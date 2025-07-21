import asyncio
import os
import sys
import time
import requests
import uvicorn
import logging
from argparse import ArgumentParser, Namespace
from typing import Optional, Coroutine, List

from sglang.srt.server_args import ServerArgs
from sglang.srt.entrypoints.http_server import (
    app as fastapi_app,
    set_global_state,
    _GlobalState,
)
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.utils import (
    kill_process_tree,
    add_api_key_middleware,
    set_uvicorn_logging_configs,
)
from sglang.utils import get_exception_traceback

from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class SGLangBackend(BaseBackend):
    """
    An adapter class that provides fine-grained control over the SGLang server lifecycle.
    """

    def __init__(self, framework_args: Namespace, backend_argv: List[str]):
        """
        Initializes the backend and prepares SGLang arguments.
        """
        super().__init__(framework_args, backend_argv)
        self.sglang_args: ServerArgs = self._parse_sglang_args(backend_argv)

        self.app = None
        self.tokenizer_manager = None
        self.server_task: Optional[asyncio.Task] = None

    def _parse_sglang_args(self, backend_argv: List[str]) -> ServerArgs:
        """
        Parses SGLang-specific arguments from the command line.
        """
        parser = ArgumentParser()
        ServerArgs.add_cli_args(parser)

        # 1. Parse the backend-specific arguments passed from the command line
        sglang_cli_args = parser.parse_args(backend_argv)

        # 2. Merge framework-level arguments (like host, port) into the SGLang args.
        # Framework arguments take precedence if they are also defined for SGLang.
        for key, value in vars(self.framework_args).items():
            if hasattr(sglang_cli_args, key) and value is not None:
                setattr(sglang_cli_args, key, value)
        
        # 3. Create the final ServerArgs instance from the combined arguments
        server_args = ServerArgs.from_cli_args(sglang_cli_args)
        server_args.check_server_args()
        
        return server_args

    async def run(self):
        """
        Orchestrates the startup and shutdown of the SGLang server.
        """
        logger.info("Starting SGLang backend server...")
        try:
            self.server_task = asyncio.create_task(self._run_server_worker())
            await self.server_task
        except asyncio.CancelledError:
            logger.info("SGLang backend server task was cancelled.")
        except Exception:
            logger.error(f"SGLang backend server failed: {get_exception_traceback()}")
            raise
        finally:
            self.cleanup()

    async def _run_server_worker(self):
        """
        The core logic to build and run the SGLang API server worker.
        This is an adaptation of SGLang's `launch_server` function.
        """
        # 1. Launch SGLang engine subprocesses (Tokenizer, Scheduler, Detokenizer).
        self.tokenizer_manager, scheduler_info = _launch_subprocesses(
            server_args=self.sglang_args
        )

        # 2. Set the global state required by SGLang's API endpoints.
        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                scheduler_info=scheduler_info,
            )
        )

        # 3. Get and configure the FastAPI app.
        self.app = fastapi_app
        self.app.server_args = self.sglang_args  # type: ignore

        if self.sglang_args.api_key:
            add_api_key_middleware(self.app, self.sglang_args.api_key)

        set_uvicorn_logging_configs()
        
        logger.info(
            f"Starting SGLang server on http://{self.sglang_args.host}:{self.sglang_args.port}"
        )

        # 4. Create server and warmup coroutines.
        server_coro = self._serve_http()
        warmup_coro = self._wait_and_warmup()

        # 5. Concurrently run the server and the warmup process.
        await asyncio.gather(server_coro, warmup_coro)

    def _serve_http(self) -> Coroutine:
        """
        Configures and creates the Uvicorn server instance as a coroutine.
        """
        assert self.app is not None, "FastAPI app has not been initialized."
        config = uvicorn.Config(
            self.app,
            host=self.sglang_args.host,
            port=self.sglang_args.port,
            log_level=self.sglang_args.log_level_http or self.sglang_args.log_level,
            timeout_keep_alive=5,
            loop="uvloop",
        )
        
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: {}  # type: ignore
        
        return server.serve()

    async def _wait_and_warmup(self):
        """
        Waits for the server to be healthy and then sends a warmup request.
        """
        headers = {}
        url = self.sglang_args.url()
        if self.sglang_args.api_key:
            headers["Authorization"] = f"Bearer {self.sglang_args.api_key}"

        # Wait until the server is launched
        success = False
        last_traceback = ""
        for _ in range(120):  # Wait for up to 2 minutes
            await asyncio.sleep(1)
            try:
                res = requests.get(url + "/get_model_info", timeout=5, headers=headers)
                if res.status_code == 200:
                    success = True
                    break
            except requests.exceptions.RequestException:
                last_traceback = get_exception_traceback()
        
        if not success:
            logger.error(f"SGLang server failed to start. Last error: {last_traceback}")
            # Cancel the main server task to trigger cleanup
            if self.server_task:
                self.server_task.cancel()
            return

        model_info = res.json()

        # Send a warmup request
        request_name = "/generate" if model_info["is_generation"] else "/encode"
        max_new_tokens = 8 if model_info["is_generation"] else 1
        json_data = {
            "text": "The capital city of France is",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
        }

        try:
            res = requests.post(url + request_name, json=json_data, headers=headers, timeout=600)
            res.raise_for_status()
        except Exception:
            logger.error(f"SGLang warmup request failed: {get_exception_traceback()}")
            if self.server_task:
                self.server_task.cancel()
            return
        
        logger.info("SGLang backend server is warmed up and ready to roll!")
        self.server_ready.set()

    def cleanup(self):
        """
        Gracefully cleans up all server resources.
        """
        logger.info("Cleaning up SGLang backend resources...")
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
        
        # SGLang's launch_server uses this to kill all subprocesses
        kill_process_tree(os.getpid(), include_parent=False)
