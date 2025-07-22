import asyncio
import os
import sys
import time
import requests
import uvicorn
import logging
import aiohttp
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
        self.server_shutdown_event: Optional[asyncio.Event] = None

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
        (
            tokenizer_manager,
            template_manager,
            scheduler_info,
            *_,
        ) = _launch_subprocesses(server_args=self.sglang_args)

        self.tokenizer_manager = tokenizer_manager

        # 2. Set the global state required by SGLang's API endpoints.
        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                scheduler_info=scheduler_info,
                template_manager=template_manager,
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

        # 4. Create and monitor tasks.
        server_task = asyncio.create_task(self._serve_http())
        
        # Wait for the model to load and the server to be ready.
        try:
            await self._wait_and_warmup()
        except Exception as e:
            logger.error(f"SGLang server failed during startup: {e}")
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)
            raise # Re-raise to trigger cleanup

        # Once ready, start the continuous health check monitor.
        health_check_task = asyncio.create_task(self._health_check_monitor())
        
        # Monitor both tasks. If one fails, the other is cancelled.
        pending = {server_task, health_check_task}
        try:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.exception():
                    raise task.exception()
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)


    def _serve_http(self) -> Coroutine:
        """
        Configures and creates the Uvicorn server instance as a coroutine.
        """
        assert self.app is not None, "FastAPI app has not been initialized."
        log_level = (
            self.sglang_args.log_level_http or self.sglang_args.log_level
        )
        config = uvicorn.Config(
            self.app,
            host=self.sglang_args.host,
            port=self.sglang_args.port,
            log_level=log_level.lower(),
            timeout_keep_alive=5,
            loop="uvloop",
        )
        
        server = uvicorn.Server(config)
        self.server_shutdown_event = server.should_exit
        server.install_signal_handlers = lambda: {}  # type: ignore
        
        return server.serve()

    async def _wait_and_warmup(self):
        """
        Waits for the server to be healthy and then sends a warmup request.
        """
        headers = {}
        base_url = self.sglang_args.url()
        if self.sglang_args.api_key:
            headers["Authorization"] = f"Bearer {self.sglang_args.api_key}"

        # Wait until the server is launched
        logger.info("Waiting for SGLang server to be ready...")
        health_url = base_url + "/get_model_info"
        
        async with aiohttp.ClientSession() as session:
            while True:
                await asyncio.sleep(1)
                try:
                    async with session.get(health_url, timeout=5, headers=headers) as res:
                        if res.status == 200:
                            model_info = await res.json()
                            logger.info("SGLang server is up.")
                            break
                except aiohttp.ClientError:
                    pass # Keep trying
        
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
            warmup_url = base_url + request_name
            try:
                async with session.post(warmup_url, json=json_data, headers=headers, timeout=600) as res:
                    res.raise_for_status()
            except Exception:
                logger.error(f"SGLang warmup request failed: {get_exception_traceback()}")
                raise # Re-raise to fail the backend startup
        
        logger.info("SGLang backend server is warmed up and ready to roll!")
        self.server_ready.set()

    async def _health_check_monitor(self):
        """
        Monitors the health of the SGLang engine continuously after startup.
        """
        headers = {}
        health_url = self.sglang_args.url() + "/get_model_info"
        if self.sglang_args.api_key:
            headers["Authorization"] = f"Bearer {self.sglang_args.api_key}"

        async with aiohttp.ClientSession() as session:
            while True:
                await asyncio.sleep(10)
                try:
                    async with session.get(health_url, timeout=5, headers=headers) as resp:
                        if resp.status != 200:
                            logger.error(f"SGLang health check returned status {resp.status}.")
                            raise RuntimeError(f"SGLang health check failed.")
                except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                    logger.error(f"SGLang engine is unresponsive: {e}")
                    raise RuntimeError("SGLang engine is unresponsive.") from e

    def cleanup(self):
        """
        Gracefully cleans up all server resources.
        """
        logger.info("Cleaning up SGLang backend resources...")
        if self.server_shutdown_event and not self.server_shutdown_event.is_set():
            self.server_shutdown_event.set()

        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
        
        # SGLang's launch_server uses this to kill all subprocesses
        kill_process_tree(os.getpid(), include_parent=False)
