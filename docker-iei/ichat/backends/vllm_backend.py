# docker-iei/ichat/backends/vllm_backend.py

import asyncio
import signal
import socket
import argparse
from argparse import Namespace
from typing import Any, Coroutine, List, Optional

import uvicorn
from fastapi import FastAPI

# VLLM imports for deep control
from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.openai.api_server import (
    build_app, build_async_engine_client, create_server_socket,
    init_app_state)
from vllm.entrypoints.openai.cli_args import (make_arg_parser,
                                              validate_parsed_serve_args)
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.utils import is_valid_ipv6_address, set_ulimit
from vllm._version import __version__ as VLLM_VERSION

from .base_backend import BaseBackend


class VLLMBackend(BaseBackend):
    """
    An adapter class that provides fine-grained control over the vLLM server lifecycle,
    going a level deeper than the standard `run_server` entrypoint. This allows for
    customization and better integration within the iChat framework.
    """

    def __init__(self, framework_args: Namespace, backend_argv: List[str]):
        """
        Initializes the VLLM backend by converting iChat arguments into a format
        that vLLM's components can understand.

        Args:
            framework_args: A Namespace object containing arguments for the iChat framework.
            backend_argv: A list of strings for backend-specific arguments.
        """
        super().__init__(framework_args, backend_argv)
        self.vllm_args = self._parse_vllm_args(backend_argv)
        
        # Placeholders for server components to be initialized in run()
        self.app: Optional[FastAPI] = None
        self.engine_client: Optional[EngineClient] = None
        self.vllm_config: Optional[VllmConfig] = None
        self.server_task: Optional[asyncio.Task] = None
        self.server_shutdown_event: Optional[asyncio.Event] = None
        self.sock: Optional[socket.socket] = None
        
    def _parse_vllm_args(self, backend_argv: List[str]) -> Namespace:
        """
        Parses vLLM-specific arguments, including unified arguments from iChat.
        """
        # 1. Create a parser for vLLM to define all its arguments
        parser = make_arg_parser()
        
        # 2. Add iChat's unified arguments to the parser for recognition
        # These arguments might not be in vLLM's default parser, so we add them
        # to ensure they are parsed correctly.
        # Note: This is a placeholder for a more robust mapping if names differ.
        if not any(opt.dest == 'model_path' for opt in parser._actions):
             parser.add_argument("--model-path", type=str, default=None, help="iChat alias for --model.")
        if not any(opt.dest == 'context_length' for opt in parser._actions):
             parser.add_argument("--context-length", type=int, default=None, help="iChat alias for --max-model-len.")

        # 3. Parse the backend-specific arguments
        vllm_args, _ = parser.parse_known_args(backend_argv)

        # 4. Map iChat unified argument names to vLLM expected names
        # As per serve.md, --model-path -> --model
        if hasattr(vllm_args, 'model_path') and vllm_args.model_path:
            vllm_args.model = vllm_args.model_path
        
        # As per serve.md, --tokenizer-path -> --tokenizer
        if hasattr(vllm_args, 'tokenizer_path') and vllm_args.tokenizer_path:
            vllm_args.tokenizer = vllm_args.tokenizer_path

        # As per serve.md, --context-length -> --max-model-len
        if hasattr(vllm_args, 'context_length') and vllm_args.context_length:
            vllm_args.max_model_len = vllm_args.context_length
            
        # 5. Merge framework arguments into vllm_args
        # This allows using framework-level settings (e.g., host, port) in the backend.
        for key, value in vars(self.framework_args).items():
            if not hasattr(vllm_args, key) or getattr(vllm_args, key) is None:
                 if value is not None:
                    setattr(vllm_args, key, value)

        # Ensure served_model_name is a list for vLLM
        if hasattr(vllm_args, 'served_model_name') and isinstance(vllm_args.served_model_name, str):
            vllm_args.served_model_name = [vllm_args.served_model_name]

        return vllm_args

    async def run(self):
        """
        Starts and manages the vLLM API server with fine-grained control.
        
        This method replaces the simple `run_server` call with its underlying
        logic to set up and run the server, allowing for deeper integration.
        """
        print("INFO:     Starting vLLM backend server with fine-grained control...")

        try:
            listen_address, self.sock = self._setup_server()
            
            # The server will run in a separate task
            self.server_task = asyncio.create_task(
                self._run_server_worker(listen_address, self.sock)
            )
            await self.server_task
            
        except asyncio.CancelledError:
            print("INFO:     VLLM backend server task was cancelled.")
        except Exception as e:
            print(f"ERROR:    An error occurred in the vLLM backend: {e}")
            raise
        finally:
            self.cleanup()

    def _setup_server(self):
        """
        Validates arguments and creates a server socket. This is an adaptation of
        vLLM's `setup_server` function.
        """
        print(f"INFO:     vLLM API server version {VLLM_VERSION}")
        
        if self.vllm_args.tool_parser_plugin and len(self.vllm_args.tool_parser_plugin) > 3:
            ToolParserManager.import_tool_parser(self.vllm_args.tool_parser_plugin)

        validate_parsed_serve_args(self.vllm_args)

        host = self.vllm_args.host
        port = self.vllm_args.port
        
        # Bind socket before engine setup to avoid race conditions
        sock_addr = (host or "0.0.0.0", port)
        sock = create_server_socket(sock_addr)

        set_ulimit()

        # Create listen address for logging
        addr, port = sock.getsockname()
        is_ssl = self.vllm_args.ssl_keyfile and self.vllm_args.ssl_certfile
        host_part = f"[{addr}]" if is_valid_ipv6_address(addr) else addr
        listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"

        return listen_address, sock

    async def _run_server_worker(self, listen_address: str, sock: socket.socket, **uvicorn_kwargs: Any) -> None:
        """
        The core logic to build and run the vLLM API server worker.
        This is an adaptation of vLLM's `run_server_worker` async function.
        """
        async with build_async_engine_client(self.vllm_args) as engine_client:
            self.engine_client = engine_client
            
            self.app = build_app(self.vllm_args)
            
            self.vllm_config = await self.engine_client.get_vllm_config()
            await init_app_state(self.engine_client, self.vllm_config, self.app.state, self.vllm_args)

            print(f"INFO:     Starting vLLM server on {listen_address}")
            
            # Start the Uvicorn server in a separate task
            server_coro = self._serve_http(sock=sock, **uvicorn_kwargs)
            warmup_coro = self._wait_and_warmup()

            await asyncio.gather(server_coro, warmup_coro)
    
    async def _wait_and_warmup(self):
        """
        Waits for the server to be healthy and then sends a warmup request.
        """
        import requests
        from vllm.utils import get_exception_traceback

        health_url = f"http://{self.vllm_args.host}:{self.vllm_args.port}/health"

        # Wait until the server is launched
        for _ in range(120):  # Wait for up to 2 minutes
            await asyncio.sleep(1)
            try:
                res = requests.get(health_url, timeout=5)
                if res.status_code == 200:
                    print("INFO:     vLLM server is healthy.")
                    self.server_ready.set()
                    return
            except requests.exceptions.RequestException:
                pass
        
        print("ERROR:    vLLM server failed to start after 2 minutes.")
        if self.server_task:
            self.server_task.cancel()


    def _serve_http(self, sock: socket.socket, **uvicorn_kwargs: Any) -> Coroutine:
        """
        Creates a coroutine to run the Uvicorn server.
        """
        assert self.app is not None, "FastAPI app is not initialized"
        
        config = uvicorn.Config(
            self.app,
            log_level=self.vllm_args.uvicorn_log_level,
            access_log=not self.vllm_args.disable_uvicorn_access_log,
            ssl_keyfile=self.vllm_args.ssl_keyfile,
            ssl_certfile=self.vllm_args.ssl_certfile,
            ssl_ca_certs=self.vllm_args.ssl_ca_certs,
            ssl_cert_reqs=self.vllm_args.ssl_cert_reqs,
            **uvicorn_kwargs
        )
        
        server = uvicorn.Server(config)
        self.server_shutdown_event = server.should_exit
        
        # Override signal handlers to allow for programmatic shutdown
        server.install_signal_handlers = lambda: {}
        
        return server.serve(sockets=[sock])

    def cleanup(self):
        """
        Gracefully cleans up server resources.
        """
        print("INFO:     Cleaning up VLLM backend resources...")
        if self.server_shutdown_event and not self.server_shutdown_event.is_set():
            self.server_shutdown_event.set()
        
        if self.sock:
            self.sock.close()
            self.sock = None
        
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            
        print("INFO:     VLLM backend has stopped.")
