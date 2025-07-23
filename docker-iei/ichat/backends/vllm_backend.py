# docker-iei/ichat/backends/vllm_backend.py

import asyncio
import signal
import socket
import argparse
import os
from argparse import Namespace
from typing import Any, Coroutine, List, Optional
import aiohttp
import psutil

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
from vllm.utils import (FlexibleArgumentParser, is_valid_ipv6_address,
                        set_ulimit)
from vllm._version import __version__ as VLLM_VERSION

from .base_backend import BaseBackend
from .. import serve


class VLLMBackend(BaseBackend):
    """
    An adapter class that provides fine-grained control over the vLLM server lifecycle,
    going a level deeper than the standard `run_server` entrypoint. This allows for
    customization and better integration within the iChat framework.
    """

    def __init__(
        self,
        framework_args: Namespace,
        backend_argv: List[str],
        backend_ready_event: asyncio.Event,
    ):
        """
        Initializes the VLLM backend by converting iChat arguments into a format
        that vLLM's components can understand.

        Args:
            framework_args: A Namespace object containing arguments for the iChat framework.
            backend_argv: A list of strings for backend-specific arguments.
            backend_ready_event: An asyncio.Event to signal when the backend is ready.
        """
        super().__init__(framework_args, backend_argv)
        self.server_ready = backend_ready_event
        self.vllm_args = self._parse_vllm_args(backend_argv)
        
        # Placeholders for server components to be initialized in run()
        self.app: Optional[FastAPI] = None
        self.engine_client: Optional[EngineClient] = None
        self.vllm_config: Optional[VllmConfig] = None
        self.server_task: Optional[asyncio.Task] = None
        self.server_shutdown_event: Optional[asyncio.Event] = None
        self.sock: Optional[socket.socket] = None
        self.worker_pid: Optional[int] = None
        
    def _parse_vllm_args(self, backend_argv: List[str]) -> Namespace:
        """
        Parses vLLM-specific arguments, including unified arguments from iChat.
        """
        # 1. Create a parser for vLLM to define all its arguments
        # The make_arg_parser function from vllm now expects an existing parser.
        parser = FlexibleArgumentParser(
            prog="vllm",
            description="vLLM-backed iChat server for OpenAI-compatible API.")
        make_arg_parser(parser)

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

        # 4. Apply mappings from iChat's unified args to vLLM's native args
        # If the user provided an iChat-specific argument, its value is mapped
        # to the corresponding vLLM argument.
        if vllm_args.model_path is not None:
            vllm_args.model = vllm_args.model_path

        if vllm_args.context_length is not None:
            vllm_args.max_model_len = vllm_args.context_length
        
        return vllm_args

    def get_backend_args(self) -> Namespace:
        """Returns the parsed and resolved arguments for the backend."""
        return self.vllm_args

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

            # Create a coroutine for the server.
            server_task = asyncio.create_task(self._serve_http(sock=sock, **uvicorn_kwargs))
            
            # The warmup must complete before we can consider the server fully running.
            # It needs the server task to be running in the background.
            await self._wait_and_warmup()
            
            # Start the health check monitor only after the server is ready.
            health_check_task = asyncio.create_task(self._health_check_monitor())
            
            # Now that warmup is done, we monitor the long-running server and health-check tasks.
            # If either one fails, we'll shut everything down.
            pending = {server_task, health_check_task}
            try:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )

                # If a task has an exception, retrieve and re-raise it.
                # This will propagate the error up to backend.run()
                for task in done:
                    if task.exception():
                        raise task.exception()
            finally:
                # Gracefully cancel all pending tasks on exit.
                for task in pending:
                    task.cancel()
                # Ensure cancellation is processed.
                await asyncio.gather(*pending, return_exceptions=True)
                print("INFO:     vLLM worker tasks have been cleaned up.")

    async def _health_check_monitor(self):
        """
        Monitors the health of the vLLM engine. If the engine becomes unhealthy
        (e.g., the worker process dies), this task will raise an exception,
        triggering a shutdown of the backend.
        """
        if not self.engine_client:
            return

        # The client holding the process handle is nested. The attribute name
        # might be 'engine' or 'engine_client' depending on the vLLM version.
        # client_with_proc = getattr(self.engine_client, 'engine', None) or \
        #                    getattr(self.engine_client, 'engine_client', None)
        # engine_process = getattr(client_with_proc, '_engine_process', None)

        # if not isinstance(engine_process, psutil.Process):
        #     print("WARN:     Could not get vLLM engine process object. "
        #           "Health checks will rely on RPC timeouts.")
        #     engine_process = None

        while True:
            await asyncio.sleep(5)
            try:
                # Priority 1: Direct process check using psutil. This is the most
                # reliable and fastest way to detect if the worker was killed.
                # if engine_process:
                #     if not engine_process.is_running() or \
                #        engine_process.status() == psutil.STATUS_ZOMBIE:
                #         raise RuntimeError("vLLM worker process is not running or is a zombie.")

                # Priority 2: Liveness check via RPC. This is for cases where
                # the process is running but stuck (unresponsive).
                await asyncio.wait_for(self.engine_client.is_sleeping(), timeout=10.0)
                # print("INFO:     vLLM engine is healthy.")

            # except psutil.NoSuchProcess:
            #     print("ERROR:    vLLM worker process no longer exists. Triggering shutdown.")
            #     raise RuntimeError("vLLM worker process does not exist.")
            
            except asyncio.TimeoutError:
                print("ERROR:    vLLM engine is unresponsive (RPC timed out). Triggering shutdown.")
                raise RuntimeError("vLLM engine is unresponsive.")

            except Exception as e:
                # This will catch any other exception and trigger a shutdown.
                print(f"ERROR:    vLLM engine health check failed: {e}. Triggering shutdown.")
                raise RuntimeError(f"vLLM engine health check failed: {e}")

    async def _wait_and_warmup(self):
        """
        Waits for the server to be healthy. This may take a long time
        for large models to load.
        """
        health_url = f"http://{self.vllm_args.host or 'localhost'}:{self.vllm_args.port}/health"

        print("INFO:     Waiting for model to load. This may take a while...")

        # Wait until the server is launched, checking every 5 seconds.
        # This will block indefinitely until the server is ready.
        while True:
            await asyncio.sleep(5)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(health_url, timeout=5) as resp:
                        if resp.status == 200:
                            print("INFO:     vLLM server is healthy.")
                            self.server_ready.set()
                            return
            except aiohttp.ClientError:
                # This is expected if the server is not up yet.
                pass


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
        
        if self.server_task and not self.server_task.done(): # This is the line I'm adding the check to
            self.server_task.cancel()
            
        print("INFO:     VLLM backend has stopped.")
