
import asyncio
import socket
from argparse import Namespace
from typing import Any, Coroutine

import uvicorn
from fastapi import FastAPI

from vllm.config import VllmConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.api_server import (
    build_app, build_async_engine_client, create_server_socket,
    init_app_state)
from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args
from vllm.entrypoints.openai.tool_parsers import ToolParserManager
from vllm.utils import is_valid_ipv6_address, set_ulimit
from vllm._version import __version__ as VLLM_VERSION

from .monitor import health_check_monitor, wait_and_warmup


def setup_server(vllm_args: Namespace):
    """
    Validates arguments and creates a server socket. This is an adaptation of
    vLLM's `setup_server` function.
    """
    print(f"INFO:     vLLM API server version {VLLM_VERSION}")
    
    if vllm_args.tool_parser_plugin and len(vllm_args.tool_parser_plugin) > 3:
        ToolParserManager.import_tool_parser(vllm_args.tool_parser_plugin)

    validate_parsed_serve_args(vllm_args)

    host = vllm_args.host
    port = vllm_args.port
    
    # Bind socket before engine setup to avoid race conditions
    sock_addr = (host or "0.0.0.0", port)
    sock = create_server_socket(sock_addr)

    set_ulimit()

    # Create listen address for logging
    addr, port = sock.getsockname()
    is_ssl = vllm_args.ssl_keyfile and vllm_args.ssl_certfile
    host_part = f"[{addr}]" if is_valid_ipv6_address(addr) else addr
    listen_address = f"http{'s' if is_ssl else ''}://{host_part}:{port}"

    return listen_address, sock


async def run_server_worker(
    vllm_args: Namespace,
    listen_address: str,
    sock: socket.socket,
    server_ready_event: asyncio.Event,
    server_shutdown_event_holder: list,
    **uvicorn_kwargs: Any
) -> None:
    """
    The core logic to build and run the vLLM API server worker.
    This is an adaptation of vLLM's `run_server_worker` async function.
    """
    async with build_async_engine_client(vllm_args) as engine_client:
        app = build_app(vllm_args)
        
        vllm_config = await engine_client.get_vllm_config()
        await init_app_state(engine_client, vllm_config, app.state, vllm_args)

        print(f"INFO:     Starting vLLM server on {listen_address}")

        # Create a coroutine for the server.
        server_task = asyncio.create_task(serve_http(app, sock, vllm_args, uvicorn_kwargs, server_shutdown_event_holder))
        
        # The warmup must complete before we can consider the server fully running.
        # It needs the server task to be running in the background.
        await wait_and_warmup(vllm_args, server_ready_event)
        
        # Start the health check monitor only after the server is ready.
        health_check_task = asyncio.create_task(health_check_monitor(engine_client))
        
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


def serve_http(
    app: FastAPI,
    sock: socket.socket,
    vllm_args: Namespace,
    uvicorn_kwargs: dict,
    shutdown_event_holder: list
) -> Coroutine:
    """
    Creates a coroutine to run the Uvicorn server.
    """
    config = uvicorn.Config(
        app,
        log_level=vllm_args.uvicorn_log_level,
        access_log=not vllm_args.disable_uvicorn_access_log,
        ssl_keyfile=vllm_args.ssl_keyfile,
        ssl_certfile=vllm_args.ssl_certfile,
        ssl_ca_certs=vllm_args.ssl_ca_certs,
        ssl_cert_reqs=vllm_args.ssl_cert_reqs,
        **uvicorn_kwargs
    )
    
    server = uvicorn.Server(config)
    shutdown_event_holder.append(server.should_exit)
    
    # Override signal handlers to allow for programmatic shutdown
    server.install_signal_handlers = lambda: {}
    
    return server.serve(sockets=[sock])

