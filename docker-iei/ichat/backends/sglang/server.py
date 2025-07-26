
import asyncio
import logging
import os
from typing import Coroutine, List

import psutil
import uvicorn
from sglang.srt.entrypoints.engine import _launch_subprocesses
from sglang.srt.entrypoints.http_server import (_GlobalState, app as fastapi_app,
                                                 set_global_state)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (add_api_key_middleware,
                                set_uvicorn_logging_configs)

from .monitor import health_check_monitor, wait_and_warmup

logger = logging.getLogger(__name__)


async def run_server_worker(
    sglang_args: ServerArgs,
    server_ready_event: asyncio.Event,
    server_shutdown_event_holder: list,
    subprocesses_holder: list
):
    """
    The core logic to build and run the SGLang API server worker.
    This is an adaptation of SGLang's `launch_server` function.
    """
    # 0. Get a list of child processes before launching new ones.
    # This allows us to identify the SGLang-specific subprocesses later.
    pre_launch_children = psutil.Process(os.getpid()).children()
    # 1. Launch SGLang engine subprocesses (Tokenizer, Scheduler, Detokenizer).
    (
        tokenizer_manager,
        template_manager,
        scheduler_info,
        *_,
    ) = _launch_subprocesses(server_args=sglang_args)

    # Store the newly created SGLang subprocesses for monitoring
    post_launch_children = psutil.Process(os.getpid()).children()
    subprocesses = [p for p in post_launch_children if p not in pre_launch_children]
    subprocesses_holder.extend(subprocesses)
    logger.info(f"Detected {len(subprocesses)} SGLang subprocesses to monitor.")

    # 2. Set the global state required by SGLang's API endpoints.
    set_global_state(
        _GlobalState(
            tokenizer_manager=tokenizer_manager,
            scheduler_info=scheduler_info,
            template_manager=template_manager,
        )
    )

    # 3. Get and configure the FastAPI app.
    app = fastapi_app
    app.server_args = sglang_args  # type: ignore

    if sglang_args.api_key:
        add_api_key_middleware(app, sglang_args.api_key)

    set_uvicorn_logging_configs()
    
    logger.info(
        f"Starting SGLang server on http://{sglang_args.host}:{sglang_args.port}"
    )

    # 4. Create and monitor tasks.
    server_task = asyncio.create_task(serve_http(app, sglang_args, server_shutdown_event_holder))
    
    # Wait for the model to load and the server to be ready.
    try:
        await wait_and_warmup(sglang_args, server_ready_event)
    except Exception as e:
        logger.error(f"SGLang server failed during startup: {e}")
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)
        raise # Re-raise to trigger cleanup

    # Once ready, start the continuous health check monitor.
    health_check_task = asyncio.create_task(health_check_monitor(subprocesses))

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


def serve_http(app, sglang_args: ServerArgs, shutdown_event_holder: list) -> Coroutine:
    """
    Configures and creates the Uvicorn server instance as a coroutine.
    """
    log_level = (
        sglang_args.log_level_http or sglang_args.log_level
    )
    config = uvicorn.Config(
        app,
        host=sglang_args.host,
        port=sglang_args.port,
        log_level=log_level.lower(),
        timeout_keep_alive=5,
        loop="uvloop",
    )
    
    server = uvicorn.Server(config)
    shutdown_event_holder.append(server.should_exit)
    server.install_signal_handlers = lambda: {}  # type: ignore
    
    return server.serve()

