
import asyncio
import logging
from typing import List

import aiohttp
import psutil

from sglang.srt.server_args import ServerArgs
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


async def wait_and_warmup(sglang_args: ServerArgs, server_ready_event: asyncio.Event):
    """
    Waits for the server to be healthy and then sends a warmup request.
    """
    headers = {}
    base_url = sglang_args.url()
    if sglang_args.api_key:
        headers["Authorization"] = f"Bearer {sglang_args.api_key}"

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
    server_ready_event.set()


async def health_check_monitor(subprocesses: List[psutil.Process]):
    """
    Monitors the health of SGLang subprocesses directly.

    This provides a more robust way to detect worker failures than HTTP checks,
    as it directly verifies if the underlying processes are running. If any
    of the essential SGLang subprocesses (e.g., scheduler, detokenizer) die,
    this monitor will detect it and trigger a server shutdown.
    """
    while True:
        await asyncio.sleep(5)
        for proc in subprocesses:
            if not proc.is_running():
                logger.error(f"SGLang subprocess with PID {proc.pid} has terminated unexpectedly.")
                raise RuntimeError("A critical SGLang subprocess has failed.")

