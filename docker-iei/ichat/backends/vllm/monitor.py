
import asyncio
from argparse import Namespace

import aiohttp

from vllm.engine.protocol import EngineClient


async def health_check_monitor(engine_client: EngineClient):
    """
    Monitors the health of the vLLM engine. If the engine becomes unhealthy
    (e.g., the worker process dies), this task will raise an exception,
    triggering a shutdown of the backend.
    """
    if not engine_client:
        return

    while True:
        await asyncio.sleep(5)
        try:
            # Liveness check via RPC. This is for cases where
            # the process is running but stuck (unresponsive).
            await asyncio.wait_for(engine_client.is_sleeping(), timeout=10.0)
            # print("INFO:     vLLM engine is healthy.")

        except asyncio.TimeoutError:
            print("ERROR:    vLLM engine is unresponsive (RPC timed out). Triggering shutdown.")
            raise RuntimeError("vLLM engine is unresponsive.")

        except Exception as e:
            # This will catch any other exception and trigger a shutdown.
            print(f"ERROR:    vLLM engine health check failed: {e}. Triggering shutdown.")
            raise RuntimeError(f"vLLM engine health check failed: {e}")

async def wait_and_warmup(vllm_args: Namespace, server_ready_event: asyncio.Event):
    """
    Waits for the server to be healthy. This may take a long time
    for large models to load.
    """
    health_url = f"http://{vllm_args.host or 'localhost'}:{vllm_args.port}/health"

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
                        server_ready_event.set()
                        return
        except aiohttp.ClientError:
            # This is expected if the server is not up yet.
            pass

