
import asyncio
import time
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from .registry import ServiceRegistry
from .worker_manager import WorkerManager

# --- Global State References ---
# These will be populated by the main gateway entrypoint
service_registry: Optional[ServiceRegistry] = None
worker_manager: Optional[WorkerManager] = None


# --- Pydantic Models for API Validation ---
class HeartbeatPayload(BaseModel):
    worker_id: str
    model_name: str
    model_path: str
    backend: str
    host: str
    port: int
    state: str = "ready"  # "initializing" or "ready"


# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown logic for the application.
    """
    print("INFO:     iChat Gateway starting up...")
    global service_registry, worker_manager
    if not service_registry:
        raise RuntimeError("ServiceRegistry has not been initialized.")

    # Start background tasks
    health_check_task = asyncio.create_task(service_registry.check_unhealthy_workers())

    # Start managed workers if configured
    if worker_manager:
        asyncio.create_task(worker_manager.start_all_workers())

    try:
        yield
    finally:
        # --- Shutdown sequence ---
        print("INFO:     iChat Gateway shutting down...")
        if worker_manager:
            await worker_manager.stop_workers()

        # Cancel background tasks
        health_check_task.cancel()
        try:
            await health_check_task
        except asyncio.CancelledError:
            pass

        print("INFO:     Gateway has shut down.")


# --- FastAPI Application ---
app = FastAPI(lifespan=lifespan)


# --- API Endpoints ---
# Control Plane: For worker and admin interaction
@app.post("/v1/workers/heartbeat")
async def receive_heartbeat(payload: HeartbeatPayload):
    assert service_registry is not None
    success, message = service_registry.register_worker(**payload.model_dump())
    if not success:
        raise HTTPException(status_code=409, detail=message)
    return {"status": "ok", "message": message}


@app.get("/v1/admin/workers")
async def list_workers():
    assert service_registry is not None
    return service_registry.get_all_workers()


# Data Plane: For client requests, with custom routing
async def _forward_request_to_worker(request: Request):
    """
    Generic function to forward a request to a worker based on the model name in the
    request body. It handles both streaming and non-streaming responses.
    """
    data = await request.json()
    model_name = data.get("model")
    assert service_registry is not None

    if not model_name:
        raise HTTPException(status_code=400, detail="Request body must include a 'model' key.")

    worker = service_registry.get_worker_for_model(model_name)

    if not worker:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found or not ready.")

    worker_url = f"http://{worker['host']}:{worker['port']}{request.url.path}"

    is_streaming = data.get("stream", False)

    session = aiohttp.ClientSession()
    try:
        response = await session.post(worker_url, json=data, timeout=None)

        if not response.ok:
            error_text = await response.text()
            await response.release()
            await session.close()
            raise HTTPException(
                status_code=response.status,
                detail=f"Error from worker: {error_text}",
            )

        if is_streaming:

            async def cleanup(resp: aiohttp.ClientResponse, sess: aiohttp.ClientSession):
                await resp.release()
                await sess.close()

            return StreamingResponse(
                response.content,
                media_type=response.headers.get("Content-Type"),
                background=BackgroundTask(cleanup, resp=response, sess=session),
            )
        else:
            response_data = await response.json()
            await response.release()
            await session.close()
            return JSONResponse(content=response_data, status_code=response.status)
    except aiohttp.ClientConnectorError:
        await session.close()
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable. Could not connect to worker for model '{model_name}'.",
        )
    except Exception as e:
        await session.close()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while communicating with worker: {str(e)}",
        )


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    return await _forward_request_to_worker(request)


@app.api_route("/v1/completions", methods=["POST"])
async def completions(request: Request):
    return await _forward_request_to_worker(request)


@app.api_route("/v1/embeddings", methods=["POST"])
async def embeddings(request: Request):
    return await _forward_request_to_worker(request)


@app.api_route("/v1/rerank", methods=["POST"])
async def rerank(request: Request):
    return await _forward_request_to_worker(request)


@app.api_route("/v1/models", methods=["GET"])
async def get_models():
    assert service_registry is not None
    workers = service_registry.get_all_workers()
    model_list = []
    # Use a set to avoid duplicate model names if multiple workers serve the same model
    registered_models = set()
    for _, worker_info in workers.items():
        model_name = worker_info.get("model_name")
        # Expose models that are either ready or in the process of initializing.
        if worker_info.get("state") in ["ready"] and model_name not in registered_models:
            model_list.append(
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(worker_info.get("last_heartbeat", time.time())),
                    "owned_by": "ichat",
                }
            )
            registered_models.add(model_name)

    return {"object": "list", "data": model_list} 