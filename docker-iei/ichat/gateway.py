
# docker-iei/ichat/gateway.py

import argparse
import asyncio
import os
import subprocess
import time
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Dict, List, Optional

import aiohttp
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask


# --- ServiceRegistry: Manages worker registration and health ---
class ServiceRegistry:
    """
    Manages the registration and health status of all workers.
    This class is thread-safe.
    """

    def __init__(self, heartbeat_timeout: int = 30, worker_manager: Optional["WorkerManager"] = None):
        self._workers: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self.heartbeat_timeout = heartbeat_timeout
        self.worker_manager = worker_manager
        self.managed_models = (
            set(worker_manager.worker_configs.keys()) if worker_manager else set()
        )

    def register_worker(self, **kwargs: Any) -> None:
        """
        Registers or updates a worker's information and heartbeat.
        A worker sending a 'terminating' state will be removed, and if it's
        a managed worker, a restart will be triggered.
        """
        worker_id = kwargs.get("worker_id")
        if not worker_id:
            return

        state = kwargs.get("state")

        if state == "terminating":
            with self._lock:
                # If it's a managed worker that is terminating, we keep it in the registry
                # with a 'terminating' state. This prevents a race condition where a client
                # might see a model as unavailable because the old worker is gone but the
                # new one hasn't registered yet. The unhealthy worker check will clean it up.
                if worker_id in self._workers:
                    worker_info = self._workers[worker_id]
                    model_name = worker_info.get("model_name")
                    is_managed = self.worker_manager and model_name in self.managed_models

                    if is_managed:
                        # For managed workers, update state to 'terminating'.
                        # The old worker entry will be replaced when the new one registers.
                        self._workers[worker_id]["state"] = "terminating"
                        self._workers[worker_id]["last_heartbeat"] = time.time()
                        print(f"INFO:     Managed worker {worker_id} ({model_name}) sent 'terminating' state. Awaiting restart.")
                    else:
                        # For unmanaged workers, remove immediately.
                        self._workers.pop(worker_id)
                        print(f"INFO:     Unmanaged worker {worker_id} ({model_name}) sent 'terminating' state. Removing from registry.")
                    
                    # Trigger restart for managed workers regardless.
                    if is_managed:
                        print(f"INFO:     Attempting to restart managed worker for model '{model_name}'...")
                        asyncio.create_task(self.worker_manager.start_worker(model_name))
            return

        # For any other state ("initializing", "ready"), perform an "upsert" operation.
        with self._lock:
            self._workers[worker_id] = {
                **kwargs,
                "last_heartbeat": time.time(),
            }

    async def check_unhealthy_workers(self) -> None:
        """
        Periodically checks for and removes workers that have missed heartbeats.
        If a managed worker times out, it will be restarted.
        """
        while True:
            await asyncio.sleep(self.heartbeat_timeout)
            
            unhealthy_workers = {}
            with self._lock:
                now = time.time()
                # Find timed-out workers
                timed_out_ids = [
                    worker_id
                    for worker_id, worker in self._workers.items()
                    if now - worker.get("last_heartbeat", 0) > self.heartbeat_timeout
                ]
                # Atomically remove them and get their info
                for worker_id in timed_out_ids:
                    unhealthy_workers[worker_id] = self._workers.pop(worker_id)

            # Process unhealthy workers outside the lock
            for worker_id, worker_info in unhealthy_workers.items():
                model_name = worker_info.get("model_name")
                print(f"INFO:     Worker {worker_id} ({model_name}) timed out. Removing from registry.")
                
                # If it's a managed worker, request a restart.
                if self.worker_manager and model_name in self.managed_models:
                    print(f"INFO:     Worker for managed model '{model_name}' timed out. Attempting to restart...")
                    # The restart is fire-and-forget from the perspective of the health check loop.
                    asyncio.create_task(self.worker_manager.start_worker(model_name))

    def get_worker_for_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Finds a ready worker that serves a given model."""
        with self._lock:
            for worker in self._workers.values():
                if worker.get("model_name") == model_name and worker.get("state") == "ready":
                    return worker.copy()
        return None

    def get_all_workers(self) -> Dict[str, Any]:
        """
        Returns a dictionary of all registered workers for admin purposes.
        """
        with self._lock:
            return self._workers.copy()


# --- WorkerManager: Manages subprocesses for managed workers ---
class WorkerManager:
    """
    Manages the lifecycle of workers defined in the config file.
    It starts them as subprocesses and ensures they are terminated gracefully.
    It can also restart workers that have failed.
    """

    def __init__(self, worker_configs: List[Dict[str, Any]], gateway_address: str):
        # Store configs in a dict for easy lookup by model_name
        self.worker_configs = {cfg["model_name"]: cfg for cfg in worker_configs}
        self.gateway_address = gateway_address
        # Store running processes by model_name to track them
        self.processes: Dict[str, subprocess.Popen] = {}

    async def start_worker(self, model_name: str):
        """Launches a single worker subprocess based on its model name."""
        if model_name not in self.worker_configs:
            print(f"ERROR:    Cannot start worker. No configuration found for model '{model_name}'.")
            return
        
        # If a process for this model is already tracked, wait for it to terminate
        # before starting a new one. This handles the restart race condition.
        if model_name in self.processes:
            process = self.processes[model_name]
            if process.poll() is None:
                print(f"WARN:     Restart requested for model '{model_name}', but process {process.pid} still exists. Waiting for it to terminate...")
                try:
                    # Use asyncio.to_thread to avoid blocking the event loop while waiting.
                    await asyncio.to_thread(process.wait, timeout=10)
                    print(f"INFO:     Old process {process.pid} for model '{model_name}' has terminated.")
                except subprocess.TimeoutExpired:
                    print(f"ERROR:    Old process {process.pid} for model '{model_name}' did not terminate in time. Killing it.")
                    process.kill()
                    # Give it a moment to die after being killed.
                    await asyncio.sleep(1)

        config = self.worker_configs[model_name]
        cmd = self._build_command(config)
        print(f"INFO:     Launching worker for model '{model_name}' with command: {' '.join(cmd)}")
        try:
            # Inherit parent environment and set/override CUDA_VISIBLE_DEVICES
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, config.get("gpu_ids", [])))
            process = subprocess.Popen(cmd, env=env)
            self.processes[model_name] = process
        except Exception as e:
            print(f"ERROR:    Failed to launch worker for {config['model_name']}: {e}")

    async def start_all_workers(self) -> None:
        """Starts all managed workers as subprocesses."""
        for model_name in self.worker_configs:
            await self.start_worker(model_name)

    async def stop_workers(self) -> None:
        """
        Terminates all managed worker subprocesses.
        """
        for process in list(self.processes.values()):
            if process.poll() is None:
                process.terminate()
        
        # Wait for processes to terminate
        for process in list(self.processes.values()):
            if process.poll() is None:
                try:
                    await asyncio.to_thread(process.wait, timeout=10)
                except subprocess.TimeoutExpired:
                    print(f"WARN:     Worker process {process.pid} did not terminate gracefully. Killing.")
                    process.kill()

    def _build_command(self, config: Dict[str, Any]) -> List[str]:
        """
        Constructs the command-line arguments for starting a worker.
        """
        cmd = [
            "python3",
            "-m",
            "ichat.serve",
            "--gateway-address",
            self.gateway_address,
        ]
        # These are framework-specific args for the worker
        framework_args = {
            "served-model-name": config.get("model_name"),
            "model-path": config.get("model_path"),
            "backend": config.get("backend"),
            "host": config.get("host"),
            "port": config.get("port"),
            "heartbeat-interval": config.get("heartbeat_interval"),
        }
        for key, value in framework_args.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])

        # Add any other backend-specific parameters
        # These are the keys from the worker config that are handled by the framework
        # and should not be passed down to the backend again.
        known_config_keys = {
            "model_name",
            "model_path",
            "backend",
            "host",
            "port",
            "heartbeat_interval",
            "gpu_ids",
        }
        backend_args = {k: v for k, v in config.items() if k not in known_config_keys}
        if backend_args:
            for key, value in backend_args.items():
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        return cmd


# --- Global State and Configuration ---
worker_manager: Optional[WorkerManager] = None
service_registry: ServiceRegistry


# --- Pydantic Models for API Validation ---
class HeartbeatPayload(BaseModel):
    worker_id: str
    model_name: str
    model_path: str
    backend: str
    host: str
    port: int
    state: str = "ready"  # "initializing" or "ready"

# --- Background Tasks are now managed in the lifespan ---

# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown logic for the application.
    """
    print("INFO:     iChat Gateway starting up...")
    
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
    service_registry.register_worker(**payload.model_dump())
    return {"status": "ok", "message": f"Heartbeat received from {payload.worker_id}"}

@app.get("/v1/admin/workers")
async def list_workers():
    return service_registry.get_all_workers()

# Data Plane: For client requests, with custom routing
@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    data = await request.json()
    model_name = data.get("model")

    if not model_name:
        raise HTTPException(status_code=400, detail="Request body must include a 'model' key.")

    worker = service_registry.get_worker_for_model(model_name)

    if not worker:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_name}' not found or not ready."
        )

    worker_url = f"http://{worker['host']}:{worker['port']}/v1/chat/completions"

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
            # Use a background task to ensure the session is closed after the
            # response has been fully streamed.
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
            return JSONResponse(
                content=response_data, status_code=response.status
            )
    except aiohttp.ClientConnectorError:
        await session.close()
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable. Could not connect to worker for model '{model_name}'.",
        )
    except Exception as e:
        await session.close()
        # Catch other potential exceptions from aiohttp or response processing
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while communicating with worker: {str(e)}",
        )


@app.api_route("/v1/models", methods=["GET"])
async def get_models():
    workers = service_registry.get_all_workers()
    model_list = []
    # Use a set to avoid duplicate model names if multiple workers serve the same model
    registered_models = set()
    for _, worker_info in workers.items():
        model_name = worker_info.get("model_name")
        # Expose models that are either ready or in the process of initializing.
        # This gives a more accurate view of service availability during startup or restarts.
        if worker_info.get("state") in ["ready", "initializing"] and model_name not in registered_models:
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


def main():
    """Main entry point for the iChat Gateway."""
    global worker_manager, service_registry

    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="iChat Gateway Service")
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yaml file.")
    args = parser.parse_args()

    # 2. Load configuration from YAML file
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR:    Config file not found at {args.config}")
        exit(1)
    
    server_settings = config.get("server_settings", {})
    host = server_settings.get("host", "0.0.0.0")
    port = server_settings.get("port", 4000)
    gateway_address = f"http://{host}:{port}"
    
    # 3. Initialize WorkerManager for managed workers
    if "managed_workers" in config:
        worker_manager = WorkerManager(config["managed_workers"], gateway_address)

    # 4. Initialize ServiceRegistry, passing worker_manager if it exists
    service_registry = ServiceRegistry(
        heartbeat_timeout=server_settings.get("heartbeat_timeout", 30),
        worker_manager=worker_manager,
    )

    # 5. Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=server_settings.get("log_level", "info"),
    )

if __name__ == "__main__":
    main()

