
import asyncio
import time
from threading import Lock
from typing import Any, Dict, Optional, Tuple

from .worker_manager import WorkerManager


class ServiceRegistry:
    """
    Manages the registration and health status of all workers.
    This class is thread-safe.
    """

    def __init__(self, heartbeat_timeout: int = 30, worker_manager: Optional[WorkerManager] = None):
        self._workers: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self.heartbeat_timeout = heartbeat_timeout
        self.worker_manager = worker_manager
        self.managed_models = (
            set(worker_manager.worker_configs.keys()) if worker_manager else set()
        )

    def register_worker(self, **kwargs: Any) -> Tuple[bool, str]:
        """
        Registers or updates a worker's information and heartbeat.
        A worker sending a 'terminating' state will be removed, and if it's
        a managed worker, a restart will be triggered.

        Returns:
            A tuple of (success, message).
        """
        worker_id = kwargs.get("worker_id")
        if not worker_id:
            return False, "worker_id not provided"

        state = kwargs.get("state")
        model_name = kwargs.get("model_name")

        if state == "terminating":
            with self._lock:
                if worker_id in self._workers:
                    worker_info = self._workers[worker_id]
                    model_name = worker_info.get("model_name")
                    is_managed = self.worker_manager and model_name in self.managed_models

                    if is_managed:
                        self._workers[worker_id]["state"] = "terminating"
                        self._workers[worker_id]["last_heartbeat"] = time.time()
                        msg = f"Managed worker {worker_id} ({model_name}) sent 'terminating' state. Awaiting restart."
                        print(f"INFO:     {msg}")
                    else:
                        self._workers.pop(worker_id, None)
                        msg = f"Unmanaged worker {worker_id} ({model_name}) sent 'terminating' state. Removing from registry."
                        print(f"INFO:     {msg}")

                    if is_managed:
                        print(f"INFO:     Attempting to restart managed worker for model '{model_name}'...")
                        asyncio.create_task(self.worker_manager.start_worker(model_name))

                    return True, msg
            return True, f"Terminating worker {worker_id} not found in registry."

        # For any other state ("initializing", "ready"), perform an "upsert" operation.
        with self._lock:
            # Check for conflicts with existing workers for the same model.
            existing_workers = [
                (w_id, w_info)
                for w_id, w_info in self._workers.items()
                if w_info.get("model_name") == model_name and w_id != worker_id
            ]

            # Prohibit registration if an active worker already exists.
            for w_id, w_info in existing_workers:
                if w_info.get("state") != "terminating":
                    message = f"Model '{model_name}' is already served by active worker {w_id} (state: {w_info.get('state')}). Registration rejected."
                    print(f"WARN:     {message}")
                    return False, message

            # Clean up old workers for the same model that are in 'terminating' state.
            old_worker_ids_to_remove = [w_id for w_id, w_info in existing_workers]
            for old_id in old_worker_ids_to_remove:
                print(
                    f"INFO:     New worker {worker_id} registering for model '{model_name}'. Replacing old terminating worker entry {old_id}."
                )
                self._workers.pop(old_id, None)

            # Register or update the current worker.
            self._workers[worker_id] = {
                **kwargs,
                "last_heartbeat": time.time(),
            }
            return True, f"Heartbeat received from {worker_id}"

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