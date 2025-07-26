
import asyncio
import os
import subprocess
from typing import Any, Dict, List


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
                print(
                    f"WARN:     Restart requested for model '{model_name}', but process {process.pid} still exists. Waiting for it to terminate..."
                )
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
            "ichat.worker",
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

        gpu_ids = config.get("gpu_ids")
        if gpu_ids is not None:
            import json
            framework_args["gpu-ids"] = json.dumps(gpu_ids)

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