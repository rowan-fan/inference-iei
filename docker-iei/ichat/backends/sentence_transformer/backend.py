import asyncio
import subprocess
import logging
from argparse import Namespace
from typing import List, Dict, Any

import aiohttp

from ..base_backend import BaseBackend

logger = logging.getLogger(__name__)

class SentenceBackend(BaseBackend):
    def __init__(self, framework_args: Namespace, backend_argv: List[str], backend_ready_event: asyncio.Event):
        super().__init__(framework_args, backend_argv, backend_ready_event)
        self.backend_args: Dict[str, Any] = self._parse_args(backend_argv)
        self.final_backend_args = self.backend_args
        self.process: asyncio.subprocess.Process = None
        self.host = self.backend_args.get("host", "127.0.0.1")
        self.port = self.backend_args.get("port")
        if self.port is None:
            raise ValueError("Port must be specified for SentenceBackend.")

    def _parse_args(self, backend_argv: List[str]) -> Dict[str, Any]:
        """A simple parser for key-value arguments."""
        args_dict = {}
        # This is a simple parsing logic, assuming --key value format
        # A more robust solution would use argparse
        it = iter(backend_argv)
        for arg in it:
            if arg.startswith('--'):
                key = arg[2:].replace('-', '_')
                try:
                    value = next(it)
                    args_dict[key] = value
                except StopIteration:
                    # Handle boolean flags like --trust-remote-code
                    args_dict[key] = True

        # Merge framework args. Framework args take precedence.
        args_dict["host"] = self.framework_args.host
        args_dict["port"] = self.framework_args.port
        args_dict["model_path"] = self.framework_args.model_path
        args_dict["model_name"] = self.framework_args.served_model_name
        args_dict["log_level"] = self.framework_args.log_level

        # Determine device from gpu_ids. This overrides any --device in backend_argv.
        gpu_ids = self.framework_args.gpu_ids
        if not gpu_ids:
            device = "cpu"
        elif len(gpu_ids) == 1:
            device = f"cuda:{gpu_ids[0]}"
        else:
            raise ValueError(f"SentenceBackend supports at most one GPU, but got gpu_ids={gpu_ids}")
        args_dict["device"] = device

        # The `api_server` does not accept `--served-model-name`, but `--model-name`.
        # We have already mapped it, so we can remove the original passthrough argument.
        if "served_model_name" in args_dict:
            del args_dict["served_model_name"]

        # The `api_server` does not accept `--gpu-ids`, but `--device`.
        # We have already mapped it, so we can remove the original passthrough argument.
        if "gpu_ids" in args_dict:
            del args_dict["gpu_ids"]

        # Remove None values so they don't get passed as "None" string
        return {k: v for k, v in args_dict.items() if v is not None}

    def get_backend_args(self) -> Dict[str, Any]:
        return self.final_backend_args

    async def run(self):
        cmd = [
            "python3", "-m", "ichat.backends.sentence_transformer.api_server"
        ]
        for key, value in self.backend_args.items():
            cmd.append(f"--{key.replace('_', '-')}")
            if not isinstance(value, bool) or value:
                 cmd.append(str(value))


        logger.info(f"Starting SentenceBackend server with command: {' '.join(cmd)}")

        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            # Create tasks to monitor stdout/stderr
            asyncio.create_task(self._log_stream(self.process.stdout, logging.INFO))
            asyncio.create_task(self._log_stream(self.process.stderr, logging.ERROR))


            # Wait for server to be ready and monitor its health
            await self._wait_and_warmup()
            
            # The health check is implicitly the process watcher
            await self._health_check_monitor()

        except asyncio.CancelledError:
            logger.info("SentenceBackend run task cancelled.")
        except Exception as e:
            logger.error(f"Failed to start or run SentenceBackend: {e}", exc_info=True)
        finally:
            self.cleanup()
            
    async def _log_stream(self, stream, log_level):
        """Read from a stream and log it."""
        while True:
            line = await stream.readline()
            if not line:
                break
            logger.log(log_level, f"[api_server] {line.decode().strip()}")

    async def _wait_and_warmup(self):
        """Poll the /health endpoint until the server is ready."""
        url = f"http://{self.host}:{self.port}/health"
        async with aiohttp.ClientSession() as session:
            for i in range(30):  # Try for 60 seconds
                try:
                    async with session.get(url, timeout=2) as response:
                        if response.status == 200:
                            logger.info("SentenceBackend API server is ready.")
                            self.server_ready.set()
                            return
                except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
                    logger.info(f"Waiting for SentenceBackend API server to be ready... (Attempt {i+1})")
                await asyncio.sleep(2)
        
        raise RuntimeError("SentenceBackend API server failed to start in time.")

    async def _health_check_monitor(self):
        """Monitor the subprocess."""
        await self.process.wait()
        
        # If wait() returns, the process has exited.
        if self.process.returncode != 0:
            raise RuntimeError(f"Sentence API server process exited with non-zero code: {self.process.returncode}")
        else:
            logger.info("Sentence API server process has exited gracefully.")


    def cleanup(self):
        """Clean up the subprocess."""
        if self.process and self.process.returncode is None:
            logger.info("Terminating SentenceBackend API server process...")
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass  # Process already gone
        self.process = None
        logger.info("SentenceBackend cleanup complete.") 