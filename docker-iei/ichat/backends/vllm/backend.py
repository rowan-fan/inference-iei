# docker-iei/ichat/backends/vllm_backend.py

import asyncio
import socket
from argparse import Namespace
from typing import List, Optional

from ..base_backend import BaseBackend
from .args import parse_vllm_args
from .server import run_server_worker, setup_server


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
        super().__init__(framework_args, backend_argv, backend_ready_event)
        self.vllm_args = parse_vllm_args(backend_argv)
        self.final_backend_args = self.vllm_args
        
        # Placeholders for server components
        self.server_task: Optional[asyncio.Task] = None
        self.server_shutdown_event: Optional[asyncio.Event] = None
        self.sock: Optional[socket.socket] = None
        
    def get_backend_args(self) -> Namespace:
        """Returns the parsed and resolved arguments for the backend."""
        return self.final_backend_args

    async def run(self):
        """
        Starts and manages the vLLM API server with fine-grained control.
        """
        print("INFO:     Starting vLLM backend server with fine-grained control...")
        shutdown_event_holder = []

        try:
            listen_address, self.sock = setup_server(self.vllm_args)
            
            self.server_task = asyncio.create_task(
                run_server_worker(
                    self.vllm_args,
                    listen_address,
                    self.sock,
                    self.server_ready,
                    shutdown_event_holder
                )
            )
            await self.server_task
            
        except asyncio.CancelledError:
            print("INFO:     VLLM backend server task was cancelled.")
        except Exception as e:
            print(f"ERROR:    An error occurred in the vLLM backend: {e}")
            raise
        finally:
            if shutdown_event_holder:
                self.server_shutdown_event = shutdown_event_holder[0]
            self.cleanup()

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
        
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
            
        print("INFO:     VLLM backend has stopped.")
