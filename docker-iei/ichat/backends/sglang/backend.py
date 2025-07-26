import asyncio
import logging
import os
from argparse import Namespace
from typing import List, Optional

import psutil
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.utils import get_exception_traceback

from ..base_backend import BaseBackend
from .args import parse_sglang_args
from .server import run_server_worker

logger = logging.getLogger(__name__)


class SGLangBackend(BaseBackend):
    """
    An adapter class that provides fine-grained control over the SGLang server lifecycle.
    """

    def __init__(
        self,
        framework_args: Namespace,
        backend_argv: List[str],
        backend_ready_event: asyncio.Event,
    ):
        """
        Initializes the backend and prepares SGLang arguments.
        """
        super().__init__(framework_args, backend_argv, backend_ready_event)
        self.sglang_args: ServerArgs = parse_sglang_args(framework_args, backend_argv)
        self.final_backend_args = self.sglang_args
        self.server_task: Optional[asyncio.Task] = None
        self.server_shutdown_event: Optional[asyncio.Event] = None
        self.subprocesses: List[psutil.Process] = []

    def get_backend_args(self) -> ServerArgs:
        """Returns the parsed and resolved arguments for the backend."""
        return self.final_backend_args

    async def run(self):
        """
        Orchestrates the startup and shutdown of the SGLang server.
        """
        logger.info("Starting SGLang backend server...")
        shutdown_event_holder = []
        subprocesses_holder = []
        
        try:
            self.server_task = asyncio.create_task(
                run_server_worker(
                    self.sglang_args,
                    self.server_ready,
                    shutdown_event_holder,
                    subprocesses_holder
                )
            )
            await self.server_task
        except asyncio.CancelledError:
            logger.info("SGLang backend server task was cancelled.")
        except Exception:
            logger.error(f"SGLang backend server failed: {get_exception_traceback()}")
            raise
        finally:
            if shutdown_event_holder:
                self.server_shutdown_event = shutdown_event_holder[0]
            self.subprocesses = subprocesses_holder
            self.cleanup()

    def cleanup(self):
        """
        Gracefully cleans up all server resources.
        """
        logger.info("Cleaning up SGLang backend resources...")
        
        try:
            if self.server_shutdown_event and not self.server_shutdown_event.is_set():
                logger.info("Requesting Uvicorn server to shut down.")
                self.server_shutdown_event.set()

            if self.server_task and not self.server_task.done():
                self.server_task.cancel()
        finally:
            # This is the most reliable way to ensure all spawned processes are cleaned up.
            logger.info("Killing SGLang process tree...")
            kill_process_tree(os.getpid(), include_parent=False)
