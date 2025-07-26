import asyncio
from argparse import Namespace
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class BaseBackend:
    """
    Base class for all backend implementations (e.g., vLLM, SGLang).
    """

    def __init__(self, framework_args: Namespace, backend_argv: List[str], server_ready_event: asyncio.Event): 
        self.framework_args = framework_args
        self.backend_argv = backend_argv
        self.server_ready = server_ready_event
        self.final_backend_args: Optional[Namespace] = None

    def get_backend_args(self) -> Optional[Namespace]:
        """
        Returns the final backend arguments, which may have been modified by
        the backend's own argument parser.
        """
        return self.final_backend_args

    def is_server_ready(self) -> bool:
        """
        Checks if the backend server is ready.
        """
        return self.server_ready.is_set()

    async def run(self):
        """
        The main entry point to start the backend server.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def cleanup(self):
        """
        Cleans up any resources used by the backend.
        This is an optional method for subclasses to implement.
        """
        pass

    async def wait_for_server_ready(self):
        """
        Waits until the backend server is fully initialized and ready to serve requests.
        """
        await self.server_ready.wait() 