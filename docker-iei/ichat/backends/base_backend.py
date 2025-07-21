import asyncio
from argparse import Namespace
from typing import List


class BaseBackend:
    """
    Base class for all backend implementations (e.g., vLLM, SGLang).
    """

    def __init__(self, framework_args: Namespace, backend_argv: List[str]):
        self.framework_args = framework_args
        self.backend_argv = backend_argv
        self.server_ready = asyncio.Event()

    async def run(self):
        """
        The main entry point to start the backend server.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def cleanup(self):
        """
        Cleans up any resources used by the backend.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    async def wait_for_server_ready(self):
        """
        Waits until the backend server is fully initialized and ready to serve requests.
        """
        await self.server_ready.wait() 