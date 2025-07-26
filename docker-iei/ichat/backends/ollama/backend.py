import asyncio
import argparse
from typing import List
import ollama
import logging
from ..base_backend import BaseBackend

logger = logging.getLogger(__name__)

class OllamaBackend(BaseBackend):
    def __init__(self, framework_args: argparse.Namespace, backend_argv: List[str], server_ready_event: asyncio.Event):
        super().__init__(framework_args, backend_argv, server_ready_event)
        self.ollama_args = self._parse_ollama_args(backend_argv)
        host = f"http://127.0.0.1:{self.ollama_args.port}"
        self.client = ollama.AsyncClient(host=host)
        self.shutdown_event = asyncio.Event()

    def _parse_ollama_args(self, backend_argv: List[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Ollama Backend Arguments")
        parser.add_argument("--model", type=str, required=True)

        parser.add_argument("--model-path", type=str, default=None)
        args, _ = parser.parse_known_args(backend_argv)
        # Merge with framework_args
        for key, value in vars(self.framework_args).items():
            if not hasattr(args, key):
                setattr(args, key, value)
        self.final_backend_args = args
        return args

    async def run(self):
        logger.info("Starting Ollama backend...")
        try:
            await self._ensure_model_ready()
            self.server_ready.set()
            logger.info("Ollama backend is ready.")
            await self.shutdown_event.wait()
        except Exception as e:
            logger.error(f"Ollama backend failed: {e}", exc_info=True)
        finally:
            self.cleanup()

    async def _ensure_model_ready(self):
        model_name = self.ollama_args.model
        host = f"http://127.0.0.1:{self.ollama_args.port}"
        try:
            await self.client.ps()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama at {host}. Is Ollama running? Error: {e}")

        try:
            response = await self.client.list()
            if any(m['name'] == model_name for m in response.get('models', [])):
                logger.info(f"Model '{model_name}' already exists in Ollama cache.")
                return
        except Exception as e:
            logger.warning(f"Could not check model list. Proceeding with pull/create. Error: {e}")

        # If model_path is provided and not a dummy value, create from file.
        if self.ollama_args.model_path and self.ollama_args.model_path.lower() not in ['cache', 'none', 'null', '']:
            logger.info(f"Creating model '{model_name}' from path '{self.ollama_args.model_path}'...")
            await self._create_from_file(model_name, self.ollama_args.model_path)
            return

        # Otherwise, check if the model exists locally in Ollama.
        try:
            response = await self.client.list()
            if any(m['name'] == model_name for m in response.get('models', [])):
                logger.info(f"Model '{model_name}' already exists in Ollama cache.")
                return
        except Exception as e:
            logger.warning(f"Could not check model list. Proceeding with pull. Error: {e}")

        # If not found locally, pull from the hub.
        logger.info(f"Pulling model '{model_name}' from Ollama Hub...")
        await self.client.pull(model_name)

    async def _create_from_file(self, model_name: str, model_path: str):
        modelfile = f'FROM {model_path}'
        try:
            await self.client.create(model=model_name, modelfile=modelfile)
            logger.info(f"Successfully created model '{model_name}' from '{model_path}'.")
        except Exception as e:
            raise RuntimeError(f"Failed to create model from file '{model_path}'. Error: {e}")

    def cleanup(self):
        logger.info("Cleaning up Ollama backend.")
        self.shutdown_event.set()