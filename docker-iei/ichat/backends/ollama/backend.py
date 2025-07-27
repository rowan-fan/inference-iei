import asyncio
import argparse
import os
import re
import json
import hashlib
from typing import List, Dict, Any, Tuple
import ollama
import logging
from ..base_backend import BaseBackend

logger = logging.getLogger(__name__)

class OllamaBackend(BaseBackend):
    def __init__(self, framework_args: argparse.Namespace, backend_argv: List[str], server_ready_event: asyncio.Event):
        super().__init__(framework_args, backend_argv, server_ready_event)
        self.ollama_args = self._parse_ollama_args(backend_argv)
        self.ollama_args.model = self.ollama_args.served_model_name
        host = f"http://127.0.0.1:{self.ollama_args.port}"
        self.client = ollama.AsyncClient(host=host)
        self.shutdown_event = asyncio.Event()
        self.model_loaded_by_backend = False

    def _parse_ollama_args(self, backend_argv: List[str]) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Ollama Backend Arguments")
        parser.add_argument("--served-model-name", type=str, required=True)

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
            logger.info(f"Ollama model list: {response}")
            if any(m.get('name') == model_name for m in response.get('models', [])):
                logger.info(f"Model '{model_name}' already exists in Ollama cache.")
                return
        except Exception as e:
            logger.warning(f"Could not check model list. Proceeding with pull/create. Error: {e}")

        # If model_path is provided and not a dummy value, create from file.
        # if self.ollama_args.model_path and self.ollama_args.model_path.lower() not in ['cache', 'none', 'null', '']:
        #     logger.info(f"Creating model '{model_name}' from path '{self.ollama_args.model_path}'...")
        #     await self._create_from_file(model_name, self.ollama_args.model_path)
        #     self.model_loaded_by_backend = True
        #     return

        # If not found locally, pull from the hub.
        logger.info(f"Pulling model '{model_name}' from Ollama Hub...")
        await self.client.pull(model_name)
        self.model_loaded_by_backend = True

    # async def _create_from_file(self, model_name: str, model_path: str):
    #     try:
    #         # 获取目录下所有文件
    #         file_names = [
    #             f for f in os.listdir(model_path)
    #             if os.path.isfile(os.path.join(model_path, f))
    #         ]
    #         if not file_names:
    #             raise RuntimeError(f"Directory '{model_path}' contains no files and no Modelfile.")

    #         # 创建文件字典，包含文件名和SHA256摘要
    #         files_dict = {}
    #         for file_name in file_names:
    #             file_path = os.path.join(model_path, file_name)
    #             # 计算文件的SHA256摘要
    #             sha256_hash = hashlib.sha256()
    #             with open(file_path, "rb") as f:
    #                 for chunk in iter(lambda: f.read(4096), b""):
    #                     sha256_hash.update(chunk)
    #             digest = sha256_hash.hexdigest()
    #             files_dict[file_name] = f"sha256:{digest}"
                
    #             # 推送文件到Ollama服务器创建blob
    #             logger.info(f"Pushing file '{file_name}' to Ollama server...")
    #             try:
    #                 with open(file_path, "rb") as f:
    #                     response = await self.client._request(
    #                         method="POST",
    #                         url=f"/api/blobs/sha256:{digest}",
    #                         data=f.read()
    #                     )
    #                 logger.info(f"Successfully pushed file '{file_name}' as blob.")
    #             except Exception as e:
    #                 logger.error(f"Failed to push file '{file_name}' as blob: {e}")
    #                 raise

    #         # 保存当前工作目录
    #         original_cwd = os.getcwd()
    #         try:
    #             # 切换到model_path目录
    #             os.chdir(model_path)
                
    #             await self.client.create(
    #                 model=model_name,
    #                 files=files_dict,
    #                 stream=False,
    #             )
    #         finally:
    #             # 恢复原始工作目录
    #             os.chdir(original_cwd)
                
    #         logger.info(f"Successfully created model '{model_name}' from files in '{model_path}'.")
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to create model from directory '{model_path}'. Error: {e}")


    def cleanup(self):
        logger.info("Cleaning up Ollama backend.")
        if self.model_loaded_by_backend:
            logger.info("Scheduling Ollama model deletion.")
            try:
                asyncio.create_task(self._async_cleanup())
            except Exception as e:
                logger.error(f"Failed to schedule Ollama model cleanup: {e}", exc_info=True)
        
        if not self.shutdown_event.is_set():
            self.shutdown_event.set()

    async def _async_cleanup(self):
        model_name = self.ollama_args.model
        logger.info(f"Attempting to delete model '{model_name}' from Ollama.")
        try:
            await self.client.delete(model=model_name)
            logger.info(f"Successfully deleted model '{model_name}' from Ollama.")
        except ollama.ResponseError as e:
            if e.status_code == 404:
                logger.warning(f"Model '{model_name}' not found in Ollama during cleanup, skipping deletion.")
            else:
                logger.error(f"Failed to delete model '{model_name}' from Ollama. Status: {e.status_code}, Error: {e.error}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during Ollama model cleanup: {e}", exc_info=True)