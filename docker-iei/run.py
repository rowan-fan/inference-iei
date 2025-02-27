from xinference.client import Client
import argparse
import requests
import os
import json
import logging
import sys
import time
import traceback
from typing import Union, Tuple, List, Dict, Any, Optional, Protocol, Callable
from abc import ABC, abstractmethod
from functools import partial

# Configure logging with timestamp, level and message format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

class AcceleratorManager(ABC):
    """
    Abstract base class for accelerator management.
    Provides interface for managing hardware accelerators like GPUs.
    """
    
    @abstractmethod
    def get_accelerator_count(self) -> int:
        """
        Get number of available accelerators.
        
        Returns:
            int: Number of available accelerators
        """
        pass
    
    @abstractmethod 
    def allocate_accelerators(self, model_type: str, required_count: int = 1, gpu_idxs: List[int] = None) -> tuple[int, list[int]]:
        """
        Allocate accelerators for model deployment.
        
        Args:
            model_type: Type of model requesting allocation
            required_count: Number of accelerators needed
            gpu_idxs: Optional list of specific GPU indices to use
            
        Returns:
            tuple[int, list[int]]: Number of allocated accelerators and their indices
        """
        pass

class NvidiaManager(AcceleratorManager):
    """
    NVIDIA GPU manager implementation.
    Handles allocation and management of NVIDIA GPUs.
    """
    
    def __init__(self, allocation_strategy: str = "default"):
        """
        Initialize NVIDIA GPU manager.
        
        Args:
            allocation_strategy: Strategy for GPU allocation, defaults to "default"
        """
        self._gpu_path = "/proc/driver/nvidia/gpus/"
        self._allocation_status = []  # Track allocation count per GPU
        self._allocation_strategy = allocation_strategy
        self._init_allocation_status()
    
    def _init_allocation_status(self):
        """Initialize GPU allocation tracking"""
        count = self.get_accelerator_count()
        self._allocation_status = [0] * count
        
    def get_accelerator_count(self) -> int:
        """
        Get count of available NVIDIA GPUs.
        
        Returns:
            int: Number of available GPUs, 0 if none found
            
        Raises:
            FileNotFoundError: If NVIDIA driver path not found
        """
        try:
            # 首先检查路径是否存在
            if not os.path.exists(self._gpu_path):
                logging.error(f"Path not found: {self._gpu_path}")
                return 0
                
            # 检查是否有GPU设备文件
            count = len([name for name in os.listdir(self._gpu_path) 
                        if os.path.isdir(os.path.join(self._gpu_path, name))])
            
            # 额外验证：检查是否可以访问/dev/nvidia*设备
            nvidia_devices = [dev for dev in os.listdir("/dev") if dev.startswith("nvidia") and not dev.endswith("ctl")]
            if not nvidia_devices:
                logging.warning("Found GPU paths in /proc but no accessible /dev/nvidia* devices. This may indicate GPU driver paths are mounted but GPUs are not accessible.")
                return 0
                
            logging.info(f"Found {count} NVIDIA GPUs")
            return count
        except (FileNotFoundError, PermissionError) as e:
            logging.error(f"GPU detection error: {str(e)}")
            return 0
            
    def allocate_accelerators(self, model_type: str, required_count: int = 1, gpu_idxs: List[int] = None) -> tuple[int, list[int]]:
        """
        Allocate GPUs based on model requirements.
        
        Args:
            model_type: Type of model requesting allocation
            required_count: Number of GPUs needed
            gpu_idxs: Optional list of specific GPU indices to use
            
        Returns:
            tuple[int, list[int]]: Number of allocated GPUs and their indices
            
        Raises:
            Exception: If no GPUs available or insufficient GPUs
        """
        total_gpus = self.get_accelerator_count()
        
        if total_gpus == 0:
            raise Exception("No GPUs available")
            
        if gpu_idxs is not None:
            return self._custom_allocation(model_type, required_count, total_gpus, gpu_idxs)
        else:
            return self._default_allocation(model_type, required_count, total_gpus)

    def _default_allocation(self, model_type: str, required_count: int, total_gpus: int) -> tuple[int, list[int]]:
        """
        Default GPU allocation strategy.
        
        Args:
            model_type: Type of model requesting allocation
            required_count: Number of GPUs needed  
            total_gpus: Total available GPUs
            
        Returns:
            tuple[int, list[int]]: Number of allocated GPUs and their indices
            
        Raises:
            Exception: If insufficient GPUs available
        """
        if model_type == 'LLM':
            # Allocate all GPUs for LLM models
            self._allocation_status = [v+1 for v in self._allocation_status]
            logging.info(f"Allocated all {total_gpus} GPUs for LLM model")
            return total_gpus, list(range(total_gpus))
        else:
            # Allocate least loaded GPUs for other models
            if total_gpus < required_count:
                raise Exception(f"Insufficient GPUs, {required_count} required but only {total_gpus} available")
                
            available_gpus = []
            for _ in range(required_count):
                idx = self._allocation_status.index(min(self._allocation_status))
                self._allocation_status[idx] += 1
                available_gpus.append(idx)
            
            logging.info(f"Allocated GPUs {available_gpus} for {model_type} model")    
            return required_count, available_gpus

    def _custom_allocation(self, model_type: str, required_count: int, total_gpus: int, gpu_idxs: List[int]) -> tuple[int, list[int]]:
        """
        Custom GPU allocation using specified indices.
        
        Args:
            model_type: Type of model requesting allocation
            required_count: Number of GPUs needed
            total_gpus: Total available GPUs
            gpu_idxs: List of specific GPU indices to use
            
        Returns:
            tuple[int, list[int]]: Number of allocated GPUs and their indices
            
        Raises:
            ValueError: If invalid GPU indices specified
            Exception: If insufficient GPUs specified
        """
        # Validate GPU indices
        for idx in gpu_idxs:
            if idx >= total_gpus or idx < 0:
                raise ValueError(f"Invalid GPU index {idx}, must be between 0 and {total_gpus-1}")
        
        # Verify sufficient GPUs
        if len(gpu_idxs) < required_count:
            raise Exception(f"Insufficient GPUs specified, {required_count} required but only {len(gpu_idxs)} provided")
        
        # Allocate specified GPUs
        for idx in gpu_idxs:
            self._allocation_status[idx] += 1
            
        logging.info(f"Custom allocated GPUs {gpu_idxs} for {model_type} model")
        return len(gpu_idxs), gpu_idxs

class CpuManager(AcceleratorManager):
    """
    CPU manager implementation.
    Handles allocation and management of CPU resources.
    """
    
    def __init__(self):
        """Initialize CPU manager"""
        self._cpu_count = os.cpu_count() or 1
        
    def get_accelerator_count(self) -> int:
        """
        Get count of available CPU cores.
        
        Returns:
            int: Number of available CPU cores
        """
        logging.info(f"Using CPU mode with {self._cpu_count} cores")
        return self._cpu_count
            
    def allocate_accelerators(self, model_type: str, required_count: int = 1, gpu_idxs: List[int] = None) -> tuple[int, list[int]]:
        """
        Allocate CPU cores for model deployment.
        
        Args:
            model_type: Type of model requesting allocation
            required_count: Number of cores needed (ignored in CPU mode)
            gpu_idxs: Not used in CPU mode
            
        Returns:
            tuple[int, list[int]]: Always returns (1, [0]) as CPU allocation is handled by the system
        """
        logging.info(f"Running {model_type} model in CPU mode")
        return 1, [0]

class ConfigValidator(Protocol):
    """Protocol defining configuration validation interface"""
    def __call__(self, config: Dict[str, Any]) -> bool:
        ...

class ConfigProcessor(Protocol):
    """Protocol defining configuration processing interface"""
    def __call__(self, config: Dict[str, Any]) -> Dict[str, Any]:
        ...

class ModelConfigRegistry:
    """
    Registry for model configuration validators and processors.
    Handles validation and processing of different model configurations.
    """
    
    def __init__(self):
        """Initialize registry with default validators and processors"""
        self._validators: Dict[str, Dict[str, ConfigValidator]] = {}
        self._processors: Dict[str, Dict[str, ConfigProcessor]] = {}
        self._version_detectors: List[Callable[[Dict[str, Any]], Optional[str]]] = []
        self._setup_default_validators()
        self._setup_default_processors()
        self._setup_version_detectors()
    
    def _setup_default_validators(self):
        """
        Set up default configuration validators for different model types and versions.
        Validates required fields in configurations.
        """
        # LLM validators
        self.register_validator("LLM", "epai1230", 
            lambda c: all(k in c for k in ["modelName", "modelFormat", "modelSizeInBillions", "quantizations"]))
        self.register_validator("LLM", "legacy",
            lambda c: all(k in c for k in ["modelName", "modelFormat", "modelFamily", "modelSizeInBillions", "quantizations"]))
            
        # Embedding validators
        self.register_validator("embedding", "epai1230",
            lambda c: all(k in c for k in ["modelName", "dimensions", "maxToken"]))
            
        # Rerank validators
        self.register_validator("rerank", "epai1230",
            lambda c: all(k in c for k in ["modelName", "type"]))
    
    def _setup_default_processors(self):
        """
        Set up default configuration processors for different model types and versions.
        Transforms raw configurations into standardized format.
        """
        def process_llm_epai1230(config: Dict[str, Any]) -> Dict[str, Any]:
            """Process LLM model configuration"""
            return {
                "modelType": config["modelType"],
                "modelName": config["modelName"],
                "modelUid": config.get("modelUid"),
                "modelPath": config.get("modelPath"),
                "modelUri": config.get("modelUri"),
                "modelFormat": config["modelFormat"],
                "modelSizeInBillions": config["modelSizeInBillions"],
                "quantizations": config.get("quantizations", ["None"]),
                "contextLength": config.get("contextLength", 2048),
                "kwargs": config.get("kwargs", {})
            }

        def process_llm_legacy(config: Dict[str, Any]) -> Dict[str, Any]:
            """Process legacy LLM model configuration"""
            return {
                "modelType": config.get("modelType", "LLM"),
                "modelName": config["modelFamily"],
                "modelUid": os.environ.get("MODEL_UID") or os.environ.get("MODEL_ID") or config["modelName"],  # Use modelName as model_id for legacy
                "modelUri": config.get("modelUri"),
                "modelFormat": config["modelFormat"],
                "modelSizeInBillions": config["modelSizeInBillions"],
                "quantizations": config.get("quantizations", ["None"]),
                "contextLength": config.get("contextLength", 2048)
            }
        
        def process_embedding_epai1230(config: Dict[str, Any]) -> Dict[str, Any]:
            """Process embedding model configuration"""
            return {
                "modelType": config["modelType"],
                "modelName": config["modelName"],
                "modelUid": config.get("modelUid"),
                "modelPath": config.get("modelPath"),
                "modelUri": config.get("modelUri"),
                "dimensions": config["dimensions"],
                "maxToken": config["maxToken"]
            }
        
        def process_rerank_epai1230(config: Dict[str, Any]) -> Dict[str, Any]:
            """Process rerank model configuration"""
            return {
                "modelType": config["modelType"],
                "modelName": config["modelName"],
                "modelUid": config.get("modelUid"),
                "modelPath": config.get("modelPath"),
                "modelUri": config.get("modelUri"),
                "type": config["type"]
            }
            
        self.register_processor("LLM", "epai1230", process_llm_epai1230)
        self.register_processor("LLM", "legacy", process_llm_legacy)
        self.register_processor("embedding", "epai1230", process_embedding_epai1230)
        self.register_processor("rerank", "epai1230", process_rerank_epai1230)
    
    def _setup_version_detectors(self):
        """
        Set up configuration version detectors.
        Detects version of configuration format.
        """
        def detect_epai1230(config: Dict[str, Any]) -> Optional[str]:
            """Detect epai1230 version"""
            return "epai1230" if config.get("configVersion") == "epai1230" else None
            
        def detect_legacy(config: Dict[str, Any]) -> Optional[str]:
            """Detect legacy version"""
            return "legacy" if "configVersion" not in config else None
            
        self._version_detectors.extend([detect_epai1230, detect_legacy])
    
    def register_validator(self, model_type: str, version: str, validator: ConfigValidator):
        """
        Register configuration validator.
        
        Args:
            model_type: Type of model
            version: Configuration version
            validator: Validator function
        """
        if model_type not in self._validators:
            self._validators[model_type] = {}
        self._validators[model_type][version] = validator
    
    def register_processor(self, model_type: str, version: str, processor: ConfigProcessor):
        """
        Register configuration processor.
        
        Args:
            model_type: Type of model
            version: Configuration version
            processor: Processor function
        """
        if model_type not in self._processors:
            self._processors[model_type] = {}
        self._processors[model_type][version] = processor
    
    def detect_version(self, config: Dict[str, Any]) -> str:
        """
        Detect configuration version.
        
        Args:
            config: Model configuration
            
        Returns:
            str: Detected version or "legacy" if unknown
        """
        for detector in self._version_detectors:
            if version := detector(config):
                return version
        return "legacy"
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            bool: True if valid, False otherwise
        """
        model_type = config.get("modelType", "LLM")
        version = self.detect_version(config)
        
        if model_type not in self._validators or version not in self._validators[model_type]:
            return False
            
        return self._validators[model_type][version](config)
    
    def process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Dict[str, Any]: Processed configuration
            
        Raises:
            ValueError: If unsupported model type or version
        """
        model_type = config.get("modelType", "LLM")
        version = self.detect_version(config)
        logging.info(f"Detect model type {model_type} with version {version}")
        
        if model_type not in self._processors or version not in self._processors[model_type]:
            raise ValueError(f"Unsupported model type or version: {model_type}, {version}")
            
        return self._processors[model_type][version](config)

class ModelConfigManager:
    """
    Manager for model configurations.
    Handles loading and retrieving model configurations.
    """
    
    def __init__(self):
        """Initialize configuration manager"""
        self.registry = ModelConfigRegistry()
        self.configs: List[Dict[str, Any]] = []
    
    def load_config_file(self, file_path: str) -> None:
        """
        Load configurations from file.
        
        Args:
            file_path: Path to configuration file
            
        Raises:
            FileNotFoundError: If configuration file not found
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, "r") as f:
            content = json.load(f)
        
        if isinstance(content, dict):
            content = [content]
        
        for config in content:
            try:
                if self.registry.validate_config(config):
                    processed_config = self.registry.process_config(config)
                    self.configs.append(processed_config)
                else:
                    logging.warning(f"Invalid configuration: {config}")
            except Exception as e:
                logging.error(f"Failed to process configuration: {str(e)}, config: {config}\nStacktrace:\n{traceback.format_exc()}")
    
    def get_configs_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get configurations by model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            List[Dict[str, Any]]: List of matching configurations
        """
        return [config for config in self.configs 
                if config.get("modelType", "LLM") == model_type]
    
    def get_config_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration by model name.
        
        Args:
            model_name: Name of model
            
        Returns:
            Optional[Dict[str, Any]]: Matching configuration or None
        """
        return next((config for config in self.configs 
                    if config["modelName"] == model_name), None)

class ModelLauncher:
    """
    Manager for launching models.
    Handles model deployment and configuration.
    """
    
    def __init__(self, client: Client, args: argparse.Namespace):
        """
        Initialize model launcher.
        
        Args:
            client: Xinference client instance
            args: Command line arguments
        """
        self.client = client
        self.args = args
        
        # Try to initialize NVIDIA manager first
        nvidia_manager = NvidiaManager()
        if nvidia_manager.get_accelerator_count() > 0:
            self.accelerator_manager = nvidia_manager
            logging.info("Using NVIDIA GPU acceleration")
        else:
            self.accelerator_manager = CpuManager()
            logging.info("No NVIDIA GPUs found, falling back to CPU mode")

    def _determine_model_engine(self, model_name: str, model_format: str, n_gpu: Union[int, str]) -> str:
        """
        Determine appropriate model engine.
        
        Args:
            model_name: Name of model
            model_format: Format of model
            
        Returns:
            str: Selected engine name
            
        Raises:
            Exception: If no suitable engine found
        """
        model_list = self.client.list_model_registrations('LLM')
        if not [m for m in model_list if not m['model_name'] == model_name]:
            raise Exception("No matching model found")
        engine_list = self.client.query_engine_by_model_name(model_name)

        # Validate specified engine
        if self.args.model_engine:
            if self.args.model_engine in engine_list:
                return self.args.model_engine
            else:
                raise Exception(f"Invalid model-engine parameter, please choose from: {str(engine_list)}")

        # Select engine based on format
        if model_format.lower() in ['pytorch', 'gptq', 'awq']:
            if n_gpu == 0 or n_gpu == "auto" and 'Transformers' in engine_list:
                return 'Transformers'
            if 'vLLM' in engine_list and os.environ.get("open_acceleration") == "True":
                return 'vLLM'
            elif 'Transformers' in engine_list:
                return 'Transformers'
            elif 'vLLM' in engine_list and os.environ.get("open_acceleration") == "False":
                return 'vLLM'
        elif model_format.lower() == 'ggufv2':
            if 'llama.cpp' in engine_list:
                return 'llama.cpp'
                
        raise Exception(
            f"No suitable engine found for model format {model_format}.\n"
            f"Available engines: {str(engine_list)}\n"
            f"Supported formats: pytorch/gptq/awq (vLLM/Transformers), ggufv2 (llama.cpp)"
        )
    
    def _prepare_launch_params(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare model launch parameters.
        
        Args:
            model_config: Model configuration
            
        Returns:
            Dict[str, Any]: Launch parameters
        """
        model_type = model_config.get("modelType", "LLM")
        
        # Handle GPU allocation
        gpu_idxs = model_config.get("gpu_idxs")
        if gpu_idxs:
            n_gpu = len(gpu_idxs)
            gpu_idx = gpu_idxs
        else:
            try:
                # Check if we're using CPU manager
                if isinstance(self.accelerator_manager, CpuManager):
                    n_gpu = 'auto'  # Set to 'auto' for CPU mode
                    gpu_idx = []
                    logging.info("Using CPU mode, setting n_gpu to 'auto'")
                else:
                    n_gpu, gpu_idx = self.accelerator_manager.allocate_accelerators(
                        model_type=model_type,
                        required_count=1
                    )
                    # If no GPUs available or allocation returned 0, set to 'auto'
                    if n_gpu == 0:
                        n_gpu = 'auto'
                        logging.info("No GPUs allocated, setting n_gpu to 'auto'")
            except Exception as e:
                logging.warning(f"GPU allocation failed: {str(e)}, falling back to CPU mode")
                n_gpu = 'auto'
                gpu_idx = []
            
        # Base parameters
        launch_params = {
            "model_name": self.args.model_name or model_config.get("modelName"),
            "model_uid": model_config.get("modelUid") or self.args.model_uid or os.environ.get("MODEL_UID") or os.environ.get("MODEL_ID") or model_config["modelName"],
            "model_type": self.args.model_type or model_type,
            "n_gpu": n_gpu,
            "gpu_idx": gpu_idx,
            "model_path": model_config.get("modelPath", "/mnt/models") or "/mnt/models"
        }
        
        # Model specific parameters
        if model_type == "LLM":
            quantizations = model_config.get("quantizations", ["None"])
            quantization = "none" if quantizations[0] == "None" else quantizations[0]
            model_engine = self.args.model_engine or self._determine_model_engine(
                model_name=launch_params["model_name"],
                model_format=self.args.model_format or model_config.get("modelFormat"),
                n_gpu=n_gpu
            )
            launch_params.update({
                "model_size_in_billions": self.args.size_in_billions or model_config.get("modelSizeInBillions"),
                "model_format": self.args.model_format or model_config.get("modelFormat"),
                "quantization": self.args.quantization or quantization,
                "model_engine": model_engine
            })
            
            # Add contextLength to max_model_len for vLLM engine
            if model_engine == "vLLM" and "contextLength" in model_config:
                # Set max_model_len directly in launch_params
                launch_params["max_model_len"] = model_config["contextLength"]
                logging.info(f"Setting max_model_len to {model_config['contextLength']} for vLLM engine")
            elif model_engine == "Transformers" and "contextLength" in model_config:
                # For Transformers engine, we need to set it in the kwargs
                launch_params["model_max_length"] = model_config["contextLength"]
                logging.info(f"Setting model_max_length to {model_config['contextLength']} for Transformers engine")
            elif model_engine == "SGLang" and "contextLength" in model_config:
                # For SGLang engine, we need to set it in the kwargs
                launch_params["context_length"] = model_config["contextLength"]
                logging.info(f"Setting context_length to {model_config['contextLength']} for SGLang engine")
            
        # Additional parameters
        if "kwargs" in model_config:
            launch_params.update(**model_config["kwargs"])
            
        return launch_params
        
    def launch_model(self, model_config: Dict[str, Any]) -> None:
        """
        Launch single model.
        
        Args:
            model_config: Model configuration
            
        Raises:
            Exception: If launch fails
        """
        try:
            launch_params = self._prepare_launch_params(model_config)
            logging.info(f"Launching model with params: {launch_params}")
            self.client.launch_model(**launch_params)
        except Exception as e:
            logging.error(f"Failed to launch model: {str(e)}\nStacktrace:\n{traceback.format_exc()}")
            raise
            
    def launch_models(self, model_configs: List[Dict[str, Any]]) -> None:
        """
        Launch multiple models.
        
        Args:
            model_configs: List of model configurations
        """
        for config in model_configs:
            self.launch_model(config)

class ModelProbe:
    """
    Model probe checker.
    Generates probe file for model health checks.
    """
    
    def __init__(self, port: int, model_names: List[str]):
        """
        Initialize model probe.
        
        Args:
            port: Service port number
            model_names: List of model names to check
        """
        self.port = port
        self.model_names = model_names
        self.base_url = f"http://0.0.0.0:{port}"
        self.sensitive_url = f"http://0.0.0.0:{os.environ.get('SENSITIVE_SVC_PORT', 39998)}"
        
    def generate_probe_file(self) -> None:
        """Generate probe check file with all necessary checks"""
        probe_content = self._generate_basic_check()
        probe_content += self._generate_model_check()
        probe_content += self._generate_sensitive_check()
        
        with open("probe.py", "w") as f:
            f.write(probe_content)
            
    def _generate_basic_check(self) -> str:
        """Generate basic service check code"""
        return f'''
import requests
import os

address = "{self.base_url}"
instances_url = "/v1/models/instances"

# Check model instance status
resp = requests.get(address + instances_url, timeout=30)
if resp.status_code != 200:
    raise Exception("Failed to get model instances")

need_ready = set({self.model_names})
instances = resp.json()
ready = set([model["model_name"] for model in instances if model["status"] == "READY"])
if ready < need_ready:
    raise Exception(f"Models not ready")
'''

    def _generate_model_check(self) -> str:
        """Generate model-specific health check code"""
        return '''
# Check various model services
headers = {}
for instance in instances:
    model_id = instance["model_uid"]
    ability = instance["model_ability"]
    
    if "chat" in ability:
        # Check LLM service
        llm_payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Hello."}],
            "max_tokens": 1,
            "temperature": 0.7,
            "stream": False
        }
        resp = requests.post(address + "/v1/chat/completions", 
                           headers=headers, json=llm_payload, timeout=30)
        if resp.status_code != 200:
            raise Exception("LLM service error")
            
    elif "embed" in ability:
        # Check Embedding service
        embed_payload = {"model": model_id, "input": "Hello"}
        resp = requests.post(address + "/v1/embeddings",
                           headers=headers, json=embed_payload, timeout=30)
        if resp.status_code != 200:
            raise Exception("Embedding service error")
            
    elif "rerank" in ability:
        # Check Rerank service
        rerank_payload = {
            "model": model_id,
            "query": "test query",
            "documents": ["test passage"]
        }
        resp = requests.post(address + "/v1/rerank",
                           headers=headers, json=rerank_payload, timeout=30)
        if resp.status_code != 200:
            raise Exception("Rerank service error")
'''

    def _generate_sensitive_check(self) -> str:
        """Generate sensitive word service check code"""
        return f'''
# Check sensitive word service
if os.environ.get("SENSITIVE_MODEL_ENABLE", "false").lower() == "true":
    sensitive_address = "{self.sensitive_url}"
    resp = requests.get(sensitive_address + "/health", timeout=30)
    if resp.status_code != 200:
        raise Exception("Sensitive word service error")
'''

def main():
    """
    Main entry point.
    Handles command line arguments and orchestrates model deployment.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Model deployment script')
    parser.add_argument('--port', type=int, required=True, help='Port number')
    parser.add_argument('--model-name', type=str, help='Model name')
    parser.add_argument('--model-type', type=str, help='Model type')
    parser.add_argument('--size-in-billions', type=int, help='Model size in billions of parameters')
    parser.add_argument('--model-uid', type=str, help='Model unique identifier')
    parser.add_argument('--model-format', type=str, help='Model format')
    parser.add_argument('--quantization', type=str, help='Quantization method')
    parser.add_argument('--model-engine', type=str, help='Model engine')
    parser.add_argument('--model-config', type=str, help='Model configuration path')
    parser.add_argument('--model-path', type=str, help='Model base directory path')
    
    args = parser.parse_args()
    
    try:
        # Initialize client
        client = Client(f"http://0.0.0.0:{args.port}")
        
        # Load and process configurations
        config_manager = ModelConfigManager()
        if args.model_config:
            config_path = args.model_config
        elif args.model_path:
            config_path = os.path.join(args.model_path, "Param.json")
        else:
            config_path = "/mnt/models/Param.json"
        
        logging.info(f"Loading configuration from: {config_path}")
        config_manager.load_config_file(config_path)
        
        # Launch models
        launcher = ModelLauncher(client, args)
        launcher.launch_models(config_manager.configs)
        
        # Generate probe file
        model_names = [config["modelName"] for config in config_manager.configs]
        probe = ModelProbe(args.port, model_names)
        probe.generate_probe_file()
        
        # Monitor service status
        logging.info("Starting service status monitoring...")
        while True:
            try:
                res = requests.get(f"http://0.0.0.0:{args.port}/status", timeout=5)
                if res.status_code != 200:
                    logging.error("Model service error")
                    raise Exception("Model service not running")
                time.sleep(1)
            except requests.exceptions.RequestException as e:
                logging.error(f"Service status check failed: {str(e)}")
                raise
                
    except Exception as e:
        logging.error(f"Program execution error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()