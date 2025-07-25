
# docker-iei/ichat/gateway/__main__.py

import argparse
import uvicorn
import yaml
import os
import re
import json

# Import the components from the gateway sub-package
try:
    from . import api
    from .registry import ServiceRegistry
    from .worker_manager import WorkerManager
    from ..utils.common import get_default_gpu_ids
except ImportError as e:
    print("ERROR: Failed to import modules. This script is designed to be run as a Python module.")
    print("Please run it from the parent directory of 'ichat', for example:")
    print("  python3 -m ichat.gateway --config ichat/config.yaml")
    print(f"Original ImportError: {e}")
    exit(1)


def main():
    """Main entry point for the iChat Gateway."""

    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="iChat Gateway Service")
    parser.add_argument("--config", type=str, required=True, help="Path to the config.yaml file.")
    args = parser.parse_args()

    # 2. Load configuration from YAML file
    try:
        with open(args.config, "r") as f:
            raw_config = f.read()

        # Substitute environment variables of the form ${VAR_NAME} or ${VAR_NAME:-default}
        def env_substituter(match):
            var_name = match.group(1)
            default_part = match.group(2) # This is like ":-some_value" or empty
            
            # First, check the environment for the variable
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value

            # If not in env, and it's ICHAT_WORKER_GPU_IDS, use our auto-detection
            if var_name == 'ICHAT_WORKER_GPU_IDS':
                default_gpus = get_default_gpu_ids()
                # The YAML loader expects a JSON-like string, so we format it
                return json.dumps(default_gpus)

            # If not in env and a default is provided in the YAML, use it
            if default_part:
                return default_part[2:]

            # If not in env and no default, we'll return a special marker to signify removal
            return "__REMOVE_FIELD__"

        # Regex to find ${VAR_NAME} or ${VAR_NAME:-default_value}
        pattern = re.compile(r'\$\{([\w_]+)((?::-[^}]+)?)\}')
        substituted_config_str = pattern.sub(env_substituter, raw_config)
        
        config = yaml.safe_load(substituted_config_str)

        # Recursively remove any fields that were marked for removal
        def remove_marked_fields(obj):
            if isinstance(obj, dict):
                return {k: remove_marked_fields(v) for k, v in obj.items() if v != "__REMOVE_FIELD__"}
            elif isinstance(obj, list):
                return [remove_marked_fields(elem) for elem in obj if elem != "__REMOVE_FIELD__"]
            else:
                return obj

        config = remove_marked_fields(config)

    except FileNotFoundError:
        print(f"ERROR:    Config file not found at {args.config}")
        exit(1)
    
    server_settings = config.get("server_settings", {})
    host = server_settings.get("host", "0.0.0.0")
    port = server_settings.get("port", 4000)
    gateway_address = f"http://{host}:{port}"
    
    # 3. Initialize WorkerManager for managed workers
    worker_manager = None
    if "managed_workers" in config:
        worker_manager = WorkerManager(config["managed_workers"], gateway_address)

    # 4. Initialize ServiceRegistry
    service_registry = ServiceRegistry(
        heartbeat_timeout=server_settings.get("heartbeat_timeout", 30),
        worker_manager=worker_manager,
    )

    # 5. Inject dependencies into the API module
    api.service_registry = service_registry
    api.worker_manager = worker_manager

    # 6. Run the server
    uvicorn.run(
        api.app,
        host=host,
        port=port,
        log_level=server_settings.get("log_level", "info"),
    )

if __name__ == "__main__":
    main() 