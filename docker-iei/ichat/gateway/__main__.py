
# docker-iei/ichat/gateway/__main__.py

import argparse
import uvicorn
import yaml

# Import the components from the gateway sub-package
try:
    from . import api
    from .registry import ServiceRegistry
    from .worker_manager import WorkerManager
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
            config = yaml.safe_load(f)
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