# docker-iei/ichat/serve.py

import asyncio
import signal
from contextlib import asynccontextmanager
from typing import Any

# Assume that the execution environment is configured so that 'ichat' is a package.
# This allows for consistent relative imports within the application.
try:
    from .backends.sglang_backend import SGLangBackend
    from .backends.vllm_backend import VLLMBackend
    from .config.args import parse_worker_args
    from .monitor.heartbeat import HeartbeatManager
    from .utils.logger import setup_logging
except ImportError as e:
    # Provide a helpful error message if modules can't be found,
    # often due to incorrect execution context.
    print(
        "ERROR: Failed to import modules. Please ensure you are running from the correct directory "
        "and the Python path is set up properly."
    )
    print(f"ImportError: {e}")
    # Exit since the application cannot start without its core components.
    exit(1)


# Global event to coordinate a graceful shutdown across the application.
shutdown_event = asyncio.Event()


def _signal_handler(*_: Any) -> None:
    """
    Signal handler for SIGINT and SIGTERM.

    When a shutdown signal is received, this function sets the global
    `shutdown_event`. This event acts as a beacon for all asynchronous tasks,
    prompting them to begin their cleanup and shutdown procedures.
    """
    print("INFO:     Shutdown signal received. Initiating graceful shutdown...")
    shutdown_event.set()


@asynccontextmanager
async def lifespan(args: Any):
    """
    An asynchronous context manager to manage the lifecycle of background services.

    It ensures that tasks like the heartbeat manager are started and stopped
    cleanly along with the main application.
    """
    heartbeat_manager = None
    # The heartbeat service is only started if a gateway address is provided,
    # enabling integration with the iChat Gateway.
    if getattr(args, "gateway_address", None):
        heartbeat_manager = HeartbeatManager(args)
        # The heartbeat runs as a background task.
        asyncio.create_task(heartbeat_manager.start())

    try:
        # Yield control back to the main application logic.
        yield
    finally:
        # This block is executed on exit, ensuring graceful shutdown of background tasks.
        if heartbeat_manager:
            print("INFO:     Stopping heartbeat manager...")
            await heartbeat_manager.stop()
        print("INFO:     Worker has shut down.")


async def main() -> None:
    """
    The main asynchronous entry point for the iChat Worker.

    This function orchestrates the entire worker lifecycle, from parsing arguments
    and setting up logging to running the inference backend and managing
    a graceful shutdown.
    """
    # 1. Parse iChat framework-specific arguments, separating them from backend-specific ones.
    framework_args, backend_argv = parse_worker_args()

    # 2. Set up the logging system. Based on configuration, logs can be
    # streamed to the gateway for centralized monitoring.
    setup_logging(
        log_level=getattr(framework_args, "log_level", "INFO"),
        stream_to_gateway=getattr(framework_args, "log_streaming", False),
        gateway_address=getattr(framework_args, "gateway_address", None),
    )

    # 3. Dynamically instantiate the specified inference backend.
    if framework_args.backend == "vllm":
        backend = VLLMBackend(framework_args, backend_argv)
    elif framework_args.backend == "sglang":
        backend = SGLangBackend(framework_args, backend_argv)
    else:
        # If an unsupported backend is requested, fail immediately.
        raise ValueError(f"Unsupported backend: {framework_args.backend}")

    # 4. Run the main service within the lifespan context to manage background tasks.
    async with lifespan(framework_args):
        # The backend's run() method is the main long-running service task.
        server_task = asyncio.create_task(backend.run())

        # Wait for either the server to exit or a shutdown signal to be received.
        await asyncio.wait(
            [server_task, shutdown_event.wait()],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # If the shutdown was triggered by a signal, cancel the server task
        # to allow it to clean up and exit gracefully.
        if shutdown_event.is_set() and not server_task.done():
            server_task.cancel()
            # Wait for the task to acknowledge cancellation.
            await asyncio.sleep(1)


if __name__ == "__main__":
    # Register signal handlers to capture termination signals (e.g., Ctrl+C)
    # and trigger the graceful shutdown process.
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        # Run the main asynchronous application.
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Suppress exceptions that are expected during a graceful shutdown.
        print("INFO:     Main task cancelled. Exiting.")
        pass
