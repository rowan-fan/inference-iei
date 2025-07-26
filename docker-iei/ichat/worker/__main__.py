
import asyncio
import signal
from contextlib import asynccontextmanager
from typing import Any, Optional
from argparse import Namespace

# Assume that the execution environment is configured so that 'ichat' is a package.
# This allows for consistent relative imports within the application.
try:
    from .args import parse_worker_args
    from .heartbeat import HeartbeatManager
    from ..utils.logger import setup_logging
    from ..backends.base_backend import BaseBackend
except ImportError as e:
    # Provide a helpful error message if modules can't be found,
    # often due to incorrect execution context.
    print("ERROR: Failed to import modules. This script is designed to be run as a Python module.")
    print("Please run it from the parent directory of 'ichat', for example:")
    print("  python3 -m ichat.worker [arguments]")
    print(f"Original ImportError: {e}")
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
async def lifespan(heartbeat_manager: Optional[HeartbeatManager]):
    """
    An asynchronous context manager to manage the lifecycle of the HeartbeatManager.
    It ensures that the heartbeat service is started and stopped cleanly along
    with the main application.
    """
    if heartbeat_manager:
        asyncio.create_task(heartbeat_manager.start())

    try:
        yield
    finally:
        if heartbeat_manager and not shutdown_event.is_set():
            print("INFO:     Stopping heartbeat manager...")
            await heartbeat_manager.stop()


async def main() -> None:
    """
    The main asynchronous entry point for the iChat Worker.

    This function orchestrates the entire worker lifecycle, from parsing arguments
    and setting up logging to running the inference backend and managing
    a graceful shutdown.
    """
    # 1. Parse iChat framework-specific arguments, separating them from backend-specific ones.
    framework_args, backend_argv = parse_worker_args()

    # 2. Set up logging.
    setup_logging(log_level=framework_args.log_level)
    # Backend-ready event to coordinate with the heartbeat manager.
    backend_ready_event = asyncio.Event()

    # 3. Dynamically instantiate the specified inference backend.
    if framework_args.backend == "vllm":
        from ..backends.vllm.backend import VLLMBackend
        backend = VLLMBackend(framework_args, backend_argv, backend_ready_event)
    elif framework_args.backend == "sglang":
        from ..backends.sglang.backend import SGLangBackend
        backend = SGLangBackend(framework_args, backend_argv, backend_ready_event)
    elif framework_args.backend == "sentence":
        from ..backends.sentence_transformer.backend import SentenceBackend
        backend = SentenceBackend(framework_args, backend_argv, backend_ready_event)
    elif framework_args.backend == "ollama":
        from ..backends.ollama.backend import OllamaBackend
        backend = OllamaBackend(framework_args, backend_argv, backend_ready_event)
    else:
        raise ValueError(f"Unsupported backend: {framework_args.backend}")

    # 4. Get the final backend arguments for heartbeat payload after backend-specific parsing.
    final_backend_args = backend.get_backend_args()

    # 5. Initialize HeartbeatManager
    heartbeat_manager = None
    if getattr(framework_args, "gateway_address", None):
        heartbeat_manager = HeartbeatManager(
            framework_args=framework_args,
            backend_args=final_backend_args,
            backend_ready=backend.server_ready,
        )

    # 6. Run the main service within the lifespan context.
    async with lifespan(heartbeat_manager):
        # Create tasks for the backend and for waiting on a shutdown signal.
        server_task = asyncio.create_task(backend.run())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        # Wait for either the server to complete or a shutdown signal to be received.
        done, pending = await asyncio.wait(
            {server_task, shutdown_task},  # Use a set for clarity
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Check if the backend task is the one that completed
        if server_task in done:
            # If the backend task finished, it might be a crash or a normal exit.
            # If the backend wasn't ready, it's a startup failure.
            if not backend.is_server_ready() and heartbeat_manager:
                # This indicates a startup failure. Enter zombie mode.
                await heartbeat_manager.enter_zombie_mode()
                # The process will now hang here, sending 'terminating' heartbeats.
                # It will only exit if the gateway/user kills it.
                await asyncio.Event().wait()  # Wait indefinitely

            # If it was a normal exit or post-startup crash, trigger graceful shutdown.
            if not shutdown_event.is_set():
                print("INFO:     Backend task completed. Initiating graceful shutdown...")
                shutdown_event.set()
        
        # At this point, a graceful shutdown has been initiated.
        # All pending tasks must be cancelled.
        for task in pending:
            task.cancel()
            
        # Await all original tasks to ensure they complete their cancellation/cleanup.
        # This prevents the script from exiting prematurely or hanging.
        await asyncio.gather(server_task, shutdown_task, return_exceptions=True)

        # Log the reason for shutdown
        if server_task.done() and not shutdown_task.done():
            try:
                exc = server_task.exception()
                if exc:
                    print(f"ERROR:    Backend exited unexpectedly with an exception: {exc}")
                else:
                    print("ERROR:    Backend exited unexpectedly without an exception.")
            except asyncio.CancelledError:
                # This can happen if shutdown was initiated elsewhere.
                print("INFO:     Backend task was cancelled during shutdown.")
        else:
            print("INFO:     Shutdown signal received, server has stopped.")

        # Final cleanup for the heartbeat manager if it was running
        if heartbeat_manager and heartbeat_manager._heartbeat_task and not heartbeat_manager._heartbeat_task.done():
            await heartbeat_manager.stop()


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