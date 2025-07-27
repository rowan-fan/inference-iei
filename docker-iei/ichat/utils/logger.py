import aiohttp
import asyncio
import logging
import queue
import threading
from logging.handlers import QueueHandler
from typing import Optional

# Global queue to hold log records for streaming
log_queue = queue.Queue()


class GatewayLogHandler(logging.Handler):
    """
    A custom logging handler that puts log records into a thread-safe queue.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def emit(self, record):
        log_queue.put(record)


class LogStreamer:
    """
    Manages the streaming of log records from a queue to the iChat Gateway.
    """

    def __init__(self, gateway_address: str):
        self.gateway_address = gateway_address
        self._session: Optional[aiohttp.ClientSession] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._should_stop = threading.Event()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _log_stream_loop(self):
        """
        The main loop for pulling logs from the queue and sending them to the gateway.
        """
        session = await self._get_session()
        stream_url = f"{self.gateway_address}/api/v1/logs/stream"

        while not self._should_stop.is_set():
            try:
                # Use a short timeout to remain responsive to the stop signal
                record = log_queue.get(block=True, timeout=0.5)
            except queue.Empty:
                continue

            try:
                log_entry = {
                    "level": record.levelname,
                    "message": self.format(record),
                }
                async with session.post(stream_url, json=log_entry, timeout=5) as response:
                    if response.status != 200:
                        print(
                            f"WARNING:  Failed to stream log to gateway. Status: {response.status}"
                        )
            except aiohttp.ClientError as e:
                print(f"WARNING:  Error streaming log to gateway: {e}")
                # Wait before retrying to avoid spamming a down gateway
                await asyncio.sleep(5)
            except Exception as e:
                print(f"ERROR:    Unexpected error in log streaming loop: {e}")

    def format(self, record: logging.LogRecord) -> str:
        """
        Formats a log record into a string, similar to a standard Formatter.
        """
        return f"[{record.filename}:{record.lineno}] {record.getMessage()}"

    def start(self):
        """Starts the log streaming task in the background."""
        print("INFO:     Starting log streamer...")
        self._stream_task = asyncio.create_task(self._log_stream_loop())
        print("INFO:     Log streamer started.")

    async def stop(self):
        """Signals the log streamer to stop and waits for it to finish."""
        if self._should_stop.is_set():
            return

        print("INFO:     Signaling log streamer to stop...")
        self._should_stop.set()

        if self._stream_task:
            await self._stream_task
        if self._session:
            await self._session.close()

        print("INFO:     Log streamer stopped.")


def setup_logging(
    log_level: str = "INFO",
    stream_to_gateway: bool = False,
    gateway_address: Optional[str] = None,
):
    """
    Configures the root logger for the application.

    Args:
        log_level: The minimum log level to capture (e.g., "INFO", "DEBUG").
        stream_to_gateway: If True, streams logs to the iChat Gateway.
        gateway_address: The base address of the iChat Gateway.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure console logging
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        "%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Configure gateway log streaming if enabled
    if stream_to_gateway and gateway_address:
        gateway_handler = GatewayLogHandler()
        root_logger.addHandler(gateway_handler)

        log_streamer = LogStreamer(gateway_address=gateway_address)
        log_streamer.start()

        # It's crucial to properly handle the lifecycle of the streamer.
        # This basic setup assumes the application will manage stopping it.
        # A more robust solution might involve returning the streamer instance.
        # For now, we rely on the main application shutdown to handle it.
        # (This part of the logic might need refinement depending on the main app structure)
    elif stream_to_gateway and not gateway_address:
        logging.warning(
            "Log streaming was requested, but no gateway address was provided."
        )

    logging.info(f"Logging configured with level {log_level}.")
    if stream_to_gateway and gateway_address:
        logging.info(f"Streaming logs to gateway at {gateway_address}")
