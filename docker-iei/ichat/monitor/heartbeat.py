# docker-iei/ichat/monitor/heartbeat.py

import asyncio
import uuid
from argparse import Namespace
from typing import Optional

import aiohttp


class HeartbeatManager:
    """
    Manages the registration and periodic heartbeating of a worker with the
    iChat Gateway. This ensures the gateway is aware of active workers and
    can route requests to them.
    """

    def __init__(self, args: Namespace):
        """
        Initializes the HeartbeatManager.

        Args:
            args: A Namespace object containing parsed command-line arguments,
                  including gateway configuration and worker details.
        """
        self.gateway_address = args.gateway_address
        self.heartbeat_interval = args.heartbeat_interval

        # Determine the model name for registration. Use the explicit
        # --served-model-name if provided, otherwise derive it from the model path.
        self.model_name = (
            args.served_model_name
            or args.model_path.strip("/").split("/")[-1]
        )

        # Generate a unique identifier for this specific worker process.
        self.worker_id = f"worker-{uuid.uuid4()}"

        # Construct the worker's address that the gateway will use to contact it.
        # Note: If the host is '0.0.0.0', this assumes the gateway can resolve
        # it correctly. In containerized environments, a reachable service name
        # or IP should be used.
        self.worker_addr = f"http://{args.host}:{args.port}"

        self._session: Optional[aiohttp.ClientSession] = None
        self._should_stop = asyncio.Event()
        self._heartbeat_task: Optional[asyncio.Task] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazily creates and returns the aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _register(self) -> bool:
        """
        Registers the worker with the gateway by sending its metadata.

        Returns:
            True if registration was successful, False otherwise.
        """
        session = await self._get_session()
        # This endpoint should correspond to the gateway's worker registration API.
        register_url = f"{self.gateway_address}/api/v1/workers"
        payload = {
            "worker_id": self.worker_id,
            "model_names": [self.model_name],
            "worker_addr": self.worker_addr,
        }

        try:
            print(f"INFO:     Registering worker {self.worker_id} for model '{self.model_name}' to gateway at {self.gateway_address}...")
            async with session.post(register_url, json=payload, timeout=10) as response:
                if response.status == 200:
                    print(f"INFO:     Worker {self.worker_id} registered successfully.")
                    return True
                else:
                    text = await response.text()
                    print(
                        f"ERROR:    Failed to register worker. Gateway returned status {response.status}: {text}"
                    )
                    return False
        except aiohttp.ClientError as e:
            print(f"ERROR:    Could not connect to gateway for registration: {e}")
            return False
        except asyncio.TimeoutError:
            print("ERROR:    Gateway registration request timed out.")
            return False

    async def _send_heartbeat(self):
        """Sends a single heartbeat signal to the gateway."""
        session = await self._get_session()
        # This endpoint is for sending periodic health checks.
        heartbeat_url = f"{self.gateway_address}/api/v1/heartbeat"
        payload = {"worker_id": self.worker_id}

        try:
            async with session.post(heartbeat_url, json=payload, timeout=10) as response:
                if response.status != 200:
                    text = await response.text()
                    print(
                        f"WARNING:  Failed to send heartbeat. Gateway returned status {response.status}: {text}"
                    )
        except aiohttp.ClientError as e:
            print(f"WARNING:  Could not send heartbeat to gateway: {e}")
        except asyncio.TimeoutError:
            print("WARNING:  Gateway heartbeat request timed out.")

    async def _heartbeat_loop(self):
        """The main loop that periodically sends heartbeats."""
        while not self._should_stop.is_set():
            await self._send_heartbeat()
            try:
                # Wait for the specified interval, but break immediately if
                # a stop signal is received.
                await asyncio.wait_for(
                    self._should_stop.wait(), timeout=self.heartbeat_interval
                )
            except asyncio.TimeoutError:
                # This is the expected behavior, triggering the next heartbeat.
                pass

    async def start(self):
        """
        Starts the heartbeat service.

        It first attempts to register the worker. If successful, it starts the
        periodic heartbeat loop in a background task.
        """
        # Retry registration a few times in case the gateway is not yet ready.
        for i in range(3):
            if await self._register():
                # On success, start the background heartbeat task.
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                return

            print(f"INFO:     Registration attempt {i + 1}/3 failed. Retrying in 5 seconds...")
            await asyncio.sleep(5)

        print(
            "ERROR:    Worker registration failed after multiple attempts. Heartbeat service will not start."
        )

    async def stop(self):
        """Stops the heartbeat manager gracefully."""
        if self._should_stop.is_set():
            return

        print("INFO:     Signaling heartbeat loop to stop...")
        self._should_stop.set()

        if self._heartbeat_task:
            # Wait for the heartbeat task to finish its current cycle and exit.
            await self._heartbeat_task

        if self._session:
            await self._session.close()

        print("INFO:     Heartbeat manager stopped.")
