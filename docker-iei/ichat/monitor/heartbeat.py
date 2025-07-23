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

    def __init__(
        self,
        framework_args: Namespace,
        backend_args: Namespace,
        backend_ready: asyncio.Event,
    ):
        """
        Initializes the HeartbeatManager.

        Args:
            framework_args: The initial arguments parsed for the iChat framework.
            backend_args: The final, resolved arguments used by the backend,
                          which may include default values (e.g., for port).
            backend_ready: An asyncio.Event that is set when the backend is
                           fully initialized and ready to serve requests.
        """
        self.gateway_address = framework_args.gateway_address
        self.heartbeat_interval = framework_args.heartbeat_interval
        self.backend_ready = backend_ready

        # Construct the payload once. It will be sent with each heartbeat.
        # This uses a mix of framework arguments (user intent) and backend
        # arguments (runtime values).
        self.payload = {
            "worker_id": f"worker-{uuid.uuid4()}",
            "model_name": (
                framework_args.served_model_name
                or (framework_args.model_path or "").strip("/").split("/")[-1]
            ),
            "model_path": framework_args.model_path,
            "backend": framework_args.backend,
            "host": backend_args.host,
            "port": backend_args.port,
        }

        self._session: Optional[aiohttp.ClientSession] = None
        self._should_stop = asyncio.Event()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.state = "initializing"

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazily creates and returns the aiohttp ClientSession."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _send_heartbeat(self, state: str):
        """
        Sends a single heartbeat signal to the gateway.
        This also serves as the registration call, as the gateway
        is expected to handle this as an "upsert" operation.
        """
        session = await self._get_session()
        heartbeat_url = f"{self.gateway_address}/v1/workers/heartbeat"

        payload = self.payload.copy()
        payload["state"] = state

        try:
            # The first heartbeat acts as registration.
            # print(f"INFO:     Sending heartbeat for worker {self.payload['worker_id']} with state '{state}'...")
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
        """
        The main loop that periodically sends heartbeats.
        It starts by sending 'initializing' state, then switches to 'ready'
        once the backend signals it's ready.
        """
        print("INFO:     Heartbeat service started. Will send initial heartbeats as 'initializing'.")

        async def wait_for_backend_ready():
            """Waits for the backend to be ready and updates the state."""
            try:
                await self.backend_ready.wait()
                self.state = "ready"
                print("INFO:     Backend is ready. Heartbeats will now be sent with state 'ready'.")
            except asyncio.CancelledError:
                pass # This is expected on shutdown

        ready_waiter_task = asyncio.create_task(wait_for_backend_ready())

        while not self._should_stop.is_set():
            await self._send_heartbeat(state=self.state)
            try:
                # Wait for the specified interval, but break immediately if
                # a stop signal is received.
                await asyncio.wait_for(
                    self._should_stop.wait(), timeout=self.heartbeat_interval
                )
            except asyncio.TimeoutError:
                # This is the expected behavior, triggering the next heartbeat.
                pass
        
        ready_waiter_task.cancel()

    async def start(self):
        """
        Starts the heartbeat service. It runs the heartbeat loop in a background
        task, which will manage the worker's state transitions.
        """
        if not self.gateway_address:
            print("INFO:     Gateway address not provided. Heartbeat service disabled.")
            return

        print(f"INFO:     Starting heartbeat service for worker {self.payload['worker_id']}.")

        # The loop now handles all state transitions internally.
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self):
        """
        Stops the heartbeat manager gracefully, sending a final 'terminating'
        heartbeat before shutting down.
        """
        if self._should_stop.is_set() or not self._heartbeat_task:
            return

        # Send a final heartbeat to notify the gateway of graceful shutdown
        self.state = "terminating"
        print(f"INFO:     Sending final heartbeat for worker {self.payload['worker_id']} with state 'terminating'...")
        await self._send_heartbeat(state=self.state)

        print("INFO:     Signaling heartbeat loop to stop...")
        self._should_stop.set()

        if self._heartbeat_task:
            # Wait for the heartbeat task to finish its current cycle and exit.
            # Add a timeout to prevent hanging.
            try:
                await asyncio.wait_for(self._heartbeat_task, timeout=5.0)
            except asyncio.TimeoutError:
                print("WARNING:  Heartbeat task did not stop gracefully. Cancelling.")
                self._heartbeat_task.cancel()

        if self._session:
            await self._session.close()

        print("INFO:     Heartbeat manager stopped.")
