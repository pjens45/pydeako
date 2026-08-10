"""
Manager to SocketConnection, ensuring that there's always an
active connection. Runs one background worker that checks
connectivity through pinging.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from ..discover import DevicesNotFoundException
from ..models import (
    device_list_request,
    device_ping_request,
    state_change_request,
    ResponseType,
)
from .utils import _Connection
from ._request import _Request

_LOGGER: logging.Logger = logging.getLogger(__package__)

CONNECTED_POLLING_INTERVAL_S = 1
CONNECTION_TIMEOUT_S = 10
WORKER_WAIT_S = 0.5
PING_WORKER_WAIT_S = 10


@dataclass
class _ManagerState:
    """State for _Manager, used by workers."""

    connecting: bool = False
    canceled: bool = False


# pylint: disable-next=too-many-instance-attributes
class _Manager:
    """Manage the socket connection to Deako local integrations."""

    maintain_worker: asyncio.Task | None = None
    connection: _Connection | None = None
    tasks: set[asyncio.Task]
    client_name: str | None
    state: _ManagerState

    def __init__(
        self,
        get_address,
        incoming_json_callback,
        client_name: str | None = None,
        on_connection_lost: Callable | None = None,
    ) -> None:
        """Initialize with get address function and incoming json callback.

        Args:
            get_address: Async callable returning (address, name) tuple.
            incoming_json_callback: Called with parsed JSON from the bridge.
            client_name: Optional client name sent in protocol messages.
            on_connection_lost: Optional callback invoked when the connection
                drops (ping timeout). Called before auto-reconnect starts.
        """
        self.get_address = get_address
        self.incoming_json_callback = incoming_json_callback
        self.pong_received = False
        self.tasks = set()
        self.client_name = client_name
        self.on_connection_lost = on_connection_lost
        # When False, every reconnect-scheduling call site is skipped
        # (init_connection devices-not-found, init_connection connect
        # timeout, maintain_connection_worker ping timeout). The pool
        # disables this so failover owns reconnect, not _Manager.
        # Default True preserves 0.x behavior for single-bridge callers.
        self.auto_reconnect: bool = True
        self.state = _ManagerState()

    async def init_connection(self) -> None:
        """Initialize the connection process."""
        if self.state.connecting:
            _LOGGER.error("Already attempting to connect")
            return
        self.state.connecting = True
        # Track the in-flight connection separately from self.connection
        # so the finally block can tear it down if this coroutine is
        # cancelled (or raises) before it is installed. A leaked
        # _Connection keeps a live run() task and holds the bridge's
        # single TCP slot until the process restarts -- the same zombie
        # class as the pool's partial-connect. `installed` guards the
        # success path so we don't close a connection we handed off.
        connection: _Connection | None = None
        installed = False
        try:
            try:
                address, name = await self.get_address()
            except DevicesNotFoundException:
                _LOGGER.warning("No devices to connect to")
                if self.auto_reconnect:
                    self.create_connection_task()
                return
            connection = _Connection(address, name, self.incoming_json)
            timeout = 0
            while (
                not connection.is_connected()
                and not connection.is_errored()
                and timeout < CONNECTION_TIMEOUT_S
            ):
                await asyncio.sleep(CONNECTED_POLLING_INTERVAL_S)
                timeout += CONNECTED_POLLING_INTERVAL_S
            if not connection.is_connected():
                # Only claim a retry when one is actually scheduled.
                # With auto_reconnect off (the pool's configuration)
                # reconnection is owned by the caller, and the old
                # unconditional "Trying again" made logs read as if
                # the manager were retrying when it was not.
                if self.auto_reconnect:
                    _LOGGER.error("Failed to connect. Trying again")
                    self.create_connection_task()
                else:
                    _LOGGER.error(
                        "Failed to connect; auto_reconnect is off, "
                        "leaving recovery to the caller",
                    )
                # connection is torn down by the finally block below
                # (not installed), so it isn't closed twice.
                return
            self.connection = connection
            installed = True
            # init connection watching
            if self.maintain_worker is None:
                self.state.canceled = False
                self.maintain_worker = asyncio.create_task(
                    self.maintain_connection_worker()
                )
        finally:
            # Always release the connecting latch, even on cancellation
            # or an unexpected error from get_address(); otherwise every
            # future init_connection() short-circuits on "Already
            # attempting to connect" and the manager wedges permanently.
            self.state.connecting = False
            # Close any connection that was created but never installed
            # (failed poll, cancelled mid-connect, unexpected raise) so
            # its run() task and socket don't outlive this call.
            if connection is not None and not installed:
                connection.close()

    def close(self) -> None:
        """Close connection."""
        _LOGGER.debug("Closing connection and canceling workers")
        self.state.canceled = True

        # Cancel and clear all pending tasks
        for task in self.tasks:
            task.cancel()
        self.tasks.clear()

        if self.maintain_worker is not None:
            self.maintain_worker.cancel()
            self.maintain_worker = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def create_connection_task(self):
        """Create an async task to initiate connection."""
        # RUF006
        # pylint: disable-next=line-too-long
        # noqa keep reference via: https://stackoverflow.com/questions/71938799/python-asyncio-create-task-really-need-to-keep-a-reference
        # even if we don't care
        task = asyncio.create_task(self.init_connection())
        self.tasks.add(task)

        def remove_task(_task):
            try:
                self.tasks.remove(_task)
            except KeyError:
                pass  # already removed

        task.add_done_callback(remove_task)

    async def maintain_connection_worker(self) -> None:
        """Monitor connection and restart if there's a failure."""
        await asyncio.sleep(PING_WORKER_WAIT_S)
        while True:
            if self.state.canceled:
                break
            self.pong_received = False
            _LOGGER.debug("Pinging for responsiveness")
            try:
                await self.send_request(
                    _Request(device_ping_request(source=self.client_name)),
                )
            except OSError as exc:
                # Send failure during health-check ping. Do not reraise;
                # falling through with pong_received=False lets the
                # "never received pong" branch below run the existing
                # connection-drop path (on_connection_lost + close +
                # reconnect under auto_reconnect).
                _LOGGER.warning("Ping send failed: %s", exc)
            await asyncio.sleep(PING_WORKER_WAIT_S)
            if self.pong_received:
                _LOGGER.debug("Pong received")
            else:
                _LOGGER.warning("Never received pong! Dumping this connection")
                if self.on_connection_lost is not None:
                    try:
                        self.on_connection_lost()
                    except Exception:  # pylint: disable=broad-exception-caught
                        _LOGGER.warning("on_connection_lost callback error")
                self.close()
                if self.auto_reconnect:
                    self.create_connection_task()
                break

    def incoming_json(self, incoming_json: dict) -> None:
        """Handle incoming json."""
        response_type = incoming_json.get("type")
        if response_type == ResponseType.PONG:
            self.pong_received = True
        else:
            self.incoming_json_callback(incoming_json)

    async def send_get_device_list(self) -> bool:
        """Send the device list request."""
        return await self.send_request(
            _Request(device_list_request(source=self.client_name)),
        )

    async def send_state_change(
        self,
        uuid,
        power,
        dim=None,
        completed_callback: Callable | None = None,
    ) -> bool:
        """Send a state change request.

        Returns True on successful enqueue, False only when there is no
        active connection. OSError from a real send failure propagates
        to the caller so failover / the pool can act on it.
        """
        return await self.send_request(
            _Request(
                state_change_request(
                    uuid, power, dim, source=self.client_name,
                ),
                completed_callback=completed_callback,
            )
        )

    async def send_request(self, req: _Request) -> bool:
        """Send a request."""
        if self.connection is not None:
            await self.connection.send_data(req.get_body_str())
            req.complete_callback()
            return True

        _LOGGER.warning("No connection to send data to")
        return False
