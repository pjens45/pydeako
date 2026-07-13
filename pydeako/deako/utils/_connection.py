"""
Class that manages a socket connection to a deako device.
"""

import asyncio
import json
import logging
from enum import Enum
from typing import Callable

from ._socket import _SocketConnection

_LOGGER: logging.Logger = logging.getLogger(__package__)


class UnknownStateException(Exception):
    """Unknown state."""


class ConnectionState(Enum):
    """Enum for connection states."""

    UNKNOWN = -1
    NOT_STARTED = 0
    CONNECTED = 1
    ERROR = 2
    CLOSED = 3


class _Connection:
    """
    Representation of a local socket connection to a Deako device.
    Continuously reads socket for messages async.
    """

    # pylint: disable=too-many-instance-attributes

    address: str
    name: str
    message_buffer: str
    loop: asyncio.AbstractEventLoop
    state: ConnectionState
    socket: _SocketConnection
    tasks: set[asyncio.Task]

    def __init__(
        self, address: str, name: str, on_data_callback: Callable[[dict], None]
    ) -> None:
        """Setup and start a socket connection."""
        self.address = address
        self.name = name
        self.loop = asyncio.get_running_loop()
        self.state = ConnectionState.NOT_STARTED
        self.on_data_callback = on_data_callback
        self.message_buffer = ""
        self.socket = _SocketConnection(address, self.loop)
        self.tasks = set()
        self.init_run()

    async def send_data(self, data_to_send: str) -> None:
        """Send data to socket.

        On send failure, state must flip to ERROR immediately so the
        rest of the state machine (and higher-level callers like the
        pool) see the failure without polling. The ``except OSError``
        is intentionally narrow: TypeError, AttributeError, and other
        programming errors should propagate as bugs, not be converted
        into connection-state changes.
        """
        _LOGGER.debug("[%s] Sending data: %s", self.address, data_to_send)
        try:
            await self.socket.send_bytes(str.encode(data_to_send + "\r\n"))
        except OSError as exc:
            _LOGGER.error("Error sending data: %s", exc)
            self.state = ConnectionState.ERROR
            raise

    async def read_socket(self) -> None:
        """Read data from socket."""
        try:
            data = await self.socket.read_bytes()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.error("Error receiving data: %s", exc)
            self.state = ConnectionState.ERROR
            return

        if not data:
            # Peer closed the connection gracefully (EOF). sock_recv
            # returns b"" immediately -- and keeps returning it -- on
            # a half-closed socket, and asyncio's sock_recv resolves
            # synchronously in that case, so without this check the
            # run() loop spins with no yield point and blocks the
            # entire event loop.
            _LOGGER.warning(
                "[%s] Connection closed by peer", self.format_name(),
            )
            self.state = ConnectionState.ERROR
            return

        self.parse_data(data)

    def parse_data(self, data: bytes) -> None:
        """
        Parse incoming bytes into json as expected. Possible to have
        data come in multiple chunks and multiple messages.

        A segment that fails to parse is buffered and retried with
        the next segment appended (messages can be split across
        chunks and across delimiters). To keep a garbled or
        truncated fragment from poisoning the buffer forever --
        every later message would be appended to it, never parse,
        and leave the connection deaf until the ping timeout dumps
        it -- a segment that parses standalone while the combined
        buffer does not is delivered on its own and the stale
        buffer prefix is dropped.
        """
        raw_string = data.decode("utf-8")
        _LOGGER.debug(
            "[%s] Raw message received: %s",
            self.format_name(),
            raw_string,
        )
        messages = raw_string.strip().split("\r\n")
        for message_str in messages:
            self.message_buffer = self.message_buffer + message_str
            try:
                message_json = json.loads(self.message_buffer)
                self.on_data_callback(message_json)
                self.message_buffer = ""
                continue
            except json.decoder.JSONDecodeError:
                pass
            if self.message_buffer != message_str:
                # The combined buffer doesn't parse. If this segment
                # parses on its own, the buffered prefix is a dead
                # fragment: drop it and deliver the segment so one
                # bad chunk can't silence the connection.
                try:
                    message_json = json.loads(message_str)
                except json.decoder.JSONDecodeError:
                    pass
                else:
                    _LOGGER.warning(
                        "Dropping unparseable buffered fragment: %s",
                        self.message_buffer[: -len(message_str)],
                    )
                    self.on_data_callback(message_json)
                    self.message_buffer = ""
                    continue
            _LOGGER.debug("Got partial message: %s", self.message_buffer)

    def init_run(self) -> None:
        """Init the run sequence and store run task."""
        # RUF006
        # pylint: disable-next=line-too-long
        # noqa keep reference via: https://stackoverflow.com/questions/71938799/python-asyncio-create-task-really-need-to-keep-a-reference
        # even if we don't care
        task = self.loop.create_task(self.run())

        def remove_task(_task):
            try:
                self.tasks.remove(_task)
            except KeyError:
                pass  # already removed

        task.add_done_callback(remove_task)
        self.tasks.add(task)

    def close(self) -> None:
        """Close our socket and cancel all pending tasks."""
        self.socket.close_socket()
        for task in self.tasks:
            task.cancel()

    def is_connected(self) -> bool:
        """Return whether or not connected."""
        return self.state == ConnectionState.CONNECTED

    def is_errored(self) -> bool:
        """Return True if this connection has failed terminally.

        A connection in ERROR or CLOSED can never become CONNECTED
        again (the run() state machine only moves forward), so
        callers polling for connection establishment can bail as
        soon as this returns True instead of waiting out their
        full timeout.
        """
        return self.state in (ConnectionState.ERROR, ConnectionState.CLOSED)

    async def run(self) -> None:
        """State machine."""
        while True:
            if self.state == ConnectionState.NOT_STARTED:
                try:
                    await self.socket.connect_socket()
                    self.state = ConnectionState.CONNECTED
                    _LOGGER.info(
                        "Connected to Deako local integrations with %s",
                        self.format_name(),
                    )
                # pylint: disable-next=broad-exception-caught
                except Exception as exc:
                    _LOGGER.error(
                        "Failed to connect %s because %s",
                        self.format_name(),
                        exc,
                    )
                    self.state = ConnectionState.ERROR
            elif self.state == ConnectionState.CONNECTED:
                try:
                    await self.read_socket()
                # pylint: disable-next=broad-exception-caught
                except Exception as exc:
                    _LOGGER.error(
                        "Failed to read socket %s because %s",
                        self.format_name(),
                        exc,
                    )
                    self.state = ConnectionState.ERROR
            elif self.state == ConnectionState.ERROR:
                try:
                    self.close()
                    self.state = ConnectionState.CLOSED
                # pylint: disable-next=broad-exception-caught
                except Exception as exc:
                    _LOGGER.error(
                        "Failed to close socket %s because %s",
                        self.format_name(),
                        exc,
                    )
                    self.state = ConnectionState.CLOSED
            elif self.state == ConnectionState.CLOSED:
                # this socket is toast
                break
            else:
                _LOGGER.error("Unknown state: %s", self.state)
                raise UnknownStateException(f"Unknown state: {self.state}")

    def format_name(self) -> str:
        """Format name."""
        return f"{self.name}@{self.address}"
