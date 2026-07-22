"""
Test the SocketConnection manager.
"""

import asyncio
import logging

import pytest
from mock import AsyncMock, Mock, patch

from ..discover import DevicesNotFoundException
from ._manager import _Manager, CONNECTION_TIMEOUT_S


def test_init():
    """Test _Manager.__init__"""
    manager = _Manager(AsyncMock(), Mock())

    assert manager is not None


@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager.asyncio")
@pytest.mark.asyncio
async def test_init_connection_already_started(
    asyncio_mock,
    create_connection_mock,
):
    """
    Test _Manager.init_connection when the connection
    sequence has already been initiated.
    """
    get_address = AsyncMock()

    manager = _Manager(get_address, Mock())
    manager.state.connecting = True

    await manager.init_connection()

    asyncio_mock.create_task.assert_not_called()

    create_connection_mock.assert_not_called()


@patch("pydeako.deako._manager._Manager.create_connection_task")
@pytest.mark.asyncio
async def test_init_connection_get_address_no_devices(
    create_connection_mock,
):
    """
    Test _Manager.init_connection with no devices found
    which restarts the connection. If we have an address,
    devices should be found.
    """
    get_address = AsyncMock()

    manager = _Manager(get_address, Mock())

    get_address.side_effect = DevicesNotFoundException()

    await manager.init_connection()

    create_connection_mock.assert_called_once()

    assert manager.state.connecting is False


@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager._Connection")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_init_connection_timeout_connecting(
    asyncio_mock,
    connection_mock,
    create_connection_mock,
):
    """Test _Manager.init_connection with timeout."""
    address, name = Mock(), Mock()
    get_address = AsyncMock()

    manager = _Manager(get_address, Mock())

    get_address.return_value = address, name
    connection_mock_instance = connection_mock.return_value
    connection_mock_instance.is_connected.return_value = False
    connection_mock_instance.is_errored.return_value = False

    await manager.init_connection()

    connection_mock.assert_called_once_with(
        address,
        name,
        manager.incoming_json,
    )

    assert asyncio_mock.sleep.call_count == CONNECTION_TIMEOUT_S

    create_connection_mock.assert_called_once()  # this is the retry


@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager._Connection")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_init_connection_bails_early_on_error(
    asyncio_mock,
    connection_mock,
    create_connection_mock,
):
    """A connection that errors ends the poll before the timeout."""
    address, name = Mock(), Mock()
    get_address = AsyncMock()

    manager = _Manager(get_address, Mock())

    get_address.return_value = address, name
    connection_mock_instance = connection_mock.return_value
    connection_mock_instance.is_connected.return_value = False
    connection_mock_instance.is_errored.return_value = True

    await manager.init_connection()

    # errored immediately: no poll sleeps burned
    assert asyncio_mock.sleep.call_count == 0
    connection_mock_instance.close.assert_called_once()
    create_connection_mock.assert_called_once()  # this is the retry


@patch("pydeako.deako._manager._Manager.maintain_connection_worker")
@patch("pydeako.deako._manager._Connection")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_init_connection(
    asyncio_mock,
    connection_mock,
    maintain_connection_worker_mock,
):
    """Test _Manager.init_connection, success"""
    address, name = Mock(), Mock()
    get_address = AsyncMock()

    manager = _Manager(get_address, Mock())

    get_address.return_value = address, name
    connection_mock_instance = connection_mock.return_value
    connection_mock_instance.is_connected.return_value = True

    await manager.init_connection()

    assert asyncio_mock.create_task.call_count == 1
    maintain_connection_worker_mock.assert_called_once()

    connection_mock.assert_called_once_with(
        address,
        name,
        manager.incoming_json,
    )

    assert not manager.state.connecting


@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager._Connection")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
# pylint: disable-next=unused-argument
async def test_init_connection_cancelled_closes_orphan(
    asyncio_mock,
    connection_mock,
    create_connection_mock,
):
    """Cancellation mid-connect closes the orphan and clears the latch.

    Regression guard for the leak where a cancelled init_connection
    abandoned a not-yet-installed _Connection (live run() task, held
    TCP slot) and left state.connecting=True, wedging every later
    connect attempt.
    """
    address, name = Mock(), Mock()
    get_address = AsyncMock()

    manager = _Manager(get_address, Mock())

    get_address.return_value = address, name
    connection_mock_instance = connection_mock.return_value
    connection_mock_instance.is_connected.return_value = False
    connection_mock_instance.is_errored.return_value = False
    # Cancel the coroutine while it is polling for the connection.
    asyncio_mock.sleep.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await manager.init_connection()

    connection_mock_instance.close.assert_called_once()
    assert not manager.state.connecting
    assert manager.connection is None


def test_close():
    """Test _Manager.close."""
    maintain_worker = Mock()
    connection = Mock()

    manager = _Manager(AsyncMock(), Mock())
    manager.maintain_worker = maintain_worker
    manager.connection = connection

    manager.close()

    maintain_worker.cancel.assert_called_once()
    connection.close.assert_called_once()

    assert manager.state.canceled


@patch("pydeako.deako._manager.asyncio")
@patch("pydeako.deako._manager._Manager.init_connection")
def test_create_connection_task(init_connection_mock, asyncio_mock):
    """Test _Manager.create_connection_task."""
    manager = _Manager(AsyncMock(), Mock())

    manager.create_connection_task()

    asyncio_mock.create_task.assert_called_once()
    task = asyncio_mock.create_task.return_value
    task.add_done_callback.assert_called_once()
    init_connection_mock.assert_called_once()
    assert len(manager.tasks) == 1


@patch("pydeako.deako._manager._Manager.close")
@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_maintain_connection_worker_canceled(
    asyncio_mock, create_connection_mock, close_mock
):
    """
    Test _Manager.maintain_connection_worker
    doesn't proceed when canceled.
    """
    manager = _Manager(AsyncMock(), Mock())
    manager.state.canceled = True

    await manager.maintain_connection_worker()

    asyncio_mock.sleep.assert_called_once_with(10)
    close_mock.assert_not_called()
    create_connection_mock.assert_not_called()


@patch("pydeako.deako._manager._Manager.close")
@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_maintain_connection_worker_no_pong(
    asyncio_mock, create_connection_mock, close_mock
):
    """Test _Manager.maintain_connection_worker doesn't receive pong."""
    manager = _Manager(AsyncMock(), Mock())

    await manager.maintain_connection_worker()

    assert len(asyncio_mock.sleep.mock_calls) == 2
    assert asyncio_mock.sleep.mock_calls[0].args[0] == 10
    assert asyncio_mock.sleep.mock_calls[1].args[0] == 10
    close_mock.assert_called_once()
    create_connection_mock.assert_called_once()


def test_incoming_json_pong():
    """Test _Manager.incoming_json with ping response."""
    incoming_json = {"type": "PING"}

    incoming_json_callback = Mock()

    manager = _Manager(AsyncMock(), incoming_json_callback)
    manager.pong_received = False

    manager.incoming_json(incoming_json)

    assert manager.pong_received


def test_incoming_json():
    """Test _Manager.incoming_json."""
    incoming_json = {"key": "value"}
    incoming_json_callback = Mock()

    manager = _Manager(AsyncMock(), incoming_json_callback)
    manager.pong_received = False

    manager.incoming_json(incoming_json)

    incoming_json_callback.assert_called_once_with(incoming_json)
    assert manager.pong_received is False


@patch("pydeako.deako._manager._Request")
@patch("pydeako.deako._manager.device_list_request")
@pytest.mark.asyncio
async def test_send_get_device_list(device_list_request_mock, request_mock):
    """Test _Manager.send_get_device_list."""
    client_name = Mock()
    request_mock_ret = Mock()

    request_mock.return_value = request_mock_ret

    manager = _Manager(AsyncMock(), Mock(), client_name=client_name)

    # Test with connection
    manager.connection = AsyncMock()
    result = await manager.send_get_device_list()
    device_list_request_mock.assert_called_once_with(source=client_name)
    request_mock.assert_called_once_with(device_list_request_mock.return_value)
    assert result is True

    # Test without connection
    manager.connection = None
    result = await manager.send_get_device_list()
    assert result is False


@pytest.mark.parametrize("completed_callback", [None, "some_callback"])
@patch("pydeako.deako._manager._Request")
@patch("pydeako.deako._manager.state_change_request")
@patch("pydeako.deako._manager._Manager.send_request")
@pytest.mark.asyncio
async def test_send_state_change(
    send_request_mock, state_change_request_mock, request_mock,
    completed_callback,
):
    """Test _Manager.send_state_change."""
    client_name = Mock()
    uuid = Mock()
    power = Mock()
    dim = Mock()
    request_mock_ret = Mock()

    request_mock.return_value = request_mock_ret

    manager = _Manager(AsyncMock(), Mock(), client_name=client_name)

    await manager.send_state_change(
        uuid, power, dim, completed_callback=completed_callback
    )

    request_mock.assert_called_once_with(
        state_change_request_mock.return_value,
        completed_callback=completed_callback,
    )
    state_change_request_mock.assert_called_once_with(
        uuid, power, dim, source=client_name
    )
    send_request_mock.assert_called_once_with(request_mock_ret)


@patch("pydeako.deako._manager._Request")
@pytest.mark.asyncio
async def test_send_request(
    request_mock,
):
    """Test _Manager.send_request with connection"""
    client_name = Mock()
    connection_mock = AsyncMock()

    request_mock.get_body_str.return_value = "some message"

    manager = _Manager(AsyncMock(), Mock(), client_name=client_name)

    manager.connection = connection_mock

    result = await manager.send_request(request_mock)

    connection_mock.send_data.assert_called_once_with("some message")
    # A successful send fires the request's optimistic-state callback
    # (N8): without this, HA state depended entirely on the bridge
    # echoing an EVENT back.
    request_mock.complete_callback.assert_called_once_with()
    assert result is True


@patch("pydeako.deako._manager._Request")
@pytest.mark.asyncio
async def test_send_request_no_connection(
    request_mock,
):
    """Test _Manager.send_request without connection"""
    client_name = Mock()
    connection_mock = AsyncMock()

    request_mock.get_body_str.return_value = "some message"

    manager = _Manager(AsyncMock(), Mock(), client_name=client_name)

    result = await manager.send_request(request_mock)

    connection_mock.send_data.assert_not_called()
    # No connection means the send never happened, so the optimistic
    # callback must NOT fire.
    request_mock.complete_callback.assert_not_called()
    assert result is False


# ---------------------------------------------------------------------------
# Phase 4 coverage: on_connection_lost callback, ping-send OSError guard,
# send_state_change contract, and auto_reconnect gating of the three
# create_connection_task() sites.
# ---------------------------------------------------------------------------


@patch("pydeako.deako._manager._Manager.close")
@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_maintain_connection_worker_invokes_on_connection_lost(
    asyncio_mock, create_connection_mock, close_mock
):
    """on_connection_lost fires once on ping timeout, before reconnect."""
    callback = Mock()

    manager = _Manager(AsyncMock(), Mock(), on_connection_lost=callback)

    await manager.maintain_connection_worker()

    assert len(asyncio_mock.sleep.mock_calls) == 2
    callback.assert_called_once()
    close_mock.assert_called_once()
    create_connection_mock.assert_called_once()


@patch("pydeako.deako._manager._Manager.close")
@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_maintain_worker_callback_exception_swallowed(
    asyncio_mock, create_connection_mock, close_mock, caplog,
):
    """Exceptions from on_connection_lost are caught, logged at WARNING."""
    callback = Mock(side_effect=RuntimeError("callback boom"))

    manager = _Manager(AsyncMock(), Mock(), on_connection_lost=callback)

    caplog.set_level(logging.WARNING, logger="pydeako.deako")

    await manager.maintain_connection_worker()

    assert len(asyncio_mock.sleep.mock_calls) == 2
    callback.assert_called_once()
    close_mock.assert_called_once()
    create_connection_mock.assert_called_once()
    assert any(
        record.levelno == logging.WARNING
        and "on_connection_lost callback error" in record.message
        for record in caplog.records
    )


@patch("pydeako.deako._manager._Manager.close")
@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_maintain_connection_worker_no_on_connection_lost(
    asyncio_mock, create_connection_mock, close_mock
):
    """Ping timeout with no callback still closes and reconnects."""
    manager = _Manager(AsyncMock(), Mock())

    assert manager.on_connection_lost is None

    await manager.maintain_connection_worker()

    assert len(asyncio_mock.sleep.mock_calls) == 2
    close_mock.assert_called_once()
    create_connection_mock.assert_called_once()


@patch("pydeako.deako._manager._Manager.close")
@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager._Manager.send_request")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_maintain_worker_handles_ping_send_oserror(
    asyncio_mock, send_request_mock, create_connection_mock, close_mock,
):
    """Ping-send OSError is caught; worker falls through to no-pong branch."""
    send_request_mock.side_effect = OSError("broken pipe")

    manager = _Manager(AsyncMock(), Mock())

    await manager.maintain_connection_worker()

    assert len(asyncio_mock.sleep.mock_calls) == 2
    send_request_mock.assert_called_once()
    close_mock.assert_called_once()
    create_connection_mock.assert_called_once()


@pytest.mark.asyncio
async def test_send_state_change_returns_false_when_disconnected():
    """send_state_change returns False only for missing connection."""
    manager = _Manager(AsyncMock(), Mock())
    manager.connection = None

    result = await manager.send_state_change(
        uuid="uuid-1", power=True, dim=None,
    )

    assert result is False


@pytest.mark.asyncio
async def test_send_state_change_propagates_oserror_when_connected():
    """Real send failure surfaces as OSError, not a silent False."""
    connection_mock = AsyncMock()
    connection_mock.send_data.side_effect = OSError("broken pipe")

    manager = _Manager(AsyncMock(), Mock())
    manager.connection = connection_mock

    with pytest.raises(OSError):
        await manager.send_state_change(
            uuid="uuid-1", power=True, dim=None,
        )


@patch("pydeako.deako._manager._Manager.create_connection_task")
@pytest.mark.asyncio
async def test_no_reconnect_when_auto_reconnect_false_devices_not_found(
    create_connection_mock,
):
    """auto_reconnect=False skips reconnect on DevicesNotFoundException."""
    get_address = AsyncMock()
    get_address.side_effect = DevicesNotFoundException()

    manager = _Manager(get_address, Mock())
    manager.auto_reconnect = False

    await manager.init_connection()

    create_connection_mock.assert_not_called()
    assert manager.state.connecting is False


@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager._Connection")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_no_reconnect_when_auto_reconnect_false_connect_timeout(
    asyncio_mock, connection_mock, create_connection_mock,
):
    """auto_reconnect=False skips reconnect at connect-timeout site."""
    address, name = Mock(), Mock()
    get_address = AsyncMock()
    get_address.return_value = address, name

    connection_mock_instance = connection_mock.return_value
    connection_mock_instance.is_connected.return_value = False
    connection_mock_instance.is_errored.return_value = False

    manager = _Manager(get_address, Mock())
    manager.auto_reconnect = False

    await manager.init_connection()

    connection_mock.assert_called_once_with(
        address, name, manager.incoming_json,
    )
    assert asyncio_mock.sleep.call_count == CONNECTION_TIMEOUT_S
    connection_mock_instance.close.assert_called_once()
    create_connection_mock.assert_not_called()


@patch("pydeako.deako._manager._Manager.close")
@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_no_reconnect_when_auto_reconnect_false_ping_timeout(
    asyncio_mock, create_connection_mock, close_mock,
):
    """auto_reconnect=False skips reconnect at ping-timeout site."""
    manager = _Manager(AsyncMock(), Mock())
    manager.auto_reconnect = False

    await manager.maintain_connection_worker()

    assert len(asyncio_mock.sleep.mock_calls) == 2
    close_mock.assert_called_once()
    create_connection_mock.assert_not_called()


@patch("pydeako.deako._manager._Manager.close")
@patch("pydeako.deako._manager._Manager.create_connection_task")
@patch("pydeako.deako._manager.asyncio", autospec=True)
@pytest.mark.asyncio
async def test_no_reconnect_auto_reconnect_false_ping_timeout_with_callback(
    asyncio_mock, create_connection_mock, close_mock,
):
    """auto_reconnect=False fires callback and closes, but no reconnect."""
    callback = Mock()

    manager = _Manager(AsyncMock(), Mock(), on_connection_lost=callback)
    manager.auto_reconnect = False

    await manager.maintain_connection_worker()

    assert len(asyncio_mock.sleep.mock_calls) == 2
    callback.assert_called_once()
    close_mock.assert_called_once()
    create_connection_mock.assert_not_called()
