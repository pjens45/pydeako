"""Tests for the two-manager DeakoConnectionPool.

Covers: module constants, `_tcp_probe`, `ConnectionPoolState`, pool
`__init__`, shared-cache accessors and callbacks, `start()` fail-fast
plus best-effort standby, terminal re-entrant `stop()`, identity-
dispatched session-lost handling, `_flip` semantics and anti-flap,
the repair supervisor's probe gating and install guard, standby
device-list verification, per-host EVENT counters, and every
`control_device` path (happy, flip+retry, degraded fail-fast).
"""
# pylint: disable=too-many-lines,protected-access

import asyncio

import pytest
from mock import AsyncMock, MagicMock, patch

from ._connection_pool import (
    ConnectionPoolState,
    DEAKO_DEFAULT_PORT,
    DeakoConnectionPool,
    PROBE_TIMEOUT,
    REPAIR_BACKOFF_MAX_S,
    REPAIR_SETTLE_S,
    REPAIR_TICK_S,
    STEP_TIMEOUT_S,
    _tcp_probe,
)
from .utils._socket import NoSocketException


# --- helpers -------------------------------------------------------

def _fake_deako(connected: bool = True) -> MagicMock:
    """Build a MagicMock that looks like a live Deako for the pool."""
    deako = MagicMock()
    deako.is_connected.return_value = connected
    deako.connect = AsyncMock()
    deako.find_devices = AsyncMock()
    deako.disconnect = AsyncMock()
    deako._control_device_strict = AsyncMock()
    deako.connection_manager = MagicMock()
    deako.connection_manager.auto_reconnect = True
    return deako


def _pool(**kwargs) -> DeakoConnectionPool:
    return DeakoConnectionPool(
        primary_host="10.0.0.1",
        failover_host="10.0.0.2",
        **kwargs,
    )


async def _drain() -> None:
    """Let scheduled tasks run."""
    for _ in range(4):
        await asyncio.sleep(0)


async def _started_pool(
    active: MagicMock | None = None,
    standby: MagicMock | None = None,
    **kwargs,
) -> tuple[DeakoConnectionPool, MagicMock, MagicMock, MagicMock]:
    """Start a pool with two fakes; return (pool, active, standby, cls)."""
    active = active if active is not None else _fake_deako()
    standby = standby if standby is not None else _fake_deako()
    with patch(
        "pydeako.deako._connection_pool.Deako",
        side_effect=[active, standby],
    ) as deako_cls:
        pool = _pool(**kwargs)
        await pool.start()
    return pool, active, standby, deako_cls


# --- module constants ----------------------------------------------

def test_module_constants_are_concrete():
    """Module defines every constant used by the pool contracts."""
    assert DEAKO_DEFAULT_PORT == 23
    assert PROBE_TIMEOUT > 0
    assert STEP_TIMEOUT_S == 2.0
    assert REPAIR_TICK_S > 0
    assert REPAIR_BACKOFF_MAX_S >= REPAIR_TICK_S
    assert REPAIR_SETTLE_S >= 0


# --- _tcp_probe ----------------------------------------------------

@pytest.mark.asyncio
async def test_tcp_probe_returns_true_on_open_connection():
    """_tcp_probe succeeds when open_connection returns cleanly."""
    writer = MagicMock()
    writer.close = MagicMock()
    writer.wait_closed = AsyncMock()
    reader = MagicMock()
    with patch(
        "pydeako.deako._connection_pool.asyncio.open_connection",
        new=AsyncMock(return_value=(reader, writer)),
    ):
        result = await _tcp_probe("10.0.0.1")
    assert result is True
    writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_tcp_probe_returns_false_on_oserror():
    """_tcp_probe returns False when open_connection raises OSError."""
    with patch(
        "pydeako.deako._connection_pool.asyncio.open_connection",
        new=AsyncMock(side_effect=OSError("refused")),
    ):
        result = await _tcp_probe("10.0.0.1")
    assert result is False


# --- ConnectionPoolState -------------------------------------------

def test_pool_state_is_frozen():
    """Snapshots are immutable."""
    snap = ConnectionPoolState(
        primary_host="a",
        failover_host="b",
        primary_connected=True,
        failover_keepalive_active=False,
        started=True,
    )
    with pytest.raises(Exception):
        snap.primary_host = "c"  # type: ignore[misc]


# --- __init__ -------------------------------------------------------

def test_init_rejects_same_hosts():
    """primary_host == failover_host raises ValueError."""
    with pytest.raises(ValueError):
        DeakoConnectionPool(
            primary_host="10.0.0.1", failover_host="10.0.0.1",
        )


def test_init_initial_state():
    """Fresh pool: not started, nothing connected, hosts recorded."""
    pool = _pool()
    snap = pool.state()
    assert snap.primary_host == "10.0.0.1"
    assert snap.failover_host == "10.0.0.2"
    assert snap.primary_connected is False
    assert snap.failover_keepalive_active is False
    assert snap.started is False
    assert pool.is_connected() is False


# --- shared cache accessors ----------------------------------------

def test_accessors_read_shared_cache():
    """Accessors answer from the pool-owned dict, not a session."""
    pool = _pool()
    pool._devices["u"] = {
        "name": "Kitchen",
        "uuid": "u",
        "dimmable": True,
        "state": {"power": True, "dim": 50},
    }
    assert pool.get_devices() is pool._devices
    assert pool.get_state("u") == {"power": True, "dim": 50}
    assert pool.get_name("u") == "Kitchen"
    assert pool.is_dimmable("u") is True
    assert pool.get_state("missing") is None


def test_set_state_callback_lands_in_shared_dict():
    """Callbacks registered before and after device load both land."""
    pool = _pool()
    early = MagicMock()
    pool.set_state_callback("u", early)
    # Entry did not exist yet; simulate a device load then re-apply.
    pool._devices["u"] = {"state": {}}
    pool._apply_callbacks()
    assert pool._devices["u"]["callback"] is early
    late = MagicMock()
    pool.set_state_callback("u", late)
    assert pool._devices["u"]["callback"] is late


# --- start() --------------------------------------------------------

@pytest.mark.asyncio
async def test_start_connects_both_sessions():
    """Happy path: active + standby sessions, supervisor running."""
    pool, active, standby, _ = await _started_pool()
    try:
        assert pool.active is active
        assert pool.standby is standby
        # Primary session did the device-list exchange; standby not.
        active.find_devices.assert_awaited_once()
        standby.find_devices.assert_not_awaited()
        # Reconnect is centralized: both managers forced off.
        assert active.connection_manager.auto_reconnect is False
        assert standby.connection_manager.auto_reconnect is False
        # Both sessions share the pool's device dict.
        assert active.devices is pool._devices
        assert standby.devices is pool._devices
        snap = pool.state()
        assert snap.started is True
        assert snap.primary_connected is True
        assert snap.failover_keepalive_active is True
        assert pool._supervisor_task is not None
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_start_raises_when_primary_unreachable():
    """Fail-fast: primary connect raises; pool unstarted, torn down."""
    broken = _fake_deako(connected=False)
    broken.connect = AsyncMock(side_effect=OSError("no route"))
    with patch(
        "pydeako.deako._connection_pool.Deako",
        return_value=broken,
    ):
        pool = _pool()
        with pytest.raises(NoSocketException):
            await pool.start()
    assert pool._started is False
    assert pool.active is None
    broken.disconnect.assert_awaited()


@pytest.mark.asyncio
async def test_start_raises_when_primary_never_connects():
    """connect() returns but never reaches CONNECTED: fail-fast."""
    broken = _fake_deako(connected=False)
    with patch(
        "pydeako.deako._connection_pool.Deako",
        return_value=broken,
    ):
        pool = _pool()
        with pytest.raises(NoSocketException):
            await pool.start()
    assert pool._started is False
    broken.disconnect.assert_awaited()


@pytest.mark.asyncio
async def test_start_proceeds_degraded_when_standby_fails():
    """Standby failure is non-fatal; supervisor owns the repair."""
    active = _fake_deako()
    broken_standby = _fake_deako()
    broken_standby.connect = AsyncMock(side_effect=OSError("refused"))
    pool, _, _, _ = await _started_pool(
        active=active, standby=broken_standby,
    )
    try:
        assert pool._started is True
        assert pool.active is active
        assert pool.standby is None
        snap = pool.state()
        assert snap.primary_connected is True
        assert snap.failover_keepalive_active is False
        broken_standby.disconnect.assert_awaited()
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_start_is_idempotent_and_single_use():
    """Second start() is a no-op; start() after stop() raises."""
    pool, active, _, deako_cls = await _started_pool()
    calls_after_first = deako_cls.call_count
    await pool.start()
    assert deako_cls.call_count == calls_after_first
    await pool.stop()
    with pytest.raises(RuntimeError):
        await pool.start()
    assert active.disconnect.await_count >= 1


# --- stop() ---------------------------------------------------------

@pytest.mark.asyncio
async def test_stop_tears_down_everything_and_is_reentrant():
    """stop(): disconnects both sessions, kills supervisor, twice-safe."""
    pool, active, standby, _ = await _started_pool()
    await pool.stop()
    active.disconnect.assert_awaited()
    standby.disconnect.assert_awaited()
    assert pool.active is None
    assert pool.standby is None
    assert pool._supervisor_task is None
    assert pool.state().started is False
    await pool.stop()  # re-entrant


# --- control_device: happy and degraded paths -----------------------

@pytest.mark.asyncio
async def test_control_device_sends_on_active():
    """Happy path: strict send on the active session only."""
    pool, active, standby, _ = await _started_pool()
    try:
        await pool.control_device("u", True, 40)
        active._control_device_strict.assert_awaited_once_with(
            "u", True, 40,
        )
        standby._control_device_strict.assert_not_awaited()
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_control_device_raises_after_stop():
    """control_device on a stopped pool raises NoSocketException."""
    pool, _, _, _ = await _started_pool()
    await pool.stop()
    with pytest.raises(NoSocketException):
        await pool.control_device("u", True)


@pytest.mark.asyncio
async def test_control_device_flips_when_active_dead():
    """Dead active + connected standby: flip first, then send."""
    fired = []
    pool, active, standby, _ = await _started_pool(
        on_failover_switch=lambda p, f: fired.append((p, f)),
    )
    try:
        active.is_connected.return_value = False
        await pool.control_device("u", False)
        assert pool.active is standby
        assert pool.primary_host == "10.0.0.2"
        assert pool.failover_host == "10.0.0.1"
        standby._control_device_strict.assert_awaited_once_with(
            "u", False, None,
        )
        assert fired == [("10.0.0.2", "10.0.0.1")]
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_control_device_fails_fast_when_nothing_usable():
    """Dead active + dead standby: quick NoSocketException, no connects."""
    pool, active, standby, deako_cls = await _started_pool()
    try:
        active.is_connected.return_value = False
        standby.is_connected.return_value = False
        calls_before = deako_cls.call_count
        with pytest.raises(NoSocketException):
            await pool.control_device("u", True)
        # Fail-fast contract: no inline session builds.
        assert deako_cls.call_count == calls_before
        # Hosts unchanged: nothing was promoted.
        assert pool.primary_host == "10.0.0.1"
        assert pool._repair_wake.is_set()
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_control_device_send_failure_flips_and_retries():
    """OSError on active send: one flip, one retry on new active."""
    pool, active, standby, _ = await _started_pool()
    try:
        active._control_device_strict = AsyncMock(
            side_effect=OSError("broken pipe"),
        )
        await pool.control_device("u", True, 75)
        active._control_device_strict.assert_awaited_once()
        standby._control_device_strict.assert_awaited_once_with(
            "u", True, 75,
        )
        assert pool.active is standby
        assert pool.primary_host == "10.0.0.2"
        # Demoted session gets torn down in the background.
        await _drain()
        active.disconnect.assert_awaited()
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_control_device_send_failure_without_standby_raises():
    """OSError on send and no connected standby: NoSocketException."""
    pool, active, standby, _ = await _started_pool()
    try:
        active._control_device_strict = AsyncMock(
            side_effect=OSError("broken pipe"),
        )
        standby.is_connected.return_value = False
        with pytest.raises(NoSocketException):
            await pool.control_device("u", True)
        assert pool.primary_host == "10.0.0.1"
        assert pool.active is active
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_control_device_retry_failure_names_both_hosts():
    """Both sends fail across the flip: exception names both hosts."""
    pool, active, standby, _ = await _started_pool()
    try:
        active._control_device_strict = AsyncMock(
            side_effect=OSError("first"),
        )
        standby._control_device_strict = AsyncMock(
            side_effect=OSError("second"),
        )
        with pytest.raises(NoSocketException) as excinfo:
            await pool.control_device("u", True)
        msg = str(excinfo.value)
        assert "10.0.0.1" in msg
        assert "10.0.0.2" in msg
    finally:
        await pool.stop()


# --- session-lost dispatch ------------------------------------------

@pytest.mark.asyncio
async def test_active_ping_timeout_triggers_flip():
    """Manager loss callback on the active promotes the standby."""
    fired = []
    active = _fake_deako()
    standby = _fake_deako()
    with patch(
        "pydeako.deako._connection_pool.Deako",
        side_effect=[active, standby],
    ) as deako_cls:
        pool = _pool(
            on_failover_switch=lambda p, f: fired.append((p, f)),
        )
        await pool.start()
    try:
        on_lost = deako_cls.call_args_list[0].kwargs[
            "on_connection_lost"
        ]
        active.is_connected.return_value = False
        on_lost()
        await _drain()
        assert pool.active is standby
        assert pool.primary_host == "10.0.0.2"
        assert fired == [("10.0.0.2", "10.0.0.1")]
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_standby_loss_drops_slot_without_flip():
    """Manager loss callback on the standby never flips roles."""
    active = _fake_deako()
    standby = _fake_deako()
    with patch(
        "pydeako.deako._connection_pool.Deako",
        side_effect=[active, standby],
    ) as deako_cls:
        pool = _pool()
        await pool.start()
    try:
        on_lost = deako_cls.call_args_list[1].kwargs[
            "on_connection_lost"
        ]
        on_lost()
        await _drain()
        assert pool.active is active
        assert pool.primary_host == "10.0.0.1"
        assert pool.standby is None
        assert pool._repair_wake.is_set()
        standby.disconnect.assert_awaited()
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_superseded_session_loss_is_ignored():
    """A loss callback from a demoted session changes nothing."""
    pool, active, standby, deako_cls = await _started_pool()
    try:
        active.is_connected.return_value = False
        assert await pool._flip("10.0.0.1") is True
        # Old active fires its loss callback late.
        on_lost = deako_cls.call_args_list[0].kwargs[
            "on_connection_lost"
        ]
        on_lost()
        await _drain()
        assert pool.active is standby
        assert pool.primary_host == "10.0.0.2"
    finally:
        await pool.stop()


# --- flip semantics -------------------------------------------------

@pytest.mark.asyncio
async def test_flip_requires_connected_standby():
    """No connected standby: flip returns False and kicks repair."""
    pool, _, standby, _ = await _started_pool()
    try:
        standby.is_connected.return_value = False
        pool._repair_wake.clear()
        assert await pool._flip("10.0.0.1") is False
        assert pool._repair_wake.is_set()
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_flip_stale_failed_host_reports_current_health():
    """A flip request for an already-replaced host is a no-op."""
    pool, active, standby, _ = await _started_pool()
    try:
        assert await pool._flip("10.0.0.99") is True
        assert pool.active is active
        assert pool.standby is standby
    finally:
        await pool.stop()


# --- repair supervisor ----------------------------------------------

@pytest.mark.asyncio
async def test_repair_standby_probe_gated():
    """Probe failure skips the connect attempt entirely."""
    pool, _, _, deako_cls = await _started_pool()
    try:
        pool.standby = None
        calls_before = deako_cls.call_count
        with patch(
            "pydeako.deako._connection_pool._tcp_probe",
            new=AsyncMock(return_value=False),
        ):
            assert await pool._repair_standby() is False
        assert deako_cls.call_count == calls_before
        assert pool.standby is None
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_repair_standby_installs_on_success():
    """Probe + connect success installs the new standby session."""
    pool, _, _, _ = await _started_pool()
    try:
        pool.standby = None
        replacement = _fake_deako()
        with patch(
            "pydeako.deako._connection_pool._tcp_probe",
            new=AsyncMock(return_value=True),
        ):
            with patch(
                "pydeako.deako._connection_pool.Deako",
                return_value=replacement,
            ):
                assert await pool._repair_standby() is True
        assert pool.standby is replacement
        assert replacement.devices is pool._devices
        assert (
            replacement.connection_manager.auto_reconnect is False
        )
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_repair_aborts_if_failover_host_changed_mid_connect():
    """A flip during repair invalidates the target; do not install."""
    pool, _, _, _ = await _started_pool()
    try:
        pool.standby = None
        replacement = _fake_deako()

        async def connect_and_mutate():
            # Simulate a concurrent flip while the connect ran.
            pool.failover_host = "10.0.0.1"
            pool.primary_host = "10.0.0.2"

        replacement.connect = AsyncMock(
            side_effect=connect_and_mutate,
        )
        with patch(
            "pydeako.deako._connection_pool._tcp_probe",
            new=AsyncMock(return_value=True),
        ):
            with patch(
                "pydeako.deako._connection_pool.Deako",
                return_value=replacement,
            ):
                assert await pool._repair_standby() is False
        assert pool.standby is None
        replacement.disconnect.assert_awaited()
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_recovered_bridge_returns_as_standby_not_active():
    """Anti-flap: repair fills the standby slot; no auto flip-back."""
    pool, active, standby, _ = await _started_pool()
    try:
        # Active dies; flip promotes standby.
        active.is_connected.return_value = False
        assert await pool._flip("10.0.0.1") is True
        assert pool.primary_host == "10.0.0.2"
        # Old primary host recovers; supervisor repairs it.
        pool._last_flip = 0.0
        recovered = _fake_deako()
        with patch(
            "pydeako.deako._connection_pool._tcp_probe",
            new=AsyncMock(return_value=True),
        ):
            with patch(
                "pydeako.deako._connection_pool.Deako",
                return_value=recovered,
            ):
                assert await pool._repair_standby() is True
        # Roles: recovered host is STANDBY; active untouched.
        assert pool.active is standby
        assert pool.standby is recovered
        assert pool.primary_host == "10.0.0.2"
        assert pool.failover_host == "10.0.0.1"
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_supervisor_loop_repairs_when_kicked():
    """End-to-end: dead standby, kick, supervisor installs repair."""
    pool, _, standby, _ = await _started_pool()
    try:
        pool.standby = None
        pool._last_flip = 0.0
        replacement = _fake_deako()
        with patch(
            "pydeako.deako._connection_pool._tcp_probe",
            new=AsyncMock(return_value=True),
        ):
            with patch(
                "pydeako.deako._connection_pool.Deako",
                return_value=replacement,
            ):
                pool._repair_wake.set()
                for _ in range(50):
                    await asyncio.sleep(0)
                    if pool.standby is replacement:
                        break
        assert pool.standby is replacement
        _ = standby
    finally:
        await pool.stop()


# --- standby verification -------------------------------------------

@pytest.mark.asyncio
async def test_verify_standby_once_runs_find_devices_once():
    """Fresh standby answers device-list exactly once; flag latches."""
    pool, _, standby, _ = await _started_pool()
    try:
        assert pool._standby_found is False
        await pool._verify_standby_once()
        standby.find_devices.assert_awaited_once()
        await pool._verify_standby_once()
        standby.find_devices.assert_awaited_once()
    finally:
        await pool.stop()


@pytest.mark.asyncio
async def test_verify_standby_tolerates_failure():
    """Device-list failure on standby is logged, session kept."""
    pool, _, standby, _ = await _started_pool()
    try:
        standby.find_devices = AsyncMock(
            side_effect=OSError("timeout"),
        )
        await pool._verify_standby_once()
        assert pool.standby is standby
        assert pool._standby_found is True
    finally:
        await pool.stop()


# --- event counters --------------------------------------------------

@pytest.mark.asyncio
async def test_event_counters_track_per_host():
    """on_event_seen wiring increments the right host's counter."""
    active = _fake_deako()
    standby = _fake_deako()
    with patch(
        "pydeako.deako._connection_pool.Deako",
        side_effect=[active, standby],
    ) as deako_cls:
        pool = _pool()
        await pool.start()
    try:
        seen_active = deako_cls.call_args_list[0].kwargs[
            "on_event_seen"
        ]
        seen_standby = deako_cls.call_args_list[1].kwargs[
            "on_event_seen"
        ]
        seen_active()
        seen_active()
        seen_standby()
        counts = pool.event_counts()
        assert counts["10.0.0.1"] == 2
        assert counts["10.0.0.2"] == 1
        # Copy, not the live dict.
        counts["10.0.0.1"] = 99
        assert pool.event_counts()["10.0.0.1"] == 2
    finally:
        await pool.stop()


# --- shared-cache continuity across flip ------------------------------

@pytest.mark.asyncio
async def test_cache_and_callbacks_survive_flip():
    """State and callbacks are continuous across a flip."""
    pool, active, standby, _ = await _started_pool()
    try:
        cb = MagicMock()
        pool._devices["u"] = {
            "name": "Lamp",
            "state": {"power": True, "dim": 10},
        }
        pool.set_state_callback("u", cb)
        active.is_connected.return_value = False
        assert await pool._flip("10.0.0.1") is True
        assert pool.active is standby
        assert pool.get_state("u") == {"power": True, "dim": 10}
        assert pool._devices["u"]["callback"] is cb
    finally:
        await pool.stop()
