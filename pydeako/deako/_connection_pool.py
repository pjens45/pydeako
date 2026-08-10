"""
Connection pool for Deako bridges: two fully-managed sessions.

Architecture (two-manager model): the pool holds a real `Deako`
session on BOTH bridges. Exactly one is `active` (all commands route
to it); the other is `standby`. The standby's session doubles as the
bridge-mode keepalive (holding TCP on port 23 is what kept the old
`_KeepAliveSocket` design warm) while also maintaining a live device
cache and a ping worker.

Failover is a pointer flip. When the active fails (send OSError or
the manager's ping timeout), the pool promotes the connected standby
in place: swap roles, retry the command once, done. There is no
teardown-then-reconnect at the worst moment, no bridge-recycle wait,
and no device-list exchange on the critical path.

Repair is background and bounded. A single supervisor task owns ALL
reconnection: it probe-gates (tcp probe before any connect attempt)
and backs off exponentially up to a cap, so a permanently missing
bridge is polled gently instead of hammered. Neither manager ever
self-reconnects (`auto_reconnect` is forced off on both); this is
load-bearing, see the Gen1 stale-bridge lesson in the integration
docs.

Anti-flap by construction: only an active failure triggers a flip,
and the supervisor only ever fills the STANDBY slot. A recovered
bridge comes back as standby; nothing auto-promotes it. Role flapping
therefore requires the active to actually fail each time.

State continuity: all sessions share one device-cache dict owned by
the pool, so a flip serves exactly the state the old active had, and
EVENT traffic from either bridge upserts the same entries. Device
state callbacks live inside that shared dict and survive flips
without replay.

Instrumentation: the pool counts EVENT deliveries per host (via
`Deako.on_event_seen`) and the supervisor logs both counters at
DEBUG. A growing standby-host counter while that host is standby is
direct evidence the firmware relays profile events to secondary
sessions.

`stop()` is terminal. A stopped pool is not restartable. Host names
(`primary_host`, `failover_host`) are `str` for the lifetime of the
pool and never nulled out on any path; degraded state is signaled
through `ConnectionPoolState`, not missing host fields.
"""
# pylint: disable=too-many-lines

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from ._deako import Deako
from .utils._socket import NoSocketException

_LOGGER: logging.Logger = logging.getLogger(__package__)

# Bridge TCP port used by `_tcp_probe`. The protocol layer parses
# "ip:port" address strings in utils/_socket.py and exposes no
# shared port constant, so the pool owns this value at module scope.
DEAKO_DEFAULT_PORT = 23

# tcp_probe connect budget.
PROBE_TIMEOUT = 3.0

# Unified bounded-teardown budget for disconnects and stops.
STEP_TIMEOUT_S = 2.0

# Supervisor cadence: base interval between repair passes when there
# is something to repair. Doubles on consecutive failures for a given
# target up to REPAIR_BACKOFF_MAX_S, resets on success. A healthy
# pool's supervisor sleeps on an Event and wakes only when kicked.
REPAIR_TICK_S = 15.0
REPAIR_BACKOFF_MAX_S = 600.0

# Grace period after a flip before the supervisor may probe the
# demoted host: the bridge that just died is often rebooting or
# recycling its single TCP slot, and probing it instantly wastes an
# attempt (and a backoff doubling) on a known-bad window.
REPAIR_SETTLE_S = 5.0


async def _tcp_probe(
    host: str,
    port: int = DEAKO_DEFAULT_PORT,
    timeout: float = PROBE_TIMEOUT,
) -> bool:
    """Return True iff ``host:port`` accepts a TCP connection.

    Opens and immediately closes a connection. Used by the repair
    supervisor to gate connect attempts so a dead host costs one
    cheap probe, not a full connect + device-list budget.
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout,
        )
    except (OSError, asyncio.TimeoutError) as exc:
        _LOGGER.debug("_tcp_probe(%s:%d) failed: %s", host, port, exc)
        return False
    try:
        writer.close()
        try:
            await writer.wait_closed()
        except (OSError, asyncio.TimeoutError):
            pass
    except Exception:  # pylint: disable=broad-exception-caught
        pass
    _ = reader
    return True


@dataclass(frozen=True)
class ConnectionPoolState:
    """Immutable snapshot of the pool's state.

    Exactly five fields, unchanged from the keepalive-era contract.
    `failover_keepalive_active` is reinterpreted as "the standby
    SESSION is connected": a live standby session holds the bridge's
    TCP slot exactly like the old keepalive socket did, so existing
    consumers keep their meaning (True == warm standby ready).
    """

    primary_host: str
    failover_host: str
    primary_connected: bool
    failover_keepalive_active: bool
    started: bool


class DeakoConnectionPool:
    """Primary/standby pool with two managed sessions.

    Public contract is unchanged from the keepalive-era pool:
    constructor signature, `start()` fail-fast on the primary,
    terminal `stop()`, `control_device()` raising
    `NoSocketException` when nothing usable remains, `state()`,
    `is_connected()`, `set_state_callback()`, and the device
    accessors. `on_failover_switch` fires after every successful
    flip with (new_primary_host, new_failover_host).

    `stop()` is terminal. Calling `start()` after `stop()` raises
    `RuntimeError`. `control_device()` after `stop()` raises
    `NoSocketException`.
    """

    # pylint: disable=too-many-instance-attributes

    active: Deako | None
    standby: Deako | None

    def __init__(
        self,
        primary_host: str,
        failover_host: str,
        client_name: str | None = None,
        on_failover_switch: Callable[[str, str], None] | None = None,
    ) -> None:
        """Initialize the pool.

        Args:
            primary_host: IP address of the primary bridge. Required.
            failover_host: IP address of the failover bridge.
                Required; single-bridge callers use `Deako` directly.
            client_name: Optional client name sent in protocol
                messages; forwarded to every `Deako` the pool owns.
            on_failover_switch: Optional sync callback invoked after
                a successful flip with the new primary host and the
                new failover host. Exceptions raised by the callback
                are logged at WARNING and swallowed.

        Concurrent `start()` calls on the same pool are unsupported:
        callers must serialize setup.
        """
        if primary_host == failover_host:
            raise ValueError(
                "DeakoConnectionPool: primary_host and "
                "failover_host must be distinct; use Deako "
                "directly for single-bridge setups",
            )
        self.primary_host: str = primary_host
        self.failover_host: str = failover_host
        self._client_name = client_name
        self._on_failover_switch = on_failover_switch

        self.active: Deako | None = None
        self.standby: Deako | None = None

        # Shared device cache. Every Deako the pool constructs has
        # its `devices` attribute pointed at this dict, so state and
        # callbacks are continuous across flips and both sessions
        # upsert the same entries from their own EVENT streams.
        self._devices: dict = {}

        # Pool-owned callback registry. Applied into the shared
        # device dict (entries may not exist yet at registration
        # time; `_apply_callbacks` re-applies after device loads).
        self._state_callbacks: dict[str, Callable[[], None]] = {}

        # Per-host EVENT delivery counters (firmware experiment:
        # does a standby session receive profile events?).
        self._event_counts: dict[str, int] = {
            primary_host: 0,
            failover_host: 0,
        }

        # Flip serialization.
        self._flip_lock: asyncio.Lock = asyncio.Lock()
        self._last_flip: float = 0.0

        # Repair supervisor.
        self._repair_wake: asyncio.Event = asyncio.Event()
        self._supervisor_task: asyncio.Task | None = None
        self._standby_found: bool = False

        # Tasks scheduled from sync manager callbacks (session-lost
        # dispatch, demoted-session teardown). Held so asyncio does
        # not GC them mid-flight; discarded on completion.
        self._bg_tasks: set[asyncio.Task] = set()

        self._started: bool = False
        self._stopped: bool = False

    # ----- Observability ---------------------------------------

    def state(self) -> ConnectionPoolState:
        """Return an immutable snapshot of pool state."""
        primary_connected = (
            self.active is not None and self.active.is_connected()
        )
        standby_connected = (
            self.standby is not None and self.standby.is_connected()
        )
        return ConnectionPoolState(
            primary_host=self.primary_host,
            failover_host=self.failover_host,
            primary_connected=primary_connected,
            failover_keepalive_active=standby_connected,
            started=self._started and not self._stopped,
        )

    def is_connected(self) -> bool:
        """Return True iff the active session is currently connected."""
        return self.active is not None and self.active.is_connected()

    def event_counts(self) -> dict[str, int]:
        """Return a copy of the per-host EVENT delivery counters."""
        return dict(self._event_counts)

    # ----- State callbacks -------------------------------------

    def set_state_callback(
        self, uuid: str, callback: Callable[[], None],
    ) -> None:
        """Register a sync state-change callback for a device.

        Stored on the pool and written into the shared device dict,
        where `Deako.update_state` invokes it. Because the dict is
        shared by both sessions, callbacks survive flips without
        replay; when both bridges relay the same event the callback
        can fire twice per change, which consumers must tolerate
        (Home Assistant's schedule-update path is idempotent).
        """
        self._state_callbacks[uuid] = callback
        entry = self._devices.get(uuid)
        if entry is not None:
            entry["callback"] = callback

    def _apply_callbacks(self) -> None:
        """Write registered callbacks into existing device entries.

        Called after any device-list load so callbacks registered
        before a device entry existed still land in the dict.
        """
        for uuid, callback in self._state_callbacks.items():
            entry = self._devices.get(uuid)
            if entry is not None:
                entry["callback"] = callback

    # ----- Device accessors ------------------------------------

    def get_devices(self) -> dict:
        """Return the shared device cache."""
        return self._devices

    def get_state(self, uuid: str) -> dict | None:
        """Get a device's state by uuid from the shared cache."""
        device = self._devices.get(uuid)
        if device is None:
            return None
        return device.get("state")

    def get_name(self, uuid: str) -> str | None:
        """Get a device's name by uuid from the shared cache."""
        device = self._devices.get(uuid)
        if device is None:
            return None
        return device.get("name")

    def is_dimmable(self, uuid: str) -> bool | None:
        """Get whether a device is dimmable from the shared cache."""
        device = self._devices.get(uuid)
        if device is None:
            return None
        return device.get("dimmable")

    # ----- Session construction --------------------------------

    def _build_session(self, host: str) -> Deako:
        """Construct a pool-owned `Deako` for ``host``.

        Wires: shared device dict, per-host EVENT counter, identity-
        dispatched connection-lost handler, and `auto_reconnect`
        FORCED OFF. The supervisor owns all reconnection; a manager
        that self-reconnects can hammer a permanently-gone bridge
        (the Gen1 stale-record failure class) and race the pool's
        role bookkeeping.
        """
        port = DEAKO_DEFAULT_PORT
        address = f"{host}:{port}"
        client = self._client_name or "pydeako"

        async def get_address():
            return address, client

        def on_event_seen() -> None:
            self._event_counts[host] = (
                self._event_counts.get(host, 0) + 1
            )

        holder: list[Deako] = []

        def on_lost() -> None:
            if holder:
                self._schedule_session_lost(holder[0])

        deako = Deako(
            get_address,
            client_name=self._client_name,
            on_connection_lost=on_lost,
            on_event_seen=on_event_seen,
        )
        holder.append(deako)
        # Share the device cache before any traffic arrives.
        deako.devices = self._devices
        deako.connection_manager.auto_reconnect = False
        return deako

    async def _connect_session(
        self, host: str, find_devices: bool,
    ) -> Deako:
        """Build and connect a session on ``host``.

        Raises on failure with the partial session torn down (a
        leaked half-connected session holds the bridge's single TCP
        slot until process restart). Cancellation mid-connect also
        tears down before propagating.
        """
        deako = self._build_session(host)
        try:
            await deako.connect()
            if not deako.is_connected():
                raise NoSocketException(
                    f"connect to {host} did not reach CONNECTED",
                )
            if find_devices:
                await deako.find_devices()
                self._apply_callbacks()
            return deako
        except BaseException:
            # BaseException: CancelledError must also tear down.
            try:
                await asyncio.wait_for(
                    deako.disconnect(), timeout=STEP_TIMEOUT_S,
                )
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            raise

    # ----- Lifecycle -------------------------------------------

    async def start(self) -> None:
        """Connect the primary session and start the supervisor.

        Fail-fast: raises `NoSocketException` if the primary cannot
        be reached (or `FindDevicesError` if it connects but the
        device-list exchange fails). The standby session is then
        attempted best-effort WITHOUT a blocking device-list
        exchange; on any failure the pool starts degraded and the
        supervisor repairs in the background.

        Idempotent after first success. `start()` after `stop()`
        raises `RuntimeError`; the pool is single-use.
        """
        if self._stopped:
            raise RuntimeError(
                "DeakoConnectionPool: start() after stop() is not "
                "supported; construct a new pool",
            )
        if self._started:
            return
        try:
            new_active = await self._connect_session(
                self.primary_host, find_devices=True,
            )
        except NoSocketException as exc:
            raise NoSocketException(
                f"start: primary {self.primary_host} unreachable: "
                f"{exc}",
            ) from exc
        except OSError as exc:
            raise NoSocketException(
                f"start: primary {self.primary_host} unreachable: "
                f"{exc}",
            ) from exc
        # FindDevicesError propagates as-is (session already torn
        # down by _connect_session), matching the previous contract.
        if await self._discard_if_stopped(new_active):
            return
        self.active = new_active
        self._started = True

        # Best-effort standby session. No device-list exchange here:
        # the shared cache is already populated by the primary, and
        # the supervisor verifies the standby answers queries later.
        try:
            standby = await self._connect_session(
                self.failover_host, find_devices=False,
            )
            if await self._discard_if_stopped(standby):
                return
            self.standby = standby
            self._standby_found = False
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.warning(
                "standby session on %s failed at start: %s; pool "
                "proceeds degraded, supervisor will repair",
                self.failover_host, exc,
            )
            self.standby = None

        self._repair_wake.set()
        self._supervisor_task = asyncio.create_task(
            self._supervisor(),
        )

    async def stop(self) -> None:
        """Terminal shutdown. Re-entrant-safe."""
        if self._stopped:
            return
        self._stopped = True
        self._repair_wake.set()
        if self._supervisor_task is not None:
            self._supervisor_task.cancel()
            try:
                await self._supervisor_task
            except (asyncio.CancelledError, Exception):  # pylint: disable=broad-exception-caught
                pass
            self._supervisor_task = None
        for task in list(self._bg_tasks):
            task.cancel()
        self._bg_tasks.clear()
        await self._teardown(self.standby)
        self.standby = None
        await self._teardown(self.active)
        self.active = None

    async def _teardown(self, deako: Deako | None) -> None:
        """Bounded, tolerant disconnect of a session (None-safe)."""
        if deako is None:
            return
        try:
            await asyncio.wait_for(
                deako.disconnect(), timeout=STEP_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            _LOGGER.warning(
                "session disconnect timed out after %ss; proceeding",
                STEP_TIMEOUT_S,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug("session disconnect error: %s", exc)

    async def _discard_if_stopped(self, deako: Deako) -> bool:
        """If the pool is stopped, disconnect ``deako``, return True."""
        if not self._stopped:
            return False
        await self._teardown(deako)
        return True

    # ----- Session-lost dispatch -------------------------------

    def _schedule_session_lost(self, deako: Deako) -> None:
        """Sync entry from a manager's ping-timeout path.

        Dispatch is by object identity, not host name, so a flip
        that already swapped roles cannot misroute a stale loss
        notification: the object either IS the current active (flip)
        or IS the current standby (drop + repair) or is neither
        (already superseded; ignore).
        """
        if self._stopped:
            return
        try:
            task = asyncio.create_task(self._on_session_lost(deako))
        except RuntimeError:
            _LOGGER.debug(
                "no running loop for session-lost dispatch; ignoring",
            )
            return
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _on_session_lost(self, deako: Deako) -> None:
        """Handle a dead session: flip if active, drop if standby."""
        if self._stopped:
            return
        if deako is self.active:
            _LOGGER.warning(
                "active session on %s lost (ping timeout); "
                "attempting flip to standby",
                self.primary_host,
            )
            flipped = await self._flip(self.primary_host)
            if not flipped:
                _LOGGER.error(
                    "active lost and no connected standby; pool "
                    "degraded, supervisor repairing",
                )
        elif deako is self.standby:
            _LOGGER.warning(
                "standby session on %s lost; supervisor will repair",
                self.failover_host,
            )
            self.standby = None
            self._standby_found = False
            await self._teardown(deako)
            self._repair_wake.set()
        else:
            _LOGGER.debug(
                "session-lost for superseded session; ignoring",
            )

    # ----- Flip -------------------------------------------------

    async def _flip(self, failed_host: str) -> bool:
        """Promote the standby in place of a failed active.

        Returns True when the pool has a usable active afterwards
        (including "another flip already handled it"), False when
        there is no connected standby to promote. Never connects;
        promotion is a pointer swap. The demoted session is torn
        down in the background and the supervisor is kicked to
        rebuild the standby slot.
        """
        async with self._flip_lock:
            if self._stopped:
                return False
            if failed_host != self.primary_host:
                # A concurrent flip already moved the primary off
                # the failed host; report current health.
                return (
                    self.active is not None
                    and self.active.is_connected()
                )
            standby = self.standby
            if standby is None or not standby.is_connected():
                self._repair_wake.set()
                return False
            demoted = self.active
            self.active = standby
            self.standby = None
            self._standby_found = False
            self.primary_host, self.failover_host = (
                self.failover_host, self.primary_host,
            )
            self._last_flip = asyncio.get_running_loop().time()
            _LOGGER.info(
                "flip succeeded: primary=%s failover=%s",
                self.primary_host, self.failover_host,
            )
            # Background teardown of the demoted session; it is
            # dead or dying and must not hold its TCP slot.
            if demoted is not None:
                task = asyncio.create_task(self._teardown(demoted))
                self._bg_tasks.add(task)
                task.add_done_callback(self._bg_tasks.discard)
            self._repair_wake.set()
            self._fire_on_failover_switch()
            return True

    def _fire_on_failover_switch(self) -> None:
        """Invoke the user switch callback, swallowing errors."""
        if self._on_failover_switch is None:
            return
        try:
            self._on_failover_switch(
                self.primary_host, self.failover_host,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.warning(
                "on_failover_switch callback error: %s", exc,
            )

    # ----- Repair supervisor ------------------------------------

    async def _supervisor(self) -> None:
        """Single owner of all reconnection. Bounded and probe-gated.

        Fills the STANDBY slot only; never touches the active and
        never promotes (anti-flap: promotion happens exclusively in
        `_flip`, triggered exclusively by active failure). Backoff
        doubles on consecutive failures up to `REPAIR_BACKOFF_MAX_S`
        and resets on success or when newly kicked by a state
        change.
        """
        backoff = REPAIR_TICK_S
        while not self._stopped:
            try:
                await asyncio.wait_for(
                    self._repair_wake.wait(), timeout=backoff,
                )
                # Kicked: a fresh condition, restart gentle.
                backoff = REPAIR_TICK_S
            except asyncio.TimeoutError:
                pass
            self._repair_wake.clear()
            if self._stopped:
                return
            _LOGGER.debug(
                "supervisor: event_counts=%s devices=%d "
                "standby_connected=%s",
                self._event_counts,
                len(self._devices),
                self.standby is not None
                and self.standby.is_connected(),
            )
            # Active recovery comes first: a pool with no active
            # cannot serve commands at all, and rebuilding it is
             # flap-free (there is nothing to flap between).
            if self.active is None or not self.active.is_connected():
                recovered = await self._repair_active()
                if recovered:
                    backoff = REPAIR_TICK_S
                    # Fall through on the next tick to refill standby.
                    self._repair_wake.set()
                else:
                    backoff = min(backoff * 2, REPAIR_BACKOFF_MAX_S)
                    _LOGGER.debug(
                        "supervisor: active repair failed; next "
                        "attempt in %.0fs", backoff,
                    )
                continue
            if (
                self.standby is not None
                and self.standby.is_connected()
            ):
                await self._verify_standby_once()
                continue
            # Respect the post-flip settle window.
            loop = asyncio.get_running_loop()
            if (
                self._last_flip
                and loop.time() - self._last_flip < REPAIR_SETTLE_S
            ):
                backoff = REPAIR_TICK_S
                continue
            repaired = await self._repair_standby()
            if repaired:
                backoff = REPAIR_TICK_S
            else:
                backoff = min(backoff * 2, REPAIR_BACKOFF_MAX_S)
                _LOGGER.debug(
                    "supervisor: standby repair failed; next "
                    "attempt in %.0fs", backoff,
                )

    async def _repair_active(self) -> bool:
        """Rebuild the active session when the pool has none.

        Runs ONLY when there is no usable active, so it carries no
        flap risk: there is nothing to flap between. Order:

        1. If a connected standby exists, promote it in place (the
           cheap path; equivalent to a flip with no failed host).
        2. Otherwise probe the current primary, then the failover
           host, and install the first that answers as the active.

        Host roles are updated so `primary_host` always names the
        active, and `on_failover_switch` fires when they change so
        consumers repaint their bridge cards.

        Device-list exchange is skipped when the shared cache is
        already populated: an outage does not invalidate uuid->name
        or dimmable mappings, and live state is reconciled by the
        EVENT stream. That keeps recovery off the slow path.
        """
        async with self._flip_lock:
            if self._stopped:
                return False
            if self.active is not None and self.active.is_connected():
                return True
            # Drop a dead active so it cannot hold a bridge TCP slot
            # we may be about to reconnect to.
            if self.active is not None:
                dead = self.active
                self.active = None
                await self._teardown(dead)
            # Cheap path: a live standby is exactly what we need.
            if self.standby is not None and self.standby.is_connected():
                promoted = self.standby
                self.standby = None
                self._standby_found = False
                self.active = promoted
                self.primary_host, self.failover_host = (
                    self.failover_host, self.primary_host,
                )
                self._last_flip = asyncio.get_running_loop().time()
                _LOGGER.info(
                    "supervisor: promoted standby to active: "
                    "primary=%s failover=%s",
                    self.primary_host, self.failover_host,
                )
                self._fire_on_failover_switch()
                return True
            # Drop a stale (disconnected) standby before probing, so
            # it cannot own the slot of a host we try to connect.
            if self.standby is not None:
                stale = self.standby
                self.standby = None
                self._standby_found = False
                await self._teardown(stale)
            for target in (self.primary_host, self.failover_host):
                if self._stopped:
                    return False
                if not await _tcp_probe(target):
                    _LOGGER.debug(
                        "supervisor: %s not accepting TCP; skipping "
                        "active connect", target,
                    )
                    continue
                try:
                    new_active = await self._connect_session(
                        target, find_devices=not self._devices,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    _LOGGER.debug(
                        "supervisor: active connect to %s failed: %s",
                        target, exc,
                    )
                    continue
                if await self._discard_if_stopped(new_active):
                    return False
                other = (
                    self.failover_host
                    if target == self.primary_host
                    else self.primary_host
                )
                changed = target != self.primary_host
                self.primary_host = target
                self.failover_host = other
                self.active = new_active
                _LOGGER.info(
                    "supervisor: active session restored on %s "
                    "(failover=%s)", target, other,
                )
                if changed:
                    self._fire_on_failover_switch()
                return True
            return False

    async def _repair_standby(self) -> bool:
        """One probe-gated attempt to rebuild the standby session."""
        target = self.failover_host
        if not await _tcp_probe(target):
            _LOGGER.debug(
                "supervisor: %s not accepting TCP; skipping "
                "connect attempt", target,
            )
            return False
        try:
            standby = await self._connect_session(
                target, find_devices=False,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.debug(
                "supervisor: standby connect to %s failed: %s",
                target, exc,
            )
            return False
        if await self._discard_if_stopped(standby):
            return False
        # The failover host may have changed while we connected
        # (a flip mid-repair). Identity dispatch keeps this safe:
        # install only if the target still matches.
        if target != self.failover_host:
            await self._teardown(standby)
            return False
        self.standby = standby
        self._standby_found = False
        _LOGGER.info(
            "supervisor: standby session restored on %s", target,
        )
        return True

    async def _verify_standby_once(self) -> None:
        """One-time device-list verification on a fresh standby.

        Data point for the two-manager experiment: does the standby
        bridge answer queries on its session? Failure is logged and
        tolerated; the session still holds the TCP slot and still
        feeds the shared cache with whatever EVENTs it relays.
        """
        if self._standby_found or self.standby is None:
            return
        self._standby_found = True
        try:
            await self.standby.find_devices()
            self._apply_callbacks()
            _LOGGER.info(
                "standby %s answered device-list (devices=%d)",
                self.failover_host, len(self._devices),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            _LOGGER.warning(
                "standby %s did not answer device-list: %s "
                "(session kept; EVENT relay may still work)",
                self.failover_host, exc,
            )

    # ----- control_device send path ----------------------------

    async def control_device(
        self, uuid: str, power: bool, dim: int | None = None,
    ) -> None:
        """Send a device control command through the active session.

        Uses `Deako._control_device_strict` so send failures raise.
        On failure of the active: exactly one flip attempt and one
        retry on the promoted standby. When nothing is usable the
        call raises `NoSocketException` QUICKLY (no inline connects;
        the supervisor repairs in the background), which matches the
        integration's drop-and-recover-in-background contract.
        """
        if self._stopped:
            raise NoSocketException("pool stopped")
        if self.active is None or not self.active.is_connected():
            # Promote a live standby if there is one; otherwise take
            # one bounded, probe-gated shot at rebuilding the active
            # inline. Without the inline attempt the first command
            # after a full outage always fails while the supervisor
            # waits out its backoff, which reads to the user as the
            # integration being down.
            recovered = await self._repair_active()
            if not recovered:
                self._repair_wake.set()
                raise NoSocketException(
                    f"no usable connection: neither "
                    f"{self.primary_host} nor {self.failover_host} "
                    f"is reachable; background repair running",
                )
        active = self.active
        if active is None:
            raise NoSocketException(
                "no active connection after flip",
            )
        failed_host = self.primary_host
        try:
            # pylint: disable-next=protected-access
            await active._control_device_strict(uuid, power, dim)
            return
        except OSError:
            # NoSocketException subclasses OSError.
            _LOGGER.warning(
                "control_device send failed on active %s; "
                "flipping to standby",
                failed_host,
            )
        flipped = await self._flip(failed_host)
        if not flipped:
            msg = (
                f"control_device failed on {failed_host} and no "
                f"connected standby to flip to"
            )
            _LOGGER.error(msg)
            raise NoSocketException(msg)
        new_active = self.active
        if new_active is None:
            raise NoSocketException(
                f"flip reported success but no active session "
                f"remains; last active host was {failed_host}",
            )
        try:
            # pylint: disable-next=protected-access
            await new_active._control_device_strict(uuid, power, dim)
        except OSError as exc:
            msg = (
                f"control_device failed twice across flip: "
                f"first active={failed_host}, "
                f"second active={self.primary_host}: {exc}"
            )
            _LOGGER.error(msg)
            raise NoSocketException(msg) from exc
