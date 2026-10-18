"""The merge lane's halt state: which lanes are halted, by whom, and the operator halt.

``SpeculativeMergeWorker`` composes one :class:`HaltState`, handing it the
lane tuple and the callback that wakes its merger.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

logger = logging.getLogger('orchestrator.merge_lane')


class HaltState:
    """Per-lane halt state and the operator halt.

    Each lane has an independent asyncio.Event (set = not halted; cleared =
    halted) and an optional owner esc_id.  Legacy all-lanes methods
    (halt_for_wip, unhalt_all_lanes, …) are preserved as backward-compatible
    shims.
    """

    def __init__(self, lanes: tuple[str, ...], *, on_resume: Callable[[], None]) -> None:
        self._lanes = lanes
        self._on_resume = on_resume
        self._lane_halt = {ln: asyncio.Event() for ln in lanes}
        for ln in lanes:
            self._lane_halt[ln].set()
        self._lane_halt_owner: dict[str, str | None] = {ln: None for ln in lanes}
        self._operator_halt = asyncio.Event()

    # ── per-lane public API ────────────────────────────────────────────────

    def halt_lane(self, lane: str, reason: str, *, owner_esc_id: str | None = None) -> None:
        """Halt a specific lane."""
        logger.warning('Merge queue lane %r halted: %s', lane, reason)
        self._lane_halt[lane].clear()
        if owner_esc_id is not None:
            self._lane_halt_owner[lane] = owner_esc_id

    def unhalt_lane(self, lane: str, reason: str | None = None) -> None:
        """Resume a specific lane."""
        logger.info(
            'Merge queue lane %r un-halted%s',
            lane,
            f' ({reason})' if reason else '',
        )
        self._lane_halt[lane].set()
        self._lane_halt_owner[lane] = None
        self._on_resume()

    def is_lane_halted(self, lane: str) -> bool:
        """True iff the given lane is currently halted."""
        return not self._lane_halt[lane].is_set()

    # ── per-lane owner API ─────────────────────────────────────────────────

    def set_lane_halt_owner(self, lane: str, esc_id: str) -> None:
        """Register the escalation that owns the halt on *lane*.

        Asserts no different owner is already set (mirrors ``set_halt_owner``
        single-owner invariant).
        """
        current = self._lane_halt_owner[lane]
        assert current is None or current == esc_id, (
            f'lane {lane!r} halt owner already set to {current!r}, '
            f'refusing to overwrite with {esc_id!r}'
        )
        self._lane_halt_owner[lane] = esc_id

    def lane_owned_by(self, esc_id: str) -> str | None:
        """Return the lane owned by *esc_id*, or None."""
        for lane, owner in self._lane_halt_owner.items():
            if owner == esc_id:
                return lane
        return None

    def unhalt_lanes_owned_by(self, esc_id: str, reason: str | None = None) -> list[str]:
        """Un-halt every lane owned by *esc_id*.  Returns list of lanes resumed."""
        resumed = []
        for lane in self._lanes:
            if self._lane_halt_owner.get(lane) == esc_id:
                # Call lane halt setter directly to avoid double-signal; signal once after.
                self._lane_halt[lane].set()
                self._lane_halt_owner[lane] = None
                logger.info('Merge queue lane %r un-halted (owner %r resolved)', lane, esc_id)
                resumed.append(lane)
        if resumed:
            self._on_resume()
        return resumed

    # ── legacy all-lanes shims (backward-compatible) ──────────────────────

    def halt_for_wip(self, reason: str) -> None:
        """Halt the merge queue due to a WIP conflict (all lanes)."""
        logger.warning('Merge queue halted for WIP: %s', reason)
        for lane in self._lanes:
            self._lane_halt[lane].clear()
            self._lane_halt_owner[lane] = None

    def operator_halt(self, reason: str) -> None:
        """Operator-initiated halt: stop the merger AND abort the in-flight verify.

        Backs the ``halt_merge_queue`` escalation tool.  Unlike
        :meth:`halt_for_wip` (the automatic WIP-conflict halt, which intentionally
        lets the verifier keep draining the items behind it), this ALSO raises the
        dedicated operator-halt signal, read through :meth:`is_operator_halted`.
        The verifier's abort-poll loop and verifier-loop top check it: the
        in-flight verify is cancelled (killing its subprocess via the existing
        CancelledError seam) and its merge is re-queued so it re-verifies after
        un-halt — the waiter's future is left pending and per-task retry
        counters are untouched (a halt is not a failure).

        Halts every lane with no owner (like ``halt_for_wip``), so
        ``is_wip_halted`` reports True and ``halt_owner_esc_id`` is None; the
        existing ``unhalt_merge_queue`` / ``force_unhalt_merge_queue`` path
        (→ ``unhalt_all_lanes``, which clears that signal) cleanly reverses
        it without tripping the active-owner refusal.  Synchronous: the verify
        abort happens asynchronously in the verifier within one
        ``VERIFY_ABANDON_POLL_SECS`` interval.
        """
        logger.warning('Merge queue operator-halted: %s', reason)
        self._operator_halt.set()
        for lane in self._lanes:
            self._lane_halt[lane].clear()
            self._lane_halt_owner[lane] = None

    def set_halt_owner(self, esc_id: str) -> None:
        """Register the escalation that owns the current halt.

        The workflow calls this right after submitting its halt-triggering
        escalation.  Sets the owner on all currently-halted lanes.
        Asserts no owner is already set — a double-register indicates a
        double-halt bug that should fail loudly.
        """
        current = self.halt_owner_esc_id
        assert current is None, (
            f'halt owner already set to {current!r}, '
            f'refusing to overwrite with {esc_id!r}'
        )
        halted = [ln for ln in self._lanes if not self._lane_halt[ln].is_set()]
        assert halted, (
            f'set_halt_owner({esc_id!r}) called when no lane is halted — '
            'halt must be active before registering an owner'
        )
        for lane in halted:
            self._lane_halt_owner[lane] = esc_id

    def is_halt_owner(self, esc_id: str) -> bool:
        """True iff esc_id owns any currently-halted lane."""
        for lane in self._lanes:
            if (not self._lane_halt[lane].is_set()
                    and self._lane_halt_owner.get(lane) == esc_id):
                return True
        return False

    def unhalt_all_lanes(self, reason: str | None = None) -> None:
        """Resume ALL lanes unconditionally, clearing every owner.

        This is the global 'resume-all' backstop — e.g. operator
        ``force_unhalt_merge_queue`` or a signal that all active halts should
        be cleared regardless of which escalation owned them.
        """
        logger.info(
            'Merge queue: all lanes un-halted%s',
            f' ({reason})' if reason else '',
        )
        for lane in self._lanes:
            self._lane_halt[lane].set()
            self._lane_halt_owner[lane] = None
        # Also clear any operator halt — this global resume-all backstop (reached
        # via unhalt_wip ← force_unhalt_merge_queue ← unhalt_merge_queue) is the
        # reversal path for operator_halt().
        self._operator_halt.clear()
        self._on_resume()

    def release_lanes_for_shutdown(self) -> None:
        """Make every lane runnable so a stopping merger never waits on a halt.

        Owners and the operator halt are left as they are, and the merger is
        not woken: shutdown has its own signal.
        """
        for lane in self._lanes:
            self._lane_halt[lane].set()

    @property
    def is_wip_halted(self) -> bool:
        """True iff at least one lane is halted."""
        return any(not self._lane_halt[ln].is_set() for ln in self._lanes)

    @property
    def is_operator_halted(self) -> bool:
        """True iff :meth:`operator_halt` is in force, the one halt that aborts a verify."""
        return self._operator_halt.is_set()

    @property
    def halt_owner_esc_id(self) -> str | None:
        """Owner of any currently-halted lane, or None."""
        for lane in self._lanes:
            if not self._lane_halt[lane].is_set():
                owner = self._lane_halt_owner.get(lane)
                if owner is not None:
                    return owner
        return None
