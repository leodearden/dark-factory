"""Offline deep-test lane singleton worker (task 1953, β2).

Implements PRD §5 C2 for the offline-deep lane: a heavy, reify-side test
suite run off the verify hot path, always from the current ``main`` head,
never gating a merge.  ``OfflineLaneWorker`` is a singleton (one loop
coroutine + a lockfile) launched and owned by ``Harness`` (see
``harness._start_offline_lane`` / ``harness._stop_offline_lane``).

Upstream dependencies this module consumes (both already landed on main):

* β1 (task 1951) — ``Harness._offline_lane_notifiee`` slot
  (``harness.py:749``) and its ``_note_offline_lane`` fan-out
  (``harness.py:5031``), wired into ``_note_merge_all``
  (``harness.py:5008``).  The notifiee is awaited synchronously on the
  merge-landed hot path, so :meth:`OfflineLaneWorker.on_post_merge` MUST
  enqueue-and-return promptly (flip a dirty flag and wake a waiter) rather
  than perform the deep-test run inline (see that method's docstring).
* δ (task 1952) — the dedicated warm ``_offline-deep`` worktree:
  ``git_ops.reset_persistent_offline_deep_worktree`` (``git_ops.py:4055``),
  ``git_ops.persistent_offline_deep_worktree_path`` (``git_ops.py:3839``),
  and the ``cleanup_merge_worktree`` prune exemption (``git_ops.py:3789``).

Contract summary (PRD §5 C2):

* **Single-flight** — one loop coroutine (:meth:`OfflineLaneWorker.run`) plus
  a lockfile (:meth:`OfflineLaneWorker.acquire_lock`) enforce that at most one
  offline-deep run is ever in flight, even across process instances.
* **Always-from-head** — each run snapshots ``head =
  await git_ops.get_main_sha()`` at RUN-START (inside
  :meth:`OfflineLaneWorker._run_once`), NOT at trigger time.  The SHAs passed
  to :meth:`on_post_merge` are advisory only and are never stored or used as
  the run head.
* **Coalescing** — an advance that lands *during* a run re-sets the dirty
  flag, producing exactly one coalesced re-run at the (new) then-current
  head afterwards; multiple advances during one run collapse to that single
  re-run.
* **Poll backstop** — a missed trigger (e.g. a crash between the merge and
  the notifiee call) is caught by a cheap periodic ``get_main_sha`` poll;
  correctness lives in the run-start snapshot, not the trigger, so a missed
  trigger only costs granularity, never correctness.
* **Never a gate** — the worker runs as an out-of-band background asyncio
  task at idle class; it never touches the merge queue or the merge lane's
  own ``target/``, and a failed pass is logged and backed off, never raised
  into the merge path.

Each run is one pass of every enabled command through
``orchestrator/src/orchestrator/offline_lane_pass.py::run_pass``.  The red path is
``orchestrator/src/orchestrator/offline_lane_episodes.py::LaneEngine``: the poll
backstop calls its tick, and runs the rechecks it says are due.
"""

from __future__ import annotations

import asyncio
import fcntl
import functools
import logging
import os
from collections.abc import Awaitable, Callable, Collection, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

from orchestrator import offline_lane_confirm, offline_lane_runs
from orchestrator.offline_lane_episodes import LaneEngine
from orchestrator.offline_lane_pass import (
    ConfirmSeam,
    DetectSeam,
    LaneLeg,
    PassOutcomeHook,
    run_pass,
)
from orchestrator.offline_lane_records import Leg, RunRecordStore

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

    from orchestrator.config import LaneCommand, OrchestratorConfig
    from orchestrator.flake_ledger import FlakeLedgerTaskClient
    from orchestrator.git_ops import GitOps

logger = logging.getLogger(__name__)

# Fixed quiescent gap after a failed run() pass (seconds).  Mirrors
# harness._BG_LOOP_FAILURE_BACKOFF_SECS (harness.py:107) — duplicated here
# (rather than imported) to avoid a harness <-> offline_lane import cycle,
# since harness.py imports OfflineLaneWorker from this module.
_OFFLINE_LANE_FAILURE_BACKOFF_SECS: float = 60.0

# Ceiling on the exponential backoff applied to CONSECUTIVE run() failures
# (task 2016 amendment). Since the Bug #2 fix means a persistently-raising
# red-handling/MCP leg now retries the head every pass instead of silently
# advancing past it, a stuck outage would otherwise re-run the entire heavy
# offline-deep suite every fixed 60s backoff forever. Doubling the backoff
# per consecutive failure (capped here) bounds that cost while preserving
# fail-open retry — never a permanent give-up.
_OFFLINE_LANE_MAX_FAILURE_BACKOFF_SECS: float = 900.0

#: Signature of the injectable heavy-suite seam: (worktree path, head SHA,
#: test-thread count) -> (return code, output tail).
SuiteRunner = Callable[[Path, str, int], Awaitable[tuple[int, str]]]

#: Signature of the injectable GENERIC per-project command seam (task 2789):
#: (LaneCommand, worktree path, head SHA) -> (return code, output tail).
CommandRunner = Callable[['LaneCommand', Path, str], Awaitable[tuple[int, str]]]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class OfflineLaneWorker:
    """Singleton offline-deep lane worker — single-flight, coalescing, from-head.

    Args:
        git_ops: The orchestrator's ``GitOps`` instance (head snapshot,
            warm-worktree reset, and the engine's ancestry read).
        config: The ``OrchestratorConfig`` instance.
        lock_path: Path to this worker's dedicated lockfile (keyword-only).
        suite_runner, infra_runner, command_runner: Optional detect seams of
            the numeric, infra and generic legs; default to
            ``orchestrator/src/orchestrator/offline_lane_runs.py``'s.
        confirm: Optional confirm seam returning a pass outcome; defaults to
            ``orchestrator/src/orchestrator/offline_lane_confirm.py::confirm``.
        task_client, escalation_queue: The engine's filing and escalation
            seams; without them it files and raises nothing, and logs so.
        pass_outcome_hooks: The section 5.7 hooks each red pass asks.
        now: The clock the run records and the engine share.
    """

    def __init__(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        *,
        lock_path: str | Path,
        suite_runner: SuiteRunner | None = None,
        infra_runner: SuiteRunner | None = None,
        command_runner: CommandRunner | None = None,
        confirm: ConfirmSeam | None = None,
        task_client: FlakeLedgerTaskClient | None = None,
        escalation_queue: EscalationQueue | None = None,
        pass_outcome_hooks: Sequence[PassOutcomeHook] = (),
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.git_ops = git_ops
        self.config = config
        self.lock_path = Path(lock_path)
        self._now = now or _utc_now
        self.records = RunRecordStore(config.project_root, clock=self._now)
        self.suite_runner: SuiteRunner = suite_runner or offline_lane_runs.suite_runner(self.records)
        self.infra_runner: SuiteRunner = infra_runner or offline_lane_runs.infra_runner(self.records)
        self.command_runner: CommandRunner = command_runner or offline_lane_runs.command_runner(self.records)
        self._confirm: ConfirmSeam = confirm or offline_lane_confirm.confirm
        self._engine = LaneEngine(
            config, git_ops, task_client=task_client, escalation_queue=escalation_queue, now=self._now,
        )
        self._hooks = tuple(pass_outcome_hooks)

        # Coalescing state — flipped by on_post_merge (and the poll
        # backstop), cleared at the START of _run_once (clear-before-snapshot).
        self._dirty: bool = False
        # Woken by on_post_merge; timed-out by run()'s poll backstop.
        self._wake: asyncio.Event = asyncio.Event()
        # The snapshot head of the most recently COMPLETED run (never an
        # advisory trigger SHA) — used by the poll backstop to detect a
        # missed advance.
        self._last_run_head: str | None = None
        # Held open for the lifetime of a successful acquire_lock() call.
        self._lock_file: IO | None = None
        # Count of consecutive run() pass failures (reset to 0 on any
        # successful _run_once) — drives the capped exponential backoff
        # below so a persistent outage never hammers the heavy suite at a
        # fixed 60s cadence forever (task 2016 amendment).
        self._consecutive_failures: int = 0

    # ------------------------------------------------------------------
    # Trigger seam (β1 consumer)
    # ------------------------------------------------------------------

    async def on_post_merge(self, task_id: str, base_sha: str, head_sha: str) -> None:
        """β1's ``on_post_merge`` notifiee — enqueue-and-return, never inline.

        Registered as ``harness._offline_lane_notifiee`` by
        ``Harness._start_offline_lane``.  Per the β1 contract
        (``harness.py:5040-5046``), this is awaited SYNCHRONOUSLY on the
        merge-landed hot path ahead of the diff fetch and the
        service-restart coordinator fan-out — it must enqueue-and-return
        promptly rather than block on the deep-test run.

        The SHAs are advisory only (never stored): the eventual run always
        snapshots its own head at run-start (see :meth:`_run_once`).
        """
        self._dirty = True
        self._wake.set()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Singleton coalescing loop — single-flight via one coroutine.

        See module docstring for the full single-flight / coalescing / poll
        backstop / fail-open contract.

        Coalescing core: whenever ``_dirty`` is set, run once and IMMEDIATELY
        re-check ``_dirty`` — since :meth:`_run_once` clears it before its own
        head snapshot, a re-set during that run (one or many times) collapses
        into exactly one further coalesced re-run at the new head, never more.
        Otherwise, clear and wait on the wake event (set by
        :meth:`on_post_merge`), bounded by ``offline_lane_poll_interval_secs``.

        Poll backstop: a wait that times out (no trigger arrived) runs
        :meth:`_poll_tick`, which polls ``get_main_sha`` — a mismatch against ``_last_run_head`` means
        an advance was missed (e.g. a crash between the merge and the
        notifiee call) and is run-worthy. Correctness lives in
        :meth:`_run_once`'s own head snapshot, not here, so a missed trigger
        only costs granularity (up to one poll interval), never correctness.

        Fail-open: modeled verbatim on ``harness._main_tip_sweep_loop``
        (``harness.py:6056``) — a failed pass is logged as a bounded
        one-line summary (exc type + message) via ``logger.error``, NEVER
        ``logger.exception`` (whose O(frame-count) traceback formatting can
        starve the event loop on a pathological traceback), and is followed
        by a fixed backoff sleep so the loop can never tight-spin.
        ``asyncio.CancelledError`` always propagates — never swallowed.
        Since the failed run never updates ``_last_run_head``, the poll
        backstop (above) naturally re-flags the still-outstanding advance
        once the backoff elapses — a broken pass costs a retry, never a
        wedge (worker leg of C7, never-a-gate).

        Bounded-cost retry (task 2016 amendment): the backoff doubles per
        CONSECUTIVE failure (reset on any successful pass), capped at
        :data:`_OFFLINE_LANE_MAX_FAILURE_BACKOFF_SECS` — so a persistent
        red-handling/MCP outage doesn't re-run the entire heavy suite at the
        fixed base cadence forever, while still never giving up permanently.
        """
        while True:
            try:
                if self._dirty:
                    await self._run_once()
                    self._consecutive_failures = 0
                    continue
                self._wake.clear()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=self.config.git.offline_lane_poll_interval_secs,
                    )
                except TimeoutError:
                    await self._poll_tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._consecutive_failures += 1
                backoff = min(
                    _OFFLINE_LANE_FAILURE_BACKOFF_SECS * (2 ** (self._consecutive_failures - 1)),
                    _OFFLINE_LANE_MAX_FAILURE_BACKOFF_SECS,
                )
                logger.error(
                    'offline-lane: run() pass failed (consecutive=%d): %s: %s',
                    self._consecutive_failures, type(exc).__name__, exc,
                )
                await asyncio.sleep(backoff)

    async def _run_once(self) -> None:
        """Snapshot head, reset the warm worktree, and run one pass of every enabled command.

        Clear-before-snapshot: ``_dirty`` is cleared FIRST, before the
        ``get_main_sha`` snapshot, so no advance is ever lost — an advance
        landing after the clear either lands inside this run's snapshot
        (fine) or re-sets ``_dirty`` for a coalesced re-run (fine).
        ``_last_run_head`` advances only after every pass returned, so a
        raising pass leaves the head for the poll backstop to retry.
        """
        self._dirty = False
        head = await self.git_ops.get_main_sha()
        if not head:
            logger.warning('offline-lane: get_main_sha returned empty head; skipping run')
            return
        pinned = await asyncio.to_thread(self._engine.pinned_pass_ids)
        await asyncio.to_thread(self.records.prune, self._now(), pinned)
        wt = await self.git_ops.reset_persistent_offline_deep_worktree(head)
        await self._run_legs(wt, head, self._enabled_legs())
        self._last_run_head = head

    async def _poll_tick(self) -> None:
        """The idle poll: the engine's tick, then the backstop for a missed advance, else the due rechecks."""
        head = await self.git_ops.get_main_sha()
        if not head:
            return
        result = await self._engine.tick(head=head)
        if head != self._last_run_head:
            self._dirty = True
        elif result.commands_due and not self._dirty:
            await self._recheck(head, result.commands_due)

    async def _recheck(self, head: str, commands: Collection[str]) -> None:
        """One pass of each due command at *head*; ``_dirty`` and ``_last_run_head`` stay the backstop's."""
        legs = tuple(leg for leg in self._enabled_legs() if leg.command in commands)
        if not legs:
            return
        wt = await self.git_ops.reset_persistent_offline_deep_worktree(head)
        await self._run_legs(wt, head, legs)

    def _enabled_legs(self) -> tuple[LaneLeg, ...]:
        git = self.config.git
        legacy = [
            leg for leg, enabled in (
                (LaneLeg.numeric(), git.offline_lane_legacy_numeric_enabled),
                (LaneLeg.infra(), git.offline_lane_infra_enabled),
            ) if enabled
        ]
        generic = [LaneLeg.generic(cmd) for cmd in git.offline_lane_commands if cmd.enabled]
        return (*legacy, *generic)

    def _detect_seam(self, target: LaneLeg, wt: Path, head: str) -> DetectSeam:
        if target.lane_command is not None:
            return functools.partial(self.command_runner, target.lane_command, wt, head)
        runner = self.suite_runner if target.leg is Leg.NUMERIC else self.infra_runner
        return functools.partial(runner, wt, head, self.config.git.offline_lane_test_threads)

    async def _run_legs(self, wt: Path, head: str, legs: Sequence[LaneLeg]) -> None:
        for target in legs:
            await run_pass(
                target,
                worktree=wt,
                head=head,
                detect=self._detect_seam(target, wt, head),
                confirm=self._confirm,
                store=self.records,
                engine=self._engine,
                hooks=self._hooks,
            )

    # ------------------------------------------------------------------
    # Lockfile singleton
    # ------------------------------------------------------------------

    def acquire_lock(self) -> bool:
        """Acquire this worker's dedicated exclusive lockfile.

        Modeled on ``harness._acquire_project_lock`` (``harness.py:212``)
        but returns ``False`` on contention instead of raising
        ``SystemExit`` — a refused acquire is an expected "second instance"
        outcome the caller (``Harness._start_offline_lane``) handles
        fail-open (log + skip), not a fatal error.
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(self.lock_path, 'w')  # noqa: SIM115
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Another instance holds the lock — read its diagnostic info.
            try:
                with open(self.lock_path) as f:
                    info = f.read().strip()
            except OSError:
                info = '(unknown)'
            logger.error(
                'offline-lane: another instance already holds the lock '
                '(holder: %s, lock file: %s) — refusing to start',
                info, self.lock_path,
            )
            lock_file.close()
            return False

        # Write diagnostic info for anyone who tries to acquire next.
        lock_file.truncate(0)
        lock_file.seek(0)
        lock_file.write(f'PID {os.getpid()} started {datetime.now(UTC).isoformat()}')
        lock_file.flush()
        self._lock_file = lock_file
        return True

    def release_lock(self) -> None:
        """Release a previously-acquired lockfile, if held."""
        if self._lock_file is not None:
            self._lock_file.close()
            self._lock_file = None
