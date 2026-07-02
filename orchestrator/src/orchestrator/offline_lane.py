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

Out of scope for β2 (deferred to β3, which depends on this task): failure
fingerprinting, dedup fix-task spawn, and escalation staging.  A run here
records only pass/fail + head + duration.
"""

from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

    from orchestrator.config import OrchestratorConfig
    from orchestrator.git_ops import GitOps

logger = logging.getLogger(__name__)

# Fixed quiescent gap after a failed run() pass (seconds).  Mirrors
# harness._BG_LOOP_FAILURE_BACKOFF_SECS (harness.py:107) — duplicated here
# (rather than imported) to avoid a harness <-> offline_lane import cycle,
# since harness.py imports OfflineLaneWorker from this module.
_OFFLINE_LANE_FAILURE_BACKOFF_SECS: float = 60.0

# Default suite-run seam script (Part A, gated separately at ζ — see module
# docstring's cross-project scope boundary).
_RUN_OFFLINE_DEEP_SCRIPT: str = 'scripts/run-offline-deep.sh'

# Default infra sub-run seam script + scope (task 1959, IE1): reify's H9
# (reify:4929) host-exclusive runner.  IE-D2 — `--scope host-infra`
# SELF-ACQUIRES the H8 Lane-X flock, so no DF-side flock is needed here.
_RUN_ALL_INFRA_SCRIPT: str = 'tests/infra/run_all.sh'
_INFRA_SCOPE: str = 'host-infra'

# Matches reify H9's per-test result lines, e.g. '  RESULT: FAIL (test_x.sh)'
# or '  RESULT: PASS (test_y.sh)'.  Only FAIL lines are of interest to the
# infra confirmation runner; the captured group is the test-FILE-name.
_INFRA_RESULT_FAIL_RE = re.compile(r'^\s*RESULT:\s*FAIL\s*\((?P<name>.+?)\)\s*$')


def _parse_infra_failures(text: str) -> list[str]:
    """Extract still-failing test-file-names from reify H9's stdout (IE1).

    Matches per-line ``RESULT: FAIL (<name>)`` lines (PASS lines and any
    other output are ignored) and returns the sorted, de-duplicated set of
    captured names.  Used by
    :meth:`OfflineLaneWorker._default_infra_confirmation_run` (kept
    module-level so it is trivially unit-testable in isolation).
    """
    names = {
        m.group('name')
        for line in text.splitlines()
        if (m := _INFRA_RESULT_FAIL_RE.match(line))
    }
    return sorted(names)

#: Signature of the injectable heavy-suite seam: (worktree path, head SHA,
#: test-thread count) -> (return code, output tail).
SuiteRunner = Callable[[Path, str, int], Awaitable[tuple[int, str]]]

#: Signature of the injectable confirmation-rerun seam (β3): (worktree path,
#: head SHA) -> confirmed-still-failing test IDs.  An empty return means the
#: original failure did not reproduce in isolation (flake, B6).
ConfirmationRunner = Callable[[Path, str], Awaitable[list[str]]]


class OfflineLaneTaskClient(Protocol):
    """Pluggable interface for filing/updating the offline-lane fix task (β3).

    Encapsulates the MCP ``submit_task`` seam so this module never talks to
    the fused-memory MCP directly — mirrors ``SuiteRunner``'s cross-project
    scope boundary. The real implementation (backed by the
    ``_post_submit_tasks`` httpx pattern) is wired by the harness; unit tests
    inject a fake.
    """

    async def submit_fix_task(self, arguments: dict) -> str:
        """Submit a new fix task (see ``build_offline_lane_fix_task_arguments``);
        returns the new task's id."""
        ...  # pragma: no cover

    async def append_suspect_range(self, task_id: str, suspect_range: str) -> None:
        """Append a new suspect commit range to an already-open fix task."""
        ...  # pragma: no cover

    async def get_status(self, task_id: str) -> str:
        """Return the current status of a previously filed fix task."""
        ...  # pragma: no cover


#: Fix-task statuses that are terminal and non-done — the fix "can't land" —
#: so :meth:`OfflineLaneWorker._maybe_promote_blocker` promotes straight to
#: escalate_blocker (B7/C4) regardless of the N-advances count.  ``blocked``
#: is deliberately excluded: it is a resolvable escalation-in-progress state,
#: not a can't-land verdict.
_TERMINAL_NON_DONE: frozenset[str] = frozenset({'cancelled', 'deferred'})


class OfflineLaneWorker:
    """Singleton offline-deep lane worker — single-flight, coalescing, from-head.

    Args:
        git_ops: The orchestrator's ``GitOps`` instance (head snapshot +
            warm-worktree reset).
        config: The ``OrchestratorConfig`` instance (reads
            ``config.git.offline_lane_test_threads`` and
            ``config.git.offline_lane_poll_interval_secs``).
        lock_path: Path to this worker's dedicated lockfile (keyword-only).
        suite_runner: Optional injectable async seam
            ``(wt_path, head, threads) -> (rc, tail)`` for the heavy suite
            run.  Defaults to :meth:`_default_run_suite` (a real
            ``scripts/run-offline-deep.sh`` subprocess invocation) when not
            supplied.
        confirmation_runner: Optional injectable async seam (β3)
            ``(wt_path, head) -> confirmed_still_failing_ids`` used by
            :meth:`_handle_red_run` to filter flakes before fingerprinting.
            Defaults to :meth:`_default_confirmation_run` when not supplied.
        infra_runner: Optional injectable async seam (task 1959, IE1),
            same ``SuiteRunner`` signature as ``suite_runner``, for the
            second (infra) sub-run: reify's ``run_all --scope host-infra``
            (H9 = reify:4929).  Defaults to :meth:`_default_run_infra` when
            not supplied.  IE-D2: the infra scope self-acquires the H8
            Lane-X flock — no DF-side flock is taken here.
        infra_confirmation_runner: Optional injectable async seam (task
            1959, IE1), same ``ConfirmationRunner`` signature as
            ``confirmation_runner``, used by :meth:`_handle_red_run` (via
            its optional ``confirmation_runner`` kwarg) to confirm an infra
            sub-run red result.  Defaults to
            :meth:`_default_infra_confirmation_run` when not supplied.
        task_client: Optional :class:`OfflineLaneTaskClient` (β3) used to
            file/update the dedup'd fix task for a confirmed red run.
            Defaults to ``None`` — the red path degrades to a log-only no-op
            when unwired (never a crash; worker leg of C7).
        escalation_queue: Optional ``EscalationQueue`` (β3) used to raise the
            L0 info escalation on a new fix-task file and the staged L2
            ``escalate_blocker`` promotion.  Defaults to ``None`` — same
            log-only degrade as ``task_client``.
    """

    def __init__(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        *,
        lock_path: str | Path,
        suite_runner: SuiteRunner | None = None,
        confirmation_runner: ConfirmationRunner | None = None,
        infra_runner: SuiteRunner | None = None,
        infra_confirmation_runner: ConfirmationRunner | None = None,
        task_client: OfflineLaneTaskClient | None = None,
        escalation_queue: EscalationQueue | None = None,
    ) -> None:
        self.git_ops = git_ops
        self.config = config
        self.lock_path = Path(lock_path)
        self.suite_runner: SuiteRunner = (
            suite_runner if suite_runner is not None else self._default_run_suite
        )
        self.confirmation_runner: ConfirmationRunner = (
            confirmation_runner
            if confirmation_runner is not None
            else self._default_confirmation_run
        )
        self.infra_runner: SuiteRunner = (
            infra_runner if infra_runner is not None else self._default_run_infra
        )
        self.infra_confirmation_runner: ConfirmationRunner = (
            infra_confirmation_runner
            if infra_confirmation_runner is not None
            else self._default_infra_confirmation_run
        )
        self.task_client = task_client
        self.escalation_queue = escalation_queue

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

        # Red-path state (β3) — keyed by compute_failing_test_set_fingerprint.
        # Maps a confirmed failing-test-set fingerprint to its open fix task
        # id; an in-flight fingerprint absorbs further same-set red advances
        # (append suspect range) instead of spawning a duplicate task (B5).
        self.open_fix_tasks: dict[str, str] = {}
        # Consecutive confirmed-red advances recorded per open fingerprint —
        # compared against config.git.offline_lane_red_advances_before_blocker
        # for the N-advances staged escalate_blocker promotion (B7/C4).
        self._red_advance_counts: dict[str, int] = {}
        # Fingerprints already promoted to escalate_blocker — promotion is
        # idempotent per fingerprint (never re-files once escalated).
        self._promoted_blockers: set[str] = set()
        # Head of the most recently PASSING run — the cheapest sound lower
        # bound for a new fix task's suspect commit range.  None until the
        # first green run is observed.
        self._last_green_head: str | None = None

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

        Poll backstop: a wait that times out (no trigger arrived) polls
        ``get_main_sha`` itself — a mismatch against ``_last_run_head`` means
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
        """
        while True:
            try:
                if self._dirty:
                    await self._run_once()
                    continue
                self._wake.clear()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=self.config.git.offline_lane_poll_interval_secs,
                    )
                except TimeoutError:
                    head = await self.git_ops.get_main_sha()
                    if head and head != self._last_run_head:
                        self._dirty = True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    'offline-lane: run() pass failed: %s: %s',
                    type(exc).__name__, exc,
                )
                await asyncio.sleep(_OFFLINE_LANE_FAILURE_BACKOFF_SECS)

    async def _run_once(self) -> None:
        """Snapshot head, reset the warm worktree, and invoke both sub-run seams.

        See module docstring — always-from-head, clear-before-snapshot.

        Clear-before-snapshot: ``_dirty`` is cleared FIRST, before the
        ``get_main_sha`` snapshot, so no advance is ever lost — an advance
        landing after the clear either lands inside this run's snapshot
        (fine) or re-sets ``_dirty`` for a coalesced re-run (fine).
        Clearing AFTER the snapshot would open a lost-update window.

        Two sub-runs per coalesced cadence, both at this ONE snapshot head
        and in the SAME ``_offline-deep`` worktree (task 1959, IE1):

        1. NUMERIC — the existing ``scripts/run-offline-deep.sh`` suite via
           :attr:`suite_runner`. Unchanged: on red it calls
           :meth:`_handle_red_run` BARE (``_handle_red_run(wt, head)``),
           preserving back-compat with the numeric-only red path.
        2. INFRA — reify's ``run_all --scope host-infra`` (H9 = reify:4929)
           via :attr:`infra_runner`, gated on
           ``config.git.offline_lane_infra_enabled`` (default False =
           numeric-only, byte-identical to prior behavior). On red, it
           reuses :meth:`_handle_red_run` with :attr:`infra_confirmation_runner`
           injected via the optional ``confirmation_runner`` kwarg — the
           entire confirm→fingerprint→file/update→escalate machinery is
           shared, unchanged (IE-D1/§6: no new dedup mechanism). IE-D2: the
           infra scope self-acquires the H8 Lane-X flock — no DF-side flock
           is taken here. IE-D4: infra ``.sh``/cargo tests never touch
           ``_offline-deep/target/``, so reusing δ's worktree does not
           violate C5's single-consumer-of-target invariant.

        Both sub-runs are logged (the run record). ``_last_green_head``
        advances only when ALL EXECUTED sub-runs passed at this head — the
        last-both-green head is a sound (if sometimes wider) lower bound for
        either sub-run's suspect range (see :meth:`_suspect_range`). When
        infra is disabled, the infra leg is vacuously green, so numeric-only
        semantics are unchanged. Never a gate (C7): a red infra result files
        a normal queued fix task — it never blocks the merge queue.
        """
        self._dirty = False
        head = await self.git_ops.get_main_sha()
        if not head:
            logger.warning('offline-lane: get_main_sha returned empty head; skipping run')
            return
        wt = await self.git_ops.reset_persistent_offline_deep_worktree(head)
        threads = self.config.git.offline_lane_test_threads

        start = time.monotonic()
        rc, _tail = await self.suite_runner(wt, head, threads)
        duration = time.monotonic() - start
        self._last_run_head = head
        logger.info(
            'offline-lane: numeric sub-run head=%s status=%s duration=%.1fs',
            head[:12], 'PASS' if rc == 0 else 'FAIL', duration,
        )
        numeric_green = rc == 0
        if not numeric_green:
            await self._handle_red_run(wt, head)

        infra_green = True
        if self.config.git.offline_lane_infra_enabled:
            start = time.monotonic()
            rc, _tail = await self.infra_runner(wt, head, threads)
            duration = time.monotonic() - start
            logger.info(
                'offline-lane: infra sub-run head=%s status=%s duration=%.1fs',
                head[:12], 'PASS' if rc == 0 else 'FAIL', duration,
            )
            infra_green = rc == 0
            if not infra_green:
                await self._handle_red_run(
                    wt, head, confirmation_runner=self.infra_confirmation_runner,
                )

        if numeric_green and infra_green:
            self._last_green_head = head

    # ------------------------------------------------------------------
    # Red-path handling — confirmation, fingerprint, dedup'd fix-task spawn,
    # staged escalation (β3, task 1954, PRD §5 C3/C4)
    # ------------------------------------------------------------------

    async def _handle_red_run(
        self, wt: Path, head: str, *, confirmation_runner: ConfirmationRunner | None = None,
    ) -> None:
        """Handle a confirmed-red ``_run_once`` pass: confirm, fingerprint, file/update.

        (1) CONFIRMATION RE-RUN: re-runs only the originally-failing tests,
        isolated/serial, ONCE, via *confirmation_runner* if supplied, else
        :attr:`confirmation_runner` (the numeric seam). An empty result
        means the failure did not reproduce — intermittent nondeterminism
        (B6): log only, no task, no escalation.

        (2)+(3) UPDATE-OR-FILE: fingerprints the confirmed failing-test SET
        (never ``main_sha`` — DB3/C3) and dispatches on whether that
        fingerprint already has an open fix task. An already-open
        fingerprint is UPDATED (:meth:`_update_existing_fix_task` — append
        the suspect range, no new task/escalation, B5); a new fingerprint is
        FILED (:meth:`_file_new_fix_task` — new fix task + INFO escalation, B4).

        The optional *confirmation_runner* kwarg (task 1959, IE1) lets the
        infra sub-run's red path reuse this entire method with its OWN
        confirmation seam (:attr:`infra_confirmation_runner`) while the
        numeric call site stays bare (``_handle_red_run(wt, head)``,
        unchanged) and keeps using :attr:`confirmation_runner`. Everything
        downstream of the confirmation re-run — fingerprint, file/update,
        escalation — is content-agnostic and unchanged, so infra
        test-file-names dedup exactly like numeric test IDs (IE-D1/§6): no
        new dedup mechanism.
        """
        runner = confirmation_runner if confirmation_runner is not None else self.confirmation_runner
        confirmed = await runner(wt, head)
        if not confirmed:
            logger.info(
                'offline-lane: intermittent nondeterminism — confirmation '
                're-run at head=%s found no still-failing test; not filing '
                'a fix task',
                head[:12],
            )
            return

        from orchestrator.workflow import compute_failing_test_set_fingerprint

        fp = compute_failing_test_set_fingerprint(confirmed)
        if fp in self.open_fix_tasks:
            await self._update_existing_fix_task(fp, head)
        else:
            await self._file_new_fix_task(fp, confirmed, head)

    def _suspect_range(self, head: str) -> str:
        """Cheapest sound over-approximation of the commits that could have
        introduced a confirmed break: the last-known-green head through the
        current run's head, or just ``head`` when no green run has been
        observed yet (``_last_green_head`` is still ``None``)."""
        if self._last_green_head:
            return f'{self._last_green_head}..{head}'
        return head

    async def _file_new_fix_task(self, fp: str, confirmed: list[str], head: str) -> None:
        """File a NEW dedup'd fix task for a confirmed-red failing-test set (B4).

        Builds the ``submit_task`` argument block via
        :func:`~orchestrator.workflow.build_offline_lane_fix_task_arguments`
        and — when :attr:`task_client` is wired — files it and records *fp*
        in :attr:`open_fix_tasks` / :attr:`_red_advance_counts` so a
        same-set recurrence updates rather than re-files (see
        :meth:`_update_existing_fix_task`). Degrades to a log-only no-op
        when :attr:`task_client` is unwired (never a crash; worker leg of
        C7) — no fingerprint is recorded in that case, since there is no
        fix task to dedup against.
        """
        from orchestrator.workflow import build_offline_lane_fix_task_arguments

        suspect = self._suspect_range(head)
        arguments = build_offline_lane_fix_task_arguments(
            confirmed, suspect, fp, self.config.project_root, head,
        )
        if self.task_client is None:
            logger.info(
                'offline-lane: confirmed red run (tests=%s) but no task_client '
                'wired — skipping fix-task file (log-only degrade)',
                ', '.join(sorted(confirmed)),
            )
            return
        task_id = await self.task_client.submit_fix_task(arguments)
        self.open_fix_tasks[fp] = task_id
        self._red_advance_counts[fp] = 1
        await self._file_info_escalation(task_id, confirmed, suspect)

    async def _file_info_escalation(
        self, task_id: str, confirmed: list[str], suspect: str,
    ) -> None:
        """File a non-blocking L0 INFO escalation announcing a new fix task (B4).

        Mirrors ``harness._file_starvation_info``: a pure operator-signal
        INFO escalation (``severity='info'``, ``level=0``), deduped against
        any already-pending L0 escalation for this fix task id so a crash
        + re-dispatch can't double-file. No-ops gracefully when no
        escalation queue is attached (same log-only degrade as
        :attr:`task_client`).
        """
        if self.escalation_queue is None:
            return

        existing = [
            e
            for e in self.escalation_queue.get_by_task(task_id, status='pending')
            if e.agent_role == 'orchestrator-offline-lane' and e.level == 0
        ]
        if existing:
            logger.debug(
                'offline-lane: pending INFO escalation already exists for '
                'fix task %s — skipping duplicate file',
                task_id,
            )
            return

        from escalation.models import Escalation

        esc = Escalation(
            id=self.escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role='orchestrator-offline-lane',
            severity='info',
            category='risk_identified',
            summary=(
                f'offline-lane: filed fix task {task_id} for '
                f'{len(confirmed)} confirmed-failing test(s)'
            )[:200],
            detail=(
                f'Failing tests: {", ".join(sorted(confirmed))}\n'
                f'Suspect commit range: {suspect}'
            ),
            suggested_action='manual_investigation',
            level=0,
        )
        self.escalation_queue.submit(esc)
        logger.info(
            'offline-lane: filed INFO escalation %s for fix task %s',
            esc.id, task_id,
        )

    async def _update_existing_fix_task(self, fp: str, head: str) -> None:
        """Update an ALREADY-OPEN fix task for a same-fingerprint red advance (B5).

        Appends the new suspect range to the open fix task rather than
        filing a duplicate task or raising a second INFO escalation, then
        bumps the red-advance count and defers to
        :meth:`_maybe_promote_blocker` for the staged L2 promotion check
        (C4).
        """
        task_id = self.open_fix_tasks[fp]
        if self.task_client is not None:
            await self.task_client.append_suspect_range(task_id, self._suspect_range(head))
        self._red_advance_counts[fp] += 1
        await self._maybe_promote_blocker(fp, task_id)

    async def _maybe_promote_blocker(self, fp: str, task_id: str) -> None:
        """Staged L2 ``escalate_blocker`` promotion when a fix task stalls (C4).

        Promotion is idempotent per fingerprint — a fingerprint already in
        :attr:`_promoted_blockers` returns immediately, so subsequent
        same-set advances never re-promote.

        Three arms, checked in order (done-clear and terminal-promote
        precede the count check, since a status query — when
        :attr:`task_client` is wired — is authoritative over the local
        advance count):

        1. DONE: the fix task landed — clear *fp* from :attr:`open_fix_tasks`
           / :attr:`_red_advance_counts` / :attr:`_promoted_blockers` so a
           genuine LATER recurrence of the same failing set files a FRESH
           task instead of being silently swallowed by stale state. No
           blocker filed.
        2. TERMINAL NON-DONE (cancelled/deferred): the fix task can't land —
           promote immediately regardless of the advance count.
        3. N-ADVANCES (B7): once the confirmed-red advance count reaches
           ``config.git.offline_lane_red_advances_before_blocker``, promote.

        Arms 1/2 are skipped when :attr:`task_client` is unwired (no status
        to query) — only the count-based arm 3 applies in that case.
        """
        if fp in self._promoted_blockers:
            return

        if self.task_client is not None:
            status = await self.task_client.get_status(task_id)
            if status == 'done':
                self.open_fix_tasks.pop(fp, None)
                self._red_advance_counts.pop(fp, None)
                self._promoted_blockers.discard(fp)
                return
            if status in _TERMINAL_NON_DONE:
                await self._file_blocker_escalation(task_id, fp)
                self._promoted_blockers.add(fp)
                return

        if self._red_advance_counts[fp] >= self.config.git.offline_lane_red_advances_before_blocker:
            await self._file_blocker_escalation(task_id, fp)
            self._promoted_blockers.add(fp)

    async def _file_blocker_escalation(self, task_id: str, fp: str) -> None:
        """File a born-at-L2 ``escalate_blocker`` for a stalled offline-lane fix task (B7/C4).

        Mirrors the deterministic-runner / merge-worker-supervisor born-at-L2
        pattern: ``severity='critical'`` + ``level=2`` routes straight to a
        human, and the ``orchestrator-`` ``agent_role`` sentinel prefix keeps
        it at level=2 past the escalation server's severity-downgrade gate.
        ``category='dependency_discovered'`` mirrors the scheduler's
        external-dependency escalations (``scheduler.py`` cancelled-dep /
        unresolved-threshold cases) — this fix task IS the (in-process)
        dependency the confirmed-red offline-lane state is blocked on.

        Deduped against any already-pending L2 blocker escalation for this
        task id (crash-safe re-dispatch guard, mirrors
        :meth:`_file_info_escalation`'s L0 dedup) in addition to the
        in-memory ``_promoted_blockers`` idempotency guard in
        :meth:`_maybe_promote_blocker`. No-ops gracefully when no escalation
        queue is attached (same log-only degrade as :attr:`task_client`).
        """
        if self.escalation_queue is None:
            return

        existing = [
            e
            for e in self.escalation_queue.get_by_task(task_id, status='pending')
            if e.agent_role == 'orchestrator-offline-lane' and e.level == 2
        ]
        if existing:
            logger.debug(
                'offline-lane: pending L2 blocker escalation already exists for '
                'fix task %s — skipping duplicate file',
                task_id,
            )
            return

        from escalation.models import Escalation

        esc = Escalation(
            id=self.escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role='orchestrator-offline-lane',
            severity='critical',
            category='dependency_discovered',
            summary=(
                f'offline-lane: fix task {task_id} has not landed — '
                f'confirmed-red offline-deep suite is stalled'
            )[:200],
            detail=(
                f'Fingerprint {fp} has been confirmed red for '
                f'{self._red_advance_counts.get(fp)} advance(s) without the '
                f'fix task landing. Manual investigation is needed to land '
                f'the fix or re-triage it.'
            ),
            suggested_action='manual_investigation',
            level=2,
        )
        self.escalation_queue.submit(esc)
        logger.info(
            'offline-lane: filed L2 blocker escalation %s for stalled fix task %s',
            esc.id, task_id,
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

    # ------------------------------------------------------------------
    # Default injectable seam implementation
    # ------------------------------------------------------------------

    async def _default_run_suite(self, wt_path: Path, head: str, threads: int) -> tuple[int, str]:
        """Default ``suite_runner`` seam — runs the real offline-deep script.

        Cross-project scope boundary: Part A's ``scripts/run-offline-deep.sh``
        and its ``DF_VERIFY_ROLE=offline`` role are not yet on reify ``main``
        (gated at ζ).  This default implementation builds the invocation
        unconditionally; wiring it to a real, present script is ζ's job.

        Modeled on ``DeterministicRunner._default_run_script``
        (``deterministic_runner.py:313``): env overlays ``DF_VERIFY_ROLE``
        onto a full ``os.environ`` copy (never a bespoke subset, so PATH /
        HOME / XDG_RUNTIME_DIR etc. survive) — the offline role's nice/ionice
        footprint lives inside the script itself (Part A), not re-implemented
        here.  ``head`` is accepted for seam-signature/logging symmetry with
        injected runners; the worktree is already checked out to it by the
        caller (:meth:`_run_once`) before this is invoked.
        """
        argv = [_RUN_OFFLINE_DEEP_SCRIPT, f'--test-threads={threads}']
        env = {**os.environ, 'DF_VERIFY_ROLE': 'offline'}
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(wt_path),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        tail = (stdout or b'').decode(errors='replace')[-2000:]
        return proc.returncode or 0, tail

    async def _default_confirmation_run(self, wt: Path, head: str) -> list[str]:
        """Default ``confirmation_runner`` seam — re-runs failures serially, isolated (β3).

        Cross-project scope boundary (same as :meth:`_default_run_suite`):
        the real reify-side confirm-failures invocation — discover which
        tests just failed, re-run them isolated/serial once, report those
        still failing — is ζ's job; this default builds the invocation
        unconditionally.

        ``--test-threads=1`` forces the confirmation re-run serial/isolated
        (a flake filter must never share state across parallel test
        workers). ``--confirm-failed`` is the confirm-failures flag
        distinguishing this from a normal suite run. ``env``/``cwd`` mirror
        :meth:`_default_run_suite` exactly (same ``DF_VERIFY_ROLE=offline``
        overlay onto a full ``os.environ`` copy). ``head`` is accepted for
        seam-signature symmetry with injected runners, same as
        :meth:`_default_run_suite`.

        Parses stdout as newline-separated test IDs; blank/whitespace-only
        lines are skipped so trailing/blank output never becomes a spurious
        "confirmed" entry.
        """
        argv = [_RUN_OFFLINE_DEEP_SCRIPT, '--test-threads=1', '--confirm-failed']
        env = {**os.environ, 'DF_VERIFY_ROLE': 'offline'}
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(wt),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        text = (stdout or b'').decode(errors='replace')
        return [line.strip() for line in text.splitlines() if line.strip()]

    async def _default_run_infra(self, wt_path: Path, head: str, threads: int) -> tuple[int, str]:
        """Default ``infra_runner`` seam — runs reify's H9 host-infra scope (IE1).

        Cross-project scope boundary (same as :meth:`_default_run_suite`):
        reify's ``tests/infra/run_all.sh`` (H9 = reify:4929) is not
        necessarily present/confirmed on every checkout — real wiring is
        gated separately via ``config.git.offline_lane_infra_enabled``
        (task 1959, IE1), exactly like ``offline_lane_enabled`` gates
        :meth:`_default_run_suite`'s ζ script. This default builds the
        invocation unconditionally.

        NO ``--test-threads``: IE-D2 — ``--scope host-infra`` SELF-ACQUIRES
        the H8 Lane-X flock and runs the host-exclusive set serially, so a
        thread count is meaningless and no DF-side flock is taken here.
        ``env``/``cwd`` mirror :meth:`_default_run_suite` exactly (same
        ``DF_VERIFY_ROLE=offline`` overlay onto a full ``os.environ``
        copy). ``head``/``threads`` are accepted only for ``SuiteRunner``
        seam-signature symmetry with the numeric runner; ``threads`` is
        unused.
        """
        argv = [_RUN_ALL_INFRA_SCRIPT, '--scope', _INFRA_SCOPE]
        env = {**os.environ, 'DF_VERIFY_ROLE': 'offline'}
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(wt_path),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        tail = (stdout or b'').decode(errors='replace')[-2000:]
        return proc.returncode or 0, tail

    async def _default_infra_confirmation_run(self, wt: Path, head: str) -> list[str]:
        """Default ``infra_confirmation_runner`` seam — re-runs H9 host-infra (IE1).

        Cross-project scope boundary (same as :meth:`_default_run_infra`).
        Re-invokes reify's H9 with the SAME argv/env/cwd as
        :meth:`_default_run_infra` — the whole host-exclusive set is small
        and self-serializes under its own flock (IE-D2), so re-running the
        full scope is an acceptable flake filter (mirrors
        :meth:`_default_confirmation_run`'s serial re-run, without needing
        a separate "confirm only these" flag since the set is already
        tiny).

        Parses stdout via the module-level :func:`_parse_infra_failures`
        helper, which extracts H9's ``RESULT: FAIL (<name>)`` lines — a
        different wire format from the numeric seam's newline-separated
        test IDs (:meth:`_default_confirmation_run`), since reify H9 emits
        one line per test with an explicit PASS/FAIL verdict rather than
        listing only failures.
        """
        argv = [_RUN_ALL_INFRA_SCRIPT, '--scope', _INFRA_SCOPE]
        env = {**os.environ, 'DF_VERIFY_ROLE': 'offline'}
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(wt),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        text = (stdout or b'').decode(errors='replace')
        return _parse_infra_failures(text)
