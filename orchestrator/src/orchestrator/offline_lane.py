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
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING

if TYPE_CHECKING:
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

#: Signature of the injectable heavy-suite seam: (worktree path, head SHA,
#: test-thread count) -> (return code, output tail).
SuiteRunner = Callable[[Path, str, int], Awaitable[tuple[int, str]]]


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
    """

    def __init__(
        self,
        git_ops: 'GitOps',
        config: 'OrchestratorConfig',
        *,
        lock_path: str | Path,
        suite_runner: SuiteRunner | None = None,
    ) -> None:
        self.git_ops = git_ops
        self.config = config
        self.lock_path = Path(lock_path)
        self.suite_runner: SuiteRunner = (
            suite_runner if suite_runner is not None else self._default_run_suite
        )

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
        """
        raise NotImplementedError

    async def _run_once(self) -> None:
        """Snapshot head, reset the warm worktree, and invoke the suite seam.

        See module docstring — always-from-head, clear-before-snapshot.

        Clear-before-snapshot: ``_dirty`` is cleared FIRST, before the
        ``get_main_sha`` snapshot, so no advance is ever lost — an advance
        landing after the clear either lands inside this run's snapshot
        (fine) or re-sets ``_dirty`` for a coalesced re-run (fine).
        Clearing AFTER the snapshot would open a lost-update window.
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
            'offline-lane: run head=%s status=%s duration=%.1fs',
            head[:12], 'PASS' if rc == 0 else 'FAIL', duration,
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
        raise NotImplementedError

    def release_lock(self) -> None:
        """Release a previously-acquired lockfile, if held."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Default injectable seam implementation
    # ------------------------------------------------------------------

    async def _default_run_suite(self, wt_path: Path, head: str, threads: int) -> tuple[int, str]:
        """Default ``suite_runner`` seam — runs the real offline-deep script.

        Cross-project scope boundary: Part A's ``scripts/run-offline-deep.sh``
        and its ``DF_VERIFY_ROLE=offline`` role are not yet on reify ``main``
        (gated at ζ).  This default implementation builds the invocation
        unconditionally; wiring it to a real, present script is ζ's job.
        """
        raise NotImplementedError
