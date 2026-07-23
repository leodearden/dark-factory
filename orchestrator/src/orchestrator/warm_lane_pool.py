"""Warm-lane pool — pure in-memory FREE/ASSIGNED state machine.

No git I/O.  Git-touching lifecycle methods (seed, acquire, reset, release)
live on GitOps.  This module holds only the concurrency-safe state machine so
the lock-based no-duplicate guarantee is unit-testable without a git repo.

PRD ζ, D9: pool size N = max_concurrent_tasks, wired by Harness at startup.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator.lane_lifecycle import IllegalLaneTransition

if TYPE_CHECKING:
    from orchestrator.lane_lifecycle import LaneLifecycle

logger = logging.getLogger(__name__)


class LaneState(Enum):
    FREE = 'free'
    ASSIGNED = 'assigned'


@dataclass(frozen=True)
class WarmLanePoolCensus:
    """Typed snapshot of warm-lane pool occupancy at one instant.

    Carried on the warm-lane exhaustion path (PRD α / W2a): appended to the
    ``WarmLanePoolExhausted`` message and emitted in the WARNING log at the
    EXHAUSTED return, so an operator sees WHY the pool is full — how many lanes
    are free, held by a dispatched task, pinned by a non-dispatched (stuck)
    task, of unknown dispatch status, or durably quarantined.

    Lives in this git-free module (not ``git_ops``) so the escalation server
    (PRD β) can import it without pulling in git plumbing.  The pure counting
    lives on :meth:`WarmLanePool.census`; ``n_quarantined`` is supplied by the
    caller (GitOps, which reads durable records) rather than computed here, so
    this module stays "pure in-memory, no git I/O".

    Size decomposition invariant (holds by construction in
    :meth:`WarmLanePool.census`)::

        size == n_free + n_assigned_dispatched
                + n_pinned_non_dispatched + n_unknown_dispatch

    ``n_quarantined`` stands OUTSIDE that sum: durable QUARANTINED records are
    not pool members (PRD Open Q5 resolved: include the count regardless).

    ``n_pinned_non_dispatched`` is the contract-fixed field name — the
    delivered_check / user-observable signal keys on it.
    """

    size: int
    n_free: int
    n_assigned_dispatched: int
    n_pinned_non_dispatched: int
    n_unknown_dispatch: int
    n_quarantined: int

    def render(self) -> str:
        """Return the stable single-line ``key=value`` string.

        The ONE format source reused verbatim by both the
        ``WarmLanePoolExhausted`` exception message and the EXHAUSTED-return
        WARNING log (and, later, the PRD β/ε consumers), so the operator-facing
        signal is identical and greppable everywhere.
        """
        return (
            f'size={self.size} '
            f'n_free={self.n_free} '
            f'n_assigned_dispatched={self.n_assigned_dispatched} '
            f'n_pinned_non_dispatched={self.n_pinned_non_dispatched} '
            f'n_unknown_dispatch={self.n_unknown_dispatch} '
            f'n_quarantined={self.n_quarantined}'
        )


class WarmLanePool:
    """Per-host pool of warm task-dispatch lanes.

    Lanes are named ``<worktree_base>/<name_prefix><k>`` for k in range(size).
    State is purely in-memory (one process, one orchestrator, one GitOps).

    All mutating operations (try_acquire, release) run under a single
    asyncio.Lock so concurrent coroutines never receive the same lane.

    Args:
        worktree_base: Directory under which lanes live.
        size: Number of lanes.  0 → pool always exhausted.
        name_prefix: Lane directory-name prefix (default ``_lane-``).
    """

    def __init__(
        self,
        worktree_base: Path,
        size: int,
        name_prefix: str = '_lane-',
        lane_lifecycle: LaneLifecycle | None = None,
    ) -> None:
        self._base = worktree_base
        self._name_prefix = name_prefix
        # Ordered dict preserves insertion order so try_acquire always hands
        # out the lowest-numbered free lane first (stable, deterministic).
        self._lanes: dict[Path, LaneState] = {
            worktree_base / f'{name_prefix}{k}': LaneState.FREE
            for k in range(size)
        }
        # branch_name -> lane_path assignment map (for live-requeue + disk backstop)
        self._assignments: dict[str, Path] = {}
        self._lock = asyncio.Lock()
        # OPTIONAL durable-record mirror (PRD dec.3, I1). None keeps the pool
        # a pure in-memory cache — every existing unit test above stays
        # byte-identical. Wired post-construction via set_lane_lifecycle
        # (GitOps constructs both its LaneLifecycle and this pool, so the
        # shared instance cannot be injected at __init__ time).
        self._lane_lifecycle = lane_lifecycle

    # ── Public properties ──────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of lanes in the pool (0 → always exhausted)."""
        return len(self._lanes)

    def set_lane_lifecycle(self, lane_lifecycle: LaneLifecycle) -> None:
        """Wire the shared durable-record writer post-construction.

        After this call, ``restore_assignment``/``note_assignment`` mirror
        their cache mutation onto the durable ``.lane-state/<lane>.json``
        record (via ``LaneLifecycle.note_assigned``) so the record and the
        in-memory cache never drift (PRD dec.3, I1). Not calling this at all
        leaves the pool a pure in-memory cache, unchanged from before this
        was added.
        """
        self._lane_lifecycle = lane_lifecycle

    # ── Mutating operations ────────────────────────────────────────────────

    async def try_acquire(self) -> Path | None:
        """Return the first FREE lane (ASSIGNED) or None on exhaustion.

        Thread/coroutine-safe: runs under the pool lock so two concurrent
        calls never receive the same lane.
        """
        async with self._lock:
            for lane, state in self._lanes.items():
                if state == LaneState.FREE:
                    self._lanes[lane] = LaneState.ASSIGNED
                    return lane
            return None

    def _match_lane(self, path: Path) -> Path | None:
        """Return the registered lane key matching *path*, or None.

        Uses exact match first (fast path), then resolved-path comparison to
        handle symlinks.  Never raises.

        Callers that need concurrency safety must acquire ``self._lock``
        before calling (the lane dict is not modified here, but read-only
        access from an async context still races if the lock is not held).
        """
        if path in self._lanes:
            return path
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        for known_lane in self._lanes:
            try:
                known_resolved = known_lane.resolve()
            except OSError:
                known_resolved = known_lane
            if known_resolved == resolved:
                return known_lane
        return None

    async def release(self, lane: Path) -> None:
        """Mark *lane* as FREE.

        Idempotent: releasing a FREE lane or an unknown path is a no-op
        (never raises).  Also drops any ``_assignments`` entry whose value
        resolves to *lane* so the assignment map stays coherent.
        """
        async with self._lock:
            matched = self._match_lane(lane)
            if matched is None:
                # Unknown path — silently ignore (idempotent)
                return
            self._lanes[matched] = LaneState.FREE
            # Drop any assignment entry pointing at this lane
            to_drop = [
                br for br, assigned in self._assignments.items()
                if assigned in (matched, lane)
            ]
            for br in to_drop:
                del self._assignments[br]

    async def acquire_for(
        self,
        branch_name: str,
        *,
        title: str | None = None,
        branch: str | None = None,
    ) -> tuple[Path, bool] | None:
        """Acquire a lane for *branch_name*, reusing the existing one if mapped.

        Returns:
            ``(lane_path, reused)`` where *reused* is True when the pool
            already had *branch_name* mapped to a lane (live-requeue path),
            or False for a fresh allocation.  Returns ``None`` if *branch_name*
            has no existing mapping AND no FREE lane is available (exhaustion).

        On a FRESH allocation (``reused is False``) and when a ``LaneLifecycle``
        is wired, the durable ASSIGNED record is written through at the SAME
        moment the in-memory state flips (single-writer coherence, PRD
        warm-lane-exhaustion-hardening W2b I1 / boundary row 1). The REUSE
        branch does NOT re-write: the record is already ``ASSIGNED:branch_name``
        (``note_assigned`` would no-op anyway). ``title``/``branch`` are
        carry-forward hints threaded from GitOps so the durable record keeps
        its task_id/title/branch (used by the GitOps acquire consolidation).

        Thread/coroutine-safe: the in-memory mutation runs under the pool lock;
        the durable write-through is issued AFTER the lock is released. There is
        NO ``await`` between the in-memory flip and the synchronous durable
        write, so on the single asyncio loop no other coroutine can interleave
        (same coherence rationale as ``note_assignment``/``restore_assignment``),
        and the blocking file I/O never runs inside the lock.
        """
        async with self._lock:
            # Reuse: if this branch is already mapped, return the same lane.
            if branch_name in self._assignments:
                return self._assignments[branch_name], True
            # Fresh: find the first FREE lane.
            fresh_lane: Path | None = None
            for lane, state in self._lanes.items():
                if state == LaneState.FREE:
                    self._lanes[lane] = LaneState.ASSIGNED
                    self._assignments[branch_name] = lane
                    fresh_lane = lane
                    break
            if fresh_lane is None:
                # Exhausted
                return None
        # Durable write-through for the FRESH allocation, outside the lock.
        self._note_assigned_durable(branch_name, fresh_lane, title=title, branch=branch)
        return fresh_lane, False

    async def reclaim_victim(
        self,
        branch_name: str,
        candidates: set[str],
        is_dispatched: Callable[[str], bool],
    ) -> tuple[str, Path] | None:
        """Steal the oldest eligible ASSIGNED lane for *branch_name*.

        Used by the reclaim-on-exhaustion safety valve (task 1933): when
        ``acquire_for`` returns None (pool exhausted), the caller can attempt
        to reclaim a non-dispatched non-terminal lane rather than returning
        EXHAUSTED immediately.

        Victim selection (under ``self._lock``, no ``await`` in critical section):
        - Iterates ``_assignments`` in insertion order (oldest-first).
        - Skips ``victim == branch_name`` (never steal from self).
        - Skips victims not in *candidates* (non-terminal set provided by the
          async candidate provider; computed BEFORE acquiring the lock).
        - Re-checks ``is_dispatched(victim)`` synchronously UNDER the lock to
          close the TOCTOU window where a candidate was re-dispatched during
          the async ``get_statuses`` call that populated *candidates*.
        - Accepts only lanes in ``LaneState.ASSIGNED``.

        On success: atomically removes ``_assignments[victim]``, sets
        ``_assignments[branch_name] = lane``, leaves lane ``ASSIGNED``, and
        returns ``(victim, lane)``.  On failure (no eligible victim): ``None``.

        Mirrors ``acquire_for``'s atomic ASSIGNED + map idiom.
        """
        async with self._lock:
            for victim, lane in list(self._assignments.items()):
                if victim == branch_name:
                    continue
                if victim not in candidates:
                    continue
                # Re-check dispatched synchronously under lock (TOCTOU guard).
                if is_dispatched(victim):
                    continue
                if self._lanes.get(lane) != LaneState.ASSIGNED:
                    continue
                # Atomically re-key: victim → branch_name, keep ASSIGNED.
                del self._assignments[victim]
                self._assignments[branch_name] = lane
                self._lanes[lane] = LaneState.ASSIGNED  # explicit, mirrors acquire_for
                return (victim, lane)
            return None

    # ── Read-only helpers ──────────────────────────────────────────────────

    def lane_paths(self) -> list[Path]:
        """Return every pool lane Path in insertion order (``_lane-0 .. _lane-{N-1}``).

        A public, git-free snapshot of the lane keys used by
        ``GitOps.prewarm_pool`` to iterate EVERY lane — including the
        high-numbered ones a lazy acquire never demands — without reaching
        into the private ``_lanes`` dict.  The list is a fresh copy of the
        dict keys (dict insertion order is stable), so the caller may iterate
        it freely; ``list(...)`` contains no ``await`` point, so no concurrent
        acquire/release coroutine can interleave during the copy (single
        asyncio loop) and no lock is needed.  Returns ``[]`` for a size-0 pool.
        """
        return list(self._lanes.keys())

    def is_lane(self, path: Path) -> bool:
        """Return True if *path* (resolved) is a known pool lane."""
        return self._match_lane(path) is not None

    def state(self, lane: Path) -> LaneState | None:
        """Return the current LaneState for *lane*, or None if unknown."""
        matched = self._match_lane(lane)
        return self._lanes[matched] if matched is not None else None

    def assignment_for(self, branch_name: str) -> Path | None:
        """Return the lane path currently assigned to *branch_name*, or None."""
        return self._assignments.get(branch_name)

    def assignments_snapshot(self) -> dict[str, Path]:
        """Return a shallow copy of the current assignment map.

        Safety is guaranteed by the single-threaded asyncio event loop: the
        ``dict()`` constructor contains no ``await`` point, so no concurrent
        ``acquire_for``/``release`` mutation (which run as coroutines) can
        interleave during the copy.  No explicit lock is needed.

        The result is intentionally decoupled from the live ``_assignments``
        dict so the caller can iterate it safely without holding any lock.
        Momentary staleness is safe: the reconciler that consumes the snapshot
        re-resolves each lane via the primitive (which is idempotent), so stale
        entries are harmless.
        """
        return dict(self._assignments)

    def census(
        self,
        is_dispatched: Callable[[str], bool] | None,
        n_quarantined: int = 0,
    ) -> WarmLanePoolCensus:
        """Classify every lane into a typed :class:`WarmLanePoolCensus`.

        Pure in-memory read over ``_lanes``/``_assignments`` — no ``await``, no
        lock, no git I/O (keeps this module git-free).  The single-threaded
        asyncio loop guarantees no concurrent ``acquire_for``/``release``
        coroutine interleaves during the loop (no ``await`` point), the same
        no-lock safety rationale as :meth:`assignments_snapshot`.

        Args:
            is_dispatched: ``branch -> bool`` predicate distinguishing a lane
                held by a live dispatched task (``n_assigned_dispatched``) from
                one pinned by a non-dispatched/stuck task
                (``n_pinned_non_dispatched``).  ``None`` at unwired construction
                sites (cli/recover/evals, or the reclaim valve off): every
                ASSIGNED lane then honestly reports ``n_unknown_dispatch``
                rather than guessing.
            n_quarantined: Pass-through count of durable QUARANTINED records,
                supplied by the git-aware caller (GitOps); this pure method
                never reads durable state itself.

        Classification per lane:
            FREE                                    -> n_free
            ASSIGNED + is_dispatched is None        -> n_unknown_dispatch
            ASSIGNED + no branch mapping             -> n_unknown_dispatch
            ASSIGNED + is_dispatched(branch) True   -> n_assigned_dispatched
            ASSIGNED + is_dispatched(branch) False  -> n_pinned_non_dispatched

        Invariant by construction:
            ``size == n_free + n_assigned_dispatched
                      + n_pinned_non_dispatched + n_unknown_dispatch``
        (``n_quarantined`` stands apart — QUARANTINED records are not pool
        members.)
        """
        # Invert branch -> lane into lane -> branch.  Each lane maps to at most
        # one branch (acquire_for/reclaim_victim maintain a 1:1 assignment), so
        # no information is lost.
        lane_to_branch: dict[Path, str] = {
            lane: branch for branch, lane in self._assignments.items()
        }
        n_free = 0
        n_assigned_dispatched = 0
        n_pinned_non_dispatched = 0
        n_unknown_dispatch = 0
        for lane, lane_state in self._lanes.items():
            if lane_state == LaneState.FREE:
                n_free += 1
                continue
            # ASSIGNED lane.
            branch = lane_to_branch.get(lane)
            if is_dispatched is None or branch is None:
                n_unknown_dispatch += 1
            elif is_dispatched(branch):
                n_assigned_dispatched += 1
            else:
                n_pinned_non_dispatched += 1
        return WarmLanePoolCensus(
            size=len(self._lanes),
            n_free=n_free,
            n_assigned_dispatched=n_assigned_dispatched,
            n_pinned_non_dispatched=n_pinned_non_dispatched,
            n_unknown_dispatch=n_unknown_dispatch,
            n_quarantined=n_quarantined,
        )

    def note_assignment(self, task_id: str, lane: Path) -> None:
        """Record *task_id* → *lane* in the assignment map.

        Used by the on-disk backstop in ``acquire_warm_lane`` to restore the
        in-memory map after a process restart (when *lane* is discovered on
        disk carrying *task_id*'s plan.json).  Does NOT change lane state —
        the caller must ensure the lane is ASSIGNED before calling.

        *task_id* is always a bare task id (e.g. ``'42'``), never a real
        ``task/<id>`` branch string — see ``_note_assigned_durable``.

        When a ``LaneLifecycle`` has been wired via ``set_lane_lifecycle``,
        also mirrors this assignment onto the durable record (best-effort;
        see ``_note_assigned_durable``). No-op when unwired.
        """
        self._assignments[task_id] = lane
        self._note_assigned_durable(task_id, lane)

    def drop_assignment(self, branch_name: str) -> None:
        """Remove the *branch_name* assignment without changing lane state.

        Used when an identity guard rejects a reuse candidate: the lane stays
        ASSIGNED (the caller resets it in-place), but the stale assignment is
        cleared so the pool no longer considers this branch mapped to the lane.
        Idempotent: silently ignored if *branch_name* has no assignment.
        """
        self._assignments.pop(branch_name, None)

    def restore_assignment(self, task_id: str, lane: Path) -> None:
        """Restore *task_id* → *lane* and mark the lane ASSIGNED.

        Used by the crash-recovery startup path to rebuild the in-memory
        assignment map after a process restart.  Unlike ``note_assignment``
        (which leaves lane state to the caller), this method atomically sets
        *both* the assignment map AND the lane state to ASSIGNED so that a
        concurrent fresh dispatch's ``try_acquire``/``acquire_for`` cannot grab
        the lane before the original task is re-dispatched.

        Startup is single-threaded, so no lock is needed (mirrors
        ``note_assignment``/``drop_assignment``).

        Unknown *lane* path → no-op (never raises).

        *task_id* is always a bare task id (e.g. ``'42'``), never a real
        ``task/<id>`` branch string — see ``_note_assigned_durable``.

        When a ``LaneLifecycle`` has been wired via ``set_lane_lifecycle``,
        also mirrors this assignment onto the durable record (best-effort;
        see ``_note_assigned_durable``). No-op when unwired.
        """
        matched = self._match_lane(lane)
        if matched is None:
            # Unknown lane path — silently ignore (idempotent).
            return
        self._lanes[matched] = LaneState.ASSIGNED
        self._assignments[task_id] = matched
        self._note_assigned_durable(task_id, matched)

    def _note_assigned_durable(
        self,
        task_id: str,
        lane: Path,
        *,
        title: str | None = None,
        branch: str | None = None,
    ) -> None:
        """Best-effort durable-record mirror for the single-writer ASSIGNED
        write (PRD dec.3, I1: record ↔ cache never drift).

        Called from ``acquire_for`` (fresh alloc), ``reclaim_victim`` (thief),
        ``restore_assignment`` and ``note_assignment`` — the pool is the SOLE
        writer of the durable ASSIGNED record (PRD warm-lane-exhaustion-
        hardening W2b I2).

        No-op when no ``LaneLifecycle`` is wired (``self._lane_lifecycle is
        None``) — the pool then stays a pure in-memory cache, byte-identical
        to its pre-record-routing behavior.

        *task_id* is forwarded to ``LaneLifecycle.note_assigned`` as
        ``task_id`` unchanged.  Callers that only have a bare task id on hand
        (e.g. GitOps's disk-backstop reuse and Harness's crash-recovery
        restore) pass a bare task id (e.g. ``'42'``, matched against
        plan.json's ``task_id`` field), never a real ``task/<id>`` branch
        string.  ``title``/``branch`` are optional carry-forward hints: when
        supplied (GitOps threads ``expected_title``/``full_branch`` from its
        acquire so the durable record keeps its title/branch) they overwrite
        the corresponding record field; when left ``None`` any value already on
        the durable record is carried forward untouched (``note_assigned``'s
        carry-forward-not-clobber contract). A caller must NOT pass a real
        branch string as *task_id* — it would land in and corrupt the durable
        record's ``task_id`` field.

        Both failure modes below are best-effort: the caller already flipped
        the in-memory cache (acquire/reclaim/crash-recovery), and this mirror
        must never crash or block that decision (I3: acquire/release always
        succeed).
        - ``IllegalLaneTransition``: the durable record conflicts (e.g. the
          lane is durably ``ASSIGNED``/``IN_USE`` for a DIFFERENT task, or
          ``QUARANTINED``) — logged and swallowed rather than stealing the
          record (mirrors ``note_assigned``'s own never-steal contract).
        - ``OSError``: the durable write itself failed (disk full, EACCES,
          read-only mount, ``.lane-state`` unwritable) — logged and swallowed
          so a transient I/O error degrades to cache-only instead of
          propagating out of the caller.
        """
        if self._lane_lifecycle is None:
            return
        try:
            self._lane_lifecycle.note_assigned(
                lane, task_id=task_id, title=title, branch=branch,
            )
        except (IllegalLaneTransition, OSError):
            logger.warning(
                'warm_lane_pool: durable record mirror for lane %s failed '
                'for task %r — cache updated, durable record left as-is',
                lane.name, task_id, exc_info=True,
            )
