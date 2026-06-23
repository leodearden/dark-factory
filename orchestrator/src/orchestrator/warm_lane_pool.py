"""Warm-lane pool — pure in-memory FREE/ASSIGNED state machine.

No git I/O.  Git-touching lifecycle methods (seed, acquire, reset, release)
live on GitOps.  This module holds only the concurrency-safe state machine so
the lock-based no-duplicate guarantee is unit-testable without a git repo.

PRD ζ, D9: pool size N = max_concurrent_tasks, wired by Harness at startup.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from pathlib import Path


class LaneState(Enum):
    FREE = 'free'
    ASSIGNED = 'assigned'


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

    # ── Public properties ──────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of lanes in the pool (0 → always exhausted)."""
        return len(self._lanes)

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

    async def acquire_for(self, branch_name: str) -> tuple[Path, bool] | None:
        """Acquire a lane for *branch_name*, reusing the existing one if mapped.

        Returns:
            ``(lane_path, reused)`` where *reused* is True when the pool
            already had *branch_name* mapped to a lane (live-requeue path),
            or False for a fresh allocation.  Returns ``None`` if *branch_name*
            has no existing mapping AND no FREE lane is available (exhaustion).

        Thread/coroutine-safe: runs under the pool lock.
        """
        async with self._lock:
            # Reuse: if this branch is already mapped, return the same lane.
            if branch_name in self._assignments:
                return self._assignments[branch_name], True
            # Fresh: find the first FREE lane.
            for lane, state in self._lanes.items():
                if state == LaneState.FREE:
                    self._lanes[lane] = LaneState.ASSIGNED
                    self._assignments[branch_name] = lane
                    return lane, False
            # Exhausted
            return None

    # ── Read-only helpers ──────────────────────────────────────────────────

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

    def note_assignment(self, branch_name: str, lane: Path) -> None:
        """Record *branch_name* → *lane* in the assignment map.

        Used by the on-disk backstop in ``acquire_warm_lane`` to restore the
        in-memory map after a process restart (when *lane* is discovered on
        disk carrying *branch_name*'s plan.json).  Does NOT change lane state —
        the caller must ensure the lane is ASSIGNED before calling.
        """
        self._assignments[branch_name] = lane

    def drop_assignment(self, branch_name: str) -> None:
        """Remove the *branch_name* assignment without changing lane state.

        Used when an identity guard rejects a reuse candidate: the lane stays
        ASSIGNED (the caller resets it in-place), but the stale assignment is
        cleared so the pool no longer considers this branch mapped to the lane.
        Idempotent: silently ignored if *branch_name* has no assignment.
        """
        self._assignments.pop(branch_name, None)

    def restore_assignment(self, branch_name: str, lane: Path) -> None:
        """Restore *branch_name* → *lane* and mark the lane ASSIGNED.

        Used by the crash-recovery startup path to rebuild the in-memory
        assignment map after a process restart.  Unlike ``note_assignment``
        (which leaves lane state to the caller), this method atomically sets
        *both* the assignment map AND the lane state to ASSIGNED so that a
        concurrent fresh dispatch's ``try_acquire``/``acquire_for`` cannot grab
        the lane before the original task is re-dispatched.

        Startup is single-threaded, so no lock is needed (mirrors
        ``note_assignment``/``drop_assignment``).

        Unknown *lane* path → no-op (never raises).
        """
        matched = self._match_lane(lane)
        if matched is None:
            # Unknown lane path — silently ignore (idempotent).
            return
        self._lanes[matched] = LaneState.ASSIGNED
        self._assignments[branch_name] = matched
