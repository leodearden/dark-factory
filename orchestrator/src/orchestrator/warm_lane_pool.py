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

    async def release(self, lane: Path) -> None:
        """Mark *lane* as FREE.

        Idempotent: releasing a FREE lane or an unknown path is a no-op
        (never raises).
        """
        resolved = lane.resolve() if lane.exists() else lane
        async with self._lock:
            # Match by resolved path to handle symlinks
            for known_lane in self._lanes:
                known_resolved = known_lane.resolve() if known_lane.exists() else known_lane
                if known_resolved == resolved or known_lane == lane:
                    self._lanes[known_lane] = LaneState.FREE
                    return
            # Unknown path — silently ignore (idempotent)

    # ── Read-only helpers ──────────────────────────────────────────────────

    def is_lane(self, path: Path) -> bool:
        """Return True if *path* (resolved) is a known pool lane."""
        # Try exact match first (fast path)
        if path in self._lanes:
            return True
        # Resolve both sides to handle symlinks
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
                return True
        return False

    def state(self, lane: Path) -> LaneState | None:
        """Return the current LaneState for *lane*, or None if unknown."""
        if lane in self._lanes:
            return self._lanes[lane]
        # Try resolved-path lookup
        try:
            resolved = lane.resolve()
        except OSError:
            return None
        for known_lane, lane_state in self._lanes.items():
            try:
                known_resolved = known_lane.resolve()
            except OSError:
                known_resolved = known_lane
            if known_resolved == resolved:
                return lane_state
        return None
