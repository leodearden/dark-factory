"""The worktrees a drift check's local leg can build, as git records them.

A warm leg verifies in a ``_driftcheck-<hex>`` tree and a cold one in a
``_merge-<hex>`` throwaway
(orchestrator/src/orchestrator/merge_lane/liveness.py::drift_check_lane).
"""
from __future__ import annotations

from pathlib import Path

from _worktree_registrations import registered_worktree_paths

from orchestrator.git_ops import WorktreeKind

DRIFT_LANE_PREFIXES = (WorktreeKind.DRIFT_CHECK.value, '_merge-')


def registered_drift_lanes(repo: Path) -> list[str]:
    """Names of every drift-leg worktree git still lists for *repo*, sorted."""
    names = (Path(p).name for p in registered_worktree_paths(repo))
    return sorted(name for name in names if name.startswith(DRIFT_LANE_PREFIXES))
