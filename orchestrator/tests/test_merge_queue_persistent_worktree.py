"""Tests for the persistent warm merge-verify worktree feature (task 1692).

All tests in this file relate to PRD κ Phase 1 of reify warmer-builds-merge-verify.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    tmp_path: Path,
    *,
    persistent: bool = False,
    safety_valve: int = 0,
    push_after_advance: bool = False,
) -> OrchestratorConfig:
    """Build a minimal OrchestratorConfig with the given persistent-worktree knobs."""
    git = GitConfig(
        push_after_advance=push_after_advance,
        persistent_merge_worktree=persistent,
        persistent_merge_worktree_safety_valve_every_n=safety_valve,
    )
    return OrchestratorConfig(project_root=tmp_path, git=git)


# ---------------------------------------------------------------------------
# Step 11 — enforce_persistent_worktree_serial_lane startup guard
# ---------------------------------------------------------------------------


class TestEnforcePersistentWorktreeSerialLane:
    """enforce_persistent_worktree_serial_lane fail-closed startup guard.

    Step 11 (RED): function/exception absent today.
    """

    def test_knob_on_bound_gt1_raises(self, tmp_path: Path):
        """persistent_merge_worktree=True + merge_ahead_bound=2 → raises PersistentWorktreeConfigError."""
        from orchestrator.merge_queue import (  # noqa: PLC0415
            PersistentWorktreeConfigError,
            enforce_persistent_worktree_serial_lane,
        )

        cfg = _make_config(tmp_path, persistent=True)
        with pytest.raises(PersistentWorktreeConfigError) as exc_info:
            enforce_persistent_worktree_serial_lane(cfg, merge_ahead_bound=2)

        msg = str(exc_info.value)
        # Message must name the bound so the operator knows what to change
        assert '2' in msg, (
            f'PersistentWorktreeConfigError must mention the bad bound (2); got: {msg!r}'
        )

    def test_knob_on_bound_1_no_raise(self, tmp_path: Path):
        """persistent_merge_worktree=True + merge_ahead_bound=1 → no raise."""
        from orchestrator.merge_queue import (  # noqa: PLC0415
            enforce_persistent_worktree_serial_lane,
        )

        cfg = _make_config(tmp_path, persistent=True)
        result = enforce_persistent_worktree_serial_lane(cfg, merge_ahead_bound=1)
        # Returns None (no return value needed)
        assert result is None

    def test_knob_off_bound_gt1_no_raise(self, tmp_path: Path):
        """persistent_merge_worktree=False + merge_ahead_bound=2 → guard inert."""
        from orchestrator.merge_queue import (  # noqa: PLC0415
            enforce_persistent_worktree_serial_lane,
        )

        cfg = _make_config(tmp_path, persistent=False)
        # Must not raise even with a large bound
        result = enforce_persistent_worktree_serial_lane(cfg, merge_ahead_bound=2)
        assert result is None
