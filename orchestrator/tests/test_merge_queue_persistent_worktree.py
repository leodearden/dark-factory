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


# ---------------------------------------------------------------------------
# Step 13 — _acquire_warm_verify_worktree unit tests
# ---------------------------------------------------------------------------


def _make_stub_git_ops(warm_path: Path) -> MagicMock:
    """Build a stub GitOps with async reset/cleanup methods that record calls."""
    stub = MagicMock()
    stub.reset_persistent_merge_worktree = AsyncMock(return_value=warm_path)
    stub.cleanup_merge_worktree = AsyncMock(return_value=None)
    stub.persistent_merge_worktree_path = warm_path
    return stub


def _make_stub_req(tmp_path: Path, *, persistent: bool) -> MagicMock:
    """Build a stub MergeRequest with config.git.persistent_merge_worktree."""
    cfg = _make_config(tmp_path, persistent=persistent)
    req = MagicMock()
    req.config = cfg
    return req


class TestAcquireWarmVerifyWorktree:
    """Unit tests for _acquire_warm_verify_worktree with stub git_ops.

    Step 13 (RED): helper absent today — ImportError expected.
    """

    @pytest.mark.asyncio
    async def test_knob_off_returns_ephemeral_unchanged(self, tmp_path: Path):
        """Knob OFF: returns merge_wt unchanged, no reset/cleanup calls."""
        from orchestrator.merge_queue import _acquire_warm_verify_worktree  # noqa: PLC0415

        warm_path = tmp_path / '_merge-verify'
        ephemeral = tmp_path / '_merge-abc123'
        stub = _make_stub_git_ops(warm_path)
        req = _make_stub_req(tmp_path, persistent=False)

        result = await _acquire_warm_verify_worktree(
            stub, req, ephemeral, 'sha-abc', safety_valve_due=False
        )

        assert result == ephemeral, 'knob OFF: must return merge_wt unchanged'
        stub.reset_persistent_merge_worktree.assert_not_called()
        stub.cleanup_merge_worktree.assert_not_called()

    @pytest.mark.asyncio
    async def test_knob_on_not_due_swaps_to_warm(self, tmp_path: Path):
        """Knob ON, safety_valve_due=False → resets warm wt, cleans up ephemeral, returns warm path."""
        from orchestrator.merge_queue import _acquire_warm_verify_worktree  # noqa: PLC0415

        warm_path = tmp_path / '_merge-verify'
        ephemeral = tmp_path / '_merge-abc123'
        stub = _make_stub_git_ops(warm_path)
        req = _make_stub_req(tmp_path, persistent=True)

        result = await _acquire_warm_verify_worktree(
            stub, req, ephemeral, 'sha-abc', safety_valve_due=False
        )

        assert result == warm_path, 'knob ON+not due: must return warm path'
        stub.reset_persistent_merge_worktree.assert_awaited_once_with('sha-abc')
        stub.cleanup_merge_worktree.assert_awaited_once_with(ephemeral)

    @pytest.mark.asyncio
    async def test_knob_on_due_returns_ephemeral_unchanged(self, tmp_path: Path):
        """Knob ON, safety_valve_due=True → returns merge_wt unchanged (cold throwaway path)."""
        from orchestrator.merge_queue import _acquire_warm_verify_worktree  # noqa: PLC0415

        warm_path = tmp_path / '_merge-verify'
        ephemeral = tmp_path / '_merge-abc123'
        stub = _make_stub_git_ops(warm_path)
        req = _make_stub_req(tmp_path, persistent=True)

        result = await _acquire_warm_verify_worktree(
            stub, req, ephemeral, 'sha-abc', safety_valve_due=True
        )

        assert result == ephemeral, 'safety_valve_due=True: must return merge_wt unchanged (cold path)'
        stub.reset_persistent_merge_worktree.assert_not_called()
        stub.cleanup_merge_worktree.assert_not_called()

    @pytest.mark.asyncio
    async def test_knob_on_not_due_none_merge_wt_no_cleanup(self, tmp_path: Path):
        """Knob ON, merge_wt=None (edge case) → no cleanup_merge_worktree call, returns warm path."""
        from orchestrator.merge_queue import _acquire_warm_verify_worktree  # noqa: PLC0415

        warm_path = tmp_path / '_merge-verify'
        stub = _make_stub_git_ops(warm_path)
        req = _make_stub_req(tmp_path, persistent=True)

        result = await _acquire_warm_verify_worktree(
            stub, req, None, 'sha-abc', safety_valve_due=False
        )

        assert result == warm_path
        stub.reset_persistent_merge_worktree.assert_awaited_once_with('sha-abc')
        stub.cleanup_merge_worktree.assert_not_called()
