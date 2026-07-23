"""Tests for the durable-record warm-lane reclaim + stale census (leaf γ, task 2891).

The durable-record complement to ``Harness._reconcile_terminal_lanes`` (which
enumerates only the IN-MEMORY ``pool.assignments_snapshot()``):

* ``Harness._reclaim_terminal_lane_records`` — a periodic pass (riding
  ``_run_warm_lane_gc_pass``) that releases warm lanes whose DURABLE
  ``.lane-state/<lane>.json`` record is assigned to a TERMINAL (done/cancelled)
  task and is not live-dispatched, gated on the task branch ref still
  resolving.  NON-terminal (pending/in-progress/blocked) lanes are NEVER
  released — the WIP-preserving invariant.

* ``Harness._stale_lane_assignment_census`` — a digest census of the
  NON-terminal durable assignments the reclaim pass deliberately leaves alone
  but which are idle past ``lane_stale_report_days``.

step-5  RED — reclaim releases done/cancelled lanes only (+ dispatch/fail-safe).
step-7  RED — branch-ref-resolve gate.
step-11 RED — census census lines + fail-safe.
step-13 RED — _maybe_write_digest populates DigestInputs.stale_lane_census.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _workflow_helpers import _build_harness

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import GitOps
from orchestrator.lane_lifecycle import LaneRecord
from orchestrator.lane_lifecycle import LaneState as DurableLaneState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_record(
    git_ops: GitOps,
    lane_name: str,
    state: DurableLaneState,
    task_id: str | None,
    *,
    updated_at: str | None = None,
) -> None:
    """Write a durable ``.lane-state/<lane>.json`` record for a synthetic lane.

    Writes the JSON directly (rather than via ``LaneLifecycle.transition``) so
    the test controls ``updated_at`` for the staleness-age census cases.
    """
    state_dir = git_ops._lane_lifecycle.state_dir
    state_dir.mkdir(parents=True, exist_ok=True)
    rec = LaneRecord(
        state=state,
        task_id=task_id,
        branch=f'task/{task_id}' if task_id else None,
        updated_at=(
            updated_at if updated_at is not None else datetime.now(UTC).isoformat()
        ),
    )
    (state_dir / f'{lane_name}.json').write_text(rec.to_json())


def _make_pool_harness(
    tmp_path: Path, *, lane_stale_report_days: float = 7.0
):
    """Harness with a real GitOps warm-lane pool (non-None) + spy collaborators.

    ``release_warm_lane``/``resolve_branch_sha`` are spied as AsyncMocks and the
    scheduler/event_store are stubbed so the pass runs on the real durable
    ``LaneLifecycle`` records without touching git or the scheduler backend.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / '.git').mkdir()  # bare minimum to satisfy path checks
    config = OrchestratorConfig(
        project_root=repo,
        max_concurrent_tasks=1,
        git=GitConfig(warm_lane_pool=True),
        lane_stale_report_days=lane_stale_report_days,
    )
    git_ops = GitOps(config.git, repo, warm_lane_pool_size=1)
    harness = _build_harness(config)
    harness.git_ops = git_ops
    harness.event_store = MagicMock()
    assert git_ops.warm_lane_pool is not None
    return harness, git_ops


def _reaped_events(harness) -> list:
    """worktree_reaped emit calls recorded on the spy event_store."""
    return [
        c
        for c in harness.event_store.emit.call_args_list
        if c.args and c.args[0] == EventType.worktree_reaped
    ]


# ---------------------------------------------------------------------------
# step-5 / step-7: _reclaim_terminal_lane_records
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReclaimTerminalLaneRecords:
    """Durable-record reclaim releases terminal-task lanes, never live ones."""

    async def test_releases_only_terminal_task_lanes(self, tmp_path: Path) -> None:
        """done + cancelled lanes released; blocked + pending NEVER (WIP-preserving)."""
        harness, git_ops = _make_pool_harness(tmp_path)
        _write_record(git_ops, '_lane-done', DurableLaneState.ASSIGNED, '100')
        _write_record(git_ops, '_lane-blocked', DurableLaneState.ASSIGNED, '200')
        _write_record(git_ops, '_lane-pending', DurableLaneState.ASSIGNED, '300')
        _write_record(git_ops, '_lane-cancelled', DurableLaneState.IN_USE, '400')

        harness.scheduler.get_statuses = AsyncMock(
            return_value=(
                {'100': 'done', '200': 'blocked', '300': 'pending', '400': 'cancelled'},
                None,
            )
        )
        harness.scheduler._dispatched = set()
        git_ops.release_warm_lane = AsyncMock()
        git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeefcafe')

        released = await harness._reclaim_terminal_lane_records()

        # (d) return value == count released
        assert released == 2
        # (a) released exactly for done + cancelled, path == worktree_base/<lane>,
        #     bare task_id
        git_ops.release_warm_lane.assert_any_call(
            git_ops.worktree_base / '_lane-done', '100'
        )
        git_ops.release_warm_lane.assert_any_call(
            git_ops.worktree_base / '_lane-cancelled', '400'
        )
        assert git_ops.release_warm_lane.await_count == 2
        # (b) blocked + pending never released
        released_paths = [
            call.args[0] for call in git_ops.release_warm_lane.await_args_list
        ]
        assert git_ops.worktree_base / '_lane-blocked' not in released_paths
        assert git_ops.worktree_base / '_lane-pending' not in released_paths
        # (f) worktree_reaped emitted per release with warm_lane_retained=True
        reaped = _reaped_events(harness)
        assert len(reaped) == 2
        for call in reaped:
            assert call.kwargs['data']['warm_lane_retained'] is True

    async def test_dispatched_terminal_lane_skipped(self, tmp_path: Path) -> None:
        """(c) A terminal task that IS is_dispatched is skipped (live-acquire guard)."""
        harness, git_ops = _make_pool_harness(tmp_path)
        _write_record(git_ops, '_lane-done', DurableLaneState.ASSIGNED, '100')

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'100': 'done'}, None)
        )
        harness.scheduler._dispatched = {'100'}  # live-acquire guard trips
        git_ops.release_warm_lane = AsyncMock()
        git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeefcafe')

        released = await harness._reclaim_terminal_lane_records()

        assert released == 0
        git_ops.release_warm_lane.assert_not_awaited()
        assert _reaped_events(harness) == []

    async def test_resolver_failed_err_aborts_pass(self, tmp_path: Path) -> None:
        """(e) get_statuses error -> no release, returns 0 (never mass-free)."""
        harness, git_ops = _make_pool_harness(tmp_path)
        _write_record(git_ops, '_lane-done', DurableLaneState.ASSIGNED, '100')

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({}, RuntimeError('db down'))
        )
        harness.scheduler._dispatched = set()
        git_ops.release_warm_lane = AsyncMock()
        git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeefcafe')

        released = await harness._reclaim_terminal_lane_records()

        assert released == 0
        git_ops.release_warm_lane.assert_not_awaited()

    async def test_resolver_failed_empty_aborts_pass(self, tmp_path: Path) -> None:
        """(e) get_statuses empty dict -> no release, returns 0 (fail-safe-wait)."""
        harness, git_ops = _make_pool_harness(tmp_path)
        _write_record(git_ops, '_lane-done', DurableLaneState.ASSIGNED, '100')

        harness.scheduler.get_statuses = AsyncMock(return_value=({}, None))
        harness.scheduler._dispatched = set()
        git_ops.release_warm_lane = AsyncMock()
        git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeefcafe')

        released = await harness._reclaim_terminal_lane_records()

        assert released == 0
        git_ops.release_warm_lane.assert_not_awaited()

    async def test_no_pool_returns_zero(self, tmp_path: Path) -> None:
        """Guard: no warm-lane pool -> returns 0, no scheduler read."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        (repo / '.git').mkdir()
        config = OrchestratorConfig(project_root=repo, max_concurrent_tasks=1)
        git_ops = GitOps(config.git, repo, warm_lane_pool_size=0)
        harness = _build_harness(config)
        harness.git_ops = git_ops
        assert git_ops.warm_lane_pool is None

        released = await harness._reclaim_terminal_lane_records()
        assert released == 0

    async def test_branch_ref_unresolved_skips_release(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Branch ref no longer resolves -> skip + WARN, never released.

        Conservative assert-before-release: a terminal task whose branch has
        already vanished is left for the in-memory reconciler / next acquire,
        never silently destroyed.
        """
        harness, git_ops = _make_pool_harness(tmp_path)
        _write_record(git_ops, '_lane-done', DurableLaneState.ASSIGNED, '100')

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'100': 'done'}, None)
        )
        harness.scheduler._dispatched = set()
        git_ops.release_warm_lane = AsyncMock()
        git_ops.resolve_branch_sha = AsyncMock(return_value=None)  # branch vanished

        with caplog.at_level('WARNING'):
            released = await harness._reclaim_terminal_lane_records()

        assert released == 0
        git_ops.release_warm_lane.assert_not_awaited()
        assert _reaped_events(harness) == []
        assert any(
            '100' in rec.message or '_lane-done' in rec.message
            for rec in caplog.records
        ), 'a WARNING naming the lane/task must be logged on the skip'

    async def test_branch_ref_resolved_releases_after_gate(
        self, tmp_path: Path
    ) -> None:
        """Branch ref resolves -> release proceeds; gate awaited with prefixed branch."""
        harness, git_ops = _make_pool_harness(tmp_path)
        _write_record(git_ops, '_lane-done', DurableLaneState.ASSIGNED, '100')

        harness.scheduler.get_statuses = AsyncMock(
            return_value=({'100': 'done'}, None)
        )
        harness.scheduler._dispatched = set()
        git_ops.release_warm_lane = AsyncMock()
        git_ops.resolve_branch_sha = AsyncMock(return_value='deadbeefcafe')

        released = await harness._reclaim_terminal_lane_records()

        assert released == 1
        git_ops.release_warm_lane.assert_awaited_once_with(
            git_ops.worktree_base / '_lane-done', '100'
        )
        # gate awaited with f'{branch_prefix}{task_id}' (branch_prefix default 'task/')
        git_ops.resolve_branch_sha.assert_awaited_once_with('task/100')
