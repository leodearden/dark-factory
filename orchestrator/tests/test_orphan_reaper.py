"""Tests for the orphan-worktree reaper (Fix B).

The reaper sweeps worktrees whose numeric id no longer maps to a live task and
routes each: quarantine (preserve) anything the work-detector flags, reap only
provably-clean dirs.  These tests exercise the reaper's ROUTING, skip rules,
and fail-safes; the git-level work detection / quarantine mechanics live in
``test_git_ops.py``.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.event_store import EventType
from orchestrator.harness import Harness
from orchestrator.warm_lane_pool import WarmLanePool


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.scheduler.get_statuses = AsyncMock(return_value=({'100': 'pending', '101': 'done'}, None))

    base = (tmp_path / '.worktrees').resolve()
    base.mkdir(parents=True, exist_ok=True)
    h.git_ops.worktree_base = base
    # Mock the git-level helpers — their behaviour is covered in test_git_ops.
    h.git_ops.worktree_has_unsaved_work = AsyncMock(return_value=False)
    h.git_ops.quarantine_worktree = AsyncMock(return_value=tmp_path / 'q-dest')
    h.git_ops.cleanup_worktree = AsyncMock()
    h.git_ops.prune_worktrees = AsyncMock()

    h.event_store = MagicMock()
    h.config.worktree_orphan_reaper_enabled = True
    return h


def _mk(base: Path, name: str) -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _emitted_types(harness: Harness) -> list:
    return [c.args[0] for c in harness.event_store.emit.call_args_list]  # type: ignore[attr-defined]


@pytest.mark.asyncio
class TestOrphanReaper:
    async def test_reaps_clean_orphan(self, harness: Harness):
        """An orphan (id not live) with no work is reaped, not quarantined."""
        wt = _mk(harness.git_ops.worktree_base, '500')
        await harness._reap_orphan_worktrees()
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '500')  # type: ignore[attr-defined]
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert EventType.worktree_reaped in _emitted_types(harness)

    async def test_quarantines_orphans_with_work(self, harness: Harness):
        """Orphans the work-detector flags (commits OR dirty WIP) are quarantined."""
        wt_commits = _mk(harness.git_ops.worktree_base, '600')
        wt_dirty = _mk(harness.git_ops.worktree_base, '601')
        harness.git_ops.worktree_has_unsaved_work = AsyncMock(return_value=True)

        await harness._reap_orphan_worktrees()

        quarantined = {
            c.args[0]
            for c in harness.git_ops.quarantine_worktree.call_args_list  # type: ignore[attr-defined]
        }
        assert quarantined == {wt_commits, wt_dirty}
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        assert _emitted_types(harness).count(EventType.worktree_quarantined) == 2

    async def test_skips_reserved_names(self, harness: Harness):
        """``_merge-*`` and ``*-skip-attempt`` are reserved — never touched."""
        _mk(harness.git_ops.worktree_base, '_merge-abc123')
        _mk(harness.git_ops.worktree_base, '700-skip-attempt')
        await harness._reap_orphan_worktrees()
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_skips_live_recovered_preserved_dispatched(self, harness: Harness):
        """Live, recovered, preserved, session, and dispatched ids are skipped."""
        base = harness.git_ops.worktree_base
        _mk(base, '100')  # live id (in get_tasks)
        _mk(base, '200')  # recovered plan
        _mk(base, '300')  # preserved worktree
        _mk(base, '400')  # dispatched
        _mk(base, '450')  # recovered session
        harness._recovered_plans = {'200': {}}
        harness._preserved_worktrees = {'300'}
        harness.scheduler._dispatched = {'400'}
        harness._recovered_sessions = {'450': {}}

        await harness._reap_orphan_worktrees()

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]

    async def test_noop_on_empty_get_tasks(self, harness: Harness):
        """Empty statuses map (transient DB failure) aborts the sweep entirely."""
        _mk(harness.git_ops.worktree_base, '500')
        harness.scheduler.get_statuses = AsyncMock(return_value=({}, None))

        await harness._reap_orphan_worktrees()

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.prune_worktrees.assert_not_called()  # type: ignore[attr-defined]

    async def test_disabled_flag_noop(self, harness: Harness):
        """Flag off → method returns before touching the DB or filesystem."""
        _mk(harness.git_ops.worktree_base, '500')
        harness.config.worktree_orphan_reaper_enabled = False

        await harness._reap_orphan_worktrees()

        harness.scheduler.get_statuses.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.prune_worktrees.assert_not_called()  # type: ignore[attr-defined]

    async def test_prune_invoked_once(self, harness: Harness):
        """A completed sweep runs ``git worktree prune`` exactly once."""
        _mk(harness.git_ops.worktree_base, '500')
        await harness._reap_orphan_worktrees()
        harness.git_ops.prune_worktrees.assert_called_once()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# γ3b step-7 RED: reaper routes to get_statuses; done task's worktree is live
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reaper_uses_get_statuses_and_spares_done_task_worktree(harness: Harness):
    """_reap_orphan_worktrees must call get_statuses (not get_tasks) to determine
    live ids, and must NOT reap the worktree of a done task (γ3b correctness crux).

    Setup:
      - get_statuses returns {500: done, 999: pending} — ALL ids including terminal.
      - get_tasks (old path) would return only [{id: 999}] — 500 absent as done.
      - Worktree '500' exists on disk.

    Correct new behaviour:
      - live_ids derives from get_statuses → {'500', '999'} → worktree '500' is live.
      - cleanup_worktree / quarantine_worktree NOT called for '500'.

    Regression guard — do NOT narrow to active-only:
      - The OLD reaper called get_tasks() with NO statuses filter; that returns ALL
        tasks including done task 500, so done worktrees were already spared.
      - The NEW reaper calls get_statuses() instead — same full id set, ~95% less
        payload.  Routing changes; done-worktree-is-live behaviour must NOT change.
      - This test guards against accidentally introducing an active-only filter
        (e.g. get_tasks(statuses=ACTIVE_TASK_STATUSES)) which WOULD drop done task
        500 from live_ids and wrongly reap its worktree.
    """
    # Simulate the old active-only path that this test guards against regressing to:
    # get_tasks returns only [{id: 999}], omitting the done task 500.
    harness.scheduler.get_tasks = AsyncMock(return_value=[{'id': 999}])
    # Add get_statuses stub — returns full id map incl. done task 500.
    harness.scheduler.get_statuses = AsyncMock(
        return_value=({'500': 'done', '999': 'pending'}, None)
    )

    # Create the worktree for the done task (it must NOT be reaped).
    _mk(harness.git_ops.worktree_base, '500')

    await harness._reap_orphan_worktrees()

    # (a) get_statuses must have been awaited (routing assertion).
    harness.scheduler.get_statuses.assert_awaited()  # type: ignore[attr-defined]
    # (b) The done task's worktree must NOT be cleaned up or quarantined.
    cleanup_paths = [c.args[0] for c in harness.git_ops.cleanup_worktree.call_args_list]  # type: ignore[attr-defined]
    quarantine_paths = [c.args[0] for c in harness.git_ops.quarantine_worktree.call_args_list]  # type: ignore[attr-defined]
    orphan_path = harness.git_ops.worktree_base / '500'
    assert orphan_path not in cleanup_paths, (
        f'Done task worktree {orphan_path} must NOT be reaped (get_statuses treats it as live)'
    )
    assert orphan_path not in quarantine_paths, (
        f'Done task worktree {orphan_path} must NOT be quarantined (get_statuses treats it as live)'
    )


# ===========================================================================
# Step-7 RED: reaper skips warm pool lanes
# ===========================================================================


@pytest.mark.asyncio
async def test_reaper_skips_pool_lanes(harness: Harness):
    """_reap_orphan_worktrees must NOT quarantine or reap pool lanes.

    quarantine_worktree is not pool-aware (it moves the dir), so moving a lane
    would leave the pool's registered lane path dangling.  The reaper must skip
    lanes entirely via the is_lane check — the same guarantee that protects a
    just-recovered lane from being immediately undone by the reaper.

    Setup:
      - worktree_orphan_reaper_enabled = True
      - get_statuses returns a non-empty live set that does NOT include any
        lane dir name ('_lane-0' is not a task id)
      - A WarmLanePool is attached with a '_lane-0' dir on disk
      - worktree_has_unsaved_work returns True (so without the skip the lane
        WOULD be quarantined)

    Assertions:
      - quarantine_worktree was NOT called for the lane
      - cleanup_worktree was NOT called for the lane
    """
    base = harness.git_ops.worktree_base
    # Attach a pool to git_ops
    pool = WarmLanePool(worktree_base=base, size=2)
    harness.git_ops.warm_lane_pool = pool
    # Create the lane dir on disk (so the iterdir loop sees it)
    lane_path = _mk(base, '_lane-0')
    # get_statuses returns a live set that does NOT include '_lane-0'
    harness.scheduler.get_statuses = AsyncMock(
        return_value=({'100': 'pending', '101': 'done'}, None)
    )
    # Simulate unsaved work so the lane WOULD be quarantined without the skip
    harness.git_ops.worktree_has_unsaved_work = AsyncMock(return_value=True)

    await harness._reap_orphan_worktrees()

    cleanup_paths = [c.args[0] for c in harness.git_ops.cleanup_worktree.call_args_list]  # type: ignore[attr-defined]
    quarantine_paths = [c.args[0] for c in harness.git_ops.quarantine_worktree.call_args_list]  # type: ignore[attr-defined]
    assert lane_path not in cleanup_paths, (
        f'Lane {lane_path} must NOT be reaped by the orphan reaper'
    )
    assert lane_path not in quarantine_paths, (
        f'Lane {lane_path} must NOT be quarantined by the orphan reaper '
        f'(quarantine_worktree is not pool-aware — it would move the dir)'
    )
