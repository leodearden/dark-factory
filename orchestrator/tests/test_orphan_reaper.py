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
from _orch_helpers import wire_scheduler_liveness_mock

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
    # Wire real (non-auto-mock) liveness-accessor behaviour (task 2235:
    # harness.py now calls scheduler.is_dispatched() instead of reaching
    # into _dispatched directly) so the reaper's live/orphan routing below
    # exercises real semantics instead of an auto-mocked (always-truthy) stub.
    wire_scheduler_liveness_mock(h.scheduler)
    h.scheduler.get_statuses = AsyncMock(return_value=({'100': 'pending', '101': 'done'}, None))

    base = (tmp_path / '.worktrees').resolve()
    base.mkdir(parents=True, exist_ok=True)
    h.git_ops.worktree_base = base
    # Task 2099: mark pool storage present by default so the new
    # mount-presence guard on _reap_orphan_worktrees does not false-trip
    # across this file's reaper-routing tests, which all assume an
    # already-mounted host — an orthogonal concern to orphan/lane routing.
    # The dedicated storage-absent tests below remove the sentinel
    # explicitly to exercise the guard itself.
    h.git_ops.mark_pool_storage_present()
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


@pytest.mark.asyncio
async def test_reaper_skips_spec_pool_lanes(harness: Harness):
    """_reap_orphan_worktrees must NOT quarantine or reap '_spec-' pool lanes.

    The merge-speculation pool is a SEPARATE WarmLanePool instance
    (name_prefix='_spec-').  A '_spec-N' dir is not '_merge-*', is not a
    member of warm_lane_pool, and its name is not a live task id — so without
    a spec-pool-aware skip it falls through to the orphan branch and is moved
    (quarantine) or removed (cleanup), stranding the spec pool and potentially
    clobbering a lane that is mid-verify if the periodic reaper fires
    concurrently.  This is the exact hazard warm_lane_pool lanes are protected
    from; spec lanes must be protected identically.
    """
    base = harness.git_ops.worktree_base
    # Attach a spec pool (with its '_spec-' prefix) to git_ops; leave the
    # plain warm pool unset to prove the spec skip stands on its own.
    spec_pool = WarmLanePool(worktree_base=base, size=2, name_prefix='_spec-')
    harness.git_ops.warm_lane_pool = None
    harness.git_ops.spec_warm_lane_pool = spec_pool
    spec_lane_path = _mk(base, '_spec-0')
    # Live set does NOT include '_spec-0'.
    harness.scheduler.get_statuses = AsyncMock(
        return_value=({'100': 'pending', '101': 'done'}, None)
    )
    # Simulate unsaved work so the lane WOULD be quarantined without the skip.
    harness.git_ops.worktree_has_unsaved_work = AsyncMock(return_value=True)

    await harness._reap_orphan_worktrees()

    cleanup_paths = [c.args[0] for c in harness.git_ops.cleanup_worktree.call_args_list]  # type: ignore[attr-defined]
    quarantine_paths = [c.args[0] for c in harness.git_ops.quarantine_worktree.call_args_list]  # type: ignore[attr-defined]
    assert spec_lane_path not in cleanup_paths, (
        f'Spec lane {spec_lane_path} must NOT be reaped by the orphan reaper'
    )
    assert spec_lane_path not in quarantine_paths, (
        f'Spec lane {spec_lane_path} must NOT be quarantined by the orphan '
        f'reaper (quarantine_worktree is not pool-aware — it would move the dir)'
    )


# ===========================================================================
# Task 2099: reaper guard on pool-storage-absent (mount-down safety)
# ===========================================================================


@pytest.mark.asyncio
class TestOrphanReaperPoolStorageAbsentGuard:
    """_reap_orphan_worktrees must refuse its ENTIRE sweep — no DB read, no
    cleanup/quarantine, no prune — when pool storage is absent (task 2099).

    Direct regression guard for the Jul-3 incident: an unmounted mountpoint
    dir must never let the reaper (and its prune_worktrees tail) treat every
    mount-resident worktree as an orphan.
    """

    async def test_storage_absent_aborts_entire_sweep(self, harness: Harness):
        from orchestrator.git_ops import POOL_ROOT_SENTINEL

        # The `harness` fixture marks pool storage present by default —
        # remove the sentinel to simulate an unmounted mountpoint with a
        # live, empty (from git's perspective) mount dir.
        (harness.git_ops.worktree_base / POOL_ROOT_SENTINEL).unlink()
        assert not harness.git_ops.pool_storage_present()

        _mk(harness.git_ops.worktree_base, '500')  # plain orphan dir
        pool = WarmLanePool(worktree_base=harness.git_ops.worktree_base, size=1)
        harness.git_ops.warm_lane_pool = pool
        _mk(harness.git_ops.worktree_base, '_lane-0')  # registered-lane-looking dir

        harness._file_pool_storage_absent_escalation = MagicMock()

        await harness._reap_orphan_worktrees()

        harness.scheduler.get_statuses.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.prune_worktrees.assert_not_called()  # type: ignore[attr-defined]
        harness._file_pool_storage_absent_escalation.assert_called_once()  # type: ignore[attr-defined]

    async def test_storage_present_control_reaps_clean_orphan(self, harness: Harness):
        """Regression guard: sentinel present (the fixture default) — the
        existing reap-clean-orphan behavior is unchanged."""
        assert harness.git_ops.pool_storage_present()
        wt = _mk(harness.git_ops.worktree_base, '500')

        await harness._reap_orphan_worktrees()

        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '500')  # type: ignore[attr-defined]
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
class TestOrphanReaperNoPoolConfiguredNoOp:
    """_reap_orphan_worktrees() must run its NORMAL sweep on a pool-less
    default host even though `.pool-root` is absent (step-17 review-fix).

    ``create_worktree`` places COLD worktrees directly at
    ``worktree_base/<branch>`` (git_ops.py:1234), so ``worktree_base.exists()``
    is True on any pool-disabled host that has ever run a task, while
    ``.pool-root`` (whose only writer, ``_seed_warm_lane`` on ``rc == 0``,
    requires a configured pool) is never written. Pre-fix, the guard fired
    the escalation and aborted the ENTIRE sweep — no DB read, no
    cleanup/quarantine, no prune — at every startup on such a host. Gating
    on ``pool_in_use()`` (task 2099 step-16) restores the normal sweep when
    no pool is configured.
    """

    async def test_runs_normal_sweep_when_no_pool_configured(self, harness: Harness):
        from orchestrator.git_ops import POOL_ROOT_SENTINEL

        # The `harness` fixture marks pool storage present by default and
        # attaches no pool — unlink the sentinel to reproduce a pool-less
        # host that has run a cold task (worktree_base exists, .pool-root
        # was never actually written by production code).
        (harness.git_ops.worktree_base / POOL_ROOT_SENTINEL).unlink()
        assert harness.git_ops.warm_lane_pool is None
        assert harness.git_ops.spec_warm_lane_pool is None
        assert not harness.git_ops.pool_storage_present()

        wt = _mk(harness.git_ops.worktree_base, '500')
        harness._file_pool_storage_absent_escalation = MagicMock()

        await harness._reap_orphan_worktrees()

        harness.scheduler.get_statuses.assert_awaited()  # type: ignore[attr-defined]
        harness.git_ops.cleanup_worktree.assert_called_once_with(wt, '500')  # type: ignore[attr-defined]
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness._file_pool_storage_absent_escalation.assert_not_called()  # type: ignore[attr-defined]
