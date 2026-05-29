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


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.scheduler.get_tasks = AsyncMock(return_value=[{'id': 100}, {'id': 101}])

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
        """Empty task list (transient DB failure) aborts the sweep entirely."""
        _mk(harness.git_ops.worktree_base, '500')
        harness.scheduler.get_tasks = AsyncMock(return_value=[])

        await harness._reap_orphan_worktrees()

        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.quarantine_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.prune_worktrees.assert_not_called()  # type: ignore[attr-defined]

    async def test_disabled_flag_noop(self, harness: Harness):
        """Flag off → method returns before touching the DB or filesystem."""
        _mk(harness.git_ops.worktree_base, '500')
        harness.config.worktree_orphan_reaper_enabled = False

        await harness._reap_orphan_worktrees()

        harness.scheduler.get_tasks.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.cleanup_worktree.assert_not_called()  # type: ignore[attr-defined]
        harness.git_ops.prune_worktrees.assert_not_called()  # type: ignore[attr-defined]

    async def test_prune_invoked_once(self, harness: Harness):
        """A completed sweep runs ``git worktree prune`` exactly once."""
        _mk(harness.git_ops.worktree_base, '500')
        await harness._reap_orphan_worktrees()
        harness.git_ops.prune_worktrees.assert_called_once()  # type: ignore[attr-defined]
