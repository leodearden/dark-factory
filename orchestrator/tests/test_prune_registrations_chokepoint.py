"""Tests for ``GitOps._prune_registrations`` — the single guarded chokepoint
for every ``git worktree prune`` invocation in git_ops.py (gitops-chokepoints
PRD, task α).

Prior to this task, the task-2099 pool-storage guard (see
test_pool_storage_guard.py) only wrapped the ``prune_worktrees()`` harness
entry point; five other call sites in this module shelled out to
``git worktree prune`` directly and were unprotected against the Jul-3
mount-down incident. This module confirms:

  1. A converted call site (``delete_solo_branch``) refuses to run
     ``git worktree prune`` when pool storage is absent, logs a
     context-tagged 'refusing to run `git worktree prune`' warning, and
     fires ``_note_pool_storage_absent``.
  2. When pool storage is present, the same call site runs the prune
     subprocess identically to today (same argv/cwd).
  3. The ``context`` argument threaded through ``_prune_registrations``
     appears verbatim in the emitted log record, for more than one caller.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.warm_lane_pool import WarmLanePool


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, tmp_path: Path) -> GitOps:
    project_root = tmp_path / 'project'
    project_root.mkdir()
    return GitOps(git_config, project_root)


def _configure_pool_storage_absent(git_ops: GitOps) -> None:
    """Simulate an unmounted mountpoint dir with a pool configured: the
    worktree_base dir exists (mount dir present) but the `.pool-root`
    sentinel was never written, and a pool is configured so pool_in_use()
    is True — same setup as test_pool_storage_guard.py's
    TestPruneWorktreesGuard.test_prune_skipped_when_storage_absent.
    """
    git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
    git_ops.warm_lane_pool = WarmLanePool(worktree_base=git_ops.worktree_base, size=1)
    assert not git_ops.pool_storage_present()
    assert git_ops.pool_in_use()


@pytest.mark.asyncio
class TestPruneRegistrationsChokepoint:
    async def test_converted_site_refuses_when_storage_absent(
        self, git_ops: GitOps, caplog,
    ):
        """delete_solo_branch (a converted raw-prune site) refuses to run
        `git worktree prune` when pool storage is absent, tagged with its
        own context, and fires the escalation callback — identical guard
        behaviour to the original prune_worktrees() chokepoint."""
        _configure_pool_storage_absent(git_ops)

        callback = Mock()
        git_ops._on_pool_storage_absent = callback
        mock_run = AsyncMock(return_value=(0, '', ''))

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', mock_run),
        ):
            await git_ops.delete_solo_branch('_solo-b2')

        # The prune subprocess must never have been invoked ...
        prune_calls = [
            c for c in mock_run.await_args_list
            if c.args and c.args[0] == ['git', 'worktree', 'prune']
        ]
        assert prune_calls == []
        callback.assert_called_once()

        refusal_records = [
            r for r in caplog.records
            if 'refusing to run `git worktree prune`' in r.message
        ]
        assert refusal_records, (
            f'Expected a refusal WARNING; got {[r.message for r in caplog.records]}'
        )
        assert any(
            r.message.startswith('delete_solo_branch:') for r in refusal_records
        ), (
            f'Expected the refusal log context-tagged with the caller; got '
            f'{[r.message for r in refusal_records]}'
        )

    async def test_converted_site_runs_when_storage_present(self, git_ops: GitOps):
        """With pool storage present, behaviour is byte-identical to today:
        the prune subprocess still runs with the same argv/cwd."""
        git_ops.mark_pool_storage_present()
        assert git_ops.pool_storage_present()
        mock_run = AsyncMock(return_value=(0, '', ''))

        with patch('orchestrator.git_ops._run', mock_run):
            await git_ops.delete_solo_branch('_solo-b2')

        mock_run.assert_any_await(
            ['git', 'worktree', 'prune'], cwd=git_ops.project_root,
        )

    async def test_context_tag_distinguishes_callers(self, git_ops: GitOps, caplog):
        """The same guard fires from two different call sites — the log
        context tag must reflect which one asked, not a hardcoded name."""
        _configure_pool_storage_absent(git_ops)
        mock_run = AsyncMock(return_value=(0, '', ''))

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', mock_run),
        ):
            await git_ops.delete_solo_branch('_solo-b2')
            await git_ops.prune_worktrees()

        messages = [r.message for r in caplog.records]
        assert any(m.startswith('delete_solo_branch:') for m in messages), messages
        assert any(m.startswith('prune_worktrees:') for m in messages), messages

    async def test_prune_worktrees_default_context_preserved(
        self, git_ops: GitOps, caplog,
    ):
        """prune_worktrees() is now a thin delegate to _prune_registrations
        but existing zero-arg harness callers must see unchanged log text
        (context defaults to 'prune_worktrees')."""
        _configure_pool_storage_absent(git_ops)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            await git_ops.prune_worktrees()

        assert any(
            r.message.startswith('prune_worktrees:') for r in caplog.records
        ), [r.message for r in caplog.records]
