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
  4. A second, sweep-style converted site (``prune_stale_merge_worktrees``)
     also routes its post-removal prune through the same guard — coverage
     is not limited to the single ``delete_solo_branch`` call site.
  5. (review-fix) An AST-based regression guard asserts NO bare
     ``['git', 'worktree', 'prune']`` argv literal exists anywhere in
     git_ops.py outside ``_prune_registrations`` itself, so reverting any
     converted site (or adding a new unconverted one) back to a raw,
     unguarded call fails this suite instead of shipping silently.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from orchestrator import git_ops as git_ops_module
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

    async def test_prune_stale_merge_worktrees_refuses_when_storage_absent(
        self, git_ops: GitOps, caplog,
    ):
        """A second, sweep-style converted site distinct from
        delete_solo_branch: prune_stale_merge_worktrees only calls
        _prune_registrations AFTER it has removed something (``if
        removed:``), so this also proves the guard fires from that
        conditional call shape, tagged with its own context — not just from
        an unconditional call site.

        The worktree removal itself (`git worktree remove --force`) is
        unrelated to the pool-storage guard and must still happen; only the
        trailing registration-prune step is gated.
        """
        _configure_pool_storage_absent(git_ops)
        merge_wt = git_ops.worktree_base / '_merge-deadbeef'

        async def _fake_iter_merge_worktrees():
            yield merge_wt, merge_wt

        git_ops._iter_merge_worktrees = _fake_iter_merge_worktrees
        mock_run = AsyncMock(return_value=(0, '', ''))

        with (
            caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'),
            patch('orchestrator.git_ops._run', mock_run),
        ):
            removed = await git_ops.prune_stale_merge_worktrees()

        assert removed == [str(merge_wt)], (
            'expected the fake merge worktree to still be removed — the '
            'pool-storage guard must only gate the trailing prune step'
        )

        prune_calls = [
            c for c in mock_run.await_args_list
            if c.args and c.args[0] == ['git', 'worktree', 'prune']
        ]
        assert prune_calls == []

        refusal_records = [
            r for r in caplog.records
            if 'refusing to run `git worktree prune`' in r.message
        ]
        assert any(
            r.message.startswith('prune_stale_merge_worktrees:')
            for r in refusal_records
        ), (
            f'Expected a prune_stale_merge_worktrees-tagged refusal; got '
            f'{[r.message for r in refusal_records]}'
        )


def _find_raw_prune_argv_lines(source: str) -> list[int]:
    """Return line numbers of bare ``['git', 'worktree', 'prune']`` list
    literals in *source* (AST-based, so docstring/comment mentions of the
    same text — e.g. this module's own module docstring — never match).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    target = ['git', 'worktree', 'prune']
    lines = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        values = [elt.value for elt in node.elts if isinstance(elt, ast.Constant)]
        if values == target:
            lines.append(node.lineno)
    return lines


def test_no_raw_prune_argv_outside_chokepoint() -> None:
    """Regression guard (review-fix): every bare ``['git', 'worktree',
    'prune']`` argv literal in git_ops.py must live inside
    ``GitOps._prune_registrations`` — the single guarded chokepoint
    (gitops-chokepoints PRD, task α).

    The four sweep-style converted sites (``create_worktree`` leftover
    cleanup, ``materialize_member_solo``, ``prune_stale_merge_worktrees``,
    ``reap_interactive_worktrees``) are not each individually exercised
    against the pool-storage-absent guard in this module — doing so would
    require reconstructing each site's own preconditions. This source-level
    guard instead makes the regression the PRD cares about — a future edit
    reverting ANY of the five converted sites (or a brand-new site) back to
    a raw, unguarded ``_run(['git', 'worktree', 'prune'])`` call — fail CI
    directly, regardless of which site regresses.
    """
    source_path = Path(git_ops_module.__file__)
    source = source_path.read_text(encoding='utf-8')
    tree = ast.parse(source)

    chokepoint = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == '_prune_registrations'
    )
    assert chokepoint.end_lineno is not None, (
        'ast.parse(source) always populates end_lineno for a real, fully '
        'parsed function node — None here would mean the AST was built '
        'without source positions'
    )
    chokepoint_lines = range(chokepoint.lineno, chokepoint.end_lineno + 1)

    offenders = [
        lineno for lineno in _find_raw_prune_argv_lines(source)
        if lineno not in chokepoint_lines
    ]
    assert offenders == [], (
        f"found raw ['git', 'worktree', 'prune'] argv literal(s) outside "
        f"_prune_registrations at line(s) {offenders} in {source_path} — "
        f"route through self._prune_registrations(context=...) instead"
    )
