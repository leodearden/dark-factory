"""Tests for merge_status Tier-3.5 git-authority resolution.

Isolated from test_server.py because:
- Real-git integration tests need subprocess/worktree fixtures incompatible
  with test_server.py's in-memory stubs.
- Isolating keeps the slower integration test grouped and avoids bloating
  the 3400-line test_server.py.

Shared patterns from the existing test suite:
- Cross-package import guard (mirrors test_server.py lines 30-54)
- _call_merge_status helper (mirrors test_server.py line 2825)
- _stub_git_ops: returns SimpleNamespace with AsyncMock methods for unit tests
- Real-git fixtures (git_repo/_init_repo/orch_config/git_ops) modeled on
  test_workflow_status_on_resume.py:47-80 and test_git_ops.py
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from escalation.queue import EscalationQueue
from escalation.server import create_server

# ---------------------------------------------------------------------------
# Cross-package orchestrator imports.
# Mirrors test_server.py lines 30-54 exactly.
# ---------------------------------------------------------------------------
try:
    from orchestrator.config import GitConfig, OrchestratorConfig  # type: ignore[reportMissingImports]
    from orchestrator.git_ops import GitOps, _run  # type: ignore[reportMissingImports]
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False
    GitConfig: Any = None  # type: ignore[assignment,misc]
    OrchestratorConfig: Any = None  # type: ignore[assignment,misc]
    GitOps: Any = None  # type: ignore[assignment,misc]
    _run: Any = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _call_merge_status(server, **kwargs) -> dict:
    """Invoke the merge_status MCP tool (async tool)."""
    tool = await server.get_tool('merge_status')
    return await tool.fn(**kwargs)


def _stub_git_ops(**overrides) -> types.SimpleNamespace:
    """Return a SimpleNamespace stub for git_ops unit tests.

    Pass keyword args matching method names; each value must be an async
    callable (e.g. AsyncMock).  Unspecified methods default to AsyncMock()
    returning None / False so callers only specify the values they care about.

    Example::

        stub = _stub_git_ops(
            resolve_branch_sha=AsyncMock(return_value='a' * 40),
            is_ancestor=AsyncMock(return_value=True),
        )
        # stub.find_merge_marker is an AsyncMock(return_value=None) by default
    """
    stub = types.SimpleNamespace(
        resolve_branch_sha=AsyncMock(return_value=None),
        is_ancestor=AsyncMock(return_value=False),
        find_merge_marker=AsyncMock(return_value=None),
    )
    for name, fn in overrides.items():
        setattr(stub, name, fn)
    return stub


# ---------------------------------------------------------------------------
# Real-git fixtures — modeled on test_workflow_status_on_resume.py:47-80
# and test_git_ops.py.  Only used by integration tests guarded with
# @pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, ...).
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def orch_config(git_repo: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=git_repo,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


@pytest.fixture
def git_ops(orch_config: OrchestratorConfig) -> GitOps:
    return GitOps(orch_config.git, orch_config.project_root)


# ---------------------------------------------------------------------------
# Unit tests — in-memory stubs, no real git required
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestMergeStatusGitAuthority:
    """Unit tests for the git-authority Tier-3.5 inside merge_status."""

    # ── step-1: is_ancestor live-branch path ─────────────────────────────────

    async def test_live_branch_on_main_returns_done(self, tmp_path: Path) -> None:
        """When tip is not None and is_ancestor(tip, 'main') → state='done'/found_on_main.

        Verifies:
        - state == 'done', kind == 'found_on_main', merge_sha == tip, generation == 1
        - is_ancestor was called with (tip, 'main')
        - find_merge_marker was NOT called (cheaper-common-path ordering: skip
          the find_merge_marker scan when the branch ref is still live)
        """
        tip = 'a' * 40
        rsb = AsyncMock(return_value=tip)
        ia = AsyncMock(return_value=True)
        fmm = AsyncMock(return_value=None)
        stub_git = _stub_git_ops(
            resolve_branch_sha=rsb,
            is_ancestor=ia,
            find_merge_marker=fmm,
        )
        stub_harness = types.SimpleNamespace(
            _merge_worker=None,
            _terminal_retention=None,
            git_ops=stub_git,
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        config = OrchestratorConfig(
            project_root=tmp_path,
            max_concurrent_tasks=1,
            git=GitConfig(main_branch='main', branch_prefix='task/', remote='origin',
                          worktree_dir='.worktrees'),
        )
        # NO event_store — durable tiers miss, so Tier-3.5 must fire
        server = create_server(esc_queue, harness=stub_harness, orch_config=config)

        result = await _call_merge_status(server, task_id='123')

        assert result.get('state') == 'done', f'Expected state=done, got: {result}'
        assert result.get('kind') == 'found_on_main', f'Expected kind=found_on_main, got: {result}'
        assert result.get('merge_sha') == tip, f'Expected merge_sha={tip!r}, got: {result}'
        assert result.get('generation') == 1, f'Expected generation=1, got: {result}'

        # Verify routing: is_ancestor called with (tip, 'main')
        ia.assert_called_once_with(tip, 'main')
        # find_merge_marker must NOT be called when tip is present
        fmm.assert_not_called()

    # ── step-3: deleted-branch find_merge_marker path (canonical 4352 shape) ──

    async def test_deleted_branch_find_merge_marker_returns_done(
        self, tmp_path: Path
    ) -> None:
        """When branch ref is gone (tip=None), find_merge_marker finds the merge commit.

        This is the canonical 4352 lost-record shape: task merged to main,
        branch+worktree deleted, retention ring/event store record lost —
        but the merge commit is findable on main via git log.

        Verifies:
        - state == 'done', kind == 'found_on_main', merge_sha == marker, generation == 1
        - find_merge_marker was called with full ref 'task/456'
        - is_ancestor is NOT called (tip is None, cheaper-common-path ordering)
        """
        marker = 'b' * 40
        rsb = AsyncMock(return_value=None)   # branch ref gone
        ia = AsyncMock(return_value=False)
        fmm = AsyncMock(return_value=marker)
        stub_git = _stub_git_ops(
            resolve_branch_sha=rsb,
            is_ancestor=ia,
            find_merge_marker=fmm,
        )
        stub_harness = types.SimpleNamespace(
            _merge_worker=None,
            _terminal_retention=None,
            git_ops=stub_git,
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        config = OrchestratorConfig(
            project_root=tmp_path,
            max_concurrent_tasks=1,
            git=GitConfig(main_branch='main', branch_prefix='task/', remote='origin',
                          worktree_dir='.worktrees'),
        )
        server = create_server(esc_queue, harness=stub_harness, orch_config=config)

        result = await _call_merge_status(server, task_id='456')

        assert result.get('state') == 'done', f'Expected state=done, got: {result}'
        assert result.get('kind') == 'found_on_main', f'Expected kind=found_on_main, got: {result}'
        assert result.get('merge_sha') == marker, f'Expected merge_sha={marker!r}, got: {result}'
        assert result.get('generation') == 1, f'Expected generation=1, got: {result}'

        # find_merge_marker called with full ref 'task/456'
        fmm.assert_called_once_with('task/456')
        # is_ancestor must NOT be called when tip is None
        ia.assert_not_called()
