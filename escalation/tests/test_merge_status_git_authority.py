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
    from orchestrator.config import (  # type: ignore[reportMissingImports]
        GitConfig,
        OrchestratorConfig,
    )
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
def orch_config(git_repo: Path) -> OrchestratorConfig:  # type: ignore[reportInvalidTypeForm]
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
def git_ops(orch_config: OrchestratorConfig) -> GitOps:  # type: ignore[reportInvalidTypeForm]
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

    # ── step-5: fire-safety + negatives + guards ──────────────────────────────

    async def _make_server_with_git_ops(
        self, tmp_path: Path, stub_git
    ) -> Any:
        """Helper: create a server with git_ops wired, no event_store."""
        esc_queue = EscalationQueue(tmp_path / 'esc')
        stub_harness = types.SimpleNamespace(
            _merge_worker=None,
            _terminal_retention=None,
            git_ops=stub_git,
        )
        config = OrchestratorConfig(
            project_root=tmp_path,
            max_concurrent_tasks=1,
            git=GitConfig(main_branch='main', branch_prefix='task/', remote='origin',
                          worktree_dir='.worktrees'),
        )
        return create_server(esc_queue, harness=stub_harness, orch_config=config)

    async def test_fire_safe_resolve_branch_sha_raises_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """(a) fire-safe: resolve_branch_sha raising → state='unknown', no raise."""
        rsb = AsyncMock(side_effect=RuntimeError('git exploded'))
        stub_git = _stub_git_ops(resolve_branch_sha=rsb)
        server = await self._make_server_with_git_ops(tmp_path, stub_git)

        result = await _call_merge_status(server, task_id='T')

        assert result.get('state') == 'unknown', f'Expected unknown, got: {result}'
        assert 'hint' in result, f'Expected hint key in unknown response: {result}'

    async def test_fire_safe_is_ancestor_raises_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """(a) fire-safe: is_ancestor raising → state='unknown', no raise."""
        tip = 'c' * 40
        rsb = AsyncMock(return_value=tip)
        ia = AsyncMock(side_effect=RuntimeError('git subprocess died'))
        stub_git = _stub_git_ops(resolve_branch_sha=rsb, is_ancestor=ia)
        server = await self._make_server_with_git_ops(tmp_path, stub_git)

        result = await _call_merge_status(server, task_id='T')

        assert result.get('state') == 'unknown', f'Expected unknown, got: {result}'
        assert 'hint' in result, f'Expected hint key in unknown response: {result}'

    async def test_genuine_miss_branch_present_not_on_main_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """(b) genuine miss: tip present, is_ancestor=False, no marker → unknown.

        A branch that exists but hasn't landed on main yet must NOT
        false-positive as 'done'.
        """
        tip = 'd' * 40
        rsb = AsyncMock(return_value=tip)
        ia = AsyncMock(return_value=False)
        fmm = AsyncMock(return_value=None)
        stub_git = _stub_git_ops(resolve_branch_sha=rsb, is_ancestor=ia,
                                  find_merge_marker=fmm)
        server = await self._make_server_with_git_ops(tmp_path, stub_git)

        result = await _call_merge_status(server, task_id='T-pending')

        assert result.get('state') == 'unknown', (
            f'Branch not on main must stay unknown, got: {result}'
        )
        # find_merge_marker must NOT be called when tip is present (cheaper path)
        fmm.assert_not_called()

    async def test_guard_orch_config_absent_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """(c) guard: git_ops present but no orch_config → straight to honest unknown."""
        stub_git = _stub_git_ops(
            resolve_branch_sha=AsyncMock(return_value='e' * 40),
            is_ancestor=AsyncMock(return_value=True),
        )
        stub_harness = types.SimpleNamespace(
            _merge_worker=None,
            _terminal_retention=None,
            git_ops=stub_git,
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        # No orch_config passed — Tier-3.5 must be skipped
        server = create_server(esc_queue, harness=stub_harness)

        result = await _call_merge_status(server, task_id='T-noconfig')

        assert result.get('state') == 'unknown', (
            f'No orch_config must yield unknown, got: {result}'
        )

    async def test_guard_git_ops_absent_returns_unknown(
        self, tmp_path: Path
    ) -> None:
        """(d) guard: harness without git_ops attr → straight to honest unknown."""
        # Harness with NO git_ops attribute at all
        stub_harness = types.SimpleNamespace(
            _merge_worker=None,
            _terminal_retention=None,
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        config = OrchestratorConfig(
            project_root=tmp_path,
            max_concurrent_tasks=1,
            git=GitConfig(main_branch='main', branch_prefix='task/', remote='origin',
                          worktree_dir='.worktrees'),
        )
        server = create_server(esc_queue, harness=stub_harness, orch_config=config)

        result = await _call_merge_status(server, task_id='T-nogit')

        assert result.get('state') == 'unknown', (
            f'No git_ops must yield unknown, got: {result}'
        )


# ---------------------------------------------------------------------------
# Real-git integration test — the canonical 4352 lost-record shape end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
class TestMergeStatusGitAuthorityIntegration:
    """End-to-end integration test using real git operations.

    Exercises the canonical 4352 shape:
        land branch to main → advance main → delete branch + worktree →
        no ring/event-store record → merge_status must return done/found_on_main.

    Real git subprocess is required to validate the find_merge_marker code
    path that the SimpleNamespace stubs cannot exercise.
    """

    async def test_4352_lost_record_returns_done_found_on_main(
        self, tmp_path: Path, git_ops: GitOps, orch_config: OrchestratorConfig  # type: ignore[reportInvalidTypeForm]
    ) -> None:
        """After merge+delete with no ring/event-store record, merge_status returns done.

        Sequence:
        1. Create worktree for task '770', commit a file.
        2. merge_to_main → advance_main (branch lands on main).
        3. cleanup_merge_worktree + cleanup_worktree (branch+worktree deleted).
        4. Build server with NO event_store, stub_harness with real git_ops.
        5. Call merge_status(task_id='770').
        6. Assert: state='done', kind='found_on_main', merge_sha==result.merge_commit,
           len(merge_sha)==40.
        """
        tid = '770'

        # --- Step 1: create worktree and commit ---
        wt_info = await git_ops.create_worktree(tid)
        assert wt_info is not None
        (wt_info.path / f'{tid}.py').write_text(f'{tid} = True\n')
        await git_ops.commit(wt_info.path, f'Add {tid}')

        # --- Step 2: merge to main ---
        merge_result = await git_ops.merge_to_main(wt_info.path, tid)
        assert merge_result.success, f'merge_to_main failed: {merge_result}'
        assert merge_result.merge_commit is not None
        assert merge_result.merge_worktree is not None

        adv = await git_ops.advance_main(merge_result.merge_commit)
        assert adv == 'advanced'

        expected_sha = merge_result.merge_commit

        # --- Step 3: delete branch + worktree (4352 shape) ---
        await git_ops.cleanup_merge_worktree(merge_result.merge_worktree)
        await git_ops.cleanup_worktree(wt_info.path, tid)

        # Verify branch is gone (confirms the find_merge_marker path will fire)
        rc, sha_out, _ = await _run(
            ['git', 'rev-parse', '--verify', f'task/{tid}'],
            cwd=git_ops.project_root,
        )
        assert rc != 0, f'Branch should be gone (non-zero rc) but rc={rc}, sha={sha_out!r}'

        # --- Step 4: server with NO event_store (ring/event-store miss) ---
        stub_harness = types.SimpleNamespace(
            _merge_worker=None,
            _terminal_retention=None,
            git_ops=git_ops,
        )
        esc_queue = EscalationQueue(tmp_path / 'esc')
        server = create_server(esc_queue, harness=stub_harness, orch_config=orch_config)

        # --- Step 5 & 6: call merge_status and assert ---
        result = await _call_merge_status(server, task_id=tid)

        assert result.get('state') == 'done', (
            f'Expected done/found_on_main after 4352 shape, got: {result}'
        )
        assert result.get('kind') == 'found_on_main', (
            f'Expected kind=found_on_main, got: {result}'
        )
        assert result.get('merge_sha') == expected_sha, (
            f'Expected merge_sha={expected_sha!r}, got: {result.get("merge_sha")!r}'
        )
        assert len(result['merge_sha']) == 40, (
            f'merge_sha must be a 40-char SHA, got: {result["merge_sha"]!r}'
        )
