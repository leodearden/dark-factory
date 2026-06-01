"""Tests for SpeculativeMergeWorker.on_merge_landed callback (task 1592).

Verifies that:
  1. A 'done' merge invokes the on_merge_landed callback with
     (task_id, base_sha, advanced_sha) where advanced_sha == outcome.merge_sha.
  2. When on_merge_landed raises, the merge still resolves 'done' (fail-open).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeOutcome, MergeRequest, SpeculativeMergeWorker

# ---------------------------------------------------------------------------
# Fixtures — mirror test_merge_queue.py
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


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
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


async def _make_branch_with_file(
    git_ops: GitOps,
    branch_name: str,
    filename: str,
    content: str,
) -> Path:
    """Create a worktree branch with one committed file and return its path."""
    worktree = (await git_ops.create_worktree(branch_name)).path
    (worktree / filename).write_text(content)
    await git_ops.commit(worktree, f'Add {filename}')
    return worktree


def _mock_verify_pass():
    """Return a mock that makes run_scoped_verification always pass."""
    return AsyncMock(return_value=type('VR', (), {'passed': True, 'summary': ''})())


def _make_request(
    task_id: str,
    branch: str,
    worktree: Path,
    config: OrchestratorConfig,
) -> MergeRequest:
    future: asyncio.Future[MergeOutcome] = asyncio.get_event_loop().create_future()
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_merge_landed_invoked_on_done(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    """on_merge_landed is awaited exactly once with (task_id, base_sha, advanced_sha)."""
    wt = await _make_branch_with_file(git_ops, 'hook-test', 'hook_file.py', 'x = 1\n')

    # Capture main SHA before the merge — this is what item.base_sha will be
    _, pre_merge_main_raw, _ = await _run(
        ['git', 'rev-parse', 'main'], cwd=git_ops.project_root,
    )
    pre_merge_main = pre_merge_main_raw.strip()

    callback = AsyncMock()
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue, on_merge_landed=callback)
    worker_task = asyncio.create_task(worker.run())

    with patch(
        'orchestrator.merge_queue.run_scoped_verification',
        _mock_verify_pass(),
    ):
        req = _make_request('hook-test', 'hook-test', wt, config)
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=60)

    assert outcome.status == 'done', f'Expected done, got: {outcome}'
    assert outcome.merge_sha is not None

    # The callback must have been awaited exactly once
    callback.assert_awaited_once()
    called_task_id, called_base_sha, called_advanced_sha = callback.call_args.args
    assert called_task_id == 'hook-test'
    assert called_base_sha == pre_merge_main, (
        f'base_sha should be pre-merge main tip; got {called_base_sha!r}'
    )
    assert called_advanced_sha == outcome.merge_sha, (
        f'advanced_sha should equal outcome.merge_sha; got {called_advanced_sha!r}'
    )

    await worker.stop()
    await worker_task


@pytest.mark.asyncio
async def test_on_merge_landed_fail_open(
    git_ops: GitOps, config: OrchestratorConfig,
) -> None:
    """When on_merge_landed raises, the merge STILL resolves 'done' (fail-open)."""
    wt = await _make_branch_with_file(git_ops, 'hook-fail', 'hook_fail.py', 'y = 2\n')

    async def _exploding_callback(task_id: str, base_sha: str, head_sha: str) -> None:
        raise RuntimeError('Simulated callback failure')

    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    worker = SpeculativeMergeWorker(git_ops, queue, on_merge_landed=_exploding_callback)
    worker_task = asyncio.create_task(worker.run())

    with patch(
        'orchestrator.merge_queue.run_scoped_verification',
        _mock_verify_pass(),
    ):
        req = _make_request('hook-fail', 'hook-fail', wt, config)
        await queue.put(req)
        outcome = await asyncio.wait_for(req.result, timeout=60)

    # Merge must still succeed despite the callback raising
    assert outcome.status == 'done', (
        f'Merge should not fail because of the callback; got: {outcome}'
    )

    await worker.stop()
    await worker_task
