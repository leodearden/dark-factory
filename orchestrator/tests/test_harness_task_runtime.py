"""Tests for Harness.task_runtime_snapshot() — the thin delegator to
orchestrator.task_runtime.build_task_runtime_snapshot (task 2634, PRD
plans/dashboard-task-runtime-endpoint-prd.md task alpha).

Exercises the delegator via the "unbound method + stub self" trick: rather
than constructing a full ~7000-line Harness, wrap a real (non-pooled)
GitOps in a types.SimpleNamespace duck-typing the two attributes the
delegator reads (git_ops, event_store) and invoke
Harness.task_runtime_snapshot(stub_self) directly.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.harness import Harness
from orchestrator.task_runtime import build_task_runtime_snapshot


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


def _make_task_artifacts(worktree_base: Path, name: str, task_id: str) -> TaskArtifacts:
    """Create <worktree_base>/<name> plus its .task-meta/<name> artifacts root."""
    worktree = worktree_base / name
    worktree.mkdir(parents=True, exist_ok=True)
    meta_root = TaskArtifacts.meta_root_for(worktree_base, name)
    ta = TaskArtifacts(worktree, meta_root)
    ta.init(task_id, f'Task {task_id}', 'desc')
    return ta


class TestHarnessDelegator:
    def test_task_runtime_snapshot_matches_free_function(self, tmp_path: Path):
        repo = tmp_path / 'repo'
        repo.mkdir()
        asyncio.run(_init_repo(repo))

        git_ops = GitOps(GitConfig(), repo)
        assert not git_ops.pool_in_use()
        worktree_base = git_ops.worktree_base

        ta = _make_task_artifacts(worktree_base, '77', '77')
        ta.write_plan({'steps': [{'id': 's1', 'status': 'done'}]})
        ta.append_iteration_log({'note': 'iter-1'})
        ta.write_review('reviewer-a', {'verdict': 'PASS', 'issues': []})

        # Stub self: a bare SimpleNamespace duck-typing only the two
        # attributes the delegator reads — no full Harness construction.
        stub_self = types.SimpleNamespace(git_ops=git_ops, event_store=None)

        result = Harness.task_runtime_snapshot(stub_self)  # type: ignore[reportArgumentType]
        expected = build_task_runtime_snapshot(git_ops=git_ops, event_store=None)

        assert result == expected
        assert len(result) == 1
        entry = result[0]
        assert entry.task_id == 77
        assert entry.loops == 1
        assert entry.attempts == 1
        assert entry.lane is None
        assert entry.phase == 'DONE'
