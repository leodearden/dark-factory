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

from test_task_runtime import _init_repo, _make_task_artifacts

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
from orchestrator.harness import Harness
from orchestrator.task_runtime import build_task_runtime_snapshot

# _init_repo / _make_task_artifacts are shared with test_task_runtime.py
# (dark_factory's established cross-test-module reuse pattern — e.g.
# test_merge_queue_dry_run_unblock.py importing from test_dry_run_unblock.py
# — rather than duplicating the fixtures verbatim in both files).


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
