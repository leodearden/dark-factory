"""Regression-guard tests for _inject_plan_tools_mcp() wiring in workflow.py.

Verifies the two genuinely new behaviors of _inject_plan_tools_mcp():
1. _inject_plan_tools_mcp(None, worktree, task_root=...) creates an mcpServers dict with a
   'plan-tools' entry equal to
   plan_tools_mcp_server(_ORCH_PROJECT_DIR, worktree, python_executable=sys.executable,
   meta_root=_meta_root_for_worktree(worktree), task_root=task_root)
   (single source of truth — no-uv hot path + worktree propagation guaranteed
   transitively; task 1776).  The 'command' key must equal sys.executable (not 'uv').
2. A pre-existing server entry (e.g. fused-memory) is preserved alongside the
   new plan-tools entry.

Also covers _meta_root_for_worktree() (task 2258, W11-ε1) — the single
DRY seam used by both the orchestrator-side TaskArtifacts construction and
this injection helper to derive the sibling `.task-meta` root for a worktree.
"""

from __future__ import annotations

import sys
from pathlib import Path

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp_lifecycle import plan_tools_mcp_server
from orchestrator.workflow import (
    _ORCH_PROJECT_DIR,
    _inject_plan_tools_mcp,
    _meta_root_for_worktree,
)

_TASK_ROOT = Path('/.task-meta/task-3155')


class TestMetaRootForWorktree:
    """_meta_root_for_worktree() derives the sibling `.task-meta` root
    (worktree-lane-lifecycle PRD task ε1) via TaskArtifacts.meta_root_for —
    never by hand-joining '.task' or '.task-meta'."""

    def test_derives_sibling_task_meta_path(self):
        assert _meta_root_for_worktree(Path('/base/wt')) == Path('/base/.task-meta/wt')

    def test_result_is_sibling_not_child_of_worktree(self):
        """The derived path must NOT live under the worktree itself — that's
        the entire point of the relocation (git add -A/clean/checkout in the
        worktree cannot reach a sibling path)."""

        worktree = Path('/base/wt')
        meta_root = _meta_root_for_worktree(worktree)

        assert worktree not in meta_root.parents
        assert meta_root.parent.parent == worktree.parent


class TestInjectPlanToolsMcp:
    """_inject_plan_tools_mcp() creates/extends mcpServers with the no-uv
    direct-interpreter plan-tools entry and preserves pre-existing servers."""

    def test_none_config_creates_plan_tools_entry(self):
        """Given mcp_config=None, result has mcpServers['plan-tools'] == helper output
        with python_executable=sys.executable (no-uv hot path, task 1776) and
        meta_root=_meta_root_for_worktree(wt) (task 2258, W11-ε1)."""

        wt = Path('/wt')
        out = _inject_plan_tools_mcp(None, wt, task_root=_TASK_ROOT)

        assert 'mcpServers' in out
        expected = plan_tools_mcp_server(
            _ORCH_PROJECT_DIR, wt, python_executable=sys.executable,
            meta_root=_meta_root_for_worktree(wt), task_root=_TASK_ROOT,
        )
        assert out['mcpServers']['plan-tools'] == expected
        # Explicit command assertion: must be the venv interpreter, not 'uv'.
        assert out['mcpServers']['plan-tools']['command'] == sys.executable

    def test_lane_and_task_roots_passed_through_to_args(self):
        """args carry '--meta-root <lane root>' then '--task-root <task root>'
        — the agent-side plan-tools server must target the same two roots the
        orchestrator uses (task 3155)."""

        wt = Path('/base/wt')
        out = _inject_plan_tools_mcp(None, wt, task_root=_TASK_ROOT)

        args = out['mcpServers']['plan-tools']['args']
        idx = args.index('--meta-root')
        assert args[idx:idx + 4] == [
            '--meta-root', str(TaskArtifacts.meta_root_for(wt.parent, wt.name)),
            '--task-root', str(_TASK_ROOT),
        ]

    def test_existing_server_is_preserved(self):
        """Pre-existing servers (e.g. fused-memory) survive alongside plan-tools."""

        existing = {
            'mcpServers': {
                'fused-memory': {'type': 'http', 'url': 'http://x/mcp'},
            },
        }
        out = _inject_plan_tools_mcp(existing, Path('/wt'), task_root=_TASK_ROOT)

        assert 'fused-memory' in out['mcpServers'], (
            f"Expected 'fused-memory' preserved, got {list(out['mcpServers'])}"
        )
        assert out['mcpServers']['fused-memory'] == {'type': 'http', 'url': 'http://x/mcp'}
        assert 'plan-tools' in out['mcpServers']
