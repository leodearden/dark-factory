"""Regression-guard tests for _inject_plan_tools_mcp() wiring in workflow.py.

Verifies the two genuinely new behaviors of _inject_plan_tools_mcp():
1. _inject_plan_tools_mcp(None, worktree) creates an mcpServers dict with a
   'plan-tools' entry equal to
   plan_tools_mcp_server(_ORCH_PROJECT_DIR, worktree, python_executable=sys.executable,
   meta_root=_meta_root_for_worktree(worktree))
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


class TestMetaRootForWorktree:
    """_meta_root_for_worktree() derives the sibling `.task-meta` root
    (worktree-lane-lifecycle PRD task ε1) via TaskArtifacts.meta_root_for —
    never by hand-joining '.task' or '.task-meta'."""

    def test_derives_sibling_task_meta_path(self):
        from orchestrator.workflow import _meta_root_for_worktree  # noqa: PLC0415

        assert _meta_root_for_worktree(Path('/base/wt')) == Path('/base/.task-meta/wt')

    def test_result_is_sibling_not_child_of_worktree(self):
        """The derived path must NOT live under the worktree itself — that's
        the entire point of the relocation (git add -A/clean/checkout in the
        worktree cannot reach a sibling path)."""
        from orchestrator.workflow import _meta_root_for_worktree  # noqa: PLC0415

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
        from orchestrator.mcp_lifecycle import plan_tools_mcp_server  # noqa: PLC0415
        from orchestrator.workflow import (  # noqa: PLC0415
            _ORCH_PROJECT_DIR,
            _inject_plan_tools_mcp,
            _meta_root_for_worktree,
        )

        wt = Path('/wt')
        out = _inject_plan_tools_mcp(None, wt)

        assert 'mcpServers' in out
        expected = plan_tools_mcp_server(
            _ORCH_PROJECT_DIR, wt, python_executable=sys.executable,
            meta_root=_meta_root_for_worktree(wt),
        )
        assert out['mcpServers']['plan-tools'] == expected
        # Explicit command assertion: must be the venv interpreter, not 'uv'.
        assert out['mcpServers']['plan-tools']['command'] == sys.executable

    def test_meta_root_flag_passed_through_to_args(self):
        """args contains '--meta-root' immediately followed by
        str(_meta_root_for_worktree(wt)) — the agent-side plan-tools server
        must target the same relocated root the orchestrator uses."""
        from orchestrator.workflow import (  # noqa: PLC0415
            _inject_plan_tools_mcp,
            _meta_root_for_worktree,
        )

        wt = Path('/wt')
        out = _inject_plan_tools_mcp(None, wt)

        args = out['mcpServers']['plan-tools']['args']
        idx = args.index('--meta-root')
        assert args[idx + 1] == str(_meta_root_for_worktree(wt))

    def test_existing_server_is_preserved(self):
        """Pre-existing servers (e.g. fused-memory) survive alongside plan-tools."""
        from orchestrator.workflow import _inject_plan_tools_mcp  # noqa: PLC0415

        existing = {
            'mcpServers': {
                'fused-memory': {'type': 'http', 'url': 'http://x/mcp'},
            },
        }
        out = _inject_plan_tools_mcp(existing, Path('/wt'))

        assert 'fused-memory' in out['mcpServers'], (
            f"Expected 'fused-memory' preserved, got {list(out['mcpServers'])}"
        )
        assert out['mcpServers']['fused-memory'] == {'type': 'http', 'url': 'http://x/mcp'}
        assert 'plan-tools' in out['mcpServers']
