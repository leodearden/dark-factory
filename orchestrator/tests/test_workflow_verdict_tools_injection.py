"""Regression-guard tests for _inject_verdict_tools_mcp() wiring in workflow.py.

Mirrors test_workflow_plan_tools_injection.py::TestInjectPlanToolsMcp. Verifies
the two genuinely new behaviors of _inject_verdict_tools_mcp():
1. _inject_verdict_tools_mcp(None, worktree, role, task_root=...) creates an mcpServers dict
   with a 'verdict-tools' entry equal to
   verdict_tools_mcp_server(_ORCH_PROJECT_DIR, worktree, role.name,
   python_executable=sys.executable, meta_root=_meta_root_for_worktree(worktree),
   task_root=task_root)
   (single source of truth — no-uv hot path guaranteed transitively). The
   'command' key must equal sys.executable (not 'uv'). role.name drives
   --verdict-role — the authoritative verdicts/<role>.json filename contract
   (α's I-AUTHORITATIVE-PATH invariant, orchestrator/src/orchestrator/mcp/
   verdict_tools.py).
2. A pre-existing server entry (e.g. fused-memory) is preserved alongside the
   new verdict-tools entry.

Task 2482, PRD mcp-verdict-servers-prd.md task β.
"""

from __future__ import annotations

import sys
from pathlib import Path

from orchestrator.agents.roles import AgentRole
from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp_lifecycle import verdict_tools_mcp_server
from orchestrator.workflow import (
    _ORCH_PROJECT_DIR,
    _inject_verdict_tools_mcp,
    _meta_root_for_worktree,
)

_TASK_ROOT = Path('/.task-meta/task-3155')


class TestInjectVerdictToolsMcp:
    """_inject_verdict_tools_mcp() creates/extends mcpServers with the no-uv
    direct-interpreter verdict-tools entry and preserves pre-existing servers."""

    def test_none_config_creates_verdict_tools_entry(self):
        """Given mcp_config=None, result has mcpServers['verdict-tools'] == helper
        output with python_executable=sys.executable (no-uv hot path) and
        meta_root=_meta_root_for_worktree(wt)."""

        role = AgentRole(name='reviewer_comprehensive', system_prompt='x', allowed_tools=[])
        wt = Path('/wt')
        out = _inject_verdict_tools_mcp(None, wt, role, task_root=_TASK_ROOT)

        assert 'mcpServers' in out
        expected = verdict_tools_mcp_server(
            _ORCH_PROJECT_DIR, wt, role.name, python_executable=sys.executable,
            meta_root=_meta_root_for_worktree(wt), task_root=_TASK_ROOT,
        )
        assert out['mcpServers']['verdict-tools'] == expected
        # Explicit command assertion: must be the venv interpreter, not 'uv'.
        assert out['mcpServers']['verdict-tools']['command'] == sys.executable

    def test_verdict_role_flag_passed_through_to_args(self):
        """args contains '--verdict-role' immediately followed by role.name —
        the authoritative selector for both the registered tool and the
        verdicts/<role>.json filename (α's contract)."""

        role = AgentRole(name='reviewer_comprehensive', system_prompt='x', allowed_tools=[])
        wt = Path('/wt')
        out = _inject_verdict_tools_mcp(None, wt, role, task_root=_TASK_ROOT)

        args = out['mcpServers']['verdict-tools']['args']
        idx = args.index('--verdict-role')
        assert args[idx + 1] == role.name

    def test_lane_and_task_roots_precede_the_verdict_role(self):
        """args carry '--meta-root <lane root>', '--task-root <task root>',
        then the '--verdict-role' tail — the agent-side verdict-tools server
        must target the same two roots the orchestrator uses (task 3155)."""

        role = AgentRole(name='reviewer_comprehensive', system_prompt='x', allowed_tools=[])
        wt = Path('/base/wt')
        out = _inject_verdict_tools_mcp(None, wt, role, task_root=_TASK_ROOT)

        args = out['mcpServers']['verdict-tools']['args']
        idx = args.index('--meta-root')
        assert args[idx:] == [
            '--meta-root', str(TaskArtifacts.meta_root_for(wt.parent, wt.name)),
            '--task-root', str(_TASK_ROOT),
            '--verdict-role', role.name,
        ]

    def test_existing_server_is_preserved(self):
        """Pre-existing servers (e.g. fused-memory) survive alongside verdict-tools —
        this is what lets a role with NO 'orchestrator' family (e.g. the reviewer)
        still acquire the verdict-tools server via the None-skeleton path."""

        role = AgentRole(name='reviewer_comprehensive', system_prompt='x', allowed_tools=[])
        existing = {
            'mcpServers': {
                'fused-memory': {'type': 'http', 'url': 'http://x/mcp'},
            },
        }
        out = _inject_verdict_tools_mcp(existing, Path('/wt'), role, task_root=_TASK_ROOT)

        assert 'fused-memory' in out['mcpServers'], (
            f"Expected 'fused-memory' preserved, got {list(out['mcpServers'])}"
        )
        assert out['mcpServers']['fused-memory'] == {'type': 'http', 'url': 'http://x/mcp'}
        assert 'verdict-tools' in out['mcpServers']
