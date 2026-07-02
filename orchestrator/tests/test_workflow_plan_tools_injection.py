"""Regression-guard tests for _inject_plan_tools_mcp() wiring in workflow.py.

Verifies the two genuinely new behaviors of _inject_plan_tools_mcp():
1. _inject_plan_tools_mcp(None, worktree) creates an mcpServers dict with a
   'plan-tools' entry equal to
   plan_tools_mcp_server(_ORCH_PROJECT_DIR, worktree, python_executable=sys.executable)
   (single source of truth — no-uv hot path + worktree propagation guaranteed
   transitively; task 1776).  The 'command' key must equal sys.executable (not 'uv').
2. A pre-existing server entry (e.g. fused-memory) is preserved alongside the
   new plan-tools entry.
"""

from __future__ import annotations

import sys
from pathlib import Path


class TestInjectPlanToolsMcp:
    """_inject_plan_tools_mcp() creates/extends mcpServers with the no-uv
    direct-interpreter plan-tools entry and preserves pre-existing servers."""

    def test_none_config_creates_plan_tools_entry(self):
        """Given mcp_config=None, result has mcpServers['plan-tools'] == helper output
        with python_executable=sys.executable (no-uv hot path, task 1776)."""
        from orchestrator.mcp_lifecycle import plan_tools_mcp_server  # noqa: PLC0415
        from orchestrator.workflow import _ORCH_PROJECT_DIR, _inject_plan_tools_mcp  # noqa: PLC0415

        wt = Path('/wt')
        out = _inject_plan_tools_mcp(None, wt)

        assert 'mcpServers' in out
        expected = plan_tools_mcp_server(_ORCH_PROJECT_DIR, wt, python_executable=sys.executable)
        assert out['mcpServers']['plan-tools'] == expected
        # Explicit command assertion: must be the venv interpreter, not 'uv'.
        assert out['mcpServers']['plan-tools']['command'] == sys.executable

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




class TestMcpRoleGates:
    """_invoke()'s per-role MCP wiring gates (reify esc-4943-54).

    A role whose AgentRole.allowed_tools / briefing references an MCP tool
    family MUST also be listed in the matching _invoke() gate, or its
    sessions are instructed to call tools that do not exist in the session.
    simple_task was the incident case: its contract (_run_simple_task
    requires a plan-tools-registered plan) was unsatisfiable, so the Lever-C
    fast-path always fell through to the architect.
    """

    def test_simple_task_in_plan_tools_gate(self):
        from orchestrator.workflow import _PLAN_TOOLS_ROLES  # noqa: PLC0415

        assert 'simple_task' in _PLAN_TOOLS_ROLES

    def test_simple_task_in_mcp_config_gate(self):
        from orchestrator.workflow import _MCP_CONFIG_ROLES  # noqa: PLC0415

        assert 'simple_task' in _MCP_CONFIG_ROLES

    def test_plan_tools_gate_covers_every_role_allowing_plan_tools(self):
        """Inverse invariant: every ROLES entry that allows any
        mcp__plan-tools__* tool must be in _PLAN_TOOLS_ROLES."""
        from orchestrator.agents.roles import ROLES  # noqa: PLC0415
        from orchestrator.workflow import _PLAN_TOOLS_ROLES  # noqa: PLC0415

        for name, role in ROLES.items():
            allows_plan_tools = any(
                t.startswith('mcp__plan-tools__') for t in role.allowed_tools
            )
            if allows_plan_tools:
                assert name in _PLAN_TOOLS_ROLES, (
                    f'{name!r} allows plan-tools tools in allowed_tools but is '
                    f'not in _PLAN_TOOLS_ROLES — its sessions would be told to '
                    f'call tools that do not exist (reify esc-4943-54 class)'
                )
