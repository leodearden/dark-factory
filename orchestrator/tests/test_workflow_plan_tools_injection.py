"""Regression-guard tests for _inject_plan_tools_mcp() wiring in workflow.py.

Verifies that:
1. _inject_plan_tools_mcp(None, worktree) creates an mcpServers dict with a
   'plan-tools' entry equal to plan_tools_mcp_server(_ORCH_PROJECT_DIR, worktree)
   (single source of truth) and its args contain the fast-start flags.
2. A pre-existing server entry (e.g. fused-memory) is preserved alongside the
   new plan-tools entry.
3. The worktree path passed in propagates through to the helper args.
"""

from __future__ import annotations

from pathlib import Path


class TestInjectPlanToolsMcp:
    """_inject_plan_tools_mcp() creates/extends mcpServers with the fast-start
    plan-tools entry and preserves pre-existing servers."""

    def test_none_config_creates_plan_tools_entry(self):
        """Given mcp_config=None, result has mcpServers['plan-tools'] == helper output."""
        from orchestrator.mcp_lifecycle import plan_tools_mcp_server  # noqa: PLC0415
        from orchestrator.workflow import _ORCH_PROJECT_DIR, _inject_plan_tools_mcp  # noqa: PLC0415

        wt = Path('/wt')
        out = _inject_plan_tools_mcp(None, wt)

        assert 'mcpServers' in out
        expected = plan_tools_mcp_server(_ORCH_PROJECT_DIR, wt)
        assert out['mcpServers']['plan-tools'] == expected

    def test_plan_tools_entry_contains_fast_start_flags(self):
        """plan-tools args in the injected config contain --no-sync and --frozen."""
        from orchestrator.workflow import _inject_plan_tools_mcp  # noqa: PLC0415

        out = _inject_plan_tools_mcp(None, Path('/wt'))
        args = out['mcpServers']['plan-tools']['args']
        assert '--no-sync' in args, f"Expected '--no-sync' in args, got {args}"
        assert '--frozen' in args, f"Expected '--frozen' in args, got {args}"

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

    def test_worktree_propagates_to_args(self):
        """The worktree path passed to _inject_plan_tools_mcp appears in plan-tools args."""
        from orchestrator.workflow import _inject_plan_tools_mcp  # noqa: PLC0415

        wt = Path('/my/special/worktree')
        out = _inject_plan_tools_mcp(None, wt)
        args = out['mcpServers']['plan-tools']['args']

        wt_idx = args.index('--worktree')
        assert args[wt_idx + 1] == str(wt), (
            f"Expected --worktree {str(wt)!r}, got {args[wt_idx + 1]!r}"
        )
