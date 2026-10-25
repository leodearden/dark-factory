"""Guards for the plan-deviation grant, doctrine splice and escalate_info contract.

What this module guards (PRD ``plans/plan-deviation-recording-prd.md`` leaf γ):
which roles hold the deviation-recording tool, the IMPLEMENTER deviation
doctrine's splice and its render-from-source parity, and the splice of the one
escalate_info contract.

Every assertion is a containment, count or equality check against a NAMED live
constant: no regex over prose, and no scraping of backticked names out of a
prompt (INV-10 guards-exercise-behaviour; PRD D6). The standing rule and its
rationale live in ``orchestrator/tests/_role_splice_contract.py``.
"""

from __future__ import annotations

import pytest

from orchestrator.agents.roles import _PLAN_DEVIATION_TOOLS, ROLES
from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools

#: The roles that execute a frozen plan and may record where its text was wrong.
_DEVIATION_TOOL_HOLDERS = frozenset({'implementer', 'debugger'})

_PLAN_TOOLS_PREFIX = 'mcp__plan-tools__'


def _bare_name(qualified: str) -> str:
    return qualified.rsplit('__', 1)[-1]


class TestPlanDeviationToolGrant:
    """The allowlist half of PRD D14: who may call the deviation-recording tool."""

    def test_the_deviation_tool_list_is_nonempty(self):
        assert _PLAN_DEVIATION_TOOLS, (
            '_PLAN_DEVIATION_TOOLS is empty, so every containment check in this '
            'class passes vacuously. Restore its record_step_deviation entry.'
        )

    def test_implementer_and_debugger_hold_every_deviation_tool(self):
        missing = {
            name: sorted(set(_PLAN_DEVIATION_TOOLS) - set(ROLES[name].allowed_tools))
            for name in sorted(_DEVIATION_TOOL_HOLDERS)
        }
        offenders = {name: tools for name, tools in missing.items() if tools}
        assert offenders == {}, (
            f'Roles missing deviation tools from allowed_tools: {offenders}. '
            'Append *_PLAN_DEVIATION_TOOLS to their allowed_tools in roles.py.'
        )

    def test_no_other_role_holds_a_deviation_tool(self):
        offenders = {
            name: sorted(set(_PLAN_DEVIATION_TOOLS) & set(role.allowed_tools))
            for name, role in sorted(ROLES.items())
            if name not in _DEVIATION_TOOL_HOLDERS
            and set(_PLAN_DEVIATION_TOOLS) & set(role.allowed_tools)
        }
        assert offenders == {}, (
            f'Roles outside {sorted(_DEVIATION_TOOL_HOLDERS)} hold deviation '
            f'tools: {offenders}. Remove the grant. SIMPLE_TASK is the likely '
            'case: it holds _PLAN_STATUS_TOOLS but has no frozen plan and no '
            'doctrine (PRD D14), so the grant must not ride on _PLAN_STATUS_TOOLS.'
        )

    @pytest.mark.asyncio
    async def test_every_deviation_tool_is_registered_on_the_plan_tools_server(self, tmp_path):
        artifacts = TaskArtifacts(tmp_path / 'wt')
        artifacts.init('5753', 'T', 'd')
        server = plan_tools.create_server(artifacts)
        registry = {tool.name for tool in await server.list_tools()}

        foreign = sorted(
            qualified for qualified in _PLAN_DEVIATION_TOOLS
            if not qualified.startswith(_PLAN_TOOLS_PREFIX)
        )
        assert foreign == [], (
            f'_PLAN_DEVIATION_TOOLS entries not on the plan-tools server: {foreign}. '
            f'Every entry must start with {_PLAN_TOOLS_PREFIX!r}.'
        )
        unregistered = sorted(
            qualified for qualified in _PLAN_DEVIATION_TOOLS
            if _bare_name(qualified) not in registry
        )
        assert unregistered == [], (
            f'_PLAN_DEVIATION_TOOLS names tools plan_tools.create_server does not '
            f'register: {unregistered}. A granted name with no registered tool '
            'leaves the role with a dead grant; fix the name or register the tool.'
        )
