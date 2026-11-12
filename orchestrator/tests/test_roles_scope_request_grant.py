"""Only the writers that the workflow briefs on scope requests may file one (task 3090).

``workflow.py::TaskWorkflow._apply_scope_requests`` applies requests after
implementer and debugger sessions only, so those two roles hold the grant and
no other role does.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.agents.roles import ARCHITECT, DEBUGGER, IMPLEMENTER, SIMPLE_TASK
from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools

_TOOL = 'report_missing_plan_file'
_GRANT = f'mcp__plan-tools__{_TOOL}'


@pytest.mark.parametrize('role', [IMPLEMENTER, DEBUGGER], ids=lambda r: r.name)
def test_the_boundary_writers_hold_the_grant(role):
    assert _GRANT in role.allowed_tools


@pytest.mark.parametrize('role', [SIMPLE_TASK, ARCHITECT], ids=lambda r: r.name)
def test_roles_without_a_boundary_do_not(role):
    assert _GRANT not in role.allowed_tools


@pytest.mark.asyncio
async def test_granted_name_resolves_to_a_registered_tool(tmp_path: Path):
    artifacts = TaskArtifacts(tmp_path)
    artifacts.init('42', 'X', 'desc', base_commit='base')
    server = plan_tools.create_server(artifacts)

    assert await server.get_tool(_TOOL) is not None
