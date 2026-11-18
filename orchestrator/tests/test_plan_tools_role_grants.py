"""The plan-tools server exposes only the tools its dispatch is granted (task 7717).

Every Claude dispatch runs under ``bypassPermissions``, where
``--allowed-tools`` neither hides nor denies a tool, so a role's plan-tools
allowlist holds only because the server registers nothing outside it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from orchestrator import mcp_lifecycle
from orchestrator.agents.roles import ARCHITECT, DEBUGGER, IMPLEMENTER, ROLES, SIMPLE_TASK
from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools
from orchestrator.workflow import compose_invocation_mcp_config

_DEVIATION_ARGS = {
    'step_id': 'step-1', 'kind': 'wrong_inference', 'observed': 'o', 'expected': 'e',
    'action_taken': 'a',
}


async def _seed_plan(artifacts: TaskArtifacts) -> None:
    async with Client(plan_tools.create_server(artifacts)) as client:
        await client.call_tool('create_plan', {
            'task_id': '42', 'title': 'X', 'analysis': 'a', 'files': ['x.py'],
        })
        await client.call_tool('add_plan_step', {
            'step_id': 'step-1', 'step_type': 'test', 'description': 'd',
        })


def _step_deviations(artifacts: TaskArtifacts) -> list:
    (step,) = artifacts.read_plan()['steps']
    return step.get('deviations', [])


async def _listed(server) -> list[str]:
    async with Client(server) as client:
        return sorted(tool.name for tool in await client.list_tools())


@pytest.fixture
def artifacts(tmp_path: Path) -> TaskArtifacts:
    artifacts = TaskArtifacts(tmp_path)
    artifacts.init('42', 'X', 'desc', base_commit='base')
    return artifacts


@pytest.mark.asyncio
async def test_a_granted_server_lists_and_dispatches_only_its_grant(artifacts):
    await _seed_plan(artifacts)
    server = plan_tools.create_server(artifacts, granted_tools=frozenset({'mark_step_done'}))

    assert await _listed(server) == ['mark_step_done']
    async with Client(server) as client:
        with pytest.raises(ToolError, match='Unknown tool'):
            await client.call_tool('record_step_deviation', _DEVIATION_ARGS)
    assert _step_deviations(artifacts) == []


def test_an_unknown_grant_is_refused(artifacts):
    with pytest.raises(ValueError, match='no_such_tool'):
        plan_tools.create_server(
            artifacts, granted_tools=frozenset({'mark_step_done', 'no_such_tool'}),
        )


@pytest.mark.asyncio
async def test_no_grant_argument_keeps_the_full_surface(artifacts):
    names = await _listed(plan_tools.create_server(artifacts))

    assert {
        'create_plan', 'add_plan_step', 'confirm_plan', 'mark_step_done',
        'record_step_deviation', 'report_missing_plan_file',
    } <= set(names)
    assert await _listed(
        plan_tools.create_server(artifacts, granted_tools=frozenset(names)),
    ) == names


def _lane_argv(tmp_path: Path) -> list[str]:
    worktree = tmp_path / '.worktrees' / '_lane-3'
    worktree.mkdir(parents=True)
    lane = TaskArtifacts.for_task(worktree, '42')
    lane.init('42', 'X', 'desc')
    return [
        '--worktree', str(worktree),
        '--meta-root', str(lane.lane_root),
        '--task-root', str(lane.task_root),
    ]


@pytest.mark.asyncio
async def test_an_unmarked_cli_launch_keeps_the_full_surface_and_warns(
    tmp_path, artifacts, caplog,
):
    with caplog.at_level(logging.WARNING, logger='orchestrator.mcp.plan_tools'):
        server = plan_tools.server_from_argv(_lane_argv(tmp_path))

    assert await _listed(server) == await _listed(plan_tools.create_server(artifacts))
    assert any(
        record.levelno == logging.WARNING and '--grants-enforced' in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_a_marked_cli_launch_with_no_grant_exposes_nothing(tmp_path):
    argv = [*_lane_argv(tmp_path), '--grants-enforced']

    assert await _listed(plan_tools.server_from_argv(argv)) == []


@pytest.mark.asyncio
@pytest.mark.parametrize('marker', [[], ['--grants-enforced']], ids=['unmarked', 'marked'])
async def test_the_cli_exposes_exactly_the_granted_tools(tmp_path, marker):
    argv = [
        *_lane_argv(tmp_path), *marker,
        '--grant', 'mark_step_done', '--grant', 'record_step_deviation',
    ]

    assert await _listed(plan_tools.server_from_argv(argv)) == [
        'mark_step_done', 'record_step_deviation',
    ]


@pytest.mark.parametrize('python_executable', ['/venv/bin/python', None])
def test_the_launcher_always_appends_the_marker_then_sorted_grant_pairs_after_the_root_args(
    tmp_path, python_executable,
):
    roots = {
        'meta_root': tmp_path / '.task-meta' / '_lane-3',
        'task_root': tmp_path / '.task-meta' / 'task-42',
    }
    plain = mcp_lifecycle.plan_tools_mcp_server(
        tmp_path / 'orchestrator', tmp_path / '_lane-3',
        python_executable=python_executable, **roots,
    )
    granted = mcp_lifecycle.plan_tools_mcp_server(
        tmp_path / 'orchestrator', tmp_path / '_lane-3',
        python_executable=python_executable, granted_tools={'b', 'a'}, **roots,
    )

    assert plain['args'][-3:] == ['--task-root', str(roots['task_root']), '--grants-enforced']
    assert granted['args'] == [*plain['args'], '--grant', 'a', '--grant', 'b']


_PLAN_TOOLS_ROLES = [ARCHITECT, IMPLEMENTER, DEBUGGER, SIMPLE_TASK]
_DEVIATION_RECORDERS = frozenset({IMPLEMENTER.name, DEBUGGER.name})
_PLAN_TOOLS_PREFIX = 'mcp__plan-tools__'


def test_the_plan_tools_family_is_exactly_the_roles_tested_here():
    assert sorted(
        role.name for role in ROLES.values() if 'plan_tools' in role.mcp_families
    ) == sorted(role.name for role in _PLAN_TOOLS_ROLES)


@pytest.mark.parametrize('role', _PLAN_TOOLS_ROLES, ids=lambda role: role.name)
def test_a_roles_grant_is_its_plan_tools_allowlist(role):
    assert role.plan_tools_grants == {
        tool.removeprefix(_PLAN_TOOLS_PREFIX)
        for tool in role.allowed_tools if tool.startswith(_PLAN_TOOLS_PREFIX)
    }


@pytest.mark.asyncio
@pytest.mark.parametrize('role', _PLAN_TOOLS_ROLES, ids=lambda role: role.name)
async def test_a_dispatch_exposes_exactly_its_roles_plan_tools_allowlist(tmp_path, role):
    worktree = tmp_path / '.worktrees' / '_lane-3'
    worktree.mkdir(parents=True)
    artifacts = TaskArtifacts.for_task(worktree, '42')
    artifacts.init('42', 'X', 'desc')
    await _seed_plan(artifacts)
    config = compose_invocation_mcp_config(
        role, mcp=None, escalation_url=None, cwd=worktree, task_root=artifacts.task_root,
        strict_mcp_config=False,
    )
    assert config is not None
    args = config['mcpServers']['plan-tools']['args']
    assert args.count('--grants-enforced') == 1
    launch = args.index('orchestrator.mcp.plan_tools')
    assert args[launch - 1] == '-m'
    server = plan_tools.server_from_argv(args[launch + 1:])

    assert await _listed(server) == sorted(role.plan_tools_grants)
    async with Client(server) as client:
        if role.name in _DEVIATION_RECORDERS:
            result = await client.call_tool('record_step_deviation', _DEVIATION_ARGS)
            assert result.data['status'] == 'ok'
        else:
            with pytest.raises(ToolError, match='Unknown tool'):
                await client.call_tool('record_step_deviation', _DEVIATION_ARGS)
    assert len(_step_deviations(artifacts)) == (role.name in _DEVIATION_RECORDERS)
