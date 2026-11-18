"""The plan-tools server exposes only the tools its dispatch is granted (task 7717).

Every Claude dispatch runs under ``bypassPermissions``, where
``--allowed-tools`` neither hides nor denies a tool, so a role's plan-tools
allowlist holds only because the server registers nothing outside it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from orchestrator import mcp_lifecycle
from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools

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
async def test_the_cli_exposes_nothing_without_a_grant(tmp_path):
    assert await _listed(plan_tools.server_from_argv(_lane_argv(tmp_path))) == []


@pytest.mark.asyncio
async def test_the_cli_exposes_exactly_the_granted_tools(tmp_path):
    argv = [
        *_lane_argv(tmp_path),
        '--grant', 'mark_step_done', '--grant', 'record_step_deviation',
    ]

    assert await _listed(plan_tools.server_from_argv(argv)) == [
        'mark_step_done', 'record_step_deviation',
    ]


@pytest.mark.parametrize('python_executable', ['/venv/bin/python', None])
def test_the_launcher_appends_sorted_grant_pairs_after_the_root_args(
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

    assert plain['args'][-2:] == ['--task-root', str(roots['task_root'])]
    assert granted['args'] == [*plain['args'], '--grant', 'a', '--grant', 'b']
