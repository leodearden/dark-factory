"""The plan-tools ``report_missing_plan_file`` tool: a writer's scope request (task 3090).

The tool records a request for an existing worktree file in the LANE
``scope_requests.json`` artifact and never writes plan.json; the workflow
applies it at the next writer boundary.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.mcp import plan_tools
from orchestrator.scope_drift import ScopeRequest

_PACKAGE = 'orchestrator/src/orchestrator'
_DECLARED = f'{_PACKAGE}/a.py'
_EXISTING = ('a', 'b', 'c', *(f'm{n}' for n in range(8)))


@pytest.fixture()
def artifacts(tmp_path) -> TaskArtifacts:
    package = tmp_path / _PACKAGE
    package.mkdir(parents=True)
    for name in _EXISTING:
        (package / f'{name}.py').write_text('')
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    a.write_plan({
        'task_id': 'test-1', 'title': 'Test task', 'analysis': 'Analysis',
        'files': [_DECLARED], 'prerequisites': [], 'steps': [],
    })
    return a


async def _report(artifacts: TaskArtifacts, path: str, reason: str) -> dict[str, Any]:
    tool = await plan_tools.create_server(artifacts).get_tool('report_missing_plan_file')
    assert tool is not None
    result = await tool.run({'path': path, 'reason': reason})
    assert result.structured_content is not None
    return result.structured_content


def _plan_bytes(artifacts: TaskArtifacts) -> bytes:
    return artifacts.artifact_path('plan.json').read_bytes()


@pytest.mark.asyncio
async def test_an_out_of_plan_path_is_recorded_and_counted(artifacts):
    before = _plan_bytes(artifacts)

    first = await _report(artifacts, './orchestrator/src/orchestrator/b.py', 'plan omitted it')
    assert first == {
        'status': 'ok',
        'path': 'orchestrator/src/orchestrator/b.py',
        'pending': 1,
    }
    second = await _report(artifacts, 'orchestrator/src/orchestrator/c.py', 'helper lives here')
    assert second['status'] == 'ok'
    assert second['pending'] == 2

    assert artifacts.read_scope_requests() == (
        ScopeRequest('orchestrator/src/orchestrator/b.py', 'plan omitted it'),
        ScopeRequest('orchestrator/src/orchestrator/c.py', 'helper lives here'),
    )
    assert _plan_bytes(artifacts) == before


@pytest.mark.asyncio
async def test_a_declared_path_is_already_in_scope_and_writes_nothing(artifacts):
    before = _plan_bytes(artifacts)

    result = await _report(artifacts, _DECLARED, 'just checking')

    assert result == {'status': 'already_in_scope', 'path': _DECLARED}
    assert artifacts.read_scope_requests() == ()
    assert not artifacts.artifact_path('scope_requests.json').exists()
    assert _plan_bytes(artifacts) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('path', 'reason', 'field'),
    [
        ('/abs/orchestrator/b.py', 'r', 'path'),
        ('orchestrator/../b.py', 'r', 'path'),
        ('.task/plan.json', 'r', 'path'),
        ('orchestrator/src/orchestrator', 'r', 'path'),
        ('', 'r', 'path'),
        ('orchestrator/src/orchestrator/b.py', '   ', 'reason'),
    ],
)
async def test_an_invalid_request_returns_an_error_and_writes_nothing(
    artifacts, path, reason, field
):
    before = _plan_bytes(artifacts)

    result = await _report(artifacts, path, reason)

    assert result['status'] == 'error'
    assert result['field'] == field
    assert isinstance(result['message'], str) and result['message']
    assert artifacts.read_scope_requests() == ()
    assert _plan_bytes(artifacts) == before


@pytest.mark.asyncio
async def test_a_path_naming_no_file_returns_an_error_and_writes_nothing(artifacts):
    before = _plan_bytes(artifacts)

    result = await _report(artifacts, f'{_PACKAGE}/typo.py', 'plan omitted it')

    assert result['status'] == 'error'
    assert result['field'] == 'path'
    assert 'typo.py' in result['message']
    assert artifacts.read_scope_requests() == ()
    assert _plan_bytes(artifacts) == before


@pytest.mark.asyncio
async def test_concurrent_reports_lose_no_request(artifacts):
    paths = [f'orchestrator/src/orchestrator/m{n}.py' for n in range(8)]

    results = await asyncio.gather(*(_report(artifacts, path, 'parallel') for path in paths))

    assert all(result['status'] == 'ok' for result in results)
    assert sorted(r.path for r in artifacts.read_scope_requests()) == sorted(paths)
    assert sorted(result['pending'] for result in results) == list(range(1, 9))
