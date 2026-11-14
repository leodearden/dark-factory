"""The unknown-argument guard on fused-memory's real servers (task 4867).

The bundled FastMCP ignores an argument its tool does not declare, so a
caller's misnamed ``metadata`` vanished and the call ran without it (escalations
esc-3147-7, esc-3671-7). Every test here drives a server built by
``create_mcp_server`` or ``_build_recon_report_components`` with the guards
installed the way ``main`` installs them.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastmcp.exceptions import ToolError
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent
from shared.mcp_unknown_arguments import UNKNOWN_ARGUMENT_CODE
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.server.main import (
    _build_recon_report_components,
    _install_tool_dispatch_guards,
)
from fused_memory.server.tools import create_mcp_server

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_PROJECT_ROOT = '/project'


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """``/project`` is no git checkout; the resolver is exercised elsewhere (test_task_tools.py)."""
    monkeypatch.setattr('fused_memory.server.tools.resolve_main_checkout', lambda p: str(p))


@pytest.fixture
def task_interceptor() -> AsyncMock:
    interceptor = AsyncMock()
    interceptor.update_task = AsyncMock(return_value={'success': True})
    interceptor.remove_tasks = AsyncMock(return_value={'successful': 1, 'failed': 0})
    return interceptor


@pytest.fixture
def server(task_interceptor: AsyncMock) -> Any:
    built = create_mcp_server(AsyncMock(), task_interceptor=task_interceptor)
    _install_tool_dispatch_guards(built, known_projects={'dark_factory': _PROJECT_ROOT})
    return built


def _refusal(exc_info: pytest.ExceptionInfo[ToolError]) -> dict[str, Any]:
    return json.loads(str(exc_info.value))


@pytest.mark.asyncio
async def test_a_misnamed_metadata_is_refused_and_nothing_is_written(server, task_interceptor):
    with pytest.raises(ToolError) as exc_info:
        await server._tool_manager.call_tool(
            'update_task',
            {'id': '7', 'project_root': _PROJECT_ROOT, 'metadata_patch': {'owner': 'x'}},
        )

    refusal = _refusal(exc_info)
    assert refusal['code'] == UNKNOWN_ARGUMENT_CODE
    assert refusal['tool'] == 'update_task'
    assert refusal['unknown'] == [{'name': 'metadata_patch', 'closest_declared': 'metadata'}]
    assert 'repaired_call' not in refusal
    task_interceptor.update_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_refusal_is_logged_on_the_server_naming_tool_and_argument(server, caplog):
    with (
        caplog.at_level('WARNING', logger='fused_memory.server.unknown_argument_guard'),
        pytest.raises(ToolError),
    ):
        await server._tool_manager.call_tool(
            'update_task',
            {'id': '7', 'project_root': _PROJECT_ROOT, 'metadata_patch': {'owner': 'secret-value'}},
        )

    [record] = [r for r in caplog.records if r.name == 'fused_memory.server.unknown_argument_guard']
    message = record.getMessage()
    assert 'update_task' in message
    assert 'metadata_patch' in message
    assert 'secret-value' not in message


@pytest.mark.asyncio
async def test_a_client_sent_ctx_is_refused_because_it_is_framework_injected(
    server, task_interceptor,
):
    with pytest.raises(ToolError) as exc_info:
        await server._tool_manager.call_tool(
            'update_task', {'id': '7', 'project_root': _PROJECT_ROOT, 'ctx': {}},
        )

    assert [entry['name'] for entry in _refusal(exc_info)['unknown']] == ['ctx']
    task_interceptor.update_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_call_naming_only_declared_arguments_runs_unchanged(server, task_interceptor):
    result = await server._tool_manager.call_tool(
        'update_task', {'id': '7', 'project_root': _PROJECT_ROOT, 'metadata': {'owner': 'x'}},
    )

    assert result == {'success': True}
    task_interceptor.update_task.assert_awaited_once()
    assert task_interceptor.update_task.call_args.kwargs['metadata'] == '{"owner": "x"}'


@pytest.mark.asyncio
async def test_arguments_none_is_an_empty_call_not_a_refusal(server):
    result = await server._tool_manager.call_tool('get_wal_status', None)

    assert not (isinstance(result, dict) and result.get('code') == UNKNOWN_ARGUMENT_CODE)


@pytest.mark.asyncio
async def test_the_refusal_reaches_a_client_as_structured_json_not_flattened(server):
    async with create_connected_server_and_client_session(server._mcp_server) as session:
        result = await session.call_tool(
            'update_task', {'id': '7', 'project_root': _PROJECT_ROOT, 'metadata_patch': {}},
        )

    assert result.isError is True
    text = result.content[0]
    assert isinstance(text, TextContent)
    refusal = json.loads(text.text)
    assert refusal['code'] == UNKNOWN_ARGUMENT_CODE
    assert refusal['unknown'][0]['name'] == 'metadata_patch'


@pytest.mark.asyncio
async def test_the_markup_override_on_a_tool_without_metadata_still_runs(
    server, task_interceptor,
):
    """The markup guard strips the override's ``metadata`` before this guard looks."""
    result = await server._tool_manager.call_tool(
        'remove_task',
        {'id': '7', 'project_root': _PROJECT_ROOT, 'metadata': {'allow_mcp_markup': True}},
    )

    assert result == {'successful': 1, 'failed': 0}
    task_interceptor.remove_tasks.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_unknown_tool_fails_open_to_the_dispatcher():
    built = FastMCP('unknown-tool')
    _install_tool_dispatch_guards(built)

    result = await built._tool_manager.call_tool('no_such_tool', {'anything': 1})

    assert isinstance(result, dict)
    assert 'no_such_tool' in result['error']
    assert UNKNOWN_ARGUMENT_CODE not in result['error']


@pytest.mark.asyncio
async def test_a_tool_with_no_usable_schema_fails_open():
    built = FastMCP('no-schema')
    calls: list[int] = []

    @built.tool()
    def echo(a: int) -> dict:
        calls.append(a)
        return {'a': a}

    tool = built._tool_manager.get_tool('echo')
    assert tool is not None
    tool.parameters = {'type': 'object'}
    _install_tool_dispatch_guards(built)

    result = await built._tool_manager.call_tool('echo', {'a': 1, 'extra': 2})

    assert calls == [1]
    assert UNKNOWN_ARGUMENT_CODE not in json.dumps(result, default=str)


@pytest.mark.asyncio
async def test_the_recon_report_server_refuses_an_undeclared_argument_too():
    from fused_memory.config.schema import (
        FusedMemoryConfig,
        ReconciliationConfig,
        ServerConfig,
    )

    config = FusedMemoryConfig(
        server=ServerConfig(recon_report_port=8003, host='127.0.0.1'),
        reconciliation=ReconciliationConfig(recon_report_state_ttl_seconds=300),
    )
    _, recon, _ = _build_recon_report_components(config)

    with pytest.raises(ToolError) as exc_info:
        await recon._tool_manager.call_tool(
            'set_stat', {'run_id': 'r1', 'key': 'k', 'value': 1, 'vlaue': 2},
        )
    declared_only = await recon._tool_manager.call_tool(
        'set_stat', {'run_id': 'r1', 'key': 'k', 'value': 1},
    )

    assert _refusal(exc_info)['unknown'] == [{'name': 'vlaue', 'closest_declared': 'value'}]
    assert UNKNOWN_ARGUMENT_CODE not in json.dumps(declared_only, default=str)
