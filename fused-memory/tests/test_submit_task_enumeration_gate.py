"""The enumeration liveness gate at the real MCP task-tool boundary (task 3056).

Reify gate 5537 shipped enumerating a Mem0 entry deleted about eleven minutes
earlier, alongside a truncated 8-char id. These tests drive ``submit_task`` and
``update_task`` through ``create_mcp_server`` so the gate is exercised with
the preceding recon hooks (``execution_class_error``,
``recurring_gate_guard_error``) in the loop, and assert what the task
interceptor actually receives. The gate's own rule matrix lives in
``tests/reconciliation/test_enumeration_guard.py``.

Harness modelled on ``test_delete_memory_citation_guard.py``:
``create_mcp_server`` + ``_tool_manager.call_tool`` + ``_parse_result``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from fused_memory.server.tools import create_mcp_server

RECON_AGENT = 'recon-stage-task_knowledge_sync'

LIVE_B3F = 'b3f55b47-0879-49e0-a26d-53bf6ea9689a'
LIVE_21F = '21f78de1-a5f3-4259-b907-16c8f2328e2f'
LIVE_23E = '23eff0cd-8515-40fb-9b3e-9a5be19cbc0c'
DELETED_0D5 = '0d542614-874b-4003-b136-76dd9d8b40f8'
LIVE_IDS = (LIVE_B3F, LIVE_21F, LIVE_23E)

VALID_SCOPE = {
    'stores': ['mem0'],
    'categories': ['observations_and_summaries', 'procedural_knowledge'],
    'exhaustive': False,
    'note': 'Stage-1 search over two categories only',
}

GATE_DESCRIPTION = '\n'.join([
    'THE FOUR ENTRIES:',
    f'- {LIVE_B3F} (observations_and_summaries)',
    f'- {DELETED_0D5} (procedural_knowledge), the likeliest deletion target',
])


def gate_5537_metadata(**overrides: Any) -> dict[str, Any]:
    return {
        'mem0_falsified_entries': [LIVE_B3F, LIVE_21F, LIVE_23E, DELETED_0D5],
        'execution_class': 'operational',
        'operational_mode': 'gate',
        'x_enumeration_scope': VALID_SCOPE,
        **overrides,
    }


def _parse_result(result):
    """Parse a FastMCP call_tool result (list of TextContent) into a dict."""
    if isinstance(result, list):
        content = result[0].text if hasattr(result[0], 'text') else str(result[0])
        return json.loads(content)
    return result


@pytest.fixture(autouse=True)
def passthrough_main_checkout(monkeypatch):
    """The tmp project_root is not a git checkout; pass it through unchanged."""
    monkeypatch.setattr(
        'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p),
    )


@pytest.fixture
def project_root(tmp_path) -> str:
    root = tmp_path / 'reify'
    root.mkdir()
    return str(root)


@pytest.fixture
def memory_service() -> AsyncMock:
    """get_memory_by_id resolves the three live entries and misses the rest."""

    async def _get(_project_id: str, memory_id: str) -> dict[str, Any] | None:
        if memory_id in LIVE_IDS:
            return {'id': memory_id, 'content': 'x', 'metadata': {}}
        return None

    service = AsyncMock()
    service.get_memory_by_id = AsyncMock(side_effect=_get)
    return service


@pytest.fixture
def task_interceptor() -> AsyncMock:
    interceptor = AsyncMock()
    interceptor.submit_task = AsyncMock(return_value={'ticket': 'tkt_x'})
    interceptor.update_task = AsyncMock(return_value={'success': True})
    interceptor.get_tasks = AsyncMock(return_value={'tasks': []})
    interceptor.get_task = AsyncMock(return_value={})
    return interceptor


@pytest.fixture
def server(memory_service, task_interceptor):
    return create_mcp_server(memory_service, task_interceptor=task_interceptor)


async def submit(
    server, project_root: str, metadata: dict[str, Any], *,
    agent_id: str = RECON_AGENT, description: str = GATE_DESCRIPTION,
) -> dict[str, Any]:
    return _parse_result(await server._tool_manager.call_tool('submit_task', {
        'project_root': project_root,
        'title': 'Human gate: decide the fate of the falsified entries',
        'description': description,
        'metadata': metadata,
        'agent_id': agent_id,
    }))


def forwarded(interceptor_method: AsyncMock) -> dict[str, Any]:
    interceptor_method.assert_awaited_once()
    return interceptor_method.await_args.kwargs


class TestSubmitTask:

    @pytest.mark.asyncio
    async def test_gate_5537_is_filed_without_the_deleted_entry(
        self, server, project_root, task_interceptor,
    ):
        result = await submit(server, project_root, gate_5537_metadata())

        assert 'error' not in result
        kwargs = forwarded(task_interceptor.submit_task)
        metadata = kwargs['metadata']
        assert metadata['mem0_falsified_entries'] == list(LIVE_IDS)
        assert metadata['x_liveness_exceptions'] == [{
            'memory_id': DELETED_0D5,
            'path': 'mem0_falsified_entries[3]',
            'outcome': 'absent',
        }]
        assert 'x_liveness_reverified_at' in metadata
        deleted_line = kwargs['description'].split('\n')[2]
        assert 'already deleted, verified' in deleted_line
        assert result['enumeration_liveness']['dropped_absent'] == 1
        assert result['enumeration_liveness']['verified'] == 3

    @pytest.mark.asyncio
    async def test_a_truncated_fragment_is_rejected_before_any_lookup(
        self, server, project_root, task_interceptor, memory_service,
    ):
        metadata = gate_5537_metadata(
            mem0_falsified_entries=[LIVE_B3F, LIVE_23E, DELETED_0D5, '21f78de1'],
        )

        result = await submit(server, project_root, metadata)

        assert result['error_type'] == 'EnumeratedIdTruncated'
        task_interceptor.submit_task.assert_not_awaited()
        memory_service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_enumerating_gate_without_a_scope_is_rejected(
        self, server, project_root, task_interceptor,
    ):
        metadata = gate_5537_metadata()
        del metadata['x_enumeration_scope']

        result = await submit(server, project_root, metadata)

        assert result['error_type'] == 'EnumerationScopeMissing'
        task_interceptor.submit_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_non_recon_caller_reaches_the_interceptor_unchanged(
        self, server, project_root, task_interceptor, memory_service,
    ):
        result = await submit(
            server, project_root, gate_5537_metadata(),
            agent_id='claude-interactive',
        )

        assert 'enumeration_liveness' not in result
        kwargs = forwarded(task_interceptor.submit_task)
        assert kwargs['description'] == GATE_DESCRIPTION
        assert kwargs['metadata']['mem0_falsified_entries'] == [
            LIVE_B3F, LIVE_21F, LIVE_23E, DELETED_0D5,
        ]
        assert 'x_liveness_reverified_at' not in kwargs['metadata']
        memory_service.get_memory_by_id.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_lookups_use_the_project_id_of_project_root(
        self, server, project_root, memory_service,
    ):
        await submit(server, project_root, gate_5537_metadata())

        project_ids = {
            call.args[0] for call in memory_service.get_memory_by_id.await_args_list
        }
        assert project_ids == {'reify'}
