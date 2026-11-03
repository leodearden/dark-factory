"""The store-identity seam's boundary rows, read as an agent receives them.

Rows B1-B7 of ``plans/escalation-store-ambiguity-prd.md`` §6.5, each driven
through a real MCP client session and never a tool's ``fn``: the in-memory
``fastmcp.Client``, and real HTTP for B5, the agent on 8103. B8, the
reconciliation-stage consumer side, is
``fused-memory/tests/reconciliation/test_recon_stage_escalation_read_boundary.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import _project_root_stores as stores
import pytest
from _pending_tool_fixtures import _file
from fastmcp import Client, FastMCP
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import TextContent

from escalation.queue import EscalationQueue
from escalation.server import create_server
from escalation.store_identity import StoreIdentity

TASK = '5534'
READS = ('get_pending_escalations', 'get_escalation', 'get_task_escalation_history')
WRITES = ('escalate_info', 'escalate_blocker')
GUARDED = READS + WRITES
_FILING = {
    'task_id': TASK,
    'agent_role': 'implementer',
    'category': 'design_concern',
    'summary': 's',
}


def arguments(name: str, escalation_id: str) -> dict[str, Any]:
    table: dict[str, dict[str, Any]] = {
        'get_pending_escalations': {'task_id': TASK},
        'get_escalation': {'escalation_id': escalation_id},
        'get_task_escalation_history': {'task_id': TASK},
        'escalate_info': _FILING,
        'escalate_blocker': _FILING,
    }
    return dict(table[name])


def pending_ids(queue: EscalationQueue) -> list[str]:
    return sorted(e.id for e in queue.get_pending())


def listed(result: CallToolResult) -> list[dict[str, Any]]:
    assert result.structured_content is not None
    records = result.structured_content['result']
    assert isinstance(records, list)
    return records


def wire_text(result: CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


def refusal(result: CallToolResult) -> str:
    assert isinstance(result.data, dict)
    assert set(result.data) == {'error'}
    return result.data['error']


async def refused_pending_read(client: Client, asserted: Path) -> None:
    refusal(
        await client.call_tool(
            'get_pending_escalations', {'task_id': TASK, 'project_root': str(asserted)},
        )
    )


@pytest.fixture
def reify_root(tmp_path: Path) -> Path:
    return stores.make_root(tmp_path, 'reify')


@pytest.fixture
def df_root(tmp_path: Path) -> Path:
    return stores.make_root(tmp_path, 'dark-factory')


@pytest.fixture
def queue(reify_root: Path) -> EscalationQueue:
    return stores.project_queue(reify_root)


@pytest.fixture
def reify_identity(queue: EscalationQueue, reify_root: Path) -> StoreIdentity:
    return stores.project_identity(queue, reify_root, project_id='reify')


@pytest.fixture
def reify_server(queue: EscalationQueue, reify_identity: StoreIdentity) -> FastMCP:
    return create_server(queue, startup_sweep=False, store_identity=reify_identity)


@pytest.fixture
def recon_queue(tmp_path: Path) -> EscalationQueue:
    return stores.recon_queue(tmp_path)


@pytest.fixture
def recon_identity(recon_queue: EscalationQueue) -> StoreIdentity:
    return stores.recon_identity(recon_queue)


@pytest.fixture
def recon_server(recon_queue: EscalationQueue, recon_identity: StoreIdentity) -> FastMCP:
    return create_server(recon_queue, startup_sweep=False, store_identity=recon_identity)


@pytest.mark.asyncio
@pytest.mark.parametrize('name', GUARDED)
async def test_b1_mismatched_assertion_is_loud(
    name: str,
    reify_server: FastMCP,
    queue: EscalationQueue,
    reify_root: Path,
    df_root: Path,
) -> None:
    seeded = _file(queue, TASK)
    before = pending_ids(queue)

    async with Client(reify_server) as client:
        refused = await client.call_tool(
            name, {**arguments(name, seeded.id), 'project_root': str(df_root)},
        )
        control = await client.call_tool('get_pending_escalations', {'task_id': TASK})

    error = refusal(refused)
    assert 'project_root mismatch' in error
    assert str(df_root) in error
    assert str(reify_root) in error
    assert pending_ids(queue) == before
    assert [e['id'] for e in listed(control)] == [seeded.id]


@pytest.mark.asyncio
@pytest.mark.parametrize('name', GUARDED)
async def test_b2_recon_server_names_itself(
    name: str,
    recon_server: FastMCP,
    recon_queue: EscalationQueue,
    reify_root: Path,
    df_root: Path,
) -> None:
    seeded = _file(recon_queue, TASK, agent_role='reconciliation-stage1')
    before = pending_ids(recon_queue)

    async with Client(recon_server) as client:
        for asserted in (df_root, reify_root):
            refused = await client.call_tool(
                name, {**arguments(name, seeded.id), 'project_root': str(asserted)},
            )

            error = refusal(refused)
            assert 'reconciliation store' in error, asserted
            assert str(recon_queue.queue_dir) in error, asserted
            assert 'project_root mismatch' not in error, asserted

    assert pending_ids(recon_queue) == before


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ('project', 'reconciliation'))
@pytest.mark.parametrize('name', READS)
async def test_b3_omitted_assertion_is_byte_identical(
    kind: str,
    name: str,
    queue: EscalationQueue,
    reify_identity: StoreIdentity,
    recon_queue: EscalationQueue,
    recon_identity: StoreIdentity,
) -> None:
    store, identity = (queue, reify_identity) if kind == 'project' else (
        recon_queue, recon_identity,
    )
    seeded = _file(store, TASK)

    answers: list[CallToolResult] = []
    for served_identity in (identity, None):
        server = create_server(store, startup_sweep=False, store_identity=served_identity)
        async with Client(server) as client:
            answers.append(await client.call_tool(name, arguments(name, seeded.id)))

    with_identity, without_identity = answers
    assert seeded.id in wire_text(with_identity)
    assert wire_text(with_identity) == wire_text(without_identity)
    assert with_identity.structured_content == without_identity.structured_content


@pytest.mark.asyncio
async def test_b4_description_carries_identity(
    recon_server: FastMCP, recon_identity: StoreIdentity,
) -> None:
    async with Client(recon_server) as client:
        tools = await client.list_tools()

    assert 'get_pending_escalations' in {tool.name for tool in tools}
    for tool in tools:
        description = tool.description or ''
        assert str(recon_identity.queue_dir) in description, tool.name
        assert 'kind=reconciliation' in description, tool.name


def test_b5_empty_pending_read_is_attributable_to_its_store(
    tmp_path: Path, serve_escalation_mcp: Callable[..., tuple[str, int, EscalationQueue]],
) -> None:
    seeding = stores.recon_queue(tmp_path)
    identity = stores.recon_identity(seeding)
    stage_record = _file(seeding, 'recon-run-1', agent_role='reconciliation-stage1')
    base_url, _port, _queue = serve_escalation_mcp(seeding.queue_dir, store_identity=identity)

    async def agent_on_8103() -> tuple[CallToolResult, CallToolResult, dict[str, str]]:
        async with Client(StreamableHttpTransport(f'{base_url}/mcp/')) as client:
            everything = await client.call_tool('get_pending_escalations', {})
            for_task = await client.call_tool('get_pending_escalations', {'task_id': TASK})
            tools = await client.list_tools()
        return everything, for_task, {tool.name: tool.description or '' for tool in tools}

    everything, for_task, descriptions = asyncio.run(agent_on_8103())

    assert [e['id'] for e in listed(everything)] == [stage_record.id]
    assert listed(for_task) == []
    description = descriptions['get_pending_escalations']
    assert str(identity.queue_dir) in description
    assert 'kind=reconciliation' in description
    assert 'get_task_escalation_history' in description


@pytest.mark.asyncio
async def test_b6_history_answers_what_pending_cannot(
    reify_server: FastMCP, queue: EscalationQueue,
) -> None:
    gate = _file(queue, TASK, level=2, agent_role='deterministic')
    queue.resolve(gate.id, 'Human reviewed the gate')
    assert len(list(queue.queue_dir.glob(f'archive/*/{gate.id}.json'))) == 1

    async with Client(reify_server) as client:
        pending = await client.call_tool('get_pending_escalations', {'task_id': TASK})
        history = await client.call_tool('get_task_escalation_history', {'task_id': TASK})

    assert listed(pending) == []
    assert history.data['count'] == 1
    record = history.data['escalations'][0]
    assert (record['id'], record['status'], record['level']) == (gate.id, 'resolved', 2)


@pytest.mark.asyncio
async def test_b7_mismatch_storm_files_exactly_one_l1(
    reify_server: FastMCP, queue: EscalationQueue, df_root: Path,
) -> None:
    async with Client(reify_server) as client:
        for _ in range(4):
            await refused_pending_read(client, df_root)
        assert stores.storm_records(queue) == []

        await refused_pending_read(client, df_root)
        storms = stores.storm_records(queue)
        assert len(storms) == 1
        assert str(df_root) in storms[0].summary
        assert str(queue.queue_dir) in storms[0].summary

        await refused_pending_read(client, df_root)
        assert len(stores.storm_records(queue)) == 1

    assert list(df_root.iterdir()) == []
