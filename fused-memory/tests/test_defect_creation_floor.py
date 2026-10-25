"""A creating write may carry the defect keys only as fresh records, refused synchronously.

The floor sits at ``TaskInterceptor.submit_task``, which both creation paths
(the ticket path and ``planning_mode``) pass before anything is persisted, and
is repeated at ``SqliteTaskBackend.add_task`` for creators that call the
backend directly. Driven over a real backend, ticket store and interceptor.
"""

from __future__ import annotations

import json
import subprocess
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
    DefectDeclaration,
    DefectOrigin,
    DefectProof,
    DefectStatus,
    NodeId,
    ProofOutcome,
    Retirement,
    RetirementCode,
    RetirementReason,
    RuledBy,
    Ruling,
    RulingSubject,
    SideOutcome,
    SideResult,
    mint_explore_ref,
    mint_human_ref,
)

from fused_memory.backends.defect_records import DefectCreationRefused
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server

_REF = mint_human_ref(datetime(2026, 10, 18, 12, 34, 56, tzinfo=UTC), 'Some summary')
_OTHER_REF = mint_explore_ref('run-1', 4)


def _declaration(ref: str = _REF, **overrides: Any) -> DefectDeclaration:
    origin = DefectOrigin.HUMAN if ref == _REF else DefectOrigin.EXPLORATORY
    fields: dict[str, Any] = {
        'ref': ref,
        'origin': origin,
        'summary': 'Some summary',
        'declared_at': datetime(2026, 10, 18, 12, 0, tzinfo=UTC),
        'declared_by': 'leo',
        'status': DefectStatus.OPEN,
    }
    return DefectDeclaration(**{**fields, **overrides})


def _proof() -> DefectProof:
    node = NodeId(module_root='fused-memory', path='tests/test_x.py', name='test_reproduces')
    return DefectProof(
        ref=_REF,
        outcome=ProofOutcome.PROVEN,
        base_sha='a' * 40,
        tip_sha='b' * 40,
        measured_at=datetime(2026, 10, 18, 13, 0, tzinfo=UTC),
        node_ids=(node,),
        base=(SideResult(node_id=node, outcome=SideOutcome.FAILED),),
        tip=(SideResult(node_id=node, outcome=SideOutcome.PASSED),),
        duration_s=1.0,
    )


def _json(record: Any) -> dict[str, Any]:
    return record.model_dump(mode='json')


_PROVEN = _json(_declaration(status=DefectStatus.PROVEN, proof=_proof()))
_FRESH = _json(_declaration())
_RULED_RETIREMENT = _json(
    Retirement(
        ref=_REF,
        reason=RetirementReason(reason=RetirementCode.FEATURE_REMOVED),
        ruling=Ruling(subject=RulingSubject.RETIREMENT, accepted=True, ruled_by=RuledBy.OPERATOR),
    )
)


@dataclass(frozen=True)
class CreationStack:
    root: str
    backend: SqliteTaskBackend
    interceptor: TaskInterceptor
    store: TicketStore
    server: Any

    async def submit(self, metadata: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return await self.interceptor.submit_task(
            self.root, title='a defect fix', description='d', metadata=metadata, **kwargs
        )

    async def task_ids(self) -> list[str]:
        return [task['id'] for task in (await self.backend.get_tasks(self.root))['tasks']]

    async def tickets(self) -> list[dict[str, Any]]:
        return await self.store.list_tickets(resolve_project_id(self.root))


def _git_init_root(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(path)], check=True, capture_output=True)
    return str(path.resolve())


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[CreationStack]:
    root = _git_init_root(tmp_path / 'project')
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=root))
    await backend.start()
    unique = uuid.uuid4().hex[:8]
    event_buffer = EventBuffer(db_path=tmp_path / f'events_{unique}.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    store = TicketStore(tmp_path / f'tickets_{unique}.db')
    await store.initialize()
    interceptor = TaskInterceptor(backend, None, event_buffer, config=None, ticket_store=store)
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    try:
        yield CreationStack(root, backend, interceptor, store, server)
    finally:
        await interceptor.close()
        await backend.close()
        await event_buffer.close()


def _assert_refused(wire: dict[str, Any], key: str, index: int | None) -> None:
    assert wire['success'] is False
    assert wire['error'] == 'defect_creation_refused'
    assert wire['key'] == key
    assert wire['index'] == index
    assert isinstance(wire['detail'], str) and wire['detail']


# --- (a)/(b) both creation paths refuse synchronously, persisting nothing -----------


@pytest.mark.asyncio
async def test_the_ticket_path_refuses_a_proven_entry_before_persisting(
    stack: CreationStack,
) -> None:
    wire = await stack.submit({DEFECTS_METADATA_KEY: [_PROVEN]})

    _assert_refused(wire, DEFECTS_METADATA_KEY, 0)
    assert f'status {DefectStatus.OPEN.value}' in wire['detail']
    assert await stack.tickets() == []
    assert await stack.task_ids() == []


@pytest.mark.asyncio
async def test_planning_mode_refuses_a_proven_entry(stack: CreationStack) -> None:
    wire = await stack.submit({DEFECTS_METADATA_KEY: [_FRESH, _PROVEN]}, planning_mode=True)

    _assert_refused(wire, DEFECTS_METADATA_KEY, 1)
    assert await stack.task_ids() == []


# --- (c) what each refusal names ---------------------------------------------------


@pytest.mark.asyncio
async def test_an_unknown_origin_is_refused_naming_every_origin(stack: CreationStack) -> None:
    wire = await stack.submit(
        {DEFECTS_METADATA_KEY: [{**_FRESH, 'origin': 'bogus'}]}, planning_mode=True
    )

    _assert_refused(wire, DEFECTS_METADATA_KEY, 0)
    for origin in DefectOrigin:
        assert f"'{origin.value}'" in wire['detail']


@pytest.mark.asyncio
async def test_a_ruled_retirement_is_refused(stack: CreationStack) -> None:
    wire = await stack.submit(
        {DEFECT_TEST_RETIREMENTS_METADATA_KEY: [_RULED_RETIREMENT]}, planning_mode=True
    )

    _assert_refused(wire, DEFECT_TEST_RETIREMENTS_METADATA_KEY, 0)
    assert 'unruled' in wire['detail']


@pytest.mark.asyncio
async def test_two_entries_sharing_a_ref_are_refused_naming_it(stack: CreationStack) -> None:
    wire = await stack.submit({DEFECTS_METADATA_KEY: [_FRESH, _FRESH]}, planning_mode=True)

    _assert_refused(wire, DEFECTS_METADATA_KEY, None)
    assert _REF in wire['detail']


# --- (d) the MCP tool returns the same refusal ---------------------------------------


@pytest.mark.asyncio
async def test_the_submit_task_tool_returns_the_refusal(stack: CreationStack) -> None:
    wire = await stack.server._tool_manager.call_tool(
        'submit_task',
        {
            'project_root': stack.root,
            'title': 'a defect fix',
            'description': 'd',
            'metadata': {DEFECTS_METADATA_KEY: [{**_FRESH, 'ref': 'bogus:1'}]},
            'planning_mode': True,
        },
    )

    _assert_refused(wire, DEFECTS_METADATA_KEY, 0)
    assert await stack.task_ids() == []


# --- (e) fresh entries are accepted as declared ------------------------------------


@pytest.mark.asyncio
async def test_fresh_open_entries_are_created(stack: CreationStack) -> None:
    entries = [_FRESH, _json(_declaration(_OTHER_REF))]

    created = await stack.submit({DEFECTS_METADATA_KEY: entries}, planning_mode=True)

    task = await stack.backend.get_task(str(created['task_id']), stack.root)
    assert task['metadata'][DEFECTS_METADATA_KEY] == entries


# --- (f) the backend backstop ----------------------------------------------------------


@pytest.mark.asyncio
async def test_add_task_called_directly_refuses_a_malformed_entry(stack: CreationStack) -> None:
    with pytest.raises(DefectCreationRefused) as excinfo:
        await stack.backend.add_task(
            project_root=stack.root,
            title='direct',
            description='d',
            metadata=json.dumps({DEFECTS_METADATA_KEY: [{**_FRESH, 'sightings': 2}]}),
        )

    _assert_refused(excinfo.value.to_error_dict(), DEFECTS_METADATA_KEY, 0)
    assert await stack.task_ids() == []
