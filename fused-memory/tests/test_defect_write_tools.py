"""The defect-record MCP tools: declare_defect, record_defect_outcome, lookup_defect_ref.

The fused-memory half of the defect-tests round trip (``plans/defect-tests-prd.md``
B17): an event written through a tool reads back through ``get_task``, every
refusal arrives as a structured dict, and update_task cannot stand in for the
two writers. Driven over a real SQLite store behind the real interceptor and
MCP server.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from shared.defects import (
    Declare,
    DeclareClosure,
    DeclareEvent,
    DefectDeclaration,
    DefectOrigin,
    DefectProof,
    DefectStatus,
    Exemption,
    ExemptionReason,
    MalformedDefectRef,
    NodeId,
    OutcomeEvent,
    Proof,
    ProofOutcome,
    Rule,
    RuledBy,
    Ruling,
    RulingSubject,
    SideOutcome,
    SideResult,
    Withdrawal,
    WithdrawalCode,
    WithdrawalReason,
    mint_explore_ref,
    mint_human_ref,
    parse_ref,
)
from shared.governed_exceptions import Debt, TaskRef

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.tools import create_mcp_server

_REF = mint_human_ref(datetime(2026, 10, 18, 12, 34, 56, tzinfo=UTC), 'Some summary')
_OTHER_REF = mint_explore_ref('run-1', 4)


def _declaration(ref: str = _REF) -> DefectDeclaration:
    origin = DefectOrigin.HUMAN if ref == _REF else DefectOrigin.EXPLORATORY
    return DefectDeclaration(
        ref=ref,
        origin=origin,
        summary='Some summary',
        declared_at=datetime(2026, 10, 18, 12, 0, tzinfo=UTC),
        declared_by='leo',
        status=DefectStatus.OPEN,
    )


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


def _event_tags(union: Any) -> list[str]:
    members = get_args(get_args(union)[0])
    return [member.model_fields['event'].default.value for member in members]


_DECLARE = _json(Declare(declaration=_declaration()))
_PROOF = _json(Proof(proof=_proof()))
_EXEMPTION_CLOSURE = _json(
    DeclareClosure(
        ref=_REF,
        closure=Exemption(reason=ExemptionReason.NEEDS_EXTERNAL_SYSTEM, owner=Debt(TaskRef(5602))),
    )
)
_OPERATOR_ACCEPTS = _json(
    Rule(
        ref=_REF,
        ruling=Ruling(subject=RulingSubject.CLOSURE, accepted=True, ruled_by=RuledBy.OPERATOR),
    )
)


def _git_init_root(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(path)], check=True, capture_output=True)
    return str(path.resolve())


@dataclass(frozen=True)
class ToolStack:
    root: str
    server: Any
    seed: Callable[..., Awaitable[str]]

    async def call(self, tool: str, **args: Any) -> dict[str, Any]:
        return await self.server._tool_manager.call_tool(tool, {'project_root': self.root, **args})

    async def defects(self, task_id: str) -> list[dict[str, Any]] | None:
        task = await self.call('get_task', id=task_id)
        return task['metadata'].get('defects')


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[ToolStack]:
    root = _git_init_root(tmp_path / 'project')
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=root))
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    interceptor = TaskInterceptor(backend, None, event_buffer)
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    seeded = 0

    async def seed(status: str = 'pending', metadata: dict[str, Any] | None = None) -> str:
        nonlocal seeded
        seeded += 1
        added = await backend.add_task(
            project_root=root,
            title=f'task {seeded}',
            description=f'seeded defect tool fixture task {seeded}',
            metadata=json.dumps(metadata) if metadata is not None else None,
            status=status,
        )
        return str(added['id'])

    try:
        yield ToolStack(root, server, seed)
    finally:
        await interceptor.close()
        await backend.close()
        await event_buffer.close()


# --- (a) the round trip -----------------------------------------------------------


@pytest.mark.asyncio
async def test_a_declared_defect_reads_back_through_get_task(stack: ToolStack) -> None:
    task_id = await stack.seed()

    reply = await stack.call('declare_defect', task_id=task_id, event=_DECLARE)

    assert reply['success'] is True
    assert reply['defects'] == [_json(_declaration())]
    assert reply['project_root'] == stack.root
    assert await stack.defects(task_id) == [_json(_declaration())]


# --- (b) each writer carries only its own events ------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('event', [_PROOF, _OPERATOR_ACCEPTS], ids=['proof', 'rule'])
async def test_declare_defect_refuses_an_outcome_event(
    stack: ToolStack, event: dict[str, Any]
) -> None:
    task_id = await stack.seed()

    reply = await stack.call('declare_defect', task_id=task_id, event=event)

    assert reply['error'] == 'defect_event_invalid'
    for tag in _event_tags(DeclareEvent):
        assert tag in reply['detail']
    assert await stack.defects(task_id) is None


@pytest.mark.asyncio
async def test_record_defect_outcome_refuses_a_declare_event(stack: ToolStack) -> None:
    task_id = await stack.seed()

    reply = await stack.call('record_defect_outcome', task_id=task_id, ref=_REF, event=_DECLARE)

    assert reply['error'] == 'defect_event_invalid'
    for tag in _event_tags(OutcomeEvent):
        assert tag in reply['detail']
    assert await stack.defects(task_id) is None


# --- (c) outcomes ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_proven_proof_proves_the_entry(stack: ToolStack) -> None:
    task_id = await stack.seed(metadata={'defects': [_json(_declaration())]})

    reply = await stack.call('record_defect_outcome', task_id=task_id, ref=_REF, event=_PROOF)

    assert reply['success'] is True
    [entry] = await stack.defects(task_id) or []
    assert entry['status'] == DefectStatus.PROVEN.value


@pytest.mark.asyncio
async def test_an_outcome_for_another_ref_is_refused(stack: ToolStack) -> None:
    task_id = await stack.seed(metadata={'defects': [_json(_declaration())]})

    reply = await stack.call(
        'record_defect_outcome', task_id=task_id, ref=_OTHER_REF, event=_PROOF
    )

    assert reply['error'] == 'defect_ref_mismatch'
    assert reply['ref'] == _OTHER_REF
    assert await stack.defects(task_id) == [_json(_declaration())]


@pytest.mark.asyncio
async def test_an_operator_ruling_exempts_a_declared_closure(stack: ToolStack) -> None:
    task_id = await stack.seed(metadata={'defects': [_json(_declaration())]})
    await stack.call('declare_defect', task_id=task_id, event=_EXEMPTION_CLOSURE)

    reply = await stack.call(
        'record_defect_outcome', task_id=task_id, ref=_REF, event=_OPERATOR_ACCEPTS
    )

    assert reply['success'] is True
    [entry] = await stack.defects(task_id) or []
    assert entry['status'] == DefectStatus.EXEMPTED.value


# --- (d)/(e) refusals from the write path ------------------------------------------


@pytest.mark.asyncio
async def test_a_terminal_task_is_refused(stack: ToolStack) -> None:
    task_id = await stack.seed(status='done')

    reply = await stack.call('declare_defect', task_id=task_id, event=_DECLARE)

    assert reply['error'] == 'defect_write_to_terminal_task'
    assert reply['status'] == 'done'
    assert reply['task_id'] == task_id


@pytest.mark.asyncio
async def test_a_duplicate_of_an_undeclared_ref_is_refused(stack: ToolStack) -> None:
    task_id = await stack.seed(metadata={'defects': [_json(_declaration())]})
    reason = WithdrawalReason(reason=WithdrawalCode.DUPLICATE_OF, ref=_OTHER_REF)
    closure = DeclareClosure(ref=_REF, closure=Withdrawal(reason=reason))

    reply = await stack.call('declare_defect', task_id=task_id, event=_json(closure))

    assert reply['error'] == 'defect_ref_not_live'
    assert reply['ref'] == _OTHER_REF
    assert reply['relation'] == 'duplicate_of'


# --- (f) the lookup ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_defect_ref_returns_the_carriers(stack: ToolStack) -> None:
    pending = await stack.seed(metadata={'defects': [_json(_declaration())]})
    done = await stack.seed(status='done', metadata={'defects': [_json(_declaration())]})

    reply = await stack.call('lookup_defect_ref', ref=_REF)

    assert reply['live'] is True
    assert reply['carriers'] == [
        {'task_id': int(pending), 'task_status': 'pending', 'defect_status': 'open'},
        {'task_id': int(done), 'task_status': 'done', 'defect_status': 'open'},
    ]


@pytest.mark.asyncio
async def test_lookup_of_a_malformed_ref_names_the_known_prefixes(stack: ToolStack) -> None:
    with pytest.raises(MalformedDefectRef) as grammar:
        parse_ref('bogus:1')

    reply = await stack.call('lookup_defect_ref', ref='bogus:1')

    assert reply == {'error': str(grammar.value), 'error_type': 'ValidationError'}


# --- (g) update_task cannot stand in for the writers --------------------------------


@pytest.mark.asyncio
async def test_the_update_task_tool_refuses_the_defect_keys(stack: ToolStack) -> None:
    task_id = await stack.seed()

    reply = await stack.call(
        'update_task', id=task_id, metadata={'defects': [_json(_declaration())]}
    )

    assert reply['error'] == 'defect_records_via_update_task'
    assert reply['key'] == 'defects'
    assert await stack.defects(task_id) is None


# --- (h) ticket-shaped ids ------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('tool', 'args'),
    [
        ('declare_defect', {'event': _DECLARE}),
        ('record_defect_outcome', {'ref': _REF, 'event': _PROOF}),
    ],
)
async def test_a_ticket_shaped_task_id_is_refused(
    stack: ToolStack, tool: str, args: dict[str, Any]
) -> None:
    reply = await stack.call(tool, task_id='tkt_0RSM2ECXBS1RPHMGYQVZ0V3QZZ', **args)

    assert reply['error_type'] == 'ValidationError'
    assert 'resolve_ticket' in reply['error']


# --- (i) the process registers the defect submodels through its own imports ----------

_REGISTRATION_PROBE = r'''
import json, sys
import fused_memory.server.tools
from shared.task_metadata import parse_metadata
entry = json.loads(sys.argv[1])
_, clean = parse_metadata({'defects': [entry]}, direction='write')
_, malformed = parse_metadata({'defects': [{**entry, 'status': 'bogus'}]}, direction='write')
print(json.dumps({
    'clean': [[w.field, w.code] for w in clean],
    'malformed': [[w.field, w.code] for w in malformed],
}))
'''


def test_importing_the_tool_server_registers_the_defect_submodels() -> None:
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join(p for p in sys.path if p)}
    completed = subprocess.run(
        [sys.executable, '-c', _REGISTRATION_PROBE, json.dumps(_json(_declaration()))],
        capture_output=True,
        text=True,
        check=True,
        env=env,
        timeout=300,
    )
    probe = json.loads(completed.stdout.strip().splitlines()[-1])

    assert probe['clean'] == []
    defect_codes = [code for field, code in probe['malformed'] if field == 'defects']
    assert defect_codes, probe
    assert 'unknown_key' not in defect_codes
