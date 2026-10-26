"""update_task never moves a task's ``defects`` or ``defect_test_retirements``, for any caller.

After creation, ``declare_defect`` and ``record_defect_outcome`` are the only
writers of both keys (``plans/defect-tests-prd.md`` decision 7). update_task
refuses either key outside ``metadata_mode='replace'``, echo included, and
under replace admits it only as a verbatim carry of the stored value. Driven
over a real SqliteTaskBackend behind a real TaskInterceptor.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
    DefectDeclaration,
    DefectOrigin,
    DefectStatus,
    Retirement,
    RetirementCode,
    RetirementReason,
    mint_explore_ref,
    mint_human_ref,
)
from shared.project_layout import tasks_db

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_backend_errors import (
    DefectRecordsWriteAuthorityError,
    DoneProvenanceWriteAuthorityError,
)
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

_REF = mint_human_ref(datetime(2026, 10, 18, 12, 34, 56, tzinfo=UTC), 'Some summary')
_OTHER_REF = mint_explore_ref('run-1', 4)
_DEFECT_KEYS = [DEFECTS_METADATA_KEY, DEFECT_TEST_RETIREMENTS_METADATA_KEY]


def _declaration_json(ref: str = _REF) -> dict[str, Any]:
    origin = DefectOrigin.HUMAN if ref == _REF else DefectOrigin.EXPLORATORY
    return DefectDeclaration(
        ref=ref,
        origin=origin,
        summary='Some summary',
        declared_at=datetime(2026, 10, 18, 12, 0, tzinfo=UTC),
        declared_by='leo',
        status=DefectStatus.OPEN,
    ).model_dump(mode='json')


def _retirement_json() -> dict[str, Any]:
    reason = RetirementReason(reason=RetirementCode.FEATURE_REMOVED)
    return Retirement(ref=_OTHER_REF, reason=reason).model_dump(mode='json')


_SEEDED = {
    'files': ['a.py'],
    'x_note': 'seeded',
    DEFECTS_METADATA_KEY: [_declaration_json()],
    DEFECT_TEST_RETIREMENTS_METADATA_KEY: [_retirement_json()],
}

_CHANGED = {
    DEFECTS_METADATA_KEY: [{**_declaration_json(), 'sightings': 2}],
    DEFECT_TEST_RETIREMENTS_METADATA_KEY: [],
}


@dataclass(frozen=True)
class FloorStack:
    root: str
    backend: SqliteTaskBackend
    interceptor: TaskInterceptor
    task_id: str

    async def metadata(self, task_id: str | None = None) -> dict[str, Any]:
        task = await self.backend.get_task(task_id or self.task_id, self.root)
        return json.loads(json.dumps(task['metadata']))

    def stored_bytes(self, task_id: str | None = None) -> str | None:
        with contextlib.closing(sqlite3.connect(tasks_db(self.root))) as conn:
            row = conn.execute(
                'SELECT metadata FROM tasks WHERE id = ?', (int(task_id or self.task_id),)
            ).fetchone()
        return row[0]

    async def update(self, metadata: dict[str, Any], **kwargs: Any) -> Any:
        return await self.backend.update_task(
            self.task_id, self.root, metadata=json.dumps(metadata), **kwargs
        )


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[FloorStack]:
    root = str(tmp_path)
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=root))
    await backend.start()
    unique = uuid.uuid4().hex[:8]
    event_buffer = EventBuffer(db_path=tmp_path / f'events_{unique}.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    store = TicketStore(tmp_path / f'tickets_{unique}.db')
    await store.initialize()
    interceptor = TaskInterceptor(backend, None, event_buffer, config=None, ticket_store=store)
    added = await backend.add_task(
        project_root=root, title='t', description='d', metadata=json.dumps(_SEEDED),
    )
    try:
        yield FloorStack(root, backend, interceptor, str(added['id']))
    finally:
        await interceptor.close()
        await backend.close()
        await event_buffer.close()


def _assert_refusal_shape(wire: dict[str, Any], task_id: str, key: str) -> None:
    hint = wire.pop('hint')
    assert wire == {
        'success': False,
        'error': 'defect_records_via_update_task',
        'task_id': task_id,
        'key': key,
    }
    assert isinstance(hint, str)
    assert 'declare_defect' in hint
    assert 'record_defect_outcome' in hint


# --- (a) outside replace, the key is refused on presence ---------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('key', _DEFECT_KEYS)
@pytest.mark.parametrize('value_source', [_SEEDED, _CHANGED], ids=['echo', 'change'])
@pytest.mark.parametrize(
    'mode',
    [{}, {'metadata_mode': 'merge'}, {'append': True}, {'metadata_mode': 'additive'}],
    ids=['default', 'merge', 'append', 'additive'],
)
async def test_update_task_refuses_a_defect_key_outside_replace(
    stack: FloorStack, key: str, value_source: dict[str, Any], mode: dict[str, Any]
) -> None:
    before = stack.stored_bytes()

    with pytest.raises(DefectRecordsWriteAuthorityError) as excinfo:
        await stack.update({key: value_source[key], 'x_note': 'edited'}, **mode)

    assert excinfo.value.key == key
    _assert_refusal_shape(excinfo.value.to_error_dict(), stack.task_id, key)
    assert stack.stored_bytes() == before


# --- (b) under replace, only a verbatim carry is admitted --------------------------


@pytest.mark.asyncio
async def test_a_replace_carrying_the_stored_records_verbatim_is_admitted(
    stack: FloorStack,
) -> None:
    blob = {**await stack.metadata(), 'x_note': 'edited'}

    await stack.update(blob, metadata_mode='replace')

    metadata = await stack.metadata()
    assert metadata['x_note'] == 'edited'
    assert metadata[DEFECTS_METADATA_KEY] == _SEEDED[DEFECTS_METADATA_KEY]
    assert metadata[DEFECT_TEST_RETIREMENTS_METADATA_KEY] == (
        _SEEDED[DEFECT_TEST_RETIREMENTS_METADATA_KEY]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('key', _DEFECT_KEYS)
async def test_a_replace_changing_a_defect_key_is_refused(stack: FloorStack, key: str) -> None:
    blob = {**await stack.metadata(), key: _CHANGED[key]}
    before = stack.stored_bytes()

    with pytest.raises(DefectRecordsWriteAuthorityError) as excinfo:
        await stack.update(blob, metadata_mode='replace')

    assert excinfo.value.key == key
    assert stack.stored_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize('key', _DEFECT_KEYS)
async def test_a_replace_dropping_a_defect_key_by_omission_is_refused(
    stack: FloorStack, key: str
) -> None:
    blob = {k: v for k, v in (await stack.metadata()).items() if k != key}
    before = stack.stored_bytes()

    with pytest.raises(DefectRecordsWriteAuthorityError) as excinfo:
        await stack.update(blob, metadata_mode='replace')

    assert excinfo.value.key == key
    assert stack.stored_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize('key', _DEFECT_KEYS)
async def test_a_replace_adding_a_defect_key_to_a_task_without_one_is_refused(
    stack: FloorStack, key: str
) -> None:
    added = await stack.backend.add_task(
        project_root=stack.root, title='bare', description='d', metadata=json.dumps({'files': []}),
    )
    bare = str(added['id'])
    blob = {**await stack.metadata(bare), key: _SEEDED[key]}
    before = stack.stored_bytes(bare)

    with pytest.raises(DefectRecordsWriteAuthorityError) as excinfo:
        await stack.backend.update_task(
            bare, stack.root, metadata=json.dumps(blob), metadata_mode='replace'
        )

    assert excinfo.value.key == key
    assert stack.stored_bytes(bare) == before


# --- (c) done_provenance keeps its precedence --------------------------------------


@pytest.mark.asyncio
async def test_done_provenance_is_refused_first_when_a_payload_carries_both(
    stack: FloorStack,
) -> None:
    payload = {
        'done_provenance': {'kind': 'merged', 'commit': 'a' * 40},
        DEFECTS_METADATA_KEY: _SEEDED[DEFECTS_METADATA_KEY],
    }
    with pytest.raises(DoneProvenanceWriteAuthorityError):
        await stack.update(payload)


# --- (d) no caller is exempt ----------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('agent_id', ['recon-stage-2', 'orchestrator-7044', None])
async def test_the_interceptor_returns_the_refusal_for_every_caller(
    stack: FloorStack, agent_id: str | None
) -> None:
    before = stack.stored_bytes()

    wire = await stack.interceptor.update_task(
        stack.task_id,
        stack.root,
        agent_id=agent_id,
        metadata=json.dumps({DEFECTS_METADATA_KEY: _SEEDED[DEFECTS_METADATA_KEY]}),
    )

    _assert_refusal_shape(dict(wire), stack.task_id, DEFECTS_METADATA_KEY)
    assert stack.stored_bytes() == before


# --- (e) the audit-trail rotation inherits the floor -----------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('key', _DEFECT_KEYS)
async def test_rewrite_audit_trail_refuses_a_changed_defect_key(
    stack: FloorStack, key: str
) -> None:
    blob = {**await stack.metadata(), key: _CHANGED[key]}
    before = stack.stored_bytes()

    with pytest.raises(DefectRecordsWriteAuthorityError) as excinfo:
        await stack.backend.rewrite_audit_trail(
            stack.task_id, stack.root, description=None, metadata=blob
        )

    assert excinfo.value.key == key
    assert stack.stored_bytes() == before


@pytest.mark.asyncio
async def test_rewrite_audit_trail_admits_the_records_carried_verbatim(
    stack: FloorStack,
) -> None:
    blob = {**await stack.metadata(), 'x_note': 'rotated'}

    await stack.backend.rewrite_audit_trail(
        stack.task_id, stack.root, description=None, metadata=blob
    )

    metadata = await stack.metadata()
    assert metadata['x_note'] == 'rotated'
    assert metadata[DEFECTS_METADATA_KEY] == _SEEDED[DEFECTS_METADATA_KEY]
