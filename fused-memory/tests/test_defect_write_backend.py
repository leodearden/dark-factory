"""SqliteTaskBackend.apply_defect_events and lookup_defect_ref over a real SQLite task store.

The backend core is the one post-creation writer of a task's defect keys; the
lookup answers which tasks declare a ref. Every case arranges its store
through the backend's public writes, or through the
``shared.project_layout.tasks_db`` file for states no writer produces.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from shared.defects import (
    Declare,
    DeclareClosure,
    DeclareRetirement,
    DefectDeclaration,
    DefectOrigin,
    DefectProof,
    DefectStatus,
    MalformedDefectRef,
    NodeId,
    Proof,
    ProofOutcome,
    Retirement,
    RetirementCode,
    RetirementReason,
    SideOutcome,
    SideResult,
    Sighting,
    Withdrawal,
    WithdrawalCode,
    WithdrawalReason,
    mint_explore_ref,
    mint_human_ref,
)
from shared.project_layout import tasks_db

from fused_memory.backends.defect_records import (
    DefectRefNotLive,
    DefectTaskTerminal,
    DefectTransitionRefused,
    StoredDefectRecordsMalformed,
)
from fused_memory.backends.defect_ref_lookup import (
    DefectRefCarrier,
    lookup_defect_ref,
)
from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_backend_errors import TaskNotFoundError
from fused_memory.config.schema import TaskmasterConfig

_REF = mint_human_ref(datetime(2026, 10, 18, 12, 34, 56, tzinfo=UTC), 'Some summary')
_OTHER_REF = mint_explore_ref('run-1', 4)
_THIRD_REF = mint_explore_ref('run-1', 5)
_DECLARED_AT = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def _declaration(ref: str = _REF) -> DefectDeclaration:
    origin = DefectOrigin.HUMAN if ref == _REF else DefectOrigin.EXPLORATORY
    return DefectDeclaration(
        ref=ref,
        origin=origin,
        summary='Some summary',
        declared_at=_DECLARED_AT,
        declared_by='leo',
        status=DefectStatus.OPEN,
    )


def _proof(ref: str = _REF) -> DefectProof:
    node = NodeId(module_root='fused-memory', path='tests/test_x.py', name='test_reproduces')
    return DefectProof(
        ref=ref,
        outcome=ProofOutcome.PROVEN,
        base_sha='a' * 40,
        tip_sha='b' * 40,
        measured_at=datetime(2026, 10, 18, 13, 0, tzinfo=UTC),
        node_ids=(node,),
        base=(SideResult(node_id=node, outcome=SideOutcome.FAILED),),
        tip=(SideResult(node_id=node, outcome=SideOutcome.PASSED),),
        duration_s=1.0,
    )


def _duplicate_of(ref: str) -> DeclareClosure:
    reason = WithdrawalReason(reason=WithdrawalCode.DUPLICATE_OF, ref=ref)
    return DeclareClosure(ref=_REF, closure=Withdrawal(reason=reason))


def _superseded_by(ref: str) -> DeclareRetirement:
    reason = RetirementReason(reason=RetirementCode.SUPERSEDED_BY, ref=ref)
    return DeclareRetirement(retirement=Retirement(ref=_REF, reason=reason))


def _json(record: Any) -> dict[str, Any]:
    return record.model_dump(mode='json')


def _git_init_root(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(['git', 'init', '-q', str(path)], check=True, capture_output=True)
    return str(path.resolve())


@dataclass(frozen=True)
class BackendStack:
    root: str
    backend: SqliteTaskBackend
    seed: Callable[..., Awaitable[str]]

    async def apply(self, task_id: str, *events: Any) -> dict[str, Any]:
        return dict(await self.backend.apply_defect_events(task_id, self.root, list(events)))

    async def metadata(self, task_id: str) -> dict[str, Any]:
        return (await self.backend.get_task(task_id, self.root))['metadata']

    def stored_bytes(self, task_id: str) -> tuple[str | None, str]:
        with contextlib.closing(sqlite3.connect(tasks_db(self.root))) as conn:
            row = conn.execute(
                'SELECT metadata, updated_at FROM tasks WHERE id = ?', (int(task_id),)
            ).fetchone()
        return row[0], row[1]

    def corrupt(self, task_id: str, raw: str) -> None:
        with contextlib.closing(sqlite3.connect(tasks_db(self.root))) as conn, conn:
            conn.execute('UPDATE tasks SET metadata = ? WHERE id = ?', (raw, int(task_id)))


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[BackendStack]:
    root = _git_init_root(tmp_path / 'project')
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=root))
    await backend.start()
    seeded = 0

    async def seed(status: str = 'pending', metadata: dict[str, Any] | None = None) -> str:
        nonlocal seeded
        seeded += 1
        added = await backend.add_task(
            project_root=root,
            title=f'task {seeded}',
            description=f'seeded defect fixture task {seeded}',
            metadata=json.dumps(metadata) if metadata is not None else None,
            status=status,
        )
        return str(added['id'])

    try:
        yield BackendStack(root, backend, seed)
    finally:
        await backend.close()


# --- (a)/(b) writes land, merged per key ---------------------------------------


@pytest.mark.asyncio
async def test_declare_writes_the_entry_and_keeps_every_other_key(stack: BackendStack) -> None:
    task_id = await stack.seed(metadata={'files': ['a.py'], 'x_note': 'kept'})

    result = await stack.apply(task_id, Declare(declaration=_declaration()))

    metadata = await stack.metadata(task_id)
    assert metadata['defects'] == [_json(_declaration())]
    assert metadata['files'] == ['a.py']
    assert metadata['x_note'] == 'kept'
    assert 'defect_test_retirements' not in metadata
    assert result['id'] == task_id
    assert result['defects'] == [_json(_declaration())]
    assert result['defect_test_retirements'] == []


@pytest.mark.asyncio
async def test_a_proof_in_a_later_call_proves_the_entry(stack: BackendStack) -> None:
    task_id = await stack.seed()
    await stack.apply(task_id, Declare(declaration=_declaration()))

    await stack.apply(task_id, Proof(proof=_proof()))

    [entry] = (await stack.metadata(task_id))['defects']
    assert entry['status'] == DefectStatus.PROVEN.value
    assert entry['proof'] == _json(_proof())


# --- (c) one write unit, all or nothing -----------------------------------------


@pytest.mark.asyncio
async def test_an_illegal_event_in_a_batch_writes_nothing(stack: BackendStack) -> None:
    task_id = await stack.seed(metadata={'files': ['a.py']})
    before = stack.stored_bytes(task_id)

    with pytest.raises(DefectTransitionRefused):
        await stack.apply(
            task_id, Declare(declaration=_declaration()), Sighting(ref=_OTHER_REF)
        )

    assert stack.stored_bytes(task_id) == before


# --- (d) terminal and absent tasks ---------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['done', 'cancelled'])
async def test_a_terminal_task_is_refused(stack: BackendStack, status: str) -> None:
    task_id = await stack.seed(status=status)
    before = stack.stored_bytes(task_id)

    with pytest.raises(DefectTaskTerminal) as excinfo:
        await stack.apply(task_id, Declare(declaration=_declaration()))

    wire = excinfo.value.to_error_dict(task_id=task_id)
    assert wire['error'] == 'defect_write_to_terminal_task'
    assert wire['status'] == status
    assert wire['task_id'] == task_id
    assert stack.stored_bytes(task_id) == before


@pytest.mark.asyncio
async def test_an_absent_task_is_not_found(stack: BackendStack) -> None:
    await stack.seed()
    with pytest.raises(TaskNotFoundError):
        await stack.apply('999', Declare(declaration=_declaration()))


# --- (e)/(f) refs another task must declare --------------------------------------


@pytest.mark.asyncio
async def test_a_duplicate_of_an_undeclared_ref_is_refused(stack: BackendStack) -> None:
    task_id = await stack.seed(metadata={'defects': [_json(_declaration())]})
    before = stack.stored_bytes(task_id)

    with pytest.raises(DefectRefNotLive) as excinfo:
        await stack.apply(task_id, _duplicate_of(_OTHER_REF))

    wire = excinfo.value.to_error_dict()
    assert wire['error'] == 'defect_ref_not_live'
    assert wire['ref'] == _OTHER_REF
    assert wire['relation'] == 'duplicate_of'
    assert stack.stored_bytes(task_id) == before


@pytest.mark.asyncio
async def test_a_duplicate_of_a_ref_a_done_task_declares_is_accepted(stack: BackendStack) -> None:
    task_id = await stack.seed(metadata={'defects': [_json(_declaration())]})
    await stack.seed(status='done', metadata={'defects': [_json(_declaration(_OTHER_REF))]})

    await stack.apply(task_id, _duplicate_of(_OTHER_REF))

    [entry] = (await stack.metadata(task_id))['defects']
    assert entry['status'] == DefectStatus.PENDING.value


@pytest.mark.asyncio
async def test_a_retirement_superseded_by_an_undeclared_ref_is_refused(
    stack: BackendStack,
) -> None:
    task_id = await stack.seed()
    before = stack.stored_bytes(task_id)

    with pytest.raises(DefectRefNotLive) as excinfo:
        await stack.apply(task_id, _superseded_by(_OTHER_REF))

    assert excinfo.value.to_error_dict()['relation'] == 'superseded_by'
    assert stack.stored_bytes(task_id) == before


@pytest.mark.asyncio
async def test_a_retirement_superseded_by_a_live_ref_lands_in_the_retirements_key(
    stack: BackendStack,
) -> None:
    task_id = await stack.seed()
    await stack.seed(metadata={'defects': [_json(_declaration(_OTHER_REF))]})
    event = _superseded_by(_OTHER_REF)

    await stack.apply(task_id, event)

    metadata = await stack.metadata(task_id)
    assert metadata['defect_test_retirements'] == [_json(event.retirement)]
    assert 'defects' not in metadata


# --- (g) the lookup ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_lookup_finds_carriers_of_every_status(stack: BackendStack) -> None:
    pending = await stack.seed(metadata={'defects': [_json(_declaration(_OTHER_REF)), _json(_declaration())]})
    done = await stack.seed(status='done', metadata={'defects': [_json(_declaration())]})
    cancelled = await stack.seed(status='cancelled', metadata={'defects': [_json(_declaration())]})
    await stack.seed(metadata={'defects': [_json(_declaration(_THIRD_REF))]})
    await stack.seed(metadata={'files': ['a.py']})
    await stack.apply(pending, Proof(proof=_proof()))

    lookup = await lookup_defect_ref(stack.backend, stack.root, _REF)

    assert lookup.ref == _REF
    assert lookup.is_live
    assert lookup.carriers == (
        DefectRefCarrier(int(pending), 'pending', DefectStatus.PROVEN),
        DefectRefCarrier(int(done), 'done', DefectStatus.OPEN),
        DefectRefCarrier(int(cancelled), 'cancelled', DefectStatus.OPEN),
    )
    assert lookup.unreadable_task_ids == ()


@pytest.mark.asyncio
async def test_lookup_pages_past_one_page(stack: BackendStack) -> None:
    carriers = [await stack.seed(metadata={'defects': [_json(_declaration())]}) for _ in range(3)]

    lookup = await lookup_defect_ref(stack.backend, stack.root, _REF, page_size=2)

    assert [carrier.task_id for carrier in lookup.carriers] == [int(c) for c in carriers]


@pytest.mark.asyncio
async def test_lookup_of_an_undeclared_ref_has_no_carriers(stack: BackendStack) -> None:
    await stack.seed(metadata={'defects': [_json(_declaration())]})

    lookup = await lookup_defect_ref(stack.backend, stack.root, _OTHER_REF)

    assert lookup.carriers == ()
    assert not lookup.is_live
    assert lookup.as_dict() == {
        'ref': _OTHER_REF,
        'live': False,
        'carriers': [],
        'unreadable_task_ids': [],
    }


@pytest.mark.asyncio
async def test_lookup_of_a_malformed_ref_raises(stack: BackendStack) -> None:
    with pytest.raises(MalformedDefectRef):
        await lookup_defect_ref(stack.backend, stack.root, 'bogus:1')


@pytest.mark.asyncio
async def test_lookup_reports_a_task_whose_entries_it_cannot_read(stack: BackendStack) -> None:
    readable = await stack.seed(metadata={'defects': [_json(_declaration())]})
    unreadable = await stack.seed(metadata={'files': ['a.py']})
    stack.corrupt(unreadable, json.dumps({'defects': [{'ref': _REF}]}))

    lookup = await lookup_defect_ref(stack.backend, stack.root, _REF)

    assert [carrier.task_id for carrier in lookup.carriers] == [int(readable)]
    assert lookup.unreadable_task_ids == (int(unreadable),)
    assert lookup.as_dict()['carriers'] == [
        {'task_id': int(readable), 'task_status': 'pending', 'defect_status': 'open'}
    ]


# --- (h) a stored blob that is not a JSON object ----------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', ['[1, 2]', 'not json'])
async def test_a_non_object_stored_blob_is_refused_and_preserved(
    stack: BackendStack, raw: str
) -> None:
    task_id = await stack.seed()
    stack.corrupt(task_id, raw)

    with pytest.raises(StoredDefectRecordsMalformed) as excinfo:
        await stack.apply(task_id, Declare(declaration=_declaration()))

    wire = excinfo.value.to_error_dict(task_id=task_id)
    assert wire['error'] == 'stored_defect_records_malformed'
    assert wire['key'] is None
    assert stack.stored_bytes(task_id)[0] == raw
