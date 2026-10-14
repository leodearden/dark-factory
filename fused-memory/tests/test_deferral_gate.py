"""Boundary rows for metadata.deferral at the status choke point (task 6525).

The rows are plans/deferral-flip-condition-prd.md §6. Each one drives the public MCP
tools against a real SQLite store, a real write journal, a real event buffer and real
processes; nothing private is patched.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from shared import task_deferral
from shared.project_layout import tasks_db

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.backends.task_backend_errors import DeferralWriteAuthorityError
from fused_memory.config.schema import TaskmasterConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.models.reconciliation import ReconciliationEvent
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.targeted import TargetedReconciler
from fused_memory.server.tools import create_mcp_server
from fused_memory.services.write_journal import WriteJournal

_EVENT_PEEK_LIMIT = 1000


class RecordingReconciler(TargetedReconciler):
    """A targeted reconciler that records each request instead of reconciling."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def reconcile_task(
        self,
        task_id: str,
        transition: str,
        project_id: str,
        task_before: dict,
        project_root: str,
        reopen_reason: str | None = None,
    ) -> dict:
        self.calls.append({'task_id': task_id, 'transition': transition})
        return {}


@dataclass
class Stack:
    root: str
    backend: SqliteTaskBackend
    interceptor: TaskInterceptor
    journal: WriteJournal
    event_buffer: EventBuffer
    reconciler: RecordingReconciler
    server: Any
    ticket_store: TicketStore

    async def call(self, tool: str, **args: Any) -> Any:
        return await self.server._tool_manager.call_tool(
            tool, {'project_root': self.root, **args},
        )

    async def seed(
        self, status: str, title: str, metadata: dict[str, Any] | None = None,
    ) -> str:
        created = await self.backend.add_task(
            project_root=self.root,
            title=title,
            status=status,
            metadata=None if metadata is None else json.dumps(metadata),
        )
        return str(created['id'])

    async def read(self, task_id: str) -> dict:
        return await self.interceptor.get_task(task_id, self.root)

    async def journal_rows(self, operation: str) -> list[dict]:
        return [
            row for row in await self.journal.get_ops_since('1970-01-01', limit=10_000)
            if row['operation'] == operation
        ]

    async def events(self, event_type: str) -> list[ReconciliationEvent]:
        buffered = await self.event_buffer.peek_buffered(
            resolve_project_id(self.root), _EVENT_PEEK_LIMIT,
        )
        return [event for event in buffered if event.type == event_type]

    def with_proc_root(self, proc_root: Path) -> Stack:
        """This stack's store behind a second interceptor that reads /proc at *proc_root*."""
        interceptor = TaskInterceptor(
            self.backend, self.reconciler, self.event_buffer, proc_root=proc_root,
        )
        interceptor.set_write_journal(self.journal)
        server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
        return replace(self, interceptor=interceptor, server=server)


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', '-C', str(root), *args], check=True, capture_output=True, text=True,
    )
    return done.stdout.strip()


def _git_main_checkout(root: Path) -> str:
    """A real main checkout, so the tool layer's project_root resolution runs unpatched."""
    root.mkdir()
    _git(root, 'init', '-q', '-b', 'main')
    _git(root, 'config', 'user.email', 'deferral-gate@test')
    _git(root, 'config', 'user.name', 'deferral-gate')
    (root / 'README').write_text('deferral gate fixture\n')
    _git(root, 'add', 'README')
    _git(root, 'commit', '-q', '-m', 'seed')
    return str(root.resolve())


@pytest_asyncio.fixture
async def stack(tmp_path: Path) -> AsyncIterator[Stack]:
    root = _git_main_checkout(tmp_path / 'proj')
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=root))
    await backend.start()
    event_buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await event_buffer.initialize()
    journal = WriteJournal(tmp_path / 'wj')
    await journal.initialize()
    ticket_store = TicketStore(tmp_path / 'tickets.db')
    await ticket_store.initialize()
    reconciler = RecordingReconciler()
    interceptor = TaskInterceptor(backend, reconciler, event_buffer, ticket_store=ticket_store)
    interceptor.set_write_journal(journal)
    server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
    built = Stack(
        root=root,
        backend=backend,
        interceptor=interceptor,
        journal=journal,
        event_buffer=event_buffer,
        reconciler=reconciler,
        server=server,
        ticket_store=ticket_store,
    )
    try:
        yield built
    finally:
        await interceptor.close()
        await journal.close()
        await event_buffer.close()
        await backend.close()


def _spawn(argv: list[str]) -> subprocess.Popen[bytes]:
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _reap(process: subprocess.Popen[bytes]) -> None:
    process.kill()
    process.wait()


def _sleep_binary() -> str:
    found = shutil.which('sleep')
    assert found is not None, 'the holder fixtures need a sleep binary on PATH'
    return found


def _claude_named(directory: Path) -> Iterator[int]:
    """A live process whose /proc/<pid>/comm reads 'claude'."""
    directory.mkdir()
    binary = directory / 'claude'
    shutil.copy(_sleep_binary(), binary)
    binary.chmod(0o755)
    process = _spawn([str(binary), '600'])
    try:
        yield process.pid
    finally:
        _reap(process)


@pytest.fixture
def claude_holder(tmp_path: Path) -> Iterator[int]:
    yield from _claude_named(tmp_path / 'first_holder')


@pytest.fixture
def second_claude_holder(tmp_path: Path) -> Iterator[int]:
    yield from _claude_named(tmp_path / 'second_holder')


@pytest.fixture
def other_process() -> Iterator[int]:
    """A live process that is not the Claude CLI."""
    process = _spawn([_sleep_binary(), '600'])
    try:
        yield process.pid
    finally:
        _reap(process)


@pytest.fixture
def dead_pid() -> int:
    """The pid of a process that has exited and been reaped."""
    true_binary = shutil.which('true')
    assert true_binary is not None
    process = _spawn([true_binary])
    process.wait()
    return process.pid


def _until_condition(condition: str) -> dict[str, Any]:
    return {'kind': 'until_condition', 'condition': condition}


async def _assert_untouched(stack: Stack, task_id: str, status: str) -> None:
    row = await stack.read(task_id)
    assert row['status'] == status, row
    assert 'deferral' not in row['metadata'], row


def _assert_invalid(result: Any, task_id: str, reason_code: str, field: str) -> None:
    assert result['success'] is False, result
    assert result['error'] == 'deferral_invalid', result
    assert result['task_id'] == task_id, result
    assert result['reason_code'] == reason_code, result
    assert result['field'] == field, result
    assert 'value' in result, result
    assert isinstance(result['hint'], str) and result['hint'], result


# Row 1: entering `deferred` needs a record.


@pytest.mark.asyncio
@pytest.mark.parametrize('from_status', ['pending', 'in-progress', 'blocked'])
async def test_entry_without_record_is_refused_and_journalled(
    stack: Stack, from_status: str,
) -> None:
    task_id = await stack.seed(from_status, f'row 1 from {from_status}')

    result = await stack.call('set_task_status', id=task_id, status='deferred')

    assert result['success'] is False, result
    assert result['error'] == 'deferral_required', result
    assert result['task_id'] == task_id
    assert result['from_status'] == from_status
    assert result['accepted_kinds'] == sorted(k.value for k in task_deferral.CALLER_KINDS)
    assert isinstance(result['alternatives'], list) and len(result['alternatives']) == 3
    assert isinstance(result['hint'], str) and result['hint']
    await _assert_untouched(stack, task_id, from_status)
    rows = await stack.journal_rows('set_task_status')
    assert len(rows) == 1, rows
    assert rows[0]['success'] == 0
    assert 'deferral_required' in rows[0]['error']


# Rows 3, 4 and 26: the record is validated, and a holder is corroborated in /proc.


@pytest.mark.asyncio
async def test_blank_condition_is_deferral_invalid(stack: Stack) -> None:
    task_id = await stack.seed('pending', 'row 3')

    result = await stack.call(
        'set_task_status', id=task_id, status='deferred', deferral=_until_condition('  '),
    )

    _assert_invalid(result, task_id, 'missing_field', 'condition')
    await _assert_untouched(stack, task_id, 'pending')


@pytest.mark.asyncio
async def test_holder_must_be_a_live_claude_process(
    stack: Stack, other_process: int, dead_pid: int,
) -> None:
    task_id = await stack.seed('pending', 'row 4')

    not_session = await stack.call(
        'set_task_status', id=task_id, status='deferred',
        deferral={'kind': 'held_by_session', 'holder_pid': other_process},
    )
    not_live = await stack.call(
        'set_task_status', id=task_id, status='deferred',
        deferral={'kind': 'held_by_session', 'holder_pid': dead_pid},
    )

    _assert_invalid(not_session, task_id, 'holder_not_session', 'holder_pid')
    _assert_invalid(not_live, task_id, 'holder_not_live', 'holder_pid')
    await _assert_untouched(stack, task_id, 'pending')


@pytest.mark.asyncio
async def test_unreadable_proc_is_holder_unverifiable(
    stack: Stack, claude_holder: int, tmp_path: Path,
) -> None:
    empty_proc = tmp_path / 'empty_proc'
    empty_proc.mkdir()
    blind = stack.with_proc_root(empty_proc)
    task_id = await stack.seed('pending', 'row 26')

    result = await blind.call(
        'set_task_status', id=task_id, status='deferred',
        deferral={'kind': 'held_by_session', 'holder_pid': claude_holder},
    )

    _assert_invalid(result, task_id, 'holder_unverifiable', 'holder_pid')
    await _assert_untouched(stack, task_id, 'pending')


# Row 25 edges and decision 3: request shapes refused before any row is read.


@pytest.mark.asyncio
async def test_request_shape_refusals(stack: Stack) -> None:
    first = await stack.seed('pending', 'shape first')
    second = await stack.seed('pending', 'shape second')
    unrecorded = await stack.seed('deferred', 'shape unrecorded')
    both = f'{first},{second}'
    condition = _until_condition('request shape')

    restamp_to_pending = await stack.call(
        'set_task_status', id=first, status='pending', restamp_only=True,
    )
    restamp_csv = await stack.call(
        'set_task_status', id=both, status='deferred', deferral=condition, restamp_only=True,
    )
    expected_csv = await stack.call(
        'set_task_status', id=both, status='deferred', deferral=condition,
        expected_stamped_at='2026-10-14T00:00:00+00:00',
    )
    deferral_on_pending = await stack.call(
        'set_task_status', id=first, status='pending', deferral=condition,
    )
    restamp_without_record = await stack.call(
        'set_task_status', id=unrecorded, status='deferred', restamp_only=True,
    )

    _assert_invalid(restamp_to_pending, first, 'field_not_allowed', 'restamp_only')
    _assert_invalid(restamp_csv, both, 'cas_needs_single_id', 'id')
    _assert_invalid(expected_csv, both, 'cas_needs_single_id', 'id')
    _assert_invalid(deferral_on_pending, first, 'status_not_deferred', 'status')
    assert restamp_without_record['success'] is False, restamp_without_record
    assert restamp_without_record['error'] == 'deferral_required', restamp_without_record
    await _assert_untouched(stack, first, 'pending')
    await _assert_untouched(stack, second, 'pending')
    await _assert_untouched(stack, unrecorded, 'deferred')


@pytest.mark.asyncio
async def test_kind_refusals(stack: Stack) -> None:
    task_id = await stack.seed('pending', 'kind refusals')

    legacy = await stack.call(
        'set_task_status', id=task_id, status='deferred', deferral={'kind': 'legacy_unknown'},
    )
    planning = await stack.call(
        'set_task_status', id=task_id, status='deferred', deferral={'kind': 'planning'},
    )
    no_carrier = await stack.call(
        'set_task_status', id=task_id, status='deferred',
        deferral={'kind': 'carried_by', 'carrier_task_id': 999_999},
    )
    self_carrier = await stack.call(
        'set_task_status', id=task_id, status='deferred',
        deferral={'kind': 'carried_by', 'carrier_task_id': int(task_id)},
    )

    _assert_invalid(legacy, task_id, 'server_only_kind', 'kind')
    _assert_invalid(planning, task_id, 'server_only_kind', 'kind')
    _assert_invalid(no_carrier, task_id, 'carrier_not_found', 'carrier_task_id')
    _assert_invalid(self_carrier, task_id, 'carrier_is_self', 'carrier_task_id')
    await _assert_untouched(stack, task_id, 'pending')


# Rows 2 and 27 (entry): each caller kind enters `deferred`, stored and echoed.

_AGENT = 'claude-test-6525'


async def _deferral_of_kind(stack: Stack, kind: str, holder_pid: int) -> dict[str, Any]:
    if kind == 'carried_by':
        carrier = await stack.seed('pending', f'carrier for {kind}')
        return {'kind': 'carried_by', 'carrier_task_id': int(carrier)}
    if kind == 'until_condition':
        expires_at = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
        return {'kind': 'until_condition', 'condition': 'row 2', 'expires_at': expires_at}
    return {'kind': 'held_by_session', 'holder_pid': holder_pid, 'grace_secs': 600}


_STORED_VARIANTS = {
    'carried_by': task_deferral.CarriedByDeferral,
    'until_condition': task_deferral.UntilConditionDeferral,
    'held_by_session': task_deferral.HeldBySessionDeferral,
}


def _valid_record(raw: object) -> task_deferral.Deferral:
    record = task_deferral.parse_record(raw)
    assert record is not None and not isinstance(record, task_deferral.InvalidRecord), record
    return record


async def _enter(stack: Stack, task_id: str, deferral: dict[str, Any]) -> dict:
    result = await stack.call(
        'set_task_status', id=task_id, status='deferred', deferral=deferral, agent_id=_AGENT,
    )
    assert 'error' not in result, result
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', sorted(_STORED_VARIANTS))
@pytest.mark.parametrize('from_status', ['pending', 'in-progress', 'blocked'])
async def test_each_caller_kind_enters_deferred(
    stack: Stack, claude_holder: int, from_status: str, kind: str,
) -> None:
    task_id = await stack.seed(from_status, f'row 2 {kind} from {from_status}')
    deferral = await _deferral_of_kind(stack, kind, claude_holder)

    result = await _enter(stack, task_id, deferral)

    row = await stack.read(task_id)
    assert row['status'] == 'deferred'
    stored = row['metadata']['deferral']
    record = _valid_record(stored)
    assert isinstance(record, _STORED_VARIANTS[kind]), record
    assert record.stamped_by == _AGENT
    assert record.deferred_at == record.stamped_at
    if isinstance(record, task_deferral.HeldBySessionDeferral):
        assert record.holder.pid == claude_holder
        assert record.holder.boot_id
    assert isinstance(result['deferral'], dict)
    assert result['deferral'] == stored


@pytest.mark.asyncio
async def test_entry_event_carries_the_record(stack: Stack) -> None:
    task_id = await stack.seed('pending', 'entry event')

    await _enter(stack, task_id, _until_condition('entry event'))

    stored = (await stack.read(task_id))['metadata']['deferral']
    payloads = [
        event.payload for event in await stack.events('task_status_changed')
        if event.payload['task_id'] == task_id
    ]
    assert len(payloads) == 1, payloads
    assert payloads[0]['deferral'] == stored


@pytest.mark.asyncio
async def test_csv_entry_applies_one_record_to_every_id(stack: Stack) -> None:
    first = await stack.seed('pending', 'csv entry first')
    second = await stack.seed('pending', 'csv entry second')

    result = await stack.call(
        'set_task_status', id=f'{first},{second}', status='deferred',
        deferral=_until_condition('csv entry'), agent_id=_AGENT,
    )

    assert result['success'] is True, result
    stored = [(await stack.read(task_id))['metadata']['deferral'] for task_id in (first, second)]
    records = [task_deferral.parse_record(raw) for raw in stored]
    assert all(isinstance(r, task_deferral.UntilConditionDeferral) for r in records), records
    assert stored[0]['condition'] == stored[1]['condition'] == 'csv entry'
    assert stored[0]['stamped_at'] == stored[1]['stamped_at']
    echoes = [one['result']['deferral'] for one in result['results']]
    assert echoes == stored


# Rows 5 and 27 (exit): leaving `deferred` removes the record and reports it.


async def _exit_arguments(stack: Stack, exit_status: str) -> dict[str, Any]:
    if exit_status != 'done':
        return {}
    head = _git(Path(stack.root), 'rev-parse', 'HEAD')
    return {'done_provenance': {'kind': 'found_on_main', 'commit': head, 'note': 'test'}}


async def _status_events(stack: Stack, task_id: str, new_status: str) -> list[dict[str, Any]]:
    return [
        event.payload for event in await stack.events('task_status_changed')
        if event.payload['task_id'] == task_id and event.payload['new_status'] == new_status
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('exit_status', ['pending', 'cancelled', 'blocked', 'done'])
async def test_exit_clears_the_record(stack: Stack, exit_status: str) -> None:
    task_id = await stack.seed('pending', f'row 5 exit {exit_status}', {'x_note': 'kept'})
    await _enter(stack, task_id, _until_condition(f'row 5 exit {exit_status}'))
    stored = (await stack.read(task_id))['metadata']['deferral']

    result = await stack.call(
        'set_task_status', id=task_id, status=exit_status,
        **await _exit_arguments(stack, exit_status),
    )

    assert 'error' not in result, result
    row = await stack.read(task_id)
    assert row['status'] == exit_status
    assert 'deferral' not in row['metadata'], row
    assert row['metadata']['x_note'] == 'kept'
    assert result['deferral_cleared'] == stored
    payloads = await _status_events(stack, task_id, exit_status)
    assert len(payloads) == 1, payloads
    assert payloads[0]['deferral_cleared'] == stored


@pytest.mark.asyncio
async def test_exit_from_unrecorded_row_echoes_none(stack: Stack) -> None:
    task_id = await stack.seed('deferred', 'unrecorded exit')

    result = await stack.call('set_task_status', id=task_id, status='pending')

    assert 'error' not in result, result
    assert 'deferral_cleared' in result, result
    assert result['deferral_cleared'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', sorted(_STORED_VARIANTS))
async def test_plain_pending_release_is_legal_for_every_kind(
    stack: Stack, claude_holder: int, kind: str,
) -> None:
    task_id = await stack.seed('pending', f'plain release {kind}')
    await _enter(stack, task_id, await _deferral_of_kind(stack, kind, claude_holder))

    result = await stack.call('set_task_status', id=task_id, status='pending')

    assert 'error' not in result, result
    assert (await stack.read(task_id))['status'] == 'pending'


# Rows 6 and 27 (re-stamp): a deferral on a deferred row replaces its record in place.


@pytest.mark.asyncio
async def test_restamp_replaces_record_without_a_status_change(stack: Stack) -> None:
    task_id = await stack.seed('pending', 'row 6 restamp')
    carrier = await stack.seed('pending', 'row 6 carrier')
    await _enter(stack, task_id, _until_condition('row 6 first'))
    first = _valid_record((await stack.read(task_id))['metadata']['deferral'])
    first_raw = (await stack.read(task_id))['metadata']['deferral']
    await stack.interceptor.drain()
    status_events_before = len(await _status_events(stack, task_id, 'deferred'))
    reconciles_before = len(stack.reconciler.calls)
    journal_before = len(await stack.journal_rows('set_task_status'))

    result = await _enter(stack, task_id, {'kind': 'carried_by', 'carrier_task_id': int(carrier)})

    await stack.interceptor.drain()
    row = await stack.read(task_id)
    assert row['status'] == 'deferred'
    stored = row['metadata']['deferral']
    second = _valid_record(stored)
    assert isinstance(second, task_deferral.CarriedByDeferral), second
    assert second.stamped_by == _AGENT
    assert second.deferred_at == first.deferred_at
    assert second.stamped_at > first.stamped_at
    assert result['deferral'] == stored
    assert 'reconciliation' not in result, result
    restamps = [
        event.payload for event in await stack.events('deferral_restamped')
        if event.payload['task_id'] == task_id
    ]
    assert len(restamps) == 1, restamps
    assert restamps[0]['old'] == first_raw
    assert restamps[0]['new'] == stored
    assert len(await _status_events(stack, task_id, 'deferred')) == status_events_before
    assert len(stack.reconciler.calls) == reconciles_before
    rows = await stack.journal_rows('set_task_status')
    assert len(rows) == journal_before + 1
    assert rows[-1]['success'] == 1


@pytest.mark.asyncio
async def test_deferred_write_without_deferral_on_deferred_row_is_still_a_no_op(
    stack: Stack,
) -> None:
    task_id = await stack.seed('pending', 'row 6 no-op')
    await _enter(stack, task_id, _until_condition('row 6 no-op'))
    stored = (await stack.read(task_id))['metadata']['deferral']

    result = await stack.call('set_task_status', id=task_id, status='deferred')

    assert result.get('no_op') is True, result
    assert (await stack.read(task_id))['metadata']['deferral'] == stored


@pytest.mark.asyncio
async def test_restamp_of_held_row_moves_the_lease(
    stack: Stack, claude_holder: int, second_claude_holder: int,
) -> None:
    task_id = await stack.seed('pending', 'row 6 lease')
    await _enter(stack, task_id, {'kind': 'held_by_session', 'holder_pid': claude_holder})

    await _enter(stack, task_id, {'kind': 'held_by_session', 'holder_pid': second_claude_holder})

    record = _valid_record((await stack.read(task_id))['metadata']['deferral'])
    assert isinstance(record, task_deferral.HeldBySessionDeferral), record
    assert record.holder.pid == second_claude_holder


# Rows 9, 25 and 10: compare-and-set, the guarded re-stamp, and the legacy shape.


def _assert_changed(
    result: Any, *, expected: str | None, current: str | None, current_status: str,
) -> None:
    assert result['success'] is False, result
    assert result['error'] == 'deferral_changed', result
    assert result['expected_stamped_at'] == expected
    if current is None:
        assert result['current_stamped_at'] is None, result
    else:
        assert task_deferral.parse_utc_timestamp(
            result['current_stamped_at'],
        ) == task_deferral.parse_utc_timestamp(current)
    assert result['current_status'] == current_status


async def _stamped_at(stack: Stack, task_id: str) -> str:
    return (await stack.read(task_id))['metadata']['deferral']['stamped_at']


@pytest.mark.asyncio
async def test_cas_refuses_a_moved_record_and_replays_success(stack: Stack) -> None:
    task_id = await stack.seed('pending', 'row 9 cas')
    await _enter(stack, task_id, _until_condition('row 9 first'))
    s1 = await _stamped_at(stack, task_id)
    await _enter(stack, task_id, _until_condition('row 9 second'))
    s2 = await _stamped_at(stack, task_id)

    moved = await stack.call(
        'set_task_status', id=task_id, status='pending', expected_stamped_at=s1,
    )
    _assert_changed(moved, expected=s1, current=s2, current_status='deferred')
    assert await _stamped_at(stack, task_id) == s2
    stored = (await stack.read(task_id))['metadata']['deferral']

    released = await stack.call(
        'set_task_status', id=task_id, status='pending', expected_stamped_at=s2,
        client_op_id='op-6525-cas',
    )
    assert 'error' not in released, released
    assert (await stack.read(task_id))['status'] == 'pending'
    assert released['deferral_cleared'] == stored

    replayed = await stack.call(
        'set_task_status', id=task_id, status='pending', expected_stamped_at=s2,
        client_op_id='op-6525-cas',
    )
    assert replayed == released


@pytest.mark.asyncio
async def test_cas_on_a_row_that_left_deferred(stack: Stack) -> None:
    task_id = await stack.seed('pending', 'row 9 left')
    expected = '2026-10-14T00:00:00+00:00'

    result = await stack.call(
        'set_task_status', id=task_id, status='blocked', expected_stamped_at=expected,
    )

    _assert_changed(result, expected=expected, current=None, current_status='pending')
    await _assert_untouched(stack, task_id, 'pending')


_LEGACY = {'kind': 'legacy_unknown', 'reason': 'migration'}


@pytest.mark.asyncio
async def test_guarded_restamp_on_unrecorded_row(stack: Stack) -> None:
    task_id = await stack.seed('deferred', 'row 25 unrecorded')

    result = await stack.call(
        'set_task_status', id=task_id, status='deferred', deferral=_LEGACY, restamp_only=True,
    )

    assert 'error' not in result, result
    stored = (await stack.read(task_id))['metadata']['deferral']
    record = _valid_record(stored)
    assert isinstance(record, task_deferral.LegacyUnknownDeferral), record
    assert record.deferred_at is None
    assert result['deferral'] == stored


@pytest.mark.asyncio
async def test_guarded_restamp_on_repended_row(stack: Stack) -> None:
    task_id = await stack.seed('deferred', 'row 25 repended')
    assert (await stack.read(task_id))['status'] == 'deferred'
    released = await stack.call('set_task_status', id=task_id, status='pending')
    assert 'error' not in released, released

    result = await stack.call(
        'set_task_status', id=task_id, status='deferred', deferral=_LEGACY, restamp_only=True,
    )

    _assert_changed(result, expected=None, current=None, current_status='pending')
    await _assert_untouched(stack, task_id, 'pending')


@pytest.mark.asyncio
async def test_guarded_restamp_expects_the_current_record(stack: Stack) -> None:
    task_id = await stack.seed('pending', 'row 25 recorded')
    await _enter(stack, task_id, _until_condition('row 25 first'))
    current = await _stamped_at(stack, task_id)
    replacement = _until_condition('row 25 second')

    unguarded = await stack.call(
        'set_task_status', id=task_id, status='deferred', deferral=replacement,
        restamp_only=True,
    )
    guarded = await stack.call(
        'set_task_status', id=task_id, status='deferred', deferral=replacement,
        restamp_only=True, expected_stamped_at=current,
    )

    _assert_changed(unguarded, expected=None, current=current, current_status='deferred')
    assert 'error' not in guarded, guarded
    stored = (await stack.read(task_id))['metadata']['deferral']
    assert stored['condition'] == 'row 25 second'


@pytest.mark.asyncio
async def test_legacy_unknown_on_recorded_row_is_server_only(stack: Stack) -> None:
    task_id = await stack.seed('pending', 'row 10 recorded')
    await _enter(stack, task_id, _until_condition('row 10'))
    stored = (await stack.read(task_id))['metadata']['deferral']

    result = await stack.call('set_task_status', id=task_id, status='deferred', deferral=_LEGACY)

    _assert_invalid(result, task_id, 'server_only_kind', 'kind')
    assert (await stack.read(task_id))['metadata']['deferral'] == stored


# Row 7, row 6's planning-born precondition, and decision 3's submit refusal.


async def _birth(stack: Stack, title: str) -> str:
    born = await stack.call(
        'submit_task', title=title, description=f'{title} body', planning_mode=True,
        agent_id=_AGENT,
    )
    assert born.get('status') == 'deferred', born
    return str(born['task_id'])


def _per_id(result: dict) -> dict[str, dict]:
    return {str(one['task_id']): one['result'] for one in result['results']}


@pytest.mark.asyncio
async def test_planning_birth_is_stamped_planning(stack: Stack) -> None:
    task_id = await _birth(stack, 'row 7 birth')

    metadata = (await stack.read(task_id))['metadata']

    record = _valid_record(metadata['deferral'])
    assert isinstance(record, task_deferral.PlanningDeferral), record
    assert record.stamped_by == _AGENT
    assert record.deferred_at == record.stamped_at
    assert metadata['human_decomposed'] is True


@pytest.mark.asyncio
async def test_commit_pending_releases_a_planning_hold(stack: Stack) -> None:
    births = [await _birth(stack, f'row 7 release {n}') for n in range(2)]

    result = await stack.call('commit_planning', task_ids=','.join(births))

    assert result['success'] is True, result
    for task_id, one in _per_id(result).items():
        assert one['deferral_cleared']['kind'] == 'planning', one
        row = await stack.read(task_id)
        assert row['status'] == 'pending'
        assert 'deferral' not in row['metadata']


@pytest.mark.asyncio
@pytest.mark.parametrize('target_status', ['pending', 'cancelled'])
async def test_commit_releases_an_unrecorded_hold(stack: Stack, target_status: str) -> None:
    task_id = await stack.seed('deferred', f'row 7 unrecorded {target_status}')

    result = await stack.call(
        'commit_planning', task_ids=task_id, target_status=target_status,
    )

    assert 'error' not in result, result
    assert result['deferral_cleared'] is None
    assert (await stack.read(task_id))['status'] == target_status


@pytest.mark.asyncio
@pytest.mark.parametrize('target_status', ['pending', 'cancelled'])
async def test_commit_refuses_a_non_planning_hold(stack: Stack, target_status: str) -> None:
    held = await stack.seed('pending', f'row 7 held {target_status}')
    await _enter(stack, held, _until_condition(f'row 7 held {target_status}'))
    stored = (await stack.read(held))['metadata']['deferral']
    birth = await _birth(stack, f'row 7 batch birth {target_status}')

    result = await stack.call(
        'commit_planning', task_ids=f'{held},{birth}', target_status=target_status,
    )

    assert result['success'] is False, result
    _assert_invalid(_per_id(result)[held], held, 'not_planning_hold', 'deferral.kind')
    held_row = await stack.read(held)
    assert held_row['status'] == 'deferred'
    assert held_row['metadata']['deferral'] == stored
    assert (await stack.read(birth))['status'] == target_status
    released = await stack.call('set_task_status', id=held, status='pending')
    assert 'error' not in released, released


@pytest.mark.asyncio
async def test_commit_deferred_needs_a_deferral(stack: Stack) -> None:
    births = [await _birth(stack, f'row 6 planning-born {n}') for n in range(2)]
    born_at = {
        task_id: _valid_record((await stack.read(task_id))['metadata']['deferral']).deferred_at
        for task_id in births
    }

    bare = await stack.call(
        'commit_planning', task_ids=','.join(births), target_status='deferred',
    )
    restamped = await stack.call(
        'commit_planning', task_ids=','.join(births), target_status='deferred',
        deferral=_until_condition('row 6 planning-born'),
    )

    assert bare['success'] is False, bare
    assert {one['error'] for one in _per_id(bare).values()} == {'deferral_required'}
    assert restamped['success'] is True, restamped
    restamp_events = {
        event.payload['task_id'] for event in await stack.events('deferral_restamped')
    }
    for task_id, one in _per_id(restamped).items():
        stored = (await stack.read(task_id))['metadata']['deferral']
        record = _valid_record(stored)
        assert isinstance(record, task_deferral.UntilConditionDeferral), record
        assert record.deferred_at == born_at[task_id]
        assert one['deferral'] == stored
        assert task_id in restamp_events


@pytest.mark.asyncio
async def test_commit_planning_attributes_agent_id(stack: Stack) -> None:
    task_id = await _birth(stack, 'row 7 attribution')

    result = await stack.call(
        'commit_planning', task_ids=task_id, target_status='deferred',
        deferral=_until_condition('row 7 attribution'), agent_id='claude-commit-6525',
    )

    assert 'error' not in result, result
    record = _valid_record((await stack.read(task_id))['metadata']['deferral'])
    assert record.stamped_by == 'claude-commit-6525'


@pytest.mark.asyncio
@pytest.mark.parametrize('planning_mode', [True, False])
async def test_submit_task_refuses_caller_deferral(stack: Stack, planning_mode: bool) -> None:
    now = datetime.now(UTC)
    forged = task_deferral.record_to_json(
        task_deferral.UntilConditionDeferral(
            condition='forged', deferred_at=now, stamped_at=now, stamped_by=_AGENT,
        )
    )
    before = len((await stack.interceptor.get_tasks(stack.root))['tasks'])

    result = await stack.call(
        'submit_task', title='row 8 forged', description='forged record',
        planning_mode=planning_mode, metadata={'deferral': forged},
    )

    assert result.get('error') == 'deferral_write_authority', result
    assert result['path'] == 'metadata.deferral'
    assert 'ticket' not in result
    assert len((await stack.interceptor.get_tasks(stack.root))['tasks']) == before


# Row 8: no metadata writer but the status choke point may add, alter or remove the record.

_MODES = ['merge', 'additive', 'replace']


async def _deferred_with_note(stack: Stack, title: str) -> tuple[str, dict[str, Any]]:
    task_id = await stack.seed('pending', title, {'x_note': 'kept'})
    await _enter(stack, task_id, _until_condition(title))
    return task_id, (await stack.read(task_id))['metadata']


def _assert_write_authority(result: Any, task_id: str) -> None:
    assert result['success'] is False, result
    assert result['error'] == 'deferral_write_authority', result
    assert result['path'] == 'metadata.deferral'
    assert result['task_id'] == task_id


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', _MODES)
async def test_update_task_cannot_alter_deferral(stack: Stack, mode: str) -> None:
    task_id, metadata = await _deferred_with_note(stack, f'row 8 alter {mode}')
    altered = {**metadata['deferral'], 'condition': 'altered'}
    payload = {**metadata, 'deferral': altered} if mode == 'replace' else {'deferral': altered}

    result = await stack.call(
        'update_task', id=task_id, metadata=payload, metadata_mode=mode,
    )

    _assert_write_authority(result, task_id)
    assert (await stack.read(task_id))['metadata']['deferral'] == metadata['deferral']


@pytest.mark.asyncio
async def test_update_task_cannot_add_or_remove_deferral(stack: Stack) -> None:
    pending = await stack.seed('pending', 'row 8 add')
    task_id, metadata = await _deferred_with_note(stack, 'row 8 remove')
    without = {k: v for k, v in metadata.items() if k != 'deferral'}

    added = await stack.call(
        'update_task', id=pending, metadata={'deferral': metadata['deferral']},
        metadata_mode='merge',
    )
    nulled = await stack.call(
        'update_task', id=task_id, metadata={'deferral': None}, metadata_mode='merge',
    )
    dropped = await stack.call(
        'update_task', id=task_id, metadata=without, metadata_mode='replace',
    )

    _assert_write_authority(added, pending)
    assert 'deferral' not in (await stack.read(pending))['metadata']
    _assert_write_authority(nulled, task_id)
    _assert_write_authority(dropped, task_id)
    assert (await stack.read(task_id))['metadata']['deferral'] == metadata['deferral']


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', _MODES)
async def test_update_task_echo_of_stored_record_is_accepted(stack: Stack, mode: str) -> None:
    task_id, metadata = await _deferred_with_note(stack, f'row 8 echo {mode}')
    echoed = {**metadata, f'x_echo_{mode}': 'written'}

    result = await stack.call(
        'update_task', id=task_id, metadata=echoed, metadata_mode=mode,
    )

    assert result.get('error') is None, result
    after = (await stack.read(task_id))['metadata']
    assert json.dumps(after['deferral'], sort_keys=True) == json.dumps(
        metadata['deferral'], sort_keys=True,
    )
    assert after[f'x_echo_{mode}'] == 'written'


@pytest.mark.asyncio
async def test_rewrite_audit_trail_carries_deferral(stack: Stack) -> None:
    task_id, metadata = await _deferred_with_note(stack, 'row 8 rotation')
    without = {k: v for k, v in metadata.items() if k != 'deferral'}
    altered = {**metadata, 'deferral': {**metadata['deferral'], 'condition': 'altered'}}

    await stack.backend.rewrite_audit_trail(
        task_id, stack.root, description=None, metadata=metadata,
    )
    assert (await stack.read(task_id))['metadata']['deferral'] == metadata['deferral']
    with pytest.raises(DeferralWriteAuthorityError):
        await stack.backend.rewrite_audit_trail(
            task_id, stack.root, description=None, metadata=without,
        )
    with pytest.raises(DeferralWriteAuthorityError):
        await stack.backend.rewrite_audit_trail(
            task_id, stack.root, description=None, metadata=altered,
        )
    assert (await stack.read(task_id))['metadata']['deferral'] == metadata['deferral']


# Row 11: the one raw-SQL exit from `deferred`, the candidate-key self-heal, drops the record.


async def _two_tasks(root: Path) -> None:
    backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(root)))
    await backend.start()
    try:
        await backend.add_task(project_root=str(root), title='row 11 canonical')
        await backend.add_task(project_root=str(root), title='row 11 duplicate')
    finally:
        await backend.close()


def _forge_v3_residual_duplicate(root: Path, metadata: dict[str, Any]) -> None:
    with sqlite3.connect(tasks_db(root)) as conn:
        conn.execute('DROP INDEX ux_tasks_candidate_key')
        conn.execute('PRAGMA user_version = 3')
        title, candidate_key = conn.execute(
            'SELECT title, candidate_key FROM tasks WHERE id = 1',
        ).fetchone()
        conn.execute(
            "UPDATE tasks SET title = ?, candidate_key = ?, status = 'deferred', "
            'metadata = ? WHERE id = 2',
            (title, candidate_key, json.dumps(metadata)),
        )


@pytest.mark.asyncio
async def test_self_heal_cancel_drops_deferral(tmp_path: Path) -> None:
    root = tmp_path / 'heal'
    root.mkdir()
    await _two_tasks(root)
    now = datetime.now(UTC)
    record = task_deferral.record_to_json(
        task_deferral.UntilConditionDeferral(
            condition='row 11', deferred_at=now, stamped_at=now, stamped_by=_AGENT,
        )
    )
    _forge_v3_residual_duplicate(root, {'deferral': record, 'x_note': 'keep'})

    healed = SqliteTaskBackend(TaskmasterConfig(project_root=str(root)))
    await healed.start()
    try:
        row = await healed.get_task('2', str(root))
    finally:
        await healed.close()

    assert row['status'] == 'cancelled', row
    assert 'deferral' not in row['metadata'], row
    assert 'auto_cancelled_by_self_heal' in row['metadata']
    assert row['metadata']['x_note'] == 'keep'
