"""Boundary rows for metadata.deferral at the status choke point (task 6525).

The rows are plans/deferral-flip-condition-prd.md §6. Each one drives the public MCP
tools against a real SQLite store, a real write journal, a real event buffer and real
processes; nothing private is patched.
"""

from __future__ import annotations

import json
import shutil
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

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
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


@pytest.fixture
def claude_holder(tmp_path: Path) -> Iterator[int]:
    """A live process whose /proc/<pid>/comm reads 'claude'."""
    binary = tmp_path / 'claude'
    shutil.copy(_sleep_binary(), binary)
    binary.chmod(0o755)
    process = _spawn([str(binary), '600'])
    try:
        yield process.pid
    finally:
        _reap(process)


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
