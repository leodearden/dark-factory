"""Task 4503 end to end: a submission the curator pool cannot judge waits for a real verdict.

Real components throughout: a :class:`TicketStore` on a file, a
:class:`TaskInterceptor`, and the :class:`TicketJanitor` that
``server/main.py::_build_ticket_janitor`` builds, ticked explicitly. Only the
curator is a double, except where a real :class:`TaskCurator` meets a real
frozen :class:`UsageGate`. Neither the janitor nor the interceptor has a clock
seam, so simulated time is the ticket's age: ``created_at`` is pinned through
an independent connection, as the store suite does.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple, TypeVar, cast
from unittest.mock import patch

import pytest
import pytest_asyncio
from escalation.queue import EscalationQueue
from shared.config_models import AccountConfig, UsageCapConfig
from shared.invocation_outcome import CapHit
from shared.project_layout import escalation_queue_dir
from shared.usage_gate import UsageGate

from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig, TicketJanitorConfig
from fused_memory.middleware.curation_deferral import DEDUPE_SKIPPED_KEY, DeferReason
from fused_memory.middleware.task_curator import (
    CandidateTask,
    CuratorDecision,
    CuratorFailureError,
    PoolWithheld,
    PreparedCandidate,
    RewrittenTask,
    TaskCurator,
    normalize_title,
)
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_janitor import TicketJanitor, curation_backlog_anchor
from fused_memory.middleware.ticket_store import AWAITING_CURATION, TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.server.main import _build_ticket_janitor

PROJECT_ROOT = '/project'
PROJECT_ID = 'project'
DEADLINE_SECONDS = 5.0
RESOLVE_TIMEOUT_SECONDS = 4.0
TINY_BACKOFF_SECONDS = 0.001
OLD_TERMINAL_BOUND = timedelta(seconds=1800)

T = TypeVar('T')


async def _bounded(awaitable: Awaitable[T]) -> T:
    return await asyncio.wait_for(awaitable, DEADLINE_SECONDS)


class _Hold:
    """A verdict that never arrives: the curation attempt is still in flight."""


HOLD = _Hold()
Verdict = CuratorDecision | BaseException | _Hold
CAPPED = CuratorDecision.deferred(DeferReason.ALL_ACCOUNTS_CAPPED)
AUTH_FAILED = CuratorDecision.deferred(DeferReason.ACCOUNTS_AUTH_FAILED)


class _Curator:
    """A curator double answering each candidate, by title, through *answer*."""

    def __init__(self, answer: Callable[[str], Verdict]) -> None:
        self._answer = answer
        self.batch_raises: deque[BaseException] = deque()
        self.asked: list[str] = []
        self.holding = asyncio.Event()

    async def _verdict(self, title: str) -> CuratorDecision:
        self.asked.append(title)
        verdict = self._answer(title)
        if isinstance(verdict, _Hold):
            self.holding.set()
            await asyncio.Event().wait()
        if isinstance(verdict, BaseException):
            raise verdict
        assert isinstance(verdict, CuratorDecision)
        return verdict

    async def prepare_candidate(
        self, candidate: CandidateTask, project_id: str, project_root: str,
    ) -> PreparedCandidate:
        return PreparedCandidate(candidate=candidate, pool=[], pool_sizes={}, prompt_tokens=0)

    async def curate_batch_prepared(
        self, prepared: list[PreparedCandidate], project_id: str, project_root: str,
    ) -> list[CuratorDecision]:
        if self.batch_raises:
            raise self.batch_raises.popleft()
        return [await self._verdict(p.candidate.title) for p in prepared]

    async def curate(
        self, candidate: CandidateTask, project_id: str, project_root: str,
    ) -> CuratorDecision:
        return await self._verdict(candidate.title)

    def note_created(self, project_id: str, candidate: CandidateTask, task_id: str) -> None:
        return None

    async def record_task(self, task_id: str, candidate: CandidateTask, project_id: str) -> None:
        return None

    async def reembed_task(self, task_id: str, candidate: CandidateTask, project_id: str) -> None:
        return None

    async def close(self) -> None:
        return None


def _in_turn(script: Mapping[str, Sequence[Verdict]]) -> Callable[[str], Verdict]:
    turns = {title: deque(verdicts) for title, verdicts in script.items()}
    return lambda title: turns[title].popleft()


class _RecordingTaskmaster:
    def __init__(self) -> None:
        self.add_task_calls: list[dict[str, Any]] = []
        self.update_task_calls: list[dict[str, Any]] = []
        self.tasks: dict[str, dict[str, Any]] = {}

    async def ensure_connected(self) -> None:
        return None

    async def add_task(self, **kwargs: Any) -> dict[str, Any]:
        self.add_task_calls.append(kwargs)
        return {'id': str(100 + len(self.add_task_calls)), 'title': kwargs.get('title')}

    async def get_task(self, task_id: str, project_root: str) -> dict[str, Any]:
        return self.tasks[task_id]

    async def update_task(self, **kwargs: Any) -> dict[str, Any]:
        self.update_task_calls.append(kwargs)
        return {'id': kwargs['task_id'], 'updated': True}


def _config(*, backoff_seconds: float = TINY_BACKOFF_SECONDS) -> FusedMemoryConfig:
    config = FusedMemoryConfig()
    config.curator = CuratorConfig(
        deferral_backoff_base_seconds=backoff_seconds,
        deferral_backoff_max_seconds=backoff_seconds,
        janitor=TicketJanitorConfig(cooldown_seconds=0.0),
    )
    return config


class _Run(NamedTuple):
    interceptor: TaskInterceptor
    janitor: TicketJanitor
    store: TicketStore


class _Server:
    """Builds fused-memory's ticket components the way a server run does, and tears them down."""

    def __init__(self, tmp_path: Path, buffer: EventBuffer, taskmaster: _RecordingTaskmaster) -> None:
        self.db_path = tmp_path / 'tickets.db'
        self._buffer = buffer
        self.taskmaster = taskmaster
        self._stores: list[TicketStore] = []
        self._interceptors: list[TaskInterceptor] = []

    async def open_store(self) -> TicketStore:
        store = TicketStore(self.db_path)
        await _bounded(store.initialize())
        self._stores.append(store)
        return store

    def interceptor(
        self, config: FusedMemoryConfig, store: TicketStore, curator: _Curator | None = None,
        **kwargs: Any,
    ) -> TaskInterceptor:
        interceptor = TaskInterceptor(
            cast(Any, self.taskmaster), None, self._buffer, config=config,
            ticket_store=store, curator=cast(TaskCurator | None, curator), **kwargs,
        )
        self._interceptors.append(interceptor)
        return interceptor

    async def boot(
        self, config: FusedMemoryConfig, curator: _Curator, *, known_projects: dict[str, str] | None = None,
    ) -> _Run:
        store = await self.open_store()
        interceptor = self.interceptor(config, store, curator)
        await _bounded(interceptor.start())
        janitor = _build_ticket_janitor(config, store, interceptor, known_projects or {})
        assert janitor is not None
        return _Run(interceptor, janitor, store)

    async def close(self) -> None:
        for interceptor in self._interceptors:
            await _bounded(interceptor.close())
        for store in self._stores:
            await store.close()


@pytest.fixture(autouse=True)
def _combine_audit_in_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv('DARK_FACTORY_DATA_DIR', str(tmp_path / 'data'))


@pytest_asyncio.fixture
async def server(tmp_path):
    buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await buffer.initialize()
    built = _Server(tmp_path, buffer, _RecordingTaskmaster())
    yield built
    await built.close()
    await buffer.close()


async def _submit(interceptor: TaskInterceptor, title: str, *, metadata: Any = None) -> str:
    response = await _bounded(interceptor.submit_task(
        PROJECT_ROOT, title=title, description=f'{title} description', metadata=metadata,
    ))
    return response['ticket']


async def _resolve(interceptor: TaskInterceptor, ticket_id: str) -> dict:
    return await _bounded(
        interceptor.resolve_ticket(ticket_id, PROJECT_ROOT, timeout_seconds=RESOLVE_TIMEOUT_SECONDS),
    )


def _parked_answer(reason: DeferReason = DeferReason.ALL_ACCOUNTS_CAPPED) -> dict:
    return {'status': AWAITING_CURATION, 'reason': f'curation-deferred: {reason}'}


async def _park(interceptor: TaskInterceptor, title: str, **kwargs: Any) -> str:
    ticket_id = await _submit(interceptor, title, **kwargs)
    assert await _resolve(interceptor, ticket_id) == _parked_answer()
    return ticket_id


async def _row(store: TicketStore, ticket_id: str) -> dict:
    row = await _bounded(store.get(ticket_id))
    assert row is not None
    return row


def _age_ticket(db_path: Path, ticket_id: str, created_at: datetime) -> None:
    with sqlite3.connect(db_path, timeout=5) as conn:
        conn.execute(
            'UPDATE tickets SET created_at = ? WHERE ticket_id = ?',
            (created_at.isoformat(), ticket_id),
        )


async def _let_tiny_backoff_pass() -> None:
    await asyncio.sleep(TINY_BACKOFF_SECONDS * 5)


_REWRITE = RewrittenTask(
    title='Existing task, widened', description='both asks', details='merged',
    files_to_modify=['a.py'], priority='high',
)


def _combine_into_existing(taskmaster: _RecordingTaskmaster) -> CuratorDecision:
    taskmaster.tasks['7'] = {'id': '7', 'title': 'Existing task', 'status': 'pending', 'metadata': {}}
    return CuratorDecision(
        action='combine', target_id='7', target_fingerprint=normalize_title('Existing task'),
        rewritten_task=_REWRITE, justification='overlaps 7',
    )


@pytest.mark.asyncio
async def test_capped_ticket_gets_a_real_combine_verdict_when_capacity_returns(server):
    curator = _Curator(_in_turn({'Overlapping work': [CAPPED, _combine_into_existing(server.taskmaster)]}))
    interceptor, janitor, store = await server.boot(_config(), curator)
    ticket_id = await _park(interceptor, 'Overlapping work')

    await _let_tiny_backoff_pass()
    await _bounded(janitor.tick())

    assert await _resolve(interceptor, ticket_id) == {
        'status': 'combined', 'task_id': '7', 'reason': 'combine: overlaps 7',
    }
    row = await _row(store, ticket_id)
    assert (row['status'], row['task_id'], row['attempts']) == ('combined', '7', 1)
    assert server.taskmaster.add_task_calls == []
    assert [call['task_id'] for call in server.taskmaster.update_task_calls] == ['7']


@pytest.mark.asyncio
async def test_a_long_outage_never_creates(server):
    curator = _Curator(lambda title: CAPPED)
    interceptor, janitor, store = await server.boot(_config(), curator)
    ticket_id = await _park(interceptor, 'Capped work')
    submitted_at = datetime.fromisoformat((await _row(store, ticket_id))['created_at'])
    ticks = 12
    simulated_step = 2 * OLD_TERMINAL_BOUND / ticks

    for tick in range(1, ticks + 1):
        _age_ticket(server.db_path, ticket_id, submitted_at - tick * simulated_step)
        await _let_tiny_backoff_pass()
        await _bounded(janitor.tick())
        assert await _resolve(interceptor, ticket_id) == _parked_answer()

    row = await _row(store, ticket_id)
    age = datetime.now(UTC) - datetime.fromisoformat(row['created_at'])
    assert age > 2 * OLD_TERMINAL_BOUND
    assert (row['status'], row['task_id'], row['resolved_at'], row['attempts']) == (
        AWAITING_CURATION, None, None, ticks + 1,
    )
    assert curator.asked == ['Capped work'] * (ticks + 1)
    assert server.taskmaster.add_task_calls == []


@pytest.mark.asyncio
async def test_parked_ticket_survives_restart_and_is_curated_after(server):
    config = _config(backoff_seconds=3600.0)
    first, first_janitor, first_store = await server.boot(config, _Curator(lambda title: CAPPED))
    ticket_id = await _park(first, 'Capped work')
    await _bounded(first_janitor.tick())
    assert (await _row(first_store, ticket_id))['status'] == AWAITING_CURATION
    await _bounded(first.close())

    curator = _Curator(lambda title: CuratorDecision(action='create', justification='nothing like it'))
    second, second_janitor, second_store = await server.boot(config, curator)
    assert await _resolve(second, ticket_id) == _parked_answer()
    assert curator.asked == []

    await _bounded(second_janitor.tick())

    assert await _resolve(second, ticket_id) == {
        'status': 'created', 'task_id': '101', 'reason': 'create: nothing like it',
    }
    assert (await _row(second_store, ticket_id))['attempts'] == 1


@pytest.mark.asyncio
async def test_requeued_in_flight_ticket_is_reparked_on_restart_not_failed(server):
    first_curator = _Curator(_in_turn({'Capped work': [CAPPED, HOLD]}))
    first, first_janitor, first_store = await server.boot(_config(), first_curator)
    ticket_id = await _park(first, 'Capped work')
    await _let_tiny_backoff_pass()
    await _bounded(first_janitor.tick())
    await _bounded(first_curator.holding.wait())
    assert (await _row(first_store, ticket_id))['status'] == 'pending'
    await _bounded(first.close())

    curator = _Curator(lambda title: CuratorDecision(action='create', justification='nothing like it'))
    second, second_janitor, second_store = await server.boot(_config(), curator)
    row = await _row(second_store, ticket_id)
    assert (row['status'], row['attempts'], row['next_attempt_at']) == (AWAITING_CURATION, 1, None)
    assert await _resolve(second, ticket_id) == _parked_answer()

    await _bounded(second_janitor.tick())

    assert (await _resolve(second, ticket_id))['status'] == 'created'
    assert (await _row(second_store, ticket_id))['attempts'] == 1


@pytest.mark.asyncio
async def test_never_deferred_pending_ticket_still_fails_server_restart(server):
    first_curator = _Curator(lambda title: HOLD)
    first, _, _ = await server.boot(_config(), first_curator)
    ticket_id = await _submit(first, 'Slow work')
    await _bounded(first_curator.holding.wait())
    await _bounded(first.close())

    second, second_janitor, second_store = await server.boot(_config(), _Curator(lambda title: CAPPED))
    await _bounded(second_janitor.tick())

    assert await _resolve(second, ticket_id) == {'status': 'failed', 'reason': 'server_restart'}
    row = await _row(second_store, ticket_id)
    assert (row['status'], row['attempts']) == ('failed', 0)
    assert server.taskmaster.add_task_calls == []


async def _frozen_gate() -> UsageGate:
    """A real one-account gate whose only account has just hit its cap, for an hour."""
    usage_cap = UsageCapConfig(
        accounts=[AccountConfig(name='acct-a', oauth_token_env='TEST_TOKEN_ACCT_A')],
        wait_for_reset=False,
    )
    with patch.dict(os.environ, {'TEST_TOKEN_ACCT_A': 'fake-token-acct-a'}):
        gate = UsageGate(usage_cap)
    async with gate.invoke_slot() as slot:
        slot.report(CapHit(resets_at=datetime.now(UTC) + timedelta(hours=1), reason='test cap'))
    return gate


async def _an_empty_corpus(
    self: TaskCurator, candidate: CandidateTask, project_id: str, project_root: str,
) -> tuple[list, dict[str, int], PoolWithheld]:
    return [], {'anchor': 0, 'module': 0, 'embedding': 0, 'dependency': 0}, PoolWithheld()


@pytest.mark.asyncio
async def test_submitter_never_hangs_on_a_frozen_pool(server):
    gate = await _frozen_gate()
    config = _config()
    config.taskmaster = None
    assert config.curator.enabled
    store = await server.open_store()
    interceptor = server.interceptor(config, store, usage_gate=gate)
    await _bounded(interceptor.start())

    try:
        with patch.object(TaskCurator, '_build_corpus', _an_empty_corpus):
            ticket_id = await _submit(interceptor, 'Frozen work')
            answer = await _resolve(interceptor, ticket_id)
    finally:
        await _bounded(gate.shutdown())

    assert answer == _parked_answer()
    row = await _row(store, ticket_id)
    assert (row['status'], row['attempts'], row['defer_reason']) == (
        AWAITING_CURATION, 1, DeferReason.ALL_ACCOUNTS_CAPPED.value,
    )
    assert server.taskmaster.add_task_calls == []


def _backlog_curator() -> _Curator:
    by_group = {'capped': CAPPED, 'auth': AUTH_FAILED}
    return _Curator(lambda title: by_group[title.split('-')[0]])


def _group_reason(title: str) -> DeferReason:
    return DeferReason.ACCOUNTS_AUTH_FAILED if title.startswith('auth') else DeferReason.ALL_ACCOUNTS_CAPPED


@pytest.mark.asyncio
async def test_one_backlog_escalation_per_outage(server, tmp_path):
    config = _config(backoff_seconds=3600.0)
    assert config.curator.janitor.curation_backlog_escalation_min_attempts == 2
    known_projects = {PROJECT_ID: str(tmp_path)}
    queue = EscalationQueue(escalation_queue_dir(str(tmp_path)))

    def pending_per_group() -> dict[DeferReason, list]:
        return {
            reason: queue.get_by_task(curation_backlog_anchor(reason), status='pending')
            for reason in DeferReason
        }

    interceptor, janitor, _ = await server.boot(config, _backlog_curator(), known_projects=known_projects)
    titles = [f'capped-{n}' for n in range(8)] + [f'auth-{n}' for n in range(3)]
    tickets = {title: await _submit(interceptor, title) for title in titles}
    for title, ticket_id in tickets.items():
        assert await _resolve(interceptor, ticket_id) == _parked_answer(_group_reason(title))
    for _ in range(2):
        await _bounded(janitor.tick())
    assert pending_per_group() == {reason: [] for reason in DeferReason}

    for _restart in range(2):
        await _bounded(interceptor.close())
        interceptor, janitor, _ = await server.boot(config, _backlog_curator(), known_projects=known_projects)
        for _ in range(3):
            await _bounded(janitor.tick())
            for title, ticket_id in tickets.items():
                assert await _resolve(interceptor, ticket_id) == _parked_answer(_group_reason(title))

    pending = pending_per_group()
    assert {reason: len(records) for reason, records in pending.items()} == {
        DeferReason.ALL_ACCOUNTS_CAPPED: 1, DeferReason.ACCOUNTS_AUTH_FAILED: 1,
    }
    [capped], [auth_failed] = pending[DeferReason.ALL_ACCOUNTS_CAPPED], pending[DeferReason.ACCOUNTS_AUTH_FAILED]
    assert (capped.severity, auth_failed.severity) == ('info', 'blocking')
    assert server.taskmaster.add_task_calls == []


_CALLER_METADATA = {'source': 'acceptance-test', 'spawn_context': 'manual'}


@pytest.mark.asyncio
async def test_a_fail_open_create_carries_the_dedupe_skipped_marker(server):
    curator = _Curator(_in_turn({'Work': [CAPPED, CuratorFailureError('llm down')]}))
    interceptor, janitor, _ = await server.boot(_config(), curator)
    ticket_id = await _park(interceptor, 'Work', metadata=_CALLER_METADATA)
    curator.batch_raises.append(CuratorFailureError('batch down'))

    await _let_tiny_backoff_pass()
    await _bounded(janitor.tick())

    reason = 'curator-failed: CuratorFailureError: llm down'
    assert await _resolve(interceptor, ticket_id) == {
        'status': 'created', 'task_id': '101', 'reason': f'create: {reason}',
    }
    [created] = server.taskmaster.add_task_calls
    metadata = json.loads(created['metadata'])
    marker = metadata.pop(DEDUPE_SKIPPED_KEY)
    assert metadata == _CALLER_METADATA
    assert marker['reason'] == reason
    assert datetime.fromisoformat(marker['at']).utcoffset() == timedelta(0)
