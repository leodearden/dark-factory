"""The ticket worker under a curator that can defer: parking, requeue and the dedupe_skipped marker.

Driven through the interceptor's public surface (``submit_task``,
``resolve_ticket``, ``cancel_ticket``, ``requeue_awaiting_curation_ticket``)
against a real :class:`TicketStore`, a recording task backend, and a scripted
curator injected through the interceptor's ``curator=`` constructor argument.
"""

import asyncio
import json
from collections import deque
from typing import Any, cast

import pytest
import pytest_asyncio

from fused_memory.middleware.task_curator import (
    CandidateTask,
    CuratorDecision,
    PreparedCandidate,
    RewrittenTask,
    TaskCurator,
    normalize_title,
)
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

PROJECT_ROOT = '/project'
RESOLVE_DEADLINE_SECONDS = 30.0

Verdict = CuratorDecision | BaseException


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

    def added_metadata(self) -> list[Any]:
        return [
            json.loads(call['metadata']) if call.get('metadata') else None
            for call in self.add_task_calls
        ]


class _ScriptedCurator:
    """A curator double answering each candidate, by title, with its next scripted verdict.

    A batch-level script entry (``batch_raises``) makes the next
    ``curate_batch_prepared`` call raise instead of answering.
    """

    def __init__(self, script: dict[str, list[Verdict]]) -> None:
        self._script = {title: deque(verdicts) for title, verdicts in script.items()}
        self.batch_raises: deque[BaseException] = deque()
        self.hold_prepare: dict[str, asyncio.Event] = {}
        self.created: list[tuple[str, str]] = []

    def _next(self, title: str) -> CuratorDecision:
        verdict = self._script[title].popleft()
        if isinstance(verdict, BaseException):
            raise verdict
        return verdict

    async def prepare_candidate(
        self, candidate: CandidateTask, project_id: str, project_root: str,
    ) -> PreparedCandidate:
        hold = self.hold_prepare.get(candidate.title)
        if hold is not None:
            await hold.wait()
        return PreparedCandidate(candidate=candidate, pool=[], pool_sizes={}, prompt_tokens=0)

    async def curate_batch_prepared(
        self, prepared: list[PreparedCandidate], project_id: str, project_root: str,
    ) -> list[CuratorDecision]:
        if self.batch_raises:
            raise self.batch_raises.popleft()
        return [self._next(p.candidate.title) for p in prepared]

    async def curate(
        self, candidate: CandidateTask, project_id: str, project_root: str,
    ) -> CuratorDecision:
        return self._next(candidate.title)

    def note_created(self, project_id: str, candidate: CandidateTask, task_id: str) -> None:
        self.created.append((candidate.title, task_id))

    async def record_task(self, task_id: str, candidate: CandidateTask, project_id: str) -> None:
        return None

    async def reembed_task(self, task_id: str, candidate: CandidateTask, project_id: str) -> None:
        return None

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _combine_audit_in_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv('DARK_FACTORY_DATA_DIR', str(tmp_path / 'data'))


@pytest_asyncio.fixture
async def store(tmp_path):
    ticket_store = TicketStore(tmp_path / 'tickets.db')
    await ticket_store.initialize()
    yield ticket_store
    await ticket_store.close()


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buffer = EventBuffer(db_path=tmp_path / 'events.db', buffer_size_threshold=100)
    await buffer.initialize()
    yield buffer
    await buffer.close()


@pytest.fixture
def taskmaster() -> _RecordingTaskmaster:
    return _RecordingTaskmaster()


@pytest_asyncio.fixture
async def make_interceptor(taskmaster, event_buffer, store):
    built: list[TaskInterceptor] = []

    def _make(curator: _ScriptedCurator | None, **kwargs: Any) -> TaskInterceptor:
        interceptor = TaskInterceptor(
            cast(Any, taskmaster), None, event_buffer,
            ticket_store=store, curator=cast(TaskCurator | None, curator), **kwargs,
        )
        built.append(interceptor)
        return interceptor

    yield _make
    for interceptor in built:
        for worker in list(interceptor._worker_tasks.values()):
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)


async def _submit(interceptor: TaskInterceptor, title: str, *, metadata: Any = None) -> str:
    response = await interceptor.submit_task(
        PROJECT_ROOT, title=title, description=f'{title} description', metadata=metadata,
    )
    return response['ticket']


async def _resolve(interceptor: TaskInterceptor, ticket_id: str) -> dict:
    return await asyncio.wait_for(
        interceptor.resolve_ticket(ticket_id, PROJECT_ROOT, timeout_seconds=RESOLVE_DEADLINE_SECONDS),
        timeout=RESOLVE_DEADLINE_SECONDS + 5,
    )


async def _row_outcome(store: TicketStore, ticket_id: str) -> dict:
    row = await store.get(ticket_id)
    assert row is not None
    return {
        'status': row['status'],
        'task_id': row['task_id'],
        'reason': row['reason'],
        'result_json': json.loads(row['result_json']) if row['result_json'] else None,
    }


_REWRITE = RewrittenTask(
    title='Existing task, widened', description='both asks', details='merged',
    files_to_modify=['a.py'], priority='high',
)

_CALLER_METADATA = {'source': 'parking-test', 'spawn_context': 'manual'}


class TestUncappedCurationUnchanged:
    """Characterises every healthy verdict end to end; written before the defer seam existed."""

    @pytest.mark.asyncio
    async def test_healthy_create(self, make_interceptor, store, taskmaster):
        curator = _ScriptedCurator({'New work': [
            CuratorDecision(action='create', justification='nothing like it'),
        ]})
        interceptor = make_interceptor(curator)

        ticket_id = await _submit(interceptor, 'New work', metadata=_CALLER_METADATA)
        answer = await _resolve(interceptor, ticket_id)

        assert answer == {'status': 'created', 'task_id': '101', 'reason': 'create: nothing like it'}
        assert await _row_outcome(store, ticket_id) == {
            'status': 'created', 'task_id': '101', 'reason': 'create: nothing like it',
            'result_json': {'id': '101', 'title': 'New work'},
        }
        assert taskmaster.add_task_calls == [{
            'project_root': PROJECT_ROOT,
            'metadata': json.dumps(_CALLER_METADATA),
            'title': 'New work',
            'description': 'New work description',
        }]

    @pytest.mark.asyncio
    async def test_drop_onto_an_existing_task(self, make_interceptor, store, taskmaster):
        curator = _ScriptedCurator({'Dup work': [
            CuratorDecision(action='drop', target_id='7', justification='same as 7'),
        ]})
        interceptor = make_interceptor(curator)

        ticket_id = await _submit(interceptor, 'Dup work', metadata=_CALLER_METADATA)
        answer = await _resolve(interceptor, ticket_id)

        assert answer == {'status': 'combined', 'task_id': '7', 'reason': 'drop: same as 7'}
        assert await _row_outcome(store, ticket_id) == {
            'status': 'combined', 'task_id': '7', 'reason': 'drop: same as 7',
            'result_json': {
                'id': '7', 'title': 'Dup work', 'deduplicated': True,
                'action': 'drop', 'justification': 'same as 7',
            },
        }
        assert taskmaster.add_task_calls == []

    @pytest.mark.asyncio
    async def test_combine_into_an_existing_task(self, make_interceptor, store, taskmaster):
        taskmaster.tasks['7'] = {'id': '7', 'title': 'Existing task', 'status': 'pending', 'metadata': {}}
        curator = _ScriptedCurator({'Overlapping work': [
            CuratorDecision(
                action='combine', target_id='7',
                target_fingerprint=normalize_title('Existing task'),
                rewritten_task=_REWRITE, justification='overlaps 7',
            ),
        ]})
        interceptor = make_interceptor(curator)

        ticket_id = await _submit(interceptor, 'Overlapping work', metadata=_CALLER_METADATA)
        answer = await _resolve(interceptor, ticket_id)

        assert answer == {'status': 'combined', 'task_id': '7', 'reason': 'combine: overlaps 7'}
        assert await _row_outcome(store, ticket_id) == {
            'status': 'combined', 'task_id': '7', 'reason': 'combine: overlaps 7',
            'result_json': {
                'id': '7', 'title': 'Existing task, widened', 'deduplicated': True,
                'action': 'combine', 'justification': 'overlaps 7',
            },
        }
        assert taskmaster.add_task_calls == []
        assert [call['task_id'] for call in taskmaster.update_task_calls] == ['7']

    @pytest.mark.asyncio
    async def test_refuse(self, make_interceptor, store, taskmaster):
        curator = _ScriptedCurator({'Dead premise': [
            CuratorDecision(action='refuse', justification='premise cancelled'),
        ]})
        interceptor = make_interceptor(curator)

        ticket_id = await _submit(interceptor, 'Dead premise', metadata=_CALLER_METADATA)
        answer = await _resolve(interceptor, ticket_id)

        assert answer == {
            'status': 'refused', 'reason': 'refused (no task created): premise cancelled',
        }
        assert await _row_outcome(store, ticket_id) == {
            'status': 'refused', 'task_id': None,
            'reason': 'refused (no task created): premise cancelled',
            'result_json': {
                'id': None, 'title': 'Dead premise', 'created': False, 'refused': True,
                'action': 'refuse', 'justification': 'premise cancelled',
            },
        }
        assert taskmaster.add_task_calls == []

    @pytest.mark.asyncio
    async def test_route_deterministic(self, make_interceptor, store, taskmaster):
        curator = _ScriptedCurator({'Restart the server': [
            CuratorDecision(
                action='route_deterministic',
                justification='operational-ask-registry: restart: ops',
            ),
        ]})
        interceptor = make_interceptor(curator)

        ticket_id = await _submit(interceptor, 'Restart the server', metadata=_CALLER_METADATA)
        answer = await _resolve(interceptor, ticket_id)

        reason = 'create: route_deterministic: operational-ask-registry: restart: ops'
        assert answer == {'status': 'created', 'task_id': '101', 'reason': reason}
        assert await _row_outcome(store, ticket_id) == {
            'status': 'created', 'task_id': '101', 'reason': reason,
            'result_json': {'id': '101', 'title': 'Restart the server'},
        }
        assert taskmaster.added_metadata() == [{
            **_CALLER_METADATA, 'task_kind': 'deterministic', 'always_escalates': True,
        }]
