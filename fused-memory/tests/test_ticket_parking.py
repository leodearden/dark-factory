"""The ticket worker under a curator that can defer: parking, requeue and the dedupe_skipped marker.

Driven through the interceptor's public surface (``submit_task``,
``resolve_ticket``, ``cancel_ticket``, ``requeue_awaiting_curation_ticket``)
against a real :class:`TicketStore`, a recording task backend, and a scripted
curator injected through the interceptor's ``curator=`` constructor argument.
"""

import asyncio
import json
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, cast
from unittest.mock import ANY, AsyncMock, patch

import pytest
import pytest_asyncio
from _curator_helpers import (
    agent_result,
    escalator_double,
    invoke_by_role,
    make_config,
    must_not_invoke,
    pool_with_ids,
    prepared,
    submit_as_one_batch,
    tripwire,
    zot_agent_result,
)
from shared.cli_invoke import AgentResult, AllAccountsCappedException
from shared.usage_gate import PoolFrozen
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.middleware.candidate_key import normalize_title
from fused_memory.middleware.curation_deferral import DEDUPE_SKIPPED_KEY, DeferReason
from fused_memory.middleware.curator_types import (
    CandidateTask,
    CuratorDecision,
    CuratorFailureError,
    PreparedCandidate,
    RewrittenTask,
)
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStatus, TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

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
    ``curate_prepared`` call raise instead of answering.
    """

    def __init__(self, script: Mapping[str, Sequence[Verdict]]) -> None:
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

    async def curate_prepared(
        self, prepared: list[PreparedCandidate], project_id: str, project_root: str,
    ) -> list[CuratorDecision]:
        if self.batch_raises:
            raise self.batch_raises.popleft()
        return [self._next(p.candidate.title) for p in prepared]

    def note_created(self, project_id: str, candidate: CandidateTask, task_id: str) -> None:
        self.created.append((candidate.title, task_id))

    async def record_task(self, task_id: str, candidate: CandidateTask, project_id: str) -> None:
        return None

    async def reembed_task(self, task_id: str, candidate: CandidateTask, project_id: str) -> None:
        return None

    async def close(self) -> None:
        return None


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
        await interceptor.close()


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
_FILED_METADATA = {**_CALLER_METADATA, 'declared_scope': ANY}


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
        [call] = taskmaster.add_task_calls
        assert {**call, 'metadata': json.loads(call['metadata'])} == {
            'project_root': PROJECT_ROOT,
            'metadata': _FILED_METADATA,
            'title': 'New work',
            'description': 'New work description',
        }

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
            **_FILED_METADATA, 'task_kind': 'deterministic', 'always_escalates': True,
            'pure_gate_from': 'operational_ask_registry',
        }]


CAPPED = CuratorDecision.deferred(DeferReason.ALL_ACCOUNTS_CAPPED)
AUTH_FAILED = CuratorDecision.deferred(DeferReason.ACCOUNTS_AUTH_FAILED)


def _parked_answer(ticket_id: str, reason: DeferReason) -> dict:
    return {
        'status': TicketStatus.AWAITING_CURATION, 'ticket': ticket_id,
        'reason': f'curation-deferred: {reason}',
    }


async def _park(interceptor: TaskInterceptor, title: str) -> str:
    ticket_id = await _submit(interceptor, title)
    assert await _resolve(interceptor, ticket_id) == _parked_answer(ticket_id, DeferReason.ALL_ACCOUNTS_CAPPED)
    return ticket_id


async def _parked_row(store: TicketStore, ticket_id: str) -> dict:
    row = await store.get(ticket_id)
    assert row is not None
    return {
        key: row[key]
        for key in ('status', 'task_id', 'reason', 'result_json', 'resolved_at', 'attempts', 'defer_reason')
    }


def _parked(reason: DeferReason, *, attempts: int = 1) -> dict:
    return {
        'status': TicketStatus.AWAITING_CURATION, 'task_id': None, 'reason': f'curation-deferred: {reason}',
        'result_json': None, 'resolved_at': None, 'attempts': attempts, 'defer_reason': reason.value,
    }


def _frozen_pool() -> PoolFrozen:
    return PoolFrozen(account_count=1, capped_account_names=('max-a',), auth_failed_account_names=())


class TestParking:
    @pytest.mark.asyncio
    async def test_a_defer_parks_the_ticket_and_creates_nothing(self, make_interceptor, store, taskmaster):
        interceptor = make_interceptor(_ScriptedCurator({'Capped work': [CAPPED]}))

        ticket_id = await _park(interceptor, 'Capped work')

        assert await _parked_row(store, ticket_id) == _parked(DeferReason.ALL_ACCOUNTS_CAPPED)
        assert taskmaster.add_task_calls == []

    @pytest.mark.asyncio
    async def test_a_waiter_is_woken_with_awaiting_curation_and_no_task_id(self, make_interceptor, store):
        curator = _ScriptedCurator({'Capped work': [CAPPED]})
        curator.hold_prepare['Capped work'] = release = asyncio.Event()
        interceptor = make_interceptor(curator)

        ticket_id = await _submit(interceptor, 'Capped work')
        waiter = asyncio.create_task(_resolve(interceptor, ticket_id))
        await asyncio.sleep(0.2)
        assert not waiter.done()
        release.set()

        assert await waiter == _parked_answer(ticket_id, DeferReason.ALL_ACCOUNTS_CAPPED)

    @pytest.mark.asyncio
    async def test_resolve_on_a_parked_ticket_returns_at_once(self, make_interceptor):
        interceptor = make_interceptor(_ScriptedCurator({'Capped work': [CAPPED]}))
        ticket_id = await _park(interceptor, 'Capped work')

        answer = await asyncio.wait_for(
            interceptor.resolve_ticket(ticket_id, PROJECT_ROOT, timeout_seconds=None),
            timeout=RESOLVE_DEADLINE_SECONDS,
        )

        assert answer == _parked_answer(ticket_id, DeferReason.ALL_ACCOUNTS_CAPPED)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'pool_error',
        [
            _frozen_pool(),
            AllAccountsCappedException(retries=3, elapsed_secs=120.0, label='curator'),
        ],
        ids=['pool-frozen', 'all-accounts-capped'],
    )
    async def test_a_pool_error_from_the_llm_call_parks_never_creates(
        self, make_interceptor, store, taskmaster, pool_error,
    ):
        interceptor = make_interceptor(_real_curator(invoke=AsyncMock(side_effect=pool_error)))

        ticket_id = await _park(interceptor, 'Capped work')

        assert await _parked_row(store, ticket_id) == _parked(DeferReason.ALL_ACCOUNTS_CAPPED)
        assert taskmaster.add_task_calls == []

    @pytest.mark.asyncio
    async def test_a_pool_error_raised_out_of_the_pipeline_parks_never_creates(
        self, make_interceptor, store, taskmaster,
    ):
        """The worker's pool arm: ``curate_prepared`` may raise a pool error, and it parks."""
        curator = _ScriptedCurator({'Capped work': []})
        curator.batch_raises.append(_frozen_pool())
        interceptor = make_interceptor(curator)

        ticket_id = await _park(interceptor, 'Capped work')

        assert await _parked_row(store, ticket_id) == _parked(DeferReason.ALL_ACCOUNTS_CAPPED)
        assert taskmaster.add_task_calls == []

    @pytest.mark.asyncio
    async def test_a_drop_onto_a_parked_sibling_is_parked_too(self, make_interceptor, store, taskmaster):
        curator = _ScriptedCurator({
            'Sibling': [AUTH_FAILED],
            'Sibling again': [CuratorDecision(
                action='drop', batch_target_index=0, justification='same as the sibling',
            )],
        })
        curator.hold_prepare['Sibling'] = release = asyncio.Event()
        interceptor = make_interceptor(curator)

        sibling = await _submit(interceptor, 'Sibling')
        duplicate = await _submit(interceptor, 'Sibling again')
        release.set()
        answers = [await _resolve(interceptor, ticket_id) for ticket_id in (sibling, duplicate)]

        assert answers == [
            _parked_answer(sibling, DeferReason.ACCOUNTS_AUTH_FAILED),
            _parked_answer(duplicate, DeferReason.ACCOUNTS_AUTH_FAILED),
        ]
        assert await _parked_row(store, duplicate) == _parked(DeferReason.ACCOUNTS_AUTH_FAILED)
        assert taskmaster.add_task_calls == []

    @pytest.mark.asyncio
    async def test_cancel_withdraws_a_parked_ticket(self, make_interceptor, store):
        interceptor = make_interceptor(_ScriptedCurator({'Capped work': [CAPPED]}))
        ticket_id = await _park(interceptor, 'Capped work')

        cancelled = await interceptor.cancel_ticket(ticket_id)

        assert cancelled == {'status': 'cancelled', 'ticket_id': ticket_id}
        assert await _resolve(interceptor, ticket_id) == {'status': 'cancelled', 'reason': 'user_cancelled'}


def _combine_into_existing(taskmaster: _RecordingTaskmaster) -> CuratorDecision:
    taskmaster.tasks['7'] = {'id': '7', 'title': 'Existing task', 'status': 'pending', 'metadata': {}}
    return CuratorDecision(
        action='combine', target_id='7', target_fingerprint=normalize_title('Existing task'),
        rewritten_task=_REWRITE, justification='overlaps 7',
    )


class TestRequeue:
    @pytest.mark.asyncio
    async def test_requeue_hands_back_a_parked_ticket_for_a_real_combine_verdict(
        self, make_interceptor, store, taskmaster,
    ):
        curator = _ScriptedCurator({'Overlapping work': [CAPPED, _combine_into_existing(taskmaster)]})
        interceptor = make_interceptor(curator)
        ticket_id = await _park(interceptor, 'Overlapping work')

        requeued = await interceptor.requeue_awaiting_curation_ticket('project', ticket_id)

        assert requeued is True
        assert await _resolve(interceptor, ticket_id) == {
            'status': 'combined', 'task_id': '7', 'reason': 'combine: overlaps 7',
        }
        row = await store.get(ticket_id)
        assert row is not None and row['attempts'] == 1
        assert taskmaster.add_task_calls == []

    @pytest.mark.asyncio
    async def test_requeue_refuses_anything_but_a_parked_ticket(self, make_interceptor, store, taskmaster):
        curator = _ScriptedCurator({
            'New work': [CuratorDecision(action='create', justification='novel')],
            'Capped work': [CAPPED, CuratorDecision(action='create', justification='novel later')],
        })
        interceptor = make_interceptor(curator)
        created = await _submit(interceptor, 'New work')
        await _resolve(interceptor, created)
        parked = await _park(interceptor, 'Capped work')

        assert await interceptor.requeue_awaiting_curation_ticket('project', created) is False
        assert await interceptor.requeue_awaiting_curation_ticket('project', 'tkt_unknown') is False
        assert await interceptor.requeue_awaiting_curation_ticket('project', parked) is True
        assert await interceptor.requeue_awaiting_curation_ticket('project', parked) is False
        assert (await _resolve(interceptor, parked))['status'] == 'created'
        assert len(taskmaster.add_task_calls) == 2

    @pytest.mark.asyncio
    async def test_requeue_after_close_leaves_the_row_parked(self, make_interceptor, tmp_path):
        interceptor = make_interceptor(_ScriptedCurator({'Capped work': [CAPPED]}))
        ticket_id = await _park(interceptor, 'Capped work')
        await interceptor.close()

        requeued = await interceptor.requeue_awaiting_curation_ticket('project', ticket_id)

        reopened = TicketStore(tmp_path / 'tickets.db')
        await reopened.initialize()
        try:
            assert requeued is False
            assert await _parked_row(reopened, ticket_id) == _parked(DeferReason.ALL_ACCOUNTS_CAPPED)
        finally:
            await reopened.close()

    @pytest.mark.asyncio
    async def test_requeue_marks_the_worker_alive_before_enqueue(self, make_interceptor, taskmaster):
        script = {'Overlapping work': [CAPPED, _combine_into_existing(taskmaster)]}
        ticket_id = await _park(make_interceptor(_ScriptedCurator(script)), 'Overlapping work')
        restarted = make_interceptor(_ScriptedCurator({'Overlapping work': script['Overlapping work'][1:]}))
        assert restarted.is_worker_alive('project') is False

        requeued = await restarted.requeue_awaiting_curation_ticket('project', ticket_id)
        alive_on_return = restarted.is_worker_alive('project')

        assert (requeued, alive_on_return) == (True, True)
        assert (await _resolve(restarted, ticket_id))['status'] == 'combined'


def _max_turns_failure() -> AgentResult:
    """A failed LLM call that is not a zero-output hang."""
    return AgentResult(success=False, output='', subtype='error_max_turns', turns=12)


def _real_curator(*, invoke: Any, escalator: Any = None) -> TaskCurator:
    """A real TaskCurator whose LLM is *invoke* and whose corpus is never touched.

    ``prepare_candidate`` hands back a bundle over an empty pool, and the
    corpus writes and reads the worker makes after a create are stubbed, so
    nothing reaches Qdrant or an embedder.
    """
    curator = TaskCurator(
        config=make_config(), taskmaster=None,
        escalator=escalator if escalator is not None else escalator_double(), invoke=invoke,
    )

    async def bundle_for(candidate: CandidateTask, *_args: Any) -> PreparedCandidate:
        return prepared(candidate.title, candidate=candidate)

    curator.prepare_candidate = AsyncMock(side_effect=bundle_for)
    curator.record_task = AsyncMock()
    curator.reembed_task = AsyncMock()
    curator.search_corpus = AsyncMock(return_value=[])
    return curator


def _escalator_raising(error: BaseException) -> AsyncMock:
    escalator = escalator_double()
    escalator.report_failure = AsyncMock(side_effect=error)
    return escalator


class TestCuratorFailuresDecidedByThePipeline:
    """Every curator failure is decided inside curate_prepared; the worker only dispatches."""

    @pytest.mark.asyncio
    async def test_a_prepare_candidate_exception_parks_corpus_failed_and_never_reaches_the_llm(
        self, make_interceptor, store, taskmaster,
    ):
        curator = _real_curator(invoke=must_not_invoke())
        curator.prepare_candidate = AsyncMock(side_effect=RuntimeError('qdrant gone'))
        interceptor = make_interceptor(curator)

        ticket_id = await _submit(interceptor, 'Work')

        assert await _resolve(interceptor, ticket_id) == _parked_answer(ticket_id, DeferReason.CORPUS_FAILED)
        assert await _parked_row(store, ticket_id) == _parked(DeferReason.CORPUS_FAILED)
        assert taskmaster.add_task_calls == []

    @pytest.mark.asyncio
    async def test_a_failed_batch_bisects_and_both_tickets_park_llm_failed(
        self, make_interceptor, store, taskmaster,
    ):
        invoke = invoke_by_role(single=_max_turns_failure(), batch=_max_turns_failure())
        curator = _real_curator(invoke=invoke)
        interceptor = make_interceptor(curator)

        ticket_ids = await submit_as_one_batch(interceptor, curator, [
            {'title': 'First work', 'description': 'first'},
            {'title': 'Second work', 'description': 'second'},
        ], project_root=PROJECT_ROOT)

        for ticket_id in ticket_ids:
            assert await _resolve(interceptor, ticket_id) == _parked_answer(ticket_id, DeferReason.LLM_FAILED)
            assert await _parked_row(store, ticket_id) == _parked(DeferReason.LLM_FAILED)
        assert [call.kwargs['role'] for call in invoke.await_args_list] == [
            'task_curator_batch', 'task_curator', 'task_curator',
        ]
        assert taskmaster.add_task_calls == []

    @pytest.mark.asyncio
    async def test_a_zero_output_hang_with_no_orchestrator_creates_marked(self, make_interceptor, taskmaster):
        no_orchestrator = CuratorFailureError('no orchestrator is running', zero_output_timeout=True)
        curator = _real_curator(
            invoke=invoke_by_role(
                single=zot_agent_result(), batch=tripwire('one ticket is a batch of one'),
            ),
            escalator=_escalator_raising(no_orchestrator),
        )
        interceptor = make_interceptor(curator)

        ticket_id = await _submit(interceptor, 'Work', metadata=_CALLER_METADATA)

        answer = await _resolve(interceptor, ticket_id)
        assert (answer['status'], answer['task_id']) == ('created', '101')
        reason = answer['reason'].removeprefix('create: ')
        assert reason.startswith('curator-failed: CuratorFailureError')
        [metadata] = taskmaster.added_metadata()
        assert _marker_of(metadata) == {'reason': reason, 'at': metadata[DEDUPE_SKIPPED_KEY]['at']}

    @pytest.mark.asyncio
    async def test_a_curator_that_raises_instead_of_deciding_parks_curator_raised(
        self, make_interceptor, store, taskmaster,
    ):
        curator = _real_curator(
            invoke=invoke_by_role(single=_max_turns_failure(), batch=_max_turns_failure()),
            escalator=_escalator_raising(RuntimeError('escalation queue corrupt')),
        )
        interceptor = make_interceptor(curator)

        ticket_id = await _submit(interceptor, 'Work')

        assert await _resolve(interceptor, ticket_id) == _parked_answer(ticket_id, DeferReason.CURATOR_RAISED)
        assert await _parked_row(store, ticket_id) == _parked(DeferReason.CURATOR_RAISED)
        assert taskmaster.add_task_calls == []

    @pytest.mark.asyncio
    async def test_a_curator_answering_fewer_decisions_than_bundles_parks_curator_raised(
        self, make_interceptor, store, taskmaster,
    ):
        curator = _real_curator(invoke=must_not_invoke())
        curator.curate_prepared = AsyncMock(return_value=[])
        interceptor = make_interceptor(curator)

        ticket_id = await _submit(interceptor, 'Work')

        assert await _resolve(interceptor, ticket_id) == _parked_answer(ticket_id, DeferReason.CURATOR_RAISED)
        assert await _parked_row(store, ticket_id) == _parked(DeferReason.CURATOR_RAISED)
        assert taskmaster.add_task_calls == []

    @pytest.mark.asyncio
    async def test_a_parked_llm_failure_resolves_on_a_later_combine(self, make_interceptor, store, taskmaster):
        taskmaster.tasks['7'] = {'id': '7', 'title': 'Existing task', 'status': 'pending', 'metadata': {}}
        llm_down = True

        def single(_kwargs: dict[str, Any]) -> AgentResult:
            if llm_down:
                return _max_turns_failure()
            return agent_result({
                'action': 'combine', 'target_id': '7', 'target_fingerprint': 'Existing task',
                'justification': 'overlaps 7',
                'rewritten_task': {
                    'title': 'Existing task, widened', 'description': 'both asks', 'details': 'merged',
                },
            })

        curator = _real_curator(invoke=invoke_by_role(
            single=single, batch=tripwire('one ticket is a batch of one'),
        ))

        async def bundle_over_task_7(candidate: CandidateTask, *_args: Any) -> PreparedCandidate:
            return prepared(candidate.title, candidate=candidate, pool=pool_with_ids(('7', 'pending')))

        curator.prepare_candidate = AsyncMock(side_effect=bundle_over_task_7)
        interceptor = make_interceptor(curator)
        ticket_id = await _submit(interceptor, 'Overlapping work')
        assert await _resolve(interceptor, ticket_id) == _parked_answer(ticket_id, DeferReason.LLM_FAILED)

        llm_down = False
        assert await interceptor.requeue_awaiting_curation_ticket('project', ticket_id) is True

        assert await _resolve(interceptor, ticket_id) == {
            'status': 'combined', 'task_id': '7', 'reason': 'combine: overlaps 7',
        }
        assert taskmaster.add_task_calls == []


def _marker_of(metadata: Any) -> dict[str, str] | None:
    marker = metadata.get(DEDUPE_SKIPPED_KEY) if isinstance(metadata, dict) else None
    if marker is None:
        return None
    stamped_at = datetime.fromisoformat(marker['at'])
    assert stamped_at.utcoffset() == timedelta(0), marker
    return marker


async def _created_metadata(interceptor: TaskInterceptor, taskmaster: _RecordingTaskmaster, *titles: str) -> list[Any]:
    ticket_ids = [await _submit(interceptor, title, metadata=_CALLER_METADATA) for title in titles]
    for ticket_id in ticket_ids:
        await _resolve(interceptor, ticket_id)
    return taskmaster.added_metadata()


def _held_batch(script: Mapping[str, Sequence[Verdict]], first: str) -> tuple[_ScriptedCurator, asyncio.Event]:
    curator = _ScriptedCurator(script)
    curator.hold_prepare[first] = release = asyncio.Event()
    return curator, release


async def _batch_created_metadata(
    interceptor: TaskInterceptor, taskmaster: _RecordingTaskmaster, release: asyncio.Event, *titles: str,
) -> list[Any]:
    ticket_ids = [await _submit(interceptor, title, metadata=_CALLER_METADATA) for title in titles]
    release.set()
    for ticket_id in ticket_ids:
        await _resolve(interceptor, ticket_id)
    return taskmaster.added_metadata()


class TestDedupeSkippedMarker:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('script', 'reason'),
        [
            (
                [CuratorDecision(action='create', degraded=True, justification='llm-failed: boom')],
                'llm-failed: boom',
            ),
            (
                [CuratorDecision(action='drop', batch_target_index=5, justification='dup')],
                'batch-target-out-of-range: local=5',
            ),
            (
                [CuratorDecision(action='combine', target_id='404', justification='overlaps 404')],
                'combine-failed: overlaps 404',
            ),
            (
                [CuratorDecision(action='drop', justification='dup of nothing')],
                'drop-failed: dup of nothing',
            ),
        ],
        ids=[
            'curator-degraded-create',
            'batch-target-out-of-range', 'combine-fell-through', 'drop-fell-through',
        ],
    )
    async def test_a_create_no_verdict_judged_is_marked(
        self, make_interceptor, taskmaster, script, reason,
    ):
        curator = _ScriptedCurator({'Work': script})

        [metadata] = await _created_metadata(make_interceptor(curator), taskmaster, 'Work')

        assert _marker_of(metadata) == {'reason': reason, 'at': metadata[DEDUPE_SKIPPED_KEY]['at']}

    @pytest.mark.asyncio
    async def test_a_drop_onto_a_sibling_that_created_nothing_is_marked(self, make_interceptor, taskmaster):
        curator, release = _held_batch({
            'Refused sibling': [CuratorDecision(action='refuse', justification='premise cancelled')],
            'Its duplicate': [CuratorDecision(action='drop', batch_target_index=0, justification='dup')],
        }, first='Refused sibling')

        [metadata] = await _batch_created_metadata(
            make_interceptor(curator), taskmaster, release, 'Refused sibling', 'Its duplicate',
        )

        assert _marker_of(metadata) == {
            'reason': 'batch-sibling-failed: degraded to create', 'at': metadata[DEDUPE_SKIPPED_KEY]['at'],
        }

    @pytest.mark.asyncio
    async def test_a_batch_cycle_coerced_to_create_is_marked(self, make_interceptor, taskmaster):
        curator, release = _held_batch({
            'Left': [CuratorDecision(action='drop', batch_target_index=1, justification='dup')],
            'Right': [CuratorDecision(action='drop', batch_target_index=0, justification='dup')],
        }, first='Left')

        created = await _batch_created_metadata(make_interceptor(curator), taskmaster, release, 'Left', 'Right')

        assert [(_marker_of(m) or {}).get('reason') for m in created] == [
            'batch-cycle: coerced to create', 'batch-cycle: coerced to create',
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'decision',
        [
            CuratorDecision(action='create', justification='nothing like it'),
            CuratorDecision(action='route_deterministic', justification='operational-ask-registry: x'),
        ],
        ids=['healthy-create', 'route-deterministic'],
    )
    async def test_a_create_a_verdict_judged_is_not_marked(self, make_interceptor, taskmaster, decision):
        [metadata] = await _created_metadata(
            make_interceptor(_ScriptedCurator({'Work': [decision]})), taskmaster, 'Work',
        )

        assert _marker_of(metadata) is None

    @pytest.mark.asyncio
    async def test_a_disabled_curator_create_is_not_marked(self, make_interceptor, store, taskmaster):
        config = FusedMemoryConfig()
        config.taskmaster = None
        config.curator.enabled = False
        interceptor = make_interceptor(None, config=config)

        ticket_id = await _submit(interceptor, 'Work', metadata=_CALLER_METADATA)
        answer = await _resolve(interceptor, ticket_id)

        assert answer['reason'] == 'create: curator-unavailable: disabled'
        assert taskmaster.added_metadata() == [_FILED_METADATA]

    @pytest.mark.asyncio
    async def test_marker_merges_into_caller_metadata(self, make_interceptor, taskmaster):
        curator = _ScriptedCurator({'Work': [
            CuratorDecision(action='create', degraded=True, justification='llm-failed: boom'),
        ]})

        [metadata] = await _created_metadata(make_interceptor(curator), taskmaster, 'Work')

        assert metadata == {
            **_FILED_METADATA,
            DEDUPE_SKIPPED_KEY: {'reason': 'llm-failed: boom', 'at': metadata[DEDUPE_SKIPPED_KEY]['at']},
        }

    @pytest.mark.asyncio
    async def test_a_curator_that_failed_to_construct_is_marked(self, make_interceptor, taskmaster):
        config = FusedMemoryConfig()
        config.taskmaster = None
        interceptor = make_interceptor(None, config=config)

        with patch(
            'fused_memory.middleware.task_interceptor.TaskCurator',
            side_effect=RuntimeError('qdrant down'),
        ):
            [metadata] = await _created_metadata(interceptor, taskmaster, 'Work')

        assert _marker_of(metadata) == {
            'reason': 'curator-unavailable: construction-failed: RuntimeError: qdrant down',
            'at': metadata[DEDUPE_SKIPPED_KEY]['at'],
        }
