"""One curation pipeline for every batch size: a single curation is a batch of one.

Contract: plans/review-all-s11-curator-pipeline-prd.md decisions 1 and 10,
§Contract "Curation pipeline", rows B6-B8. ``TaskCurator.curate_prepared`` is
the only pipeline. It applies one ordered guard chain to every bundle, sends
what is left to one LLM dispatch, and maps every LLM failure, a pool error
included, to a decision in one place.

Driven through public seams only: the invoker and escalator injected through
the constructor, and ``curate_prepared`` itself. No private name is patched.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
import yaml
from _curator_helpers import (
    agent_result,
    batch_create_result,
    candidate_titles,
    candidates_in,
    escalator_double,
    invoke_by_role,
    make_config,
    pool_with_ids,
    prepared,
    submit_as_one_batch,
    tripwire,
    zot_agent_result,
)
from shared.cli_invoke import AgentResult, AllAccountsCappedException
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.backends.sqlite_task_backend import SqliteTaskBackend
from fused_memory.config.schema import FusedMemoryConfig, TaskmasterConfig
from fused_memory.middleware.curation_deferral import DeferReason
from fused_memory.middleware.curator_types import (
    CandidateTask,
    CuratorFailureError,
    PreparedCandidate,
)
from fused_memory.middleware.task_curator import TaskCurator
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.ticket_store import TicketStore
from fused_memory.reconciliation.event_buffer import EventBuffer

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

PROJECT_ID = 'p'
PROJECT_ROOT = '/x'

_BLOCKLISTED = CandidateTask(
    title='Convert FIX C relay-flag deletion: search-then-delete',
    description='Metric fixc_flags_deleted_not_found is not tracked.',
)
_OPERATIONAL = CandidateTask(
    title='Run prune_recon_cycle_summaries --apply against live Mem0',
    description='Operational --apply run to collapse pre-existing piles.',
)
_RECENTLY_CREATED = CandidateTask(
    title='Teach the janitor to sweep stale tickets',
    description='The first filing.',
    files_to_modify=['fused-memory/src/fused_memory/middleware/ticket_janitor.py'],
)


def _max_turns_failure() -> AgentResult:
    """An LLM call that ran out of turns: a failure, but not a zero-output hang."""
    return AgentResult(success=False, output='', subtype='error_max_turns', turns=12)


def _create_each_in_batch(kwargs: dict[str, Any]) -> AgentResult:
    return batch_create_result(candidates_in(kwargs['prompt']))


def _llm_bound(title: str) -> PreparedCandidate:
    return prepared(title, candidate=CandidateTask(title=title, description=f'{title} body'))


def _roles(invoke: AsyncMock) -> list[str]:
    return [call.kwargs['role'] for call in invoke.await_args_list]


def _prompts(invoke: AsyncMock, role: str) -> list[str]:
    return [
        call.kwargs['prompt'] for call in invoke.await_args_list if call.kwargs['role'] == role
    ]


def _write_registry(path: Path, *, title_subs: list[str], desc_subs: list[str]) -> str:
    path.write_text(yaml.dump([{
        'name': 'pipeline_test_entry',
        'reason': 'pipeline test reason',
        'title_substrings': title_subs,
        'description_substrings': desc_subs,
    }]), encoding='utf-8')
    return str(path)


def _blocklist_config(tmp_path: Path) -> FusedMemoryConfig:
    config = make_config()
    config.curator.cancelled_premise_blocklist_path = _write_registry(
        tmp_path / 'blocklist.yaml',
        title_subs=['search-then-delete', 'fix c'],
        desc_subs=['fixc_flags_deleted_not_found'],
    )
    return config


def _operational_config(tmp_path: Path) -> FusedMemoryConfig:
    config = make_config()
    config.curator.operational_ask_registry_path = _write_registry(
        tmp_path / 'operational_ask_registry.yaml',
        title_subs=['prune_recon_cycle_summaries'],
        desc_subs=['--apply'],
    )
    return config


def _no_setup(_curator: TaskCurator) -> None:
    return None


def _note_recent_create(curator: TaskCurator) -> None:
    curator.note_created(PROJECT_ID, _RECENTLY_CREATED, '41')


# (config builder, the guard-hit candidate, per-curator setup)
_GUARD_HITS: dict[str, tuple[
    Callable[[Path], FusedMemoryConfig], CandidateTask, Callable[[TaskCurator], None],
]] = {
    'blocklist-refuse': (_blocklist_config, _BLOCKLISTED, _no_setup),
    'pre-llm-exact-match': (lambda _tmp: make_config(), _RECENTLY_CREATED, _note_recent_create),
    'operational-ask-route': (_operational_config, _OPERATIONAL, _no_setup),
}


@pytest.mark.asyncio
class TestOnePipeline:
    @pytest.mark.parametrize('guard', sorted(_GUARD_HITS))
    async def test_a_guard_hit_decides_the_same_alone_or_inside_a_batch(self, guard, tmp_path):
        """B7: the size-1 decision equals the in-batch one, field for field."""
        build_config, hit, setup = _GUARD_HITS[guard]

        def fresh() -> TaskCurator:
            curator = TaskCurator(
                config=build_config(tmp_path), taskmaster=None, cwd=tmp_path,
                escalator=escalator_double(),
                invoke=invoke_by_role(
                    single=tripwire('a guard hit must not reach the LLM'),
                    batch=_create_each_in_batch,
                ),
            )
            setup(curator)
            return curator

        bundle = prepared(hit.title, candidate=hit)
        [alone] = await fresh().curate_prepared([bundle], PROJECT_ID, PROJECT_ROOT)
        in_batch = await fresh().curate_prepared(
            [_llm_bound('Neighbour A'), bundle, _llm_bound('Neighbour B')],
            PROJECT_ID, PROJECT_ROOT,
        )

        assert alone.action != 'create'
        assert alone == in_batch[1]
        assert [d.action for d in (in_batch[0], in_batch[2])] == ['create', 'create']

    @pytest.mark.parametrize('guard', sorted(_GUARD_HITS))
    async def test_a_guard_hit_keeps_its_decision_beside_a_sibling_whose_failure_raises(
        self, guard, tmp_path,
    ):
        """B7: a raise while deciding the LLM-bound sibling defers only that sibling."""
        build_config, hit, setup = _GUARD_HITS[guard]

        def fresh() -> TaskCurator:
            escalator = escalator_double()
            escalator.report_failure = AsyncMock(side_effect=OSError('escalation state unwritable'))
            curator = TaskCurator(
                config=build_config(tmp_path), taskmaster=None, cwd=tmp_path, escalator=escalator,
                invoke=invoke_by_role(
                    single=_max_turns_failure(), batch=tripwire('one LLM-bound sibling is a batch of one'),
                ),
            )
            setup(curator)
            return curator

        bundle = prepared(hit.title, candidate=hit)
        [alone] = await fresh().curate_prepared([bundle], PROJECT_ID, PROJECT_ROOT)
        in_batch = await fresh().curate_prepared(
            [bundle, _llm_bound('Neighbour')], PROJECT_ID, PROJECT_ROOT,
        )

        assert alone.action != 'defer'
        assert in_batch[0] == alone
        assert (in_batch[1].action, in_batch[1].defer_reason) == ('defer', DeferReason.CURATOR_RAISED)

    async def test_the_exact_match_runs_inside_a_batch(self):
        invoke = invoke_by_role(
            single=tripwire('a batch of two must use the batch spec'),
            batch=_create_each_in_batch,
        )
        curator = TaskCurator(
            config=make_config(), taskmaster=None, escalator=escalator_double(), invoke=invoke,
        )
        curator.note_created(PROJECT_ID, _RECENTLY_CREATED, '41')
        reworded = CandidateTask(
            title=_RECENTLY_CREATED.title,
            description='Filed again, worded differently.',
            files_to_modify=list(_RECENTLY_CREATED.files_to_modify),
        )
        assert reworded.payload_hash() != _RECENTLY_CREATED.payload_hash()

        decisions = await curator.curate_prepared(
            [_llm_bound('Neighbour A'), prepared(reworded.title, candidate=reworded),
             _llm_bound('Neighbour B')],
            PROJECT_ID, PROJECT_ROOT,
        )

        assert (decisions[1].action, decisions[1].target_id, decisions[1].justification) == (
            'drop', '41', 'pre-llm-exact-match',
        )
        assert _roles(invoke) == ['task_curator_batch']
        [prompt] = _prompts(invoke, 'task_curator_batch')
        assert candidates_in(prompt) == 2
        assert reworded.title not in prompt

    async def test_a_failed_batch_bottoms_out_in_single_calls_over_the_prepared_pools(self):
        """B8: the leaves invoke over the bundles' own pools, never rebuilding a corpus.

        ``_build_corpus`` is left real: with no embedder configured, a leaf
        that re-entered the guard chain and rebuilt its corpus would defer.
        """
        def echo_pool(kwargs: dict[str, Any]) -> AgentResult:
            seen = [pid for pid in ('71', '72') if f'[Task {pid}]' in kwargs['prompt']]
            return agent_result({'action': 'create', 'justification': f'saw {seen}'})

        invoke = invoke_by_role(single=echo_pool, batch=_max_turns_failure())
        curator = TaskCurator(
            config=make_config(), taskmaster=None, escalator=escalator_double(), invoke=invoke,
        )

        decisions = await curator.curate_prepared(
            [prepared('Left', pool=pool_with_ids(('71', 'pending'))),
             prepared('Right', pool=pool_with_ids(('72', 'pending')))],
            PROJECT_ID, PROJECT_ROOT,
        )

        assert _roles(invoke) == ['task_curator_batch', 'task_curator', 'task_curator']
        singles = _prompts(invoke, 'task_curator')
        assert sorted('[Task 71]' in p for p in singles) == [False, True]
        assert sorted('[Task 72]' in p for p in singles) == [False, True]
        assert [(d.action, d.justification) for d in decisions] == [
            ('create', "saw ['71']"), ('create', "saw ['72']"),
        ]

    async def test_a_failure_with_no_orchestrator_is_decided_not_raised(self):
        escalator = escalator_double()
        escalator.report_failure = AsyncMock(
            side_effect=CuratorFailureError('no orchestrator is running for this project'),
        )
        curator = TaskCurator(
            config=make_config(), taskmaster=None, escalator=escalator,
            invoke=invoke_by_role(
                single=lambda kwargs: (
                    zot_agent_result() if 'Hung' in kwargs['prompt'] else _max_turns_failure()
                ),
                batch=tripwire('each call here is a batch of one'),
            ),
        )

        [failed] = await curator.curate_prepared([prepared('Failed')], PROJECT_ID, PROJECT_ROOT)
        [hung] = await curator.curate_prepared([prepared('Hung')], PROJECT_ID, PROJECT_ROOT)

        assert (failed.action, failed.defer_reason) == ('defer', DeferReason.LLM_FAILED)
        assert (hung.action, hung.degraded, hung.degraded_by_zot) == ('create', True, True)
        assert hung.justification.startswith('curator-failed: CuratorFailureError')
        assert escalator.report_failure.await_count == 2

    async def test_a_hang_with_no_escalator_wired_is_not_called_escalated(self):
        curator = TaskCurator(
            config=make_config(), taskmaster=None, escalator=None,
            invoke=invoke_by_role(
                single=zot_agent_result(), batch=tripwire('one bundle is a batch of one'),
            ),
        )

        [hung] = await curator.curate_prepared([prepared('Hung')], PROJECT_ID, PROJECT_ROOT)

        assert (hung.action, hung.degraded_by_zot) == ('create', True)
        assert hung.justification.startswith('curator-failed: CuratorFailureError')

    async def test_a_half_that_raises_leaves_no_sibling_call_in_flight(self):
        in_flight: set[str] = set()
        never = asyncio.Event()

        async def answer(**kwargs: Any) -> AgentResult:
            titles = candidate_titles(kwargs['prompt'])
            if titles != ['Slow']:
                return _max_turns_failure()
            in_flight.add('Slow')
            try:
                await never.wait()
            finally:
                in_flight.discard('Slow')
            return _max_turns_failure()

        escalator = escalator_double()
        escalator.report_failure = AsyncMock(side_effect=OSError('escalation state unwritable'))
        curator = TaskCurator(
            config=make_config(), taskmaster=None, escalator=escalator,
            invoke=AsyncMock(side_effect=answer),
        )

        try:
            decisions = await asyncio.wait_for(
                curator.curate_prepared(
                    [_llm_bound('Fails'), _llm_bound('Slow')], PROJECT_ID, PROJECT_ROOT,
                ),
                timeout=5.0,
            )
            assert in_flight == set()
        finally:
            never.set()

        assert [(d.action, d.defer_reason) for d in decisions] == [
            ('defer', DeferReason.CURATOR_RAISED),
        ] * 2

    async def test_an_escalator_that_raises_defers_every_candidate_curator_raised(self):
        escalator = escalator_double()
        escalator.report_failure = AsyncMock(side_effect=RuntimeError('escalation queue corrupt'))
        curator = TaskCurator(
            config=make_config(), taskmaster=None, escalator=escalator,
            invoke=invoke_by_role(single=_max_turns_failure(), batch=_max_turns_failure()),
        )

        decisions = await curator.curate_prepared(
            [prepared('First'), prepared('Second')], PROJECT_ID, PROJECT_ROOT,
        )

        assert [(d.action, d.defer_reason) for d in decisions] == [
            ('defer', DeferReason.CURATOR_RAISED), ('defer', DeferReason.CURATOR_RAISED),
        ]

    async def test_a_pool_error_defers_each_candidate_after_one_round_trip(self):
        invoke = AsyncMock(side_effect=AllAccountsCappedException(
            retries=3, elapsed_secs=120.0, label='task-curator-batch[p]',
        ))
        curator = TaskCurator(
            config=make_config(), taskmaster=None, escalator=escalator_double(), invoke=invoke,
        )

        decisions = await curator.curate_prepared(
            [prepared('First'), prepared('Second')], PROJECT_ID, PROJECT_ROOT,
        )

        assert [(d.action, d.defer_reason) for d in decisions] == [
            ('defer', DeferReason.ALL_ACCOUNTS_CAPPED), ('defer', DeferReason.ALL_ACCOUNTS_CAPPED),
        ]
        assert invoke.await_count == 1

    async def test_a_failed_corpus_is_deferred_before_the_breaker_is_consulted(self):
        """PRD decision 10 orders guard 9 (corpus failed) before guard 10 (breaker open)."""
        config = make_config()
        config.curator.zero_output_breaker_threshold = 1
        config.curator.zero_output_breaker_cooldown_seconds = 600.0
        curator = TaskCurator(
            config=config, taskmaster=None, escalator=escalator_double(),
            invoke=invoke_by_role(single=zot_agent_result(), batch=_max_turns_failure()),
        )
        [hung] = await curator.curate_prepared([prepared('Hung')], PROJECT_ID, PROJECT_ROOT)
        assert hung.degraded_by_zot

        [broken] = await curator.curate_prepared(
            [prepared('Broken corpus', corpus_error='qdrant unreachable')],
            PROJECT_ID, PROJECT_ROOT,
        )
        [healthy] = await curator.curate_prepared(
            [prepared('Healthy corpus')], PROJECT_ID, PROJECT_ROOT,
        )

        assert broken.defer_reason is DeferReason.CORPUS_FAILED
        assert healthy.defer_reason is DeferReason.ZERO_OUTPUT_BREAKER_OPEN



# --- The zero-output breaker gates every dispatch, bisected halves included --

# A call's answer, keyed on its candidates' titles: returned at once, or after
# the first zero-output hang has been reported (and so the breaker opened).
_Scripted = AgentResult | tuple[str, AgentResult]


def _breaker_scene(
    script: dict[tuple[str, ...], _Scripted],
) -> tuple[TaskCurator, AsyncMock, AsyncMock]:
    """A curator whose breaker opens on its first hang, and its invoker and escalator.

    Each call yields once before it answers, as a call in flight would, so
    halves dispatched together are all invoked before either answers.
    ``report_failure`` signals a hang synchronously, and the failure mapper
    records the hang on the breaker before it next yields, so a call scripted
    to wait resumes only once the breaker is open.
    """
    config = make_config()
    config.curator.zero_output_breaker_threshold = 1
    config.curator.zero_output_breaker_cooldown_seconds = 600.0
    zot_reported = asyncio.Event()

    def report_failure(**kwargs: Any) -> None:
        if kwargs.get('zero_output_timeout') is True:
            zot_reported.set()

    escalator = escalator_double()
    escalator.report_failure = AsyncMock(side_effect=report_failure)

    async def answer(**kwargs: Any) -> AgentResult:
        titles = tuple(candidate_titles(kwargs['prompt']))
        if titles not in script:
            pytest.fail(f'unscripted curator call over {titles!r}')
        scripted = script[titles]
        await asyncio.sleep(0)
        if isinstance(scripted, tuple):
            _wait, scripted = scripted
            await zot_reported.wait()
        return scripted

    invoke = AsyncMock(side_effect=answer)
    curator = TaskCurator(config=config, taskmaster=None, escalator=escalator, invoke=invoke)
    return curator, invoke, escalator


def _invoked_titles(invoke: AsyncMock) -> list[tuple[str, ...]]:
    return sorted(
        tuple(candidate_titles(call.kwargs['prompt'])) for call in invoke.await_args_list
    )


@pytest.mark.asyncio
class TestTheBreakerGatesEveryDispatch:
    async def test_leaves_bisected_after_the_breaker_opens_defer_without_a_call(self):
        curator, invoke, escalator = _breaker_scene({
            ('A', 'B', 'C'): _max_turns_failure(),
            ('A',): zot_agent_result(),
            ('B', 'C'): ('wait', _max_turns_failure()),
        })

        decisions = await asyncio.wait_for(
            curator.curate_prepared(
                [_llm_bound(title) for title in 'ABC'], PROJECT_ID, PROJECT_ROOT,
            ),
            timeout=5.0,
        )

        assert (decisions[0].action, decisions[0].degraded_by_zot) == ('create', True)
        assert [(d.action, d.defer_reason) for d in decisions[1:]] == [
            ('defer', DeferReason.ZERO_OUTPUT_BREAKER_OPEN),
            ('defer', DeferReason.ZERO_OUTPUT_BREAKER_OPEN),
        ]
        assert [
            candidate_titles(prompt) for prompt in _prompts(invoke, 'task_curator')
        ] == [['A']]
        assert escalator.report_failure.await_count == 1

    async def test_a_half_dispatched_after_the_breaker_opens_makes_no_call(self):
        curator, invoke, escalator = _breaker_scene({
            ('A', 'B', 'C', 'D', 'E', 'F'): _max_turns_failure(),
            ('A', 'B', 'C'): _max_turns_failure(),
            ('D', 'E', 'F'): ('wait', _max_turns_failure()),
            ('A',): zot_agent_result(),
            ('B', 'C'): ('wait', _max_turns_failure()),
        })

        decisions = await asyncio.wait_for(
            curator.curate_prepared(
                [_llm_bound(title) for title in 'ABCDEF'], PROJECT_ID, PROJECT_ROOT,
            ),
            timeout=5.0,
        )

        assert decisions[0].degraded_by_zot
        assert [(d.action, d.defer_reason) for d in decisions[1:]] == [
            ('defer', DeferReason.ZERO_OUTPUT_BREAKER_OPEN),
        ] * 5
        assert _invoked_titles(invoke) == sorted([
            ('A', 'B', 'C', 'D', 'E', 'F'), ('A', 'B', 'C'), ('D', 'E', 'F'), ('A',), ('B', 'C'),
        ])
        assert escalator.report_failure.await_count == 1


# --- B6: the consumer sees the exact match inside a worker batch -------------


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


@pytest_asyncio.fixture
async def backend(tmp_path):
    task_backend = SqliteTaskBackend(TaskmasterConfig(project_root=str(tmp_path)))
    await task_backend.start()
    yield task_backend
    await task_backend.close()


def _task(title: str, description: str, files: list[str]) -> dict[str, Any]:
    return {
        'title': title, 'description': description,
        'metadata': json.dumps({'files': files, 'spawn_context': 'manual'}),
    }


@pytest.mark.asyncio
class TestExactDuplicateInsideAWorkerBatch:
    async def test_a_reworded_duplicate_in_a_batch_drops_onto_the_task_it_repeats(
        self, store, event_buffer, backend, tmp_path,
    ):
        root = str(tmp_path)
        invoke = invoke_by_role(
            single=agent_result({'action': 'create', 'justification': 'new'}),
            batch=_create_each_in_batch,
        )
        curator = TaskCurator(
            config=make_config(), taskmaster=backend, escalator=escalator_double(),
            invoke=invoke,
        )

        async def bundle_for(candidate: CandidateTask, *_args: Any) -> PreparedCandidate:
            return prepared(candidate.title, candidate=candidate)

        curator.prepare_candidate = AsyncMock(side_effect=bundle_for)
        curator.record_task = AsyncMock()
        interceptor = TaskInterceptor(
            backend, None, event_buffer, ticket_store=store, curator=curator,
        )
        try:
            first = await interceptor.submit_task(
                root, **_task('Task A', 'The first filing.', ['a.py']),
            )
            task_a = (await interceptor.resolve_ticket(first['ticket'], root, 30.0))['task_id']

            b, duplicate, c = await submit_as_one_batch(interceptor, curator, [
                _task('Task B', 'Other work.', ['b.py']),
                _task('Task A', 'The same ask, filed again in other words.', ['a.py']),
                _task('Task C', 'More other work.', ['c.py']),
            ], project_root=root)
            answers = [
                await interceptor.resolve_ticket(ticket, root, 30.0)
                for ticket in (b, duplicate, c)
            ]
            created = [answers[0], answers[2]]
            assert [a['status'] for a in created] == ['created', 'created']
            created_titles = [
                (await backend.get_task(a['task_id'], root))['title'] for a in created
            ]
        finally:
            await interceptor.close()

        assert answers[1] == {
            'status': 'combined', 'task_id': task_a, 'reason': 'drop: pre-llm-exact-match',
        }
        assert created_titles == ['Task B', 'Task C']
        [batch_prompt] = _prompts(invoke, 'task_curator_batch')
        assert candidates_in(batch_prompt) == 2
        assert 'title: Task A' not in batch_prompt
