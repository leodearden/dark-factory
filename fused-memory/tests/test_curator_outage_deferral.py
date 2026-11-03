"""The curator defers, rather than fabricating a create, when it cannot judge a submission.

A failed corpus assembly, a failed LLM call and an open zero-output breaker each
yield ``CuratorDecision.deferred(reason)``: never a cached verdict and never a
step of the degraded streak. A zero-output hang on the candidate's own call is
the one outage that still creates, marked ``degraded_by_zot``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from _curator_helpers import (
    agent_result,
    empty_corpus,
    escalator_double,
    make_config,
    pool_with_ids,
    prepared,
    zot_agent_result,
)
from shared.cli_invoke import AgentResult

from fused_memory.middleware.curation_deferral import DeferReason
from fused_memory.middleware.curator_types import (
    CuratorDecision,
    PoolWithheld,
    PreparedCandidate,
    empty_pool_sizes,
)
from fused_memory.middleware.task_curator import TaskCurator

DEADLINE_SECS = 30.0
ZOT_ESCALATION_ID = 'esc-curator-7'


def _curator(
    invoke: AsyncMock, escalator: AsyncMock | None = None, *, breaker_cooldown: float = 600.0,
) -> TaskCurator:
    config = make_config()
    config.curator.zero_output_breaker_threshold = 1
    config.curator.zero_output_breaker_cooldown_seconds = breaker_cooldown
    return TaskCurator(config=config, taskmaster=None, escalator=escalator, invoke=invoke)


def _max_turns() -> AgentResult:
    return AgentResult(
        success=False, output='partial', subtype='error_max_turns',
        timed_out=False, turns=3, cost_usd=0.05, duration_ms=9_000,
    )


def _combine_into_42() -> AgentResult:
    return agent_result({
        'action': 'combine',
        'target_id': '42',
        'justification': 'same work',
        'rewritten_task': {'title': 'T merged', 'details': 'both halves'},
    })


async def _curate(
    curator: TaskCurator, bundle: PreparedCandidate, project_id: str = 'p',
) -> CuratorDecision:
    return await asyncio.wait_for(
        curator.curate(bundle.candidate, project_id, '/x', prepared=bundle),
        DEADLINE_SECS,
    )


async def _curate_unprepared(curator: TaskCurator, bundle: PreparedCandidate) -> CuratorDecision:
    return await asyncio.wait_for(curator.curate(bundle.candidate, 'p', '/x'), DEADLINE_SECS)


async def _open_breaker(curator: TaskCurator, project_id: str = 'p') -> None:
    """Open *curator*'s breaker; its LLM must answer this call with a ZOT."""
    opener = await _curate(curator, prepared('Breaker opener'), project_id)
    assert opener.degraded_by_zot


class TestCurateDefersOnCuratorOutage:
    @pytest.mark.asyncio
    async def test_a_failed_corpus_defers_and_is_not_cached(self):
        escalator = escalator_double()
        corpus = AsyncMock(side_effect=[
            RuntimeError('qdrant down'),
            (pool_with_ids(('42', 'pending')), {**empty_pool_sizes(), 'module': 1}, PoolWithheld()),
        ])
        invoke = AsyncMock(return_value=_combine_into_42())
        curator = _curator(invoke, escalator)

        with patch.object(curator, '_build_corpus', new=corpus):
            first = await _curate_unprepared(curator, prepared('T'))
            second = await _curate_unprepared(curator, prepared('T'))

        assert first.action == 'defer'
        assert first.defer_reason is DeferReason.CORPUS_FAILED
        escalator.report_failure.assert_not_awaited()
        assert (second.action, second.target_id) == ('combine', '42')
        invoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_non_zot_llm_failure_defers_and_keeps_its_diagnostics(self):
        escalator = escalator_double()
        invoke = AsyncMock(return_value=_max_turns())
        curator = _curator(invoke, escalator)

        first = await _curate(curator, prepared('T'))
        await _curate(curator, prepared('T'))

        assert first.action == 'defer'
        assert first.defer_reason is DeferReason.LLM_FAILED
        assert escalator.report_failure.await_args_list[0].kwargs['zero_output_timeout'] is False
        assert invoke.await_count == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'error',
        [RuntimeError('llm down'), FileNotFoundError(2, 'No such file or directory', 'claude')],
        ids=['runtime-error', 'claude-binary-missing'],
    )
    async def test_an_unexpected_llm_exception_defers_and_reports(self, error):
        escalator = escalator_double()
        curator = _curator(AsyncMock(side_effect=error), escalator)

        decision = await _curate(curator, prepared('T'))

        assert decision.action == 'defer'
        assert decision.defer_reason is DeferReason.LLM_FAILED
        escalator.report_failure.assert_awaited_once()
        assert 'unexpected-exception' in escalator.report_failure.await_args.kwargs['subtype']

    @pytest.mark.asyncio
    async def test_an_open_breaker_defers_without_calling_the_llm(self):
        invoke = AsyncMock(return_value=zot_agent_result())
        curator = _curator(invoke, escalator_double())
        await _open_breaker(curator)

        decision = await _curate(curator, prepared('Second distinct candidate'))

        assert decision.action == 'defer'
        assert decision.defer_reason is DeferReason.ZERO_OUTPUT_BREAKER_OPEN
        assert decision.degraded_by_zot is False
        invoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_zot_on_the_candidates_own_call_still_creates(self):
        escalator = escalator_double()
        escalator.report_failure = AsyncMock(return_value=ZOT_ESCALATION_ID)
        curator = _curator(AsyncMock(return_value=zot_agent_result()), escalator)

        decision = await _curate(curator, prepared('ZOT candidate'))

        assert decision.action == 'create'
        assert decision.degraded_by_zot is True
        assert decision.zot_escalation_id == ZOT_ESCALATION_ID

    @pytest.mark.asyncio
    async def test_a_defer_neither_advances_nor_resets_the_degraded_streak(self):
        escalator = escalator_double()
        cooldown = 0.2
        unusable = agent_result({'action': 'bogus'})
        invoke = AsyncMock(side_effect=[
            *[unusable] * 4,
            _max_turns(),
            RuntimeError('llm down'),
            zot_agent_result(),
            unusable,
        ])
        curator = _curator(invoke, escalator, breaker_cooldown=cooldown)
        corpus = AsyncMock(side_effect=RuntimeError('qdrant down'))

        with patch.object(curator, '_build_corpus', new=corpus):
            for i in range(4):
                assert (await _curate(curator, prepared(f'T{i}'))).degraded
            deferrals = [
                await _curate(curator, prepared('Corpus', corpus_error='qdrant down')),
                await _curate(curator, prepared('Max turns')),
                await _curate(curator, prepared('Raises')),
            ]
            assert (await _curate(curator, prepared('ZOT elsewhere'), project_id='q')).degraded_by_zot
            deferrals.append(await _curate(curator, prepared('Breaker open')))
            assert [d.defer_reason for d in deferrals] == [
                DeferReason.CORPUS_FAILED,
                DeferReason.LLM_FAILED,
                DeferReason.LLM_FAILED,
                DeferReason.ZERO_OUTPUT_BREAKER_OPEN,
            ]
            escalator.report_consecutive_degraded.assert_not_awaited()
            await asyncio.sleep(cooldown * 1.5)
            await _curate(curator, prepared('T5'))

        escalator.report_consecutive_degraded.assert_awaited_once()
        assert escalator.report_consecutive_degraded.await_args.kwargs['streak'] == 5


async def _curate_batch(
    curator: TaskCurator, bundles: list[PreparedCandidate], project_id: str = 'p',
) -> list[CuratorDecision]:
    return await asyncio.wait_for(
        curator.curate_batch_prepared(bundles, project_id, '/x'), DEADLINE_SECS,
    )


class TestBatchDefersOnCuratorOutage:
    @pytest.mark.asyncio
    async def test_a_failed_corpus_bundle_defers_and_stays_out_of_the_prompt(self):
        batch = [
            prepared('Healthy alpha'),
            prepared('Broken beta', corpus_error='qdrant down'),
            prepared('Healthy gamma'),
        ]
        invoke = AsyncMock(return_value=agent_result({'decisions': [
            {'candidate_index': 0, 'action': 'create', 'justification': 'new alpha'},
            {'candidate_index': 1, 'action': 'create', 'justification': 'new gamma'},
        ]}))
        curator = _curator(invoke, escalator_double())

        decisions = await _curate_batch(curator, batch)

        assert [d.action for d in decisions] == ['create', 'defer', 'create']
        assert decisions[1].defer_reason is DeferReason.CORPUS_FAILED
        assert [decisions[0].justification, decisions[2].justification] == [
            'new alpha', 'new gamma',
        ]
        invoke.assert_awaited_once()
        batch_call = invoke.await_args
        assert batch_call is not None
        assert 'Broken beta' not in batch_call.kwargs['prompt']

    @pytest.mark.asyncio
    async def test_an_open_breaker_defers_every_llm_bound_candidate(self):
        invoke = AsyncMock(return_value=zot_agent_result())
        curator = _curator(invoke, escalator_double())
        await _open_breaker(curator)

        decisions = await _curate_batch(
            curator, [prepared('Alpha'), prepared('Beta'), prepared('Alpha')],
        )

        assert [d.action for d in decisions] == ['defer', 'defer', 'drop']
        assert {decisions[0].defer_reason, decisions[1].defer_reason} == {
            DeferReason.ZERO_OUTPUT_BREAKER_OPEN,
        }
        assert decisions[2].batch_target_index == 0
        invoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_breaker_deferrals_neither_advance_nor_reset_the_degraded_streak(self):
        escalator = escalator_double()
        cooldown = 0.2
        unusable = agent_result({'action': 'bogus'})
        curator = _curator(
            AsyncMock(side_effect=[*[unusable] * 4, zot_agent_result(), unusable]),
            escalator, breaker_cooldown=cooldown,
        )

        for i in range(4):
            assert (await _curate(curator, prepared(f'T{i}'))).degraded
        await _open_breaker(curator, project_id='q')
        decisions = await _curate_batch(curator, [prepared('Alpha'), prepared('Beta')])
        assert [d.action for d in decisions] == ['defer', 'defer']
        escalator.report_consecutive_degraded.assert_not_awaited()

        await asyncio.sleep(cooldown * 1.5)
        await _curate(curator, prepared('T5'))

        escalator.report_consecutive_degraded.assert_awaited_once()
        assert escalator.report_consecutive_degraded.await_args.kwargs['streak'] == 5

    @pytest.mark.asyncio
    async def test_failed_bisect_leaves_defer(self):
        batch_failed = AgentResult(
            success=False, output='boom', subtype='error_max_turns', cost_usd=0.0,
        )
        invoke = AsyncMock(side_effect=[batch_failed, _max_turns(), _max_turns()])
        curator = _curator(invoke, escalator_double())

        with patch.object(curator, '_build_corpus', side_effect=empty_corpus):
            decisions = await _curate_batch(curator, [prepared('Alpha'), prepared('Beta')])

        assert [(d.action, d.defer_reason) for d in decisions] == [
            ('defer', DeferReason.LLM_FAILED), ('defer', DeferReason.LLM_FAILED),
        ]
        assert invoke.await_count == 3
