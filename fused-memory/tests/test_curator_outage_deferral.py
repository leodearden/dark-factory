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
    EMPTY_POOL_SIZES,
    agent_result,
    make_config,
    pool_with_ids,
    prepared,
    zot_agent_result,
)
from shared.cli_invoke import AgentResult

from fused_memory.middleware.curation_deferral import DeferReason
from fused_memory.middleware.task_curator import (
    CuratorDecision,
    PoolWithheld,
    PreparedCandidate,
    TaskCurator,
)

INVOKE = 'fused_memory.middleware.task_curator.invoke_with_cap_retry'
DEADLINE_SECS = 30.0
ZOT_ESCALATION_ID = 'esc-curator-7'


def _escalator() -> AsyncMock:
    escalator = AsyncMock()
    escalator.report_failure = AsyncMock(return_value=None)
    escalator.report_consecutive_degraded = AsyncMock(return_value=None)
    return escalator


def _curator(escalator: AsyncMock | None = None, *, breaker_cooldown: float = 600.0) -> TaskCurator:
    config = make_config()
    config.curator.zero_output_breaker_threshold = 1
    config.curator.zero_output_breaker_cooldown_seconds = breaker_cooldown
    return TaskCurator(config=config, taskmaster=None, escalator=escalator)


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
    with patch(INVOKE, new=AsyncMock(return_value=zot_agent_result())):
        opener = await _curate(curator, prepared('Breaker opener'), project_id)
    assert opener.degraded_by_zot


class TestCurateDefersOnCuratorOutage:
    @pytest.mark.asyncio
    async def test_a_failed_corpus_defers_and_is_not_cached(self):
        escalator = _escalator()
        curator = _curator(escalator)
        corpus = AsyncMock(side_effect=[
            RuntimeError('qdrant down'),
            (pool_with_ids(('42', 'pending')), {**EMPTY_POOL_SIZES, 'module': 1}, PoolWithheld()),
        ])
        invoke = AsyncMock(return_value=_combine_into_42())

        with patch(INVOKE, new=invoke), patch.object(curator, '_build_corpus', new=corpus):
            first = await _curate_unprepared(curator, prepared('T'))
            second = await _curate_unprepared(curator, prepared('T'))

        assert first.action == 'defer'
        assert first.defer_reason is DeferReason.CORPUS_FAILED
        escalator.report_failure.assert_not_awaited()
        assert (second.action, second.target_id) == ('combine', '42')
        invoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_non_zot_llm_failure_defers_and_keeps_its_diagnostics(self):
        escalator = _escalator()
        curator = _curator(escalator)
        invoke = AsyncMock(return_value=_max_turns())

        with patch(INVOKE, new=invoke):
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
        escalator = _escalator()
        curator = _curator(escalator)

        with patch(INVOKE, new=AsyncMock(side_effect=error)):
            decision = await _curate(curator, prepared('T'))

        assert decision.action == 'defer'
        assert decision.defer_reason is DeferReason.LLM_FAILED
        escalator.report_failure.assert_awaited_once()
        assert 'unexpected-exception' in escalator.report_failure.await_args.kwargs['subtype']

    @pytest.mark.asyncio
    async def test_an_open_breaker_defers_without_calling_the_llm(self):
        curator = _curator(_escalator())
        await _open_breaker(curator)
        invoke = AsyncMock(return_value=zot_agent_result())

        with patch(INVOKE, new=invoke):
            decision = await _curate(curator, prepared('Second distinct candidate'))

        assert decision.action == 'defer'
        assert decision.defer_reason is DeferReason.ZERO_OUTPUT_BREAKER_OPEN
        assert decision.degraded_by_zot is False
        invoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_zot_on_the_candidates_own_call_still_creates(self):
        escalator = _escalator()
        escalator.report_failure = AsyncMock(return_value=ZOT_ESCALATION_ID)
        curator = _curator(escalator)

        with patch(INVOKE, new=AsyncMock(return_value=zot_agent_result())):
            decision = await _curate(curator, prepared('ZOT candidate'))

        assert decision.action == 'create'
        assert decision.degraded_by_zot is True
        assert decision.zot_escalation_id == ZOT_ESCALATION_ID

    @pytest.mark.asyncio
    async def test_a_defer_neither_advances_nor_resets_the_degraded_streak(self):
        escalator = _escalator()
        cooldown = 0.2
        curator = _curator(escalator, breaker_cooldown=cooldown)
        unusable = agent_result({'action': 'bogus'})
        invoke = AsyncMock(side_effect=[
            *[unusable] * 4,
            _max_turns(),
            RuntimeError('llm down'),
            zot_agent_result(),
            unusable,
        ])
        corpus = AsyncMock(side_effect=RuntimeError('qdrant down'))

        with patch(INVOKE, new=invoke), patch.object(curator, '_build_corpus', new=corpus):
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


async def _empty_corpus(*_args, **_kwargs):
    return [], dict(EMPTY_POOL_SIZES), PoolWithheld()


class TestBatchDefersOnCuratorOutage:
    @pytest.mark.asyncio
    async def test_a_failed_corpus_bundle_defers_and_stays_out_of_the_prompt(self):
        curator = _curator(_escalator())
        batch = [
            prepared('Healthy alpha'),
            prepared('Broken beta', corpus_error='qdrant down'),
            prepared('Healthy gamma'),
        ]
        invoke = AsyncMock(return_value=agent_result({'decisions': [
            {'candidate_index': 0, 'action': 'create', 'justification': 'new alpha'},
            {'candidate_index': 1, 'action': 'create', 'justification': 'new gamma'},
        ]}))

        with patch(INVOKE, new=invoke):
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
        curator = _curator(_escalator())
        await _open_breaker(curator)
        invoke = AsyncMock(return_value=zot_agent_result())

        with patch(INVOKE, new=invoke):
            decisions = await _curate_batch(
                curator, [prepared('Alpha'), prepared('Beta'), prepared('Alpha')],
            )

        assert [d.action for d in decisions] == ['defer', 'defer', 'drop']
        assert {decisions[0].defer_reason, decisions[1].defer_reason} == {
            DeferReason.ZERO_OUTPUT_BREAKER_OPEN,
        }
        assert decisions[2].batch_target_index == 0
        invoke.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_breaker_deferrals_neither_advance_nor_reset_the_degraded_streak(self):
        escalator = _escalator()
        cooldown = 0.2
        curator = _curator(escalator, breaker_cooldown=cooldown)
        unusable = agent_result({'action': 'bogus'})

        with patch(INVOKE, new=AsyncMock(return_value=unusable)):
            for i in range(4):
                assert (await _curate(curator, prepared(f'T{i}'))).degraded
        await _open_breaker(curator, project_id='q')
        decisions = await _curate_batch(curator, [prepared('Alpha'), prepared('Beta')])
        assert [d.action for d in decisions] == ['defer', 'defer']
        escalator.report_consecutive_degraded.assert_not_awaited()

        await asyncio.sleep(cooldown * 1.5)
        with patch(INVOKE, new=AsyncMock(return_value=unusable)):
            await _curate(curator, prepared('T5'))

        escalator.report_consecutive_degraded.assert_awaited_once()
        assert escalator.report_consecutive_degraded.await_args.kwargs['streak'] == 5

    @pytest.mark.asyncio
    async def test_failed_bisect_leaves_defer(self):
        curator = _curator(_escalator())
        batch_failed = AgentResult(
            success=False, output='boom', subtype='error_max_turns', cost_usd=0.0,
        )
        invoke = AsyncMock(side_effect=[batch_failed, _max_turns(), _max_turns()])

        with patch(INVOKE, new=invoke), \
             patch.object(curator, '_build_corpus', side_effect=_empty_corpus):
            decisions = await _curate_batch(curator, [prepared('Alpha'), prepared('Beta')])

        assert [(d.action, d.defer_reason) for d in decisions] == [
            ('defer', DeferReason.LLM_FAILED), ('defer', DeferReason.LLM_FAILED),
        ]
        assert invoke.await_count == 3
