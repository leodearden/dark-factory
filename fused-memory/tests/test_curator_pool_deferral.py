"""The curator defers, rather than fabricating a create, when its pool cannot serve.

Task 4503. An unavailable account pool — every account capped past the
cap-retry patience, or no account admissible at call start — yields
``CuratorDecision.deferred(reason)`` for each LLM-bound candidate: never an
exception, never a cached verdict, never a report, and never a step of the
degraded streak. The interceptor parks what is returned.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _curator_helpers import (
    agent_result,
    empty_corpus,
    escalator_double,
    make_config,
    pool_with_ids,
    prepared,
)
from shared.cli_invoke import AgentResult, AllAccountsCappedException
from shared.config_models import AccountConfig, UsageCapConfig
from shared.invocation_outcome import AuthFailed, CapHit, InvocationOutcome
from shared.usage_gate import PoolFrozen, UsageGate

from fused_memory.middleware.curation_deferral import DeferReason
from fused_memory.middleware.task_curator import (
    CuratorDecision,
    PreparedCandidate,
    TaskCurator,
)

INVOKE = 'fused_memory.middleware.task_curator.invoke_with_cap_retry'
DEADLINE_SECS = 30.0


def _capped(label: str = 'task-curator[p]') -> AllAccountsCappedException:
    return AllAccountsCappedException(retries=3, elapsed_secs=120.0, label=label)


def _frozen() -> PoolFrozen:
    return PoolFrozen(
        account_count=1, capped_account_names=('acct-a',), auth_failed_account_names=(),
    )


async def _curate(curator: TaskCurator, bundle: PreparedCandidate) -> CuratorDecision:
    return await asyncio.wait_for(
        curator.curate(bundle.candidate, 'p', '/x', prepared=bundle),
        DEADLINE_SECS,
    )


async def _curate_batch(
    curator: TaskCurator, bundles: list[PreparedCandidate],
) -> list[CuratorDecision]:
    return await asyncio.wait_for(
        curator.curate_batch_prepared(bundles, 'p', '/x'), DEADLINE_SECS,
    )


async def _frozen_gate(outcome: InvocationOutcome) -> UsageGate:
    """A real one-account gate whose only account *outcome* has just closed."""
    env = {'TEST_TOKEN_ACCT_A': 'fake-token-acct-a'}
    config = UsageCapConfig(
        accounts=[AccountConfig(name='acct-a', oauth_token_env='TEST_TOKEN_ACCT_A')],
        wait_for_reset=False,
    )
    with patch.dict(os.environ, env):
        gate = UsageGate(config)
    gate._run_probe = AsyncMock(return_value=True)
    async with gate.invoke_slot() as slot:
        slot.report(outcome)
    return gate


def _cap_until_later() -> CapHit:
    return CapHit(resets_at=datetime.now(UTC) + timedelta(hours=1), reason='test cap')


FROZEN_OUTCOMES = [
    pytest.param(_cap_until_later, DeferReason.ALL_ACCOUNTS_CAPPED, id='capped'),
    pytest.param(lambda: AuthFailed(status=401), DeferReason.ACCOUNTS_AUTH_FAILED, id='auth-failed'),
]


def _gated_curator(gate: UsageGate, tmp_path: Path) -> TaskCurator:
    return TaskCurator(
        config=make_config(), taskmaster=None, usage_gate=gate, config_dir_base=tmp_path,
    )


class TestDeferDecision:
    def test_deferred_sets_action_reason_and_justification(self):
        decision = CuratorDecision.deferred(
            DeferReason.ALL_ACCOUNTS_CAPPED, pool_sizes={'module': 2}, latency_ms=7,
        )

        assert decision.action == 'defer'
        assert decision.defer_reason is DeferReason.ALL_ACCOUNTS_CAPPED
        assert decision.justification == 'curation-deferred: all-accounts-capped'
        assert decision.pool_sizes == {'module': 2}
        assert decision.latency_ms == 7
        assert decision.degraded is False
        assert decision.to_log_fields()['defer_reason'] == 'all-accounts-capped'

    def test_defer_reason_iff_action_defer(self):
        with pytest.raises(ValueError, match='defer_reason'):
            CuratorDecision(action='defer')
        with pytest.raises(ValueError, match='defer_reason'):
            CuratorDecision(action='create', defer_reason=DeferReason.ALL_ACCOUNTS_CAPPED)

    @pytest.mark.asyncio
    async def test_the_model_cannot_emit_defer(self):
        curator = TaskCurator(config=make_config(), taskmaster=None)
        response = agent_result({'action': 'defer', 'justification': 'pool looks busy'})

        with patch(INVOKE, new=AsyncMock(return_value=response)):
            decision = await _curate(curator, prepared('T'))

        assert decision.action == 'create'
        assert decision.degraded is True
        assert decision.defer_reason is None
        assert decision.justification == "invalid-action: 'defer'"


class TestCurateDefersOnUnavailablePool:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('error', [_capped, _frozen], ids=['capped', 'frozen'])
    async def test_all_accounts_capped_defers_and_reports_nothing(self, error):
        escalator = escalator_double()
        curator = TaskCurator(config=make_config(), taskmaster=None, escalator=escalator)

        with patch(INVOKE, new=AsyncMock(side_effect=error())):
            decision = await _curate(curator, prepared('T'))

        assert decision.action == 'defer'
        assert decision.defer_reason is DeferReason.ALL_ACCOUNTS_CAPPED
        escalator.report_failure.assert_not_awaited()
        escalator.report_consecutive_degraded.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_defer_is_never_cached(self):
        curator = TaskCurator(config=make_config(), taskmaster=None)
        bundle = prepared('T', pool_with_ids(('42', 'pending')))
        combine = agent_result({
            'action': 'combine',
            'target_id': '42',
            'justification': 'same work',
            'rewritten_task': {'title': 'T merged', 'details': 'both halves'},
        })
        invoke = AsyncMock(side_effect=[_capped(), combine])

        with patch(INVOKE, new=invoke):
            first = await _curate(curator, bundle)
            second = await _curate(curator, bundle)

        assert first.action == 'defer'
        assert second.action == 'combine'
        assert second.target_id == '42'
        assert invoke.await_count == 2

    @pytest.mark.asyncio
    async def test_a_defer_neither_advances_nor_resets_the_degraded_streak(self):
        escalator = escalator_double()
        curator = TaskCurator(config=make_config(), taskmaster=None, escalator=escalator)
        unusable = agent_result({'action': 'bogus'})
        invoke = AsyncMock(side_effect=[unusable] * 4 + [_capped(), unusable])

        with patch(INVOKE, new=invoke):
            for i in range(4):
                assert (await _curate(curator, prepared(f'T{i}'))).degraded
            assert (await _curate(curator, prepared('T4'))).action == 'defer'
            escalator.report_consecutive_degraded.assert_not_awaited()
            await _curate(curator, prepared('T5'))

        escalator.report_consecutive_degraded.assert_awaited_once()
        assert escalator.report_consecutive_degraded.await_args.kwargs['streak'] == 5

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('outcome', 'reason'), FROZEN_OUTCOMES)
    async def test_frozen_real_gate_defers_without_parking(self, tmp_path, outcome, reason):
        gate = await _frozen_gate(outcome())
        curator = _gated_curator(gate, tmp_path)

        decision = await _curate(curator, prepared('T'))

        assert decision.action == 'defer'
        assert decision.defer_reason is reason


class TestBatchDefersOnUnavailablePool:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(('outcome', 'reason'), FROZEN_OUTCOMES)
    async def test_frozen_real_gate_defers_every_llm_bound_candidate(
        self, tmp_path, outcome, reason,
    ):
        gate = await _frozen_gate(outcome())
        curator = _gated_curator(gate, tmp_path)

        decisions = await _curate_batch(curator, [prepared('A'), prepared('B')])

        assert [d.action for d in decisions] == ['defer', 'defer']
        assert {d.defer_reason for d in decisions} == {reason}

    @pytest.mark.asyncio
    async def test_batch_cap_defers_llm_bound_and_keeps_deterministic_verdicts(self):
        curator = TaskCurator(config=make_config(), taskmaster=None)
        batch = [prepared('A'), prepared('B'), prepared('A')]

        invoke = AsyncMock(side_effect=_capped('task-curator-batch[p]'))
        with patch(INVOKE, new=invoke):
            decisions = await _curate_batch(curator, batch)

        invoke.assert_awaited_once()
        assert [d.action for d in decisions] == ['defer', 'defer', 'drop']
        assert decisions[2].batch_target_index == 0
        assert decisions[2].justification == 'pre-batch-dedup: identical payload_hash'

    @pytest.mark.asyncio
    async def test_bisect_leaf_cap_defers_instead_of_raising(self):
        curator = TaskCurator(config=make_config(), taskmaster=None)
        batch_failed = AgentResult(
            success=False, output='boom', subtype='error_max_turns', cost_usd=0.0,
        )
        invoke = AsyncMock(side_effect=[batch_failed, _capped(), _capped()])

        with patch(INVOKE, new=invoke), \
             patch.object(curator, '_build_corpus', side_effect=empty_corpus):
            decisions = await _curate_batch(curator, [prepared('A'), prepared('B')])

        assert [d.action for d in decisions] == ['defer', 'defer']
        assert invoke.await_count == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize('error', [_capped, _frozen], ids=['capped', 'frozen'])
    async def test_curate_batch_prepared_never_raises_pool_unavailable(self, error):
        escalator = escalator_double()
        curator = TaskCurator(config=make_config(), taskmaster=None, escalator=escalator)

        with patch(INVOKE, new=AsyncMock(side_effect=error())):
            decisions = await _curate_batch(curator, [prepared('A'), prepared('B')])

        assert [d.action for d in decisions] == ['defer', 'defer']
        escalator.report_failure.assert_not_awaited()
