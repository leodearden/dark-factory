"""What one Claude CLI process cost, when a resumed process reports its session's running total.

Drives only the public API of ``shared.process_cost``. The recorded sessions
come from ``fixtures/cumulative_cost/sessions.json``; its README says which
fields are verbatim and why the totals are cumulative.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from shared.process_cost import (
    ProcessCost,
    ProcessCostBasis,
    derive_process_cost,
    next_resume_prior,
)

_SESSIONS = json.loads(
    (Path(__file__).parent / 'fixtures' / 'cumulative_cost' / 'sessions.json').read_text()
)
STEWARD_4895 = '59607ecd-9ecd-4f90-9ce6-bc71a7fb33e9'
CROSSED_UPGRADE_POSITIVE = 'f615388b-10b7-4dc9-b86b-4d011dcb0069'
CROSSED_UPGRADE_NEGATIVE = '4bd5a89f-1992-4691-a968-191f2ccc9abf'
_LOGGER = 'shared.process_cost'


def _stdout(session_id: str, index: int) -> dict[str, Any]:
    return _SESSIONS[session_id][index]['stdout']


def _reported(session_id: str, index: int) -> float:
    return _stdout(session_id, index)['total_cost_usd']


def _usage(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read: int = 0,
    cache_write_1h: int = 0,
) -> dict[str, Any]:
    return {
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'cache_read_input_tokens': cache_read,
        'cache_creation_input_tokens': cache_write_1h,
        'cache_creation': {
            'ephemeral_1h_input_tokens': cache_write_1h,
            'ephemeral_5m_input_tokens': 0,
        },
    }


def _result(
    *,
    total: Any,
    usage: dict[str, Any] | None,
    model_usage: dict[str, Any] | None = None,
    is_error: bool = False,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        'type': 'result',
        'subtype': 'error_during_execution' if is_error else 'success',
        'is_error': is_error,
        'session_id': 'synthetic-session',
        'total_cost_usd': total,
    }
    if usage is not None:
        data['usage'] = usage
    if model_usage is not None:
        data['modelUsage'] = model_usage
    return data


def _model_entry(canonical: str, cost: float) -> dict[str, Any]:
    return {'canonicalModel': canonical, 'costUSD': cost}


def _single_model(model: str) -> dict[str, Any]:
    return {model: _model_entry(model, 1.0)}


def _estimate(data: dict[str, Any], served_model_id: str | None = None) -> float:
    """The usage estimate, read through a resumed call with no prior."""
    return derive_process_cost(
        data, resumed=True, prior_total_usd=None, served_model_id=served_model_id,
    ).cost_usd


# One million output tokens, priced per model: the cheapest way to tell rates apart.
_ONE_MILLION_OUT = _usage(output_tokens=1_000_000)


class TestFreshProcessIsReported:
    def test_fresh_cost_is_the_reported_total(self) -> None:
        cost = derive_process_cost(
            _stdout(STEWARD_4895, 0), resumed=False, prior_total_usd=None,
        )

        assert cost.cost_usd == pytest.approx(2.288609, abs=1e-9)
        assert cost.basis is ProcessCostBasis.REPORTED
        assert cost.reported_total_usd == pytest.approx(2.288609, abs=1e-9)
        assert cost.delta_clamped is False

    def test_fresh_call_ignores_a_supplied_prior(self) -> None:
        cost = derive_process_cost(
            _stdout(STEWARD_4895, 0), resumed=False, prior_total_usd=100.0,
        )

        assert cost.cost_usd == pytest.approx(2.288609, abs=1e-9)
        assert cost.basis is ProcessCostBasis.REPORTED
        assert cost.delta_clamped is False


class TestResumeWithPriorIsSessionDelta:
    def test_each_resume_costs_its_delta_and_the_session_sums_to_its_total(self) -> None:
        fresh = derive_process_cost(
            _stdout(STEWARD_4895, 0), resumed=False, prior_total_usd=None,
        )
        resumes = [
            derive_process_cost(
                _stdout(STEWARD_4895, i),
                resumed=True,
                prior_total_usd=_reported(STEWARD_4895, i - 1),
            )
            for i in range(1, 5)
        ]

        assert [c.cost_usd for c in resumes] == pytest.approx(
            [1.1174095, 1.8764015, 2.0331165, 1.3074325], abs=1e-9,
        )
        assert [c.basis for c in resumes] == [ProcessCostBasis.SESSION_DELTA] * 4
        assert [c.delta_clamped for c in resumes] == [False] * 4
        assert fresh.cost_usd + sum(c.cost_usd for c in resumes) == pytest.approx(
            8.622969, abs=1e-9,
        )


class TestRejectedDeltaIsClamped:
    def test_negative_delta_falls_back_to_the_usage_estimate(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            cost = derive_process_cost(
                _stdout(CROSSED_UPGRADE_NEGATIVE, 1),
                resumed=True,
                prior_total_usd=1.7857264999999998,
            )

        assert cost.cost_usd == pytest.approx(1.5471005, abs=1e-9)
        assert cost.cost_usd >= 0
        assert cost.basis is ProcessCostBasis.USAGE_ESTIMATE
        assert cost.delta_clamped is True
        assert CROSSED_UPGRADE_NEGATIVE in caplog.text

    def test_over_subtracted_positive_delta_falls_back_to_the_usage_estimate(self) -> None:
        cost = derive_process_cost(
            _stdout(CROSSED_UPGRADE_POSITIVE, 2), resumed=True, prior_total_usd=1.2188795,
        )

        assert cost.cost_usd == pytest.approx(3.6470905, abs=1e-9)
        assert cost.cost_usd != pytest.approx(2.428211, abs=1e-6)
        assert cost.basis is ProcessCostBasis.USAGE_ESTIMATE
        assert cost.delta_clamped is True

    def test_pre_upgrade_resume_below_its_prior_is_clamped(self) -> None:
        cost = derive_process_cost(
            _stdout(CROSSED_UPGRADE_POSITIVE, 1), resumed=True, prior_total_usd=2.423063,
        )

        assert cost.cost_usd == pytest.approx(1.2188795, abs=1e-9)
        assert cost.basis is ProcessCostBasis.USAGE_ESTIMATE
        assert cost.delta_clamped is True


class TestResumeWithoutPrior:
    def test_resume_without_prior_is_the_usage_estimate(self) -> None:
        cost = derive_process_cost(
            _stdout(STEWARD_4895, 2), resumed=True, prior_total_usd=None,
        )

        assert cost.cost_usd == pytest.approx(1.8764015, abs=1e-9)
        assert cost.basis is ProcessCostBasis.USAGE_ESTIMATE
        assert cost.delta_clamped is False

    def test_estimate_never_exceeds_the_reported_total(self) -> None:
        data = _result(
            total=3.0,
            usage=_usage(output_tokens=400_000),
            model_usage=_single_model('claude-opus-5'),
        )

        cost = derive_process_cost(data, resumed=True, prior_total_usd=None)

        assert cost.cost_usd == pytest.approx(3.0, abs=1e-9)
        assert cost.basis is ProcessCostBasis.USAGE_ESTIMATE


class TestFailedRunWhoseUsageUnderStatesSpend:
    def test_delta_above_the_usage_estimate_is_accepted(self) -> None:
        data = _result(
            total=7.0,
            usage=_usage(output_tokens=40_000),
            model_usage=_single_model('claude-opus-5'),
            is_error=True,
        )

        cost = derive_process_cost(data, resumed=True, prior_total_usd=5.0)

        assert cost.cost_usd == pytest.approx(2.0, abs=1e-9)
        assert cost.basis is ProcessCostBasis.SESSION_DELTA
        assert cost.delta_clamped is False


class TestUnpricedModel:
    @pytest.mark.parametrize('model_usage', [
        None,
        {'claude-opus-9': _model_entry('claude-opus-9', 4.0)},
    ], ids=['no-model-usage', 'unknown-canonical-model'])
    def test_resume_without_prior_reports_the_total_and_warns(
        self, caplog, model_usage,
    ) -> None:
        data = _result(total=4.0, usage=_usage(output_tokens=1000), model_usage=model_usage)

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            cost = derive_process_cost(data, resumed=True, prior_total_usd=None)

        assert cost.cost_usd == pytest.approx(4.0, abs=1e-9)
        assert cost.basis is ProcessCostBasis.REPORTED
        assert cost.delta_clamped is False
        assert 'synthetic-session' in caplog.text

    def test_non_negative_delta_is_accepted(self) -> None:
        data = _result(total=4.0, usage=_usage(output_tokens=1000))

        cost = derive_process_cost(data, resumed=True, prior_total_usd=3.0)

        assert cost.cost_usd == pytest.approx(1.0, abs=1e-9)
        assert cost.basis is ProcessCostBasis.SESSION_DELTA
        assert cost.delta_clamped is False

    def test_negative_delta_reports_the_total_and_is_clamped(self) -> None:
        data = _result(total=4.0, usage=_usage(output_tokens=1000))

        cost = derive_process_cost(data, resumed=True, prior_total_usd=6.0)

        assert cost.cost_usd == pytest.approx(4.0, abs=1e-9)
        assert cost.basis is ProcessCostBasis.REPORTED
        assert cost.delta_clamped is True


class TestPricing:
    """Each sample is a measured full CLI stdout; together they pin the rate table."""

    @pytest.mark.parametrize(('model_key', 'canonical', 'usage', 'expected'), [
        (
            'claude-opus-5-5', 'claude-opus-5-5',
            _usage(input_tokens=200, output_tokens=108_774,
                   cache_read=22_678_057, cache_write_1h=353_628),
            9.5409154,
        ),
        (
            'claude-sonnet-5', 'claude-sonnet-5',
            _usage(input_tokens=160, output_tokens=69_997,
                   cache_read=10_949_113, cache_write_1h=179_374),
            3.6076086,
        ),
        (
            'claude-fable-5-1', 'claude-fable-5-1',
            _usage(input_tokens=226, output_tokens=6595,
                   cache_read=430_982, cache_write_1h=75_895),
            1.9576555,
        ),
        (
            'claude-haiku-4-5-20251001', 'claude-haiku-4-5',
            _usage(input_tokens=9192, output_tokens=20),
            0.009292,
        ),
    ], ids=['opus-5-5', 'sonnet-5', 'fable-5-1', 'haiku-4-5'])
    def test_measured_sample_prices_exactly(
        self, model_key, canonical, usage, expected,
    ) -> None:
        data = _result(
            total=1000.0, usage=usage,
            model_usage={model_key: _model_entry(canonical, 1.0)},
        )

        assert _estimate(data) == pytest.approx(expected, abs=1e-9)

    def test_cache_write_without_a_split_is_priced_at_the_5m_rate(self) -> None:
        usage = {'cache_creation_input_tokens': 1_000_000}
        data = _result(total=1000.0, usage=usage, model_usage=_single_model('claude-opus-5'))

        assert _estimate(data) == pytest.approx(6.25, abs=1e-9)

    def test_cache_write_remainder_beyond_the_split_is_priced_at_the_5m_rate(self) -> None:
        usage = {
            'cache_creation_input_tokens': 1_000_000,
            'cache_creation': {
                'ephemeral_1h_input_tokens': 400_000,
                'ephemeral_5m_input_tokens': 100_000,
            },
        }
        data = _result(total=1000.0, usage=usage, model_usage=_single_model('claude-opus-5'))

        assert _estimate(data) == pytest.approx(4.0 + 3.75, abs=1e-9)


class TestPricingModelChoice:
    def test_dominant_model_usage_entry_prices_the_usage_block(self) -> None:
        data = _result(
            total=1000.0,
            usage=_ONE_MILLION_OUT,
            model_usage={
                'claude-haiku-4-5-20251001': _model_entry('claude-haiku-4-5', 0.009),
                'claude-sonnet-5-20260901': _model_entry('claude-sonnet-5', 3.6),
            },
        )

        assert _estimate(data) == pytest.approx(10.0, abs=1e-9)

    def test_served_model_prices_the_usage_block(self) -> None:
        data = _result(
            total=1000.0, usage=_ONE_MILLION_OUT,
            model_usage={'claude-opus-5': _model_entry('claude-opus-5', 9.0)},
        )

        assert _estimate(data, served_model_id='claude-opus-5-5') == pytest.approx(
            20.0, abs=1e-9,
        )

    def test_unpriced_served_model_falls_back_to_the_dominant_entry(self) -> None:
        data = _result(
            total=1000.0, usage=_ONE_MILLION_OUT,
            model_usage={'claude-opus-5': _model_entry('claude-opus-5', 9.0)},
        )

        assert _estimate(data, served_model_id='claude-opus-9') == pytest.approx(
            25.0, abs=1e-9,
        )

    def test_no_served_model_uses_the_dominant_entry(self) -> None:
        data = _result(
            total=1000.0, usage=_ONE_MILLION_OUT,
            model_usage={'claude-opus-5': _model_entry('claude-opus-5', 9.0)},
        )

        assert _estimate(data, served_model_id=None) == pytest.approx(25.0, abs=1e-9)


class TestMalformedReportedTotal:
    @pytest.mark.parametrize('total', ['absent', '2.5', True], ids=['absent', 'string', 'bool'])
    def test_malformed_total_is_none_and_never_raises(self, total) -> None:
        data = _result(total=total, usage=_usage(output_tokens=1000))
        if total == 'absent':
            del data['total_cost_usd']

        cost = derive_process_cost(data, resumed=False, prior_total_usd=None)

        assert cost.reported_total_usd is None
        assert cost.cost_usd == 0.0


class TestNextResumePrior:
    @pytest.mark.parametrize(
        ('armed_session', 'armed_prior', 'result_session', 'result_total', 'expected'),
        [
            ('S', 5.0, 'S', 7.3, 7.3),
            ('S', 5.0, 'S', 0.0, 5.0),
            ('S', 5.0, 'S', None, 5.0),
            ('S', None, 'S', 4.0, 4.0),
            ('S', 5.0, 'T', 1.0, 1.0),
            (None, None, 'T', None, None),
        ],
        ids=[
            'later-total', 'zero-keeps-high-water', 'none-keeps-armed',
            'first-figure', 'new-session-never-inherits', 'nothing-known',
        ],
    )
    def test_next_prior(
        self, armed_session, armed_prior, result_session, result_total, expected,
    ) -> None:
        assert next_resume_prior(
            armed_session_id=armed_session,
            armed_prior_usd=armed_prior,
            result_session_id=result_session,
            result_reported_total_usd=result_total,
        ) == expected


class TestValueTypes:
    def test_process_cost_is_frozen(self) -> None:
        cost = ProcessCost(
            cost_usd=1.0, basis=ProcessCostBasis.REPORTED, reported_total_usd=1.0,
        )

        frozen_field = 'cost_usd'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(cost, frozen_field, 2.0)

    def test_basis_vocabulary_is_closed(self) -> None:
        assert {b.value for b in ProcessCostBasis} == {
            'reported', 'session_delta', 'usage_estimate',
        }
