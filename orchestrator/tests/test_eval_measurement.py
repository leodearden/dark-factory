"""Tests for evals/measurement.py — scored or unmeasured, and why, as typed values.

The classifier cases are the ones that pinned the retired
``metrics.detect_invocation_error``; they now assert the typed
:class:`Unmeasured` it is replaced by. Every refusal shape here is a REAL
``shared.cli_invoke.AgentResult`` so ``classify_invocation`` sees real fields.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import build_usage_gate
from shared.cli_invoke import AgentResult, AllAccountsCappedException
from shared.config_models import AccountConfig
from shared.invocation_outcome import (
    NON_CAP_CLI_ERROR_MARKERS,
    CliLocalError,
    classify_invocation,
)

from orchestrator.evals.measurement import (
    MeasurementStage,
    Scored,
    Unmeasured,
    UnmeasuredCause,
    classify_unmeasured,
    clip_reason,
    coerce_cost_usd,
    exception_reason,
    json_answer,
    measured_call,
)

# The exact payload the Claude CLI emitted during the 2026-07-27 architect-effort
# campaign (plans/eval-architect-effort-verdict-2026-07-27.md, "Data hygiene").
_CAP_TEXT = "You've hit your session limit · resets 8pm"


def _cap_agent_result(cost_usd: float = 0.0) -> AgentResult:
    return AgentResult(
        success=False, output=_CAP_TEXT, cost_usd=cost_usd, duration_ms=1200,
        turns=0, subtype='error', api_error_status=429,
    )


def _codex_cap_agent_result() -> AgentResult:
    """A CODEX cap body: matched only by the codex-backend table, never claude's."""
    return AgentResult(
        success=False, output='stream error: usage limit reached for this account',
        cost_usd=0.0, duration_ms=900, turns=0, subtype='error',
    )


def _wedged_agent_result() -> AgentResult:
    """A zero-output CLI wedge: full timeout, zero transcript turns, zero cost."""
    return AgentResult(
        success=False, output='', cost_usd=0.0, duration_ms=600_000, turns=0,
        timed_out=True, transcript_turns=0,
    )


def _flushed_server_error_result(status: int = 529) -> AgentResult:
    """A watchdog-SIGTERM-flushed 5xx: ``timed_out`` and a 5xx status both true.

    Mirrors shared/tests/test_server_error.py::_incident_result.
    """
    return AgentResult(
        success=False, output='', subtype='error_empty_output', timed_out=True,
        transcript_turns=0, duration_ms=127_000, api_error_status=status,
    )


def _cli_local_error_result(marker: str) -> AgentResult:
    """A local CLI/usage fault — explicitly NOT a cap (the reify-3604 fix)."""
    return AgentResult(
        success=False, output=f'error: {marker}', cost_usd=0.0, duration_ms=300,
        turns=0, subtype='error',
    )


def _server_error_result(status: int = 529) -> AgentResult:
    """A fast, zero-cost 5xx refusal carrying no cap body and no timeout."""
    return AgentResult(
        success=False, output='', cost_usd=0.0, duration_ms=1100, turns=0,
        subtype='error', api_error_status=status,
    )


def _classify(result: AgentResult, *, backend: str = 'claude') -> Unmeasured | None:
    return classify_unmeasured(result, stage=MeasurementStage.ARCHITECT, backend=backend)


class TestVocabulary:
    """Cause values are the report's by-cause bucket labels: they must not move."""

    def test_unmeasured_cause_values(self):
        assert {c.value for c in UnmeasuredCause} == {
            'cap_hit', 'auth_failed', 'model_not_found', 'api_error',
            'server_error', 'wedge', 'timeout', 'cap_exhausted',
            'harness_error', 'raised', 'unparseable', 'legacy_unclassified',
        }

    def test_measurement_stage_values(self):
        assert {s.value for s in MeasurementStage} == {
            'architect', 'judge', 'pairwise_judge', 'matcher', 'assessor',
            'synthesizer',
        }


class TestOutcomeInvariants:
    def test_record_is_plain_json_strings_without_cost(self):
        record = Unmeasured(
            MeasurementStage.ARCHITECT, UnmeasuredCause.CAP_HIT, 'r', cost_usd=0.5,
        ).to_record()

        assert record == {'stage': 'architect', 'cause': 'cap_hit', 'reason': 'r'}
        assert all(type(v) is str for v in record.values())

    def test_unmeasured_is_frozen(self):
        u: Any = Unmeasured(
            MeasurementStage.JUDGE, UnmeasuredCause.RAISED, 'r', cost_usd=0.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            u.reason = 'other'

    def test_scored_is_frozen(self):
        s: Any = Scored(1, cost_usd=0.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.value = 2

    def test_bare_string_stage_is_rejected_naming_field_and_value(self):
        bare_stage: Any = 'architect'
        with pytest.raises(TypeError, match=r"stage.*'architect'"):
            Unmeasured(bare_stage, UnmeasuredCause.CAP_HIT, 'r', cost_usd=0.0)

    def test_bare_string_cause_is_rejected_naming_field_and_value(self):
        bare_cause: Any = 'cap_hit'
        with pytest.raises(TypeError, match=r"cause.*'cap_hit'"):
            Unmeasured(MeasurementStage.ARCHITECT, bare_cause, 'r', cost_usd=0.0)

    @pytest.mark.parametrize('bad', [float('nan'), -1.0, float('inf')])
    def test_scored_rejects_non_finite_or_negative_spend(self, bad):
        with pytest.raises(ValueError, match='cost_usd') as exc:
            Scored(1, cost_usd=bad)
        assert repr(bad) in str(exc.value)

    @pytest.mark.parametrize('bad', [float('nan'), -1.0, float('inf')])
    def test_unmeasured_rejects_non_finite_or_negative_spend(self, bad):
        with pytest.raises(ValueError, match='cost_usd') as exc:
            Unmeasured(
                MeasurementStage.JUDGE, UnmeasuredCause.RAISED, 'r', cost_usd=bad,
            )
        assert repr(bad) in str(exc.value)


class TestReasonHelpers:
    def test_clip_reason_collapses_whitespace_and_clips(self):
        clipped = clip_reason('a\n  b' + 'x' * 200)

        assert clipped == ('a b' + 'x' * 200)[:80]
        assert len(clipped) <= 80

    def test_exception_reason_names_the_type(self):
        assert exception_reason(RuntimeError('judge   exploded')) == (
            'RuntimeError: judge exploded'
        )


class TestCoerceCostUsd:
    def test_a_real_figure_passes_through(self):
        assert coerce_cost_usd(1.5) == 1.5

    @pytest.mark.parametrize('placeholder', [MagicMock(), None, '3'])
    def test_a_non_number_is_zero(self, placeholder):
        # A bare MagicMock's float() is 1.0: an isinstance gate, not float().
        assert coerce_cost_usd(placeholder) == 0.0

    @pytest.mark.parametrize('nonsense', [float('nan'), float('inf'), -2.0])
    def test_a_nonsense_number_is_zero_with_one_warning(self, nonsense, caplog):
        caplog.set_level(logging.WARNING, logger='orchestrator.evals.measurement')

        assert coerce_cost_usd(nonsense) == 0.0
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1


class TestClassifyUnmeasured:
    def test_cap_payload_is_a_cap_hit_naming_the_reason(self):
        u = classify_unmeasured(
            _cap_agent_result(cost_usd=0.3), stage=MeasurementStage.JUDGE,
            backend='claude',
        )

        assert isinstance(u, Unmeasured)
        assert u.stage is MeasurementStage.JUDGE
        assert u.cause is UnmeasuredCause.CAP_HIT
        assert 'session limit' in u.reason
        assert u.cost_usd == 0.3

    def test_spend_is_coerced(self):
        u = _classify(_cap_agent_result(cost_usd=math.nan))

        assert u is not None and u.cost_usd == 0.0

    def test_successful_run_quoting_cap_text_is_not_unmeasured(self):
        ok = AgentResult(
            success=True, output=f'the CLI printed: {_CAP_TEXT}', cost_usd=1.5,
            duration_ms=9000, turns=12,
        )
        assert _classify(ok) is None

    def test_body_less_429_is_an_api_error(self):
        bare = AgentResult(
            success=False, output='', stderr='', cost_usd=0.0, duration_ms=800,
            turns=0, api_error_status=429,
        )

        u = _classify(bare)

        assert u is not None
        assert u.cause is UnmeasuredCause.API_ERROR
        assert u.reason == 'HTTP 429'

    def test_auth_failure_is_auth_failed(self):
        unauthorized = AgentResult(
            success=False, output='', cost_usd=0.0, duration_ms=200, turns=0,
            api_error_status=401,
        )

        u = _classify(unauthorized)

        assert u is not None
        assert u.cause is UnmeasuredCause.AUTH_FAILED
        assert u.reason == 'HTTP 401'

    def test_ordinary_content_failure_is_not_unmeasured(self):
        """An architect that really ran and planned badly is a CONTENT signal."""
        content_failure = AgentResult(
            success=False, output='I could not complete the plan', cost_usd=1.2,
            duration_ms=45000, turns=8, api_error_status=None,
        )
        assert _classify(content_failure) is None

    def test_zero_output_wedge_is_a_wedge(self):
        u = _classify(_wedged_agent_result())

        assert u is not None
        assert u.cause is UnmeasuredCause.WEDGE

    def test_body_less_429_outranks_the_wedge_tier(self):
        wedged_429 = AgentResult(
            success=False, output='', cost_usd=0.0, duration_ms=600_000, turns=0,
            timed_out=True, transcript_turns=0, api_error_status=429,
        )

        u = _classify(wedged_429)

        assert u is not None
        assert u.cause is UnmeasuredCause.API_ERROR

    def test_near_cap_warning_is_not_unmeasured(self):
        near = AgentResult(
            success=False, output="You're close to your usage limit for this session",
            cost_usd=0.9, duration_ms=30_000, turns=6,
        )
        assert _classify(near) is None

    def test_cli_local_error_is_not_unmeasured(self):
        local = _cli_local_error_result(next(iter(NON_CAP_CLI_ERROR_MARKERS)))
        assert isinstance(classify_invocation(local, strict_confirm=True), CliLocalError)

        assert _classify(local) is None

    def test_codex_cap_pattern_fires_only_under_the_codex_backend(self):
        codex_capped = _codex_cap_agent_result()

        u = _classify(codex_capped, backend='codex')

        assert u is not None and u.cause is UnmeasuredCause.CAP_HIT
        assert _classify(codex_capped, backend='claude') is None

    @pytest.mark.parametrize('status', [500, 502, 503, 529, 599])
    def test_every_5xx_status_in_the_band_is_a_server_error(self, status):
        u = _classify(_server_error_result(status))

        assert u is not None
        assert u.cause is UnmeasuredCause.SERVER_ERROR
        assert str(status) in u.reason

    def test_timed_out_flushed_5xx_is_a_server_error_not_a_wedge(self):
        u = _classify(_flushed_server_error_result(529))

        assert u is not None
        assert u.cause is UnmeasuredCause.SERVER_ERROR
        assert '529' in u.reason

    def test_near_cap_warning_outranks_the_server_error_tier(self):
        near_with_5xx = AgentResult(
            success=False, output="You're close to your usage limit for this session",
            cost_usd=0.9, duration_ms=30_000, turns=6, api_error_status=529,
        )
        assert _classify(near_with_5xx) is None


def _answer(output: str, *, cost_usd: float = 0.0, structured_output: Any = None) -> AgentResult:
    return AgentResult(
        success=True, output=output, cost_usd=cost_usd, duration_ms=1000,
        turns=1, structured_output=structured_output,
    )


def _recording_invoke(
    *results: AgentResult | BaseException,
) -> tuple[Callable[..., Awaitable[AgentResult]], list[dict[str, Any]]]:
    """An invoke function answering *results* in order and recording its kwargs."""
    pending = list(results)
    calls: list[dict[str, Any]] = []

    async def invoke(**kwargs: Any) -> AgentResult:
        calls.append(kwargs)
        answer = pending.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    return invoke, calls


def _never_parse(result: AgentResult) -> Any:
    raise AssertionError('parse must not run on a refused invocation')


def _raising_parse(exc: BaseException) -> Callable[[AgentResult], Any]:
    def parse(result: AgentResult) -> Any:
        raise exc

    return parse


class TestJsonAnswer:
    def test_structured_output_is_returned_as_is_and_preferred(self):
        payload = {'x': 1}

        assert json_answer(_answer('{"y": 0}', structured_output=payload)) is payload

    @pytest.mark.parametrize('empty', [None, {}])
    def test_empty_structured_output_falls_back_to_the_output_text(self, empty):
        assert json_answer(_answer('{"y": 0}', structured_output=empty)) == {'y': 0}

    def test_invalid_json_is_a_value_error(self):
        with pytest.raises(json.JSONDecodeError):
            json_answer(_answer('not json'))

    @pytest.mark.parametrize('output', ['[1, 2]', '"text"'])
    def test_a_non_object_payload_is_a_value_error(self, output):
        with pytest.raises(ValueError, match='JSON object'):
            json_answer(_answer(output))


@pytest.mark.asyncio
class TestMeasuredCall:
    async def test_ungated_success_is_scored_with_its_spend(self):
        invoke, calls = _recording_invoke(_answer('{"x": 1}', cost_usd=0.4))

        outcome = await measured_call(
            invoke, json_answer, stage=MeasurementStage.JUDGE, backend='claude',
            prompt='P', model='opus',
        )

        assert outcome == Scored(value={'x': 1}, cost_usd=0.4)
        assert len(calls) == 1
        assert calls[0]['prompt'] == 'P'
        assert calls[0]['model'] == 'opus'
        assert calls[0]['backend'] == 'claude'

    async def test_transport_refusal_is_unmeasured_and_never_parsed(self):
        invoke, _ = _recording_invoke(_cap_agent_result(cost_usd=0.3))

        outcome = await measured_call(
            invoke, _never_parse, stage=MeasurementStage.JUDGE, backend='claude',
            prompt='P',
        )

        assert isinstance(outcome, Unmeasured)
        assert outcome.stage is MeasurementStage.JUDGE
        assert outcome.cause is UnmeasuredCause.CAP_HIT
        assert outcome.cost_usd == 0.3

    async def test_unparseable_answer_is_unmeasured_with_one_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger='orchestrator.evals.measurement')
        invoke, _ = _recording_invoke(_answer('not json', cost_usd=0.7))

        outcome = await measured_call(
            invoke, json_answer, stage=MeasurementStage.JUDGE, backend='claude',
            label='Plan judge for df_task_7', prompt='P',
        )

        assert isinstance(outcome, Unmeasured)
        assert outcome.cause is UnmeasuredCause.UNPARSEABLE
        assert outcome.cost_usd == 0.7
        assert 'JSONDecodeError' in outcome.reason
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert 'Plan judge for df_task_7' in warnings[0].getMessage()

    @pytest.mark.parametrize('exc', [ValueError('non-finite'), KeyError('plan_quality')])
    async def test_a_parse_rejection_is_unparseable(self, exc):
        invoke, _ = _recording_invoke(_answer('{}', cost_usd=0.1))

        outcome = await measured_call(
            invoke, _raising_parse(exc), stage=MeasurementStage.JUDGE,
            backend='claude', prompt='P',
        )

        assert isinstance(outcome, Unmeasured)
        assert outcome.cause is UnmeasuredCause.UNPARSEABLE

    async def test_a_parse_programming_error_propagates(self):
        invoke, _ = _recording_invoke(_answer('{}'))

        with pytest.raises(RuntimeError, match='bug'):
            await measured_call(
                invoke, _raising_parse(RuntimeError('bug')),
                stage=MeasurementStage.JUDGE, backend='claude', prompt='P',
            )

    async def test_healthy_result_quoting_cap_text_is_scored(self):
        invoke, _ = _recording_invoke(_answer(f'the CLI printed: {_CAP_TEXT}'))

        outcome = await measured_call(
            invoke, lambda result: result.output, stage=MeasurementStage.JUDGE,
            backend='claude', prompt='P',
        )

        assert outcome == Scored(value=f'the CLI printed: {_CAP_TEXT}', cost_usd=0.0)

    async def test_an_invoke_error_propagates_unchanged(self):
        boom = RuntimeError('transport exploded')
        invoke, _ = _recording_invoke(boom)

        with pytest.raises(RuntimeError) as exc:
            await measured_call(
                invoke, json_answer, stage=MeasurementStage.JUDGE,
                backend='claude', prompt='P',
            )
        assert exc.value is boom

    async def test_cap_patience_exhausted_is_unmeasured(self):
        invoke, _ = _recording_invoke(AllAccountsCappedException(3, 10.0, 'x'))

        outcome = await measured_call(
            invoke, _never_parse, stage=MeasurementStage.JUDGE, backend='claude',
            prompt='P',
        )

        assert isinstance(outcome, Unmeasured)
        assert outcome.cause is UnmeasuredCause.CAP_EXHAUSTED
        assert outcome.cost_usd == 0.0

    async def test_gated_failover_banks_spend_per_attempt(self):
        gate = build_usage_gate(
            [
                AccountConfig(name='max-a', oauth_token_env='CLAUDE_OAUTH_A'),
                AccountConfig(name='max-b', oauth_token_env='CLAUDE_OAUTH_B'),
            ],
            ['tok-a', 'tok-b'],
        )
        invoke, calls = _recording_invoke(
            _cap_agent_result(cost_usd=0.2), _answer('{"x": 2}', cost_usd=0.5),
        )

        with patch('shared.cli_invoke.asyncio.sleep', AsyncMock()):
            outcome = await measured_call(
                invoke, json_answer, stage=MeasurementStage.JUDGE,
                backend='claude', usage_gate=gate, prompt='P',
            )

        assert isinstance(outcome, Scored)
        assert outcome.value == {'x': 2}
        assert outcome.cost_usd == pytest.approx(0.7)
        assert len(calls) == 2
        assert calls[1]['oauth_token'] == 'tok-b'
        assert calls[1]['prompt'] == 'P'
