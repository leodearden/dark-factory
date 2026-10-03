"""Tests for measure_write_triage_judge_call.py — the judge call-cost probe (task 6076).

The probe's pure core is driven with literals; its live edge drives the SHIPPED
``judge_write`` against a faked ``AsyncOpenAI``. No network, no API key.
"""
from __future__ import annotations

import functools
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _fm_helpers import load_script_module

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.server.write_triage import (
    OUTCOME_STORED,
    JudgeUsage,
    TriageJudgeVerdict,
)
from fused_memory.server.write_triage_judge import (
    _ELIDED_MARKER,
    build_judge_prompt,
    resolve_judge_candidate_count,
    resolve_judge_timeout,
    select_judge_candidates,
)

SCRIPTS = Path(__file__).parent.parent / 'scripts'

_TEXTS = [
    'The merge worker stands off while the index lock is held.',
    'A docs-only commit skips pyright and finishes quickly.',
    'Never run git stash in any dark-factory checkout.',
]


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'measure_write_triage_judge_call.py', 'measure_write_triage_judge_call',
    )


def _sample(seconds: float, input_tokens: int | None = 1_000, error: str | None = None):
    outcome = None if error else OUTCOME_STORED
    return _mod().CallSample(seconds, input_tokens, outcome, error)


class TestWorstCaseSlate:
    """Every field elided, every id production-shaped, all of the slate shown."""

    @pytest.mark.parametrize('width', [1, 5, 20])
    def test_the_slate_has_width_unique_36_char_ids(self, width: int) -> None:
        _, candidates = _mod().worst_case_slate(_TEXTS, width, 100)
        assert len(candidates) == width
        assert len({c.id for c in candidates}) == width
        assert {len(c.id) for c in candidates} == {36}

    def test_it_is_deterministic(self) -> None:
        first = _mod().worst_case_slate(_TEXTS, 5, 100)
        second = _mod().worst_case_slate(_TEXTS, 5, 100)
        assert first[0] == second[0]
        assert [(c.id, c.content) for c in first[1]] == [(c.id, c.content) for c in second[1]]

    def test_every_field_is_over_the_cap_and_built_from_the_texts(self) -> None:
        entry, candidates = _mod().worst_case_slate(_TEXTS, 5, 100)
        for field in [entry, *(c.content for c in candidates)]:
            assert len(field) > 100
            assert set(field.split('\n\n')) <= set(_TEXTS)

    def test_fields_are_not_all_the_same_text(self) -> None:
        entry, candidates = _mod().worst_case_slate(_TEXTS, 2, 100)
        assert len({entry, *(c.content for c in candidates)}) > 1

    def test_the_cosines_strictly_descend(self) -> None:
        _, candidates = _mod().worst_case_slate(_TEXTS, 20, 100)
        scores = [c.metadata['store_score'] for c in candidates]
        assert all(a > b for a, b in zip(scores, scores[1:], strict=False))

    @pytest.mark.parametrize('width', [5, 10, 20])
    def test_the_judge_keeps_the_whole_slate(self, width: int) -> None:
        _, candidates = _mod().worst_case_slate(_TEXTS, width, 100)
        kept = select_judge_candidates(candidates, width, canonical_id=candidates[0].id)
        assert [c.id for c in kept] == [c.id for c in candidates]

    @pytest.mark.parametrize('width', [5, 10, 20])
    def test_every_rendered_field_is_elided(self, width: int) -> None:
        entry, candidates = _mod().worst_case_slate(_TEXTS, width, 100)
        prompt = build_judge_prompt(entry, candidates, field_chars=100)
        assert prompt.count(_ELIDED_MARKER) == width + 1

    def test_no_text_to_build_from_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _mod().worst_case_slate(['', ''], 5, 100)


class TestSummarizeWidth:
    def test_seconds_are_nearest_rank_order_statistics(self) -> None:
        samples = [_sample(float(s)) for s in range(1, 21)]
        row = _mod().summarize_width(5, samples, timeout_seconds=19.5)
        assert row['width'] == 5
        assert row['calls'] == 20
        assert row['seconds']['p95'] == 19.0
        assert row['seconds']['median'] == 10.0
        assert row['seconds']['max'] == 20.0

    @pytest.mark.parametrize(('timeout', 'within'), [(19.5, True), (19.0, False), (18.0, False)])
    def test_p95_is_judged_strictly_below_the_timeout(
        self, timeout: float, within: bool,
    ) -> None:
        samples = [_sample(float(s)) for s in range(1, 21)]
        row = _mod().summarize_width(5, samples, timeout_seconds=timeout)
        assert row['p95_within_timeout'] is within

    def test_errors_are_counted_by_type_and_kept_out_of_the_token_summary(self) -> None:
        samples = [
            _sample(1.0, 1_000),
            _sample(2.0, 3_000),
            _sample(60.0, None, 'TimeoutError'),
            _sample(60.0, None, 'TimeoutError'),
            _sample(0.5, None, 'APIError'),
        ]
        row = _mod().summarize_width(5, samples, timeout_seconds=15.0)
        assert row['errors'] == {'TimeoutError': 2, 'APIError': 1}
        assert row['input_tokens']['n'] == 2
        assert (row['input_tokens']['min'], row['input_tokens']['max']) == (1_000, 3_000)

    def test_unreported_usage_is_unmeasured_never_zero(self) -> None:
        samples = [_sample(1.0, None), _sample(2.0, None)]
        row = _mod().summarize_width(5, samples, timeout_seconds=15.0)
        assert row['input_tokens']['n'] == 0
        assert row['input_tokens']['median'] is None
        assert row['input_tokens']['max'] is None

    def test_outcomes_are_tallied(self) -> None:
        samples = [_sample(1.0), _sample(2.0), _sample(3.0, None, 'TimeoutError')]
        row = _mod().summarize_width(5, samples, timeout_seconds=15.0)
        assert row['outcomes'] == {OUTCOME_STORED: 2}


def _clock(*ticks: float):
    return iter(ticks).__next__


class TestMeasureWidth:
    @pytest.mark.asyncio
    async def test_each_call_is_timed_and_its_usage_read_off_the_verdict(self) -> None:
        call_judge = AsyncMock(side_effect=[
            TriageJudgeVerdict(OUTCOME_STORED, usage=JudgeUsage(1234, 7)),
            TriageJudgeVerdict(OUTCOME_STORED, usage=None),
        ])
        samples = await _mod().measure_width(
            'svc', 'entry', ['c'], calls=2,
            clock=_clock(10.0, 11.5, 20.0, 20.25), call_judge=call_judge,
        )
        assert call_judge.await_count == 2
        call_judge.assert_awaited_with('svc', 'entry', ['c'])
        assert samples == [
            _mod().CallSample(1.5, 1234, OUTCOME_STORED, None),
            _mod().CallSample(0.25, None, OUTCOME_STORED, None),
        ]

    @pytest.mark.asyncio
    async def test_a_failed_call_is_recorded_and_measurement_continues(self) -> None:
        call_judge = AsyncMock(side_effect=[
            TimeoutError(),
            TriageJudgeVerdict(OUTCOME_STORED, usage=JudgeUsage(900, 5)),
        ])
        samples = await _mod().measure_width(
            'svc', 'entry', ['c'], calls=2,
            clock=_clock(0.0, 60.0, 61.0, 62.0), call_judge=call_judge,
        )
        assert samples == [
            _mod().CallSample(60.0, None, None, 'TimeoutError'),
            _mod().CallSample(1.0, 900, OUTCOME_STORED, None),
        ]


def _openai_client(input_tokens: int = 1234) -> MagicMock:
    """A fake ``AsyncOpenAI`` that is its own async context manager, as the SDK is."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.responses.create = AsyncMock(return_value=types.SimpleNamespace(
        output_text='{"verdict": "distinct"}',
        usage=types.SimpleNamespace(
            input_tokens=input_tokens, output_tokens=7, output_tokens_details=None,
        ),
        status='completed',
        incomplete_details=None,
    ))
    return client


class TestLiveJudgeCall:
    """The live edge drives the SHIPPED judge_write at the configured cap."""

    @pytest.mark.asyncio
    async def test_it_returns_the_shipped_verdict_with_the_providers_usage(self) -> None:
        service = types.SimpleNamespace(config=FusedMemoryConfig())
        service.config.write_triage.judge_field_chars = 50
        entry, candidates = _mod().worst_case_slate(_TEXTS, 5, 50)
        client = _openai_client()
        with patch('openai.AsyncOpenAI', return_value=client):
            verdict = await _mod().live_judge_call(service, entry, candidates)
        assert verdict.usage is not None
        assert verdict.usage.input_tokens == 1234
        assert verdict.outcome == OUTCOME_STORED

    @pytest.mark.asyncio
    async def test_every_field_reaches_the_wire_cut_at_the_configured_cap(self) -> None:
        service = types.SimpleNamespace(config=FusedMemoryConfig())
        service.config.write_triage.judge_field_chars = 50
        entry, candidates = _mod().worst_case_slate(_TEXTS, 5, 50)
        client = _openai_client()
        with patch('openai.AsyncOpenAI', return_value=client):
            await _mod().live_judge_call(service, entry, candidates)
        rendered = client.responses.create.call_args.kwargs['input']
        for field in [entry, *(c.content for c in candidates)]:
            assert field[:50] + _ELIDED_MARKER in rendered
            assert field[:51] not in rendered

    @pytest.mark.asyncio
    @pytest.mark.parametrize('width', [10, 20])
    async def test_a_configured_width_shows_the_judge_the_whole_slate(self, width: int) -> None:
        """judge_write trims to judge_candidate_count, so the probe must widen it."""
        service = types.SimpleNamespace(config=FusedMemoryConfig())
        _mod().configure_for_width(service.config, width)
        entry, candidates = _mod().worst_case_slate(_TEXTS, width, 50)
        client = _openai_client()
        with patch('openai.AsyncOpenAI', return_value=client):
            await _mod().live_judge_call(service, entry, candidates)
        rendered = client.responses.create.call_args.kwargs['input']
        assert sum(1 for c in candidates if f'id: {c.id}\n' in rendered) == width


class TestProvenanceAndReport:
    def test_provenance_resolves_the_shipped_config(self) -> None:
        service = types.SimpleNamespace(config=FusedMemoryConfig())
        provenance = _mod().probe_provenance(service, calls_per_width=20)
        assert provenance['field_chars'] == 4_000
        assert provenance['judge_provider'] == 'openai'
        assert provenance['judge_model'] == 'gpt-4o-mini'
        assert provenance['widths'] == [5, 10, 20]
        assert provenance['calls_per_width'] == 20

    def test_the_recorded_timeout_is_the_shipped_one_not_the_measurement_ceiling(
        self,
    ) -> None:
        """The tail is OBSERVED past the timeout, but judged against the shipped one."""
        service = types.SimpleNamespace(config=FusedMemoryConfig())
        provenance = _mod().probe_provenance(service, calls_per_width=20)
        _mod().configure_for_width(service.config, 20)

        assert provenance['timeout_seconds'] == 15.0
        assert resolve_judge_timeout(service) == _mod()._MEASUREMENT_CEILING_SECONDS > 15.0
        assert resolve_judge_candidate_count(service) == 20

    def test_rows_are_reported_in_width_order(self) -> None:
        rows = [
            _mod().summarize_width(w, [_sample(1.0)], timeout_seconds=15.0)
            for w in (20, 5, 10)
        ]
        report = _mod().build_report(rows, provenance={'field_chars': 4_000})
        assert [r['width'] for r in report['widths']] == [5, 10, 20]
        assert report['provenance'] == {'field_chars': 4_000}
