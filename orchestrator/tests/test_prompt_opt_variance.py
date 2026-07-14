"""Tests for orchestrator.evals.prompt_opt.variance — the paired variance gate (T6).

See plans/tier1-prompt-optimization-prd.md T6 / D-5: acceptance must never be
bare tie-rejection over a noisy (haiku) matcher. `textual_lr` is the bounded
edit-budget schedule (LR 4 decaying to 2); `paired_delta` and
`measure_repeatability_band` build the pre-measured noise floor;
`evaluate_acceptance` accepts ONLY when the paired delta strictly exceeds
that band.
"""

from __future__ import annotations

import pytest

from orchestrator.evals.prompt_opt.variance import (
    AcceptanceRecord,
    evaluate_acceptance,
    measure_repeatability_band,
    paired_delta,
    textual_lr,
)


class TestTextualLr:
    def test_step_zero_is_start(self) -> None:
        assert textual_lr(0) == 4

    def test_monotone_non_increasing(self) -> None:
        values = [textual_lr(s) for s in range(8)]
        assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))

    def test_never_exceeds_start(self) -> None:
        assert all(textual_lr(s) <= 4 for s in range(10))

    def test_never_below_end(self) -> None:
        assert all(textual_lr(s) >= 2 for s in range(10))

    def test_decays_then_clamps_at_end(self) -> None:
        assert textual_lr(1) < textual_lr(0)
        assert textual_lr(2) == 2
        assert textual_lr(10) == 2

    def test_custom_start_end(self) -> None:
        assert textual_lr(0, start=6, end=3) == 6
        assert textual_lr(10, start=6, end=3) == 3


class TestPairedDelta:
    def test_mean_per_item_difference(self) -> None:
        base = [0.5, 0.6, 0.7]
        cand = [0.6, 0.6, 0.9]
        assert paired_delta(base, cand) == pytest.approx(0.1)

    def test_negative_when_candidate_worse(self) -> None:
        assert paired_delta([0.8, 0.8], [0.5, 0.5]) == pytest.approx(-0.3)

    def test_zero_when_identical(self) -> None:
        assert paired_delta([0.5, 0.5], [0.5, 0.5]) == 0.0

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError):
            paired_delta([0.5, 0.5], [0.5])


class TestMeasureRepeatabilityBand:
    def test_zero_for_identical_batches(self) -> None:
        batches = [[0.5, 0.6, 0.7], [0.5, 0.6, 0.7], [0.5, 0.6, 0.7]]
        assert measure_repeatability_band(batches) == 0.0

    def test_positive_for_spread_batches(self) -> None:
        batches = [[0.5, 0.6, 0.7], [0.55, 0.6, 0.65], [0.45, 0.6, 0.75]]
        assert measure_repeatability_band(batches) > 0.0

    def test_non_negative(self) -> None:
        batches = [[0.1, 0.9], [0.9, 0.1]]
        assert measure_repeatability_band(batches) >= 0.0

    def test_requires_at_least_two_batches(self) -> None:
        with pytest.raises(ValueError):
            measure_repeatability_band([[0.5, 0.6]])


class TestEvaluateAcceptance:
    def test_returns_acceptance_record(self) -> None:
        record = evaluate_acceptance([[0.5]], [[0.9]], band=0.1)
        assert isinstance(record, AcceptanceRecord)

    def test_accepts_when_delta_strictly_exceeds_band(self) -> None:
        base_repeats = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
        cand_repeats = [[0.9, 0.9], [0.9, 0.9], [0.9, 0.9]]
        record = evaluate_acceptance(base_repeats, cand_repeats, band=0.1)

        assert record.accepted is True
        assert record.paired_delta == pytest.approx(0.4)
        assert record.band == 0.1

    def test_rejects_zero_tie_delta(self) -> None:
        base_repeats = [[0.5, 0.5], [0.5, 0.5]]
        cand_repeats = [[0.5, 0.5], [0.5, 0.5]]
        record = evaluate_acceptance(base_repeats, cand_repeats, band=0.0)

        assert record.accepted is False
        assert record.paired_delta == pytest.approx(0.0)

    def test_rejects_positive_but_within_band_delta(self) -> None:
        # Never bare tie-rejection: a POSITIVE delta must still be rejected
        # when it does not clear the measured noise floor.
        base_repeats = [[0.5, 0.5], [0.5, 0.5]]
        cand_repeats = [[0.52, 0.52], [0.52, 0.52]]
        record = evaluate_acceptance(base_repeats, cand_repeats, band=0.05)

        assert record.paired_delta == pytest.approx(0.02)
        assert record.paired_delta > 0
        assert record.accepted is False

    def test_reason_is_a_populated_string(self) -> None:
        record = evaluate_acceptance([[0.5]], [[0.9]], band=0.1)
        assert isinstance(record.reason, str)
        assert record.reason
