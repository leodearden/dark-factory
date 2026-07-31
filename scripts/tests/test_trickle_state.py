"""Tests for scripts/legibility/trickle_state.py — the nightly trickle
run-state recorder that lets a probe answer WHY a night produced nothing.

The classifier under test is not a tuned heuristic; it is a derivation from
:class:`legibility.sampling.SampleResult`'s CONSERVATION INVARIANT::

    total_records == zero_signal_dropped + dedupe_collapsed
                     + below_sampling_cut + budget_skipped + len(selected)

Every counter fixture below therefore SATISFIES that identity — an
inconsistent fixture would prove nothing about a classifier derived from it.

The three outcomes:

- ``productive`` — ``selected > 0``; digests were built.
- ``barren``     — ``selected == 0`` and real, distinct, non-duplicate signal
  reached the sampling/budget stage and NOTHING was digested.
- ``quiet``      — everything else, which BY THE INVARIANT means every
  enumerated record left by the zero-signal or dedupe door (or nothing was
  enumerated at all). That is exactly the "genuinely quiet night" PRD
  decision 7 protects, so a quiet or dormant project can never be
  classified barren — the no-false-alarm guarantee as a PROOF from the
  invariant rather than a threshold someone picked.
"""
from __future__ import annotations

import pytest

from legibility.trickle_state import (
    OUTCOME_BARREN,
    OUTCOME_PRODUCTIVE,
    OUTCOME_QUIET,
    classify_run,
)


def _counters(
    *,
    zero_signal_dropped=0,
    dedupe_collapsed=0,
    below_sampling_cut=0,
    budget_skipped=0,
    selected_count=0,
):
    """Build a classify_run kwargs dict whose ``total_records`` is DERIVED
    from the other five counters, so every fixture satisfies SampleResult's
    conservation identity by construction."""
    return dict(
        total_records=(
            zero_signal_dropped
            + dedupe_collapsed
            + below_sampling_cut
            + budget_skipped
            + selected_count
        ),
        zero_signal_dropped=zero_signal_dropped,
        dedupe_collapsed=dedupe_collapsed,
        below_sampling_cut=below_sampling_cut,
        budget_skipped=budget_skipped,
        selected_count=selected_count,
    )


class TestClassifyRun:
    """The three-valued absence classifier."""

    def test_selected_is_productive(self):
        assert classify_run(**_counters(selected_count=3)) == OUTCOME_PRODUCTIVE

    def test_partially_truncated_night_is_still_productive(self):
        """A night that digested SOMETHING and skipped the rest on budget is
        the byte budget working as designed — never barren."""
        result = classify_run(**_counters(selected_count=2, budget_skipped=9))
        assert result == OUTCOME_PRODUCTIVE

    def test_selected_with_every_other_door_open_is_productive(self):
        result = classify_run(
            **_counters(
                selected_count=1,
                budget_skipped=4,
                below_sampling_cut=7,
                dedupe_collapsed=2,
                zero_signal_dropped=5,
            )
        )
        assert result == OUTCOME_PRODUCTIVE

    def test_budget_door_is_barren(self):
        """Reproduction of the real 2026-07-16..29 incident: candidates
        existed, competed, and were ALL discarded on the byte budget."""
        result = classify_run(**_counters(selected_count=0, budget_skipped=4))
        assert result == OUTCOME_BARREN

    def test_sampling_cut_door_is_barren(self):
        """The sibling absence mode task 3270 does NOT cover: real, distinct
        signal held back by the sampling cut, nothing digested. Different
        remedy (sampling.top_fraction/per_stratum_min, never
        budgets.max_daily_digest_bytes) — see SampleResult's docstring."""
        result = classify_run(
            **_counters(selected_count=0, below_sampling_cut=3, budget_skipped=0)
        )
        assert result == OUTCOME_BARREN

    def test_both_doors_open_is_barren(self):
        result = classify_run(
            **_counters(selected_count=0, below_sampling_cut=3, budget_skipped=4)
        )
        assert result == OUTCOME_BARREN

    def test_dormant_project_is_quiet(self):
        """All counters zero — nothing was even enumerated. A dormant
        project is a legitimate state, not a degradation."""
        assert classify_run(**_counters()) == OUTCOME_QUIET

    def test_all_zero_signal_is_quiet(self):
        result = classify_run(**_counters(zero_signal_dropped=17))
        assert result == OUTCOME_QUIET

    def test_zero_signal_plus_dedupe_only_is_quiet(self):
        result = classify_run(
            **_counters(zero_signal_dropped=6, dedupe_collapsed=3)
        )
        assert result == OUTCOME_QUIET

    def test_outcome_constants_are_distinct_strings(self):
        outcomes = {OUTCOME_PRODUCTIVE, OUTCOME_QUIET, OUTCOME_BARREN}
        assert len(outcomes) == 3
        assert all(isinstance(o, str) and o for o in outcomes)


class TestQuietNightNeverBarren:
    """PRD decision 7's no-false-alarm guarantee, in executable form.

    Given its own named test rather than hiding inside a parametrize list:
    this is THE property that lets a progress probe exist at all without
    re-opening the false-alarm objection decision 7 raised against
    git-history probes.
    """

    @pytest.mark.parametrize("zero_signal_dropped", range(0, 6))
    @pytest.mark.parametrize("dedupe_collapsed", range(0, 6))
    def test_a_quiet_night_is_never_barren(
        self, zero_signal_dropped, dedupe_collapsed
    ):
        """With BOTH cut counters at 0 and nothing selected, no combination
        of zero-signal/dedupe volume may ever classify barren."""
        result = classify_run(
            **_counters(
                zero_signal_dropped=zero_signal_dropped,
                dedupe_collapsed=dedupe_collapsed,
                below_sampling_cut=0,
                budget_skipped=0,
                selected_count=0,
            )
        )
        assert result == OUTCOME_QUIET, (
            f"zero_signal_dropped={zero_signal_dropped} "
            f"dedupe_collapsed={dedupe_collapsed} classified {result!r}; a "
            f"night where every record left by the zero-signal or dedupe "
            f"door is exactly the 'genuinely quiet night' decision 7 "
            f"protects and must never alarm."
        )
