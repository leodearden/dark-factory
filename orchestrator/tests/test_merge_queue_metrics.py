"""Tests for ι=1894: retries-per-landing + drift-at-detection metrics.

Covers:
  step-01 RED  — Pure MergeMetrics accumulator (class not yet created)
  step-03 RED  — snapshot() emits 'metrics' key from worker._merge_metrics
  step-05 RED  — Worker wiring helpers _note_merge_started/_note_merge_landing/
                 _note_merge_retry/_note_conflict_detected
"""
from __future__ import annotations

import asyncio
import pytest

from orchestrator.merge_queue import MergeMetrics, SpeculativeMergeWorker


# ---------------------------------------------------------------------------
# step-01: MergeMetrics pure accumulator
# ---------------------------------------------------------------------------


class TestMergeMetrics:
    """Pure unit tests for the MergeMetrics accumulator.

    RED until step-02 GREEN adds MergeMetrics to merge_queue.py.
    """

    def test_initial_state_zero_counts(self):
        """Fresh accumulator has zero landings, retries, and empty drift."""
        m = MergeMetrics()
        assert m.landings == 0
        assert m.retries == 0
        assert m.main_position == 0

    def test_main_position_equals_landings(self):
        """main_position tracks the number of landings (each landing advances main by 1)."""
        m = MergeMetrics()
        m.record_landing()
        assert m.main_position == 1
        m.record_landing()
        m.record_landing()
        assert m.main_position == 3

    def test_retries_per_landing_none_when_no_landings(self):
        """retries_per_landing is None when landings == 0 (no div-by-zero)."""
        m = MergeMetrics()
        assert m.retries_per_landing is None

    def test_retries_per_landing_zero_when_no_retries(self):
        """retries_per_landing is 0.0 when there are landings but no retries."""
        m = MergeMetrics()
        m.record_landing()
        assert m.retries_per_landing == 0.0

    def test_retries_per_landing_exact_arithmetic(self):
        """retries_per_landing = retries / landings exactly (3 retries / 2 landings == 1.5)."""
        m = MergeMetrics()
        m.record_landing()
        m.record_landing()
        m.record_retry()
        m.record_retry()
        m.record_retry()
        assert m.retries_per_landing == 1.5

    def test_record_drift_populates_summary(self):
        """record_drift(n) feeds the drift summary with correct values."""
        m = MergeMetrics()
        m.record_drift(3)
        s = m.drift_summary()
        assert s['count'] == 1
        assert s['last'] == 3
        assert s['mean'] == 3.0
        assert s['max'] == 3

    def test_drift_summary_multi_sample(self):
        """drift_summary() returns correct count/last/mean/max across multiple samples."""
        m = MergeMetrics()
        m.record_drift(2)
        m.record_drift(6)
        m.record_drift(4)
        s = m.drift_summary()
        assert s['count'] == 3
        assert s['last'] == 4
        assert s['mean'] == pytest.approx(4.0)
        assert s['max'] == 6

    def test_drift_summary_empty_when_no_drifts(self):
        """drift_summary() returns all-zero/None values when no drifts recorded."""
        m = MergeMetrics()
        s = m.drift_summary()
        assert s['count'] == 0
        assert s['last'] is None
        assert s['mean'] is None
        assert s['max'] is None

    def test_drift_buffer_is_bounded(self):
        """Drift samples are bounded — oldest samples drop past the window cap."""
        m = MergeMetrics(drift_window=5)
        for i in range(10):
            m.record_drift(i)
        s = m.drift_summary()
        # Only the 5 most-recent samples (5..9) remain
        assert s['count'] == 5
        assert s['max'] == 9
        # The 5 oldest (0..4) are gone — mean of 5..9 = 7.0
        assert s['mean'] == pytest.approx(7.0)

    def test_as_snapshot_shape(self):
        """as_snapshot() returns the expected dict keys."""
        m = MergeMetrics()
        m.record_landing()
        m.record_retry()
        m.record_drift(2)
        snap = m.as_snapshot()
        assert 'retries_per_landing' in snap
        assert 'drift_at_detection' in snap
        assert 'landings_total' in snap
        assert 'retries_total' in snap
        assert snap['landings_total'] == 1
        assert snap['retries_total'] == 1
        assert snap['retries_per_landing'] == 1.0
        assert snap['drift_at_detection']['count'] == 1
        assert snap['drift_at_detection']['last'] == 2

    def test_as_snapshot_none_rpl_with_no_landings(self):
        """as_snapshot() carries retries_per_landing=None when landings==0."""
        m = MergeMetrics()
        snap = m.as_snapshot()
        assert snap['retries_per_landing'] is None
