"""Tests for ι=1894: retries-per-landing + drift-at-detection metrics.

Covers:
  step-01 RED  — Pure MergeMetrics accumulator (class not yet created)
  step-03 RED  — snapshot() emits 'metrics' key from worker._merge_metrics
  step-05 RED  — Worker wiring helpers _note_merge_started/_note_merge_landing/
                 _note_merge_retry/_note_conflict_detected
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchestrator.git_ops import GitOps
from orchestrator.merge_queue import MergeMetrics, SpeculativeMergeWorker

# ---------------------------------------------------------------------------
# Worker factory (mirrors test_merge_queue_main_health._make_git_ops pattern)
# ---------------------------------------------------------------------------


def _make_bare_worker(tmp_path: Path | None = None) -> SpeculativeMergeWorker:
    """Build a SpeculativeMergeWorker with a mocked GitOps (no real git repo)."""
    git_ops = MagicMock(spec=GitOps)
    git_ops.project_root = tmp_path  # None-safe: __init__ guards on this
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


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


# ---------------------------------------------------------------------------
# step-03: snapshot() emits 'metrics' key
# ---------------------------------------------------------------------------


class TestWorkerSnapshotMetricsKey:
    """snapshot() must include a top-level 'metrics' key from _merge_metrics.

    RED until step-04 GREEN adds _merge_metrics to __init__ and 'metrics' to
    snapshot().
    """

    def test_snapshot_has_metrics_key(self):
        """snapshot() includes a 'metrics' key."""
        worker = _make_bare_worker()
        snap = worker.snapshot()
        assert 'metrics' in snap

    def test_snapshot_metrics_matches_accumulator(self):
        """snapshot()['metrics'] equals worker._merge_metrics.as_snapshot()."""
        worker = _make_bare_worker()
        # Drive the accumulator directly
        worker._merge_metrics.record_landing()
        worker._merge_metrics.record_landing()
        worker._merge_metrics.record_retry()
        worker._merge_metrics.record_drift(3)
        snap = worker.snapshot()
        expected = worker._merge_metrics.as_snapshot()
        assert snap['metrics'] == expected

    def test_snapshot_metrics_retries_per_landing_correct(self):
        """snapshot()['metrics']['retries_per_landing'] is numerically correct."""
        worker = _make_bare_worker()
        worker._merge_metrics.record_landing()
        worker._merge_metrics.record_landing()
        worker._merge_metrics.record_retry()
        worker._merge_metrics.record_retry()
        worker._merge_metrics.record_retry()
        snap = worker.snapshot()
        assert snap['metrics']['retries_per_landing'] == pytest.approx(1.5)

    def test_snapshot_metrics_drift_at_detection_correct(self):
        """snapshot()['metrics']['drift_at_detection'] has correct fields."""
        worker = _make_bare_worker()
        worker._merge_metrics.record_drift(5)
        snap = worker.snapshot()
        dd = snap['metrics']['drift_at_detection']
        assert dd['count'] == 1
        assert dd['last'] == 5
        assert dd['max'] == 5

    def test_snapshot_metrics_zero_state(self):
        """snapshot()['metrics'] on a fresh worker has None rpl and zero landings/retries."""
        worker = _make_bare_worker()
        snap = worker.snapshot()
        m = snap['metrics']
        assert m['retries_per_landing'] is None
        assert m['landings_total'] == 0
        assert m['retries_total'] == 0
        assert m['drift_at_detection']['count'] == 0

    def test_snapshot_backward_compat_keys_present(self):
        """Pre-existing snapshot keys are still present alongside 'metrics'."""
        worker = _make_bare_worker()
        snap = worker.snapshot()
        for key in ('entries', 'depth', 'head_of_line', 'suffix_conflict_graph'):
            assert key in snap, f"pre-existing key '{key}' missing from snapshot"


# ---------------------------------------------------------------------------
# step-05: worker wiring helpers
# ---------------------------------------------------------------------------


class TestWorkerWiringHelpers:
    """Worker helper methods delegate to _merge_metrics and _drift_base.

    RED until step-06 GREEN adds the four helper methods to SpeculativeMergeWorker.
    """

    def test_note_merge_started_stashes_main_position(self):
        """_note_merge_started(rid) stores current main_position into _drift_base."""
        worker = _make_bare_worker()
        # Advance main_position via 2 landings, then start a request
        worker._merge_metrics.record_landing()
        worker._merge_metrics.record_landing()
        worker._note_merge_started('req-abc')
        assert worker._drift_base['req-abc'] == 2

    def test_note_merge_started_at_zero(self):
        """_note_merge_started stashes position 0 on a fresh worker."""
        worker = _make_bare_worker()
        worker._note_merge_started('req-zero')
        assert worker._drift_base['req-zero'] == 0

    def test_note_merge_landing_increments_landings_and_pops_drift_base(self):
        """_note_merge_landing increments landings and pops _drift_base entry."""
        worker = _make_bare_worker()
        worker._note_merge_started('req-land')
        assert 'req-land' in worker._drift_base

        worker._note_merge_landing('req-land')

        assert worker._merge_metrics.landings == 1
        assert 'req-land' not in worker._drift_base

    def test_note_merge_landing_pops_only_own_entry(self):
        """_note_merge_landing only pops its own request_id from _drift_base."""
        worker = _make_bare_worker()
        worker._note_merge_started('req-a')
        worker._note_merge_started('req-b')
        worker._note_merge_landing('req-a')
        assert 'req-b' in worker._drift_base
        assert 'req-a' not in worker._drift_base

    def test_note_merge_retry_increments_retries(self):
        """_note_merge_retry() increments the retries counter."""
        worker = _make_bare_worker()
        assert worker._merge_metrics.retries == 0
        worker._note_merge_retry()
        assert worker._merge_metrics.retries == 1
        worker._note_merge_retry()
        assert worker._merge_metrics.retries == 2

    def test_note_conflict_detected_records_drift_and_pops(self):
        """_note_conflict_detected(rid) records drift = (main_position - base) and pops."""
        worker = _make_bare_worker()
        # Start req-c at position 0
        worker._note_merge_started('req-c')
        # Simulate 3 clean landings (main advances by 3)
        worker._merge_metrics.record_landing()
        worker._merge_metrics.record_landing()
        worker._merge_metrics.record_landing()
        # Detect conflict → drift should be 3 - 0 = 3
        worker._note_conflict_detected('req-c')

        snap = worker.snapshot()
        dd = snap['metrics']['drift_at_detection']
        assert dd['count'] == 1
        assert dd['last'] == 3
        assert 'req-c' not in worker._drift_base

    def test_concrete_scenario_drift_equals_intervening_landings(self):
        """Full scenario: start A at pos 0, land 3 times, conflict A → drift=3."""
        worker = _make_bare_worker()
        worker._note_merge_started('req-A')
        # 3 clean landings by other requests
        worker._note_merge_landing('req-X')
        worker._note_merge_landing('req-Y')
        worker._note_merge_landing('req-Z')
        # conflict on req-A
        worker._note_conflict_detected('req-A')
        snap = worker.snapshot()
        assert snap['metrics']['drift_at_detection']['last'] == 3

    def test_note_conflict_detected_noop_when_not_started(self):
        """_note_conflict_detected for unknown rid doesn't raise (defensive)."""
        worker = _make_bare_worker()
        # Should not raise even if req was never started
        worker._note_conflict_detected('req-missing')
