"""Tests for the shared single-shot-CRITICAL filing gate (task 2558).

`critical_filing_gate(*, rerun_confirmed=False, streak=0, threshold=None)` is the
pure predicate both the verify-host-unreachable filer (consecutive-streak arm)
and the main-tip sweep filer (re-run arm) delegate to.  A single observation is
never sufficient to file a destructive/ref-move CRITICAL: filing requires either
a positive re-run confirmation OR a consecutive-streak count reaching an explicit
threshold.
"""

from __future__ import annotations

from orchestrator.critical_gate import critical_filing_gate


class TestCriticalFilingGate:
    """critical_filing_gate = rerun_confirmed or (threshold is not None and streak >= threshold)."""

    # --- (a) rerun_confirmed short-circuits regardless of streak/threshold ---

    def test_rerun_confirmed_true_wins_with_defaults(self):
        """rerun_confirmed=True → True even with streak=0, threshold=None (rerun-only caller)."""
        assert critical_filing_gate(rerun_confirmed=True) is True

    def test_rerun_confirmed_true_wins_regardless_of_streak_and_threshold(self):
        """rerun_confirmed=True → True no matter the streak/threshold values."""
        assert critical_filing_gate(rerun_confirmed=True, streak=0, threshold=None) is True
        assert critical_filing_gate(rerun_confirmed=True, streak=1, threshold=99) is True

    # --- (b) streak arm disabled when threshold is None ---

    def test_threshold_none_disables_streak_arm(self):
        """rerun_confirmed=False, threshold=None → False for any streak (no accidental 0>=0 trip)."""
        assert critical_filing_gate(rerun_confirmed=False, streak=0, threshold=None) is False
        assert critical_filing_gate(rerun_confirmed=False, streak=5, threshold=None) is False

    # --- (c) streak arm: exact streak >= threshold ---

    def test_streak_below_threshold_is_false(self):
        """threshold=3: streak=1 and streak=2 → False (under-threshold observation insufficient)."""
        assert critical_filing_gate(rerun_confirmed=False, streak=1, threshold=3) is False
        assert critical_filing_gate(rerun_confirmed=False, streak=2, threshold=3) is False

    def test_streak_at_or_above_threshold_is_true(self):
        """threshold=3: streak=3 and streak=4 → True (streak >= threshold)."""
        assert critical_filing_gate(rerun_confirmed=False, streak=3, threshold=3) is True
        assert critical_filing_gate(rerun_confirmed=False, streak=4, threshold=3) is True

    # --- (d) boundary: streak == threshold, no rerun ---

    def test_streak_equals_threshold_boundary_is_true(self):
        """threshold=3, streak=3, rerun_confirmed=False → True (boundary, exact >=)."""
        assert critical_filing_gate(rerun_confirmed=False, streak=3, threshold=3) is True


class TestRecordRunnerUnavailableDelegation:
    """Regression guard: MergeQueue._record_runner_unavailable delegation is behavior-preserving.

    Step-7 refactored ``SpeculativeMergeWorker._record_runner_unavailable`` to
    delegate to ``critical_filing_gate`` (passing an int ``threshold`` and no
    ``rerun_confirmed``).  This pins — inside this task's own diff — that the
    delegation still fires exactly at ``streak >= _unreachable_escalate_after_n``
    and not before, including the ``streak == threshold`` boundary and
    ``rerun_confirmed`` defaulting ``False``.  (Fuller tracker coverage lives in
    test_merge_queue_multihost_wiring.py, which is outside this task's scope.)
    """

    def _make_worker(self, *, escalate_after_n):
        import asyncio
        from unittest.mock import MagicMock

        from orchestrator.merge_queue import SpeculativeMergeWorker

        git_ops = MagicMock()
        git_ops.project_root = None
        q: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops=git_ops, queue=q)
        worker._unreachable_escalate_after_n = escalate_after_n
        return worker

    def test_delegation_fires_exactly_at_threshold(self):
        """streak below N → False; streak == N (boundary) and beyond → True."""
        import time

        worker = self._make_worker(escalate_after_n=3)
        t = time.time()
        r1 = worker._record_runner_unavailable('host1', 'ssh timeout', t)  # streak=1
        r2 = worker._record_runner_unavailable('host1', 'ssh timeout', t + 1)  # streak=2
        r3 = worker._record_runner_unavailable('host1', 'ssh timeout', t + 2)  # streak=3 == N
        r4 = worker._record_runner_unavailable('host1', 'ssh timeout', t + 3)  # streak=4 > N
        assert r1 is False  # 1 < 3 (single observation insufficient)
        assert r2 is False  # 2 < 3 (under-threshold observation insufficient)
        assert r3 is True  # 3 >= 3 (exact boundary, matches pre-refactor streak >= threshold)
        assert r4 is True  # 4 >= 3 (persistent alarm condition)
