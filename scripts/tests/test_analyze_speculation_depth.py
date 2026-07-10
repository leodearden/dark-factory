"""Tests for analyze_speculation_depth.py — merge-verify p_good/p_flake calibration.

Pure estimator tests run on synthetic in-memory event lists (no DB, no clock)
so expected statistics are exact hand-derived fractions — mirrors the
scripts/tests/test_reviewer_redundancy_diagnostic.py importable-module
pattern via conftest.py sys.path insertion.

Fixtures
--------
F1: 3 tasks, all eventually land, no never-landed task.
    A: [passed=True]                         (1 attempt, lands)
    B: [passed=False, passed=True]           (2 attempts, lands)
    C: [passed=True]                         (1 attempt, lands)
    -> 4 merge_verify events total, 3 passed.

F2: F1 + a never-landed task D.
    D: [passed=False, passed=False]          (2 attempts, never lands)
    -> 6 merge_verify events total, 3 passed.
"""
from __future__ import annotations

import analyze_speculation_depth as mod


def _mv(task_id, passed, *, attempt=0, depth=None):
    """Build a synthetic merge_verify event dict."""
    return {
        'task_id': task_id,
        'data': {'passed': passed, 'attempt': attempt, 'depth': depth},
    }


def _ma(task_id, outcome):
    """Build a synthetic merge_attempt event dict."""
    return {'task_id': task_id, 'data': {'outcome': outcome}}


# ---------------------------------------------------------------------------
# Fixture F1: all land, no never-landed task
# ---------------------------------------------------------------------------

F1_MERGE_VERIFY = [
    _mv('A', True),
    _mv('B', False),
    _mv('B', True),
    _mv('C', True),
]

# ---------------------------------------------------------------------------
# Fixture F2: F1 + a never-landed task D
# ---------------------------------------------------------------------------

F2_MERGE_VERIFY = [
    *F1_MERGE_VERIFY,
    _mv('D', False),
    _mv('D', False),
]

F2_MERGE_ATTEMPT = [
    _ma('A', 'done'),
    _ma('B', 'done'),
    _ma('C', 'done'),
    _ma('D', 'conflict'),
]


class TestComputeCalibrationF1:
    """F1: 3/4 attempts pass, all 3 tasks eventually land."""

    def test_per_attempt_pass_rate(self):
        cal = mod.compute_calibration(F1_MERGE_VERIFY, [])
        assert cal['per_attempt_pass_rate'] == 0.75

    def test_p_flake_via_attempts_to_first_pass_agrees(self):
        cal = mod.compute_calibration(F1_MERGE_VERIFY, [])
        assert cal['p_flake_landed'] == 0.75
        assert cal['p_flake_landed'] == cal['per_attempt_pass_rate']

    def test_retry_histogram(self):
        cal = mod.compute_calibration(F1_MERGE_VERIFY, [])
        assert cal['retry_histogram'] == {1: 2, 2: 1}

    def test_landed_and_never_counts(self):
        cal = mod.compute_calibration(F1_MERGE_VERIFY, [])
        assert cal['landed'] == 3
        assert cal['never'] == 0


class TestComputeCalibrationF2:
    """F2 = F1 + never-landed D: per-attempt and p_flake(landed) diverge."""

    def test_per_attempt_pass_rate_drops(self):
        cal = mod.compute_calibration(F2_MERGE_VERIFY, [])
        assert cal['per_attempt_pass_rate'] == 0.5

    def test_p_flake_landed_unchanged_from_f1(self):
        cal = mod.compute_calibration(F2_MERGE_VERIFY, [])
        assert cal['p_flake_landed'] == 0.75

    def test_diverges_from_per_attempt(self):
        cal = mod.compute_calibration(F2_MERGE_VERIFY, [])
        assert cal['p_flake_landed'] != cal['per_attempt_pass_rate']

    def test_land_rate_is_p_good_upper_bound(self):
        cal = mod.compute_calibration(F2_MERGE_VERIFY, [])
        assert cal['land_rate'] == 0.75

    def test_never_count(self):
        cal = mod.compute_calibration(F2_MERGE_VERIFY, [])
        assert cal['never'] == 1

    def test_terminal_outcome_tally(self):
        cal = mod.compute_calibration(F2_MERGE_VERIFY, F2_MERGE_ATTEMPT)
        assert cal['terminal_outcomes'] == {'done': 3, 'conflict': 1}

    def test_genuine_conflict_rate(self):
        cal = mod.compute_calibration(F2_MERGE_VERIFY, F2_MERGE_ATTEMPT)
        assert cal['genuine_conflict_rate'] == 0.25

    def test_p_good_bracket(self):
        cal = mod.compute_calibration(F2_MERGE_VERIFY, F2_MERGE_ATTEMPT)
        assert cal['p_good_bracket'] == (0.75, 0.75)
