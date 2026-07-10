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

import json
import sqlite3
from datetime import UTC, datetime, timedelta

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

# ---------------------------------------------------------------------------
# Fixture F3: real depth signal — depth0/depth1/depth2 buckets.
# ---------------------------------------------------------------------------

F3_MERGE_VERIFY = [
    _mv('d0-1', True, depth=0),
    _mv('d0-2', True, depth=0),
    _mv('d1-1', True, depth=1),
    _mv('d1-2', False, depth=1),
    _mv('d2-1', False, depth=2),
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


# ---------------------------------------------------------------------------
# compute_per_depth — buckets merge_verify events by a None-safe int()
# coercion of data['depth']; absent/None/unparseable depths are skipped.
# ---------------------------------------------------------------------------

class TestComputePerDepth:
    def test_pass_total_per_depth(self):
        per_depth = mod.compute_per_depth(F3_MERGE_VERIFY)
        counts = {d: (v['pass'], v['total']) for d, v in per_depth.items()}
        assert counts == {0: (2, 2), 1: (1, 2), 2: (0, 1)}

    def test_rate_per_depth(self):
        per_depth = mod.compute_per_depth(F3_MERGE_VERIFY)
        rates = {d: v['rate'] for d, v in per_depth.items()}
        assert rates == {0: 1.0, 1: 0.5, 2: 0.0}

    def test_string_depth_coerced_to_int(self):
        # speculative_merge's _emit_speculative str-converts every value,
        # so a real event can carry depth as the string "1".
        events = [_mv('s', True, depth='1')]
        per_depth = mod.compute_per_depth(events)
        assert set(per_depth) == {1}
        assert per_depth[1]['pass'] == 1
        assert per_depth[1]['total'] == 1

    def test_none_depth_excluded(self):
        events = [_mv('n', True, depth=None), _mv('z', True, depth=0)]
        per_depth = mod.compute_per_depth(events)
        assert set(per_depth) == {0}

    def test_missing_depth_key_excluded(self):
        # historical (pre-2340) events carry no 'depth' key at all.
        events = [{'task_id': 'm', 'data': {'passed': True}}]
        per_depth = mod.compute_per_depth(events)
        assert not per_depth

    def test_unparseable_depth_excluded(self):
        events = [_mv('u', True, depth='not-a-number')]
        per_depth = mod.compute_per_depth(events)
        assert not per_depth

    def test_no_depth_anywhere_yields_empty_result(self):
        # F2: every event has an explicit depth=None — no real signal.
        # This is the fallback trigger for format_report's CONFOUNDED path.
        per_depth = mod.compute_per_depth(F2_MERGE_VERIFY)
        assert not per_depth


# ---------------------------------------------------------------------------
# format_report — NO-depth calibration falls back to a runner-split section
# explicitly labelled CONFOUNDED, with a warning (no real depth signal).
# ---------------------------------------------------------------------------

class TestFormatReportNoDepth:
    """format_report(calibration, per_depth=None) on F2 (no event carries a
    depth field, so per_depth is None/empty — the fallback trigger)."""

    @classmethod
    def setup_class(cls):
        cls.cal = mod.compute_calibration(F2_MERGE_VERIFY, F2_MERGE_ATTEMPT)
        cls.report = mod.format_report(cls.cal, None)

    def test_contains_per_attempt_rate(self):
        assert '0.5' in self.report  # per_attempt_pass_rate == 0.5

    def test_contains_both_p_flake_numbers(self):
        # method 1 (per-attempt == 0.5) and method 2 (attempts-to-first-pass
        # among landed == 0.75) must BOTH be surfaced so an operator can
        # confirm the two independent computations agree (or flag a
        # divergence, as here — F2 has a never-landed task).
        assert '0.5' in self.report
        assert '0.75' in self.report

    def test_contains_p_good_bracket(self):
        assert '0.75' in self.report  # bracket == (0.75, 0.75)

    def test_contains_retry_histogram(self):
        assert '1x:2' in self.report
        assert '2x:1' in self.report

    def test_contains_terminal_tally(self):
        assert 'done=3' in self.report
        assert 'conflict=1' in self.report

    def test_contains_confounded_runner_split_with_warning(self):
        assert 'CONFOUNDED' in self.report
        assert 'runner' in self.report.lower()
        assert 'WARNING' in self.report


# ---------------------------------------------------------------------------
# main(argv) — end-to-end CLI smoke test against a tmp sqlite runs.db
# ---------------------------------------------------------------------------

_EVENTS_SCHEMA = """
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL
);
"""


def _seed_events_db(path, *, merge_verify_events, merge_attempt_events):
    """Write a tiny events table (mirrors event_store.py's real schema).

    Rows are timestamped within the last day so a --since window of 30 (or
    even 1) days includes them.
    """
    conn = sqlite3.connect(path)
    conn.executescript(_EVENTS_SCHEMA)
    now = datetime.now(UTC)
    rows = []
    for i, ev in enumerate(merge_verify_events):
        ts = (now - timedelta(minutes=i)).isoformat()
        rows.append(
            (ts, 'run-1', ev['task_id'], 'merge_verify', None, None,
             json.dumps(ev['data']), None)
        )
    for i, ev in enumerate(merge_attempt_events):
        ts = (now - timedelta(minutes=i)).isoformat()
        rows.append(
            (ts, 'run-1', ev['task_id'], 'merge_attempt', None, None,
             json.dumps(ev['data']), None)
        )
    conn.executemany(
        'INSERT INTO events '
        '(timestamp, run_id, task_id, event_type, phase, role, data, cost_usd) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        rows,
    )
    conn.commit()
    conn.close()


class TestMainCLI:
    """main([db_path, '--since', '30']) end-to-end against a tmp sqlite DB."""

    def test_main_smoke_reports_per_attempt_rate(self, tmp_path, capsys):
        db_path = str(tmp_path / 'runs.db')
        _seed_events_db(
            db_path,
            merge_verify_events=F2_MERGE_VERIFY,
            merge_attempt_events=F2_MERGE_ATTEMPT,
        )

        ret = mod.main([db_path, '--since', '30'])

        captured = capsys.readouterr()
        assert ret == 0
        assert '0.5' in captured.out  # F2 per-attempt pass rate == 3/6 == 0.5
