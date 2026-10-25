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

import sqlite3
from datetime import UTC, datetime, timedelta

import analyze_speculation_depth as mod
import pytest


def _mv(task_id, passed, *, attempt=0, depth=None, chain_items=None):
    """Build a synthetic merge_verify event dict; ``chain_items`` is written only when given."""
    data = {'passed': passed, 'attempt': attempt, 'depth': depth}
    if chain_items is not None:
        data['chain_items'] = chain_items
    return {'task_id': task_id, 'data': data}


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
# Fixture F3: chain_items buckets 1/2/4, with legacy depth labels riding along.
# ---------------------------------------------------------------------------

F3_MERGE_VERIFY = [
    _mv('c1-1', True, depth=2, chain_items=1),
    _mv('c1-2', True, depth=0, chain_items=1),
    _mv('c2-1', True, depth=1, chain_items=2),
    _mv('c2-2', False, depth=1, chain_items=2),
    _mv('c4-1', False, depth=3, chain_items=4),
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
# compute_per_chain_items / is_legacy_deep_label — the analyzer keys on
# chain_items, never on the legacy depth label.
# ---------------------------------------------------------------------------

class TestComputePerChainItems:
    def test_pass_total_per_chain_items(self):
        per_chain_items = mod.compute_per_chain_items(F3_MERGE_VERIFY)
        # None is format_report's CONFOUNDED trigger; F3 always contributes.
        assert per_chain_items is not None
        counts = {k: (v['pass'], v['total']) for k, v in per_chain_items.items()}
        assert counts == {1: (2, 2), 2: (1, 2), 4: (0, 1)}, f'{counts=}'

    def test_rate_per_chain_items(self):
        per_chain_items = mod.compute_per_chain_items(F3_MERGE_VERIFY)
        assert per_chain_items is not None
        rates = {k: v['rate'] for k, v in per_chain_items.items()}
        assert rates == {1: 1.0, 2: 0.5, 4: 0.0}, f'{rates=}'

    @pytest.mark.parametrize(
        'untrusted', [None, '2', True, 0, -1, 2.0],
        ids=['none', 'str', 'bool', 'zero', 'negative', 'float'],
    )
    def test_untrusted_chain_items_is_skipped_not_coerced(self, untrusted):
        events = [{'task_id': 'u', 'data': {'passed': True, 'chain_items': untrusted}}]
        assert mod.compute_per_chain_items(events) is None, f'{untrusted=!r}'

    def test_depth_label_without_chain_items_contributes_no_bucket(self):
        events = [_mv('legacy', True, depth=3), _mv('live', True, depth=3, chain_items=1)]
        per_chain_items = mod.compute_per_chain_items(events)
        assert per_chain_items is not None
        assert set(per_chain_items) == {1}, f'{per_chain_items=}'

    def test_no_chain_items_anywhere_yields_none(self):
        assert mod.compute_per_chain_items(F2_MERGE_VERIFY) is None

    def test_the_depth_curve_has_no_reader_left(self):
        assert not hasattr(mod, 'compute_per_depth')


class TestIsLegacyDeepLabel:
    @pytest.mark.parametrize('depth', [3, '2'], ids=['int', 'str'])
    def test_pre_chain_items_deep_label(self, depth):
        assert mod.is_legacy_deep_label({'depth': depth}), f'{depth=!r}'

    @pytest.mark.parametrize(
        'data',
        [{'depth': 1}, {'depth': 0}, {'depth': None}, {}, {'depth': 'abc'},
         {'depth': 3, 'chain_items': 1}],
        ids=['one', 'zero', 'none', 'absent', 'unparseable', 'post-chain_items'],
    )
    def test_not_a_legacy_deep_label(self, data):
        assert not mod.is_legacy_deep_label(data), f'{data=}'


# ---------------------------------------------------------------------------
# format_report — the chain_items curve when there is one, else a runner-split
# section explicitly labelled CONFOUNDED with a warning.
# ---------------------------------------------------------------------------

class TestFormatReportNoChainItems:
    """format_report(calibration, None) on F2 (no event carries chain_items)."""

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

    def test_fallback_names_the_missing_chain_items_signal(self):
        assert 'carries chain_items >= 1' in self.report, self.report
        assert 'task 2340' not in self.report, self.report

    def test_no_legacy_line_when_there_are_no_legacy_rows(self):
        assert 'legacy depth>=2 rows' not in self.report, self.report


class TestFormatReportChainItems:
    @classmethod
    def setup_class(cls):
        cal = mod.compute_calibration(F3_MERGE_VERIFY, [])
        cls.report = mod.format_report(cal, mod.compute_per_chain_items(F3_MERGE_VERIFY), 2)

    def test_block_header_names_the_truthful_field(self):
        assert '=== per-chain_items P(pass|chain_items) ===' in self.report, self.report

    def test_one_line_per_bucket(self):
        lines = self.report.splitlines()
        assert '  chain_items=1: 2/2 = 1.000' in lines, self.report
        assert '  chain_items=2: 1/2 = 0.500' in lines, self.report
        assert '  chain_items=4: 0/1 = 0.000' in lines, self.report
        assert not any(line.lstrip().startswith('depth=') for line in lines), self.report

    def test_legacy_rows_line(self):
        assert 'legacy depth>=2 rows (pre-chain_items, not in the curve): 2' in self.report, (
            self.report
        )


# ---------------------------------------------------------------------------
# main(argv) — end-to-end CLI smoke test against a tmp sqlite runs.db
# ---------------------------------------------------------------------------

def _recent_events(*, merge_verify_events, merge_attempt_events) -> list[dict]:
    """Turn synthetic event dicts into ``make_runs_db`` events rows.

    Rows are timestamped within the last day so a --since window of 30 (or
    even 1) days includes them.
    """
    now = datetime.now(UTC)
    events = []
    for event_type, source in (
        ('merge_verify', merge_verify_events),
        ('merge_attempt', merge_attempt_events),
    ):
        for i, ev in enumerate(source):
            events.append({
                'event_type': event_type,
                'task_id': ev['task_id'],
                'timestamp': (now - timedelta(minutes=i)).isoformat(),
                'data': ev['data'],
            })
    return events


class TestMainCLI:
    """main([db_path, '--since', '30']) end-to-end against a tmp sqlite DB."""

    def test_main_smoke_reports_per_attempt_rate(self, make_runs_db, capsys):
        db_path = make_runs_db(_recent_events(
            merge_verify_events=F2_MERGE_VERIFY,
            merge_attempt_events=F2_MERGE_ATTEMPT,
        ))

        ret = mod.main([str(db_path), '--since', '30'])

        captured = capsys.readouterr()
        assert ret == 0
        assert '0.5' in captured.out  # F2 per-attempt pass rate == 3/6 == 0.5

    def test_main_opens_db_read_only(self, make_runs_db, monkeypatch):
        """main's handle is opened with a mode=ro URI, so a write on it raises."""
        db_path = make_runs_db(_recent_events(
            merge_verify_events=F1_MERGE_VERIFY,
            merge_attempt_events=[],
        ))
        monkeypatch.setattr(mod, 'load_events', _load_events_attempting_write)

        with pytest.raises(sqlite3.OperationalError, match='readonly'):
            mod.main([str(db_path)])

    def test_main_reports_chain_items_and_counts_legacy_rows(self, make_runs_db, capsys):
        db_path = make_runs_db(_recent_events(
            merge_verify_events=[
                _mv('legacy', True, depth=3),
                _mv('live', True, depth=3, chain_items=4),
            ],
            merge_attempt_events=[],
        ))

        assert mod.main([str(db_path), '--since', '30']) == 0

        out = capsys.readouterr().out
        assert '  chain_items=4: 1/1 = 1.000' in out, out
        assert 'legacy depth>=2 rows (pre-chain_items, not in the curve): 1' in out, out
        assert 'depth=3' not in out, out


def _load_events_attempting_write(conn, event_type, since_days):
    """A load_events stand-in that attempts a write on the handle main passes it."""
    conn.execute(
        "INSERT INTO events (timestamp, run_id, event_type) VALUES ('t', 'r', 'x')"
    )
