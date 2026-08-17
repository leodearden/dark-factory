"""Tests for the flake-ledger operator report (plans/flake-ledger-prd.md, task ι).

DB-backed fixtures seed ``flake_occurrence`` / ``flake_debt`` with
``flake_ledger.ensure_schema`` plus raw ``sqlite3`` INSERTs, never through
``record_flake_occurrence`` / ``open_debt`` / ``resolve_debt``.  That follows
``test_flake_ledger.py``'s stated convention of bypassing the module under test for
on-disk truth, and it additionally keeps ι's suite decoupled from the two write paths
tasks ζ and η are concurrently rewriting.

STRUCTURAL CONSTRAINT — sync and async tests live in strictly separate classes.
``orchestrator/pyproject.toml`` sets no ``asyncio_mode`` (pytest-asyncio STRICT) and
promotes the mark-mismatch warning to an ERROR-level filterwarning.  ι's report path is
ENTIRELY synchronous, so every test here is a plain sync ``def test_`` and no class in
this file carries ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from orchestrator.cli import main
from orchestrator.flake_ledger import (
    UNKNOWN_TEST_ID,
    DebtRow,
    FlakeCallSite,
    FlakeOccurrenceRow,
    FlakeVerdict,
    ensure_schema,
)
from orchestrator.flake_report import (
    DebtReportRow,
    FlakeLedgerReport,
    GateBlindCounter,
    NonConvergenceCounter,
    SystemicCounter,
    _parse_stamp,
    build_chains,
    build_report,
    compute_gate_blind,
    compute_non_convergence,
    compute_systemic,
    format_age,
    render_report,
)

_NOW = datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)


def _seed_occurrence(
    db_path: Path,
    *,
    test_id: str,
    observed_at: str,
    verdict: str = FlakeVerdict.passes_in_isolation,
    call_site: str = FlakeCallSite.merge_gate,
    psi: float | None = None,
) -> None:
    """Insert one flake_occurrence row with raw sqlite3.

    Deliberately NOT via record_flake_occurrence: test_flake_ledger.py's header
    establishes the raw-sqlite convention for on-disk truth, and here it additionally
    keeps ι's suite off the write paths tasks ζ and η are concurrently rewriting.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            'INSERT INTO flake_occurrence '
            '(observed_at, test_id, project_id, verdict, call_site, runner, '
            ' merge_sha, task_id, psi_cpu_some10, detail) '
            "VALUES (?, ?, 'dark_factory', ?, ?, 'local', NULL, NULL, ?, '{}')",
            (observed_at, test_id, str(verdict), str(call_site), psi),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_debt(
    db_path: Path,
    *,
    test_id: str,
    opened_at: str,
    resolved_at: str | None = None,
    owner_task_id: str | None = '3552',
    open_count: int = 1,
    prior_resolved_at: str | None = None,
    prior_resolving_commit: str | None = None,
    last_occurrence_at: str | None = None,
) -> None:
    """Insert one flake_debt row with raw sqlite3 (never through open_debt/resolve_debt)."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            'INSERT INTO flake_debt '
            '(test_id, project_id, opened_at, resolved_at, owner_task_id, open_count, '
            ' prior_resolved_at, prior_resolving_commit, last_occurrence_at) '
            "VALUES (?, 'dark_factory', ?, ?, ?, ?, ?, ?, ?)",
            (
                test_id,
                opened_at,
                resolved_at,
                owner_task_id,
                open_count,
                prior_resolved_at,
                prior_resolving_commit,
                last_occurrence_at or opened_at,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _occ(
    *,
    test_id: str = 'tests/test_a.py::test_one',
    verdict: str = FlakeVerdict.passes_in_isolation,
    observed_at: str = '2026-08-08T12:00:00+00:00',
    call_site: str = FlakeCallSite.merge_gate,
    psi: float | None = None,
    row_id: int = 1,
) -> FlakeOccurrenceRow:
    """A hand-built occurrence row — the pure counter tests need no DB at all."""
    return FlakeOccurrenceRow(
        id=row_id,
        observed_at=observed_at,
        test_id=test_id,
        project_id='dark_factory',
        verdict=str(verdict),
        call_site=str(call_site),
        runner='local',
        merge_sha=None,
        task_id=None,
        psi_cpu_some10=psi,
        detail='{}',
    )


def _debt(
    *,
    test_id: str = 'tests/test_a.py::test_one',
    opened_at: str = '2026-08-09T12:00:00+00:00',
    resolved_at: str | None = None,
    owner_task_id: str | None = '3552',
    open_count: int = 1,
    prior_resolved_at: str | None = None,
    prior_resolving_commit: str | None = None,
    last_occurrence_at: str = '2026-08-09T12:00:00+00:00',
) -> DebtRow:
    """A hand-built debt row — the pure counter tests need no DB at all."""
    return DebtRow(
        test_id=test_id,
        project_id='dark_factory',
        opened_at=opened_at,
        resolved_at=resolved_at,
        owner_task_id=owner_task_id,
        open_count=open_count,
        prior_resolved_at=prior_resolved_at,
        prior_resolving_commit=prior_resolving_commit,
        last_occurrence_at=last_occurrence_at,
    )


class TestStampAndAge:
    """The two rendering primitives every other section leans on."""

    def test_canonical_utc_stamp_parses_to_aware_utc(self):
        parsed = _parse_stamp('2026-08-08T12:00:00+00:00')
        assert parsed is not None
        assert parsed.tzinfo is not None, 'a parsed stamp must be timezone-AWARE'
        assert parsed.utcoffset() == timedelta(0)
        assert parsed == datetime(2026, 8, 8, 12, 0, 0, tzinfo=UTC)

    def test_naive_stamp_is_assumed_utc(self):
        # Mirrors flake_ledger._canonicalize_utc: a naive value gets UTC ATTACHED,
        # never .astimezone(), which would apply the HOST's local offset and silently
        # shift the stamp by the dispatcher's timezone.
        parsed = _parse_stamp('2026-08-08T12:00:00')
        assert parsed is not None
        assert parsed.tzinfo is not None
        assert parsed == datetime(2026, 8, 8, 12, 0, 0, tzinfo=UTC)

    def test_z_and_offset_spellings_are_the_same_instant(self):
        # flake_ledger stores canonicalised '+00:00', but a hand-built stamp may
        # arrive as '…Z'.  Both spellings must name one instant, or an age computed
        # from one and a window bound from the other disagree.
        assert _parse_stamp('2026-08-08T12:00:00Z') == _parse_stamp('2026-08-08T12:00:00+00:00')

    def test_malformed_stamp_returns_none_and_does_not_raise(self):
        # One bad row must not take down the whole report: a read path an operator
        # cannot rely on is worse than a row rendered 'unknown'.
        assert _parse_stamp('not-a-date') is None

    def test_none_stamp_returns_none(self):
        assert _parse_stamp(None) is None

    def test_format_age_renders_days_and_hours(self):
        assert format_age(timedelta(days=3, hours=4)) == '3d 4h'

    def test_format_age_renders_zero_days_explicitly(self):
        assert format_age(timedelta(hours=5)) == '0d 5h'

    def test_format_age_of_none_says_unknown_not_zero(self):
        # The dangerous direction: an unparseable opened_at rendered as '0d 0h' makes a
        # stale debt row look brand new and suppresses it from the age backstop.
        rendered = format_age(None)
        assert 'unknown' in rendered, rendered
        assert rendered != '0d 0h'


class TestGateBlindCounter:
    """§5.6 class 1 — the unconfirmable rate, i.e. "the gate has gone blind"."""

    def test_rate_over_a_mixed_window(self):
        rows = [
            _occ(verdict=FlakeVerdict.unconfirmable, test_id=UNKNOWN_TEST_ID, row_id=i)
            for i in range(3)
        ] + [
            _occ(
                verdict=(
                    FlakeVerdict.passes_in_isolation if i % 2 else FlakeVerdict.fails_in_isolation
                ),
                test_id=f'tests/test_a.py::test_{i}',
                row_id=10 + i,
            )
            for i in range(9)
        ]
        counter = compute_gate_blind(rows)
        assert counter.unconfirmable == 3
        assert counter.confirmed == 9
        assert counter.total == 12
        assert counter.rate == pytest.approx(0.25)

    def test_empty_window_has_a_none_rate_never_zero(self):
        # α's read_occurrences docstring names the hazard directly: a zero-row answer
        # "reads as 'healthy' rather than as a broken query".  0.00 for an empty window
        # is precisely the silent degradation this PRD exists to end.
        counter = compute_gate_blind([])
        assert counter.rate is None
        assert counter.total == 0
        assert counter.exceeds_threshold is False

    def test_below_the_observation_floor_is_insufficient(self):
        two_rows = [
            _occ(verdict=FlakeVerdict.unconfirmable, row_id=1),
            _occ(verdict=FlakeVerdict.passes_in_isolation, row_id=2),
        ]
        assert compute_gate_blind(two_rows).sufficient is False

    def test_at_or_above_the_observation_floor_is_sufficient(self):
        twelve_rows = [_occ(row_id=i) for i in range(12)]
        assert compute_gate_blind(twelve_rows).sufficient is True

    def test_sentinel_test_id_rows_still_count(self):
        # The sentinel IS the class-1 signal (§5.6: "6 unconfirmable lines sat at INFO
        # for a month").  Dropping these rows because they name no test would reproduce
        # exactly the blindness this counter exists to measure.
        rows = [
            _occ(verdict=FlakeVerdict.unconfirmable, test_id=UNKNOWN_TEST_ID, row_id=i)
            for i in range(4)
        ]
        counter = compute_gate_blind(rows)
        assert counter.unconfirmable == 4
        assert counter.total == 4

    def test_exceeds_threshold_requires_sufficiency(self):
        # 1-of-2 unconfirmable is rate 0.5 — well over the 0.25 threshold — but the §10
        # min_observations floor exists precisely so the class cannot fire on it.
        rows = [
            _occ(verdict=FlakeVerdict.unconfirmable, row_id=1),
            _occ(verdict=FlakeVerdict.passes_in_isolation, row_id=2),
        ]
        counter = compute_gate_blind(rows)
        assert counter.rate == pytest.approx(0.5)
        assert counter.sufficient is False
        assert counter.exceeds_threshold is False

    def test_exceeds_threshold_when_sufficient_and_over_rate(self):
        rows = [
            _occ(verdict=FlakeVerdict.unconfirmable, test_id=UNKNOWN_TEST_ID, row_id=i)
            for i in range(6)
        ] + [_occ(verdict=FlakeVerdict.fails_in_isolation, row_id=10 + i) for i in range(6)]
        counter = compute_gate_blind(rows)
        assert counter.sufficient is True
        assert counter.rate == pytest.approx(0.5)
        assert counter.exceeds_threshold is True


class TestNonConvergenceCounter:
    """§5.6 class 2 — a de-flake cycle that is not converging."""

    def test_recurrence_is_open_count_over_one(self):
        rows = [
            _debt(test_id='a', open_count=3),
            _debt(test_id='b', open_count=1),
        ]
        counter = compute_non_convergence(rows, _NOW)
        assert counter.open_tests == 2
        assert counter.recurrent_tests == 1

    def test_over_age_uses_the_injected_now(self):
        rows = [
            _debt(test_id='old', opened_at='2026-08-06T12:00:00+00:00'),  # 4d before now
            _debt(test_id='new', opened_at='2026-08-08T12:00:00+00:00'),  # 2d before now
        ]
        assert compute_non_convergence(rows, _NOW).over_age_tests == 1

    def test_exactly_at_the_threshold_is_not_over_age(self):
        # Strictly greater: a row opened exactly age_days ago has not yet EXCEEDED the
        # bound, so pinning the boundary keeps the backstop from firing a day early.
        rows = [_debt(test_id='boundary', opened_at='2026-08-07T12:00:00+00:00')]  # exactly 3d
        assert compute_non_convergence(rows, _NOW).over_age_tests == 0

    def test_missing_owner_counts_as_an_invariant_breach(self):
        rows = [
            _debt(test_id='unowned', owner_task_id=None),
            _debt(test_id='owned', owner_task_id='3552'),
        ]
        assert compute_non_convergence(rows, _NOW).unowned_tests == 1

    def test_malformed_opened_at_is_counted_and_never_treated_as_age_zero(self):
        rows = [_debt(test_id='bad', opened_at='not-a-date')]
        counter = compute_non_convergence(rows, _NOW)
        assert counter.unparseable_opened_at == 1
        # Never silently age-zero: that would hide a stale row from the backstop.
        assert counter.over_age_tests == 0

    def test_oldest_age_reflects_the_oldest_parseable_row(self):
        rows = [
            _debt(test_id='bad', opened_at='not-a-date'),
            _debt(test_id='old', opened_at='2026-08-06T12:00:00+00:00'),
            _debt(test_id='new', opened_at='2026-08-09T12:00:00+00:00'),
        ]
        assert compute_non_convergence(rows, _NOW).oldest_age == timedelta(days=4)

    def test_oldest_age_is_none_when_nothing_parses(self):
        rows = [_debt(test_id='bad', opened_at='not-a-date')]
        assert compute_non_convergence(rows, _NOW).oldest_age is None


class TestSystemicCounter:
    """§5.6 class 3 — systemic host pressure, not N independent flaky tests."""

    def test_four_distinct_tests_in_one_window_exceeds(self):
        rows = [
            _occ(test_id=f'tests/test_{i}.py::test_x', observed_at=f'2026-08-08T12:{i:02d}:00+00:00',
                 row_id=i)
            for i in range(4)
        ]
        counter = compute_systemic(rows)
        assert counter.peak_distinct_tests == 4
        assert counter.exceeds_threshold is True

    def test_one_test_suppressed_six_times_is_not_systemic(self):
        # §5.6's own discriminator, pinned: one test suppressing repeatedly is class 2
        # (non-convergence), six DIFFERENT tests suppressing at once is class 3.
        rows = [
            _occ(test_id='tests/test_a.py::test_one',
                 observed_at=f'2026-08-08T12:{i * 5:02d}:00+00:00', row_id=i)
            for i in range(6)
        ]
        counter = compute_systemic(rows)
        assert counter.peak_distinct_tests == 1
        assert counter.exceeds_threshold is False

    def test_the_window_actually_bounds_the_count(self):
        rows = [
            _occ(test_id=f'tests/test_{i}.py::test_x',
                 observed_at=f'2026-08-08T{12 + i * 2:02d}:00:00+00:00', row_id=i)
            for i in range(4)
        ]
        counter = compute_systemic(rows)
        assert counter.peak_distinct_tests < 4, 'a 6-hour spread must not read as one window'
        assert counter.exceeds_threshold is False

    def test_only_passes_in_isolation_rows_count(self):
        # fails_in_isolation is a real red, not a suppression — four of them are four
        # genuine failures, and counting them would manufacture a systemic-pressure
        # signal out of ordinary breakage.
        rows = [
            _occ(test_id=f'tests/test_{i}.py::test_x', verdict=FlakeVerdict.fails_in_isolation,
                 observed_at=f'2026-08-08T12:{i:02d}:00+00:00', row_id=i)
            for i in range(4)
        ]
        counter = compute_systemic(rows)
        assert counter.peak_distinct_tests == 0
        assert counter.exceeds_threshold is False

    def test_peak_window_psi_carries_the_max_seen(self):
        rows = [
            _occ(test_id=f'tests/test_{i}.py::test_x', observed_at=f'2026-08-08T12:{i:02d}:00+00:00',
                 psi=float(i) * 10.0, row_id=i)
            for i in range(4)
        ]
        assert compute_systemic(rows).peak_window_psi == pytest.approx(30.0)

    def test_peak_window_psi_is_none_when_every_row_lacks_it(self):
        # A missing PSI read must never render as 0.0 pressure — that would read as
        # "the host was idle", the opposite of "we could not measure the host".
        rows = [
            _occ(test_id=f'tests/test_{i}.py::test_x', observed_at=f'2026-08-08T12:{i:02d}:00+00:00',
                 psi=None, row_id=i)
            for i in range(4)
        ]
        assert compute_systemic(rows).peak_window_psi is None

    def test_empty_input_is_zero_and_below_threshold(self):
        counter = compute_systemic([])
        assert counter.peak_distinct_tests == 0
        assert counter.peak_window_start is None
        assert counter.exceeds_threshold is False


class TestChains:
    """Per-test recurrence chains — the union of open debt and windowed occurrences."""

    @staticmethod
    def _lookup(rows: dict[str, DebtRow]):
        """An injected debt_lookup, so the chain unit tests need no DB."""
        return lambda test_id: rows.get(test_id)

    def test_a_test_with_occurrences_but_no_debt_row_still_appears(self):
        # This is what makes task κ's signal observable: a CHRONIC-FLAKY marker produces
        # an occurrence BEFORE any debt exists, and κ's acceptance is that it appears in
        # THE REPORT (and files no second de-flake task).
        occurrences = [_occ(test_id='tests/test_new.py::test_x')]
        chains = build_chains(occurrences, [], self._lookup({}))
        assert [c.test_id for c in chains] == ['tests/test_new.py::test_x']
        assert chains[0].debt is None
        assert chains[0].occurrence_count == 1

    def test_call_site_counts_separate_chronic_marker_from_the_gates(self):
        occurrences = [
            _occ(test_id='t', call_site=FlakeCallSite.chronic_marker, row_id=1,
                 observed_at='2026-08-08T12:00:00+00:00'),
            _occ(test_id='t', call_site=FlakeCallSite.merge_gate, row_id=2,
                 observed_at='2026-08-08T12:01:00+00:00'),
            _occ(test_id='t', call_site=FlakeCallSite.merge_gate, row_id=3,
                 observed_at='2026-08-08T12:02:00+00:00'),
            _occ(test_id='t', call_site=FlakeCallSite.main_probe, row_id=4,
                 observed_at='2026-08-08T12:03:00+00:00'),
        ]
        chains = build_chains(occurrences, [], self._lookup({}))
        assert chains[0].call_site_counts == {
            'chronic_marker': 1,
            'merge_gate': 2,
            'main_probe': 1,
        }
        assert chains[0].occurrence_count == 4

    def test_sentinel_occurrences_produce_no_chain(self):
        # A sentinel names no test, so it can own no chain — open_debt itself REFUSES it
        # for the same reason.  It still counts toward the gate-blind rate (TestGateBlind).
        occurrences = [_occ(test_id=UNKNOWN_TEST_ID, verdict=FlakeVerdict.unconfirmable)]
        assert build_chains(occurrences, [], self._lookup({})) == []

    def test_open_debt_with_no_windowed_occurrences_still_appears(self):
        debt = _debt(test_id='quiet', open_count=2, prior_resolved_at='2026-07-01T00:00:00+00:00',
                     prior_resolving_commit='abc1234')
        chains = build_chains([], [debt], self._lookup({'quiet': debt}))
        assert [c.test_id for c in chains] == ['quiet']
        assert chains[0].occurrence_count == 0
        assert chains[0].debt is not None
        assert chains[0].debt.open_count == 2
        assert chains[0].debt.prior_resolved_at == '2026-07-01T00:00:00+00:00'
        assert chains[0].debt.prior_resolving_commit == 'abc1234'

    def test_a_currently_resolved_test_with_occurrences_still_shows_its_chain(self):
        # The motivating case: test_spawn_claude.py, 7 de-flake tasks in 7 weeks, must
        # read as ONE chain even between cycles.  list_open_debt filters on
        # resolved_at IS NULL, so a chain built from it alone goes blank exactly when
        # each fix appears to have worked.
        resolved = _debt(test_id='chronic', resolved_at='2026-08-01T00:00:00+00:00',
                         open_count=7, prior_resolved_at='2026-07-25T00:00:00+00:00',
                         prior_resolving_commit='deadbee')
        chains = build_chains([_occ(test_id='chronic')], [], self._lookup({'chronic': resolved}))
        assert [c.test_id for c in chains] == ['chronic']
        assert chains[0].debt is not None
        assert chains[0].debt.open_count == 7
        assert chains[0].debt.prior_resolving_commit == 'deadbee'

    def test_ordering_is_deterministic_recurrence_first_then_test_id(self):
        occurrences = [
            _occ(test_id='zzz', row_id=1),
            _occ(test_id='aaa', row_id=2),
            _occ(test_id='mmm', row_id=3),
        ]
        lookup = self._lookup({'mmm': _debt(test_id='mmm', open_count=5)})
        chains = build_chains(occurrences, [], lookup)
        assert [c.test_id for c in chains] == ['mmm', 'aaa', 'zzz']
        # Byte-stability of the rendered report depends on this being repeatable.
        assert [c.test_id for c in build_chains(occurrences, [], lookup)] == ['mmm', 'aaa', 'zzz']

    def test_last_observed_at_is_the_most_recent_occurrence(self):
        occurrences = [
            _occ(test_id='t', observed_at='2026-08-08T12:00:00+00:00', row_id=1),
            _occ(test_id='t', observed_at='2026-08-09T09:00:00+00:00', row_id=2),
        ]
        chains = build_chains(occurrences, [], self._lookup({}))
        assert chains[0].last_observed_at == '2026-08-09T09:00:00+00:00'


class TestBuildReport:
    """build_report against a real on-disk ledger, seeded with raw sqlite3."""

    def _seeded(self, tmp_path: Path) -> Path:
        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)
        # Two open debt rows, deliberately out of insertion order relative to opened_at
        # so the pinned `opened_at, test_id` print order is actually exercised.
        _seed_debt(db_path, test_id='tests/test_b.py::test_two',
                   opened_at='2026-08-09T12:00:00+00:00', owner_task_id='4001')
        _seed_debt(db_path, test_id='tests/test_a.py::test_one',
                   opened_at='2026-08-06T12:00:00+00:00', owner_task_id='4000', open_count=3,
                   prior_resolved_at='2026-07-20T00:00:00+00:00',
                   prior_resolving_commit='cafe123')
        for i in range(6):
            _seed_occurrence(db_path, test_id='tests/test_a.py::test_one',
                             observed_at=f'2026-08-09T1{i}:00:00+00:00')
        for i in range(3):
            _seed_occurrence(db_path, test_id=UNKNOWN_TEST_ID,
                             observed_at=f'2026-08-09T0{i}:00:00+00:00',
                             verdict=FlakeVerdict.unconfirmable)
        _seed_occurrence(db_path, test_id='tests/test_b.py::test_two',
                         observed_at='2026-08-09T20:00:00+00:00',
                         verdict=FlakeVerdict.fails_in_isolation)
        return db_path

    def test_report_sees_the_db(self, tmp_path):
        report = build_report(self._seeded(tmp_path), now=_NOW)
        assert report.db_present is True

    def test_open_debt_rows_carry_owner_and_age_in_pinned_order(self, tmp_path):
        report = build_report(self._seeded(tmp_path), now=_NOW)
        # α pins list_open_debt's `opened_at, test_id` ordering as contractual because
        # ι prints this list; relied on directly rather than re-sorted here.
        assert [r.debt.test_id for r in report.open_debt] == [
            'tests/test_a.py::test_one',
            'tests/test_b.py::test_two',
        ]
        assert [r.debt.owner_task_id for r in report.open_debt] == ['4000', '4001']
        assert report.open_debt[0].age == timedelta(days=4)
        assert report.open_debt[1].age == timedelta(days=1)

    def test_chains_cover_both_seeded_tests(self, tmp_path):
        report = build_report(self._seeded(tmp_path), now=_NOW)
        assert {c.test_id for c in report.chains} == {
            'tests/test_a.py::test_one',
            'tests/test_b.py::test_two',
        }
        by_id = {c.test_id: c for c in report.chains}
        assert by_id['tests/test_a.py::test_one'].occurrence_count == 6
        assert by_id['tests/test_a.py::test_one'].debt is not None
        assert by_id['tests/test_a.py::test_one'].debt.open_count == 3
        assert by_id['tests/test_a.py::test_one'].debt.prior_resolving_commit == 'cafe123'

    def test_all_three_counters_are_populated(self, tmp_path):
        report = build_report(self._seeded(tmp_path), now=_NOW)
        # 3 unconfirmable + 7 confirmed == 10 observations, rate 0.3, over the floor of 8.
        assert report.gate_blind.unconfirmable == 3
        assert report.gate_blind.confirmed == 7
        assert report.gate_blind.rate == pytest.approx(0.3)
        assert report.gate_blind.sufficient is True
        assert report.gate_blind.exceeds_threshold is True
        # Class 2: one recurrent (open_count 3), one over the 3d age bound.
        assert report.non_convergence.open_tests == 2
        assert report.non_convergence.recurrent_tests == 1
        assert report.non_convergence.over_age_tests == 1
        assert report.non_convergence.unowned_tests == 0
        # Class 3: one test suppressed six times is NOT systemic.
        assert report.systemic.peak_distinct_tests == 1
        assert report.systemic.exceeds_threshold is False

    def test_window_since_is_derived_from_now_and_window_hours(self, tmp_path):
        report = build_report(self._seeded(tmp_path), now=_NOW, window_hours=24)
        since = _parse_stamp(report.window_since)
        assert since == _NOW - timedelta(hours=24)

    def test_an_occurrence_older_than_the_window_is_excluded_from_the_counters(self, tmp_path):
        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)
        _seed_debt(db_path, test_id='tests/test_old.py::test_x',
                   opened_at='2026-01-01T00:00:00+00:00')
        # Far outside any window: 2025.
        _seed_occurrence(db_path, test_id='tests/test_old.py::test_x',
                         observed_at='2025-01-01T00:00:00+00:00')
        report = build_report(db_path, now=_NOW, window_hours=168)
        assert report.gate_blind.total == 0, 'a pre-window occurrence must not be counted'
        # ...but its test still resolves a debt row, so the chain remains visible.
        assert [c.test_id for c in report.chains] == ['tests/test_old.py::test_x']
        assert report.chains[0].debt is not None
        assert report.chains[0].occurrence_count == 0


class TestReadOnlyContract:
    """The report must never write — including never PROVISIONING a DB it finds absent."""

    def test_an_absent_ledger_is_not_provisioned(self, tmp_path):
        # α's readers all route through `_open`, which does
        # `parent.mkdir(parents=True, exist_ok=True)` -> connect (which CREATES the
        # file) -> `executescript(_SCHEMA)`.  So an unguarded "read-only" report would
        # provision data/orchestrator/runs.db plus WAL sidecars as a side effect of
        # PRINTING.  Neither the file nor its parent exists here.
        db_path = tmp_path / 'project' / 'data' / 'orchestrator' / 'runs.db'
        assert not db_path.parent.exists()

        build_report(db_path, now=_NOW)

        assert not db_path.exists(), 'the report must not create the ledger DB'
        assert not db_path.parent.exists(), 'the report must not create data/orchestrator/'
        assert not (tmp_path / 'project' / 'data' / 'orchestrator' / 'runs.db-wal').exists()
        assert not (tmp_path / 'project' / 'data' / 'orchestrator' / 'runs.db-shm').exists()

    def test_an_absent_ledger_reports_absence_honestly(self, tmp_path):
        db_path = tmp_path / 'project' / 'data' / 'orchestrator' / 'runs.db'
        report = build_report(db_path, now=_NOW)
        assert report.db_present is False
        assert report.open_debt == []
        assert report.chains == []
        # Absence must not render as a healthy 0.0 rate — that is exactly the silent
        # degradation the PRD exists to end.
        assert report.gate_blind.rate is None
        assert report.gate_blind.sufficient is False

    def test_the_report_module_binds_no_write_entry_point(self):
        # A direct structural check on the read-only contract, and the one that survives
        # `from x import y` binding: if a future edit imports a writer into this module,
        # this fails immediately rather than waiting for a call site to exercise it.
        import orchestrator.flake_report as report_module

        for writer in ('record_flake_occurrence', 'open_debt', 'resolve_debt'):
            assert not hasattr(report_module, writer), (
                f'flake_report must not bind the write entry point {writer}'
            )

    def test_build_report_succeeds_with_every_writer_booby_trapped(self, tmp_path, monkeypatch):
        # Plain SYNC raisers even though open_debt/resolve_debt are `async def`: the
        # assertion is that they are never AWAITED, so an AsyncMock would only be needed
        # if the code under test were expected to await them — precisely what this
        # forbids.  A sync raiser also fails louder if one is called and not awaited.
        def _boom(*args, **kwargs):
            raise AssertionError('the read-only report reached a ledger WRITE entry point')

        for writer in ('record_flake_occurrence', 'open_debt', 'resolve_debt'):
            monkeypatch.setattr(f'orchestrator.flake_ledger.{writer}', _boom)

        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)
        _seed_debt(db_path, test_id='tests/test_a.py::test_one',
                   opened_at='2026-08-09T12:00:00+00:00')
        _seed_occurrence(db_path, test_id='tests/test_a.py::test_one',
                         observed_at='2026-08-09T13:00:00+00:00')

        report = build_report(db_path, now=_NOW)
        assert report.db_present is True
        assert len(report.open_debt) == 1
        assert report.gate_blind.total == 1


class TestTruncationHonesty:
    """A limit-capped read must not silently yield a rate over an unknown window."""

    def _seed_n(self, tmp_path: Path, n: int) -> Path:
        db_path = tmp_path / 'runs.db'
        ensure_schema(db_path)
        for i in range(n):
            _seed_occurrence(db_path, test_id=f'tests/test_{i}.py::test_x',
                             observed_at=f'2026-08-09T12:00:{i:02d}+00:00')
        return db_path

    def test_a_filled_limit_marks_the_counter_truncated(self, tmp_path):
        db_path = self._seed_n(tmp_path, 10)
        report = build_report(db_path, now=_NOW, occurrence_limit=5)
        assert report.gate_blind.truncated is True

    def test_a_limit_above_the_row_count_is_not_truncated(self, tmp_path):
        db_path = self._seed_n(tmp_path, 10)
        report = build_report(db_path, now=_NOW, occurrence_limit=500)
        assert report.gate_blind.truncated is False

    def test_the_render_caveats_a_truncated_window(self, tmp_path):
        # α's read_occurrences docstring: "a count over a limit-capped read is a count
        # over an unknown window, and dividing by it is meaningless".  A silently-wrong
        # rate is worse than a caveated one, so the caveat must reach the operator.
        db_path = self._seed_n(tmp_path, 10)
        rendered = render_report(build_report(db_path, now=_NOW, occurrence_limit=5))
        caveats = [ln for ln in rendered.splitlines() if 'TRUNCATED' in ln]
        assert caveats, rendered
        assert 'PARTIAL' in caveats[0], caveats

    def test_an_untruncated_render_carries_no_caveat(self, tmp_path):
        db_path = self._seed_n(tmp_path, 10)
        rendered = render_report(build_report(db_path, now=_NOW, occurrence_limit=500))
        # Scanned per-line and case-sensitively on purpose: the db_path printed in the
        # header sits under pytest's tmp_path, whose directory name embeds this test's
        # own name — a bare `'truncated' not in rendered.lower()` matches the word
        # "untruncated" in that path and fails against a perfectly correct render.
        assert not [ln for ln in rendered.splitlines() if 'TRUNCATED' in ln], rendered


def _report(
    *,
    db_present: bool = True,
    open_debt: list[DebtReportRow] | None = None,
    chains: list | None = None,
    gate_blind: GateBlindCounter | None = None,
    non_convergence: NonConvergenceCounter | None = None,
    systemic: SystemicCounter | None = None,
) -> FlakeLedgerReport:
    """A hand-built report object — the render tests need no DB."""
    return FlakeLedgerReport(
        db_path=Path('/proj/data/orchestrator/runs.db'),
        db_present=db_present,
        generated_at=_NOW,
        window_since='2026-08-03T12:00:00+00:00' if db_present else None,
        open_debt=open_debt or [],
        chains=chains or [],
        gate_blind=gate_blind or compute_gate_blind([]),
        non_convergence=non_convergence or compute_non_convergence([], _NOW),
        systemic=systemic or compute_systemic([]),
    )


class TestRenderReport:
    """render_report's operator-facing contract."""

    def test_three_section_headings_are_present(self):
        rendered = render_report(_report()).upper()
        assert 'OPEN DEBT' in rendered
        assert 'RECURRENCE CHAIN' in rendered
        assert 'HEALTH COUNTER' in rendered

    def test_an_open_row_shows_test_id_owner_and_age(self):
        # INV-7 is discharged by an owner AND a bound both being operator-visible.
        row = DebtReportRow(debt=_debt(test_id='tests/test_a.py::test_one',
                                       owner_task_id='3552'), age=timedelta(days=2, hours=3))
        rendered = render_report(_report(open_debt=[row]))
        assert 'tests/test_a.py::test_one' in rendered
        assert '3552' in rendered
        assert '2d 3h' in rendered

    def test_a_missing_owner_renders_loudly_not_as_a_blank_cell(self):
        # A debt row with no owner means a flaky test is silently no longer blocking
        # anything and nobody owns fixing it (§5.7/§5.9) — the single most consequential
        # state this report can encounter.  A blank cell would render the breach as
        # missing data instead of as the breach it is.
        row = DebtReportRow(debt=_debt(test_id='orphan', owner_task_id=None),
                            age=timedelta(days=1))
        rendered = render_report(_report(open_debt=[row]))
        breach_lines = [ln for ln in rendered.splitlines() if 'orphan' in ln]
        assert breach_lines, rendered
        assert 'NO OWNER' in breach_lines[0].upper(), breach_lines

    def test_an_unparseable_opened_at_renders_unknown_not_zero(self):
        row = DebtReportRow(debt=_debt(test_id='badclock', opened_at='not-a-date'), age=None)
        line = [ln for ln in render_report(_report(open_debt=[row])).splitlines()
                if 'badclock' in ln][0]
        assert 'unknown' in line, line
        assert '0d 0h' not in line, line

    def test_a_chain_shows_its_cycle_fields_and_call_site_split(self):
        chain = build_chains(
            [
                _occ(test_id='chronic', call_site=FlakeCallSite.chronic_marker, row_id=1),
                _occ(test_id='chronic', call_site=FlakeCallSite.merge_gate, row_id=2,
                     observed_at='2026-08-08T13:00:00+00:00'),
                _occ(test_id='chronic', call_site=FlakeCallSite.main_probe, row_id=3,
                     observed_at='2026-08-08T14:00:00+00:00'),
            ],
            [],
            lambda t: _debt(test_id='chronic', open_count=7,
                            prior_resolved_at='2026-07-25T00:00:00+00:00',
                            prior_resolving_commit='deadbee'),
        )
        rendered = render_report(_report(chains=chain))
        assert 'chronic' in rendered
        assert '7' in rendered
        assert '2026-07-25' in rendered
        assert 'deadbee' in rendered
        # κ's signal must stay distinguishable from the two gates.
        assert 'chronic_marker' in rendered
        assert 'merge_gate' in rendered
        assert 'main_probe' in rendered

    def test_insufficient_observations_is_said_not_a_rate(self):
        two = [_occ(verdict=FlakeVerdict.unconfirmable, row_id=1), _occ(row_id=2)]
        rendered = render_report(_report(gate_blind=compute_gate_blind(two)))
        assert 'insufficient observations' in rendered.lower(), rendered

    def test_a_none_rate_renders_as_n_a_never_zero(self):
        rendered = render_report(_report(gate_blind=compute_gate_blind([])))
        # Matched on the full phrase, not a bare 'rate': the header's "generated_at"
        # contains that substring and would otherwise be picked as the rate line.
        rate_lines = [ln for ln in rendered.splitlines() if 'unconfirmable rate' in ln.lower()]
        assert rate_lines, rendered
        assert 'n/a' in rate_lines[0].lower(), rate_lines
        assert '0.00' not in rate_lines[0], rate_lines

    def test_all_three_counters_print(self):
        rendered = render_report(_report()).lower()
        assert 'gate blind' in rendered
        assert 'non-convergence' in rendered
        assert 'systemic' in rendered

    def test_an_empty_report_says_so_explicitly(self):
        rendered = render_report(_report()).lower()
        assert rendered.strip(), 'an empty report must not render as an empty string'
        assert 'no open debt' in rendered, rendered
        assert 'no occurrences' in rendered, rendered

    def test_render_is_byte_deterministic(self):
        report = _report(open_debt=[DebtReportRow(debt=_debt(), age=timedelta(days=1))])
        assert render_report(report) == render_report(report)

    def test_the_db_path_is_stated_and_absence_is_explicit(self):
        assert '/proj/data/orchestrator/runs.db' in render_report(_report())
        absent = render_report(_report(db_present=False)).lower()
        assert 'no ledger' in absent or 'not present' in absent, absent


class TestFlakeLedgerCommand:
    """`orchestrator flake-ledger` end to end.  All sync — the report path has no async."""

    def _project(self, tmp_path: Path, *, seed: bool = True) -> tuple[Path, Path]:
        """Write a config YAML pointing at a tmp project; optionally seed its ledger.

        The project root IS ``tmp_path``, not a ``tmp_path/project`` subdir.  conftest's
        autouse ``_isolate_orch_config`` pins ``ORCH_PROJECT_ROOT`` to ``tmp_path``, and
        ``settings_customise_sources`` (config.py:4635) orders ``env_settings`` AHEAD of
        the YAML source — so a nested root would be silently overridden and the command
        would resolve its ledger path somewhere this fixture never seeded.  Writing the
        same value into the YAML keeps config and env in agreement, which is exactly the
        arrangement that fixture's docstring prescribes for config-loading tests.  Do
        not unset the env var: the pin is a deliberate isolation guard that stops tests
        writing scheduler state into the live tree.
        """
        import yaml

        project = tmp_path
        # Config at the project root — CLAUDE.md's canonical
        # <project_root>/dark-factory-orchestrator.yaml layout.
        cfg = project / 'dark-factory-orchestrator.yaml'
        cfg.write_text(yaml.dump({'project_root': str(project)}))
        if seed:
            db_path = project / 'data' / 'orchestrator' / 'runs.db'
            db_path.parent.mkdir(parents=True)
            ensure_schema(db_path)
            _seed_debt(db_path, test_id='tests/test_a.py::test_one',
                       opened_at='2026-08-06T12:00:00+00:00', owner_task_id='4242')
            _seed_occurrence(db_path, test_id='tests/test_a.py::test_one',
                             observed_at='2026-08-09T12:00:00+00:00')
        return cfg, project

    def test_the_command_is_registered(self):
        assert 'flake-ledger' in main.commands

    def test_it_prints_the_report_and_exits_zero(self, tmp_path):
        cfg, _ = self._project(tmp_path)
        result = CliRunner().invoke(main, ['flake-ledger', '--config', str(cfg)])
        assert result.exit_code == 0, result.output
        assert 'OPEN DEBT' in result.output, result.output
        assert 'RECURRENCE CHAINS' in result.output, result.output
        assert 'HEALTH COUNTERS' in result.output, result.output
        assert 'tests/test_a.py::test_one' in result.output, result.output
        assert '4242' in result.output, result.output

    def test_no_config_and_no_env_exits_one_with_an_error(self, tmp_path, monkeypatch):
        # OPERATIONS.md §3's no-auto-discovery rule: the orchestrator never guesses the
        # target project from cwd, and this command is no exception.
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        result = CliRunner().invoke(main, ['flake-ledger'])
        assert result.exit_code == 1, result.output
        assert 'Error:' in result.output, result.output

    def test_a_project_with_no_data_dir_is_never_provisioned(self, tmp_path):
        cfg, project = self._project(tmp_path, seed=False)
        assert not (project / 'data').exists()
        result = CliRunner().invoke(main, ['flake-ledger', '--config', str(cfg)])
        assert result.exit_code == 0, result.output
        assert 'NO LEDGER' in result.output, result.output
        assert not (project / 'data' / 'orchestrator' / 'runs.db').exists(), result.output
        assert not (project / 'data').exists(), result.output

    def test_every_writer_booby_trapped_and_the_command_still_succeeds(
        self, tmp_path, monkeypatch
    ):
        # cli.py lazy-imports inside command bodies, so patching at the SOURCE module
        # path is what actually intercepts.  Plain sync raisers even though
        # open_debt/resolve_debt are `async def`: the assertion is that they are never
        # awaited, which an AsyncMock would only be needed to permit.
        def _boom(*args, **kwargs):
            raise AssertionError('the flake-ledger command reached a ledger WRITE path')

        for writer in ('record_flake_occurrence', 'open_debt', 'resolve_debt'):
            monkeypatch.setattr(f'orchestrator.flake_ledger.{writer}', _boom)
        cfg, _ = self._project(tmp_path)
        result = CliRunner().invoke(main, ['flake-ledger', '--config', str(cfg)])
        assert result.exit_code == 0, result.output
