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

from datetime import UTC, datetime, timedelta

import pytest

from orchestrator.flake_ledger import (
    UNKNOWN_TEST_ID,
    DebtRow,
    FlakeCallSite,
    FlakeOccurrenceRow,
    FlakeVerdict,
)
from orchestrator.flake_report import (
    _parse_stamp,
    compute_gate_blind,
    compute_non_convergence,
    compute_systemic,
    format_age,
)

_NOW = datetime(2026, 8, 10, 12, 0, 0, tzinfo=UTC)


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
