"""Owner-liveness corroboration of ι's flake-ledger report (task 5875).

A stored ``owner_task_id`` is a snapshot.  These tests pin how the report reads each open
row's owner live, counts a DONE or a cancelled/gone owner as a §5.9 breach, and says so
when it could not read.

STRUCTURAL CONSTRAINT — sync and async tests live in strictly separate classes.
``orchestrator/pyproject.toml`` sets no ``asyncio_mode`` (pytest-asyncio STRICT) and
promotes the mark-mismatch warning to an ERROR-level filterwarning, so a sync ``def`` in
an ``@pytest.mark.asyncio`` class is an error.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from shared.task_statuses import OwnerLiveness

from orchestrator.flake_ledger import DebtRow, DueSummary
from orchestrator.flake_report import (
    DebtReportRow,
    FlakeLedgerReport,
    OwnerReadFailed,
    OwnerStatuses,
    build_report,
    compute_gate_blind,
    compute_non_convergence,
    compute_systemic,
    corroborate_owners,
)

NOW = datetime(2026, 11, 16, 12, 0, 0, tzinfo=UTC)


def _row(
    test_id: str,
    owner_task_id: str | None,
    *,
    opened_at: str = '2026-11-14T12:00:00+00:00',
    open_count: int = 1,
) -> DebtRow:
    return DebtRow(
        test_id=test_id,
        project_id='dark_factory',
        opened_at=opened_at,
        resolved_at=None,
        owner_task_id=owner_task_id,
        open_count=open_count,
        prior_resolved_at=None,
        prior_resolving_commit=None,
        last_occurrence_at=opened_at,
    )


class TestOwnerStatuses:
    def test_each_status_classifies_as_the_ledger_classifies_it(self):
        expected = {
            'done': OwnerLiveness.DONE,
            'cancelled': OwnerLiveness.CLOSED,
            'deferred': OwnerLiveness.DEFERRED,
            'pending': OwnerLiveness.OPEN,
            'in-progress': OwnerLiveness.OPEN,
            'blocked': OwnerLiveness.OPEN,
        }
        for status, liveness in expected.items():
            assert OwnerStatuses({'7': status}).liveness('7') is liveness, status

    def test_an_id_the_read_omitted_is_a_corroborated_absence(self):
        assert OwnerStatuses({}).liveness('9') is OwnerLiveness.CLOSED

    def test_the_value_holds_its_own_read_only_copy(self):
        answer = {'7': 'pending'}
        statuses = OwnerStatuses(answer)
        answer['7'] = 'done'
        answer['8'] = 'done'
        assert statuses.liveness('7') is OwnerLiveness.OPEN
        assert statuses.liveness('8') is OwnerLiveness.CLOSED


class TestNonConvergenceOwnerCounts:
    ROWS = [
        _row('t/done_a', '101'),
        _row('t/done_b', '102'),
        _row('t/cancelled', '103'),
        _row('t/gone', '104'),
        _row('t/pending', '105'),
        _row('t/deferred', '106'),
        _row('t/unowned', None),
    ]
    STATUSES = OwnerStatuses({
        '101': 'done',
        '102': 'done',
        '103': 'cancelled',
        '105': 'pending',
        '106': 'deferred',
    })

    def test_done_and_closed_owners_are_counted_separately(self):
        nc = compute_non_convergence(self.ROWS, NOW, owner_statuses=self.STATUSES)
        assert nc.owner_done_tests == 2
        assert nc.owner_closed_tests == 2

    def test_the_other_counts_are_unchanged_by_the_owner_read(self):
        read = compute_non_convergence(self.ROWS, NOW, owner_statuses=self.STATUSES)
        unread = compute_non_convergence(self.ROWS, NOW)
        assert read.unowned_tests == 1
        assert (read.open_tests, read.recurrent_tests, read.over_age_tests) == (
            unread.open_tests, unread.recurrent_tests, unread.over_age_tests,
        )

    def test_not_read_is_none_never_zero(self):
        nc = compute_non_convergence(self.ROWS, NOW)
        assert nc.owner_done_tests is None
        assert nc.owner_closed_tests is None


def _report(open_rows: list[DebtRow], *, age_days: int = 5) -> FlakeLedgerReport:
    def _age(row: DebtRow) -> timedelta:
        return NOW - datetime.fromisoformat(row.opened_at)

    return FlakeLedgerReport(
        db_path=Path('/proj/data/orchestrator/runs.db'),
        db_present=True,
        generated_at=NOW,
        window_since='2026-11-09T12:00:00+00:00',
        open_debt=[DebtReportRow(debt=r, age=_age(r)) for r in open_rows],
        chains=[],
        gate_blind_by_call_site={'merge_gate': compute_gate_blind([])},
        systemic_by_call_site={'merge_gate': compute_systemic([])},
        non_convergence=compute_non_convergence(open_rows, NOW, age_days=age_days),
        due=DueSummary(
            size=0, oldest=None, stale=0, unnamed_stale=0, stale_sample=(), pending_since={},
        ),
    )


_Answer = tuple[Mapping[str, str], Exception | None]


class _NotAPair(_Answer):
    """An answer that claims the pair shape but holds one element, so it will not unpack."""

    def __new__(cls) -> _NotAPair:
        return super().__new__(cls, ({},))


class _Reader:
    """An owner-status reader answering a canned value, recording every call."""

    def __init__(self, answer: _Answer = ({}, None), *, raises: Exception | None = None,
                 hang: bool = False) -> None:
        self.answer = answer
        self.raises = raises
        self.hang = hang
        self.calls: list[list[str]] = []

    async def __call__(self, ids: list[str]) -> _Answer:
        self.calls.append(list(ids))
        if self.hang:
            await asyncio.Event().wait()
        if self.raises is not None:
            raise self.raises
        return self.answer


@pytest.mark.asyncio
class TestCorroborateOwners:
    ROWS = [
        _row('t/a', '202', opened_at='2026-11-01T12:00:00+00:00'),
        _row('t/b', '101'),
        _row('t/c', '202'),
        _row('t/unowned', None),
    ]

    async def test_reads_the_distinct_owners_once_and_counts_them(self):
        reader = _Reader(({'101': 'done', '202': 'pending'}, None))
        out = await corroborate_owners(_report(self.ROWS), reader)
        assert reader.calls == [['101', '202']]
        assert isinstance(out.owner_read, OwnerStatuses)
        assert dict(out.owner_read.statuses) == {'101': 'done', '202': 'pending'}
        assert out.non_convergence.owner_done_tests == 1
        assert out.non_convergence.owner_closed_tests == 0

    async def test_returns_a_new_report_and_keeps_every_other_field(self):
        report = _report(self.ROWS)
        out = await corroborate_owners(report, _Reader(({'101': 'done'}, None)))
        assert out is not report
        assert report.owner_read is None
        for field in fields(report):
            if field.name not in {'owner_read', 'non_convergence'}:
                assert getattr(out, field.name) == getattr(report, field.name), field.name
        assert out.non_convergence.age_threshold_days == 5
        assert out.non_convergence.over_age_tests == report.non_convergence.over_age_tests == 1

    async def test_an_in_band_error_is_a_failed_read(self):
        out = await corroborate_owners(
            _report(self.ROWS), _Reader(({}, RuntimeError('fm down'))),
        )
        assert isinstance(out.owner_read, OwnerReadFailed)
        assert 'fm down' in out.owner_read.error
        assert out.non_convergence.owner_done_tests is None
        assert out.non_convergence.owner_closed_tests is None

    async def test_a_raising_reader_is_a_failed_read_not_a_raise(self):
        out = await corroborate_owners(_report(self.ROWS), _Reader(raises=OSError('refused')))
        assert isinstance(out.owner_read, OwnerReadFailed)
        assert 'refused' in out.owner_read.error

    async def test_a_hung_reader_is_bounded_by_the_budget(self):
        out = await asyncio.wait_for(
            corroborate_owners(_report(self.ROWS), _Reader(hang=True), budget_secs=0.05),
            timeout=5,
        )
        assert isinstance(out.owner_read, OwnerReadFailed)
        assert 'Timeout' in out.owner_read.error
        assert '0.05' in out.owner_read.error

    async def test_an_answer_that_is_not_a_pair_is_a_failed_read(self):
        out = await corroborate_owners(_report(self.ROWS), _Reader(_NotAPair()))
        assert isinstance(out.owner_read, OwnerReadFailed)

    @pytest.mark.parametrize('rows', [[], [_row('t/unowned', None)]])
    async def test_no_owned_rows_reads_nothing_and_counts_zero(self, rows):
        reader = _Reader(({}, None))
        out = await corroborate_owners(_report(rows), reader)
        assert reader.calls == []
        assert out.owner_read == OwnerStatuses({})
        assert out.non_convergence.owner_done_tests == 0
        assert out.non_convergence.owner_closed_tests == 0

    async def test_an_unmeasured_report_is_returned_unread(self, tmp_path):
        report = build_report(tmp_path / 'absent' / 'runs.db', now=NOW)
        reader = _Reader(({}, None))
        out = await corroborate_owners(report, reader)
        assert reader.calls == []
        assert out.owner_read is None
