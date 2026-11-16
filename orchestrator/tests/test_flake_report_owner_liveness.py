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

from datetime import UTC, datetime

from shared.task_statuses import OwnerLiveness

from orchestrator.flake_ledger import DebtRow
from orchestrator.flake_report import (
    OwnerStatuses,
    compute_non_convergence,
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
