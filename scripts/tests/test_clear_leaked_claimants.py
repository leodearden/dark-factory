"""Tests for scripts/clear_leaked_claimants.py (task 4625).

Every expectation is hand-derived from the tier table in
docs/prds/claimant-invariant-enforcement.md D3, never computed by calling the
shared predicates, and every store is a synthetic temp tasks.db.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from _task_db_scan import UNREADABLE_STORE_ERRORS
from clear_leaked_claimants import Tier, take_census

NOW = datetime(2026, 10, 11, 12, tzinfo=UTC)
STALE = NOW - timedelta(hours=2)
FRESH = NOW - timedelta(minutes=1)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _claimed(task_id, status, claimant, heartbeat, **extra):
    heartbeat_at = None if heartbeat is None else _iso(heartbeat)
    return {
        'id': task_id, 'status': status,
        'claimant_run_id': claimant, 'heartbeat_at': heartbeat_at, **extra,
    }


POSITIVE_CONTROL = _claimed(4028, 'done', 'run-a/4028-x/pid=1', NOW - timedelta(days=50))
FREEFORM_CANCELLED = _claimed(5225, 'cancelled', 'agent-esc-5053-2-docs-fix', None)
FRESH_DONE = _claimed(101, 'done', 'run-c/101-x/pid=3', FRESH)
STALE_PENDING = _claimed(103, 'pending', 'run-e/103-x/pid=5', STALE)
STALE_MERGE_DEFERRED = _claimed(109, 'merge-deferred', 'run-k/109-x/pid=11', STALE)

TIER_TABLE_ROWS = [
    POSITIVE_CONTROL,
    FREEFORM_CANCELLED,
    FRESH_DONE,
    _claimed(102, 'done', None, FRESH),
    STALE_PENDING,
    _claimed(104, 'pending', 'run-f/104-x/pid=6', FRESH),
    _claimed(105, 'in-progress', 'run-g/105-x/pid=7', STALE),
    _claimed(106, 'infra-hold', 'run-h/106-x/pid=8', STALE),
    _claimed(107, 'blocked', 'run-i/107-x/pid=9', STALE),
    _claimed(108, 'deferred', 'run-j/108-x/pid=10', STALE, metadata={'infra_hold': True}),
    STALE_MERGE_DEFERRED,
    _claimed(110, 'pending', '  ', STALE),
]


@pytest.fixture
def seed_root(tmp_path, make_tasks_db):
    def _seed(rows, name='proj') -> str:
        root = tmp_path / name
        store_dir = root / '.taskmaster' / 'tasks'
        store_dir.mkdir(parents=True)
        make_tasks_db(rows, directory=store_dir)
        return str(root)

    return _seed


class TestCensus:
    def test_classifies_each_row_by_the_d3_tier_table(self, seed_root):
        census = take_census(seed_root(TIER_TABLE_ROWS), NOW)

        assert {(v.task_id, v.tier) for v in census.violations} == {
            (4028, Tier.TERMINAL),
            (5225, Tier.TERMINAL),
            (101, Tier.TERMINAL),
            (103, Tier.HYGIENE),
            (109, Tier.HYGIENE),
        }

    def test_violation_carries_the_row_verbatim_and_the_measurement_time(self, seed_root):
        census = take_census(seed_root(TIER_TABLE_ROWS), NOW)
        by_id = {v.task_id: v for v in census.violations}

        for row in (POSITIVE_CONTROL, FREEFORM_CANCELLED, STALE_PENDING):
            violation = by_id[row['id']]
            assert violation.tag == 'master'
            assert violation.status == row['status']
            assert violation.claimant_run_id == row['claimant_run_id']
            assert violation.heartbeat_at == row['heartbeat_at']
            assert violation.measured_at == NOW

    def test_reports_how_many_rows_it_read(self, seed_root):
        census = take_census(seed_root(TIER_TABLE_ROWS), NOW)

        assert census.rows_in_store == len(TIER_TABLE_ROWS)

    def test_totals_by_tier_and_by_tier_and_status(self, seed_root):
        totals = take_census(seed_root(TIER_TABLE_ROWS), NOW).totals()

        assert totals['by_tier'] == {Tier.TERMINAL: 3, Tier.HYGIENE: 2}
        assert totals['by_tier_status'] == {
            Tier.TERMINAL: {'done': 2, 'cancelled': 1},
            Tier.HYGIENE: {'pending': 1, 'merge-deferred': 1},
        }

    def test_zero_tiers_are_reported_not_omitted(self, seed_root):
        totals = take_census(seed_root([POSITIVE_CONTROL]), NOW).totals()

        assert totals['by_tier'] == {Tier.TERMINAL: 1, Tier.HYGIENE: 0}

    def test_an_empty_stub_store_is_unreadable_not_a_clean_zero(self, tmp_path):
        root = tmp_path / 'stub'
        store = root / '.taskmaster' / 'tasks' / 'tasks.db'
        store.parent.mkdir(parents=True)
        store.write_bytes(b'')

        with pytest.raises(UNREADABLE_STORE_ERRORS):
            take_census(str(root), NOW)

