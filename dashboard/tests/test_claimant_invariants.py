"""Claimant-invariant census over real tasks.db files (PRD docs/prds/claimant-invariant-detection.md).

Every scan reads a genuine sqlite file through a real :class:`DbPool`, so the
production ``?mode=ro`` open and ``aiosqlite.Row`` row factory are exercised.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from _tasks_db_fixture import task_row, write_tasks_db

from dashboard.data.claimant_invariants import HygieneTier, TerminalTier
from dashboard.data.db import DbPool
from dashboard.data.root_invariants import RootInvariants, ScanStatus, scan_root_invariants

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
FRESH = (NOW - timedelta(minutes=1)).isoformat()
STALE = (NOW - timedelta(hours=1)).isoformat()

_CLAIMANT = 'run-a/s/pid=1'


def _db_path(root: Path) -> Path:
    return root / '.taskmaster' / 'tasks' / 'tasks.db'


async def _scan_file(root: Path, db_path: Path) -> dict[str, Any]:
    pool = DbPool()
    try:
        conn = await pool.get(db_path)
        record = await scan_root_invariants(
            root, db_path, conn, now=NOW, pending_escalations=None
        )
    finally:
        await pool.close_all()
    return record.to_wire()


async def _scan(root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return await _scan_file(root, write_tasks_db(_db_path(root), rows))


def _ledger(claimant: str) -> dict[str, Any]:
    return {
        'claimant_exception': {
            'claimant_run_id': claimant,
            'target_status': 'done',
            'agent_id': None,
            'tag': None,
            'stamped_at': FRESH,
        }
    }


def _violation_ids(wire: dict[str, Any]) -> list[int]:
    return [v['task_id'] for v in wire['terminal_tier']['violations']]


async def test_b1_terminal_claimant_is_a_violation(tmp_path):
    root = tmp_path / 'proj-a'
    wire = await _scan(
        root,
        [
            task_row(1, 'done', _CLAIMANT, STALE),
            task_row(2, 'done'),
            task_row(3, 'pending'),
        ],
    )
    assert wire['status'] == 'ok'
    assert wire['error'] is None
    assert wire['rows_scanned'] == 3
    assert wire['measured_at'] == NOW.isoformat()
    assert wire['root'] == str(root)
    assert wire['project'] == root.name
    assert wire['terminal_tier']['count'] == 1
    assert wire['terminal_tier']['unledgered_count'] == 1
    assert wire['terminal_tier']['violations'] == [
        {
            'tag': 'master',
            'task_id': 1,
            'status': 'done',
            'claimant_run_id': _CLAIMANT,
            'heartbeat_at': STALE,
            'ledgered': False,
        }
    ]


async def test_b4_heartbeat_residue_without_claimant_is_not_a_violation(tmp_path):
    wire = await _scan(tmp_path / 'proj', [task_row(1, 'done', None, FRESH), task_row(2, 'pending')])
    assert wire['terminal_tier']['count'] == 0
    assert wire['terminal_tier']['violations'] == []
    assert wire['hygiene_tier'] == {'count': 0, 'stale_count': 0}


async def test_b4_fresh_claimant_on_terminal_row_still_violates(tmp_path):
    wire = await _scan(tmp_path / 'proj', [task_row(1, 'cancelled', _CLAIMANT, FRESH)])
    assert _violation_ids(wire) == [1]


async def test_b3_stale_hygiene_claimants_are_counted_not_alarmed(tmp_path):
    rows = [
        task_row(i, status, _CLAIMANT, STALE)
        for i, status in enumerate(['pending', 'deferred', 'review', 'merge-deferred'], start=1)
    ]
    wire = await _scan(tmp_path / 'proj', rows)
    assert wire['hygiene_tier'] == {'count': 4, 'stale_count': 4}
    assert wire['terminal_tier']['count'] == 0


async def test_b3_fresh_hygiene_claimant_counts_but_is_not_stale(tmp_path):
    rows = [
        task_row(1, 'pending', _CLAIMANT, STALE),
        task_row(2, 'deferred', _CLAIMANT, STALE),
        task_row(3, 'review', _CLAIMANT, STALE),
        task_row(4, 'merge-deferred', _CLAIMANT, STALE),
        task_row(5, 'pending', _CLAIMANT, FRESH),
    ]
    wire = await _scan(tmp_path / 'proj', rows)
    assert wire['hygiene_tier'] == {'count': 5, 'stale_count': 4}


async def test_b3_in_progress_and_blocked_claimants_are_in_neither_tier(tmp_path):
    rows = [task_row(1, 'in-progress', _CLAIMANT, STALE), task_row(2, 'blocked', _CLAIMANT, STALE)]
    wire = await _scan(tmp_path / 'proj', rows)
    assert wire['hygiene_tier'] == {'count': 0, 'stale_count': 0}
    assert wire['terminal_tier']['count'] == 0


async def test_b7_ledgered_violation_is_listed_but_not_unledgered(tmp_path):
    wire = await _scan(
        tmp_path / 'proj', [task_row(1, 'done', _CLAIMANT, STALE, metadata=_ledger(_CLAIMANT))]
    )
    assert wire['terminal_tier']['count'] == 1
    assert wire['terminal_tier']['unledgered_count'] == 0
    assert wire['terminal_tier']['violations'][0]['ledgered'] is True


async def test_b7_ledger_naming_another_claimant_does_not_cover(tmp_path):
    wire = await _scan(
        tmp_path / 'proj', [task_row(1, 'done', _CLAIMANT, STALE, metadata=_ledger('run-b/s/pid=2'))]
    )
    assert wire['terminal_tier']['unledgered_count'] == 1
    assert wire['terminal_tier']['violations'][0]['ledgered'] is False


async def test_unreadable_metadata_never_hides_a_violation(tmp_path):
    wire = await _scan(tmp_path / 'proj', [task_row(1, 'done', _CLAIMANT, STALE, metadata='{not json')])
    assert _violation_ids(wire) == [1]
    assert wire['terminal_tier']['violations'][0]['ledgered'] is False


async def test_blank_claimant_is_decided_by_the_predicate_not_sql(tmp_path):
    wire = await _scan(tmp_path / 'proj', [task_row(1, 'done', '   ', STALE)])
    assert wire['terminal_tier']['count'] == 0


async def test_violations_are_ordered_by_tag_then_id_and_carry_their_tag(tmp_path):
    rows = [
        task_row(9, 'done', _CLAIMANT, STALE, tag='master'),
        task_row(7, 'done', _CLAIMANT, STALE, tag='master'),
        task_row(7, 'cancelled', _CLAIMANT, STALE, tag='feature'),
    ]
    wire = await _scan(tmp_path / 'proj', rows)
    assert [(v['tag'], v['task_id']) for v in wire['terminal_tier']['violations']] == [
        ('feature', 7),
        ('master', 7),
        ('master', 9),
    ]


async def test_metadata_is_decoded_before_the_shared_predicate_sees_it(tmp_path):
    """The legacy infra_hold overload is only honoured on a decoded dict."""
    wire = await _scan(
        tmp_path / 'proj', [task_row(1, 'pending', _CLAIMANT, STALE, metadata={'infra_hold': True})]
    )
    assert wire['hygiene_tier'] == {'count': 1, 'stale_count': 0}


def _assert_blind(wire: dict[str, Any]) -> None:
    """D-2: a blind root names its error and makes no zero-count claim."""
    assert wire['status'] == 'blind'
    assert isinstance(wire['error'], str) and wire['error']
    assert wire['terminal_tier'] is None
    assert wire['hygiene_tier'] is None
    assert wire['chains'] is None
    assert wire['measured_at'] == NOW.isoformat()


async def test_b2_absent_db_is_blind_not_clean(tmp_path):
    root = tmp_path / 'proj'
    db_path = _db_path(root)
    record = await scan_root_invariants(
        root, db_path, None, now=NOW, pending_escalations=None
    )
    wire = record.to_wire()
    _assert_blind(wire)
    assert str(db_path) in wire['error']
    assert 'not found' in wire['error']
    assert wire['rows_scanned'] is None


async def test_b2_present_but_unopened_db_is_blind_with_distinct_wording(tmp_path):
    root = tmp_path / 'proj'
    db_path = write_tasks_db(_db_path(root), [task_row(1, 'done', _CLAIMANT, STALE)])
    record = await scan_root_invariants(
        root, db_path, None, now=NOW, pending_escalations=None
    )
    wire = record.to_wire()
    _assert_blind(wire)
    assert str(db_path) in wire['error']
    assert 'could not be opened read-only' in wire['error']
    assert 'not found' not in wire['error']
    assert wire['rows_scanned'] is None


async def test_b2_zero_byte_file_is_blind_with_the_sqlite_error(tmp_path):
    root = tmp_path / 'proj'
    db_path = _db_path(root)
    db_path.parent.mkdir(parents=True)
    db_path.touch()
    wire = await _scan_file(root, db_path)
    _assert_blind(wire)
    assert 'OperationalError' in wire['error']
    assert 'no such table' in wire['error']
    assert wire['rows_scanned'] is None


async def test_b2_pre_claimant_schema_is_blind(tmp_path):
    root = tmp_path / 'proj'
    db_path = _db_path(root)
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute('CREATE TABLE tasks (tag TEXT, id INTEGER, status TEXT, metadata TEXT)')
        conn.execute("INSERT INTO tasks VALUES ('master', 1, 'done', NULL)")
        conn.commit()
    finally:
        conn.close()
    wire = await _scan_file(root, db_path)
    _assert_blind(wire)
    assert 'no such column' in wire['error']


async def test_b2_a_pooled_connection_closed_under_the_scan_is_blind_not_raised(tmp_path):
    root = tmp_path / 'proj'
    db_path = write_tasks_db(_db_path(root), [task_row(1, 'done', _CLAIMANT, STALE)])
    pool = DbPool()
    conn = await pool.get(db_path)
    await pool.close_all()
    record = await scan_root_invariants(
        root, db_path, conn, now=NOW, pending_escalations=None
    )
    wire = record.to_wire()
    _assert_blind(wire)
    assert wire['error'].startswith('ValueError: ')
    assert wire['rows_scanned'] is None


async def test_b2_zero_rows_scanned_is_blind(tmp_path):
    wire = await _scan(tmp_path / 'proj', [])
    _assert_blind(wire)
    assert wire['rows_scanned'] == 0
    assert '0 rows' in wire['error']


_OK_TIERS: dict[str, Any] = {
    'terminal_tier': TerminalTier(()),
    'hygiene_tier': HygieneTier(count=0, stale_count=0),
    'chains': (),
}
_BLIND_TIERS: dict[str, Any] = {'terminal_tier': None, 'hygiene_tier': None, 'chains': None}


@pytest.mark.parametrize(
    ('status', 'fields', 'offending'),
    [
        (ScanStatus.OK, {'rows_scanned': 0, 'error': None, **_OK_TIERS}, 'rows_scanned'),
        (ScanStatus.OK, {'rows_scanned': 3, 'error': 'boom', **_OK_TIERS}, 'error'),
        (ScanStatus.BLIND, {'rows_scanned': None, 'error': None, **_BLIND_TIERS}, 'error'),
        (
            ScanStatus.BLIND,
            {'rows_scanned': None, 'error': 'boom', **_BLIND_TIERS, 'terminal_tier': TerminalTier(())},
            'terminal_tier',
        ),
        (ScanStatus.OK, {'rows_scanned': 3, 'error': None, **_OK_TIERS, 'chains': None}, 'chains'),
        (
            ScanStatus.BLIND,
            {'rows_scanned': None, 'error': 'boom', **_BLIND_TIERS, 'chains': ()},
            'chains',
        ),
    ],
)
def test_record_invariant_is_enforced_at_construction(tmp_path, status, fields, offending):
    with pytest.raises(ValueError) as excinfo:
        RootInvariants(root=tmp_path, status=status, measured_at=NOW, **fields)
    assert status.value in str(excinfo.value)
    assert offending in str(excinfo.value)
