"""Tests for shared.scheduler_state: the scheduler snapshot's one typed shape.

The literal key set and record shapes below are the contract
(plans/review-all-s14-shared-leaves-prd.md, decision 12 and invariant I4): the
producer writes exactly these keys and every reader parses them.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from shared.project_layout import runs_db, scheduler_state_json
from shared.scheduler_state import (
    CheckedSchedulerState,
    MalformedSnapshotError,
    OverrideRecord,
    ParkRecord,
    PauseRow,
    PinQueueEntry,
    RequeueCooldown,
    SchedulerStateSnapshot,
    read_scheduler_state_checked,
)

_SNAPSHOT_KEYS = {
    'skip_counts',
    'parks',
    'park_stacks',
    'pin_reservations',
    'effective_priorities',
    'pin_queue',
    'overrides',
    'current_holders',
    'held_by_task',
    'holder_limits',
    'successor_of',
    'lock_depth',
    'is_paused',
    'pause_reason',
    'pause_class',
    'requeue_cooldowns',
    'snapshot_at',
}

_INSTALLED_AT = '2026-10-15T12:00:00+00:00'

_PIN_STACK_ENTRY = {
    'owner': 'P',
    'rank': -1,
    'shadowed': False,
    'installed_at': _INSTALLED_AT,
    'source': 'pin',
    'pin_order': 1,
}
_FAIRNESS_STACK_ENTRY = {
    'owner': 'T3',
    'rank': 2,
    'shadowed': True,
    'installed_at': '',
    'source': 'fairness',
    'tier': 'medium',
}


def _populated() -> SchedulerStateSnapshot:
    return SchedulerStateSnapshot(
        skip_counts={'T3': 2},
        parks={'P': ParkRecord(modules=('pin/mod',), installed_at=_INSTALLED_AT)},
        park_stacks={'pin/mod': (_FAIRNESS_STACK_ENTRY, _PIN_STACK_ENTRY)},
        pin_reservations={'P': ParkRecord(modules=('pin/mod', 'pin/other'), installed_at=_INSTALLED_AT)},
        effective_priorities={'T3': 'high'},
        pin_queue=(PinQueueEntry(task_id='P', order=1),),
        overrides={
            'T1': OverrideRecord(
                boost_tier='high',
                pinned=True,
                reserve_now=True,
                ttl_until='2026-10-16T00:00:00+00:00',
            ),
        },
        current_holders={'mod/a': 'T7'},
        held_by_task={'T6': ('mod/a',), 'T7': ('mod/a',)},
        holder_limits={'mod/a': sys.maxsize},
        successor_of={'T7': 'T6'},
        lock_depth=3,
        is_paused=True,
        pause_reason='operator halt',
        pause_class='operator',
        requeue_cooldowns={
            'T1': RequeueCooldown(
                armed_secs=31.25,
                envelope_secs=60.0,
                transient=True,
                transient_count=1,
                armed_at=_INSTALLED_AT,
                expires_at='2026-10-15T12:00:31.250000+00:00',
            ),
        },
        snapshot_at=datetime.now(UTC),
    )


class TestEmpty:
    def test_no_producer_has_written(self) -> None:
        assert SchedulerStateSnapshot.empty().snapshot_at is None

    def test_wire_form_has_exactly_the_contract_keys(self) -> None:
        assert set(SchedulerStateSnapshot.empty().to_json()) == _SNAPSHOT_KEYS

    def test_wire_form_values_are_empty(self) -> None:
        assert SchedulerStateSnapshot.empty().to_json() == {
            'skip_counts': {},
            'parks': {},
            'park_stacks': {},
            'pin_reservations': {},
            'effective_priorities': {},
            'pin_queue': [],
            'overrides': {},
            'current_holders': {},
            'held_by_task': {},
            'holder_limits': {},
            'successor_of': {},
            'lock_depth': None,
            'is_paused': False,
            'pause_reason': None,
            'pause_class': None,
            'requeue_cooldowns': {},
            'snapshot_at': None,
        }


class TestRoundTrip:
    def test_from_json_of_to_json_is_identity(self) -> None:
        snap = _populated()
        assert SchedulerStateSnapshot.from_json(snap.to_json()) == snap

    def test_round_trip_survives_a_json_text_hop(self) -> None:
        snap = _populated()
        text = json.dumps(snap.to_json())
        assert SchedulerStateSnapshot.from_json(json.loads(text)) == snap

    def test_populated_wire_form_has_exactly_the_contract_keys(self) -> None:
        assert set(_populated().to_json()) == _SNAPSHOT_KEYS


class TestWireShape:
    def test_snapshot_at_is_isoformat(self) -> None:
        snap = _populated()
        assert snap.snapshot_at is not None
        assert snap.to_json()['snapshot_at'] == snap.snapshot_at.isoformat()

    def test_park_record(self) -> None:
        assert _populated().to_json()['parks'] == {
            'P': {'modules': ['pin/mod'], 'installed_at': _INSTALLED_AT},
        }

    def test_pin_reservation_record(self) -> None:
        assert _populated().to_json()['pin_reservations'] == {
            'P': {'modules': ['pin/mod', 'pin/other'], 'installed_at': _INSTALLED_AT},
        }

    def test_pin_queue(self) -> None:
        assert _populated().to_json()['pin_queue'] == [{'task_id': 'P', 'order': 1}]

    def test_override_record(self) -> None:
        assert _populated().to_json()['overrides'] == {
            'T1': {
                'boost_tier': 'high',
                'pinned': True,
                'reserve_now': True,
                'ttl_until': '2026-10-16T00:00:00+00:00',
            },
        }

    def test_requeue_cooldown(self) -> None:
        assert _populated().to_json()['requeue_cooldowns'] == {
            'T1': {
                'armed_secs': 31.25,
                'envelope_secs': 60.0,
                'transient': True,
                'transient_count': 1,
                'armed_at': _INSTALLED_AT,
                'expires_at': '2026-10-15T12:00:31.250000+00:00',
            },
        }

    def test_park_stack_entries_pass_through_key_for_key(self) -> None:
        stack = _populated().to_json()['park_stacks']['pin/mod']
        assert stack == [_FAIRNESS_STACK_ENTRY, _PIN_STACK_ENTRY]
        assert 'tier' not in stack[1]

    def test_scalars(self) -> None:
        wire = _populated().to_json()
        assert wire['skip_counts'] == {'T3': 2}
        assert wire['effective_priorities'] == {'T3': 'high'}
        assert wire['current_holders'] == {'mod/a': 'T7'}
        assert wire['held_by_task'] == {'T6': ['mod/a'], 'T7': ['mod/a']}
        assert wire['holder_limits'] == {'mod/a': sys.maxsize}
        assert wire['successor_of'] == {'T7': 'T6'}
        assert wire['lock_depth'] == 3
        assert wire['is_paused'] is True
        assert wire['pause_reason'] == 'operator halt'
        assert wire['pause_class'] == 'operator'


class TestRollingRestart:
    def test_unknown_keys_from_a_newer_producer_are_ignored(self) -> None:
        wire = _populated().to_json()
        newer = {
            **wire,
            'added_by_a_newer_producer': {'x': 1},
            'parks': {'P': {**wire['parks']['P'], 'added_nested': True}},
        }
        parsed = SchedulerStateSnapshot.from_json(newer)
        assert parsed == SchedulerStateSnapshot.from_json(wire)
        assert 'added_by_a_newer_producer' not in parsed.to_json()
        assert 'added_nested' not in parsed.to_json()['parks']['P']

    def test_older_producer_without_pause_class_reads_none(self) -> None:
        wire = _populated().to_json()
        del wire['pause_class']
        parsed = SchedulerStateSnapshot.from_json(wire)
        assert parsed.is_paused is True
        assert parsed.pause_class is None

    def test_older_producer_missing_keys_fill_with_defaults(self) -> None:
        parsed = SchedulerStateSnapshot.from_json({'current_holders': {'m': '7'}})
        assert parsed.snapshot_at is None
        assert parsed.to_json() == {
            **SchedulerStateSnapshot.empty().to_json(),
            'current_holders': {'m': '7'},
        }


class TestHolderTaskIds:
    def test_every_holder_of_a_shared_key_is_named(self) -> None:
        snap = SchedulerStateSnapshot.from_json({
            'current_holders': {'docs/shared.md': 'B'},
            'held_by_task': {'A': ['docs/shared.md'], 'B': ['docs/shared.md']},
        })
        assert snap.holder_task_ids() == frozenset({'A', 'B'})

    def test_an_older_producer_without_held_by_task_reads_current_holders(self) -> None:
        snap = SchedulerStateSnapshot.from_json({'current_holders': {'m': '7', 'n': '8'}})
        assert snap.held_by_task == {}
        assert snap.holder_task_ids() == frozenset({'7', '8'})

    def test_no_holders_names_no_task(self) -> None:
        assert SchedulerStateSnapshot.empty().holder_task_ids() == frozenset()


class TestHoldersByKey:
    def test_every_holder_of_each_held_key_is_named(self) -> None:
        snap = SchedulerStateSnapshot.from_json({
            'current_holders': {'docs/shared.md': 'B', 'src/x.py': 'B'},
            'held_by_task': {'A': ['docs/shared.md'], 'B': ['docs/shared.md', 'src/x.py']},
        })
        assert snap.holders_by_key() == {
            'docs/shared.md': frozenset({'A', 'B'}),
            'src/x.py': frozenset({'B'}),
        }

    def test_an_older_producer_without_held_by_task_reads_current_holders(self) -> None:
        snap = SchedulerStateSnapshot.from_json({'current_holders': {'k': '7'}})
        assert snap.holders_by_key() == {'k': frozenset({'7'})}

    def test_no_holders_maps_no_key(self) -> None:
        assert SchedulerStateSnapshot.empty().holders_by_key() == {}


class TestHolderLimit:
    @pytest.mark.parametrize('limit', [1, 2, sys.maxsize])
    def test_a_published_limit_is_read(self, limit: int) -> None:
        snap = SchedulerStateSnapshot.from_json({'holder_limits': {'k': limit}})
        assert snap.holder_limit('k') == limit

    def test_a_key_without_a_published_limit_reads_exclusive(self) -> None:
        snap = SchedulerStateSnapshot.from_json({
            'current_holders': {'k': '7'},
            'holder_limits': {'other': sys.maxsize},
        })
        assert snap.holder_limit('k') == 1

    def test_an_older_producer_that_published_no_limits_reads_exclusive(self) -> None:
        snap = SchedulerStateSnapshot.from_json({'current_holders': {'k': '7'}})
        assert snap.holder_limits == {}
        assert snap.holder_limit('k') == 1

    def test_an_unlimited_limit_survives_a_json_text_hop(self) -> None:
        snap = SchedulerStateSnapshot(holder_limits={'k': sys.maxsize})
        parsed = SchedulerStateSnapshot.from_json(json.loads(json.dumps(snap.to_json())))
        assert parsed.holder_limit('k') == sys.maxsize


class TestMalformedInput:
    @pytest.mark.parametrize('body', [[], 'x', None, 3])
    def test_non_mapping_body(self, body: Any) -> None:
        with pytest.raises(MalformedSnapshotError):
            SchedulerStateSnapshot.from_json(body)

    def test_naive_snapshot_at(self) -> None:
        with pytest.raises(MalformedSnapshotError):
            SchedulerStateSnapshot.from_json({'snapshot_at': '2026-01-01T00:00:00'})

    @pytest.mark.parametrize('depth', [True, '4', 4.0])
    def test_lock_depth_is_a_strict_int(self, depth: object) -> None:
        with pytest.raises(MalformedSnapshotError):
            SchedulerStateSnapshot.from_json({'lock_depth': depth})

    def test_park_installed_at_must_be_a_string(self) -> None:
        with pytest.raises(MalformedSnapshotError):
            SchedulerStateSnapshot.from_json(
                {'parks': {'P': {'modules': ['m'], 'installed_at': 12345}}},
            )

    def test_current_holder_must_be_a_string(self) -> None:
        with pytest.raises(MalformedSnapshotError):
            SchedulerStateSnapshot.from_json({'current_holders': {'m': 7}})

    def test_held_by_task_lock_keys_must_be_strings(self) -> None:
        with pytest.raises(MalformedSnapshotError):
            SchedulerStateSnapshot.from_json({'held_by_task': {'T1': [7]}})

    def test_holder_limits_must_be_ints(self) -> None:
        with pytest.raises(MalformedSnapshotError):
            SchedulerStateSnapshot.from_json({'holder_limits': {'k': 'many'}})


class TestFrozen:
    def test_field_assignment_is_refused(self) -> None:
        snap = SchedulerStateSnapshot.empty()
        with pytest.raises(ValidationError):
            snap.is_paused = True


_NOW = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
_SNAPSHOT_AT = _NOW - timedelta(seconds=90)
_ROW_PAUSE_AT = '2026-11-27T11:58:00+00:00'

# A minimal transcription of the scheduler_state table in
# orchestrator/src/orchestrator/run_store.py (shared cannot import orchestrator).
_SCHEDULER_STATE_DDL = (
    'CREATE TABLE scheduler_state ('
    ' project_id TEXT PRIMARY KEY,'
    ' pause_reason TEXT NOT NULL,'
    ' pause_at TEXT NOT NULL,'
    ' set_by_run_id TEXT NOT NULL,'
    ' pause_class TEXT)'
)


def _write_snapshot(root: Path, snapshot: SchedulerStateSnapshot) -> None:
    path = scheduler_state_json(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot.to_json()))


def _write_snapshot_body(root: Path, body: str) -> None:
    path = scheduler_state_json(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def _seed_runs_db(root: Path, *rows: tuple[str, str, str, str | None], with_table: bool = True) -> None:
    """Create runs.db; each row is (project_id, pause_reason, pause_at, pause_class)."""
    path = runs_db(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        if with_table:
            conn.execute(_SCHEDULER_STATE_DDL)
            conn.executemany(
                'INSERT INTO scheduler_state '
                '(project_id, pause_reason, pause_at, set_by_run_id, pause_class) '
                'VALUES (?, ?, ?, ?, ?)',
                [(pid, reason, at, 'run-1', cls) for pid, reason, at, cls in rows],
            )
        else:
            conn.execute('CREATE TABLE unrelated (x INTEGER)')
        conn.commit()
    finally:
        conn.close()


def _paused_snapshot(reason: str = 'snapshot reason', pause_class: str = 'operator') -> SchedulerStateSnapshot:
    return SchedulerStateSnapshot(
        is_paused=True,
        pause_reason=reason,
        pause_class=pause_class,
        snapshot_at=_SNAPSHOT_AT,
    )


def _running_snapshot() -> SchedulerStateSnapshot:
    return SchedulerStateSnapshot(snapshot_at=_SNAPSHOT_AT)


class TestFromBytes:
    def test_valid_bytes_parse_as_from_json_of_the_same_dict(self) -> None:
        wire = _populated().to_json()
        parsed = SchedulerStateSnapshot.from_bytes(json.dumps(wire).encode())
        assert parsed == SchedulerStateSnapshot.from_json(wire)

    @pytest.mark.parametrize(
        'raw', [b'', b'{not json', b'\xff\xfe', b'[]', b'{"lock_depth": true}'],
    )
    def test_bytes_that_are_not_a_snapshot_raise_malformed(self, raw: bytes) -> None:
        with pytest.raises(MalformedSnapshotError):
            SchedulerStateSnapshot.from_bytes(raw)

    def test_the_checked_view_parses_as_itself(self) -> None:
        parsed = CheckedSchedulerState.from_bytes(json.dumps(_populated().to_json()).encode())
        assert isinstance(parsed, CheckedSchedulerState)


class TestCheckedReaderTruthTable:
    def test_no_files_reads_as_running_and_unaged(self, tmp_path: Path) -> None:
        view = read_scheduler_state_checked(tmp_path, now=_NOW)
        assert view.is_paused is False
        assert view.snapshot_age_s is None
        assert view.pause_row is None
        assert view.pause_disagreement is False

    def test_a_paused_snapshot_without_runs_db_serves_the_snapshot(self, tmp_path: Path) -> None:
        _write_snapshot(tmp_path, _paused_snapshot('halted', 'cost_ceiling'))
        view = read_scheduler_state_checked(tmp_path, now=_NOW)
        assert (view.is_paused, view.pause_reason, view.pause_class) == (True, 'halted', 'cost_ceiling')
        assert view.snapshot_age_s == (_NOW - _SNAPSHOT_AT).total_seconds()
        assert view.pause_row is None
        assert view.pause_disagreement is False

    def test_both_paused_with_different_reasons_agree_and_serve_the_snapshot(self, tmp_path: Path) -> None:
        _write_snapshot(tmp_path, _paused_snapshot('first pause', 'operator'))
        _seed_runs_db(tmp_path, ('dark_factory', 'later pause', _ROW_PAUSE_AT, 'cost_ceiling'))
        view = read_scheduler_state_checked(tmp_path, now=_NOW)
        assert view.is_paused is True
        assert view.pause_reason == 'first pause'
        assert view.pause_class == 'operator'
        assert view.pause_row == PauseRow(
            reason='later pause', pause_at=_ROW_PAUSE_AT, pause_class='cost_ceiling',
        )
        assert view.pause_disagreement is False

    def test_a_running_snapshot_over_a_pause_row_serves_the_row(self, tmp_path: Path) -> None:
        _write_snapshot(tmp_path, _running_snapshot())
        _seed_runs_db(tmp_path, ('dark_factory', 'cost_ceiling_orch_exceeded', _ROW_PAUSE_AT, 'cost_ceiling'))
        view = read_scheduler_state_checked(tmp_path, now=_NOW)
        assert view.is_paused is True
        assert view.pause_reason == 'cost_ceiling_orch_exceeded'
        assert view.pause_class == 'cost_ceiling'
        assert view.pause_disagreement is True

    def test_a_paused_snapshot_over_an_empty_table_serves_running(self, tmp_path: Path) -> None:
        _write_snapshot(tmp_path, _paused_snapshot())
        _seed_runs_db(tmp_path)
        view = read_scheduler_state_checked(tmp_path, now=_NOW)
        assert (view.is_paused, view.pause_reason, view.pause_class) == (False, None, None)
        assert view.pause_row is None
        assert view.pause_disagreement is True

    def test_a_row_without_a_snapshot_file_serves_the_row(self, tmp_path: Path) -> None:
        _seed_runs_db(tmp_path, ('dark_factory', 'halted', _ROW_PAUSE_AT, 'cost_ceiling'))
        view = read_scheduler_state_checked(tmp_path, now=_NOW)
        assert (view.is_paused, view.pause_reason, view.pause_class) == (True, 'halted', 'cost_ceiling')
        assert view.snapshot_age_s is None
        assert view.pause_disagreement is False

    def test_runs_db_without_the_table_serves_the_snapshot(self, tmp_path: Path) -> None:
        _write_snapshot(tmp_path, _paused_snapshot('halted', 'operator'))
        _seed_runs_db(tmp_path, with_table=False)
        view = read_scheduler_state_checked(tmp_path, now=_NOW)
        assert (view.is_paused, view.pause_reason, view.pause_class) == (True, 'halted', 'operator')
        assert view.pause_row is None
        assert view.pause_disagreement is False


class TestCheckedReaderRow:
    def test_a_null_pause_class_is_served_raw(self, tmp_path: Path) -> None:
        _seed_runs_db(tmp_path, ('dark_factory', 'legacy halt', _ROW_PAUSE_AT, None))
        view = read_scheduler_state_checked(tmp_path, now=_NOW)
        assert view.pause_row is not None
        assert view.pause_row.pause_class is None
        assert view.pause_class is None

    def test_the_latest_row_is_served(self, tmp_path: Path) -> None:
        _seed_runs_db(
            tmp_path,
            ('a', 'older', '2026-11-27T10:00:00+00:00', 'operator'),
            ('b', 'newer', '2026-11-27T11:00:00+00:00', 'cost_ceiling'),
        )
        view = read_scheduler_state_checked(tmp_path, now=_NOW)
        assert view.pause_row == PauseRow(
            reason='newer', pause_at='2026-11-27T11:00:00+00:00', pause_class='cost_ceiling',
        )

    def test_the_reader_never_creates_runs_db(self, tmp_path: Path) -> None:
        _write_snapshot(tmp_path, _paused_snapshot())
        read_scheduler_state_checked(tmp_path, now=_NOW)
        assert not runs_db(tmp_path).exists()


class TestCheckedReaderMalformedSnapshot:
    @pytest.mark.parametrize('body', ['{not json', '[]'])
    def test_a_malformed_body_reads_as_unreadable_and_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, body: str,
    ) -> None:
        _write_snapshot_body(tmp_path, body)
        with caplog.at_level(logging.WARNING, logger='shared.scheduler_state'):
            view = read_scheduler_state_checked(tmp_path, now=_NOW)
        assert view == CheckedSchedulerState()
        assert any(
            record.name == 'shared.scheduler_state' and record.levelno == logging.WARNING
            for record in caplog.records
        )

    def test_a_malformed_body_over_a_row_serves_the_row(self, tmp_path: Path) -> None:
        _write_snapshot_body(tmp_path, '{not json')
        _seed_runs_db(tmp_path, ('dark_factory', 'halted', _ROW_PAUSE_AT, 'cost_ceiling'))
        view = read_scheduler_state_checked(tmp_path, now=_NOW)
        assert view.is_paused is True
        assert view.pause_disagreement is False


class TestCheckedReaderClock:
    def test_a_naive_now_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='timezone-aware'):
            read_scheduler_state_checked(tmp_path, now=datetime(2026, 11, 27, 12, 0))


class TestCheckedWireForm:
    def _view(self, tmp_path: Path) -> CheckedSchedulerState:
        _write_snapshot(tmp_path, _running_snapshot())
        _seed_runs_db(tmp_path, ('dark_factory', 'halted', _ROW_PAUSE_AT, 'cost_ceiling'))
        return read_scheduler_state_checked(tmp_path, now=_NOW)

    def test_wire_form_adds_exactly_the_reader_keys(self, tmp_path: Path) -> None:
        assert set(self._view(tmp_path).to_json()) == _SNAPSHOT_KEYS | {
            'snapshot_age_s', 'pause_row', 'pause_disagreement',
        }

    def test_pause_row_serialises_as_its_raw_strings(self, tmp_path: Path) -> None:
        assert self._view(tmp_path).to_json()['pause_row'] == {
            'reason': 'halted', 'pause_at': _ROW_PAUSE_AT, 'pause_class': 'cost_ceiling',
        }

    def test_from_json_of_to_json_is_identity(self, tmp_path: Path) -> None:
        view = self._view(tmp_path)
        assert CheckedSchedulerState.from_json(view.to_json()) == view
