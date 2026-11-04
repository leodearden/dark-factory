"""strict_event_reads: event-store reads that raise instead of reading as empty (task 3820)."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from orchestrator.event_store import EventStore, EventType
from orchestrator.strict_event_reads import EventReadError, task_ids_with_event

GATE = EventType.zero_progress_requeue


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / 'runs.db'


def test_each_task_id_from_every_run_is_read_once(db):
    EventStore(db, run_id='earlier-run').emit(GATE, task_id='10', data={})
    current = EventStore(db, run_id='this-run')
    current.emit(GATE, task_id='10', data={})
    current.emit(GATE, task_id='11', data={})
    current.emit(GATE, data={})
    current.emit(EventType.zero_progress_requeue_recovered, task_id='12', data={})

    assert task_ids_with_event(db, GATE) == frozenset({'10', '11'})


def test_a_store_without_matching_rows_reads_as_empty(db):
    EventStore(db, run_id='this-run').emit(EventType.zero_progress_requeue_recovered, task_id='12', data={})

    assert task_ids_with_event(db, GATE) == frozenset()


@pytest.mark.parametrize('spelling', [Path, str])
def test_a_missing_store_raises_naming_it_and_is_not_created(db, spelling):
    with pytest.raises(EventReadError, match=re.escape(str(db))):
        task_ids_with_event(spelling(db), GATE)

    assert list(db.parent.iterdir()) == []


def test_a_corrupt_store_raises(db):
    EventStore(db, run_id='this-run')
    db.write_bytes(b'not a database ' * 64)

    with pytest.raises(EventReadError, match='not a database'):
        task_ids_with_event(db, GATE)


def test_a_store_without_the_events_table_raises(db):
    sqlite3.connect(db).close()

    with pytest.raises(EventReadError, match='no such table'):
        task_ids_with_event(db, GATE)
