"""Tests for the synchronous durable write-ahead journal (EventJournal).

EventJournal is the durable-at-enqueue store backing EventQueue: each event is
persisted synchronously (stdlib sqlite3, WAL + synchronous=FULL) BEFORE the
in-memory ``put_nowait``, so a hard kill between enqueue and drain no longer
silently drops in-flight events. These tests exercise the store in isolation;
the EventQueue-level durability/recovery behaviour is covered in
test_event_queue.py.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.event_journal import EventJournal


def _make_event(
    project_id: str = 'test-project',
    event_type: EventType = EventType.task_created,
    agent_id: str | None = 'agent-x',
) -> ReconciliationEvent:
    """Mirror of test_event_queue.py's _make_event, with a non-None agent_id
    so round-trip assertions cover the nullable column."""
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=event_type,
        source=EventSource.agent,
        project_id=project_id,
        timestamp=datetime.now(UTC),
        payload={'test': True, 'nested': {'a': 1}},
        agent_id=agent_id,
    )


def test_append_is_durable_across_reopen(tmp_path):
    """append() durably persists a row: a FRESH EventJournal opened on the SAME
    path (simulating a process restart) returns the event from load_unprocessed().
    """
    db_path = tmp_path / 'event_journal.db'
    event = _make_event()

    journal = EventJournal(db_path)
    journal.append(event)
    # Close cleanly — the durability guarantee is the commit, not the close;
    # a fresh connection over the same file must still see the row.
    journal.close()

    # Simulate a process restart: brand-new EventJournal over the same file.
    reopened = EventJournal(db_path)
    try:
        recovered = reopened.load_unprocessed()
    finally:
        reopened.close()

    assert len(recovered) == 1
    got = recovered[0]
    # Round-trip every load-bearing field.
    assert got.id == event.id
    assert got.type == event.type
    assert got.source == event.source
    assert got.project_id == event.project_id
    assert got.timestamp == event.timestamp
    assert got.agent_id == event.agent_id
    assert got.payload == event.payload


def test_mark_processed_removes_only_that_row(tmp_path):
    """mark_processed(id) removes exactly that event; others remain unprocessed.

    And mark_processed on an unknown id is a harmless no-op.
    """
    db_path = tmp_path / 'event_journal.db'
    journal = EventJournal(db_path)
    try:
        first = _make_event()
        second = _make_event()
        journal.append(first)
        journal.append(second)

        # Unknown id → no-op (does not raise, removes nothing).
        journal.mark_processed('nonexistent-id')
        assert {e.id for e in journal.load_unprocessed()} == {first.id, second.id}

        # Processing the first removes only its row.
        journal.mark_processed(first.id)
        remaining = journal.load_unprocessed()
        assert [e.id for e in remaining] == [second.id]
    finally:
        journal.close()
