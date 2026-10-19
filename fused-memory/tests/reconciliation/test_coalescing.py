"""Tests for the push-time coalescing rule (task 4278, recon lever 3).

Covers `fused_memory.reconciliation.coalescing`: the one task-id normaliser,
which buffered task events share a fold key, and how two folded payloads merge.
These are pure functions; buffer behaviour lives in
test_event_buffer_coalescing.py.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.coalescing import coalesce_key, task_id_key


def _event(event_type: EventType, payload: dict) -> ReconciliationEvent:
    return ReconciliationEvent(
        id=str(uuid.uuid4()),
        type=event_type,
        source=EventSource.agent,
        project_id='dark_factory',
        timestamp=datetime(2026, 10, 19, 12, 0, tzinfo=UTC),
        payload=payload,
        agent_id=None,
    )


# ── task_id_key ────────────────────────────────────────────────────────


@pytest.mark.parametrize(('value', 'expected'), [('123', '123'), (123, '123')])
def test_task_id_key_merges_int_and_str_forms(value, expected):
    assert task_id_key(value) == expected


@pytest.mark.parametrize(
    'value',
    ['', None, True, False, 7.0, ['7'], {'a': 1}],
    ids=['empty', 'none', 'true', 'false', 'float', 'list', 'dict'],
)
def test_task_id_key_rejects_non_ids(value):
    assert task_id_key(value) is None


# ── coalesce_key ───────────────────────────────────────────────────────


def test_status_change_keys_on_task_id():
    event = _event(EventType.task_status_changed, {'task_id': '123'})
    assert coalesce_key(event) == 'task_status_changed:123'


def test_status_change_int_and_str_task_ids_share_a_key():
    as_int = _event(EventType.task_status_changed, {'task_id': 123})
    as_str = _event(EventType.task_status_changed, {'task_id': '123'})
    assert coalesce_key(as_int) == coalesce_key(as_str) == 'task_status_changed:123'


def test_update_task_modification_keys_on_task_id_and_operation():
    event = _event(EventType.task_modified, {'task_id': '123', 'operation': 'update_task'})
    assert coalesce_key(event) == 'task_modified:123:update_task'


@pytest.mark.parametrize(
    'payload',
    [
        {'task_id': '123', 'operation': 'add_dependency', 'depends_on': '9'},
        {'task_id': '123', 'operation': 'remove_dependency', 'depends_on': '9'},
        {'task_id': '123'},
    ],
    ids=['add_dependency', 'remove_dependency', 'no_operation'],
)
def test_other_task_modifications_never_fold(payload):
    assert coalesce_key(_event(EventType.task_modified, payload)) is None


@pytest.mark.parametrize(
    'payload',
    [{}, {'task_id': None}, {'task_id': ''}, {'task_id': True}],
    ids=['missing', 'none', 'empty', 'bool'],
)
def test_status_change_without_a_task_id_never_folds(payload):
    assert coalesce_key(_event(EventType.task_status_changed, payload)) is None


#: The key every EventType gets for a payload that carries everything a fold
#: key could read.  Listing EVERY member (the coverage test below enforces it)
#: forces whoever adds an EventType to decide whether it folds.
_KEY_FOR_FULL_PAYLOAD: dict[EventType, str | None] = {
    EventType.task_status_changed: 'task_status_changed:7',
    EventType.task_modified: 'task_modified:7:update_task',
    EventType.task_created: None,
    EventType.task_deleted: None,
    EventType.deferral_restamped: None,
    EventType.memory_added: None,
    EventType.memory_updated: None,
    EventType.memory_deleted: None,
    EventType.episode_added: None,
}


def test_every_event_type_has_a_decided_fold_key():
    assert set(_KEY_FOR_FULL_PAYLOAD) == set(EventType)


@pytest.mark.parametrize('event_type', list(EventType), ids=lambda t: t.value)
def test_only_status_changes_and_task_updates_fold(event_type):
    event = _event(event_type, {'task_id': '7', 'operation': 'update_task'})
    assert coalesce_key(event) == _KEY_FOR_FULL_PAYLOAD.get(event_type, 'UNDECIDED')
