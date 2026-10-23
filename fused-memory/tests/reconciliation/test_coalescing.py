"""Tests for the push-time coalescing rule (task 4278, recon lever 3).

Covers `fused_memory.reconciliation.coalescing`: the one task-id normaliser,
which buffered task events share a fold key, and how two folded payloads merge.
These are pure functions; buffer behaviour lives in
test_event_buffer_coalescing.py.
"""

from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.coalescing import (
    COALESCED_COUNT_KEY,
    COALESCED_LAST_ID_KEY,
    COALESCED_LAST_TIMESTAMP_KEY,
    COALESCED_STATUS_PATH_KEY,
    COALESCED_STATUS_PATH_TRUNCATED_KEY,
    MAX_COALESCED_STATUS_PATH,
    CoalescedFold,
    coalesce_key,
    merge_coalesced_payload,
    task_id_key,
)


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


# ── merge_coalesced_payload ────────────────────────────────────────────

_T0 = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)


def _at(minutes: int) -> datetime:
    return _T0 + timedelta(minutes=minutes)


def _status(old: str, new: str, **extra) -> dict:
    return {'task_id': '644', 'old_status': old, 'new_status': new,
            '_project_root': '/p', **extra}


def _fold_status(
    existing: dict, incoming: dict, *, incoming_ts: datetime,
    row_ts: datetime = _T0, incoming_id: str | None = None,
) -> CoalescedFold:
    fold = merge_coalesced_payload(
        existing, incoming,
        event_type=EventType.task_status_changed,
        incoming_id=incoming_id or str(uuid.uuid4()),
        incoming_ts=incoming_ts,
        row_ts=row_ts,
    )
    assert fold is not None
    return fold


def test_newer_status_fold_rewinds_old_status_and_advances_new_status():
    incoming_ts = _at(5)
    fold = merge_coalesced_payload(
        _status('pending', 'in-progress'), _status('in-progress', 'done'),
        event_type=EventType.task_status_changed,
        incoming_id='evt-b', incoming_ts=incoming_ts, row_ts=_T0,
    )
    assert fold is not None
    assert fold.payload['old_status'] == 'pending'
    assert fold.payload['new_status'] == 'done'
    assert fold.payload[COALESCED_COUNT_KEY] == 2
    assert fold.payload[COALESCED_LAST_ID_KEY] == 'evt-b'
    assert fold.payload[COALESCED_LAST_TIMESTAMP_KEY] == incoming_ts.isoformat()
    assert fold.payload[COALESCED_STATUS_PATH_KEY] == ['pending', 'in-progress', 'done']
    assert fold.row_timestamp == _T0
    assert fold.folded_timestamp == incoming_ts


def test_a_third_newer_fold_appends_one_path_entry():
    second = _fold_status(
        _status('pending', 'in-progress'), _status('in-progress', 'done'),
        incoming_ts=_at(5),
    )
    third = _fold_status(second.payload, _status('done', 'review'), incoming_ts=_at(9))
    assert third.payload[COALESCED_COUNT_KEY] == 3
    assert third.payload[COALESCED_STATUS_PATH_KEY] == [
        'pending', 'in-progress', 'done', 'review']
    assert third.payload['new_status'] == 'review'


def test_redelivery_of_the_last_folded_event_is_not_a_fold():
    folded = _fold_status(
        _status('pending', 'in-progress'), _status('in-progress', 'done'),
        incoming_ts=_at(5), incoming_id='evt-b',
    )
    assert merge_coalesced_payload(
        folded.payload, _status('in-progress', 'done'),
        event_type=EventType.task_status_changed,
        incoming_id='evt-b', incoming_ts=_at(5), row_ts=_T0,
    ) is None


def test_late_older_arrival_takes_old_status_and_moves_the_row_earlier():
    row_ts = _at(10)
    older_ts = _at(2)
    existing = _status('in-progress', 'done', done_provenance={'kind': 'merged'})
    fold = _fold_status(
        existing, _status('pending', 'in-progress', _project_root='/older'),
        incoming_ts=older_ts, row_ts=row_ts,
    )
    assert fold.payload['old_status'] == 'pending'
    assert fold.payload['new_status'] == 'done'
    assert fold.payload['done_provenance'] == {'kind': 'merged'}
    assert fold.payload['_project_root'] == '/p'
    assert fold.payload[COALESCED_STATUS_PATH_KEY] == ['pending', 'in-progress', 'done']
    assert fold.payload[COALESCED_COUNT_KEY] == 2
    assert fold.row_timestamp == older_ts
    assert fold.folded_timestamp == row_ts


def test_middle_arrival_only_bumps_the_count():
    two = _fold_status(
        _status('pending', 'in-progress'), _status('blocked', 'done'),
        incoming_ts=_at(10), incoming_id='evt-newest',
    )
    middle_ts = _at(4)
    three = _fold_status(two.payload, _status('in-progress', 'blocked'), incoming_ts=middle_ts)
    assert three.payload[COALESCED_COUNT_KEY] == 3
    for key in ('old_status', 'new_status', COALESCED_STATUS_PATH_KEY,
                COALESCED_LAST_ID_KEY, COALESCED_LAST_TIMESTAMP_KEY):
        assert three.payload[key] == two.payload[key], key
    assert three.row_timestamp == _T0
    assert three.folded_timestamp == middle_ts


def test_status_path_is_capped_and_flagged_but_new_status_stays_newest():
    payload = _status('s0', 's1')
    folds = MAX_COALESCED_STATUS_PATH + 5
    for n in range(1, folds + 1):
        payload = _fold_status(payload, _status(f's{n}', f's{n + 1}'),
                               incoming_ts=_at(n)).payload
    assert len(payload[COALESCED_STATUS_PATH_KEY]) == MAX_COALESCED_STATUS_PATH
    assert payload[COALESCED_STATUS_PATH_TRUNCATED_KEY] is True
    assert payload['new_status'] == f's{folds + 1}'
    assert payload[COALESCED_COUNT_KEY] == folds + 1


def test_an_uncapped_path_carries_no_truncation_flag():
    fold = _fold_status(
        _status('pending', 'in-progress'), _status('in-progress', 'done'),
        incoming_ts=_at(5),
    )
    assert COALESCED_STATUS_PATH_TRUNCATED_KEY not in fold.payload


def test_producer_keys_come_from_the_newest_constituent_only():
    reopened = _fold_status(
        _status('pending', 'done', done_provenance={'kind': 'merged'}),
        _status('done', 'pending', reopen_reason='regressed', reopen_from='done'),
        incoming_ts=_at(5),
    )
    assert reopened.payload['reopen_reason'] == 'regressed'
    assert reopened.payload['reopen_from'] == 'done'
    assert 'done_provenance' not in reopened.payload

    undeferred = _fold_status(
        _status('pending', 'deferred', deferral={'until': '2026-11-01'}),
        _status('deferred', 'pending', deferral_cleared={'by': 'gate'}),
        incoming_ts=_at(5),
    )
    assert undeferred.payload['deferral_cleared'] == {'by': 'gate'}
    assert 'deferral' not in undeferred.payload


def test_task_update_fold_takes_newest_producer_keys_and_records_no_path():
    existing = {'task_id': '636', 'operation': 'update_task', '_project_root': '/old'}
    incoming = {'task_id': '636', 'operation': 'update_task', '_project_root': '/new'}
    fold = merge_coalesced_payload(
        existing, incoming,
        event_type=EventType.task_modified,
        incoming_id='evt-b', incoming_ts=_at(5), row_ts=_T0,
    )
    assert fold is not None
    assert fold.payload['_project_root'] == '/new'
    assert fold.payload[COALESCED_COUNT_KEY] == 2
    assert COALESCED_STATUS_PATH_KEY not in fold.payload
    assert COALESCED_STATUS_PATH_TRUNCATED_KEY not in fold.payload


def test_merge_does_not_mutate_the_existing_payload():
    existing = _fold_status(
        _status('pending', 'in-progress'), _status('in-progress', 'done'),
        incoming_ts=_at(5),
    ).payload
    before = copy.deepcopy(existing)
    _fold_status(existing, _status('done', 'review'), incoming_ts=_at(9))
    _fold_status(existing, _status('draft', 'pending'), incoming_ts=_at(-5))
    assert existing == before
