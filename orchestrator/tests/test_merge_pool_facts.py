"""The bounds of the merge lane's pick records (task 9908).

A pick record is capped in what it lists and how many are held, says how many
waiting requests it left out, and never lets a failure reach the pick.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

from orchestrator.merge_pool_facts import (
    PICK_RECORD_CAP,
    WAITING_CAP,
    PickRecords,
    queued_request_fields,
)
from orchestrator.suffix_graph import EMPTY_SUFFIX_CONFLICT_GRAPH
from orchestrator.task_priority import TaskPriority

_NOW = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)


def _req(n: int) -> Any:
    return SimpleNamespace(
        request_id=f'mr-{n}', task_id=str(n), lane='normal',
        first_enqueued_at=float(n),
    )


def _records() -> PickRecords:
    return PickRecords(
        lanes=('high', 'normal'), is_reserved=lambda rid: False,
        is_lane_halted=lambda lane: False, task_priority=None,
    )


def test_a_long_waiting_list_is_capped_and_counts_what_it_left_out() -> None:
    records = _records()
    waiting = [_req(n) for n in range(1, WAITING_CAP + 6)]

    records.record(_req(0), 'normal', {'high': [], 'normal': waiting}, EMPTY_SUFFIX_CONFLICT_GRAPH)

    record: dict[str, Any] = records.take('mr-0')
    assert len(record['waiting']) == WAITING_CAP
    assert record['waiting_dropped'] == 5
    assert record['waiting'][0]['first_enqueued_at'] == 1.0


def test_unclaimed_records_are_forgotten_oldest_first() -> None:
    records = _records()
    for n in range(PICK_RECORD_CAP + 1):
        records.record(_req(n), 'normal', {'high': [], 'normal': []}, EMPTY_SUFFIX_CONFLICT_GRAPH)

    assert records.take('mr-0') == {}
    assert records.take(f'mr-{PICK_RECORD_CAP}')['request_id'] == f'mr-{PICK_RECORD_CAP}'


def test_a_record_is_taken_once() -> None:
    records = _records()
    records.record(_req(1), 'high', {'high': [], 'normal': []}, EMPTY_SUFFIX_CONFLICT_GRAPH)

    assert records.take('mr-1')['lane'] == 'high'
    assert records.take('mr-1') == {}


def test_a_failure_while_recording_leaves_no_record_and_does_not_raise() -> None:
    records = _records()
    broken_graph: Any = SimpleNamespace()

    records.record(_req(1), 'normal', {'high': [], 'normal': [_req(2)]}, broken_graph)

    assert records.take('mr-1') == {}


def _queued_req(*, waited: timedelta) -> Any:
    return SimpleNamespace(
        request_id='mr-q', lane='normal', snapshot_tip='abc123',
        first_enqueued_at=(_NOW - waited).timestamp(),
    )


def test_merge_queued_keeps_the_keys_it_had() -> None:
    req = _queued_req(waited=timedelta(seconds=5))

    fields = queued_request_fields(req, TaskPriority(tier='high', pinned=False), now=_NOW)

    assert {k: fields[k] for k in (
        'request_id', 'lane', 'tier', 'pinned', 'head_sha', 'first_enqueued_at',
    )} == {
        'request_id': 'mr-q', 'lane': 'normal', 'tier': 'high', 'pinned': False,
        'head_sha': 'abc123', 'first_enqueued_at': req.first_enqueued_at,
    }


def test_without_an_anchor_the_age_is_from_the_first_enqueue() -> None:
    req = _queued_req(waited=timedelta(seconds=3600.5))
    priority = TaskPriority(tier='pinned', pinned=True, base_tier='low', pin_order=1)

    fields = queued_request_fields(req, priority, now=_NOW)

    assert (fields['base_tier'], fields['age_from'], fields['age_secs']) == (
        'low', 'enqueue', 3600.5,
    )


def test_an_anchor_measures_the_age_from_the_anchor() -> None:
    req = _queued_req(waited=timedelta(seconds=5))
    priority = TaskPriority(tier='low', base_tier='low', age_anchor=_NOW - timedelta(hours=2))

    fields = queued_request_fields(req, priority, now=_NOW)

    assert (fields['age_from'], fields['age_secs']) == ('anchor', 7200.0)
