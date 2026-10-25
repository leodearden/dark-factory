"""The offline lane's escalation records on a real queue."""

from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from escalation.queue import EscalationQueue

from orchestrator.offline_lane_escalations import (
    LANE_ESCALATION_ROLE,
    OFFLINE_LANE_SENTINEL_PREFIX,
    STUCK_FILING_CATEGORY,
    UNLANDED_RED_CATEGORY,
    StuckFiling,
    UnlandedRed,
    amend_stuck_filing,
    escalation_state,
    file_stuck_filing,
    file_unlanded_red,
    resolve_stuck_filing,
    sentinel_task_id,
)

COMMAND = 'Nightly Deep'
FILED_AT = datetime(2026, 10, 1, 13, 21, 4, tzinfo=UTC)
DUE_SINCE = datetime(2026, 10, 2, 9, 0, 0, tzinfo=UTC)


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'queue')


RED = UnlandedRed(
    command=COMMAND,
    covering_task_id='7123',
    task_status='in-progress',
    nodes=('tests/test_a.py::test_one', 'tests/test_b.py::test_two'),
    filed_at=FILED_AT,
    record_dirs=('/data/orchestrator/offline-lane/nightly-deep/p1',),
)
STUCK = StuckFiling(
    command=COMMAND,
    nodes=('tests/test_a.py::test_one',),
    due_since=DUE_SINCE,
    reason='planning mode refused: server busy',
)


def _red(**overrides: object) -> UnlandedRed:
    return replace(RED, **overrides)


def _stuck(**overrides: object) -> StuckFiling:
    return replace(STUCK, **overrides)


def _record(queue: EscalationQueue, escalation_id: str):
    record = queue.get(escalation_id)
    assert record is not None
    return record


def test_sentinel_is_the_prefix_and_the_command_slug():
    assert sentinel_task_id(COMMAND) == OFFLINE_LANE_SENTINEL_PREFIX + 'nightly-deep'
    assert sentinel_task_id('a/b c') == '__offline_lane__a-b-c'


def test_unlanded_red_is_a_blocking_level_one_record_on_the_sentinel(queue: EscalationQueue):
    escalation_id = file_unlanded_red(queue, _red())

    record = _record(queue, escalation_id)
    assert record.level == 1
    assert record.severity == 'blocking'
    assert record.category == UNLANDED_RED_CATEGORY == 'dependency_discovered'
    assert record.agent_role == LANE_ESCALATION_ROLE == 'orchestrator-offline-lane'
    assert record.task_id == '__offline_lane__nightly-deep'
    assert record.status == 'pending'
    assert escalation_id.startswith('esc-__offline_lane__nightly-deep-')


def test_unlanded_red_names_the_task_status_nodes_time_and_record_dirs(queue: EscalationQueue):
    record = _record(queue, file_unlanded_red(queue, _red()))

    assert '7123' in record.summary and 'in-progress' in record.summary
    assert len(record.summary) <= 200
    for expected in (
        '7123',
        'in-progress',
        'tests/test_a.py::test_one',
        'tests/test_b.py::test_two',
        FILED_AT.isoformat(),
        '/data/orchestrator/offline-lane/nightly-deep/p1',
    ):
        assert expected in record.detail


def test_unlanded_red_summary_is_capped_at_two_hundred_characters(queue: EscalationQueue):
    record = _record(queue, file_unlanded_red(queue, _red(task_status='x' * 400)))

    assert len(record.summary) == 200


def test_two_unlanded_records_on_one_sentinel_are_both_pending_and_distinct(
    queue: EscalationQueue,
):
    first = file_unlanded_red(queue, _red(covering_task_id='7123'))
    second = file_unlanded_red(queue, _red(covering_task_id='7124'))

    assert first != second
    assert escalation_state(queue, first) == 'pending'
    assert escalation_state(queue, second) == 'pending'
    assert {e.id for e in queue.get_pending()} == {first, second}


def test_queue_errors_propagate(tmp_path: Path):
    queue_dir = tmp_path / 'queue'
    queue = EscalationQueue(queue_dir)
    shutil.rmtree(queue_dir)
    queue_dir.write_text('not a directory')

    with pytest.raises(OSError):
        file_unlanded_red(queue, _red())


def test_stuck_filing_is_a_blocking_level_one_infra_record_on_the_sentinel(
    queue: EscalationQueue,
):
    record = _record(queue, file_stuck_filing(queue, _stuck()))

    assert record.level == 1
    assert record.severity == 'blocking'
    assert record.category == STUCK_FILING_CATEGORY == 'infra_issue'
    assert record.agent_role == LANE_ESCALATION_ROLE
    assert record.task_id == '__offline_lane__nightly-deep'
    assert 'planning mode refused: server busy' in record.summary
    assert len(record.summary) <= 200
    for expected in (DUE_SINCE.isoformat(), 'tests/test_a.py::test_one', 'server busy'):
        assert expected in record.detail


def test_amending_a_stuck_filing_with_a_changed_reason_records_it(queue: EscalationQueue):
    escalation_id = file_stuck_filing(queue, _stuck())

    amend_stuck_filing(queue, escalation_id, _stuck(reason='the task status is unreadable'))

    record = _record(queue, escalation_id)
    assert len(record.amendments) == 1
    assert 'the task status is unreadable' in record.amendments[0]['detail']
    assert record.status == 'pending'


def test_amending_a_stuck_filing_with_identical_text_is_a_no_op(queue: EscalationQueue):
    escalation_id = file_stuck_filing(queue, _stuck())
    changed = _stuck(reason='the task status is unreadable')
    amend_stuck_filing(queue, escalation_id, changed)
    amend_stuck_filing(queue, escalation_id, changed)
    amend_stuck_filing(queue, escalation_id, changed)

    assert len(_record(queue, escalation_id).amendments) == 1


def test_resolving_a_stuck_filing_is_read_back_as_resolved_by_the_lane(queue: EscalationQueue):
    escalation_id = file_stuck_filing(queue, _stuck())

    resolve_stuck_filing(queue, escalation_id)

    assert escalation_state(queue, escalation_id) == 'resolved'
    assert _record(queue, escalation_id).resolved_by == LANE_ESCALATION_ROLE
    assert queue.get_pending() == []


def test_state_of_an_unknown_escalation_is_none(queue: EscalationQueue):
    assert escalation_state(queue, 'esc-__offline_lane__nightly-deep-99') is None
