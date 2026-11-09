"""Tests for the shared deduped born-at-L2 filer (task 2987 amendment)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from escalation.categories import EscalationCategory
from escalation.queue import EscalationQueue

from orchestrator.born_at_l2 import BornAtL2Alarm, file_deduped_born_at_l2
from orchestrator.event_store import EventStore, EventType

_ALARM = BornAtL2Alarm(
    root_cause='probe_condition',
    sentinel_task='__probe_condition__',
    role='orchestrator-probe-condition',
    category=EscalationCategory.INFRA_ISSUE,
    suggested_action='investigate_probe_condition',
    created_reason='probe-condition-threshold',
)


@pytest.fixture
def queue(tmp_path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


@pytest.fixture
def event_store() -> MagicMock:
    return MagicMock(spec=EventStore)


def _file(queue, event_store=None, *, summary='probe tripped', detail='probe detail'):
    return file_deduped_born_at_l2(
        queue, _ALARM, summary=summary, detail=detail,
        event_store=event_store, filing_claimant_run_id='run-7',
    )


def _logged_warning_for(caplog, error: BaseException) -> bool:
    return any(
        r.levelname == 'WARNING' and r.exc_info is not None and r.exc_info[1] is error
        for r in caplog.records
    )


def test_files_an_urgent_level_two_escalation_for_a_human(queue, event_store):
    escalation = _file(queue, event_store)

    assert escalation is not None
    (pending,) = queue.get_pending()
    assert pending.id == escalation.id
    assert (pending.level, pending.severity) == (2, 'urgent')
    assert pending.task_id == _ALARM.sentinel_task
    assert pending.agent_role == _ALARM.role
    assert pending.category == _ALARM.category
    assert pending.category == 'infra_issue'
    assert pending.root_cause == _ALARM.root_cause
    assert pending.suggested_action == _ALARM.suggested_action
    assert (pending.summary, pending.detail) == ('probe tripped', 'probe detail')
    assert pending.filing_claimant_run_id == 'run-7'


def test_emits_escalation_created_with_the_alarm_reason(queue, event_store):
    escalation = _file(queue, event_store)

    assert escalation is not None
    event_store.emit.assert_called_once()
    (event_type,) = event_store.emit.call_args.args
    assert event_type is EventType.escalation_created
    assert event_store.emit.call_args.kwargs['task_id'] == _ALARM.sentinel_task
    data = event_store.emit.call_args.kwargs['data']
    assert data['escalation_id'] == escalation.id
    assert data['reason'] == _ALARM.created_reason


def test_long_summary_is_cut_to_one_line_limit(queue):
    escalation = _file(queue, summary='x' * 500)

    assert escalation is not None
    assert len(escalation.summary) == 200


def test_pending_l2_with_the_same_root_cause_dedups(queue, event_store):
    _file(queue)

    assert _file(queue, event_store) is None
    assert len(queue.get_pending()) == 1
    event_store.emit.assert_not_called()


def test_refiles_once_a_human_resolved_the_pending_one(queue):
    first = _file(queue)
    assert first is not None
    queue.resolve(first.id, 'investigated')

    second = _file(queue)

    assert second is not None and second.id != first.id
    assert [e.id for e in queue.get_pending()] == [second.id]


def test_no_queue_is_a_no_op(event_store):
    assert _file(None, event_store) is None
    event_store.emit.assert_not_called()


def test_files_without_an_event_store(queue):
    assert _file(queue, event_store=None) is not None
    assert len(queue.get_pending()) == 1


def test_submit_failure_is_logged_not_raised(event_store, caplog):
    queue = MagicMock(spec=EscalationQueue)
    queue.find_pending_l2_by_root_cause.return_value = None
    queue.make_id.return_value = 'esc-probe-1'
    queue.submit.side_effect = RuntimeError('queue down')

    with caplog.at_level('WARNING'):
        assert _file(queue, event_store) is None

    assert _logged_warning_for(caplog, queue.submit.side_effect)
    event_store.emit.assert_not_called()


def test_emit_failure_still_returns_the_filed_escalation(queue, event_store, caplog):
    event_store.emit.side_effect = RuntimeError('events db locked')

    with caplog.at_level('WARNING'):
        escalation = _file(queue, event_store)

    assert escalation is not None
    assert [e.id for e in queue.get_pending()] == [escalation.id]
    assert _logged_warning_for(caplog, event_store.emit.side_effect)
