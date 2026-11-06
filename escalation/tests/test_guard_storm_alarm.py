"""file_storm_once over a real queue: one open alarm per burst, never a dropped one.

Both guards' servers exercise the filer end to end
(``test_markup_middleware_registration.py`` and
``test_uuid_prefix_guard_registration.py``). This file covers what no server
reaches on its own: a queue read that fails, a burst of another outcome
folding into an open alarm, and a note about it that cannot land.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from escalation import guard_storm_alarm
from escalation.categories import EscalationCategory
from escalation.guard_storm_alarm import GuardStormAlarm, file_storm_once
from escalation.models import Escalation
from escalation.queue import AmendResult, EscalationQueue

ALARM = GuardStormAlarm(
    anchor_task_id='guard-storm-alarm-test',
    category=EscalationCategory.UUID_PREFIX_STORM,
    agent_role='test-guard',
    condition='the test burst is active',
    suggested_action='nothing; this alarm is a test fixture',
)
RECORD = {
    'error_type': 'uuid_prefix_boundary_storm',
    'count': 3,
    'threshold': 3,
    'window_seconds': 3600,
    'outcome': 'resolver_unavailable',
    'project': 'reify',
}
TIMED_OUT = {**RECORD, 'reason': 'timed_out'}
UNREACHABLE = {**RECORD, 'reason': 'unreachable'}


class _UnreadableQueue(EscalationQueue):
    """A real queue whose pending-record read fails, as an unreadable queue dir would."""

    def get_by_task(
        self, task_id: str, status: str | None = None, level: int | None = None,
        agent_role: str | None = None,
    ) -> list[Escalation]:
        raise OSError(f'cannot list pending records for {task_id}')


class _UnamendableQueue(EscalationQueue):
    """A real queue whose amend fails, as an unwritable record would."""

    def amend(
        self, escalation_id: str, *, root_cause: str = '', summary: str = '',
        detail: str = '', options: list[str] | None = None, agent_role: str = '',
    ) -> AmendResult:
        raise OSError(f'cannot rewrite {escalation_id}')


class _ResolvedMeanwhileQueue(EscalationQueue):
    """A real queue whose open alarm is resolved between the pending read and the note."""

    def amend(
        self, escalation_id: str, *, root_cause: str = '', summary: str = '',
        detail: str = '', options: list[str] | None = None, agent_role: str = '',
    ) -> AmendResult:
        self.resolve(escalation_id, 'resolved by an operator meanwhile')
        return super().amend(
            escalation_id, root_cause=root_cause, summary=summary, detail=detail,
            options=options, agent_role=agent_role,
        )


def _pending_alarms(queue_dir: Path) -> list[Escalation]:
    return sorted(
        EscalationQueue(queue_dir).get_by_task(ALARM.anchor_task_id, status='pending'),
        key=lambda esc: esc.id,
    )


def _notes(alarm: Escalation) -> list[str]:
    return [note.get('summary', '') for note in alarm.amendments]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.name == guard_storm_alarm.__name__ and record.levelno == logging.WARNING
    ]


def test_a_second_fire_folds_into_the_open_alarm(tmp_path: Path):
    queue = EscalationQueue(tmp_path)

    first = file_storm_once(queue, ALARM, RECORD)
    second = file_storm_once(queue, ALARM, RECORD)

    assert second == first
    assert [esc.id for esc in _pending_alarms(tmp_path)] == [first]


def test_a_fold_of_the_outcome_the_alarm_names_adds_no_note(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.INFO, logger=guard_storm_alarm.__name__)
    queue = EscalationQueue(tmp_path)

    file_storm_once(queue, ALARM, TIMED_OUT)
    file_storm_once(queue, ALARM, TIMED_OUT)

    [alarm] = _pending_alarms(tmp_path)
    assert _notes(alarm) == []
    assert _warnings(caplog) == []


def test_a_fold_of_another_outcome_is_noted_on_the_open_alarm_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    """A slow resolver's alarm must not keep saying "slow" once it is down."""
    caplog.set_level(logging.INFO, logger=guard_storm_alarm.__name__)
    queue = EscalationQueue(tmp_path)
    first = file_storm_once(queue, ALARM, TIMED_OUT)
    assert first is not None

    folded = [file_storm_once(queue, ALARM, record) for record in (UNREACHABLE, UNREACHABLE, TIMED_OUT)]

    assert folded == [first] * 3
    [alarm] = _pending_alarms(tmp_path)
    assert ' resolver_unavailable (timed_out) MCP call(s) ' in alarm.summary
    [note] = _notes(alarm)
    assert ' resolver_unavailable (unreachable) MCP call(s) ' in note
    [warning] = _warnings(caplog)
    assert first in warning.getMessage()
    assert 'resolver_unavailable (unreachable)' in warning.getMessage()


@pytest.mark.parametrize('queue_type', [_UnamendableQueue, _ResolvedMeanwhileQueue])
def test_a_new_outcome_that_cannot_be_noted_files_an_alarm_of_its_own(
    tmp_path: Path, queue_type: type[EscalationQueue],
):
    queue = queue_type(tmp_path)
    first = file_storm_once(queue, ALARM, TIMED_OUT)

    second = file_storm_once(queue, ALARM, UNREACHABLE)

    assert second is not None and second != first
    [own] = [esc for esc in _pending_alarms(tmp_path) if esc.id == second]
    assert ' resolver_unavailable (unreachable) MCP call(s) ' in own.summary


def test_the_summary_names_the_reason_when_the_record_carries_one(tmp_path: Path):
    file_storm_once(EscalationQueue(tmp_path), ALARM, TIMED_OUT)

    [alarm] = _pending_alarms(tmp_path)
    assert 'resolver_unavailable (timed_out)' in alarm.summary


def test_a_reasonless_record_keeps_its_summary(tmp_path: Path):
    file_storm_once(EscalationQueue(tmp_path), ALARM, RECORD)

    [alarm] = _pending_alarms(tmp_path)
    assert 'resolver_unavailable MCP call(s)' in alarm.summary


def test_a_failed_read_files_the_alarm_rather_than_dropping_it(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    caplog.set_level(logging.ERROR, logger=guard_storm_alarm.__name__)

    filed = file_storm_once(_UnreadableQueue(tmp_path), ALARM, RECORD)

    alarms = _pending_alarms(tmp_path)
    assert [esc.id for esc in alarms] == [filed]
    assert (alarms[0].category, alarms[0].agent_role, alarms[0].level) == (
        ALARM.category, ALARM.agent_role, 1,
    )
    errors = [
        record for record in caplog.records
        if record.name == guard_storm_alarm.__name__ and record.levelno == logging.ERROR
    ]
    assert len(errors) == 1
    assert errors[0].exc_info is not None
