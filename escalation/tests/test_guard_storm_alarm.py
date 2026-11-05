"""file_storm_once over a real queue: one open alarm per burst, never a dropped one.

Both guards' servers exercise the filer end to end
(``test_markup_middleware_registration.py`` and
``test_uuid_prefix_guard_registration.py``). This file covers the one branch no
server reaches: a queue read that fails.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from escalation import guard_storm_alarm
from escalation.categories import EscalationCategory
from escalation.guard_storm_alarm import GuardStormAlarm, file_storm_once
from escalation.models import Escalation
from escalation.queue import EscalationQueue

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


class _UnreadableQueue(EscalationQueue):
    """A real queue whose pending-record read fails, as an unreadable queue dir would."""

    def get_by_task(
        self, task_id: str, status: str | None = None, level: int | None = None,
        agent_role: str | None = None,
    ) -> list[Escalation]:
        raise OSError(f'cannot list pending records for {task_id}')


def _pending_alarms(queue_dir: Path) -> list[Escalation]:
    return EscalationQueue(queue_dir).get_by_task(ALARM.anchor_task_id, status='pending')


def test_a_second_fire_folds_into_the_open_alarm(tmp_path: Path):
    queue = EscalationQueue(tmp_path)

    first = file_storm_once(queue, ALARM, RECORD)
    second = file_storm_once(queue, ALARM, RECORD)

    assert second == first
    assert [esc.id for esc in _pending_alarms(tmp_path)] == [first]


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
