"""file_or_refresh over a real queue: one pending record per sentinel and level, refreshed in place."""

from __future__ import annotations

import dataclasses
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from escalation.categories import EscalationCategory, UnregisteredEscalationCategory
from escalation.file_or_refresh import RefreshableAlarm, file_or_refresh
from escalation.models import Escalation
from escalation.queue import EscalationQueue

ALARM = RefreshableAlarm(
    task_id='file-or-refresh-test',
    level=2,
    agent_role='orchestrator-test-guard',
    severity='critical',
    category=EscalationCategory.CLEANUP_NEEDED,
    suggested_action='nothing; test fixture',
)
L1_ALARM = RefreshableAlarm(
    task_id=ALARM.task_id,
    level=1,
    agent_role='orchestrator-test-guard',
    severity='blocking',
    category=EscalationCategory.INFRA_ISSUE,
    suggested_action='nothing; test fixture',
)


class _SubmitFailsQueue(EscalationQueue):
    def submit(self, escalation: Escalation) -> str:
        raise OSError('disk full')


class _ResolvedAfterLookupQueue(EscalationQueue):
    def get_by_task(
        self, task_id: str, status: str | None = None, level: int | None = None,
        agent_role: str | None = None,
    ) -> list[Escalation]:
        found = super().get_by_task(task_id, status=status, level=level, agent_role=agent_role)
        if found:
            self.resolve(found[0].id, 'resolved in the gap', resolved_by='test')
        return found


class _LookupBarrierQueue(EscalationQueue):
    def __init__(self, queue_dir: Path, parties: int) -> None:
        super().__init__(queue_dir)
        self._after_lookup = threading.Barrier(parties, timeout=30)

    def get_by_task(
        self, task_id: str, status: str | None = None, level: int | None = None,
        agent_role: str | None = None,
    ) -> list[Escalation]:
        found = super().get_by_task(task_id, status=status, level=level, agent_role=agent_role)
        self._after_lookup.wait()
        return found


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


def _pending(queue: EscalationQueue, level: int = ALARM.level) -> list[Escalation]:
    return queue.get_by_task(ALARM.task_id, status='pending', level=level)


class TestFirstCallFiles:
    def test_files_one_pending_record_from_the_alarm_and_the_call(self, queue):
        filed = file_or_refresh(
            queue, ALARM, summary='first summary', detail='first detail',
            filing_claimant_run_id='run-a',
        )

        assert filed.id == 'esc-file-or-refresh-test-1'
        assert filed.task_id == ALARM.task_id
        assert filed.level == ALARM.level
        assert filed.agent_role == ALARM.agent_role
        assert filed.severity == ALARM.severity
        assert filed.category == ALARM.category
        assert filed.suggested_action == ALARM.suggested_action
        assert filed.summary == 'first summary'
        assert filed.detail == 'first detail'
        assert filed.filing_claimant_run_id == 'run-a'
        assert filed.status == 'pending'
        assert filed.dedupe_count == 0
        assert filed == queue.get(filed.id)


class TestRepeatCallRefreshes:
    def test_refreshes_the_pending_record_in_place_and_keeps_the_first_filer(self, queue):
        first = file_or_refresh(
            queue, ALARM, summary='first summary', detail='first detail',
            filing_claimant_run_id='run-a',
        )
        file_or_refresh(
            queue, ALARM, summary='second summary', detail='second detail',
            filing_claimant_run_id='run-b',
        )

        pending = _pending(queue)
        assert len(pending) == 1
        refreshed = pending[0]
        assert refreshed.id == first.id
        assert refreshed.summary == 'second summary'
        assert refreshed.detail == 'second detail'
        assert refreshed.dedupe_count == 1
        assert refreshed.timestamp >= first.timestamp
        assert refreshed.filing_claimant_run_id == 'run-a'
        assert refreshed.agent_role == ALARM.agent_role
        assert refreshed.severity == ALARM.severity
        assert refreshed.category == ALARM.category
        assert refreshed.suggested_action == ALARM.suggested_action

    def test_a_refresh_mints_no_id(self, queue):
        for summary in ('first', 'second'):
            file_or_refresh(
                queue, ALARM, summary=summary, detail='detail',
                filing_claimant_run_id='run-a',
            )

        assert queue.make_id(ALARM.task_id) == 'esc-file-or-refresh-test-2'


class TestOnlyAPendingRecordAtTheAlarmLevelIsRefreshed:
    def test_a_pending_record_at_another_level_is_left_alone(self, queue):
        l1 = file_or_refresh(
            queue, L1_ALARM, summary='l1 summary', detail='l1 detail',
            filing_claimant_run_id='run-a',
        )
        file_or_refresh(
            queue, ALARM, summary='l2 summary', detail='l2 detail',
            filing_claimant_run_id='run-a',
        )

        assert len(queue.get_by_task(ALARM.task_id, status='pending')) == 2
        untouched = queue.get(l1.id)
        assert untouched is not None
        assert untouched.dedupe_count == 0
        assert untouched.summary == 'l1 summary'

    def test_a_record_no_longer_pending_is_not_refreshed(self, queue):
        first = file_or_refresh(
            queue, ALARM, summary='first', detail='detail', filing_claimant_run_id='run-a',
        )
        queue.resolve(first.id, 'fixed', resolved_by='test')

        second = file_or_refresh(
            queue, ALARM, summary='second', detail='detail', filing_claimant_run_id='run-a',
        )

        assert second.id == 'esc-file-or-refresh-test-2'
        assert second.dedupe_count == 0
        assert [esc.id for esc in _pending(queue)] == [second.id]
        resolved = queue.get(first.id)
        assert resolved is not None
        assert resolved.status == 'resolved'


class TestARefreshCannotRaceTheRecord:
    def test_a_record_resolved_after_the_lookup_stays_resolved_and_the_alarm_is_filed_afresh(
        self, queue, tmp_path,
    ):
        first = file_or_refresh(
            queue, ALARM, summary='first', detail='detail', filing_claimant_run_id='run-a',
        )
        gap_queue = _ResolvedAfterLookupQueue(tmp_path / 'escalations')

        second = file_or_refresh(
            gap_queue, ALARM, summary='second', detail='detail', filing_claimant_run_id='run-b',
        )

        assert second.id == 'esc-file-or-refresh-test-2'
        assert second.dedupe_count == 0
        assert second.status == 'pending'
        assert second.filing_claimant_run_id == 'run-b'
        resolved = queue.get(first.id)
        assert resolved is not None
        assert resolved.status == 'resolved'
        assert [esc.id for esc in queue.get_pending()] == [second.id]

    def test_concurrent_refreshes_each_count(self, queue, tmp_path):
        file_or_refresh(
            queue, ALARM, summary='first', detail='detail', filing_claimant_run_id='run-a',
        )
        barrier_queue = _LookupBarrierQueue(tmp_path / 'escalations', parties=2)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    file_or_refresh, barrier_queue, ALARM, summary=f'refresh {n}',
                    detail='detail', filing_claimant_run_id='run-b',
                )
                for n in range(2)
            ]
            for future in futures:
                future.result()

        pending = _pending(queue)
        assert len(pending) == 1
        assert pending[0].dedupe_count == 2


class TestFailures:
    def test_an_unregistered_category_is_refused_at_construction(self):
        with pytest.raises(UnregisteredEscalationCategory):
            dataclasses.replace(ALARM, category='not_a_category')

    def test_a_level_0_alarm_is_refused_at_construction(self):
        with pytest.raises(ValueError, match='level=0'):
            dataclasses.replace(ALARM, level=0)

    def test_a_failing_submit_propagates(self, tmp_path):
        queue = _SubmitFailsQueue(tmp_path / 'escalations')

        with pytest.raises(OSError, match='disk full'):
            file_or_refresh(
                queue, ALARM, summary='summary', detail='detail',
                filing_claimant_run_id='run-a',
            )
