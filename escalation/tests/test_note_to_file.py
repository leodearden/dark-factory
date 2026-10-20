"""The note-to-file escalation class at both of its enforcement points: the
queue's only pending-write path and the server's filing chokepoint
(plans/plan-deviation-recording-prd.md D2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _filing_tools import call_info

from escalation.action_effects import WORKFLOW_NONE, TaskEffect, effect_for
from escalation.archive import archive_dir_for_date
from escalation.models import KNOWN_SEVERITIES, NOTE_TO_FILE_CLASS, Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server

CLAIMANT = 'run-1/sess-1/pid=7'
EVAL_FIXTURE_ID = 'df_task_2430_adv_plan'

_NOTE_FILING: dict[str, Any] = {
    'task_id': '5751',
    'agent_role': 'implementer',
    'category': 'design_concern',
    'summary': 'deviated from step-4: kept the helper private',
    'severity': 'info',
    'note_to_file': True,
}


def _note(esc_id: str = 'esc-5751-1', *, note_to_file: bool = True) -> Escalation:
    return Escalation(
        id=esc_id,
        task_id='5751',
        agent_role='implementer',
        severity='info',
        category='design_concern',
        summary='deviated from step-4: kept the helper private',
        note_to_file=note_to_file,
    )


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'queue')


class TestQueueNeverWritesANotePending:
    """queue.submit is the only path that creates a pending record, so it is
    where the never-pending invariant is enforced redundantly."""

    def test_submit_refuses_a_note_before_writing(self, queue: EscalationQueue):
        notified: list[Escalation] = []
        queue.set_notify_callback(notified.append)

        with pytest.raises(ValueError, match='never pending') as excinfo:
            queue.submit(_note('esc-5751-7'))

        assert 'esc-5751-7' in str(excinfo.value)
        assert 'submit_resolved' in str(excinfo.value)
        assert list(queue.queue_dir.rglob('esc-*.json')) == []
        assert queue.get_pending() == []
        assert notified == []

    def test_submit_resolved_files_a_note_into_the_archive(self, queue: EscalationQueue):
        record = queue.submit_resolved(
            _note(), 'r', resolved_by='x', resolution_class=NOTE_TO_FILE_CLASS,
        )

        assert record.resolved_at is not None
        archived = archive_dir_for_date(queue.queue_dir, record.resolved_at) / f'{record.id}.json'
        assert archived.is_file()
        assert queue.get_pending() == []

    def test_submit_still_queues_an_ordinary_record_pending(self, queue: EscalationQueue):
        queue.submit(_note(note_to_file=False))

        assert [esc.id for esc in queue.get_pending()] == ['esc-5751-1']


def _server(queue: EscalationQueue, *, status: str | None = 'in-progress') -> tuple[object, AsyncMock]:
    status_lookup = AsyncMock(return_value=status)
    server = create_server(
        queue,
        startup_sweep=False,
        task_status_lookup=status_lookup,
        task_claimant_lookup=AsyncMock(return_value=CLAIMANT),
    )
    return server, status_lookup


async def _file_note(queue: EscalationQueue, **overrides: Any) -> dict[str, Any]:
    server, _ = _server(queue)
    return await call_info(server, **{**_NOTE_FILING, **overrides})


class TestNoteToFileIsFiledResolved:
    @pytest.mark.asyncio
    async def test_response_and_record_are_resolved_note_to_file(self, queue):
        result = await _file_note(queue)

        assert result == {
            'id': result['id'], 'status': 'resolved',
            'resolution_class': NOTE_TO_FILE_CLASS, 'level': 0,
        }
        record = queue.get(result['id'])
        assert record is not None
        assert record.status == 'resolved'
        assert record.resolution_class == NOTE_TO_FILE_CLASS
        assert record.resolution_action == 'close_only'
        assert record.resolved_by == 'escalation-server'
        assert record.note_to_file is True

    @pytest.mark.asyncio
    async def test_record_is_archived_at_filing(self, queue):
        result = await _file_note(queue)

        record = queue.get(result['id'])
        assert record is not None and record.resolved_at is not None
        archived = archive_dir_for_date(queue.queue_dir, record.resolved_at) / f'{record.id}.json'
        assert archived.is_file()
        assert not (queue.queue_dir / f'{record.id}.json').exists()
        assert queue.get_pending() == []


class TestNeverPending:
    @pytest.mark.asyncio
    async def test_notify_callback_never_fires(self, queue):
        notified: list[Escalation] = []
        queue.set_notify_callback(notified.append)

        await _file_note(queue)

        assert notified == []
        assert queue.get_pending() == []


class TestNoTaskEffect:
    @pytest.mark.asyncio
    async def test_resolve_callback_sees_a_no_effect_action(self, queue):
        seen: list[Escalation] = []
        queue.set_resolve_callback(seen.append)

        await _file_note(queue)

        assert len(seen) == 1
        (rec,) = seen
        assert rec.resolution_action == 'close_only'
        assert effect_for(rec.resolution_action, rec.level, rec.category) == TaskEffect(None, WORKFLOW_NONE)


class TestIdentityKept:
    @pytest.mark.asyncio
    async def test_archived_note_carries_the_filing_claimant(self, queue):
        result = await _file_note(queue)

        record = queue.get(result['id'])
        assert record is not None
        assert record.filing_claimant_run_id == CLAIMANT


class TestGatesAndDedupeNotEvaluated:
    @pytest.mark.asyncio
    async def test_terminal_task_status_is_never_looked_up(self, queue):
        server, status_lookup = _server(queue, status='done')

        result = await call_info(server, **_NOTE_FILING)

        status_lookup.assert_not_awaited()
        assert result['resolution_class'] == NOTE_TO_FILE_CLASS
        record = queue.get(result['id'])
        assert record is not None
        assert record.resolution_class == NOTE_TO_FILE_CLASS

    @pytest.mark.asyncio
    async def test_terminal_state_is_the_bug_still_files_a_resolved_note(self, queue):
        result = await _file_note(queue, terminal_state_is_the_bug=True)

        assert result['status'] == 'resolved'
        assert result['resolution_class'] == NOTE_TO_FILE_CLASS
        assert queue.get_pending() == []

    @pytest.mark.asyncio
    async def test_review_suggestions_still_files_a_resolved_note(self, queue):
        result = await _file_note(queue, category='review_suggestions')

        assert result['status'] == 'resolved'
        assert result['resolution_class'] == NOTE_TO_FILE_CLASS
        assert queue.get_pending() == []

    @pytest.mark.asyncio
    async def test_identical_infra_notes_are_never_deduped(self, queue):
        summary = 'sandbox denied a write under the lane root'
        parent = Escalation(
            id='esc-5751-900', task_id='5751', agent_role='implementer',
            severity='info', category='infra_issue', summary=summary,
        )
        queue.submit(parent)

        first = await _file_note(queue, category='infra_issue', summary=summary)
        second = await _file_note(queue, category='infra_issue', summary=summary)

        assert first['id'] != second['id']
        for result in (first, second):
            record = queue.get(result['id'])
            assert record is not None
            assert record.status == 'resolved'
            assert record.resolution_class == NOTE_TO_FILE_CLASS
        reread_parent = queue.get(parent.id)
        assert reread_parent is not None
        assert reread_parent.dedupe_count == 0


class TestSeverityGuard:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', sorted(KNOWN_SEVERITIES - {'info'}))
    async def test_non_info_note_is_refused_and_writes_nothing(self, queue, severity):
        server, _ = _server(queue)
        before = sorted(queue.queue_dir.rglob('*'))

        result = await call_info(server, **{**_NOTE_FILING, 'severity': severity})

        assert 'error' in result
        assert result['code'] == 'note_to_file_requires_info'
        assert sorted(queue.queue_dir.rglob('*')) == before


class TestDefaultUnchanged:
    @pytest.mark.asyncio
    async def test_info_without_the_flag_is_still_queued_pending(self, queue):
        server, _ = _server(queue)
        filing = {k: v for k, v in _NOTE_FILING.items() if k != 'note_to_file'}

        result = await call_info(server, **filing)

        assert result['status'] == 'queued'
        assert [esc.id for esc in queue.get_pending()] == [result['id']]


class TestEvalLaneStillWins:
    @pytest.mark.asyncio
    async def test_eval_fixture_note_is_contained_by_gate_0(self, queue):
        result = await _file_note(queue, task_id=EVAL_FIXTURE_ID)

        assert result['status'] == 'resolved'
        assert result['resolved_by'] == 'escalation-eval-lane-containment'
        record = queue.get(result['id'])
        assert record is not None
        assert record.resolved_by == 'escalation-eval-lane-containment'
        assert queue.get_pending() == []
