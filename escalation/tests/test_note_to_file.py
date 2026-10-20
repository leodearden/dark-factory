"""The note-to-file escalation class at both of its enforcement points: the
queue's only pending-write path and the server's filing chokepoint
(plans/plan-deviation-recording-prd.md D2).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from escalation.archive import archive_dir_for_date
from escalation.models import NOTE_TO_FILE_CLASS, Escalation
from escalation.queue import EscalationQueue


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
