"""EscalationQueue.patch_resolution_metadata(append_to_resolution=...) (task 5311)."""

from __future__ import annotations

from pathlib import Path

import pytest

from escalation.models import Escalation
from escalation.queue import EscalationQueue


def _escalation(esc_id: str, *, status: str = 'pending') -> Escalation:
    return Escalation(
        id=esc_id,
        task_id='5260',
        agent_role='harness-stranded-blocked-reaper',
        severity='blocking',
        category='stranded_blocked',
        summary='stranded',
        level=1,
        status=status,
    )


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


def _archived(queue: EscalationQueue, esc_id: str) -> str:
    queue.submit(_escalation(esc_id))
    queue.resolve(esc_id, 'resumed by steward', resolved_by='steward')
    return esc_id


class TestAppendToResolution:
    def test_appends_to_archived_record_in_place(self, queue: EscalationQueue) -> None:
        esc_id = _archived(queue, 'esc-5260-1')

        patched = queue.patch_resolution_metadata(esc_id, append_to_resolution='[MARK] note')

        expected = 'resumed by steward\n\n[MARK] note'
        assert patched is not None
        assert patched.resolution == expected
        stored = queue.get(esc_id)
        assert stored is not None
        assert stored.resolution == expected
        assert not (queue.queue_dir / f'{esc_id}.json').exists()
        assert len(list((queue.queue_dir / 'archive').rglob(f'{esc_id}.json'))) == 1

    def test_repeat_is_idempotent(self, queue: EscalationQueue) -> None:
        esc_id = _archived(queue, 'esc-5260-1')

        queue.patch_resolution_metadata(esc_id, append_to_resolution='[MARK] note')
        queue.patch_resolution_metadata(esc_id, append_to_resolution='[MARK] note')

        stored = queue.get(esc_id)
        assert stored is not None
        assert stored.resolution is not None
        assert stored.resolution.count('[MARK] note') == 1
        assert stored.resolution == 'resumed by steward\n\n[MARK] note'

    def test_idempotency_is_keyed_on_the_exact_note_text(self, queue: EscalationQueue) -> None:
        queue.submit(_escalation('esc-5260-1'))
        queue.resolve('esc-5260-1', 'already says [MARK] note', resolved_by='steward')

        queue.patch_resolution_metadata('esc-5260-1', append_to_resolution='[MARK] note')
        queue.patch_resolution_metadata('esc-5260-1', append_to_resolution='[MARK] other note')

        stored = queue.get('esc-5260-1')
        assert stored is not None
        assert stored.resolution == 'already says [MARK] note\n\n[MARK] other note'

    def test_empty_string_resolution_becomes_the_note(self, queue: EscalationQueue) -> None:
        queue.submit(_escalation('esc-5260-1'))
        queue.resolve('esc-5260-1', '', resolved_by='steward')

        patched = queue.patch_resolution_metadata('esc-5260-1', append_to_resolution='[MARK] note')

        assert patched is not None
        assert patched.resolution == '[MARK] note'

    def test_none_resolution_becomes_the_note(self, queue: EscalationQueue) -> None:
        queue.submit(_escalation('esc-5260-1', status='resolved'))

        patched = queue.patch_resolution_metadata('esc-5260-1', append_to_resolution='[MARK] note')

        assert patched is not None
        assert patched.resolution == '[MARK] note'
        stored = queue.get('esc-5260-1')
        assert stored is not None
        assert stored.resolution == '[MARK] note'

    def test_pending_record_is_left_unchanged(self, queue: EscalationQueue) -> None:
        queue.submit(_escalation('esc-5260-1'))

        assert queue.patch_resolution_metadata(
            'esc-5260-1', append_to_resolution='[MARK] note',
        ) is None
        stored = queue.get('esc-5260-1')
        assert stored is not None
        assert stored.status == 'pending'
        assert stored.resolution is None

    def test_unknown_id_returns_none(self, queue: EscalationQueue) -> None:
        assert queue.patch_resolution_metadata(
            'esc-nope-1', append_to_resolution='[MARK] note',
        ) is None


class TestExistingPatchBehaviourPreserved:
    def test_resolved_by_alone_leaves_resolution_untouched(self, queue: EscalationQueue) -> None:
        esc_id = _archived(queue, 'esc-5260-1')

        patched = queue.patch_resolution_metadata(esc_id, resolved_by='x')

        assert patched is not None
        assert patched.resolved_by == 'x'
        assert patched.resolution == 'resumed by steward'

    def test_combined_call_applies_both(self, queue: EscalationQueue) -> None:
        esc_id = _archived(queue, 'esc-5260-1')

        queue.patch_resolution_metadata(
            esc_id, resolved_by='x', append_to_resolution='[MARK] note',
        )

        stored = queue.get(esc_id)
        assert stored is not None
        assert stored.resolved_by == 'x'
        assert stored.resolution == 'resumed by steward\n\n[MARK] note'
