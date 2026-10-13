"""``resolution_action`` is written by ``EscalationQueue.resolve``, inside its lock.

The stamp lands in the same per-id critical section as the status
check-and-set, so it is atomic with the transition it describes.  No caller
pre-writes a snapshot of the record to carry it there.
"""

from __future__ import annotations

import json
from pathlib import Path

from escalation.models import Escalation
from escalation.queue import EscalationQueue, ResolveOutcome

SWEEP_TEXT = 'Auto-dismissed: steward interrupted (attempt cap)'
LATE_TEXT = "the steward's real finding: verify never ran"
STEWARD = 'claude-task-3902-steward'


def _seed_pending(queue: EscalationQueue, esc_id: str = 'esc-3902-1') -> str:
    queue.submit(Escalation(
        id=esc_id,
        task_id='3902',
        agent_role='implementer',
        severity='blocking',
        category='task_failure',
        summary='a pending L0',
    ))
    return esc_id


def _archive_copies(queue: EscalationQueue, esc_id: str) -> list[Path]:
    return list((queue.queue_dir / 'archive').rglob(f'{esc_id}.json'))


def _blank_outcome() -> ResolveOutcome:
    return {
        'applied': False,
        'prior_status': None,
        'prior_resolved_by': None,
        'late_resolution_captured': False,
        'resolution_class_corrected': None,
    }


class TestQueueResolveStampsResolutionAction:

    def test_applied_resolve_stamps_the_action_on_the_archived_record_and_the_callback(
        self, tmp_path: Path,
    ):
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _seed_pending(queue)
        seen: list[Escalation] = []
        queue.set_resolve_callback(seen.append)

        esc = queue.resolve(esc_id, 'abandoning', dismiss=True, resolution_action='abandon')

        assert esc is not None and esc.resolution_action == 'abandon'
        archived = _archive_copies(queue, esc_id)
        assert len(archived) == 1, f'expected exactly one archive copy: {archived}'
        assert json.loads(archived[0].read_text())['resolution_action'] == 'abandon'
        assert not (queue.queue_dir / f'{esc_id}.json').exists()
        assert len(seen) == 1 and seen[0].resolution_action == 'abandon', (
            'the harness reads the action off the resolve-callback argument'
        )

    def test_resolve_without_resolution_action_leaves_it_unset(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _seed_pending(queue)

        queue.resolve(esc_id, 'ok')

        stored = queue.get(esc_id)
        assert stored is not None and stored.resolution_action is None

    def test_already_terminal_resolve_does_not_stamp_the_action(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')
        esc_id = _seed_pending(queue)
        queue.resolve(esc_id, SWEEP_TEXT, dismiss=True, resolved_by='auto-dismissed')
        outcome = _blank_outcome()

        queue.resolve(
            esc_id, LATE_TEXT, resolved_by=STEWARD,
            resolution_action='resume', outcome=outcome,
        )

        assert outcome['applied'] is False
        assert outcome['late_resolution_captured'] is True
        stored = queue.get(esc_id)
        assert stored is not None
        assert stored.resolution_action is None, (
            'the terminal state belongs to whoever closed the record first'
        )
        assert stored.status == 'dismissed'
        assert stored.resolved_by == 'auto-dismissed'
        assert not (queue.queue_dir / f'{esc_id}.json').exists()
