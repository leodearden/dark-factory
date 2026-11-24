"""The merge-halt release marker on Escalation, and its sole writer EscalationQueue.release_merge_halt."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from escalation.models import Escalation
from escalation.queue import EscalationQueue


def _submit_halt_record(queue: EscalationQueue, task_id: str = '4843') -> Escalation:
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='orchestrator',
        severity='blocking',
        category='wip_conflict',
        summary='preserved merge-halt owner',
        level=1,
    )
    queue.submit(esc)
    return esc


class TestReleaseMarkerFields:
    def test_fresh_escalation_carries_no_release(self):
        esc = Escalation(
            id='esc-1-1', task_id='1', agent_role='orchestrator', severity='blocking',
            category='wip_conflict', summary='s',
        )

        assert esc.halt_released_at is None
        assert esc.halt_release_note == ''

    def test_release_fields_round_trip_through_json(self):
        esc = Escalation(
            id='esc-1-1', task_id='1', agent_role='orchestrator', severity='blocking',
            category='wip_conflict', summary='s',
            halt_released_at='2026-11-19T00:00:00+00:00', halt_release_note='clean at restart',
        )

        restored = Escalation.from_json(esc.to_json())

        assert restored.halt_released_at == '2026-11-19T00:00:00+00:00'
        assert restored.halt_release_note == 'clean at restart'

    def test_legacy_record_without_release_keys_takes_defaults(self):
        data = Escalation(
            id='esc-1-1', task_id='1', agent_role='orchestrator', severity='blocking',
            category='wip_conflict', summary='s',
        ).to_dict()
        del data['halt_released_at']
        del data['halt_release_note']

        restored = Escalation.from_dict(data)

        assert restored.halt_released_at is None
        assert restored.halt_release_note == ''


class TestReleaseMergeHalt:
    def test_stamps_release_without_changing_disposition(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')
        esc = _submit_halt_record(queue)

        result = queue.release_merge_halt(esc.id, note='project_root clean')

        assert result is not None
        assert result.halt_released_at is not None
        datetime.fromisoformat(result.halt_released_at)
        assert result.halt_release_note == 'project_root clean'
        assert result.updated_at is not None
        assert result.status == 'pending'
        assert result.level == 1
        assert result.category == 'wip_conflict'
        assert (queue.queue_dir / f'{esc.id}.json').exists()

    def test_release_is_durable_across_queue_instances(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')
        esc = _submit_halt_record(queue)

        queue.release_merge_halt(esc.id, note='project_root clean')

        reloaded = EscalationQueue(tmp_path / 'esc').get(esc.id)
        assert reloaded is not None
        assert reloaded.halt_released_at is not None
        assert reloaded.halt_release_note == 'project_root clean'

    def test_second_release_keeps_the_first_stamp(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')
        esc = _submit_halt_record(queue)
        first = queue.release_merge_halt(esc.id, note='first')
        assert first is not None

        second = queue.release_merge_halt(esc.id, note='second')

        assert second is not None
        assert second.halt_released_at == first.halt_released_at
        assert second.halt_release_note == 'first'
        assert second.updated_at == first.updated_at
        reloaded = queue.get(esc.id)
        assert reloaded is not None
        assert reloaded.halt_release_note == 'first'
        assert reloaded.updated_at == first.updated_at

    def test_unknown_id_returns_none(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')

        assert queue.release_merge_halt('esc-does-not-exist', note='n') is None

    def test_resolved_record_is_refused_and_not_resurrected(self, tmp_path: Path):
        queue = EscalationQueue(tmp_path / 'esc')
        esc = _submit_halt_record(queue)
        queue.resolve(esc.id, 'Fixed')

        result = queue.release_merge_halt(esc.id, note='n')

        assert result is None
        assert not (queue.queue_dir / f'{esc.id}.json').exists()
        archived = queue.get(esc.id)
        assert archived is not None
        assert archived.halt_released_at is None
