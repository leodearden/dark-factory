"""Escalation.member_folded_at: the fold writer stamps when each member joined an L2."""

from __future__ import annotations

from pathlib import Path

from escalation.models import Escalation
from escalation.queue import EscalationQueue


def _submit_l2(queue: EscalationQueue) -> Escalation:
    l2 = Escalation(
        id=queue.make_id('task-1'),
        task_id='task-1',
        agent_role='escalation-watcher-auto',
        severity='blocking',
        category='design_concern',
        summary='L2 cluster',
        level=2,
        root_cause='Bad merge strategy',
        members=['esc-l1-0'],
    )
    queue.submit(l2)
    return l2


def test_fold_stamps_each_appended_member_with_the_write_time(tmp_path: Path):
    queue = EscalationQueue(tmp_path / 'esc')
    l2 = _submit_l2(queue)

    folded = queue.add_members_to_l2(l2.id, ['esc-l1-1', 'esc-l1-2'])

    assert folded is not None
    assert set(folded.member_folded_at) == {'esc-l1-1', 'esc-l1-2'}
    assert folded.member_folded_at['esc-l1-1'] == folded.updated_at
    assert folded.member_folded_at['esc-l1-2'] == folded.updated_at
    assert 'esc-l1-0' not in folded.member_folded_at


def test_refold_of_present_member_keeps_its_first_stamp(tmp_path: Path):
    queue = EscalationQueue(tmp_path / 'esc')
    l2 = _submit_l2(queue)
    first = queue.add_members_to_l2(l2.id, ['esc-l1-1'])
    assert first is not None
    first_stamp = first.member_folded_at['esc-l1-1']

    second = queue.add_members_to_l2(l2.id, ['esc-l1-1', 'esc-l1-3'])

    assert second is not None
    assert second.member_folded_at['esc-l1-1'] == first_stamp
    assert second.member_folded_at['esc-l1-3'] == second.updated_at
    assert set(second.member_folded_at) == {'esc-l1-1', 'esc-l1-3'}


def test_fold_appending_no_new_member_stamps_nothing(tmp_path: Path):
    queue = EscalationQueue(tmp_path / 'esc')
    l2 = _submit_l2(queue)

    folded = queue.add_members_to_l2(l2.id, ['esc-l1-0'])

    assert folded is not None
    assert folded.member_folded_at == {}


def test_stamp_survives_a_fresh_read(tmp_path: Path):
    queue = EscalationQueue(tmp_path / 'esc')
    l2 = _submit_l2(queue)
    folded = queue.add_members_to_l2(l2.id, ['esc-l1-1'])
    assert folded is not None

    reread = EscalationQueue(tmp_path / 'esc').get(l2.id)

    assert reread is not None
    assert reread.member_folded_at == folded.member_folded_at


def test_legacy_record_without_the_key_loads_empty():
    legacy = {
        'id': 'esc-task-1-1',
        'task_id': 'task-1',
        'agent_role': 'escalation-watcher-auto',
        'severity': 'blocking',
        'category': 'design_concern',
        'summary': 'L2 cluster',
        'level': 2,
        'members': ['esc-l1-0'],
    }

    assert Escalation.from_dict(legacy).member_folded_at == {}
