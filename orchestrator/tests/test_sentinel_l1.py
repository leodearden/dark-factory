"""The sentinel-L1 lifecycle: filed once per episode, resolved by the filer when its condition clears."""

from __future__ import annotations

import logging

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.sentinel_l1 import SentinelL1, file_sentinel_l1, resolve_sentinel_l1


@pytest.fixture
def queue(tmp_path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


def _l1(sentinel: str) -> SentinelL1:
    return SentinelL1(
        sentinel=sentinel,
        category='infra_issue',
        summary=f'{sentinel} condition holds',
        detail='The condition this sentinel watches is holding.',
        suggested_action='none',
    )


def _pending_ids(queue: EscalationQueue, sentinel: str) -> list[str]:
    return [esc.id for esc in queue.get_by_task(sentinel, status='pending')]


class _DownQueue(EscalationQueue):
    def get_by_task(self, task_id, status=None, level=None, agent_role=None):
        raise RuntimeError('escalation queue is down')


def test_resolves_the_open_l1_filed_under_its_sentinel(queue):
    file_sentinel_l1(queue, _l1('__s__'))
    [first_id] = _pending_ids(queue, '__s__')

    resolve_sentinel_l1(queue, '__s__', 'condition cleared')

    resolved = queue.get(first_id)
    assert resolved is not None
    assert resolved.status == 'resolved'
    assert resolved.resolution == 'condition cleared'
    assert resolved.resolved_by == 'orchestrator-scheduler'
    assert not queue.has_open_l1('__s__')

    file_sentinel_l1(queue, _l1('__s__'))
    [second_id] = _pending_ids(queue, '__s__')
    assert second_id != first_id


def test_leaves_other_sentinels_and_other_roles_open(queue):
    file_sentinel_l1(queue, _l1('__a__'))
    file_sentinel_l1(queue, _l1('__b__'))
    [scheduler_b] = _pending_ids(queue, '__b__')
    steward_a = queue.submit(Escalation(
        id=queue.make_id('__a__'),
        task_id='__a__',
        agent_role='steward',
        severity='blocking',
        category='infra_issue',
        summary='a steward L1 on the same sentinel',
        detail='Filed by another role; not the scheduler’s to resolve.',
        suggested_action='none',
        level=1,
    ))

    resolve_sentinel_l1(queue, '__a__', 'condition cleared')

    still_pending = {esc.id for esc in queue.get_pending()}
    assert still_pending == {scheduler_b, steward_a}


def test_without_a_queue_it_is_a_no_op():
    assert resolve_sentinel_l1(None, '__s__', 'x') is None


def test_a_failing_queue_is_logged_and_swallowed(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger='orchestrator.sentinel_l1')

    resolve_sentinel_l1(_DownQueue(tmp_path / 'esc'), '__s__', 'condition cleared')

    warnings = [
        r for r in caplog.records
        if r.name == 'orchestrator.sentinel_l1' and r.levelno == logging.WARNING
    ]
    assert any('__s__' in r.getMessage() for r in warnings)
