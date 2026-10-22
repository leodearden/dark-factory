"""Promoting a level-0 escalation to level 1 keeps its finding intact.

``orchestrator.l0_promotion.promote_l0_to_l1`` copies a pending L0 verbatim
into a new level-1 record and only then dismisses the L0, naming the L1.
These tests drive it against a real ``EscalationQueue`` on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from orchestrator.l0_promotion import promote_l0_to_l1

TASK_ID = '7533'
WHY = 'architect filed it in lieu of a plan'
SUFFIX = '\n\n[durable refs] branch=task/7533'


def _submit_l0(queue: EscalationQueue, *, severity: str = 'blocking') -> Escalation:
    l0 = Escalation(
        id=queue.make_id(TASK_ID),
        task_id=TASK_ID,
        agent_role='architect',
        severity=severity,
        category='design_concern',
        summary='Leo decision needed: RECOMMEND KEEP max_concurrent_tasks=24',
        detail='Line one of the finding.\nLine two of the finding.',
        suggested_action='Leo decision needed: keep max_concurrent_tasks=24',
        level=0,
        evidence=[{
            'observation': 'p90 dispatch latency 41s at 24 slots',
            'measured_at': 'HEAD=5b999b3316',
            'ref': 'scheduler log 2026-10-22',
        }],
        options=['A: keep', 'B: lower'],
        root_cause='r',
    )
    queue.submit(l0)
    return l0


def _promote(queue: EscalationQueue, l0: Escalation) -> Escalation:
    return promote_l0_to_l1(
        queue, l0,
        agent_role='orchestrator-workflow',
        why=WHY,
        detail_suffix=SUFFIX,
        filing_claimant_run_id='run-x',
        resolved_by='orchestrator-workflow',
    )


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'escalations')


class TestVerbatimPromotion:
    def test_l1_carries_the_finding_verbatim(self, queue: EscalationQueue):
        l0 = _submit_l0(queue)

        l1 = _promote(queue, l0)

        assert l1.id != l0.id
        assert l1.level == 1
        assert l1.status == 'pending'
        assert (l1.task_id, l1.category, l1.severity) == (
            l0.task_id, l0.category, l0.severity,
        )
        assert l1.summary == l0.summary
        assert l1.detail.startswith(l0.detail)
        assert l0.id in l1.detail
        assert WHY in l1.detail
        assert l1.detail.endswith(SUFFIX)
        assert l1.suggested_action == l0.suggested_action
        assert l1.options == l0.options
        assert l1.evidence == l0.evidence
        assert l1.root_cause == l0.root_cause
        assert l1.worktree is None
        assert l1.agent_role == 'orchestrator-workflow'
        assert l1.filing_claimant_run_id == 'run-x'

    def test_only_the_l1_stays_pending(self, queue: EscalationQueue):
        l0 = _submit_l0(queue)

        l1 = _promote(queue, l0)

        assert [e.id for e in queue.get_by_task(TASK_ID, status='pending')] == [l1.id]

    def test_l0_is_dismissed_naming_the_l1(self, queue: EscalationQueue):
        l0 = _submit_l0(queue)

        l1 = _promote(queue, l0)

        stored = queue.get(l0.id)
        assert stored is not None
        assert stored.status == 'dismissed'
        assert stored.resolution is not None
        assert l1.id in stored.resolution
        assert WHY in stored.resolution
        assert stored.resolved_by == 'orchestrator-workflow'


def test_info_severity_is_never_inflated(queue: EscalationQueue):
    l0 = _submit_l0(queue, severity='info')

    l1 = _promote(queue, l0)

    assert l1.severity == 'info'


class TestPrecondition:
    def test_level_one_record_is_rejected(self, queue: EscalationQueue):
        l1 = Escalation(
            id=queue.make_id(TASK_ID), task_id=TASK_ID,
            agent_role='orchestrator-workflow', severity='blocking',
            category='design_concern', summary='already an L1', level=1,
        )
        queue.submit(l1)

        with pytest.raises(ValueError) as excinfo:
            _promote(queue, l1)

        message = str(excinfo.value)
        assert l1.id in message
        assert 'level' in message and '1' in message
        assert 'pending' in message
        assert [e.id for e in queue.get_by_task(TASK_ID)] == [l1.id]

    def test_dismissed_l0_is_rejected(self, queue: EscalationQueue):
        l0 = _submit_l0(queue)
        queue.resolve(l0.id, 'handled elsewhere', dismiss=True, resolved_by='test')
        dismissed = queue.get(l0.id)
        assert dismissed is not None

        with pytest.raises(ValueError) as excinfo:
            _promote(queue, dismissed)

        message = str(excinfo.value)
        assert l0.id in message
        assert 'level' in message and '0' in message
        assert 'dismissed' in message
        assert [e.id for e in queue.get_by_task(TASK_ID)] == [l0.id]


class _L1SubmitFails(EscalationQueue):
    def submit(self, escalation: Escalation) -> str:
        if escalation.level >= 1:
            raise OSError('disk full')
        return super().submit(escalation)


def test_failed_l1_submit_leaves_the_l0_pending(tmp_path: Path):
    queue = _L1SubmitFails(tmp_path / 'escalations')
    l0 = _submit_l0(queue)

    with pytest.raises(OSError):
        _promote(queue, l0)

    stored = queue.get(l0.id)
    assert stored is not None
    assert stored.status == 'pending'
