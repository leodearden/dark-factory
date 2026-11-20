"""Tests for ReviewCheckpoint._promote_reviewer_escalations().

The deep reviewer files its escalations under review_escalation_task_id, a
task id no steward watches, so the checkpoint promotes each pending L0 to
level 1 itself. These tests drive it against a real ``EscalationQueue``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from _orch_helpers import pydantic_spec
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.review_checkpoint import ReviewCheckpoint, review_escalation_task_id
from orchestrator.verify import VerifyResult

WHY = 'review checkpoint has no steward'


def _submit_reviewer_l0(
    queue: EscalationQueue,
    task_id: str,
    *,
    level: int = 0,
    severity: str = 'blocking',
) -> Escalation:
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='reviewer',
        severity=severity,
        category='design_concern',
        summary='Two modules each build the same record',
        detail='Line one of the finding.\nLine two of the finding.',
        suggested_action='investigate',
        worktree='/home/leo/src/dark-factory',
        level=level,
        evidence=[{
            'observation': 'two hand-rolled Escalation(...) builds',
            'measured_at': 'HEAD=9856ed15dc',
            'ref': 'review 20260418T120000',
        }],
        options=['A: fold onto the helper', 'B: leave both'],
        root_cause='r',
        project_id='dark_factory',
    )
    queue.submit(esc)
    return esc


class TestReviewerEscalationPromotion:
    """ReviewCheckpoint._promote_reviewer_escalations promotes reviewer L0s to L1."""

    @pytest.fixture
    def checkpoint(self, tmp_path: Path) -> ReviewCheckpoint:
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.project_root = tmp_path
        cp = ReviewCheckpoint(config, mcp=MagicMock(), usage_gate=None)
        cp.escalation_queue = EscalationQueue(tmp_path / 'escalations')
        return cp

    def test_no_queue_is_noop(self, tmp_path: Path):
        config = MagicMock(spec_set=pydantic_spec(OrchestratorConfig))
        config.project_root = tmp_path
        cp = ReviewCheckpoint(config, mcp=MagicMock(), usage_gate=None)
        # escalation_queue defaults to None
        assert cp._promote_reviewer_escalations('20260418T120000') == 0

    def test_no_pending_escalations(self, checkpoint: ReviewCheckpoint):
        assert checkpoint._promote_reviewer_escalations('20260418T120000') == 0

    def test_promotes_l0_to_l1(self, checkpoint: ReviewCheckpoint):
        review_id = '20260418T120000'
        synthetic = review_escalation_task_id(review_id)
        assert checkpoint.escalation_queue is not None
        original = _submit_reviewer_l0(checkpoint.escalation_queue, synthetic)

        count = checkpoint._promote_reviewer_escalations(review_id)
        assert count == 1

        pending = checkpoint.escalation_queue.get_pending()
        l1s = [e for e in pending if e.level == 1]
        assert len(l1s) == 1
        l1 = l1s[0]
        assert (l1.task_id, l1.category, l1.severity) == (
            original.task_id, original.category, original.severity,
        )
        assert l1.summary == original.summary
        assert l1.detail.startswith(original.detail)
        assert f'[promoted from {original.id}: {WHY}]' in l1.detail
        assert (
            l1.suggested_action, l1.evidence, l1.options, l1.root_cause, l1.project_id,
        ) == (
            original.suggested_action, original.evidence, original.options,
            original.root_cause, original.project_id,
        )
        assert l1.worktree == original.worktree
        assert l1.agent_role == 'review-checkpoint'

        refreshed = checkpoint.escalation_queue.get(original.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        assert refreshed.resolved_by == 'review-checkpoint'
        assert refreshed.resolution is not None
        assert refreshed.resolution.startswith('Auto-promoted to level 1')
        assert l1.id in refreshed.resolution
        assert WHY in refreshed.resolution

    def test_only_matches_own_review_id(self, checkpoint: ReviewCheckpoint):
        """An escalation from a different review_id is not promoted."""
        assert checkpoint.escalation_queue is not None
        _submit_reviewer_l0(
            checkpoint.escalation_queue,
            review_escalation_task_id('20260418T000000'),
        )
        assert checkpoint._promote_reviewer_escalations('20260418T999999') == 0

    def test_does_not_touch_l1(self, checkpoint: ReviewCheckpoint):
        review_id = '20260418T120000'
        synthetic = review_escalation_task_id(review_id)
        assert checkpoint.escalation_queue is not None
        _submit_reviewer_l0(checkpoint.escalation_queue, synthetic, level=1)
        assert checkpoint._promote_reviewer_escalations(review_id) == 0

    def test_prompt_names_the_task_id_promotion_keys_on(
        self, checkpoint: ReviewCheckpoint,
    ):
        review_id = '20260418T120000'
        tid = review_escalation_task_id(review_id)
        prompt = checkpoint._build_prompt(
            mode='focused',
            phase1=VerifyResult(
                passed=True, summary='OK', test_output='', lint_output='', type_output='',
            ),
            modules=['orchestrator'],
            briefing_content='',
            review_id=review_id,
        )

        # Delimited on purpose: the agent_id `claude-review-<id>` already
        # contains the bare `review-<id>` substring.
        assert f'`{tid}`' in prompt
        assert f'escalate_info(task_id="{tid}"' in prompt

        assert checkpoint.escalation_queue is not None
        _submit_reviewer_l0(checkpoint.escalation_queue, tid, severity='info')
        assert checkpoint._promote_reviewer_escalations(review_id) == 1
        l1s = [e for e in checkpoint.escalation_queue.get_pending() if e.level == 1]
        assert [e.task_id for e in l1s] == [tid]
