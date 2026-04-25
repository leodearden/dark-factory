"""Tests for Harness._reap_orphan_l0_escalations() and ReviewCheckpoint._promote_reviewer_escalations()."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig
from orchestrator.harness import Harness
from orchestrator.review_checkpoint import ReviewCheckpoint


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
    )


@pytest.fixture
def harness(tmp_path: Path, git_config: GitConfig) -> Harness:
    """Real Harness with a real EscalationQueue on tmp_path.

    Other internals are mocked; only the reaper path is under test.
    """
    config = MagicMock()
    config.git = git_config
    config.project_root = tmp_path
    config.usage_cap.enabled = False
    config.review.enabled = False
    config.orphan_l0_reaper_enabled = True
    config.orphan_l0_timeout_secs = 60.0
    config.orphan_l0_check_interval_secs = 1.0
    config.sandbox.backend = 'auto'

    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(config)

    h._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    return h


def _submit_aged(
    queue: EscalationQueue,
    task_id: str,
    seconds_ago: float,
    *,
    level: int = 0,
    category: str = 'design_concern',
) -> Escalation:
    """Submit an escalation whose timestamp is ``seconds_ago`` in the past."""
    ts = (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat()
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='integration-reviewer',
        severity='info',
        category=category,
        summary=f'aged {seconds_ago}s',
        detail='detail',
        suggested_action='investigate',
        timestamp=ts,
        level=level,
    )
    queue.submit(esc)
    return esc


class TestOrphanL0Reaper:
    """Harness._reap_orphan_l0_escalations promotes aged orphan L0s to L1."""

    def test_no_queue_is_noop(self, harness: Harness):
        harness._escalation_queue = None
        assert harness._reap_orphan_l0_escalations() == 0

    def test_empty_queue(self, harness: Harness):
        assert harness._reap_orphan_l0_escalations() == 0

    def test_young_l0_not_promoted(self, harness: Harness):
        assert harness._escalation_queue is not None
        _submit_aged(harness._escalation_queue, 'review-abc', seconds_ago=10.0)
        assert harness._reap_orphan_l0_escalations() == 0
        pending = harness._escalation_queue.get_pending()
        assert len(pending) == 1
        assert pending[0].level == 0

    def test_aged_orphan_l0_promoted(self, harness: Harness):
        assert harness._escalation_queue is not None
        original = _submit_aged(
            harness._escalation_queue, 'review-abc', seconds_ago=300.0,
        )

        count = harness._reap_orphan_l0_escalations()
        assert count == 1

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        # Original is dismissed
        refreshed = harness._escalation_queue.get(original.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        assert refreshed.resolved_by == 'harness-orphan-reaper'

        # New L1 exists
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 1
        l1 = l1s[0]
        assert l1.task_id == 'review-abc'
        assert l1.agent_role == 'harness-orphan-reaper'
        assert l1.category == 'design_concern'
        assert l1.suggested_action == 'manual_intervention'
        assert l1.status == 'pending'
        assert original.summary in l1.summary  # original summary preserved

    def test_active_workflow_l0_not_promoted(self, harness: Harness):
        """An L0 for a task_id with an active workflow is left alone."""
        assert harness._escalation_queue is not None
        import asyncio
        _submit_aged(harness._escalation_queue, 'task-42', seconds_ago=300.0)
        harness._escalation_events['task-42'] = asyncio.Event()

        assert harness._reap_orphan_l0_escalations() == 0

    def test_l1_not_touched(self, harness: Harness):
        """Level-1 escalations are never promoted (they're already at the top)."""
        assert harness._escalation_queue is not None
        _submit_aged(
            harness._escalation_queue, 'review-abc', seconds_ago=300.0, level=1,
        )
        assert harness._reap_orphan_l0_escalations() == 0
        pending = harness._escalation_queue.get_pending()
        assert len(pending) == 1
        assert pending[0].level == 1

    def test_multiple_orphans_all_promoted(self, harness: Harness):
        assert harness._escalation_queue is not None
        _submit_aged(harness._escalation_queue, 'review-a', seconds_ago=300.0)
        _submit_aged(harness._escalation_queue, 'review-b', seconds_ago=300.0)
        _submit_aged(harness._escalation_queue, 'review-c', seconds_ago=10.0)  # young

        assert harness._reap_orphan_l0_escalations() == 2

        pending = harness._escalation_queue.get_pending()
        l1s = [e for e in pending if e.level == 1]
        l0s = [e for e in pending if e.level == 0]
        assert len(l1s) == 2
        assert len(l0s) == 1  # the young one remains


class TestReviewerEscalationPromotion:
    """ReviewCheckpoint._promote_reviewer_escalations promotes reviewer L0s to L1."""

    @pytest.fixture
    def checkpoint(self, tmp_path: Path) -> ReviewCheckpoint:
        config = MagicMock()
        config.project_root = tmp_path
        cp = ReviewCheckpoint(config, mcp=MagicMock(), usage_gate=None)
        cp.escalation_queue = EscalationQueue(tmp_path / 'escalations')
        return cp

    def test_no_queue_is_noop(self, tmp_path: Path):
        config = MagicMock()
        config.project_root = tmp_path
        cp = ReviewCheckpoint(config, mcp=MagicMock(), usage_gate=None)
        # escalation_queue defaults to None
        assert cp._promote_reviewer_escalations('20260418T120000') == 0

    def test_no_pending_escalations(self, checkpoint: ReviewCheckpoint):
        assert checkpoint._promote_reviewer_escalations('20260418T120000') == 0

    def test_promotes_l0_to_l1(self, checkpoint: ReviewCheckpoint):
        review_id = '20260418T120000'
        synthetic = f'review-{review_id}'
        assert checkpoint.escalation_queue is not None
        original = _submit_aged(
            checkpoint.escalation_queue, synthetic, seconds_ago=1.0,
        )

        count = checkpoint._promote_reviewer_escalations(review_id)
        assert count == 1

        # Original dismissed
        refreshed = checkpoint.escalation_queue.get(original.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        assert refreshed.resolved_by == 'review-checkpoint'

        # L1 created
        pending = checkpoint.escalation_queue.get_pending()
        l1s = [e for e in pending if e.level == 1]
        assert len(l1s) == 1
        l1 = l1s[0]
        assert l1.task_id == synthetic
        assert l1.agent_role == 'review-checkpoint'
        assert l1.category == original.category
        assert l1.summary == original.summary
        assert l1.detail == original.detail
        assert l1.suggested_action == 'manual_intervention'

    def test_only_matches_own_review_id(self, checkpoint: ReviewCheckpoint):
        """An escalation from a different review_id is not promoted."""
        assert checkpoint.escalation_queue is not None
        _submit_aged(
            checkpoint.escalation_queue,
            'review-20260418T000000',
            seconds_ago=1.0,
        )
        assert checkpoint._promote_reviewer_escalations('20260418T999999') == 0

    def test_does_not_touch_l1(self, checkpoint: ReviewCheckpoint):
        review_id = '20260418T120000'
        synthetic = f'review-{review_id}'
        assert checkpoint.escalation_queue is not None
        _submit_aged(
            checkpoint.escalation_queue, synthetic, seconds_ago=1.0, level=1,
        )
        assert checkpoint._promote_reviewer_escalations(review_id) == 0
