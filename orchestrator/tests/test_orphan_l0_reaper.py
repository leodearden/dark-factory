"""Tests for Harness._reap_orphan_l0_escalations() and ReviewCheckpoint._promote_reviewer_escalations()."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import pydantic_spec
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.harness import Harness
from orchestrator.review_checkpoint import ReviewCheckpoint


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Real Harness with a real EscalationQueue on tmp_path.

    Other internals are mocked; only the reaper path is under test.
    """
    mock_orch_config.orphan_l0_reaper_enabled = True
    mock_orch_config.orphan_l0_timeout_secs = 60.0
    mock_orch_config.orphan_l0_check_interval_secs = 1.0

    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    return h


def _submit_aged(
    queue: EscalationQueue,
    task_id: str,
    seconds_ago: float,
    *,
    level: int = 0,
    category: str = 'design_concern',
    worktree: str | None = None,
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
        worktree=worktree,
        level=level,
    )
    queue.submit(esc)
    return esc


def _submit_aged_done_step_commit_orphan(
    queue: EscalationQueue,
    task_id: str,
    seconds_ago: float,
    *,
    step_id: str = 'step-7',
    stale_commit: str = 'fb62c8e439abc123',
) -> Escalation:
    """Submit an aged L0 matching the done-step-commit class filed by
    ``TaskWorkflow._escalate_unreconciled_done_step`` (workflow.py:5488) —
    ``agent_role='orchestrator'``, ``category='infra_issue'``,
    ``suggested_action='verify_wip_reconciliation'``.
    """
    ts = (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat()
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='orchestrator',
        severity='info',
        category='infra_issue',
        summary=(
            f"Done step {step_id}'s commit {stale_commit[:10]} is orphaned "
            'and could not be auto-reconciled against WIP tip <none>'
        ),
        detail=(
            f'Step {step_id} recorded commit {stale_commit}, which is no '
            'longer reachable from HEAD.'
        ),
        suggested_action='verify_wip_reconciliation',
        timestamp=ts,
        level=0,
    )
    queue.submit(esc)
    return esc


class TestOrphanL0Reaper:
    """Harness._reap_orphan_l0_escalations promotes aged orphan L0s to L1."""

    @pytest.mark.asyncio
    async def test_no_queue_is_noop(self, harness: Harness):
        harness._escalation_queue = None
        assert await harness._reap_orphan_l0_escalations() == 0

    @pytest.mark.asyncio
    async def test_empty_queue(self, harness: Harness):
        assert await harness._reap_orphan_l0_escalations() == 0

    @pytest.mark.asyncio
    async def test_young_l0_not_promoted(self, harness: Harness):
        assert harness._escalation_queue is not None
        _submit_aged(harness._escalation_queue, 'review-abc', seconds_ago=10.0)
        assert await harness._reap_orphan_l0_escalations() == 0
        pending = harness._escalation_queue.get_pending()
        assert len(pending) == 1
        assert pending[0].level == 0

    @pytest.mark.asyncio
    async def test_aged_orphan_l0_promoted(self, harness: Harness):
        assert harness._escalation_queue is not None
        original = _submit_aged(
            harness._escalation_queue, 'review-abc', seconds_ago=300.0,
            worktree='/home/leo/src/dark-factory/.worktrees/review-abc',
        )

        count = await harness._reap_orphan_l0_escalations()
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
        # A1: the promoted L1 cites a durable branch ref, not the ephemeral
        # worktree (which is reaped before a human reads it).
        assert l1.worktree is None
        assert 'branch=task/review-abc' in (l1.detail or '')

    @pytest.mark.asyncio
    async def test_aged_orphan_l0_with_open_l1_dismissed_not_promoted(
        self, harness: Harness,
    ):
        """B2: an aged orphan L0 for a task that already has an open L1 is
        dismissed by the reaper, not promoted to a duplicate "echo" L1.

        Reproduces the 3843/3861/3555 echo pattern: a still-pending L0 was
        re-promoted ~10 min after the workflow had already raised an L1 for
        the same condition.
        """
        assert harness._escalation_queue is not None
        # An L1 is already open for this task (e.g. raised by the workflow).
        _submit_aged(
            harness._escalation_queue, 'task-99', seconds_ago=5.0, level=1,
        )
        # A separate aged orphan L0 exists for the same task.
        orphan = _submit_aged(
            harness._escalation_queue, 'task-99', seconds_ago=300.0, level=0,
        )

        count = await harness._reap_orphan_l0_escalations()
        assert count == 0  # nothing promoted

        # The orphan L0 was dismissed by the reaper...
        refreshed = harness._escalation_queue.get(orphan.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        assert refreshed.resolved_by == 'harness-orphan-reaper'

        # ...and no second L1 was created — only the pre-existing one remains.
        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 1

    @pytest.mark.asyncio
    async def test_active_workflow_l0_not_promoted(self, harness: Harness):
        """An L0 for a task_id with an active workflow is left alone."""
        assert harness._escalation_queue is not None
        import asyncio
        _submit_aged(harness._escalation_queue, 'task-42', seconds_ago=300.0)
        harness._escalation_events['task-42'] = asyncio.Event()

        assert await harness._reap_orphan_l0_escalations() == 0

    @pytest.mark.asyncio
    async def test_l1_not_touched(self, harness: Harness):
        """Level-1 escalations are never promoted (they're already at the top)."""
        assert harness._escalation_queue is not None
        _submit_aged(
            harness._escalation_queue, 'review-abc', seconds_ago=300.0, level=1,
        )
        assert await harness._reap_orphan_l0_escalations() == 0
        pending = harness._escalation_queue.get_pending()
        assert len(pending) == 1
        assert pending[0].level == 1

    @pytest.mark.asyncio
    async def test_multiple_orphans_all_promoted(self, harness: Harness):
        assert harness._escalation_queue is not None
        _submit_aged(harness._escalation_queue, 'review-a', seconds_ago=300.0)
        _submit_aged(harness._escalation_queue, 'review-b', seconds_ago=300.0)
        _submit_aged(harness._escalation_queue, 'review-c', seconds_ago=10.0)  # young

        assert await harness._reap_orphan_l0_escalations() == 2

        pending = harness._escalation_queue.get_pending()
        l1s = [e for e in pending if e.level == 1]
        l0s = [e for e in pending if e.level == 0]
        assert len(l1s) == 2
        assert len(l0s) == 1  # the young one remains

    @pytest.mark.asyncio
    async def test_done_step_commit_orphan_dismissed_when_subject_terminal_merged(
        self, harness: Harness,
    ):
        """Task 2725: a done-step-commit-class orphan (filed by
        ``TaskWorkflow._escalate_unreconciled_done_step``) whose subject
        task is terminal ('done') and merged
        (``done_provenance.kind='merged'``) is a rebase-superseded false
        positive — the step's content landed on main under a new SHA, so
        the reaper dismisses the orphan instead of promoting a duplicate
        manual-triage L1.
        """
        assert harness._escalation_queue is not None
        orphan = _submit_aged_done_step_commit_orphan(
            harness._escalation_queue, 'task-2679', seconds_ago=300.0,
        )
        harness.scheduler.get_task = AsyncMock(
            return_value={
                'status': 'done',
                'metadata': {'done_provenance': {'kind': 'merged'}},
            },
        )

        count = await harness._reap_orphan_l0_escalations()
        assert count == 0  # nothing promoted

        refreshed = harness._escalation_queue.get(orphan.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        assert refreshed.resolved_by == 'harness-orphan-reaper'

        # No new L1 was created for this orphan.
        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'metadata',
        [
            pytest.param(
                {'done_provenance': {'kind': 'found_on_main'}}, id='found_on_main',
            ),
            pytest.param(
                {'done_provenance': {'kind': 'operational-verified'}},
                id='operational-verified',
            ),
            pytest.param({}, id='provenance-absent'),
        ],
    )
    async def test_done_step_commit_orphan_promoted_when_subject_not_merged(
        self, harness: Harness, metadata: dict,
    ):
        """Task 2725: a done-step-commit-class orphan whose subject task is
        'done' but NOT in the terminal+merged family must still be
        promoted — the dismiss skip is scoped narrowly to positively
        confirmed merges only:

        - 'found_on_main' is deliberately excluded — it is the class this
          skip might otherwise mask, and it already has its own dedicated
          landing guards (PRD 5dd39a4c42, batch 2674-2683).
        - 'operational-verified' is a commitless closure kind — never had
          a step commit to orphan in the first place.
        - Absent provenance can't be positively confirmed benign — fail
          open (promote).

        Fails against step-2's naive status=='done' impl, which wrongly
        dismisses all three instead of promoting.
        """
        assert harness._escalation_queue is not None
        orphan = _submit_aged_done_step_commit_orphan(
            harness._escalation_queue, 'task-not-merged', seconds_ago=300.0,
        )
        harness.scheduler.get_task = AsyncMock(
            return_value={'status': 'done', 'metadata': metadata},
        )

        count = await harness._reap_orphan_l0_escalations()
        assert count == 1

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 1
        assert l1s[0].agent_role == 'harness-orphan-reaper'
        assert l1s[0].task_id == 'task-not-merged'

        refreshed = harness._escalation_queue.get(orphan.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        # Dismissed via the PROMOTION path (superseded-by-L1), not the
        # benign rebase-superseded skip message.
        assert refreshed.resolution is not None
        assert 'Auto-promoted to level 1' in refreshed.resolution
        assert 'rebase-superseded' not in refreshed.resolution


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
