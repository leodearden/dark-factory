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
from orchestrator.harness import (
    Harness,
    _has_fresh_dispatch,
    _has_fresh_merge_phase,
    _has_fresh_stamp,
)
from orchestrator.review_checkpoint import ReviewCheckpoint


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Real Harness with a real EscalationQueue on tmp_path.

    Other internals are mocked; only the reaper path is under test.
    """
    mock_orch_config.orphan_l0_reaper_enabled = True
    mock_orch_config.orphan_l0_timeout_secs = 60.0
    mock_orch_config.orphan_l0_check_interval_secs = 1.0
    # Task 2931: crisp test grace for the divergence-class routing.latest
    # freshness gate, decoupled from the 300.0 source default.
    mock_orch_config.orphan_l0_dispatch_freshness_secs = 120.0

    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    # Task 2878: h.scheduler is a MagicMock (Scheduler is patched above), so
    # is_actively_held() would otherwise return a truthy Mock by default.
    # Default to "not held" so the existing "no live holder -> promote"
    # semantics are preserved for every test in this file; tests exercising
    # the live-recheck gate override this explicitly.
    h.scheduler.is_actively_held = MagicMock(return_value=False)

    # Task 2931: once the reaper consults get_task for the divergence class
    # (step-2), divergence tests that don't stub get_task (notably
    # test_genuine_orphan_with_divergence_still_promoted) would otherwise hit
    # an un-awaitable MagicMock. Default to a benign no-routing.latest return
    # (never fresh -> promote), mirroring task 2878's is_actively_held default.
    # Tests exercising fresh/stale dispatch override this explicitly.
    h.scheduler.get_task = AsyncMock(
        return_value={'status': 'in-progress', 'metadata': {}},
    )

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
    async def test_live_holder_l0_not_promoted(self, harness: Harness):
        """Task 2878: kill the stale-snapshot FP race against live
        redispatches.

        A divergence-class L0 (filed by
        ``TaskWorkflow._check_scope_invariant`` when metadata.files trails
        plan.files during in-flight scope reconciliation) is aged past
        ``orphan_l0_timeout_secs`` and its task_id is no longer in
        ``_escalation_events`` — the dict is popped on workflow-slot exit
        and goes stale across a coordinated batch-redispatch tick, even
        though the scheduler still shows the task actively holding its
        metadata.files locks (a live/re-dispatched workflow). The reaper
        must re-check live holder state via ``Scheduler.is_actively_held``
        before promoting, and skip (fail-safe wait) when it is True.

        RED on main: the reaper ignores live holder state and promotes the
        L0 unconditionally once it's absent from ``_escalation_events`` and
        aged out, so count == 1 and an L1 is filed instead of 0/none.
        """
        assert harness._escalation_queue is not None
        ts = (datetime.now(UTC) - timedelta(seconds=300.0)).isoformat()
        original = Escalation(
            id=harness._escalation_queue.make_id('task-42'),
            task_id='task-42',
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=(
                'plan.files/metadata.files divergence detected for task task-42'
            ),
            detail='detail',
            # MUST NOT be 'verify_wip_reconciliation' — that would route
            # through _is_done_step_commit_orphan's get_task branch, which
            # this test does not stub.
            suggested_action='investigate_and_retry',
            timestamp=ts,
            level=0,
        )
        harness._escalation_queue.submit(original)
        # Deliberately NOT added to harness._escalation_events, so the
        # stale-snapshot gate alone would not skip it.

        # Scheduler shows the task actively holding its metadata.files
        # locks right now — a live/re-dispatched workflow.
        harness.scheduler.is_actively_held = MagicMock(return_value=True)

        assert await harness._reap_orphan_l0_escalations() == 0

        refreshed = harness._escalation_queue.get(original.id)
        assert refreshed is not None
        assert refreshed.status == 'pending'

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 0

    @pytest.mark.asyncio
    async def test_genuine_orphan_with_divergence_still_promoted(
        self, harness: Harness,
    ):
        """Task 2878 boundary guard: the live-recheck gate must not
        over-suppress a genuinely stranded orphan.

        Same divergence-class aged L0 as
        ``test_live_holder_l0_not_promoted``, but here the task has no
        live holder (``is_actively_held`` stays at the pre-1 fixture
        default of False — no dispatch slot, no module lock, no recent
        cancel stamp): a genuinely dead workflow. It must still be
        promoted, both before and after the step-2 impl — this pins that
        the gate defers only live tasks, never drops a genuine orphan.
        """
        assert harness._escalation_queue is not None
        ts = (datetime.now(UTC) - timedelta(seconds=300.0)).isoformat()
        original = Escalation(
            id=harness._escalation_queue.make_id('task-77'),
            task_id='task-77',
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=(
                'plan.files/metadata.files divergence detected for task task-77'
            ),
            detail='detail',
            suggested_action='investigate_and_retry',
            timestamp=ts,
            level=0,
        )
        harness._escalation_queue.submit(original)
        # Not in _escalation_events, and is_actively_held stays False (the
        # pre-1 fixture default) — genuinely stranded, no live holder.

        assert await harness._reap_orphan_l0_escalations() == 1

        refreshed = harness._escalation_queue.get(original.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        assert refreshed.resolved_by == 'harness-orphan-reaper'

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 1
        assert l1s[0].task_id == 'task-77'

    @pytest.mark.asyncio
    async def test_live_lockfree_dispatch_divergence_not_promoted(
        self, harness: Harness,
    ):
        """Task 2931: defer a divergence-class L0 while its task is live
        inside a lock-free reviewer_comprehensive/resettled_adjudicator
        stage.

        Those stages ``self._invoke(...)`` an agent role that holds NO
        module locks and is absent from ``_dispatched``, so
        ``is_actively_held`` correctly returns False — yet the task is
        genuinely live mid-dispatch, its ``metadata.files`` legitimately
        lagging ``plan.files`` during in-flight scope reconciliation. The FP
        recurred post-2878 (esc-2865-19, esc-2869-10). The liveness signal
        these stages DO leave behind is ``metadata.routing.latest.decided_at``
        (stamped fresh per LLM invocation). Here it is 5s old — well within
        the 120s test grace — so the reaper must defer (not promote) the
        divergence L0.

        RED on main: the reaper has no routing-freshness gate, so the aged,
        not-held orphan is promoted (count == 1, an L1 filed) instead of 0.
        """
        assert harness._escalation_queue is not None
        harness.config.orphan_l0_dispatch_freshness_secs = 120.0
        # Reviewer/adjudicator stage holds no locks and isn't dispatched.
        harness.scheduler.is_actively_held = MagicMock(return_value=False)

        ts = (datetime.now(UTC) - timedelta(seconds=300.0)).isoformat()
        original = Escalation(
            id=harness._escalation_queue.make_id('task-42'),
            task_id='task-42',
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=(
                'plan.files/metadata.files divergence detected for task task-42'
            ),
            detail='detail',
            suggested_action='investigate_and_retry',
            timestamp=ts,
            level=0,
        )
        harness._escalation_queue.submit(original)
        # Deliberately NOT in _escalation_events (stale-snapshot gate would
        # not skip it) and is_actively_held is False (no locks) — only the
        # routing-freshness signal proves the task is live.

        fresh_decided_at = (
            datetime.now(UTC) - timedelta(seconds=5.0)
        ).isoformat()
        harness.scheduler.get_task = AsyncMock(
            return_value={
                'status': 'in-progress',
                'metadata': {
                    'routing': {'latest': {'decided_at': fresh_decided_at}},
                },
            },
        )

        assert await harness._reap_orphan_l0_escalations() == 0

        refreshed = harness._escalation_queue.get(original.id)
        assert refreshed is not None
        assert refreshed.status == 'pending'

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'case',
        [
            pytest.param('stale-routing', id='stale-routing'),
            pytest.param('get-task-none', id='get-task-none'),
            pytest.param('no-routing', id='no-routing'),
        ],
    )
    async def test_stranded_divergence_still_promoted(
        self, harness: Harness, case: str,
    ):
        """Task 2931 boundary guard: the routing-freshness gate defers ONLY
        live/fresh dispatches and never drops a genuinely stranded orphan
        (preserves task 2878's boundary).

        Same aged (300s) divergence-class L0 and ``is_actively_held=False``
        as ``test_live_lockfree_dispatch_divergence_not_promoted``, but here
        the ``routing.latest`` liveness signal is NOT fresh, so
        ``_has_fresh_dispatch`` returns False and the orphan is promoted:

        - 'stale-routing': ``decided_at`` is 300s old (> the 120s grace) —
          the decision aged out, so the task is stranded, not live.
        - 'get-task-none': ``get_task`` cannot confirm the task (``None``) —
          fail open (promote).
        - 'no-routing': ``metadata`` has no ``routing`` key — no liveness
          signal to confirm — fail open (promote).
        """
        assert harness._escalation_queue is not None
        harness.config.orphan_l0_dispatch_freshness_secs = 120.0
        harness.scheduler.is_actively_held = MagicMock(return_value=False)

        # Build the get_task return at runtime so relative timestamps compute
        # against the current sweep clock.
        if case == 'stale-routing':
            stale_decided_at = (
                datetime.now(UTC) - timedelta(seconds=300.0)
            ).isoformat()
            harness.scheduler.get_task = AsyncMock(
                return_value={
                    'status': 'in-progress',
                    'metadata': {
                        'routing': {
                            'latest': {'decided_at': stale_decided_at},
                        },
                    },
                },
            )
        elif case == 'get-task-none':
            harness.scheduler.get_task = AsyncMock(return_value=None)
        else:  # 'no-routing'
            harness.scheduler.get_task = AsyncMock(
                return_value={'status': 'in-progress', 'metadata': {}},
            )

        ts = (datetime.now(UTC) - timedelta(seconds=300.0)).isoformat()
        original = Escalation(
            id=harness._escalation_queue.make_id('task-stranded'),
            task_id='task-stranded',
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=(
                'plan.files/metadata.files divergence detected for task '
                'task-stranded'
            ),
            detail='detail',
            suggested_action='investigate_and_retry',
            timestamp=ts,
            level=0,
        )
        harness._escalation_queue.submit(original)
        # Not in _escalation_events, is_actively_held False — genuinely
        # stranded, and the routing signal is non-fresh for every case.

        assert await harness._reap_orphan_l0_escalations() == 1

        refreshed = harness._escalation_queue.get(original.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        assert refreshed.resolved_by == 'harness-orphan-reaper'
        assert refreshed.resolution is not None
        assert 'Auto-promoted to level 1' in refreshed.resolution

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 1
        assert l1s[0].task_id == 'task-stranded'

    @pytest.mark.asyncio
    async def test_live_merge_phase_divergence_not_promoted(
        self, harness: Harness,
    ):
        """Task 2991: defer a divergence-class L0 while its task is live in
        the pre-enqueue MERGE phase.

        The MERGE-phase loop (rebase + scoped verify + queue submit) makes NO
        LLM calls, so it never refreshes
        ``metadata.routing.latest.decided_at`` — a legitimately-live
        merge-stage task therefore fails the task-2931 ``_has_fresh_dispatch``
        gate and (pre-2991) is false-promoted exactly like the pre-2931 bug
        (cluster esc-2789-22: esc-2789-21, esc-2885-7, both
        workflow_state='merge'). The durable liveness signal it DOES leave is
        ``metadata.merge_phase_liveness.entered_at``, stamped at merge entry.
        Here it is 5s old — within the 120s test grace — and ``routing.latest``
        is absent, so ONLY the merge-phase gate can defer.

        RED on main: the reaper has no merge-phase-freshness gate, so the
        aged, not-held orphan is promoted (count == 1, an L1 filed) instead of
        0.
        """
        assert harness._escalation_queue is not None
        harness.config.orphan_l0_merge_phase_freshness_secs = 120.0
        # Merge-phase workflow holds no locks and isn't dispatched.
        harness.scheduler.is_actively_held = MagicMock(return_value=False)

        ts = (datetime.now(UTC) - timedelta(seconds=300.0)).isoformat()
        original = Escalation(
            id=harness._escalation_queue.make_id('task-merge'),
            task_id='task-merge',
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=(
                'plan.files/metadata.files divergence detected for task '
                'task-merge'
            ),
            detail='detail',
            suggested_action='investigate_and_retry',
            timestamp=ts,
            level=0,
        )
        harness._escalation_queue.submit(original)
        # Deliberately NOT in _escalation_events and is_actively_held False —
        # only the durable merge-phase stamp proves the task is live.

        fresh_entered_at = (
            datetime.now(UTC) - timedelta(seconds=5.0)
        ).isoformat()
        # routing.latest deliberately ABSENT (a merge phase makes no LLM
        # calls), so _has_fresh_dispatch is False and only the merge-phase
        # gate can defer.
        harness.scheduler.get_task = AsyncMock(
            return_value={
                'status': 'in-progress',
                'workflow_state': 'merge',
                'metadata': {
                    'merge_phase_liveness': {'entered_at': fresh_entered_at},
                },
            },
        )

        assert await harness._reap_orphan_l0_escalations() == 0

        refreshed = harness._escalation_queue.get(original.id)
        assert refreshed is not None
        assert refreshed.status == 'pending'

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 0

    @pytest.mark.asyncio
    async def test_post_enqueue_restart_still_promotes_known_accepted_gap(
        self, harness: Harness,
    ):
        """Characterization test — pins a KNOWN, ACCEPTED gap, not desired
        behaviour (review amendment; see the comment at the discharge site in
        ``_submit_to_merge_queue``).

        The durable stamp is discharged at the enqueue boundary, so the
        POST-enqueue window (request on the merge journal; worker rebasing /
        verifying / merging — the code's own ETA heuristic is 600s and its
        cold-verify ceiling 7200s) carries no durable liveness signal:
        ``merge_phase_liveness`` is gone and ``routing.latest`` is still stale
        (the merge worker makes no LLM calls either). In-process that window
        is covered by ``is_actively_held``; after a restart ``_dispatched`` is
        empty, and neither restart path re-stamps —
        ``Harness._recover_pending_merges`` hands the rebuilt request to the
        WORKER without re-entering ``_run_merge_phase``, and
        ``_reconcile_stranded_in_progress`` LEAVEs (does not re-dispatch) a
        task that has an open escalation, which the merge-entry divergence L0
        is. So this class of restart still promotes.

        Same setup as the pre-enqueue restart test above, differing ONLY in
        that the stamp has been discharged — which is exactly the difference
        the gap is made of.
        """
        assert harness._escalation_queue is not None
        harness.config.orphan_l0_merge_phase_freshness_secs = 120.0

        bare_scheduler = MagicMock()
        bare_scheduler.is_actively_held = MagicMock(return_value=False)
        bare_scheduler.get_task = AsyncMock(
            return_value={
                'status': 'in-progress',
                'workflow_state': 'merge',
                # Enqueued: stamp discharged, routing.latest still stale.
                'metadata': {},
            },
        )
        harness.scheduler = bare_scheduler

        ts = (datetime.now(UTC) - timedelta(seconds=300.0)).isoformat()
        original = Escalation(
            id=harness._escalation_queue.make_id('task-postq'),
            task_id='task-postq',
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=(
                'plan.files/metadata.files divergence detected for task '
                'task-postq'
            ),
            detail='detail',
            suggested_action='investigate_and_retry',
            timestamp=ts,
            level=0,
        )
        harness._escalation_queue.submit(original)

        # ACCEPTED: promoted to a human-facing L1 despite the merge being
        # genuinely live in the queue. Moving the discharge to the merge
        # terminal outcome would close this (deferral bounded by
        # orphan_l0_merge_phase_freshness_secs, never suppression) — filed as
        # follow-up rather than changed here, because the enqueue boundary is
        # where task 2753's in-memory grace clears too and the pairing is
        # pinned by TestEnqueueClearWiring.
        assert await harness._reap_orphan_l0_escalations() == 1

    @pytest.mark.asyncio
    async def test_merge_phase_deferral_survives_restart_no_inmemory_state(
        self, harness: Harness,
    ):
        """Task 2991 fix-direction (b): the merge-phase deferral relies SOLELY
        on durable task metadata, not the per-process
        ``Scheduler._merge_phase_at``.

        Simulate immediately-after-orchestrator-restart: a bare MagicMock
        scheduler carrying NO in-memory merge-phase state at all (no
        ``_merge_phase_at`` populated, ``note_merge_phase_entered`` not yet
        re-run for this process). The only liveness evidence is the durable
        ``metadata.merge_phase_liveness.entered_at`` stamp persisted before
        the restart. The reaper must still defer — proving restart-
        survivability across the exact window in which esc-2789-21 /
        esc-2885-7 were not re-dispatched.
        """
        assert harness._escalation_queue is not None
        harness.config.orphan_l0_merge_phase_freshness_secs = 120.0

        # Bare, freshly-restarted scheduler: only get_task (durable metadata)
        # + is_actively_held are configured; nothing merge-phase in-memory.
        bare_scheduler = MagicMock()
        bare_scheduler.is_actively_held = MagicMock(return_value=False)
        fresh_entered_at = (
            datetime.now(UTC) - timedelta(seconds=5.0)
        ).isoformat()
        bare_scheduler.get_task = AsyncMock(
            return_value={
                'status': 'in-progress',
                'workflow_state': 'merge',
                'metadata': {
                    'merge_phase_liveness': {'entered_at': fresh_entered_at},
                },
            },
        )
        harness.scheduler = bare_scheduler

        ts = (datetime.now(UTC) - timedelta(seconds=300.0)).isoformat()
        original = Escalation(
            id=harness._escalation_queue.make_id('task-restart'),
            task_id='task-restart',
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=(
                'plan.files/metadata.files divergence detected for task '
                'task-restart'
            ),
            detail='detail',
            suggested_action='investigate_and_retry',
            timestamp=ts,
            level=0,
        )
        harness._escalation_queue.submit(original)

        assert await harness._reap_orphan_l0_escalations() == 0

        refreshed = harness._escalation_queue.get(original.id)
        assert refreshed is not None
        assert refreshed.status == 'pending'

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 0
        # The deferral's sole liveness input was durable metadata via
        # get_task — no in-memory merge-phase accessor was consulted.
        bare_scheduler.get_task.assert_awaited_once_with('task-restart')

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'case',
        [
            pytest.param('stale-merge-phase', id='stale-merge-phase'),
            pytest.param('no-merge-phase', id='no-merge-phase'),
            pytest.param('get-task-none', id='get-task-none'),
        ],
    )
    async def test_stranded_merge_phase_divergence_still_promoted(
        self, harness: Harness, case: str,
    ):
        """Task 2991 boundary guard: the merge-phase gate defers ONLY fresh
        stamps and never drops a genuinely stranded orphan (preserves the
        task 2878/2931 boundary).

        Same aged (300s) divergence-class L0 and ``is_actively_held=False`` as
        the live case, but here ``routing.latest`` is ALSO stale so
        ``_has_fresh_dispatch`` is False too — the promotion therefore rests
        solely on the merge-phase gate ALSO being non-fresh:

        - 'stale-merge-phase': ``entered_at`` 300s old (> the 120s grace) —
          the task stopped cycling through merge, so it is stranded.
        - 'no-merge-phase': ``metadata`` has no ``merge_phase_liveness`` key —
          no signal to confirm — fail open (promote).
        - 'get-task-none': ``get_task`` returns ``None`` — fail open (promote).
        """
        assert harness._escalation_queue is not None
        harness.config.orphan_l0_dispatch_freshness_secs = 120.0
        harness.config.orphan_l0_merge_phase_freshness_secs = 120.0
        harness.scheduler.is_actively_held = MagicMock(return_value=False)

        # routing.latest is stale in every non-None case so the dispatch gate
        # (task 2931) is also False — isolating the merge-phase gate as the
        # sole remaining defer opportunity.
        stale_ts = (datetime.now(UTC) - timedelta(seconds=300.0)).isoformat()
        if case == 'stale-merge-phase':
            harness.scheduler.get_task = AsyncMock(
                return_value={
                    'status': 'in-progress',
                    'workflow_state': 'merge',
                    'metadata': {
                        'routing': {'latest': {'decided_at': stale_ts}},
                        'merge_phase_liveness': {'entered_at': stale_ts},
                    },
                },
            )
        elif case == 'no-merge-phase':
            harness.scheduler.get_task = AsyncMock(
                return_value={
                    'status': 'in-progress',
                    'metadata': {
                        'routing': {'latest': {'decided_at': stale_ts}},
                    },
                },
            )
        else:  # 'get-task-none'
            harness.scheduler.get_task = AsyncMock(return_value=None)

        ts = (datetime.now(UTC) - timedelta(seconds=300.0)).isoformat()
        original = Escalation(
            id=harness._escalation_queue.make_id('task-merge-stranded'),
            task_id='task-merge-stranded',
            agent_role='orchestrator',
            severity='blocking',
            category='infra_issue',
            summary=(
                'plan.files/metadata.files divergence detected for task '
                'task-merge-stranded'
            ),
            detail='detail',
            suggested_action='investigate_and_retry',
            timestamp=ts,
            level=0,
        )
        harness._escalation_queue.submit(original)

        assert await harness._reap_orphan_l0_escalations() == 1

        refreshed = harness._escalation_queue.get(original.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        assert refreshed.resolved_by == 'harness-orphan-reaper'
        assert refreshed.resolution is not None
        assert 'Auto-promoted to level 1' in refreshed.resolution

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 1
        assert l1s[0].task_id == 'task-merge-stranded'

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
        # Positively distinguish this benign rebase-superseded dismiss path
        # from the has_open_l1 dismiss branch and the promotion path, both
        # of which also use resolved_by='harness-orphan-reaper'.
        assert refreshed.resolution is not None
        assert 'rebase-superseded' in refreshed.resolution

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

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'kind',
        [
            pytest.param(
                'dispatch-gate-marker-found', id='dispatch-gate-marker-found',
            ),
            pytest.param(
                'dispatch-gate-content-equivalent',
                id='dispatch-gate-content-equivalent',
            ),
        ],
    )
    async def test_done_step_commit_orphan_dismissed_for_other_merged_kinds(
        self, harness: Harness, kind: str,
    ):
        """Task 2725: the merged family covers three done_provenance.kind
        values. 'merged' is covered by
        test_done_step_commit_orphan_dismissed_when_subject_terminal_merged
        above — this locks in the remaining two dispatch-gate kinds.
        """
        assert harness._escalation_queue is not None
        orphan = _submit_aged_done_step_commit_orphan(
            harness._escalation_queue, 'task-merged-family', seconds_ago=300.0,
        )
        harness.scheduler.get_task = AsyncMock(
            return_value={
                'status': 'done',
                'metadata': {'done_provenance': {'kind': kind}},
            },
        )

        count = await harness._reap_orphan_l0_escalations()
        assert count == 0  # nothing promoted

        refreshed = harness._escalation_queue.get(orphan.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        assert refreshed.resolved_by == 'harness-orphan-reaper'
        # Positively distinguish this benign rebase-superseded dismiss path
        # from the has_open_l1 dismiss branch and the promotion path, both
        # of which also use resolved_by='harness-orphan-reaper'.
        assert refreshed.resolution is not None
        assert 'rebase-superseded' in refreshed.resolution

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'status',
        [
            pytest.param('pending', id='pending'),
            pytest.param('in-progress', id='in-progress'),
            pytest.param('blocked', id='blocked'),
        ],
    )
    async def test_done_step_commit_orphan_promoted_when_subject_still_live(
        self, harness: Harness, status: str,
    ):
        """Task 2725: a done-step-commit orphan whose subject task has not
        yet reached a terminal status is promoted as before — the
        rebase-superseded false positive only exists once the task has
        actually reached done+merged.
        """
        assert harness._escalation_queue is not None
        orphan = _submit_aged_done_step_commit_orphan(
            harness._escalation_queue, 'task-still-live', seconds_ago=300.0,
        )
        harness.scheduler.get_task = AsyncMock(
            return_value={'status': status, 'metadata': {}},
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
        assert l1s[0].task_id == 'task-still-live'

        refreshed = harness._escalation_queue.get(orphan.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        assert refreshed.resolution is not None
        assert 'Auto-promoted to level 1' in refreshed.resolution

    @pytest.mark.asyncio
    async def test_non_step_commit_class_orphan_unaffected_by_status_lookup(
        self, harness: Harness,
    ):
        """Task 2725: the terminal+merged skip is scoped to the
        done-step-commit class only. A differently-classed aged orphan
        (agent_role='integration-reviewer', category='design_concern' —
        see ``_submit_aged``) on a done+merged task is promoted exactly as
        before, and the class-gate short-circuits before ever awaiting
        ``scheduler.get_task``.
        """
        assert harness._escalation_queue is not None
        orphan = _submit_aged(
            harness._escalation_queue, 'task-other-class', seconds_ago=300.0,
        )
        harness.scheduler.get_task = AsyncMock(
            return_value={
                'status': 'done',
                'metadata': {'done_provenance': {'kind': 'merged'}},
            },
        )

        count = await harness._reap_orphan_l0_escalations()
        assert count == 1
        harness.scheduler.get_task.assert_not_awaited()

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 1
        assert l1s[0].task_id == 'task-other-class'

        refreshed = harness._escalation_queue.get(orphan.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'

    @pytest.mark.asyncio
    async def test_done_step_commit_orphan_promoted_when_get_task_returns_none(
        self, harness: Harness,
    ):
        """Task 2725: fail-safe — if scheduler.get_task cannot confirm the
        subject task (``None`` on failure or absence), the orphan is
        promoted rather than silently dismissed.
        """
        assert harness._escalation_queue is not None
        orphan = _submit_aged_done_step_commit_orphan(
            harness._escalation_queue, 'task-unknown', seconds_ago=300.0,
        )
        harness.scheduler.get_task = AsyncMock(return_value=None)

        count = await harness._reap_orphan_l0_escalations()
        assert count == 1

        all_escs = [
            harness._escalation_queue.get(p.stem)
            for p in (harness._escalation_queue.queue_dir).glob('esc-*.json')
        ]
        l1s = [e for e in all_escs if e and e.level == 1]
        assert len(l1s) == 1
        assert l1s[0].task_id == 'task-unknown'

        refreshed = harness._escalation_queue.get(orphan.id)
        assert refreshed is not None
        assert refreshed.status == 'dismissed'
        assert refreshed.resolution is not None
        assert 'Auto-promoted to level 1' in refreshed.resolution


class TestHasFreshDispatch:
    """Task 2931: direct unit coverage of ``_has_fresh_dispatch``'s
    ``except (ValueError, TypeError)`` fail-open branch.

    The reaper tests (``test_live_lockfree_dispatch_divergence_not_promoted``
    and ``test_stranded_divergence_still_promoted``) already exercise the
    fresh / stale / get-task-None / no-routing-key paths indirectly. The
    unparseable / tz-naive ``decided_at`` branch is NOT covered by them, yet
    it is exactly the branch that would silently flip the gate from 'defer'
    to 'promote' — defeating the divergence-FP suppression without any test
    catching it — if the routing writer's timestamp format ever drifted
    (a non-ISO string, or a tz-naive stamp on an older Python). Pin the
    documented fail-safe directly: both malformed inputs must return False
    (fail open -> the caller promotes).
    """

    def test_garbage_decided_at_fails_open(self):
        """A non-ISO / garbage ``decided_at`` raises ValueError inside
        ``datetime.fromisoformat`` -> caught -> False (fail open)."""
        task = {
            'status': 'in-progress',
            'metadata': {
                'routing': {'latest': {'decided_at': 'not-a-timestamp'}},
            },
        }
        assert _has_fresh_dispatch(task, datetime.now(UTC), 120.0) is False

    def test_tz_naive_decided_at_fails_open(self):
        """A tz-naive ``decided_at`` subtracted from a tz-aware ``now`` raises
        TypeError -> caught -> False (fail open).

        The stamp is deliberately FRESH (5s old, well within the 120s grace),
        so the ONLY reason for False is the tz mismatch, not staleness — this
        pins the fail-open branch specifically, not the stale-decision path.
        """
        naive_fresh = (
            (datetime.now(UTC) - timedelta(seconds=5.0))
            .replace(tzinfo=None)
            .isoformat()
        )
        task = {
            'status': 'in-progress',
            'metadata': {
                'routing': {'latest': {'decided_at': naive_fresh}},
            },
        }
        now = datetime.now(UTC)  # tz-aware
        assert _has_fresh_dispatch(task, now, 120.0) is False


class TestHasFreshMergePhase:
    """Task 2991: direct unit coverage of ``_has_fresh_merge_phase``'s
    ``except (ValueError, TypeError)`` fail-open branch.

    The reaper tests (``test_live_merge_phase_divergence_not_promoted`` and
    ``test_stranded_merge_phase_divergence_still_promoted``) already exercise
    the fresh / stale / get-task-None / no-merge_phase_liveness-key paths
    indirectly. The unparseable / tz-naive ``entered_at`` branch is NOT
    covered by them, yet it is exactly the branch that would silently flip the
    gate from 'defer' to 'promote' — defeating the merge-phase-FP suppression
    without any test catching it — if the stamp writer's timestamp format ever
    drifted (a non-ISO string, or a tz-naive stamp). Pin the documented
    fail-safe directly: both malformed inputs must return False (fail open ->
    the caller promotes), mirroring ``TestHasFreshDispatch``.
    """

    def test_garbage_entered_at_fails_open(self):
        """A non-ISO / garbage ``entered_at`` raises ValueError inside
        ``datetime.fromisoformat`` -> caught -> False (fail open)."""
        task = {
            'status': 'in-progress',
            'metadata': {
                'merge_phase_liveness': {'entered_at': 'not-a-timestamp'},
            },
        }
        assert _has_fresh_merge_phase(task, datetime.now(UTC), 120.0) is False

    def test_tz_naive_entered_at_fails_open(self):
        """A tz-naive ``entered_at`` subtracted from a tz-aware ``now`` raises
        TypeError -> caught -> False (fail open).

        The stamp is deliberately FRESH (5s old, well within the 120s grace),
        so the ONLY reason for False is the tz mismatch, not staleness — this
        pins the fail-open branch specifically, not the stale-stamp path.
        """
        naive_fresh = (
            (datetime.now(UTC) - timedelta(seconds=5.0))
            .replace(tzinfo=None)
            .isoformat()
        )
        task = {
            'status': 'in-progress',
            'metadata': {
                'merge_phase_liveness': {'entered_at': naive_fresh},
            },
        }
        now = datetime.now(UTC)  # tz-aware
        assert _has_fresh_merge_phase(task, now, 120.0) is False


class TestHasFreshStamp:
    """The shared path-walking implementation behind both liveness gates.

    ``_has_fresh_dispatch`` and ``_has_fresh_merge_phase`` differ ONLY in
    their key path, so they delegate to ``_has_fresh_stamp`` (amendment for
    the code-duplication review finding). The two wrapper classes above stay
    as the per-gate contract pins; this class pins the generic parts they
    share — the path walk itself and the fail-safe branches that a future
    refinement (tz-naive handling, negative-delta clamping) would touch.
    """

    _PATH = ('merge_phase_liveness', 'entered_at')

    def test_walks_arbitrary_path_and_measures_freshness(self):
        fresh = (datetime.now(UTC) - timedelta(seconds=5.0)).isoformat()
        task = {'metadata': {'a': {'b': {'c': fresh}}}}
        assert _has_fresh_stamp(task, datetime.now(UTC), 120.0, 'a', 'b', 'c')
        # Same stamp, tighter grace -> stale.
        assert _has_fresh_stamp(
            task, datetime.now(UTC), 1.0, 'a', 'b', 'c',
        ) is False

    def test_stamp_newer_than_now_is_fresh(self):
        """A stamp written AFTER the sweep snapshot yields a negative delta,
        which is ``< grace_secs`` -> fresh. Pins the documented "newer than
        the sweep snapshot" wording for both gates at once."""
        future = (datetime.now(UTC) + timedelta(seconds=30.0)).isoformat()
        task = {'metadata': {'merge_phase_liveness': {'entered_at': future}}}
        assert _has_fresh_stamp(task, datetime.now(UTC), 120.0, *self._PATH)

    @pytest.mark.parametrize(
        'task',
        [
            None,                                        # unreadable task
            {},                                          # no metadata
            {'metadata': 'not-a-dict'},                  # non-dict metadata
            {'metadata': {}},                            # no such path
            {'metadata': {'merge_phase_liveness': 'x'}},  # non-dict node
            {'metadata': {'merge_phase_liveness': {}}},  # no leaf
            {'metadata': {'merge_phase_liveness': {'entered_at': 17}}},  # non-str
        ],
        ids=[
            'task-none', 'no-metadata', 'metadata-non-dict', 'path-missing',
            'node-non-dict', 'leaf-missing', 'leaf-non-str',
        ],
    )
    def test_malformed_inputs_fail_open(self, task):
        """Every guard returns False (never raises), so the reaper PROMOTES
        (surfaces) rather than silently suppressing whenever liveness cannot
        be positively confirmed — task 2878's boundary guard."""
        assert _has_fresh_stamp(task, datetime.now(UTC), 120.0, *self._PATH) is False

    def test_empty_path_fails_open(self):
        """Degenerate call (no key path) must not raise on the leaf unpack."""
        task = {'metadata': {'merge_phase_liveness': {'entered_at': 'x'}}}
        assert _has_fresh_stamp(task, datetime.now(UTC), 120.0) is False


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
