"""The reviewer leg of the info-L0 router, driven through the workflow.

plans/info-l0-disposition-router-prd.md D2 / D11 (task 5224): the reviewer is
briefed with its task's open info notes, and the review loop applies the
dispositions its verdict carries.
"""

from __future__ import annotations

import logging

import pytest
from _info_l0_router_fixtures import (
    TASK_ID,
    DispositioningReviewer,
    LostVerdictReviewer,
    RouterRun,
    disposition_entry,
)
from _workflow_helpers import (
    _derive_meta_root_like_production as _derive_meta_root_like_production,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.agents.review_briefing import ReviewBriefing
from orchestrator.escalation_router import REVIEWER_LEG_ID, read_hold
from orchestrator.workflow import WorkflowOutcome

STARVATION = 'orchestrator-starvation-watchdog'
UNKNOWN_ID = 'esc-42-999'
CORRUPTION = {
    'agent_role': 'orchestrator', 'category': 'infra_issue',
    'suggested_action': 'investigate_log_corruption',
}


def _file_note(
    queue: EscalationQueue, *, task_id: str = TASK_ID, severity: str = 'info',
    agent_role: str = 'implementer', category: str = 'design_concern',
    suggested_action: str = 'create_followup_task',
) -> Escalation:
    note = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role=agent_role,
        severity=severity, category=category,
        summary=f'{agent_role} {severity} note', detail='Seen while editing foo.py.',
        suggested_action=suggested_action,
    )
    queue.submit(note)
    return note


@pytest.mark.asyncio
class TestReviewerIsBriefedWithItsNotes:
    async def test_the_diff_only_reviewer_sees_exactly_the_eligible_notes(
        self, tmp_path, monkeypatch,
    ):
        run = RouterRun(tmp_path, monkeypatch, DispositioningReviewer())
        eligible = _file_note(run.queue)
        _file_note(run.queue, severity='blocking')
        _file_note(run.queue, agent_role=STARVATION)
        _file_note(run.queue, task_id='99')
        _file_note(run.queue, **CORRUPTION)

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.briefed == [(ReviewBriefing.DIFF_ONLY, [eligible.id])]

    async def test_the_context_complete_reviewer_sees_the_same_notes(
        self, tmp_path, monkeypatch,
    ):
        run = RouterRun(
            tmp_path, monkeypatch, DispositioningReviewer(),
            briefing=ReviewBriefing.CONTEXT_COMPLETE,
        )
        eligible = _file_note(run.queue)
        _file_note(run.queue, agent_role=STARVATION)

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.briefed == [(ReviewBriefing.CONTEXT_COMPLETE, [eligible.id])]

    async def test_a_disabled_router_briefs_no_notes(self, tmp_path, monkeypatch):
        run = RouterRun(tmp_path, monkeypatch, DispositioningReviewer(), router_enabled=False)
        _file_note(run.queue)

        assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.briefed == [(ReviewBriefing.DIFF_ONLY, [])]

    async def test_a_queue_fault_costs_the_notes_not_the_review(
        self, tmp_path, monkeypatch, caplog,
    ):
        run = RouterRun(tmp_path, monkeypatch, DispositioningReviewer())
        _file_note(run.queue)
        run.fail_the_first_note_read_after_verification()

        with caplog.at_level(logging.WARNING, logger='orchestrator.workflow'):
            assert await run.run() == WorkflowOutcome.DONE

        assert run.briefing.briefed == [(ReviewBriefing.DIFF_ONLY, [])]
        assert any(
            record.levelno == logging.WARNING and 'disk unreadable' in record.getMessage()
            for record in caplog.records
        )


# ---------------------------------------------------------------------------
# The review loop applies the verdict's dispositions
# (boundary rows #1, #2 first half, #17)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReviewLoopAppliesDispositions:
    @pytest.mark.parametrize(('disposition', 'resolution_class'), [
        ('addressed', 'addressed'),
        ('no_action', 'observation-consumed'),
    ])
    async def test_a_closing_disposition_closes_the_note_before_the_exit(
        self, tmp_path, monkeypatch, disposition, resolution_class,
    ):
        stub = DispositioningReviewer()
        run = RouterRun(tmp_path, monkeypatch, stub)
        note = _file_note(run.queue)
        stub.note_dispositions = [disposition_entry(note.id, disposition)]

        assert await run.run() == WorkflowOutcome.DONE

        closed = run.current(note)
        assert (closed.resolution_class, closed.resolved_by) == (resolution_class, REVIEWER_LEG_ID)
        (row,) = run.rows(note.id)
        assert (row['by'], row['class'], row['exit_kind']) == (
            REVIEWER_LEG_ID, resolution_class, None,
        )
        assert run.curator.calls_named('submit_task') == []

    async def test_work_leaves_the_note_for_the_exit_router_to_convert(
        self, tmp_path, monkeypatch,
    ):
        stub = DispositioningReviewer()
        run = RouterRun(tmp_path, monkeypatch, stub)
        note = _file_note(run.queue)
        stub.note_dispositions = [disposition_entry(note.id, 'work')]

        assert await run.run() == WorkflowOutcome.DONE

        reviewer_row, finalizer_row = run.rows(note.id)
        assert (reviewer_row['by'], reviewer_row['class'], reviewer_row['reason']) == (
            REVIEWER_LEG_ID, 'defer', 'work',
        )
        assert (finalizer_row['by'], finalizer_row['class'], finalizer_row['exit_kind']) == (
            'workflow-finalizer', 'convert-pending', 'done',
        )
        hold = read_hold(run.current(note))
        assert hold is not None and hold.ticket == 'tkt_1'
        (submitted,) = run.curator.calls_named('submit_task')
        assert 'work rationale' in submitted['description']

    async def test_work_overrides_the_authors_no_action_declaration_at_the_exit(
        self, tmp_path, monkeypatch,
    ):
        stub = DispositioningReviewer()
        run = RouterRun(tmp_path, monkeypatch, stub)
        note = _file_note(run.queue, suggested_action='no action needed for this task')
        stub.note_dispositions = [disposition_entry(note.id, 'work')]

        assert await run.run() == WorkflowOutcome.DONE

        _reviewer_row, finalizer_row = run.rows(note.id)
        assert (finalizer_row['class'], finalizer_row['exit_kind']) == ('convert-pending', 'done')
        assert run.current(note).status == 'pending'
        assert len(run.curator.calls_named('submit_task')) == 1

    async def test_a_disposition_of_a_note_the_reviewer_was_not_shown_is_rejected(
        self, tmp_path, monkeypatch,
    ):
        stub = DispositioningReviewer()
        run = RouterRun(tmp_path, monkeypatch, stub)
        foreign = _file_note(run.queue, task_id='99')
        stub.note_dispositions = [
            disposition_entry(UNKNOWN_ID, 'addressed'), disposition_entry(foreign.id, 'no_action'),
        ]

        assert await run.run() == WorkflowOutcome.DONE

        after = run.current(foreign)
        assert (after.status, after.triaged_by, after.resolution) == ('pending', None, None)
        for esc_id in (UNKNOWN_ID, foreign.id):
            (row,) = run.rows(esc_id)
            assert (row['by'], row['class'], row['task_id']) == (
                REVIEWER_LEG_ID, 'rejected_disposition', TASK_ID,
            )

    async def test_a_lost_verdict_carries_no_dispositions_and_the_exit_router_routes_the_note(
        self, tmp_path, monkeypatch,
    ):
        stub = LostVerdictReviewer()
        run = RouterRun(tmp_path, monkeypatch, stub)
        note = _file_note(run.queue)
        stub.note_dispositions = [disposition_entry(note.id, 'addressed')]

        assert await run.run() == WorkflowOutcome.BLOCKED

        artifacts = run.workflow.artifacts
        assert artifacts is not None
        ((role, review),) = artifacts.read_reviews().items()
        assert review['verdict'] == 'ERROR'
        envelope = artifacts.read_verdict(role)
        assert envelope is not None
        assert envelope['verdict']['verdict'] == 'MAYBE'
        assert envelope['verdict']['note_dispositions'] == stub.note_dispositions

        rows = run.rows(note.id)
        assert [row['by'] for row in rows] == ['workflow-finalizer']
        assert (rows[0]['class'], rows[0]['exit_kind']) == ('convert-pending', 'blocked')
        hold = read_hold(run.current(note))
        assert hold is not None and hold.ticket == 'tkt_1'

    async def test_a_harness_notice_is_left_to_the_exit_router(self, tmp_path, monkeypatch):
        stub = DispositioningReviewer()
        run = RouterRun(tmp_path, monkeypatch, stub)
        notice = _file_note(run.queue, **CORRUPTION)
        stub.note_dispositions = [disposition_entry(notice.id, 'no_action')]

        assert await run.run() == WorkflowOutcome.DONE

        reviewer_row, finalizer_row = run.rows(notice.id)
        assert (reviewer_row['by'], reviewer_row['class']) == (REVIEWER_LEG_ID, 'rejected_disposition')
        assert run.current(notice).resolution_class != 'observation-consumed'
        assert (finalizer_row['by'], finalizer_row['exit_kind']) == ('workflow-finalizer', 'done')
        hold = read_hold(run.current(notice))
        assert hold is not None and hold.ticket == 'tkt_1'
