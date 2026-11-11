"""The info-L0 router boundary table, plans/info-l0-disposition-router-prd.md §Boundary-test sketch.

Each row joins two or more real router call sites over one escalation-queue
directory.  The weekend records (esc-5050-2, esc-5131-1, esc-4314-1..3,
esc-4851-*) are unrecoverable, so the factories below rebuild their shapes.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from _info_l0_router_fixtures import (
    TASK_ID,
    DispositioningReviewer,
    LostVerdictReviewer,
    RouterRun,
    reaper_harness,
    router_reaper_config,
)
from _workflow_helpers import (
    _derive_meta_root_like_production as _derive_meta_root_like_production,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.escalation_router import (
    AGGREGATE_ROLE,
    FALLBACK_ACTION,
    REVIEWER_LEG_ID,
    ROUTER_ID,
    read_hold,
)
from orchestrator.harness import Harness
from orchestrator.workflow import WorkflowOutcome

ORPHAN_REAPER = 'harness-orphan-reaper'
STARVATION = 'orchestrator-starvation-watchdog'
AUTHOR_DETAIL = (
    'Seen while editing foo.py: bar.py carries a private copy of foo.helper; '
    'fold it into one helper in a follow-up.'
)
CREATED = {'status': 'created', 'task_id': '9001'}


def _timestamp(seconds_ago: float) -> str:
    return (datetime.now(UTC) - timedelta(seconds=seconds_ago)).isoformat()


def implementer_note(
    queue: EscalationQueue, task_id: str = TASK_ID, *,
    suggested_action: str = 'create_followup_task', seconds_ago: float = 0.0,
) -> Escalation:
    """The esc-5050-2 / esc-5131-1 shape: an implementer's info note about adjacent work."""
    note = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role='implementer',
        severity='info', category='design_concern',
        summary='helper duplicated between foo.py and bar.py', detail=AUTHOR_DETAIL,
        suggested_action=suggested_action, timestamp=_timestamp(seconds_ago),
    )
    queue.submit(note)
    return note


def architect_note(queue: EscalationQueue, task_id: str = TASK_ID) -> Escalation:
    """The esc-4314-1 shape: the ARCHITECT filed it, while planning."""
    note = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role='architect',
        severity='info', category='design_concern',
        summary='plan touches a module with a pending refactor',
        detail='The refactor of lib.py in flight elsewhere may conflict with this plan.',
        suggested_action='create_followup_task',
    )
    queue.submit(note)
    return note


def orphan_reaper_l1(queue: EscalationQueue, source: Escalation) -> Escalation:
    """The esc-4314-2 shape: the pre-router reaper's verbatim info promotion of *source*."""
    l1 = Escalation(
        id=queue.make_id(source.task_id), task_id=source.task_id, agent_role=ORPHAN_REAPER,
        severity='info', category=source.category,
        summary=f'Orphan L0 (900s old, no active workflow): {source.summary}',
        detail=source.detail, suggested_action='manual_intervention', level=1,
    )
    queue.submit(l1)
    return l1


def watcher_l2(
    queue: EscalationQueue, members: list[Escalation], *, severity: str,
    task_id: str = TASK_ID,
) -> Escalation:
    """The esc-4314-3 shape: escalation-watcher-auto's L2 over a cluster of L1s."""
    l2 = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role='escalation-watcher-auto',
        severity=severity, category='design_concern',
        summary=f'{len(members)} escalation(s) need a human',
        detail='a cluster the watcher promoted for a human',
        level=2, members=[m.id for m in members],
    )
    queue.submit(l2)
    return l2


def starvation_notice(queue: EscalationQueue, task_id: str, *, seconds_ago: float = 0.0) -> Escalation:
    """The esc-4851-* shape: one of the weekend's 489 starvation-watchdog notices."""
    notice = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role=STARVATION,
        severity='info', category='infra_issue',
        summary=f'task {task_id} starved of a slot for 30 min',
        detail='No slot became free for this task in the last starvation window.',
        suggested_action='none', timestamp=_timestamp(seconds_ago),
    )
    queue.submit(notice)
    return notice


def _disposition(esc_id: str, disposition: str) -> dict:
    return {'escalation_id': esc_id, 'disposition': disposition, 'rationale': f'{disposition} rationale'}


def _harness_after(run: RouterRun, config, **knobs) -> Harness:
    """A reaper/amnesty Harness over a second queue instance on *run*'s queue
    directory, sharing its curator and event store, reading the task row the
    run left."""
    statuses = run.scheduler.statuses.get(TASK_ID, [])
    return reaper_harness(
        router_reaper_config(config, **knobs), EscalationQueue(run.queue.queue_dir),
        curator=run.curator, event_store=run.workflow.event_store,
        task_row={'status': statuses[-1] if statuses else None, 'metadata': {}},
    )


async def _reap(run: RouterRun, config, **knobs) -> Harness:
    harness = _harness_after(run, config, **knobs)
    await harness._reap_orphan_l0_escalations()
    return harness


def _l1s(queue: EscalationQueue, *, agent_role: str, task_id: str = TASK_ID) -> list[Escalation]:
    return [e for e in queue.get_by_task(task_id, level=1) if e.agent_role == agent_role]


def _trail(run: RouterRun, esc_id: str) -> list[tuple]:
    return [(row['class'], row['by'], row['exit_kind']) for row in run.rows(esc_id)]


async def _held_after_a_work_verdict(tmp_path, monkeypatch) -> tuple[RouterRun, Escalation]:
    reviewer = DispositioningReviewer()
    run = RouterRun(tmp_path, monkeypatch, reviewer)
    note = implementer_note(run.queue)
    reviewer.note_dispositions = [_disposition(note.id, 'work')]

    assert await run.run() == WorkflowOutcome.DONE

    held = run.current(note)
    hold = read_hold(held)
    assert hold is not None and hold.ticket == 'tkt_1'
    assert (held.status, held.triaged_by) == ('pending', ROUTER_ID)
    return run, note


def _assert_promoted_with_the_hint(run: RouterRun, note: Escalation, reason: str) -> None:
    (l1,) = _l1s(run.queue, agent_role=AGGREGATE_ROLE)
    assert l1.suggested_action == FALLBACK_ACTION
    assert l1.detail.startswith(note.detail)
    assert reason in l1.detail
    assert run.current(note).status == 'dismissed'
    assert [e.id for e in run.queue.get_pending() if e.suggested_action == 'manual_intervention'] == []
    assert _l1s(run.queue, agent_role=ORPHAN_REAPER) == []


@pytest.mark.asyncio
class TestRow01AddressedClosesAtTheReviewer:
    async def test_the_reviewer_closes_it_and_no_reaper_or_amnesty_l1_ever_follows(
        self, tmp_path, monkeypatch, mock_orch_config,
    ):
        reviewer = DispositioningReviewer()
        run = RouterRun(tmp_path, monkeypatch, reviewer)
        note = implementer_note(run.queue)
        reviewer.note_dispositions = [_disposition(note.id, 'addressed')]

        assert await run.run() == WorkflowOutcome.DONE

        closed = run.current(note)
        assert (closed.resolution_class, closed.resolved_by) == ('addressed', REVIEWER_LEG_ID)
        assert _trail(run, note.id) == [('addressed', REVIEWER_LEG_ID, None)]

        harness = await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)
        await harness._dismiss_stale_escalations()

        assert run.queue.get_by_task(TASK_ID, level=1) == []
        assert _trail(run, note.id) == [('addressed', REVIEWER_LEG_ID, None)]


@pytest.mark.asyncio
class TestRow02WorkConvertsThroughTheHold:
    async def test_the_finalizer_holds_it_and_the_reaper_converts_it(
        self, tmp_path, monkeypatch, mock_orch_config,
    ):
        run, note = await _held_after_a_work_verdict(tmp_path, monkeypatch)

        (submitted,) = run.curator.calls_named('submit_task')
        assert submitted['metadata']['escalation_id'] == note.id
        assert submitted['metadata']['spawned_from'] == TASK_ID
        assert 'work rationale' in submitted['description']

        run.curator.set_ticket('tkt_1', CREATED)
        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)

        closed = run.current(note)
        assert (closed.status, closed.resolution_class) == ('dismissed', 'converted')
        assert 'converted:9001' in (closed.resolution or '')
        assert _trail(run, note.id) == [
            ('defer', REVIEWER_LEG_ID, None),
            ('convert-pending', 'workflow-finalizer', 'done'),
            ('converted', 'orphan-reaper', 'orphan'),
        ]
        assert _l1s(run.queue, agent_role=ORPHAN_REAPER) == []


@pytest.mark.asyncio
class TestRow03RefusedTicketPromotesWithTheHint:
    async def test_one_router_l1_carries_the_authors_text_and_no_manual_intervention(
        self, tmp_path, monkeypatch, mock_orch_config,
    ):
        run, note = await _held_after_a_work_verdict(tmp_path, monkeypatch)

        run.curator.set_ticket('tkt_1', {'status': 'refused', 'reason': 'cancelled premise'})
        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)

        _assert_promoted_with_the_hint(run, note, 'ticket_refused')


@pytest.mark.asyncio
class TestRow04UndecidedTicketWaitsThenTimesOut:
    async def test_inside_the_bound_the_hold_stays_and_past_it_the_note_promotes(
        self, tmp_path, monkeypatch, mock_orch_config,
    ):
        run, note = await _held_after_a_work_verdict(tmp_path, monkeypatch)
        held = run.current(note)

        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0, ticket_timeout_secs=900.0)

        after = run.current(note)
        assert (after.status, after.triage_note) == ('pending', held.triage_note)
        assert run.queue.get_by_task(TASK_ID, level=1) == []

        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0, ticket_timeout_secs=0.0)

        _assert_promoted_with_the_hint(run, note, 'ticket_timeout')


@pytest.mark.asyncio
class TestRow17LostVerdictIsRoutedAtTheExit:
    async def test_the_discarded_verdict_writes_nothing_and_the_hold_converts(
        self, tmp_path, monkeypatch, mock_orch_config,
    ):
        reviewer = LostVerdictReviewer()
        run = RouterRun(tmp_path, monkeypatch, reviewer)
        note = implementer_note(run.queue)
        reviewer.note_dispositions = [_disposition(note.id, 'addressed')]

        assert await run.run() == WorkflowOutcome.BLOCKED

        assert _trail(run, note.id) == [('convert-pending', 'workflow-finalizer', 'blocked')]
        hold = read_hold(run.current(note))
        assert hold is not None and hold.ticket == 'tkt_1'

        run.curator.set_ticket('tkt_1', CREATED)
        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)

        assert run.current(note).resolution_class == 'converted'
        assert _trail(run, note.id)[-1] == ('converted', 'orphan-reaper', 'orphan')
