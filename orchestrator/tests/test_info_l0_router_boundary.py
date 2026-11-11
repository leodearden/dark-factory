"""The info-L0 router boundary table, plans/info-l0-disposition-router-prd.md §Boundary-test sketch.

Each row joins two or more real router call sites over one escalation-queue
directory.  The weekend records (esc-5050-2, esc-5131-1, esc-4314-1..3,
esc-4851-*) are unrecoverable, so the factories below rebuild their shapes.
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _info_l0_router_fixtures import (
    TASK_ID,
    DispositioningReviewer,
    LostVerdictReviewer,
    RouterRun,
    assignment,
    handing_off_steward,
    reaper_harness,
    router_reaper_config,
)
from _merge_queue_doubles import ResolvingMergeQueue
from _orch_helpers import VERIFY_CLI_PER_TEST_TIMEOUT, wait_responsive
from _workflow_helpers import FakeScheduler, _build_harness
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
from orchestrator.event_store import EventType
from orchestrator.harness import Harness
from orchestrator.merge_lane.worker import MergeOutcome
from orchestrator.scheduler import Scheduler
from orchestrator.workflow import WorkflowOutcome

ORPHAN_REAPER = 'harness-orphan-reaper'
STARVATION = 'orchestrator-starvation-watchdog'
AUTHOR_DETAIL = (
    'Seen while editing foo.py: bar.py carries a private copy of foo.helper; '
    'fold it into one helper in a follow-up.'
)
CREATED = {'status': 'created', 'task_id': '9001'}
SUPERSEDED = MergeOutcome('superseded', superseded_by='mr-train-5050', merge_sha='5050' * 10)


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


def blocking_l0(queue: EscalationQueue, task_id: str = TASK_ID) -> Escalation:
    """An agent's blocking L0, the record a steward takes."""
    esc = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role='implementer',
        severity='blocking', category='task_failure',
        summary='cannot proceed: lib.py contract is ambiguous',
        detail='The plan and the existing tests disagree on greet().',
    )
    queue.submit(esc)
    return esc


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


def _recording_steward(constructions: list[str]) -> type:
    """A steward that records each construction and does nothing else."""

    class _Steward:
        def __init__(self, wt_path, cfg_dir):
            constructions.append(str(wt_path))

        def set_outcome_channel(self, channel) -> None:
            pass

        def set_wip_probe(self, probe) -> None:
            pass

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    return _Steward


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


@pytest.mark.asyncio
class TestRow05CoalesceParkHoldsTheNote:
    """E1 (tasks 5050, 4314 run 1): the merge is absorbed by a coalesced train."""

    async def test_the_merge_deferred_exit_holds_the_note_and_no_orphan_l1_follows(
        self, tmp_path, monkeypatch, mock_orch_config,
    ):
        run = RouterRun(
            tmp_path, monkeypatch, DispositioningReviewer(),
            merge_queue=ResolvingMergeQueue(SUPERSEDED),
        )
        note = implementer_note(run.queue)

        assert await run.run() == WorkflowOutcome.MERGE_DEFERRED

        written = run.scheduler.statuses[TASK_ID]
        assert written[-1] == 'merge-deferred'
        assert 'done' not in written
        assert _trail(run, note.id) == [('convert-pending', 'workflow-finalizer', 'merge_deferred')]
        assert read_hold(run.current(note)) is not None

        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)

        assert _l1s(run.queue, agent_role=ORPHAN_REAPER) == []

        run.curator.set_ticket('tkt_1', CREATED)
        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)

        assert run.current(note).resolution_class == 'converted'


class _DoneFlipOutrunsTheTail(FakeScheduler):
    """The first 'done' write lands, then the tail stays busy until it is cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.done_written = asyncio.Event()

    async def set_task_status(self, task_id: str, status: str, **kwargs) -> None:
        await super().set_task_status(task_id, status, **kwargs)
        if status == 'done' and not self.done_written.is_set():
            self.done_written.set()
            await asyncio.Event().wait()


@pytest.mark.asyncio
class TestRow06TerminalStatusCancelDuringTheTail:
    """E2 (task 5131): the row flips done while the workflow is still in its
    tail, and the terminal-status watcher soft-cancels the slot.

    ``build_workflow`` is patched to hand the slot this run's workflow; the
    stand-in binds the slot's ``cancel_event`` to it as the real factory does,
    so the watcher's soft cancel reaches the workflow.
    """

    @pytest.mark.timeout(VERIFY_CLI_PER_TEST_TIMEOUT)
    async def test_the_note_is_held_before_the_locks_are_released_and_no_steward_starts(
        self, tmp_path, monkeypatch, mock_orch_config,
    ):
        constructions: list[str] = []
        flips = _DoneFlipOutrunsTheTail()
        run = RouterRun(
            tmp_path, monkeypatch, DispositioningReviewer(),
            scheduler=flips, steward_factory=_recording_steward(constructions),
        )
        note = implementer_note(run.queue)
        events = run.workflow.event_store
        assert events is not None
        config = run.workflow.config.model_copy(update={'terminal_status_hard_cancel_polls': 3})
        harness = _build_harness(config)
        scheduler = Scheduler(config)
        monkeypatch.setattr(scheduler, 'dispatch_tool', AsyncMock(return_value={}))
        monkeypatch.setattr(scheduler, 'get_statuses', flips.get_statuses)
        scheduler.event_store = events
        harness.scheduler = scheduler
        harness.event_store = events
        assert scheduler.lock_table.try_acquire(TASK_ID, ['lib'])

        def _build_this_runs_workflow(**kwargs):
            run.workflow._cancel_event = kwargs['cancel_event']
            return run.workflow

        with patch('orchestrator.harness.build_workflow', side_effect=_build_this_runs_workflow):
            slot = asyncio.create_task(harness._run_slot(assignment(), asyncio.Semaphore(0)))
            await wait_responsive(flips.done_written.wait(), label='the merged row flips done')
            assert await harness._scan_for_terminal_active_tasks() == 1
            report = await wait_responsive(slot, label='the soft-cancelled slot unwinds')

        assert report is not None and report.outcome == WorkflowOutcome.DONE
        assert constructions == []
        assert _trail(run, note.id) == [('convert-pending', 'workflow-finalizer', 'cancelled')]
        assert read_hold(run.current(note)) is not None
        (routed,) = [
            row for row in events.fetch_events_by_type(EventType.info_l0_dispositioned)
            if row['data']['escalation_id'] == note.id
        ]
        released = [
            row for row in events.fetch_events_by_type(EventType.lock_released)
            if row['task_id'] == TASK_ID
        ]
        assert released, 'the slot never released its module locks'
        assert routed['id'] < released[0]['id']

        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)

        assert _l1s(run.queue, agent_role=ORPHAN_REAPER) == []


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=False)


def _file_bytes(queue: EscalationQueue, esc: Escalation) -> bytes:
    return (queue.queue_dir / f'{esc.id}.json').read_bytes()


class _FilesAtMergeEntry(FakeScheduler):
    """Files records when ``_stamp_merge_phase_entered`` persists its stamp.

    That stamp is written after the post-implementer gate and before the
    merge-entry gate (``TaskWorkflow._run_merge_phase``), so a record filed
    here is first seen by the merge-entry gate.
    """

    def __init__(self, file_records: Callable[[], list[Escalation]]) -> None:
        super().__init__()
        self._file_records = file_records
        self.filed: list[Escalation] = []
        self.snapshots: dict[str, bytes] = {}
        self.branch_head: str | None = None

    async def update_task(self, task_id: str, metadata: str | dict, *, append: bool = False) -> bool:
        if not self.filed and isinstance(metadata, dict) and 'merge_phase_liveness' in metadata:
            self.filed = self._file_records()
        return await super().update_task(task_id, metadata, append=append)


def _merge_entry_run(
    tmp_path: Path, monkeypatch, file_records: Callable[[EscalationQueue], list[Escalation]],
) -> tuple[RouterRun, _FilesAtMergeEntry, list[str]]:
    queue = EscalationQueue(tmp_path / 'escalation_queue')
    constructions: list[str] = []
    holder: dict[str, RouterRun] = {}

    def _file() -> list[Escalation]:
        records = file_records(queue)
        scheduler.snapshots = {e.id: _file_bytes(queue, e) for e in records}
        worktree = holder['run'].workflow.worktree
        assert worktree is not None
        scheduler.branch_head = _git(worktree, 'rev-parse', 'HEAD').stdout.strip()
        return records

    scheduler = _FilesAtMergeEntry(_file)
    run = RouterRun(
        tmp_path, monkeypatch, DispositioningReviewer(), queue=queue, scheduler=scheduler,
        steward_factory=_recording_steward(constructions),
    )
    holder['run'] = run
    return run, scheduler, constructions


@pytest.mark.asyncio
class TestRow07InfoL2AtMergeEntryMerges:
    """E3 (task 4314 run 2): an info L2 over an info L1 is open at merge entry."""

    async def test_the_task_merges_and_the_cluster_is_untouched(self, tmp_path, monkeypatch):
        def _cluster(queue: EscalationQueue) -> list[Escalation]:
            source = architect_note(queue)
            l1 = orphan_reaper_l1(queue, source)
            queue.resolve(source.id, f'Auto-promoted to level 1: {l1.id}', dismiss=True, resolved_by=ORPHAN_REAPER)
            return [watcher_l2(queue, [l1], severity='info'), l1]

        run, scheduler, constructions = _merge_entry_run(tmp_path, monkeypatch, _cluster)

        assert await run.run() == WorkflowOutcome.DONE

        assert len(scheduler.filed) == 2, 'the merge-entry stamp never fired'
        repo = run.workflow.config.project_root
        assert scheduler.branch_head
        assert _git(repo, 'merge-base', '--is-ancestor', scheduler.branch_head, 'main').returncode == 0
        assert 'blocked' not in scheduler.statuses[TASK_ID]
        for esc in scheduler.filed:
            assert _file_bytes(run.queue, esc) == scheduler.snapshots[esc.id]
        assert constructions == []


@pytest.mark.asyncio
class TestRow08BlockingL2AtMergeEntryGates:
    async def test_the_task_blocks_naming_the_l2_and_no_steward(self, tmp_path, monkeypatch):
        run, scheduler, constructions = _merge_entry_run(
            tmp_path, monkeypatch, lambda queue: [watcher_l2(queue, [], severity='blocking')],
        )
        repo = run.workflow.config.project_root
        main_before = _git(repo, 'rev-parse', 'main').stdout.strip()

        report = await run.workflow.run()

        (l2,) = scheduler.filed
        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason == f'Gated by open L2 {l2.id}; no steward applicable'
        assert 'Steward re-escalated' not in report.detail
        assert _git(repo, 'rev-parse', 'main').stdout.strip() == main_before
        assert constructions == []
        assert _file_bytes(run.queue, l2) == scheduler.snapshots[l2.id]


@pytest.mark.asyncio
class TestRow14BlockedExitKeepsInfoNotes:
    async def test_the_steward_hands_off_the_blocker_and_the_finalizer_holds_the_note(
        self, tmp_path, monkeypatch, mock_orch_config,
    ):
        queue = EscalationQueue(tmp_path / 'escalation_queue')
        run = RouterRun(
            tmp_path, monkeypatch, DispositioningReviewer(), queue=queue,
            steward_factory=handing_off_steward(queue, TASK_ID, severity='blocking'),
        )
        filed: list[Escalation] = []

        async def _escalate() -> WorkflowOutcome:
            filed.extend([blocking_l0(queue), implementer_note(queue)])
            return WorkflowOutcome.ESCALATED

        monkeypatch.setattr(run.workflow, '_execute_verify_review_loop', _escalate)

        report = await asyncio.wait_for(run.workflow.run(), 60)

        blocker, note = filed
        assert report.outcome == WorkflowOutcome.BLOCKED
        assert report.reason == 'Steward re-escalated to human'
        assert run.current(blocker).status == 'dismissed'
        assert _trail(run, note.id) == [('convert-pending', 'workflow-finalizer', 'blocked')]
        assert read_hold(run.current(note)) is not None

        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)

        assert _l1s(run.queue, agent_role=ORPHAN_REAPER) == []


async def _requeued_incarnation(tmp_path: Path, monkeypatch) -> tuple[RouterRun, Escalation]:
    """Incarnation 1 exits REQUEUED; the exit, not what produced it, is under test."""
    run = RouterRun(tmp_path / 'incarnation-1', monkeypatch, DispositioningReviewer())
    note = implementer_note(run.queue)

    async def _drive() -> WorkflowOutcome:
        await run.scheduler.set_task_status(TASK_ID, 'pending')
        return WorkflowOutcome.REQUEUED

    run.workflow._drive = _drive

    assert await run.run() == WorkflowOutcome.REQUEUED

    after = run.current(note)
    assert (after.status, after.triaged_by) == ('pending', None)
    assert _trail(run, note.id) == [('defer', 'workflow-finalizer', 'requeued')]
    return run, note


@pytest.mark.asyncio
class TestRow15RequeuedExitDefers:
    async def test_the_next_incarnations_reviewer_is_briefed_and_closes_it(self, tmp_path, monkeypatch):
        first, note = await _requeued_incarnation(tmp_path, monkeypatch)
        reviewer = DispositioningReviewer()
        second = RouterRun(tmp_path / 'incarnation-2', monkeypatch, reviewer, queue=first.queue)
        reviewer.note_dispositions = [_disposition(note.id, 'addressed')]

        assert await second.run() == WorkflowOutcome.DONE

        assert [ids for _generation, ids in second.briefing.briefed] == [[note.id]]
        closed = second.current(note)
        assert (closed.resolution_class, closed.resolved_by) == ('addressed', REVIEWER_LEG_ID)

    async def test_with_no_next_incarnation_the_reaper_holds_it_once_it_ages(
        self, tmp_path, monkeypatch, mock_orch_config,
    ):
        run, note = await _requeued_incarnation(tmp_path, monkeypatch)
        deferred = run.current(note)

        await _reap(run, mock_orch_config, orphan_timeout_secs=60.0)

        assert run.current(note) == deferred

        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)

        hold = read_hold(run.current(note))
        assert hold is not None and hold.ticket == 'tkt_1'
        assert _trail(run, note.id)[-1] == ('convert-pending', 'orphan-reaper', 'orphan')
