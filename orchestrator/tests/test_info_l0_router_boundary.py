"""The info-L0 router boundary table, plans/info-l0-disposition-router-prd.md §Boundary-test sketch.

Each row joins two or more real router call sites over one escalation-queue
directory.  The weekend records (esc-5050-2, esc-5131-1, esc-4314-1..3,
esc-4851-*) are unrecoverable, so the factories below rebuild their shapes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _curator_fake import FakeCurator
from _info_l0_router_fixtures import (
    TASK_ID,
    DispositioningReviewer,
    LostVerdictReviewer,
    RouterRun,
    assignment,
    disposition_entry,
    disposition_events,
    disposition_rows,
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
from escalation.classify import DONE_STEP_COMMIT_ORPHAN_CLASS
from escalation.disposition import ReviewerDisposition
from escalation.models import Escalation
from escalation.pins import classify_pins
from escalation.queue import EscalationQueue

from orchestrator.escalation_router import (
    AGGREGATE_CATEGORY,
    AGGREGATE_ROLE,
    FALLBACK_ACTION,
    OVERFLOW_SENTINEL,
    REVIEWER_LEG_ID,
    ROUTER_ID,
    Outcome,
    RouterPorts,
    apply_reviewer_disposition,
    read_hold,
    status_info_sentinel,
)
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness
from orchestrator.merge_lane.worker import MergeOutcome
from orchestrator.review_payload import NoteDisposition
from orchestrator.scheduler import Scheduler
from orchestrator.workflow import WorkflowOutcome

ORPHAN_REAPER = 'harness-orphan-reaper'
STARVATION = 'orchestrator-starvation-watchdog'
MERGE_SKEW = 'orchestrator-merge-skew-tripwire'
AGED_SECS = 300.0
ORPHAN_AGE_PREFIX = re.compile(r'^Orphan L0 \(\d+s old, ')
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


def mechanical_notice(
    queue: EscalationQueue, task_id: str, *, agent_role: str, category: str, summary: str,
    detail: str, suggested_action: str = 'none', seconds_ago: float = 0.0,
) -> Escalation:
    """An info notice filed by a harness monitor rather than an agent."""
    notice = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role=agent_role,
        severity='info', category=category, summary=summary, detail=detail,
        suggested_action=suggested_action, timestamp=_timestamp(seconds_ago),
    )
    queue.submit(notice)
    return notice


def starvation_notice(queue: EscalationQueue, task_id: str, *, seconds_ago: float = 0.0) -> Escalation:
    """The esc-4851-* shape: one of the weekend's 489 starvation-watchdog notices."""
    return mechanical_notice(
        queue, task_id, agent_role=STARVATION, category='infra_issue',
        summary=f'task {task_id} starved of a slot for 30 min',
        detail='No slot became free for this task in the last starvation window.',
        seconds_ago=seconds_ago,
    )


def merge_skew_notice(queue: EscalationQueue, task_id: str, *, seconds_ago: float = 0.0) -> Escalation:
    """The merge_lane/skew_tripwire.py shape: a landing overlaps an in-flight branch."""
    return mechanical_notice(
        queue, task_id, agent_role=MERGE_SKEW, category='risk_identified',
        summary=f'Pipeline landing 5050aaaa5050 touches load-bearing files also present on 1 in-flight branch(es): {task_id}',
        detail=f'In-flight branches with overlapping edits:\n- task {task_id} (branch task/{task_id}): lib.py',
        suggested_action='Port the landed change into each named in-flight branch before it merges.',
        seconds_ago=seconds_ago,
    )


def done_step_orphan_notice(queue: EscalationQueue, task_id: str, *, seconds_ago: float = 0.0) -> Escalation:
    """The TaskWorkflow._escalate_unreconciled_done_step shape: a done step's commit is orphaned."""
    return mechanical_notice(
        queue, task_id, agent_role='orchestrator', category='infra_issue',
        summary="Done step step-1's commit 4851bbbb48 is orphaned and could not be auto-reconciled",
        detail='Step step-1 recorded commit 4851bbbb48, which is no longer reachable from HEAD.',
        suggested_action='verify_wip_reconciliation', seconds_ago=seconds_ago,
    )


def _harness_after(run: RouterRun, config, **knobs) -> Harness:
    """A reaper/amnesty Harness over a second queue instance on *run*'s queue
    directory, sharing its curator and event store, reading the task row the
    run left."""
    statuses = run.scheduler.statuses.get(TASK_ID, [])
    return reaper_harness(
        router_reaper_config(config, **knobs), EscalationQueue(run.queue.queue_dir),
        curator=run.curator, event_store=run.event_store,
        task_row={'status': statuses[-1] if statuses else None, 'metadata': {}},
    )


async def _sweep_orphans(harness: Harness) -> None:
    """One orphan-reaper pass; the suite's one call of that private seam."""
    await harness._reap_orphan_l0_escalations()


async def _amnesty(harness: Harness) -> None:
    """One startup-amnesty pass; the suite's one call of that private seam."""
    await harness._dismiss_stale_escalations()


async def _reap(run: RouterRun, config, **knobs) -> Harness:
    harness = _harness_after(run, config, **knobs)
    await _sweep_orphans(harness)
    return harness


def _l1s(queue: EscalationQueue, *, agent_role: str, task_id: str = TASK_ID) -> list[Escalation]:
    return queue.get_by_task(task_id, level=1, agent_role=agent_role)


def _trail(store: EventStore | None, esc_id: str) -> list[tuple]:
    """Each routing of *esc_id*: (class, by, exit_kind)."""
    return [(row['class'], row['by'], row['exit_kind']) for row in disposition_rows(store, esc_id)]


async def _held_after_a_work_verdict(tmp_path, monkeypatch) -> tuple[RouterRun, Escalation]:
    reviewer = DispositioningReviewer()
    run = RouterRun(tmp_path, monkeypatch, reviewer)
    note = implementer_note(run.queue)
    reviewer.note_dispositions = [disposition_entry(note.id, 'work')]

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
        reviewer.note_dispositions = [disposition_entry(note.id, 'addressed')]

        assert await run.run() == WorkflowOutcome.DONE

        closed = run.current(note)
        assert (closed.resolution_class, closed.resolved_by) == ('addressed', REVIEWER_LEG_ID)
        assert _trail(run.event_store, note.id) == [('addressed', REVIEWER_LEG_ID, None)]

        harness = await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)
        await _amnesty(harness)

        assert run.queue.get_by_task(TASK_ID, level=1) == []
        assert _trail(run.event_store, note.id) == [('addressed', REVIEWER_LEG_ID, None)]


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
        assert _trail(run.event_store, note.id) == [
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
        reviewer.note_dispositions = [disposition_entry(note.id, 'addressed')]

        assert await run.run() == WorkflowOutcome.BLOCKED

        assert _trail(run.event_store, note.id) == [('convert-pending', 'workflow-finalizer', 'blocked')]
        hold = read_hold(run.current(note))
        assert hold is not None and hold.ticket == 'tkt_1'

        run.curator.set_ticket('tkt_1', CREATED)
        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)

        assert run.current(note).resolution_class == 'converted'
        assert _trail(run.event_store, note.id)[-1] == ('converted', 'orphan-reaper', 'orphan')


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
        assert _trail(run.event_store, note.id) == [('convert-pending', 'workflow-finalizer', 'merge_deferred')]
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

    ``build_workflow`` is patched to build the slot this run's workflow with the
    slot's ``cancel_event``, as the real factory does, so the watcher's soft
    cancel reaches the workflow.
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
        events = run.event_store
        config = run.config.model_copy(update={'terminal_status_hard_cancel_polls': 3})
        harness = _build_harness(config)
        scheduler = Scheduler(config)
        monkeypatch.setattr(scheduler, 'dispatch_tool', AsyncMock(return_value={}))
        monkeypatch.setattr(scheduler, 'get_statuses', flips.get_statuses)
        scheduler.event_store = events
        harness.scheduler = scheduler
        harness.event_store = events
        assert scheduler.lock_table.try_acquire(TASK_ID, ['lib'])

        def _build_this_runs_workflow(**kwargs):
            return run.build_workflow(cancel_event=kwargs['cancel_event'])

        with patch('orchestrator.harness.build_workflow', side_effect=_build_this_runs_workflow):
            slot = asyncio.create_task(harness._run_slot(assignment(), asyncio.Semaphore(0)))
            await wait_responsive(flips.done_written.wait(), label='the merged row flips done')
            assert await harness._scan_for_terminal_active_tasks() == 1
            report = await wait_responsive(slot, label='the soft-cancelled slot unwinds')

        assert report is not None and report.outcome == WorkflowOutcome.DONE
        assert constructions == []
        assert _trail(events, note.id) == [('convert-pending', 'workflow-finalizer', 'cancelled')]
        assert read_hold(run.current(note)) is not None
        (routed,) = disposition_events(events, note.id)
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
        repo = run.config.project_root
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
        repo = run.config.project_root
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
        assert _trail(run.event_store, note.id) == [('convert-pending', 'workflow-finalizer', 'blocked')]
        assert read_hold(run.current(note)) is not None

        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)

        assert _l1s(run.queue, agent_role=ORPHAN_REAPER) == []


def _exit_without_driving(run: RouterRun, status: str, outcome: WorkflowOutcome) -> None:
    """Replace ``_drive`` with one that writes *status* and returns *outcome*; the exit finalizer stays real."""

    async def _drive() -> WorkflowOutcome:
        await run.scheduler.set_task_status(TASK_ID, status)
        return outcome

    run.workflow._drive = _drive


async def _requeued_incarnation(tmp_path: Path, monkeypatch) -> tuple[RouterRun, Escalation]:
    """Incarnation 1 exits REQUEUED; the exit, not what produced it, is under test."""
    run = RouterRun(tmp_path / 'incarnation-1', monkeypatch, DispositioningReviewer())
    note = implementer_note(run.queue)
    _exit_without_driving(run, 'pending', WorkflowOutcome.REQUEUED)

    assert await run.run() == WorkflowOutcome.REQUEUED

    after = run.current(note)
    assert (after.status, after.triaged_by) == ('pending', None)
    assert _trail(run.event_store, note.id) == [('defer', 'workflow-finalizer', 'requeued')]
    return run, note


@pytest.mark.asyncio
class TestRow15RequeuedExitDefers:
    async def test_the_next_incarnations_reviewer_is_briefed_and_closes_it(self, tmp_path, monkeypatch):
        first, note = await _requeued_incarnation(tmp_path, monkeypatch)
        reviewer = DispositioningReviewer()
        second = RouterRun(tmp_path / 'incarnation-2', monkeypatch, reviewer, queue=first.queue)
        reviewer.note_dispositions = [disposition_entry(note.id, 'addressed')]

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
        assert _trail(run.event_store, note.id)[-1] == ('convert-pending', 'orphan-reaper', 'orphan')


@dataclass
class _ReaperSite:
    """The reaper and the amnesty over one real queue, with no workflow."""

    harness: Harness
    queue: EscalationQueue
    curator: FakeCurator
    events: EventStore | None

    async def reap(self) -> None:
        await _sweep_orphans(self.harness)

    def current(self, esc: Escalation) -> Escalation:
        current = self.queue.get(esc.id)
        assert current is not None
        return current

    def trail(self, esc_id: str) -> list[tuple]:
        return _trail(self.events, esc_id)

    def aggregates(self, sentinel: str) -> list[Escalation]:
        return [
            e for e in self.queue.get_by_task(sentinel, status='pending', level=1, agent_role=AGGREGATE_ROLE)
            if e.category == AGGREGATE_CATEGORY
        ]

    def every_record(self) -> list[dict]:
        """Every record on disk, archive included."""
        return [json.loads(path.read_text()) for path in self.queue.queue_dir.rglob('esc-*.json')]


def _site(tmp_path: Path, config, *, name: str = 'escalations', events: bool = True, **knobs) -> _ReaperSite:
    queue = EscalationQueue(tmp_path / name)
    curator = FakeCurator()
    store = EventStore(tmp_path / f'{name}-runs.db', run_id='run-test') if events else None
    harness = reaper_harness(
        router_reaper_config(config, **knobs), queue, curator=curator, event_store=store,
    )
    return _ReaperSite(harness, queue, curator, store)


def _listed(esc: Escalation) -> str:
    """How an aggregate's detail lists one record."""
    return f'{esc.id} (task {esc.task_id})'


def _seed(n: int, first_task: int, file_one: Callable[[str], Escalation]) -> list[Escalation]:
    """*n* records on distinct tasks; durable writes are not under test here."""
    with patch('os.fsync'):
        return [file_one(str(first_task + i)) for i in range(n)]


@pytest.mark.asyncio
class TestRow11MechanicalStormClosesUnderOneAggregatePerClass:
    """The esc-4851 storm: 489 starvation notices, beside two other mechanical classes."""

    async def test_no_open_notice_pins_its_task_and_one_sweep_closes_every_one_status_info(
        self, tmp_path, mock_orch_config,
    ):
        site = _site(tmp_path, mock_orch_config, events=False)  # 492 durable event commits cost seconds
        storm = _seed(489, 10_000, lambda tid: starvation_notice(site.queue, tid, seconds_ago=AGED_SECS))
        skew = _seed(2, 7_000, lambda tid: merge_skew_notice(site.queue, tid, seconds_ago=AGED_SECS))
        done_step = _seed(1, 8_000, lambda tid: done_step_orphan_notice(site.queue, tid, seconds_ago=AGED_SECS))
        classes = {STARVATION: storm, MERGE_SKEW: skew, DONE_STEP_COMMIT_ORPHAN_CLASS: done_step}
        notices = [e for records in classes.values() for e in records]
        open_by_task = site.queue.get_pending_by_task()
        assert all([r.id for r in open_by_task[e.task_id]] == [e.id] for e in notices)
        assert not any(classify_pins(e.task_id, open_by_task[e.task_id], live_claimant=False).pins for e in notices)

        await site.reap()

        assert all(site.current(e).resolution_class == 'status-info' for e in notices)
        aggregates = {}
        for class_key, records in classes.items():
            (aggregate,) = site.aggregates(status_info_sentinel(class_key))
            aggregates[class_key] = aggregate
            assert all(_listed(e) in aggregate.detail for e in records)
            others = [e for key, rs in classes.items() if key != class_key for e in rs]
            assert not any(e.id in aggregate.detail for e in others)
        pending = site.queue.get_pending()
        assert [e for e in pending if e.agent_role == ORPHAN_REAPER] == []
        assert [e for e in pending if e.level >= 2] == []
        assert site.curator.calls == []
        pending_by_task = site.queue.get_pending_by_task()
        assert [e.id for e in notices if pending_by_task.get(e.task_id)] == []

        later = _seed(3, 20_000, lambda tid: starvation_notice(site.queue, tid, seconds_ago=AGED_SECS))
        await site.reap()

        (still,) = site.aggregates(status_info_sentinel(STARVATION))
        assert still.id == aggregates[STARVATION].id
        assert all(_listed(e) in still.amendments[-1]['detail'] for e in later)


@pytest.mark.asyncio
class TestRow12UnknownMechanicalRoleGoesToTheCurator:
    async def test_an_unregistered_monitor_is_converted_never_status_info(self, tmp_path, mock_orch_config):
        site = _site(tmp_path, mock_orch_config)
        notice = mechanical_notice(
            site.queue, '42', agent_role='orchestrator-some-new-monitor', category='infra_issue',
            summary='some new monitor tripped on task 42', detail='A monitor no class registry knows yet.',
            seconds_ago=AGED_SECS,
        )

        await site.reap()

        hold = read_hold(site.current(notice))
        assert hold is not None and hold.ticket == 'tkt_1'
        assert site.trail(notice.id) == [('convert-pending', 'orphan-reaper', 'orphan')]
        assert site.aggregates(status_info_sentinel('orchestrator-some-new-monitor')) == []

        site.curator.set_ticket('tkt_1', CREATED)
        await site.reap()

        assert site.current(notice).resolution_class == 'converted'
        assert 'status-info' not in [row[0] for row in site.trail(notice.id)]


def _aged_blocking_orphan(queue: EscalationQueue, task_id: str, worktree: Path) -> Escalation:
    """The test_orphan_l0_reaper.py aged-orphan shape."""
    esc = Escalation(
        id=queue.make_id(task_id), task_id=task_id, agent_role='integration-reviewer',
        severity='blocking', category='design_concern', summary='Plan drift detected',
        detail='The integration reviewer found drift between plan and code.',
        worktree=str(worktree), timestamp=_timestamp(AGED_SECS),
    )
    queue.submit(esc)
    return esc


def _promotion_fields(esc: Escalation) -> dict:
    """Every field but the record's identity, its times, and the summary's age prefix,
    which reads the wall clock at the sweep."""
    fields = {k: v for k, v in asdict(esc).items() if k not in ('id', 'timestamp', 'resolved_at')}
    age_prefix = ORPHAN_AGE_PREFIX.match(esc.summary)
    assert age_prefix is not None, esc.summary
    fields['summary'] = esc.summary[age_prefix.end():]
    return fields


@pytest.mark.asyncio
class TestRow13BlockingOrphanIsPromotedAsItWasBeforeTheRouter:
    """'Unchanged' is measured against the router-disabled reaper over the same record."""

    async def test_the_promotion_equals_the_router_disabled_one(self, tmp_path, mock_orch_config):
        task_id = '4242'
        worktree = tmp_path / '.worktrees' / task_id
        routed = _site(tmp_path, mock_orch_config, name='router-on')
        on_orphan = _aged_blocking_orphan(routed.queue, task_id, worktree)
        note = implementer_note(routed.queue, task_id, seconds_ago=AGED_SECS)
        await routed.reap()

        disabled = _site(tmp_path, mock_orch_config, name='router-off', router_enabled=False)
        off_orphan = _aged_blocking_orphan(disabled.queue, task_id, worktree)
        await disabled.reap()

        assert read_hold(routed.current(note)) is not None
        (on_l1,) = _l1s(routed.queue, agent_role=ORPHAN_REAPER, task_id=task_id)
        (off_l1,) = _l1s(disabled.queue, agent_role=ORPHAN_REAPER, task_id=task_id)
        assert _promotion_fields(on_l1) == _promotion_fields(off_l1)
        on, off = routed.current(on_orphan), disabled.current(off_orphan)
        assert (on.status, on.resolved_by, on.resolution) == (off.status, off.resolved_by, off.resolution)
        assert on_l1.suggested_action == 'manual_intervention'
        assert f'branch=task/{task_id}' in on_l1.detail


@pytest.mark.asyncio
class TestRow16RestartAmnestyRoutesInfoNotes:
    async def test_one_amnesty_pass_routes_the_notes_and_sweeps_only_the_strand(
        self, tmp_path, mock_orch_config,
    ):
        site = _site(tmp_path, mock_orch_config)
        note = implementer_note(site.queue, '42', seconds_ago=AGED_SECS)
        notice = starvation_notice(site.queue, '43', seconds_ago=AGED_SECS)
        strand = _aged_blocking_orphan(site.queue, '5189', tmp_path / '.worktrees' / '5189')

        await _amnesty(site.harness)

        assert read_hold(site.current(note)) is not None
        assert site.trail(note.id) == [('convert-pending', 'startup-amnesty', 'restart')]
        assert site.current(notice).resolution_class == 'status-info'
        (aggregate,) = site.aggregates(status_info_sentinel(STARVATION))
        assert _listed(notice) in aggregate.detail
        swept = site.current(strand)
        assert (swept.status, swept.resolved_by, swept.resolution_class) == (
            'dismissed', 'auto-dismissed', 'stale-strand',
        )
        assert site.events is not None
        (event,) = site.events.fetch_events_by_type(EventType.stale_l0_strand_dismissed)
        assert event['data']['escalation_id'] == strand.id
        assert [
            r['id'] for r in site.every_record()
            if r['severity'] == 'info' and r.get('resolved_by') == 'auto-dismissed'
        ] == []

        site.curator.set_ticket('tkt_1', CREATED)
        await site.reap()

        assert site.current(note).resolution_class == 'converted'


@pytest.mark.asyncio
class TestRow19ConversionCapOverflows:
    async def test_past_the_cap_notes_promote_under_one_overflow_aggregate_per_sweep(
        self, tmp_path, mock_orch_config,
    ):
        site = _site(tmp_path, mock_orch_config, max_conversions=2)
        notes = [implementer_note(site.queue, str(4000 + i), seconds_ago=AGED_SECS) for i in range(4)]

        await site.reap()

        assert len(site.curator.calls_named('submit_task')) == 2
        held = [n for n in notes if read_hold(site.current(n)) is not None]
        overflow = [n for n in notes if site.current(n).status == 'dismissed']
        assert (len(held), len(overflow)) == (2, 2)
        for note in overflow:
            (l1,) = _l1s(site.queue, agent_role=AGGREGATE_ROLE, task_id=note.task_id)
            assert l1.suggested_action == FALLBACK_ACTION
            assert l1.detail.startswith(note.detail)
            assert 'conversion_cap' in l1.detail
            assert note.id in l1.detail
        (aggregate,) = site.aggregates(OVERFLOW_SENTINEL)
        assert all(_listed(n) in aggregate.detail for n in overflow)
        assert not any(n.id in aggregate.detail for n in held)

        fresh = [implementer_note(site.queue, str(5000 + i), seconds_ago=AGED_SECS) for i in range(2)]
        await site.reap()

        assert len(site.curator.calls_named('submit_task')) == 4
        assert all(read_hold(site.current(n)) is not None for n in fresh)


@pytest.mark.asyncio
class TestRow18DeliberateHoldIsUnaffected:
    """A parked L2 and a separate L2 are holds, so every router path leaves them alone.

    Both holds are blocking: pins.py link 1 makes a parked INFO record non-pinning.
    """

    async def test_every_router_path_leaves_both_holds_byte_identical(
        self, tmp_path, monkeypatch, mock_orch_config,
    ):
        run = RouterRun(tmp_path, monkeypatch, DispositioningReviewer())
        queue = run.queue
        to_park = Escalation(
            id=queue.make_id(TASK_ID), task_id=TASK_ID, agent_role='steward', severity='blocking',
            category='design_concern', summary='needs a human ruling', level=1,
        )
        queue.submit(to_park)
        parked = queue.park(to_park.id, 'parked for a human', resolved_by='escalation-watcher-auto')
        assert parked is not None
        assert (parked.status, parked.level, parked.resolution_action) == ('pending', 2, 'park')
        separate = watcher_l2(queue, [], severity='blocking')
        note = implementer_note(queue)
        held_bytes = {hold.id: _file_bytes(queue, hold) for hold in (parked, separate)}
        events = run.event_store

        def _assert_the_holds_are_untouched() -> None:
            assert {hold_id: (queue.queue_dir / f'{hold_id}.json').read_bytes() for hold_id in held_bytes} == held_bytes
            rows = disposition_rows(events)
            assert {r['class'] for r in rows if r['escalation_id'] in held_bytes} <= {'rejected_disposition'}
            assert {r['escalation_id'] for r in rows if r['class'] != 'rejected_disposition'} == {note.id}
            assert run.scheduler.statuses[TASK_ID] == ['done']
            pins = classify_pins(TASK_ID, queue.get_by_task(TASK_ID, status='pending'), live_claimant=False)
            assert pins.pins
            assert set(pins.queue_handoff) == set(held_bytes)

        _exit_without_driving(run, 'done', WorkflowOutcome.DONE)
        assert await run.run() == WorkflowOutcome.DONE
        assert read_hold(run.current(note)) is not None
        _assert_the_holds_are_untouched()

        ports = RouterPorts(
            queue=queue, event_store=events, dispatch_tool=run.curator,
            project_root=str(run.config.project_root), filing_claimant_run_id=None,
        )
        for hold_id in held_bytes:
            applied = apply_reviewer_disposition(
                NoteDisposition(hold_id, ReviewerDisposition.ADDRESSED, 'looks done'), ports,
                task_id=TASK_ID, enabled=True,
            )
            assert applied.outcome is Outcome.REJECTED
        _assert_the_holds_are_untouched()

        await _reap(run, mock_orch_config, orphan_timeout_secs=0.0)
        _assert_the_holds_are_untouched()

        await _amnesty(_harness_after(run, mock_orch_config))
        _assert_the_holds_are_untouched()


class _FailsOneClose(EscalationQueue):
    """A queue whose close of the armed record fails once; every other write lands."""

    def __init__(self, queue_dir: Path) -> None:
        super().__init__(queue_dir)
        self.armed: str | None = None

    def resolve(self, escalation_id: str, resolution: str, dismiss: bool = False, **kwargs) -> Escalation | None:
        if escalation_id == self.armed and kwargs.get('resolution_class') is not None:
            self.armed = None
            raise OSError('No space left on device')
        return super().resolve(escalation_id, resolution, dismiss, **kwargs)


@pytest.mark.asyncio
class TestRow20RouterFailureNeverChangesTheExit:
    async def test_a_failed_close_promotes_router_error_and_the_task_still_merges(
        self, tmp_path, monkeypatch, caplog,
    ):
        queue = _FailsOneClose(tmp_path / 'escalation_queue')
        run = RouterRun(tmp_path, monkeypatch, DispositioningReviewer(), queue=queue)
        note = implementer_note(queue, suggested_action='No action needed')
        queue.armed = note.id
        repo = run.config.project_root
        main_before = _git(repo, 'rev-parse', 'main').stdout.strip()

        with caplog.at_level(logging.ERROR, logger='orchestrator.escalation_router'):
            assert await run.run() == WorkflowOutcome.DONE

        assert queue.armed is None, 'the finalizer never attempted the close'
        assert _git(repo, 'rev-parse', 'main').stdout.strip() != main_before
        assert run.current(note).status == 'dismissed'
        (l1,) = _l1s(queue, agent_role=AGGREGATE_ROLE)
        assert 'router_error' in l1.detail
        assert note.id in l1.detail
        (row,) = run.rows(note.id)
        assert (row['class'], row['reason'], row['by']) == ('promoted', 'router_error', 'workflow-finalizer')
        assert [
            (record.escalation_id, record.by, record.exit_kind) for record in caplog.records
            if record.name == 'orchestrator.escalation_router' and record.levelno == logging.ERROR
        ] == [(note.id, 'workflow-finalizer', 'done')]
