"""The info-L0 router's workflow call sites (plans/info-l0-disposition-router-prd.md).

D10: the five blanket level-0 dismissals a workflow makes on its way out leave
an info note the router owns pending for it, and dismiss everything else
exactly as before.  The exit finalizer then routes every such note on every
terminal exit, and the post-merge tail starts no steward for one (D7).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _curator_fake import FakeCurator
from _orch_helpers import wait_responsive
from _workflow_helpers import (
    AgentStub,
    FakeScheduler,
    _build_workflow_with_escalation,
    _make_git_ops,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig, TimeoutsConfig, UnblockAutoConfig
from orchestrator.escalation_router import ROUTER_ID, hold_note, read_hold
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome, WorkflowState
from orchestrator.workflow_types import (
    StewardInterrupted,
    StewardReescalatedL1,
    StewardResolved,
    TerminalReport,
)

TASK_ID = '42'


def _config(project_root: Path, *, router_enabled: bool) -> OrchestratorConfig:
    """Steward waits short enough to expire inside a test; no dry-run unblock."""
    return OrchestratorConfig(
        project_root=project_root,
        max_concurrent_tasks=1,
        steward_completion_timeout=0.25,
        timeouts=TimeoutsConfig(steward=0.25),
        unblock_auto=UnblockAutoConfig(enabled=False),
        info_l0_router_enabled=router_enabled,
        git=GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
    )


def _assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id=TASK_ID,
        task={
            'id': TASK_ID, 'title': 'X', 'description': 'Y', 'status': 'pending',
            'metadata': {'files': ['lib']}, 'dependencies': [],
        },
        modules=['lib'],
    )


def _workflow(tmp_path: Path, *, router_enabled: bool) -> tuple[TaskWorkflow, EscalationQueue]:
    config = _config(tmp_path, router_enabled=router_enabled)
    workflow, _scheduler, queue = _build_workflow_with_escalation(
        config, _make_git_ops(config.project_root), _assignment(), AgentStub(), tmp_path,
        spawn_merge_worker=False,
    )
    return workflow, queue


def _submit(
    queue: EscalationQueue, *, severity: str, level: int = 0,
    suggested_action: str = 'create_followup_task',
) -> Escalation:
    esc = Escalation(
        id=queue.make_id(TASK_ID), task_id=TASK_ID, agent_role='implementer',
        severity=severity, category='task_failure' if severity != 'info' else 'design_concern',
        summary=f'{severity} L{level} note', detail=f'{severity} detail',
        suggested_action=suggested_action, level=level,
    )
    queue.submit(esc)
    return esc


def _status(queue: EscalationQueue, esc: Escalation) -> str:
    current = queue.get(esc.id)
    assert current is not None
    return current.status


def _wip_interrupting_steward() -> type:
    """A steward that publishes a wip-present interruption as soon as it starts."""

    class _Steward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002 — factory signature
            self._channel: asyncio.Queue | None = None

        def set_outcome_channel(self, channel: asyncio.Queue) -> None:
            self._channel = channel

        def set_wip_probe(self, probe) -> None:  # noqa: ARG002 — steward interface
            pass

        async def start(self) -> None:
            assert self._channel is not None
            self._channel.put_nowait(StewardInterrupted('attempt_cap', wip_commits_present=True))

        async def stop(self) -> None:
            pass

    return _Steward


async def _blocked_fall_through(workflow: TaskWorkflow, queue: EscalationQueue, tmp_path: Path) -> None:  # noqa: ARG001
    assert await workflow._mark_blocked('synthetic failure') == WorkflowOutcome.BLOCKED


async def _skip_escalation_sweep(workflow: TaskWorkflow, queue: EscalationQueue, tmp_path: Path) -> None:  # noqa: ARG001
    await workflow._mark_blocked('synthetic failure', skip_escalation=True)


async def _wip_interruption(workflow: TaskWorkflow, queue: EscalationQueue, tmp_path: Path) -> None:  # noqa: ARG001
    worktree = tmp_path / 'wt'
    worktree.mkdir()
    workflow.worktree = worktree
    workflow._steward_factory = _wip_interrupting_steward()
    assert await workflow._mark_blocked('synthetic failure') == WorkflowOutcome.REQUEUED


async def _open_l1_override(workflow: TaskWorkflow, queue: EscalationQueue, tmp_path: Path) -> None:  # noqa: ARG001
    _submit(queue, severity='blocking', level=1)
    channel: asyncio.Queue = asyncio.Queue()
    channel.put_nowait(StewardInterrupted('timeout', wip_commits_present=True))
    workflow._steward_outcome_channel = channel
    assert isinstance(await workflow._await_steward_completion(), StewardReescalatedL1)


async def _steward_wait_timeout(workflow: TaskWorkflow, queue: EscalationQueue, tmp_path: Path) -> None:  # noqa: ARG001
    await asyncio.wait_for(workflow._wait_for_resolution(), 10)
    assert workflow._steward_wait_expired


DISMISSAL_SITES = {
    'blocked-fall-through': _blocked_fall_through,
    'skip-escalation-sweep': _skip_escalation_sweep,
    'wip-interruption': _wip_interruption,
    'open-l1-override': _open_l1_override,
    'steward-wait-timeout': _steward_wait_timeout,
}


@pytest.mark.asyncio
class TestBlanketDismissalsSpareRouterOwnedNotes:
    """D10 (boundary #14 is the BLOCKED fall-through)."""

    @pytest.mark.parametrize('site', DISMISSAL_SITES.values(), ids=DISMISSAL_SITES.keys())
    async def test_the_info_note_survives_and_the_blocking_l0_is_dismissed(
        self, tmp_path: Path, site,
    ):
        workflow, queue = _workflow(tmp_path, router_enabled=True)
        blocking = _submit(queue, severity='blocking')
        info = _submit(queue, severity='info')

        await site(workflow, queue, tmp_path)

        assert _status(queue, blocking) == 'dismissed'
        assert _status(queue, info) == 'pending'

    @pytest.mark.parametrize('site', DISMISSAL_SITES.values(), ids=DISMISSAL_SITES.keys())
    async def test_with_the_router_disabled_the_info_note_is_dismissed_too(
        self, tmp_path: Path, site,
    ):
        workflow, queue = _workflow(tmp_path, router_enabled=False)
        blocking = _submit(queue, severity='blocking')
        info = _submit(queue, severity='info')

        await site(workflow, queue, tmp_path)

        assert _status(queue, blocking) == 'dismissed'
        assert _status(queue, info) == 'dismissed'


def _finalizer_workflow(
    tmp_path: Path, *, router_enabled: bool = True,
) -> tuple[TaskWorkflow, EscalationQueue, FakeCurator, list[str]]:
    """A workflow whose exit the router can observe: a real event store, the
    curator double as ``dispatch_tool``, and every other cleanup a spy."""
    workflow, queue = _workflow(tmp_path, router_enabled=router_enabled)
    workflow.event_store = EventStore(tmp_path / 'runs.db', run_id='run-test')
    curator = FakeCurator()
    workflow.scheduler.dispatch_tool = curator  # type: ignore[method-assign]
    return workflow, queue, curator, _spy_cleanups(workflow)


def _spy_cleanups(workflow: TaskWorkflow) -> list[str]:
    calls: list[str] = []

    async def _heartbeat() -> None:
        calls.append('stop_claimant_heartbeat')

    async def _worktree() -> None:
        calls.append('cleanup_done_worktree')

    def _config_dir() -> None:
        calls.append('cleanup_config_dir')

    async def _release_lane(_task_id: str) -> bool:
        calls.append('release_lane')
        return True

    workflow._stop_claimant_heartbeat = _heartbeat  # type: ignore[method-assign]
    workflow._maybe_cleanup_done_worktree = _worktree  # type: ignore[method-assign]
    workflow._cleanup_config_dir = _config_dir  # type: ignore[method-assign]
    workflow.git_ops.release_lane_for_terminal_task = _release_lane  # type: ignore[method-assign]
    return calls


def _scheduler(workflow: TaskWorkflow) -> FakeScheduler:
    scheduler = workflow.scheduler
    assert isinstance(scheduler, FakeScheduler)
    return scheduler


async def _exit_with(workflow: TaskWorkflow, outcome: WorkflowOutcome, *, row: str) -> TerminalReport:
    """Run ``run()`` with a ``_drive`` that writes the exit's row and returns *outcome*."""

    async def _drive() -> WorkflowOutcome:
        await _scheduler(workflow).set_task_status(TASK_ID, row)
        return outcome

    workflow._drive = _drive  # type: ignore[method-assign]
    return await workflow.run()


def _current(queue: EscalationQueue, esc: Escalation) -> Escalation:
    current = queue.get(esc.id)
    assert current is not None
    return current


def _dispositions(workflow: TaskWorkflow) -> list[dict]:
    store = workflow.event_store
    assert store is not None
    return [row['data'] for row in store.fetch_events_by_type(EventType.info_l0_dispositioned)]


def _unrouted_info_notes(queue: EscalationQueue) -> list[Escalation]:
    return [
        e for e in queue.get_by_task(TASK_ID, status='pending', level=0)
        if e.severity == 'info' and read_hold(e) is None
    ]


async def _reaper_l1s_after_aging(config: MagicMock, queue: EscalationQueue) -> list[Escalation]:
    """One orphan-reaper pass over *queue*'s directory with every note past a
    zero-second orphan timeout; returns the verbatim promotions it filed."""
    config.orphan_l0_timeout_secs = 0.0
    config.orphan_l0_dispatch_freshness_secs = 120.0
    config.info_l0_router_enabled = True
    config.info_l0_router_ticket_timeout_secs = 900.0
    config.info_l0_router_max_conversions_per_sweep = 20
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        harness = Harness(config)
    harness.scheduler.is_actively_held = MagicMock(return_value=False)
    harness.scheduler.get_task = AsyncMock(return_value={'status': 'done', 'metadata': {}})
    harness.scheduler.dispatch_tool = FakeCurator()
    reaper_queue = EscalationQueue(queue.queue_dir)
    harness._escalation_queue = reaper_queue
    await harness._reap_orphan_l0_escalations()
    return [
        e for e in reaper_queue.get_pending()
        if e.level == 1 and e.agent_role == 'harness-orphan-reaper'
    ]


TERMINAL_EXITS = [
    pytest.param(WorkflowOutcome.MERGE_DEFERRED, 'merge-deferred', 'merge_deferred', id='merge-deferred'),
    pytest.param(WorkflowOutcome.DONE, 'done', 'done', id='done'),
    pytest.param(WorkflowOutcome.BLOCKED, 'blocked', 'blocked', id='blocked'),
]


@pytest.mark.asyncio
class TestExitFinalizerRoutesInfoNotes:
    """D7: the workflow's own exit is an info note's consumer (boundaries #5, #6, #15)."""

    async def test_the_finalizer_runs_after_the_steward_stops_and_before_the_lane_is_released(
        self, tmp_path: Path,
    ):
        workflow, _queue = _workflow(tmp_path, router_enabled=True)

        names = [name for name, _fn in workflow._on_terminal_cleanups()]

        assert 'disposition_info_l0s' in names
        assert names.index('disposition_info_l0s') == names.index('stop_steward') + 1
        assert names.index('disposition_info_l0s') < names.index('release_lane')

    @pytest.mark.parametrize(('outcome', 'row', 'exit_kind'), TERMINAL_EXITS)
    async def test_a_terminal_exit_leaves_no_info_note_unrouted(
        self, tmp_path: Path, mock_orch_config, outcome, row, exit_kind,
    ):
        workflow, queue, curator, _cleanups = _finalizer_workflow(tmp_path)
        work = _submit(queue, severity='info')
        observation = _submit(queue, severity='info', suggested_action='No action needed')

        await _exit_with(workflow, outcome, row=row)

        assert _unrouted_info_notes(queue) == []
        hold = read_hold(_current(queue, work))
        assert hold is not None and hold.ticket == 'tkt_1'
        assert len(curator.calls_named('submit_task')) == 1
        assert _current(queue, observation).resolution_class == 'observation-consumed'
        rows = _dispositions(workflow)
        assert sorted(r['class'] for r in rows) == ['convert-pending', 'observation-consumed']
        assert {(r['by'], r['exit_kind']) for r in rows} == {('workflow-finalizer', exit_kind)}
        assert await _reaper_l1s_after_aging(mock_orch_config, queue) == []

    async def test_a_requeued_exit_defers_the_note_to_the_next_incarnation(self, tmp_path: Path):
        workflow, queue, curator, _cleanups = _finalizer_workflow(tmp_path)
        note = _submit(queue, severity='info')

        await _exit_with(workflow, WorkflowOutcome.REQUEUED, row='pending')

        after = _current(queue, note)
        assert (after.status, after.triaged_by) == ('pending', None)
        (row,) = _dispositions(workflow)
        assert (row['class'], row['by'], row['exit_kind']) == ('defer', 'workflow-finalizer', 'requeued')
        assert curator.calls == []

    async def test_a_soft_cancel_on_a_done_row_routes_inside_the_shielded_cleanup(
        self, tmp_path: Path, mock_orch_config,
    ):
        workflow, queue, _curator, _cleanups = _finalizer_workflow(tmp_path)
        note = _submit(queue, severity='info')
        driving = asyncio.Event()

        async def _drive() -> WorkflowOutcome:
            driving.set()
            await asyncio.Event().wait()
            raise AssertionError('the soft cancel never interrupted _drive')

        workflow._drive = _drive  # type: ignore[method-assign]
        running = asyncio.create_task(workflow.run())
        await wait_responsive(driving.wait(), label='the workflow is driving')
        await _scheduler(workflow).set_task_status(TASK_ID, 'done')
        workflow._cancel_event.set()
        report = await asyncio.wait_for(running, 10)

        assert report.outcome == WorkflowOutcome.DONE
        assert read_hold(_current(queue, note)) is not None
        (row,) = _dispositions(workflow)
        assert (row['class'], row['by'], row['exit_kind']) == (
            'convert-pending', 'workflow-finalizer', 'cancelled',
        )
        assert await _reaper_l1s_after_aging(mock_orch_config, queue) == []


@pytest.mark.asyncio
class TestExitFinalizerNoOps:
    async def test_eval_mode_leaves_the_note_alone(self, tmp_path: Path):
        workflow, queue, curator, _cleanups = _finalizer_workflow(tmp_path)
        workflow._worktree_external = True
        note = _submit(queue, severity='info')

        await _exit_with(workflow, WorkflowOutcome.DONE, row='done')

        assert _current(queue, note).status == 'pending'
        assert curator.calls == []
        assert _dispositions(workflow) == []

    async def test_with_the_router_disabled_the_note_is_left_alone(self, tmp_path: Path):
        workflow, queue, curator, _cleanups = _finalizer_workflow(tmp_path, router_enabled=False)
        note = _submit(queue, severity='info')

        await _exit_with(workflow, WorkflowOutcome.DONE, row='done')

        assert _current(queue, note).status == 'pending'
        assert curator.calls == []
        assert _dispositions(workflow) == []

    async def test_a_held_note_is_left_for_the_reaper(self, tmp_path: Path):
        workflow, queue, curator, _cleanups = _finalizer_workflow(tmp_path)
        note = _submit(queue, severity='info')
        queue.stamp_triage(note.id, triaged_by=ROUTER_ID, triage_note=hold_note('tkt_9', datetime.now(UTC)))
        held = _current(queue, note)

        await _exit_with(workflow, WorkflowOutcome.DONE, row='done')

        after = _current(queue, note)
        assert (after.status, after.triage_note) == ('pending', held.triage_note)
        assert curator.calls == []

    async def test_an_unreadable_queue_does_not_stop_the_later_cleanups(self, tmp_path: Path):
        workflow, queue, _curator, cleanups = _finalizer_workflow(tmp_path)
        workflow.state = WorkflowState.DONE
        queue.get_by_task = MagicMock(side_effect=OSError('queue dir unreadable'))  # type: ignore[method-assign]

        await _exit_with(workflow, WorkflowOutcome.DONE, row='done')

        assert cleanups[-2:] == ['release_lane', 'cleanup_config_dir']


def _recording_steward(started: list[str]) -> type:
    """A steward that records its start and resolves at once."""

    class _Steward:
        def __init__(self, wt_path, cfg_dir):  # noqa: ARG002 — factory signature
            self._channel: asyncio.Queue | None = None

        def set_outcome_channel(self, channel: asyncio.Queue) -> None:
            self._channel = channel

        def set_wip_probe(self, probe) -> None:  # noqa: ARG002 — steward interface
            pass

        async def start(self) -> None:
            started.append('started')
            assert self._channel is not None
            self._channel.put_nowait(StewardResolved(resolution_text='resolved'))

        async def stop(self) -> None:
            pass

    return _Steward


async def _post_merge_tail(workflow: TaskWorkflow, tmp_path: Path) -> list[str]:
    """Drive ``_merge_and_finalise`` past a successful merge; returns steward starts."""
    worktree = tmp_path / 'wt'
    worktree.mkdir()
    workflow.worktree = worktree
    started: list[str] = []
    workflow._steward_factory = _recording_steward(started)
    workflow._run_merge_phase = AsyncMock(return_value=None)  # type: ignore[method-assign]
    workflow._clear_merge_retry_pending = AsyncMock()  # type: ignore[method-assign]
    workflow._clear_merge_phase_entered = AsyncMock()  # type: ignore[method-assign]
    workflow._write_completion_to_memory = AsyncMock()  # type: ignore[method-assign]
    workflow._finalise_merged_done = AsyncMock(return_value=WorkflowOutcome.DONE)  # type: ignore[method-assign]
    assert await workflow._merge_and_finalise(f'task/{TASK_ID}') == WorkflowOutcome.DONE
    return started


@pytest.mark.asyncio
class TestPostMergeTailStartsNoStewardForInfoNotes:
    """D7: info leftovers are the finalizer's, never a steward's."""

    async def test_only_an_info_note_pending_starts_no_steward(self, tmp_path: Path):
        workflow, queue = _workflow(tmp_path, router_enabled=True)
        _submit(queue, severity='info')

        assert await _post_merge_tail(workflow, tmp_path) == []

    async def test_a_blocking_l0_pending_still_starts_the_steward(self, tmp_path: Path):
        workflow, queue = _workflow(tmp_path, router_enabled=True)
        _submit(queue, severity='blocking')
        _submit(queue, severity='info')

        assert await _post_merge_tail(workflow, tmp_path) == ['started']
