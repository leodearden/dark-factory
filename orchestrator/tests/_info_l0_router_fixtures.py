"""Drivers for the info-L0 router's real call sites over one shared escalation queue.

plans/info-l0-disposition-router-prd.md: the router runs at a workflow's review
loop and exit (``RouterRun``), at the orphan reaper and at the startup amnesty
(``reaper_harness``).  Building every site over the same queue directory and
the same ``FakeCurator`` lets a test follow one note through the whole chain.
A consuming test module that runs a full workflow must re-export
``_workflow_helpers._derive_meta_root_like_production`` so pytest applies it.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from _curator_fake import FakeCurator
from _merge_lane_fakes import lane_scene_config
from _workflow_helpers import (
    AgentStub,
    FakeBriefing,
    FakeMcp,
    FakeScheduler,
    _init_repo,
    _make_git_ops,
    agent_side_artifacts,
)
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.cli_invoke import AgentResult

from orchestrator.agents.review_briefing import ReviewBriefing, ReviewSubject
from orchestrator.agents.roles import STEWARD
from orchestrator.config import (
    GitConfig,
    OrchestratorConfig,
    ReviewConfig,
    SandboxConfig,
    TimeoutsConfig,
    UnblockAutoConfig,
)
from orchestrator.escalation_rulings import Ruling
from orchestrator.event_store import EventStore, EventType
from orchestrator.harness import Harness
from orchestrator.mcp.verdict_tools import _envelope
from orchestrator.merge_lane import MergeLane
from orchestrator.scheduler import TaskAssignment
from orchestrator.verify import VerifyResult
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome

TASK_ID = '42'
PASSED = VerifyResult(
    passed=True, test_output='', lint_output='', type_output='', summary='All checks passed',
)


def run_config(
    project_root: Path, *, briefing: ReviewBriefing, router_enabled: bool,
) -> OrchestratorConfig:
    return lane_scene_config(
        project_root,
        GitConfig(
            main_branch='main', branch_prefix='task/', remote='origin',
            worktree_dir='.worktrees',
        ),
        max_concurrent_tasks=1,
        max_execute_iterations=5,
        max_verify_attempts=3,
        max_review_cycles=2,
        judge_after_each_iteration=False,
        max_amendment_rounds=1,
        reviewer_stagger_secs=0.0,
        steward_completion_timeout=0.25,
        timeouts=TimeoutsConfig(steward=0.25),
        unblock_auto=UnblockAutoConfig(enabled=False),
        info_l0_router_enabled=router_enabled,
        review=ReviewConfig(reviewer_briefing=briefing),
        sandbox=SandboxConfig(enabled=False),
    )


def disposition_entry(esc_id: str, disposition: str) -> dict:
    """One entry of a reviewer verdict's ``note_dispositions``."""
    return {'escalation_id': esc_id, 'disposition': disposition, 'rationale': f'{disposition} rationale'}


def disposition_events(store: EventStore | None, esc_id: str | None = None) -> list[dict]:
    """The ``info_l0_dispositioned`` events in *store*, only *esc_id*'s when given."""
    assert store is not None
    return [
        row for row in store.fetch_events_by_type(EventType.info_l0_dispositioned)
        if esc_id is None or row['data']['escalation_id'] == esc_id
    ]


def disposition_rows(store: EventStore | None, esc_id: str | None = None) -> list[dict]:
    """The data of :func:`disposition_events`."""
    return [row['data'] for row in disposition_events(store, esc_id)]


def assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id=TASK_ID,
        task={
            'id': TASK_ID, 'title': 'X', 'description': 'Y', 'status': 'pending',
            'metadata': {'files': ['lib']}, 'dependencies': [],
        },
        modules=['lib'],
    )


class DispositioningReviewer(AgentStub):
    """A reviewer whose PASS verdict carries *note_dispositions*."""

    def __init__(self) -> None:
        super().__init__()
        self.note_dispositions: list[dict] = []

    def _reviewer(self, role: str, output_schema: dict | None) -> AgentResult:
        result = super()._reviewer(role, output_schema)
        assert result.structured_output is not None
        result.structured_output['note_dispositions'] = self.note_dispositions
        return result


class LostVerdictReviewer(DispositioningReviewer):
    """A reviewer whose verdict file is present but off-contract, so the workflow discards it."""

    async def invoke_agent(self, prompt: str, system_prompt: str, cwd: Path, **kwargs) -> AgentResult:
        role = self._detect_role(system_prompt)
        if not role.startswith('reviewer'):
            return await super().invoke_agent(prompt, system_prompt, cwd, **kwargs)
        self.calls.append(role)
        artifacts = agent_side_artifacts(cwd)
        artifacts.write_verdict(role, _envelope(role, 'sid', {
            'reviewer': role, 'verdict': 'MAYBE', 'issues': [], 'summary': 's',
            'note_dispositions': self.note_dispositions,
        }))
        return AgentResult(success=True, output='')


class NoteRecordingBriefing(FakeBriefing):
    """Canned prompts that also record which notes each reviewer briefing listed."""

    def __init__(self) -> None:
        self.briefed: list[tuple[ReviewBriefing, list[str]]] = []

    async def build_reviewer_prompt(
        self, reviewer_type: str, diff: str, context: str | None = None,
        *, amendment_suggestions: list[dict] | None = None,
        task: dict | None = None,
        open_info_notes: Sequence[Escalation] | None = None,
        rulings: Sequence[Ruling] = (),
    ) -> str:
        self._record(ReviewBriefing.DIFF_ONLY, open_info_notes)
        return await super().build_reviewer_prompt(reviewer_type, diff, context)

    async def build_context_complete_reviewer_prompt(
        self, reviewer_type: str, subject: ReviewSubject, context: str | None = None,
        *, amendment_suggestions: list[dict] | None = None,
        open_info_notes: Sequence[Escalation] | None = None,
        rulings: Sequence[Ruling] = (),
    ) -> str:
        self._record(ReviewBriefing.CONTEXT_COMPLETE, open_info_notes)
        return await super().build_context_complete_reviewer_prompt(reviewer_type, subject, context)

    def _record(self, generation: ReviewBriefing, notes: Sequence[Escalation] | None) -> None:
        self.briefed.append((generation, [note.id for note in notes or ()]))


class RouterRun:
    """One full ``TaskWorkflow`` run: AgentStub agents, real git, a real
    ``EscalationQueue`` and ``EventStore``, and the curator double as
    ``dispatch_tool``.

    ``queue`` shares a queue with another site (a second incarnation, a
    fault-injecting subclass); ``scheduler`` replaces the ``FakeScheduler``;
    ``merge_queue`` replaces the merge queue and starts no merge worker.
    """

    def __init__(
        self, tmp_path: Path, monkeypatch, reviewer: DispositioningReviewer, *,
        briefing: ReviewBriefing = ReviewBriefing.DIFF_ONLY, router_enabled: bool = True,
        queue: EscalationQueue | None = None, scheduler: FakeScheduler | None = None,
        merge_queue: asyncio.Queue | None = None, steward_factory: type | None = None,
    ) -> None:
        repo = tmp_path / 'repo'
        repo.mkdir(parents=True)
        _init_repo(repo)
        self.config = run_config(repo, briefing=briefing, router_enabled=router_enabled)
        self._git_ops = _make_git_ops(repo)
        self.queue = queue if queue is not None else EscalationQueue(tmp_path / 'escalation_queue')
        self.curator = FakeCurator()
        self.scheduler = scheduler if scheduler is not None else FakeScheduler()
        self.scheduler.dispatch_tool = self.curator
        self.briefing = NoteRecordingBriefing()
        self.event_store = EventStore(tmp_path / 'runs.db', run_id='run-test')
        self._steward_factory = steward_factory
        self._merge_queue: asyncio.Queue = merge_queue if merge_queue is not None else asyncio.Queue()
        self._merge_worker = None
        if merge_queue is None:
            self._merge_worker = MergeLane(self._git_ops, self._merge_queue)
            asyncio.create_task(self._merge_worker.run(), name='test-merge-worker')
        self._monkeypatch = monkeypatch
        monkeypatch.setattr('orchestrator.workflow.invoke_agent', reviewer.invoke_agent)
        monkeypatch.setattr(
            'orchestrator.workflow.run_scoped_verification', AsyncMock(return_value=PASSED),
        )
        self.workflow = self.build_workflow()

    def build_workflow(self, *, cancel_event: asyncio.Event | None = None) -> TaskWorkflow:
        """A fresh ``TaskWorkflow`` over this run's collaborators; it becomes ``self.workflow``.

        A harness slot builds its workflow with the ``cancel_event`` it registers.
        """
        self.workflow = TaskWorkflow(
            assignment=assignment(), config=self.config, git_ops=self._git_ops,
            scheduler=self.scheduler, briefing=self.briefing, mcp=FakeMcp(),
            escalation_queue=self.queue, steward_factory=self._steward_factory,
            merge_queue=self._merge_queue, merge_worker=self._merge_worker,
            event_store=self.event_store, cancel_event=cancel_event,
        )
        return self.workflow

    def fail_the_first_note_read_after_verification(self) -> None:
        """The reviewer's note read is the first pending-record read once verification passes."""
        armed = False
        read = self.queue.get_by_task

        def get_by_task(*args, **kwargs):
            nonlocal armed
            if armed and kwargs.get('status') == 'pending':
                armed = False
                raise OSError('disk unreadable')
            return read(*args, **kwargs)

        async def verify(*_args, **_kwargs) -> VerifyResult:
            nonlocal armed
            armed = True
            return PASSED

        self._monkeypatch.setattr(self.queue, 'get_by_task', get_by_task)
        self._monkeypatch.setattr('orchestrator.workflow.run_scoped_verification', verify)

    async def run(self) -> WorkflowOutcome:
        return (await self.workflow.run()).outcome

    def current(self, note: Escalation) -> Escalation:
        current = self.queue.get(note.id)
        assert current is not None
        return current

    def rows(self, esc_id: str) -> list[dict]:
        return disposition_rows(self.event_store, esc_id)


def router_reaper_config(
    base: MagicMock, *, router_enabled: bool = True, orphan_timeout_secs: float = 60.0,
    ticket_timeout_secs: float = 900.0, max_conversions: int = 20,
) -> MagicMock:
    """A copy of a ``mock_orch_config`` with the orphan-reaper and info-L0-router knobs set.

    *base* is left as it was, so two sites built from one fixture keep their own knobs.
    """
    config = copy.copy(base)
    config.orphan_l0_reaper_enabled = True
    config.orphan_l0_timeout_secs = orphan_timeout_secs
    config.orphan_l0_check_interval_secs = 1.0
    config.info_l0_router_enabled = router_enabled
    config.info_l0_router_ticket_timeout_secs = ticket_timeout_secs
    config.info_l0_router_max_conversions_per_sweep = max_conversions
    return config


def reaper_harness(
    config: MagicMock, queue: EscalationQueue, *, curator: FakeCurator,
    event_store: EventStore | None = None, task_row: dict | None = None,
) -> Harness:
    """A real Harness over *queue* whose reaper and amnesty can be driven.

    ``Harness.start()`` is bypassed: the scheduler is a mock that holds no
    task, answers ``get_task`` with *task_row*, and dispatches to *curator*.
    """
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        harness = Harness(config)
    harness.scheduler.is_actively_held = MagicMock(return_value=False)
    harness.scheduler.get_task = AsyncMock(
        return_value={'status': 'in-progress', 'metadata': {}} if task_row is None else task_row,
    )
    harness.scheduler.dispatch_tool = curator
    harness.event_store = event_store
    harness._escalation_queue = queue
    return harness


def handing_off_steward(
    queue: EscalationQueue, task_id: str, *, filer_role: str = STEWARD.name,
    severity: str | None = None,
) -> type:
    """A steward that hands the gating L0 to a human: files a level-1 under
    *filer_role* and dismisses the L0.  With the steward's role, both producers
    look like this in the queue — ``TaskSteward``'s own give-up path and its
    agent's ``escalate_blocker(level=1)``.  It acts after a short delay, as the
    real steward's background loop does, so the wait has begun before it hands
    off.  With *severity* it hands off only the L0s of that severity and stays
    silent on the rest."""

    class _FakeSteward:
        def __init__(self, wt_path, cfg_dir):
            self.give_up_task: asyncio.Task | None = None

        def set_outcome_channel(self, channel) -> None:
            pass

        def set_wip_probe(self, probe) -> None:
            pass

        async def _give_up(self) -> None:
            await asyncio.sleep(0.05)
            for esc in queue.get_by_task(task_id, status='pending', level=0):
                if severity is not None and esc.severity != severity:
                    continue
                queue.submit(Escalation(
                    id=queue.make_id(task_id),
                    task_id=task_id,
                    agent_role=filer_role,
                    severity=esc.severity,
                    category=esc.category,
                    summary=f'Steward re-escalation: {esc.summary}',
                    level=1,
                ))
                queue.resolve(
                    esc.id, 'Auto-dismissed: re-escalated to level 1',
                    dismiss=True, resolved_by='steward',
                )

        async def start(self) -> None:
            self.give_up_task = asyncio.create_task(self._give_up())

        async def stop(self) -> None:
            if self.give_up_task is not None:
                await self.give_up_task

    return _FakeSteward
