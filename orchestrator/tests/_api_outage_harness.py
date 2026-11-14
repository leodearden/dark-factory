"""One simulated hour of a provider 5xx outage, driven through the real dispatch stack.

``run_incident_burst`` dispatches a scripted cast through the real
``Harness._run_slot`` -> real ``TaskWorkflow`` (from the real ``build_workflow``)
-> real ``Harness._apply_retry_cap`` -> a real ``Scheduler`` (park-stop wired to
the production callback) and a real ``EscalationQueue``, on one simulated clock.
Only boundaries owned by other PRDs or other processes are faked: the agent CLI
(``invoke_with_cap_retry`` returns scripted ``AgentResult``s), the worktree and
already-merged probes, the briefing, the steward factory, and fused-memory (an
in-memory task store behind the scheduler's ``mcp_session`` seam).
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import random
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _orch_helpers import HermeticMcpSession
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.scheduler_state import RequeueCooldown

from orchestrator.agents.invoke import AgentResult
from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import OrchestratorConfig, TranscriptArchiveConfig
from orchestrator.harness import Harness, TaskReport
from orchestrator.scheduler import RequeueRecord, Scheduler, TaskAssignment
from orchestrator.workflow import TaskWorkflow, WorkflowOutcome
from orchestrator.workflow import build_workflow as real_build_workflow

__all__ = [
    'BURST_SIZE',
    'KILLS_PER_TASK',
    'PROVIDER_529_ERROR',
    'SEED',
    'BurstCast',
    'BurstLedger',
    'DispatchRecord',
    'PhaseSnapshot',
    'ZeroOutputEvidence',
    'incident_529_result',
    'run_incident_burst',
]

BURST_SIZE = 15
KILLS_PER_TASK = 5
SEED = 3320

PROVIDER_529_ERROR = (
    'API Error: 529 {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}'
)

PHASE_A = 'A'

_EPOCH = datetime(2026, 7, 29, 9, 0, tzinfo=UTC)
_RUN_ID = 'run-3320-incident-burst'
_BURST_FIRST_ID = 9001
_STAGGER_SECS = 20.0
_KILL_DURATION_MS = 120_000


def incident_529_result() -> AgentResult:
    """The SIGTERM-flushed 529 kill: zero output, the provider error at the end of the stream."""
    return AgentResult(
        success=False,
        output='x' * 10_000 + PROVIDER_529_ERROR,
        timed_out=True,
        turns=0,
        cost_usd=0.0,
        duration_ms=_KILL_DURATION_MS,
        transcript_turns=0,
        api_error_status=529,
    )


@dataclass(frozen=True)
class ZeroOutputEvidence:
    """The fields of a dispatch's zero-output evidence file that the gate reads."""

    api_error_status: int | None
    output_tail: str

    @classmethod
    def read(cls, path: Path) -> ZeroOutputEvidence:
        data = json.loads(path.read_text())
        return cls(api_error_status=data['api_error_status'], output_tail=data['output_tail'])


@dataclass(frozen=True)
class DispatchRecord:
    """One dispatch's report, and what the operator read paths showed right after it."""

    task_id: str
    phase: str
    at: float
    report: TaskReport
    cli_calls: int
    evidence: tuple[ZeroOutputEvidence, ...]
    cooldown: RequeueCooldown | None
    transient_requeue_count: int
    requeue_history: tuple[RequeueRecord, ...]
    parked_live_count: int
    parked_window_churn_count: int
    paused: bool

    @property
    def genuine_requeue_count(self) -> int:
        """Requeues counted against ``requeue_cap``; the cast files no non-counting requeue."""
        return len(self.requeue_history) - self.transient_requeue_count


@dataclass(frozen=True)
class PhaseSnapshot:
    pending_escalations: tuple[Escalation, ...]
    paused: bool
    parked_live_count: int
    parked_window_churn_count: int


@dataclass(frozen=True)
class BurstCast:
    burst: tuple[str, ...]

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.burst


@dataclass(frozen=True)
class BurstLedger:
    config: OrchestratorConfig
    cast: BurstCast
    dispatches: tuple[DispatchRecord, ...]
    after_phase_a: PhaseSnapshot
    status_writes: Mapping[str, tuple[str, ...]]
    escalations_by_task: Mapping[str, tuple[Escalation, ...]]
    park_stop_trip_wired: bool

    def dispatches_of(self, task_id: str) -> tuple[DispatchRecord, ...]:
        return tuple(r for r in self.dispatches if r.task_id == task_id)

    def kills_of(self, task_id: str) -> tuple[DispatchRecord, ...]:
        """The task's phase-A dispatches: for a burst task, its incident kills."""
        return tuple(r for r in self.dispatches_of(task_id) if r.phase == PHASE_A)


async def run_incident_burst(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BurstLedger:
    simulation = _Simulation(tmp_path, monkeypatch)
    await simulation.run_phase_a()
    return simulation.ledger()


class _SimClock:
    """Seconds since the incident began; feeds the scheduler's monotonic and wall clocks."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def wall(self) -> datetime:
        return _EPOCH + timedelta(seconds=self.now)

    def advance_to(self, at: float) -> None:
        assert at >= self.now, f'the simulated clock may not run backwards ({self.now} -> {at})'
        self.now = at


class _OutageTaskStore(HermeticMcpSession):
    """fused-memory as the scheduler sees it: one status per task and an ordered write log."""

    def __init__(self, task_ids: tuple[str, ...]) -> None:
        super().__init__()
        self._status = dict.fromkeys(task_ids, 'pending')
        self._writes: dict[str, list[str]] = {tid: [] for tid in task_ids}

    async def call_tool(self, name: str, arguments: dict, timeout: float = 30) -> dict:
        if name == 'set_task_status':
            task_id, status = str(arguments['id']), str(arguments['status'])
            self._status[task_id] = status
            self._writes[task_id].append(status)
        if name == 'get_task':
            task_id = str(arguments['id'])
            return self._envelope(json.dumps({'id': task_id, 'status': self._status[task_id]}))
        return await super().call_tool(name, arguments, timeout)

    def writes(self) -> dict[str, tuple[str, ...]]:
        return {tid: tuple(log) for tid, log in self._writes.items()}


class _CliScript:
    """``invoke_with_cap_retry`` for the cast: each task's n-th call returns ``script(n)``.

    Calls are attributed to a task by the ``cwd`` its dispatch's worktree registered.
    """

    def __init__(self, scripts: Mapping[str, Callable[[int], AgentResult]]) -> None:
        self._scripts = scripts
        self._task_by_cwd: dict[Path, str] = {}
        self._calls: Counter[str] = Counter()

    def register_worktree(self, worktree: Path, task_id: str) -> None:
        self._task_by_cwd[worktree] = task_id

    def calls(self, task_id: str) -> int:
        return self._calls[task_id]

    def next_duration_secs(self, task_id: str) -> float:
        return self._scripts[task_id](self._calls[task_id] + 1).duration_ms / 1000

    async def __call__(self, **kwargs: object) -> AgentResult:
        task_id = self._task_by_cwd[Path(str(kwargs['cwd']))]
        self._calls[task_id] += 1
        return self._scripts[task_id](self._calls[task_id])


def _burst_script(n: int) -> AgentResult:
    if n <= KILLS_PER_TASK:
        return incident_529_result()
    raise AssertionError(f'burst task invoked a {n}th time; the script ends at kill {KILLS_PER_TASK}')


class _WorkflowFactory:
    """``build_workflow`` for ``_run_slot``: the real workflow, its peripheral phases stubbed."""

    def __init__(
        self, dispatch_root: Path, cli: _CliScript, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        self._dispatch_root = dispatch_root
        self._cli = cli
        self._monkeypatch = monkeypatch
        self._sequence = itertools.count(1)
        self._latest: dict[str, TaskWorkflow] = {}

    def latest(self, task_id: str) -> TaskWorkflow:
        return self._latest[task_id]

    def __call__(self, **kwargs) -> TaskWorkflow:
        assignment: TaskAssignment = kwargs['assignment']
        task_id = assignment.task_id
        wf = real_build_workflow(
            **{**kwargs, 'steward_factory': None, 'initial_plan': _one_step_plan(task_id)},
        )
        lane = self._dispatch_root / f'{next(self._sequence):03d}-{task_id}' / 'lane'
        self._monkeypatch.setattr(wf, '_setup_worktree_and_artifacts', self._setup(wf, lane))
        for probe in (
            '_recover_if_already_merged',
            '_resume_merge_retry_if_pending',
            '_recover_before_execute',
        ):
            self._monkeypatch.setattr(wf, probe, AsyncMock(return_value=None))
        self._latest[task_id] = wf
        return wf

    def _setup(self, wf: TaskWorkflow, lane: Path):
        async def setup_worktree_and_artifacts(branch_name: str) -> None:
            lane.mkdir(parents=True)
            artifacts = TaskArtifacts(lane, *wf._artifact_roots(lane))
            artifacts.init(wf.task_id, wf.task.get('title', ''), wf.task.get('description', ''))
            wf.worktree = lane
            wf.artifacts = artifacts
            self._cli.register_worktree(lane, wf.task_id)

        return setup_worktree_and_artifacts


def _one_step_plan(task_id: str) -> dict:
    return {
        'task_id': task_id,
        'title': f'task {task_id}',
        'files': [],
        'analysis': 'incident-burst simulation',
        'prerequisites': [],
        'steps': [{
            'id': 'step-1', 'type': 'impl', 'description': 'do the work', 'status': 'pending',
            'commit': None, 'prerequisites': [], 'files': [],
        }],
    }


def _code_default_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OrchestratorConfig:
    """Package defaults for every PRD-owned knob; only substrate knobs are overridden."""
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(tmp_path / 'no-such-config.yaml'))
    return OrchestratorConfig(
        project_root=tmp_path,
        transcript_archive=TranscriptArchiveConfig(enabled=False),
        inter_iteration_rebase=False,
    )


def _build_harness(
    config: OrchestratorConfig, store: _OutageTaskStore, clock: _SimClock, queue_dir: Path,
) -> tuple[Harness, Scheduler]:
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        harness = Harness(config)
    scheduler = Scheduler(
        config,
        callbacks=harness.scheduler.callbacks,
        mcp_session=store,
        time_source=clock,
        monotonic_clock_source=clock,
        wall_time_source=clock.wall,
        jitter_source=random.Random(SEED).uniform,
    )
    scheduler.finish_startup()
    harness.scheduler = scheduler
    harness._escalation_queue = EscalationQueue(queue_dir)
    harness._run_id = _RUN_ID
    harness.briefing.build_implementer_prompt = AsyncMock(return_value='Implement step-1.')
    return harness, scheduler


class _Simulation:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.cast = BurstCast(burst=tuple(str(_BURST_FIRST_ID + i) for i in range(BURST_SIZE)))
        self.clock = _SimClock()
        self.store = _OutageTaskStore(self.cast.all_ids)
        self.cli = _CliScript(dict.fromkeys(self.cast.burst, _burst_script))
        self.config = _code_default_config(tmp_path, monkeypatch)
        self.harness, self.scheduler = _build_harness(
            self.config, self.store, self.clock, tmp_path / 'escalations',
        )
        self.factory = _WorkflowFactory(tmp_path / 'dispatches', self.cli, monkeypatch)
        monkeypatch.setattr('orchestrator.harness.build_workflow', self.factory)
        monkeypatch.setattr('orchestrator.workflow.invoke_with_cap_retry', self.cli)
        self._sequence = itertools.count()
        self.dispatches: list[DispatchRecord] = []
        self.after_phase_a: PhaseSnapshot | None = None

    async def run_phase_a(self) -> None:
        arrivals: list[tuple[float, int, str]] = []
        for i, task_id in enumerate(self.cast.burst):
            first = i * _STAGGER_SECS + self.cli.next_duration_secs(task_id)
            heapq.heappush(arrivals, (first, next(self._sequence), task_id))
        while arrivals:
            at, _, task_id = heapq.heappop(arrivals)
            record = await self._dispatch(task_id, at, PHASE_A)
            if record.report.outcome == WorkflowOutcome.REQUEUED and self.cli.calls(task_id) < KILLS_PER_TASK:
                heapq.heappush(arrivals, (self._redispatch_at(record), next(self._sequence), task_id))
        self.after_phase_a = self._snapshot()

    def _redispatch_at(self, record: DispatchRecord) -> float:
        if record.cooldown is None:
            raise AssertionError(f'task {record.task_id} requeued with no cooldown armed')
        return record.at + record.cooldown.armed_secs + self.cli.next_duration_secs(record.task_id)

    async def _dispatch(self, task_id: str, at: float, phase: str) -> DispatchRecord:
        self.clock.advance_to(at)
        await self.scheduler.set_task_status(task_id, 'in-progress')
        calls_before = self.cli.calls(task_id)
        slot = asyncio.Semaphore(1)
        await slot.acquire()
        report = await self.harness._run_slot(_assignment(task_id), slot)
        assert report is not None, f'task {task_id}: _run_slot returned no report'
        record = DispatchRecord(
            task_id=task_id,
            phase=phase,
            at=at,
            report=report,
            cli_calls=self.cli.calls(task_id) - calls_before,
            evidence=_evidence(self.factory.latest(task_id)),
            cooldown=self.scheduler.get_state_snapshot().requeue_cooldowns.get(task_id),
            transient_requeue_count=self.scheduler.transient_requeue_count(task_id),
            requeue_history=tuple(self.scheduler.requeue_history(task_id)),
            parked_live_count=self.scheduler.parked_live_count,
            parked_window_churn_count=self.scheduler.parked_window_churn_count,
            paused=self.scheduler.is_paused,
        )
        self.dispatches.append(record)
        return record

    def _snapshot(self) -> PhaseSnapshot:
        queue = self.harness._escalation_queue
        assert queue is not None
        return PhaseSnapshot(
            pending_escalations=tuple(queue.get_pending()),
            paused=self.scheduler.is_paused,
            parked_live_count=self.scheduler.parked_live_count,
            parked_window_churn_count=self.scheduler.parked_window_churn_count,
        )

    def ledger(self) -> BurstLedger:
        queue = self.harness._escalation_queue
        assert queue is not None and self.after_phase_a is not None
        return BurstLedger(
            config=self.config,
            cast=self.cast,
            dispatches=tuple(self.dispatches),
            after_phase_a=self.after_phase_a,
            status_writes=self.store.writes(),
            escalations_by_task={tid: tuple(queue.get_by_task(tid)) for tid in self.cast.all_ids},
            park_stop_trip_wired=self.scheduler.callbacks.on_park_stop_trip is not None,
        )


def _assignment(task_id: str) -> TaskAssignment:
    return TaskAssignment(
        task_id=task_id,
        task={
            'id': task_id, 'title': f'task {task_id}', 'description': 'incident-burst cast member',
            'status': 'pending', 'metadata': {}, 'dependencies': [],
        },
        modules=['lib'],
    )


def _evidence(wf: TaskWorkflow) -> tuple[ZeroOutputEvidence, ...]:
    if wf.artifacts is None:
        return ()
    pattern = Path(wf.artifacts.zero_output_evidence_glob())
    return tuple(ZeroOutputEvidence.read(p) for p in sorted(pattern.parent.glob(pattern.name)))
