"""Test doubles for the shadow coordinator (orchestrator/shadow_eval.py::ShadowCoordinator).

Imported by bare module name (``from _shadow_coordinator_doubles import ...``),
like ``_merge_queue_doubles`` -- ``orchestrator/tests/`` has no ``__init__.py``.

Only the coordinator's effect boundaries are faked: the runner, the clock, the
escalation sink, the host sample and the isolation probe. The stores are real,
on one tmp ``runs.db``, so a test observes the rows and events production would.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from shared.cost_store import CostStore
from shared.psi import PsiSample

from orchestrator.config import PsiAdmissionConfig, ShadowEvalConfig
from orchestrator.evals.configs import (
    GLM_BASE_URL,
    GLM_MODEL,
    MINIMAX_BASE_URL,
    MINIMAX_MODEL,
    EvalConfig,
)
from orchestrator.evals.metrics import EvalMetrics
from orchestrator.event_store import EventStore, EventType
from orchestrator.run_store import RunStore
from orchestrator.shadow_eval import (
    CellRequest,
    CellRunOutcome,
    RunEnd,
    ShadowCoordinator,
    ShadowCoordinatorDeps,
    StormTrip,
)
from orchestrator.shadow_types import ShadowShape

PROJECT_ID = 'dark_factory'
RUN_ID = 'run-shadow-live'
START = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)

INCUMBENTS: dict[ShadowShape, str] = {
    ShadowShape.IMPLEMENTER: 'incumbent-implementer',
    ShadowShape.ARCHITECT: 'incumbent-architect',
    ShadowShape.ARCHITECT_CONSEQUENCE: 'incumbent-consequence',
    ShadowShape.END_TO_END: 'incumbent-end-to-end',
}


class FakeClock:
    """An aware-UTC clock that moves only when told to."""

    def __init__(self, start: datetime = START) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> None:
        self._now += delta


def metered_native_config() -> EvalConfig:
    return EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max')


def glm_subscription_config(token: str = 'zai-test-token') -> EvalConfig:
    return EvalConfig(
        'glm-5.3-endpoint', 'claude', GLM_MODEL, 'high',
        env_overrides={'ANTHROPIC_BASE_URL': GLM_BASE_URL, 'ANTHROPIC_AUTH_TOKEN': token},
    )


def minimax_metered_config(token: str = 'minimax-test-token') -> EvalConfig:
    return EvalConfig(
        'minimax-m3-endpoint', 'claude', MINIMAX_MODEL, 'high',
        env_overrides={'ANTHROPIC_BASE_URL': MINIMAX_BASE_URL, 'ANTHROPIC_AUTH_TOKEN': token},
    )


def blank_token_endpoint_config() -> EvalConfig:
    return minimax_metered_config(token='')


def completed(
    *,
    result_path: Path = Path('/tmp/shadow-results/cell.json'),
    cost_usd: float = 0.25,
    cost_credits: float = 0.0,
    plan: dict | None = None,
    **metrics: Any,
) -> CellRunOutcome:
    fields = {'tests_pass': True, 'lint_clean': True, 'typecheck_clean': True, 'plan_steps': 2}
    fields.update(metrics)
    return CellRunOutcome(
        end=RunEnd.COMPLETED,
        metrics=EvalMetrics(**fields),
        result_path=result_path,
        cost_usd=cost_usd,
        cost_credits=cost_credits,
        plan=plan,
    )


@dataclass
class Script:
    """What the fake runner does for one (shape, candidate)."""

    outcome: CellRunOutcome | None = None
    prepare_raises: BaseException | None = None
    run_raises: BaseException | None = None
    block: asyncio.Event | None = None
    before_return: Callable[[CellRequest], None] | None = None


class FakeRunner:
    """Implements the ShadowRunner protocol and records every call."""

    def __init__(self, worktree_root: Path, default: CellRunOutcome | None = None) -> None:
        self.worktree_root = worktree_root
        self.default = default or completed()
        self.scripts: dict[tuple[ShadowShape, str], Script] = {}
        self.prepared: list[CellRequest] = []
        self.ran: list[CellRequest] = []
        self.run_started = asyncio.Event()

    def script(self, shape: ShadowShape, candidate: str, **kwargs: Any) -> Script:
        script = Script(**kwargs)
        self.scripts[(shape, candidate)] = script
        return script

    def _script_for(self, request: CellRequest) -> Script:
        return self.scripts.get((request.cell.shape, request.cell.candidate), Script())

    def worktree_for(self, request: CellRequest) -> Path:
        return self.worktree_root / f'shadow_{request.cell.task_id}_{request.cell.cell_id}'

    async def prepare(self, request: CellRequest) -> Path:
        self.prepared.append(request)
        script = self._script_for(request)
        if script.prepare_raises is not None:
            raise script.prepare_raises
        return self.worktree_for(request)

    async def run(self, request: CellRequest, worktree: Path) -> CellRunOutcome:
        self.ran.append(request)
        self.run_started.set()
        script = self._script_for(request)
        if script.block is not None:
            await script.block.wait()
        if script.before_return is not None:
            script.before_return(request)
        if script.run_raises is not None:
            raise script.run_raises
        return script.outcome or self.default


class RecordingEscalator:
    """The storm escalation sink: records each trip, and raises when told to."""

    def __init__(self) -> None:
        self.trips: list[StormTrip] = []
        self.raises: BaseException | None = None

    async def __call__(self, trip: StormTrip) -> None:
        self.trips.append(trip)
        if self.raises is not None:
            raise self.raises


def make_task(
    task_id: str,
    *,
    base_sha: str | None = 'a' * 40,
    title: str = 'Add a shadow coordinator',
    description: str = 'Implement the coordinator core.',
) -> dict:
    metadata = {} if base_sha is None else {'branch_base_sha': base_sha}
    return {'id': task_id, 'title': title, 'description': description, 'metadata': metadata}


def shadow_config(**overrides: Any) -> ShadowEvalConfig:
    fields: dict[str, Any] = {
        'enabled': True,
        'sample_rate': 1.0,
        'shapes': [ShadowShape.IMPLEMENTER],
        'candidates': ['claude-sonnet-max'],
        'max_concurrent': 10,
    }
    fields.update(overrides)
    return ShadowEvalConfig(**fields)


def calm_sample() -> PsiSample:
    return PsiSample(cpu_some10=0.0, mem_some10=0.0, mem_full10=0.0, io_some10=0.0, read_ok=True)


@dataclass
class CoordinatorRig:
    """One coordinator wired to real stores on tmp_path and the fakes above."""

    coordinator: ShadowCoordinator
    config: ShadowEvalConfig
    store: RunStore
    events: EventStore
    costs: CostStore
    runner: FakeRunner
    clock: FakeClock
    escalator: RecordingEscalator
    candidates: dict[str, EvalConfig]
    host: list[PsiSample] = field(default_factory=list)
    isolation: list[bool] = field(default_factory=list)

    def events_of(self, event_type: EventType) -> list[dict]:
        """Every *event_type* row on the db, across runs, oldest first."""
        return self.events.fetch_events_by_type_all_runs(event_type)


def default_candidates() -> dict[str, EvalConfig]:
    return {
        config.name: config
        for config in (
            metered_native_config(),
            glm_subscription_config(),
            minimax_metered_config(),
        )
    }


def build_coordinator(
    tmp_path: Path,
    costs: CostStore,
    *,
    config: ShadowEvalConfig | None = None,
    candidates: dict[str, EvalConfig] | None = None,
    resolve_candidate: Callable[[str], EvalConfig | None] | None = None,
    runner: FakeRunner | None = None,
    clock: FakeClock | None = None,
    escalator: RecordingEscalator | None = None,
    store: RunStore | None = None,
    run_id: str = RUN_ID,
    psi_config: PsiAdmissionConfig | None = None,
) -> CoordinatorRig:
    db_path = tmp_path / 'runs.db'
    config = config or shadow_config()
    candidates = default_candidates() if candidates is None else candidates
    runner = runner or FakeRunner(tmp_path / 'eval-worktrees')
    clock = clock or FakeClock()
    escalator = escalator or RecordingEscalator()
    store = store or RunStore(db_path)
    events = EventStore(db_path, run_id)
    host = [calm_sample()]
    isolation = [True]
    deps = ShadowCoordinatorDeps(
        store=store,
        events=events,
        costs=costs,
        runner=runner,
        escalate=escalator,
        clock=clock.now,
        isolation=lambda: isolation[0],
        read_host_sample=lambda: host[0],
        psi_config=psi_config or PsiAdmissionConfig(),
        resolve_candidate=resolve_candidate or candidates.get,
    )
    coordinator = ShadowCoordinator(
        config, deps, project_id=PROJECT_ID, run_id=run_id, incumbents=INCUMBENTS,
    )
    return CoordinatorRig(
        coordinator=coordinator,
        config=config,
        store=store,
        events=events,
        costs=costs,
        runner=runner,
        clock=clock,
        escalator=escalator,
        candidates=candidates,
        host=host,
        isolation=isolation,
    )
