"""The live shadow-eval coordinator: which cells open, how they run, settle and pause.

Each orchestrator process owns one coordinator. It turns production's phase
events into shadow cells, drives each cell's run, settles the cell against
production's own outcome, and pauses itself on a storm. The pure policy
(sampling, the open-gate walk, the storm guards) is module functions over
typed snapshots; the coordinator only gathers those snapshots and records
their consequences. Design: plans/live-shadow-eval-prd.md C2.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol

from shared.cost_store import CostStore
from shared.psi import PsiSample

from orchestrator.config import PsiAdmissionConfig, ShadowEvalConfig
from orchestrator.evals.configs import CANDIDATE_ENDPOINT_PRICES, EvalConfig, get_config_by_name
from orchestrator.evals.live_fixture import live_stratum
from orchestrator.evals.metrics import EvalMetrics
from orchestrator.evals.task_sampler import repo_of_project
from orchestrator.event_store import EventStore, EventType, ShadowSkipReason
from orchestrator.run_store import (
    CellChange,
    RunStore,
    ShadowCell,
    ShadowCellQuery,
    ShadowStormRecord,
)
from orchestrator.shadow_scoring import ShadowReference, consequence_leg_warranted
from orchestrator.shadow_types import (
    CostBasis,
    PlanSource,
    ShadowCellState,
    ShadowShape,
    ShadowStormCause,
)
from orchestrator.workflow_types import WorkflowOutcome, WorkflowState

logger = logging.getLogger(__name__)

__all__ = [
    'CREDIT_WINDOW',
    'EXPIRE_BATCH_LIMIT',
    'NOT_LANDED_TERMINALS',
    'PRODUCTION_COST_BASIS',
    'SHAPE_TRIGGERS',
    'TRAILING_WINDOW',
    'CellEndReason',
    'CellRequest',
    'CellRunOutcome',
    'CostPair',
    'CostRatioVerdict',
    'ExpireSweep',
    'HostVerdict',
    'MergeFinalized',
    'OpenDecision',
    'OpenInputs',
    'RatioSkip',
    'RunEnd',
    'RunEndedCell',
    'ShadowCoordinator',
    'ShadowCoordinatorDeps',
    'ShadowRunner',
    'StormFacts',
    'StormState',
    'StormTrip',
    'budget_exhausted',
    'candidate_has_credentials',
    'cost_basis_of',
    'cost_ratio_verdict',
    'decide_open',
    'evaluate_storm',
    'failure_streak_tripped',
    'host_verdict',
    'run_end_change',
    'runner_raised_reason',
    'sample_point',
    'sampled_in',
    'shape_enabled',
    'shapes_triggered_by',
    'storm_window_start',
]


# --------------------------------------------------------------------------- #
# Deterministic sampling (PRD decision 8)                                      #
# --------------------------------------------------------------------------- #


def sample_point(
    task_id: str,
    candidate: str,
    shape: ShadowShape,
    stratum: tuple[str, str, str],
    seed: int,
) -> float:
    """The cell's fixed point in ``[0, 1)``, identical in every process.

    sha256 rather than ``hash()``, which is salted per process, so both
    projects' coordinators cut the same set.
    """
    canonical = json.dumps(
        [task_id, candidate, shape.value, list(stratum), seed], separators=(',', ':'),
    )
    digest = hashlib.sha256(canonical.encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64


def sampled_in(point: float, rate: float) -> bool:
    """Threshold sampling: raising *rate* only ever adds cells to the cut."""
    return point < rate


# --------------------------------------------------------------------------- #
# Host load, credentials and cost basis                                        #
# --------------------------------------------------------------------------- #


class HostVerdict(StrEnum):
    """What the host-load gate read; ``unreadable`` is the explicit fail-open arm."""

    CLEAR = 'clear'
    SATURATED = 'saturated'
    UNREADABLE = 'unreadable'


def host_verdict(sample: PsiSample, cfg: PsiAdmissionConfig) -> HostVerdict:
    """Mirror ``scheduler.py::Scheduler._phase_psi_gate`` for an optional cell.

    A disabled gate is clear; an unreadable sample never holds but is named,
    so a cell opened blind is told apart from one opened on a calm host. No
    in-flight floor applies: a cell is optional work.
    """
    if not cfg.enabled:
        return HostVerdict.CLEAR
    if not sample.read_ok:
        return HostVerdict.UNREADABLE
    if sample.saturated(cfg):
        return HostVerdict.SATURATED
    return HostVerdict.CLEAR


_CREDENTIAL_ENV_KEYS = ('ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_API_KEY')


def candidate_has_credentials(config: EvalConfig) -> bool:
    """False when the candidate names a credential key but it is blank."""
    return not any(
        key in config.env_overrides and not config.env_overrides[key].strip()
        for key in _CREDENTIAL_ENV_KEYS
    )


def cost_basis_of(config: EvalConfig) -> CostBasis:
    price = CANDIDATE_ENDPOINT_PRICES.get(config.model)
    return CostBasis.METERED if price is None else price.cost_basis


# --------------------------------------------------------------------------- #
# Triggers and the open decision                                               #
# --------------------------------------------------------------------------- #

# The consequence leg 2 is never phase-triggered: it opens from leg 1's run.
SHAPE_TRIGGERS: Mapping[tuple[WorkflowState, bool], frozenset[ShadowShape]] = MappingProxyType({
    (WorkflowState.PLAN, True): frozenset({ShadowShape.ARCHITECT, ShadowShape.END_TO_END}),
    (WorkflowState.PLAN, False): frozenset({ShadowShape.IMPLEMENTER}),
})


def shapes_triggered_by(phase: WorkflowState, entering: bool) -> frozenset[ShadowShape]:
    return SHAPE_TRIGGERS.get((phase, entering), frozenset())


def shape_enabled(shape: ShadowShape, enabled_shapes: frozenset[ShadowShape]) -> bool:
    """Whether *shape* is on; the architect cell is also consequence leg 1."""
    if shape is ShadowShape.ARCHITECT:
        return bool(enabled_shapes & {ShadowShape.ARCHITECT, ShadowShape.ARCHITECT_CONSEQUENCE})
    return shape in enabled_shapes


def budget_exhausted(spent_usd: float, cap_usd: float) -> bool:
    """The daily USD cap rule, shared by the open gate and the storm guard."""
    return spent_usd >= cap_usd


@dataclass(frozen=True)
class OpenInputs:
    """Everything one open decision reads, gathered as a snapshot."""

    task_id: str
    shape: ShadowShape
    candidate: str
    stratum: tuple[str, str, str]
    base_sha: str | None
    seed: int
    sample_rate: float
    enabled: bool
    enabled_shapes: frozenset[ShadowShape]
    end_to_end_candidates: frozenset[str]
    paused: bool
    already_open: bool
    has_credentials: bool
    isolation_available: bool
    credits_exhausted_recently: bool
    spent_usd: float
    daily_budget_usd: float
    held_cells: int
    max_concurrent: int
    host: HostVerdict


@dataclass(frozen=True)
class OpenDecision:
    """Open, or a skip with its reason (INV-11); ``host`` records what the load gate read."""

    open: bool
    reason: ShadowSkipReason | None
    host: HostVerdict

    def __post_init__(self) -> None:
        if self.open is not (self.reason is None):
            raise ValueError(
                f'OpenDecision: an open decision has no reason and a skip names one '
                f'(got open={self.open!r}, reason={self.reason!r})'
            )


def _shape_disabled(inputs: OpenInputs) -> bool:
    return (
        not inputs.enabled
        or not shape_enabled(inputs.shape, inputs.enabled_shapes)
        or (
            inputs.shape is ShadowShape.END_TO_END
            and inputs.candidate not in inputs.end_to_end_candidates
        )
    )


def _sampled_out(inputs: OpenInputs) -> bool:
    point = sample_point(
        inputs.task_id, inputs.candidate, inputs.shape, inputs.stratum, inputs.seed,
    )
    return not sampled_in(point, inputs.sample_rate)


# Walked in order: the first refusing gate is the reason. Sampling precedes
# every resource gate, so a sampled-out cell never reports a load reason.
_OPEN_GATES: tuple[tuple[ShadowSkipReason, Callable[[OpenInputs], bool]], ...] = (
    (ShadowSkipReason.SHAPE_DISABLED, _shape_disabled),
    (ShadowSkipReason.PAUSED, lambda i: i.paused),
    (ShadowSkipReason.ALREADY_OPEN, lambda i: i.already_open),
    (ShadowSkipReason.SAMPLED_OUT, _sampled_out),
    (ShadowSkipReason.NO_BASE_SHA, lambda i: not (i.base_sha or '').strip()),
    (ShadowSkipReason.NO_CREDENTIALS, lambda i: not i.has_credentials),
    (ShadowSkipReason.ISOLATION_UNAVAILABLE, lambda i: not i.isolation_available),
    (ShadowSkipReason.CREDITS_EXHAUSTED, lambda i: i.credits_exhausted_recently),
    (ShadowSkipReason.BUDGET_EXHAUSTED, lambda i: budget_exhausted(i.spent_usd, i.daily_budget_usd)),
    (ShadowSkipReason.CONCURRENCY_FULL, lambda i: i.held_cells >= i.max_concurrent),
    (ShadowSkipReason.HOST_SATURATED, lambda i: i.host is HostVerdict.SATURATED),
)


def decide_open(inputs: OpenInputs) -> OpenDecision:
    reason = next((reason for reason, refuses in _OPEN_GATES if refuses(inputs)), None)
    return OpenDecision(open=reason is None, reason=reason, host=inputs.host)


# --------------------------------------------------------------------------- #
# Storm escape policy (PRD decision 10, INV-4)                                 #
# --------------------------------------------------------------------------- #

# The one window shared by the daily budget, the cost ratio and the open gate's budget.
TRAILING_WINDOW = timedelta(hours=24)

# PRD decision 16: production's invocations ledger meters token USD.
PRODUCTION_COST_BASIS = CostBasis.METERED

_STREAK_STATES = frozenset({ShadowCellState.FAILED, ShadowCellState.EXPIRED})


def storm_window_start(now: datetime, resumed_at: datetime | None) -> datetime:
    """A resume restarts every storm window, so old failures cannot re-trip at once."""
    trailing = now - TRAILING_WINDOW
    return trailing if resumed_at is None else max(trailing, resumed_at)


def failure_streak_tripped(states_newest_first: Sequence[ShadowCellState], threshold: int) -> bool:
    """Whether the newest *threshold* run-ended cells all failed or expired."""
    newest = states_newest_first[:threshold]
    return len(newest) >= threshold and all(state in _STREAK_STATES for state in newest)


class RatioSkip(StrEnum):
    """Why no cost ratio was measured: a named skip, never a ratio of 0 (INV-11)."""

    COST_BASIS_MISMATCH = 'cost_basis_mismatch'
    NO_PAIRS = 'no_pairs'
    NO_PRODUCTION_COST = 'no_production_cost'


@dataclass(frozen=True)
class CostPair:
    """Settled cells' summed cost beside production's summed cost for the same tasks."""

    cell_cost_usd: float
    cell_basis: CostBasis
    production_cost_usd: float
    production_basis: CostBasis


@dataclass(frozen=True)
class CostRatioVerdict:
    """A measured cell/production cost ratio, or the named reason none was measured."""

    ratio: float | None
    skipped: RatioSkip | None
    ceiling: float

    def __post_init__(self) -> None:
        if (self.ratio is None) is (self.skipped is None):
            raise ValueError(
                f'CostRatioVerdict: exactly one of ratio and skipped is set '
                f'(got ratio={self.ratio!r}, skipped={self.skipped!r})'
            )

    @property
    def tripped(self) -> bool:
        return self.ratio is not None and self.ratio > self.ceiling


def cost_ratio_verdict(pairs: Sequence[CostPair], ceiling: float) -> CostRatioVerdict:
    """Compare only same-basis pairs: across bases the guard could never trip (decision 16)."""
    if not pairs:
        return CostRatioVerdict(None, RatioSkip.NO_PAIRS, ceiling)
    matching = [pair for pair in pairs if pair.cell_basis is pair.production_basis]
    if not matching:
        return CostRatioVerdict(None, RatioSkip.COST_BASIS_MISMATCH, ceiling)
    production = sum(pair.production_cost_usd for pair in matching)
    if production <= 0:
        return CostRatioVerdict(None, RatioSkip.NO_PRODUCTION_COST, ceiling)
    cells = sum(pair.cell_cost_usd for pair in matching)
    return CostRatioVerdict(cells / production, None, ceiling)


@dataclass(frozen=True)
class RunEndedCell:
    """A cell whose run has ended, as the failure streak reads it."""

    cell_id: str
    state: ShadowCellState


@dataclass(frozen=True)
class StormFacts:
    """What the storm guards read, gathered over the current storm window."""

    spent_usd: float
    daily_budget_usd: float
    run_ends_newest_first: tuple[RunEndedCell, ...]
    failure_streak_pause: int
    ratios: Mapping[str, CostRatioVerdict]


@dataclass(frozen=True)
class StormTrip:
    """A tripped storm guard, with the numbers that tripped it."""

    cause: ShadowStormCause
    detail: str


@dataclass(frozen=True)
class StormState:
    """Whether shadow evaluation is paused, and by which guard."""

    paused: bool
    cause: ShadowStormCause | None
    detail: str | None

    def __post_init__(self) -> None:
        if self.paused is (self.cause is None):
            raise ValueError(
                f'StormState: a pause names its cause and only a pause does '
                f'(got paused={self.paused!r}, cause={self.cause!r})'
            )

    @classmethod
    def of(cls, record: ShadowStormRecord) -> StormState:
        return cls(paused=record.paused, cause=record.cause, detail=record.detail)


_UNPAUSED = StormState(paused=False, cause=None, detail=None)


def _budget_trip(facts: StormFacts) -> str | None:
    if not budget_exhausted(facts.spent_usd, facts.daily_budget_usd):
        return None
    return (
        f'metered shadow spend {facts.spent_usd:.2f} USD reached the daily cap '
        f'{facts.daily_budget_usd:.2f} USD'
    )


def _streak_trip(facts: StormFacts) -> str | None:
    streak = facts.run_ends_newest_first[:facts.failure_streak_pause]
    if not failure_streak_tripped([cell.state for cell in streak], facts.failure_streak_pause):
        return None
    cells = ', '.join(f'{cell.cell_id} ({cell.state.value})' for cell in streak)
    return f'{len(streak)} consecutive failed or expired cells: {cells}'


def _ratio_trip(facts: StormFacts) -> str | None:
    tripped = sorted(
        (candidate, verdict) for candidate, verdict in facts.ratios.items() if verdict.tripped
    )
    if not tripped:
        return None
    candidate, verdict = tripped[0]
    return (
        f'candidate {candidate!r} cell/production cost ratio {verdict.ratio:.2f} '
        f'exceeds the ceiling {verdict.ceiling:.2f}'
    )


_STORM_GUARDS: tuple[tuple[ShadowStormCause, Callable[[StormFacts], str | None]], ...] = (
    (ShadowStormCause.DAILY_BUDGET, _budget_trip),
    (ShadowStormCause.FAILURE_STREAK, _streak_trip),
    (ShadowStormCause.COST_RATIO, _ratio_trip),
)


def evaluate_storm(facts: StormFacts) -> StormTrip | None:
    for cause, trip_detail in _STORM_GUARDS:
        detail = trip_detail(facts)
        if detail is not None:
            return StormTrip(cause, detail)
    return None


# --------------------------------------------------------------------------- #
# Typed boundary values of a cell's run                                        #
# --------------------------------------------------------------------------- #

# The Coding Plan's rolling allowance window (PRD decision 16).
CREDIT_WINDOW = timedelta(hours=5)


class RunEnd(StrEnum):
    """How a cell's run ended, as its runner reports it."""

    COMPLETED = 'completed'
    USAGE_CAP = 'usage_cap'
    CREDITS_EXHAUSTED = 'credits_exhausted'


class CellEndReason(StrEnum):
    """The ``shadow_cells.reason`` of a cell that ended without settling (C1)."""

    SETTLE_DEADLINE = 'settle_deadline'
    ORPHANED_BY_RESTART = 'orphaned_by_restart'
    USAGE_CAP = 'usage_cap'
    CREDITS_EXHAUSTED = ShadowSkipReason.CREDITS_EXHAUSTED.value
    RUNNER_RAISED = 'runner_raised'
    CANCELLED_AT_SHUTDOWN = 'cancelled_at_shutdown'


def runner_raised_reason(exc: BaseException) -> str:
    """The one builder of a failed cell's exception-class reason."""
    return f'{CellEndReason.RUNNER_RAISED.value}:{type(exc).__name__}'


@dataclass(frozen=True)
class CellRunOutcome:
    """What a runner reports when a cell's run ends; a completed run has its result."""

    end: RunEnd
    metrics: EvalMetrics | None = None
    result_path: Path | None = None
    cost_usd: float = 0.0
    cost_credits: float = 0.0
    plan: dict | None = None

    def __post_init__(self) -> None:
        if self.end is RunEnd.COMPLETED and (self.metrics is None or self.result_path is None):
            raise ValueError(
                f'CellRunOutcome: a completed run carries its metrics and result_path '
                f'(got metrics={self.metrics!r}, result_path={self.result_path!r})'
            )


@dataclass(frozen=True)
class CellRequest:
    """Everything a runner needs for one cell.

    ``task`` is the production task record the cell shadows. ``plan`` is the
    candidate plan a consequence leg 2 executes and ``None`` for every other
    shape; an implementer cell's runner reads production's own plan.
    """

    cell: ShadowCell
    task: dict
    candidate: EvalConfig
    plan: dict | None = None


class ShadowRunner(Protocol):
    """Two stages, so every C1 state is real: a prepared worktree, then a run in it."""

    async def prepare(self, request: CellRequest) -> Path: ...

    async def run(self, request: CellRequest, worktree: Path) -> CellRunOutcome: ...


_CAP_REASONS: Mapping[RunEnd, CellEndReason] = MappingProxyType({
    RunEnd.USAGE_CAP: CellEndReason.USAGE_CAP,
    RunEnd.CREDITS_EXHAUSTED: CellEndReason.CREDITS_EXHAUSTED,
})


def run_end_change(outcome: CellRunOutcome, at: datetime) -> CellChange:
    """The one mapping from a run's outcome to the running cell's successor."""
    if outcome.end is RunEnd.COMPLETED:
        return CellChange(
            ShadowCellState.AWAITING_REFERENCE,
            at,
            result_path=outcome.result_path,
            cost_usd=outcome.cost_usd,
            cost_credits=outcome.cost_credits,
        )
    return CellChange(
        ShadowCellState.CAP_EXCLUDED,
        at,
        reason=_CAP_REASONS[outcome.end].value,
        cost_usd=outcome.cost_usd,
        cost_credits=outcome.cost_credits,
    )


# --------------------------------------------------------------------------- #
# Production's outcome, as a settle reads it                                   #
# --------------------------------------------------------------------------- #

# Only these advance main with a merge_sha. 'superseded' is replaced by a later
# request and 'conflict'/'blocked' may be retried, so none of them is final.
_LANDING_STATES = frozenset({'done', 'done_wip_recovery'})

# The production terminals that mean the task will not land.
NOT_LANDED_TERMINALS = frozenset({WorkflowOutcome.BLOCKED, WorkflowOutcome.CANCELLED})


def _required_text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'merge_finalized payload: {name} must be a non-blank str (got {value!r})')
    return value


@dataclass(frozen=True)
class MergeFinalized:
    """The fields of a ``merge_finalized`` payload that a settle reads, parsed once."""

    task_id: str
    state: str
    merge_sha: str | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MergeFinalized:
        sha = payload.get('merge_sha')
        return cls(
            task_id=_required_text(payload, 'task_id'),
            state=_required_text(payload, 'state'),
            merge_sha=sha if isinstance(sha, str) and sha.strip() else None,
        )

    def reference(self) -> ShadowReference | None:
        """The landed reference, or ``None`` when this outcome landed nothing."""
        if self.state in _LANDING_STATES and self.merge_sha is not None:
            return ShadowReference.landed(self.merge_sha)
        return None


# Awaiting cells accumulate for up to settle_deadline_hours, so one sweep's
# writes are capped here rather than by anything upstream (INV-8).
EXPIRE_BATCH_LIMIT = 50


@dataclass(frozen=True)
class ExpireSweep:
    """One bounded expiry sweep: what it expired, and what it left for the next (INV-11)."""

    expired: tuple[str, ...]
    deferred: tuple[str, ...]


# --------------------------------------------------------------------------- #
# The coordinator                                                              #
# --------------------------------------------------------------------------- #

_HELD_STATES = frozenset(state for state in ShadowCellState if state.held_by_run)

# The runs a failure streak reads. A cap-excluded run says nothing about the
# candidate, so it neither extends a streak nor breaks one.
_RUN_ENDED_STATES = frozenset({
    ShadowCellState.AWAITING_REFERENCE,
    ShadowCellState.SETTLED,
    ShadowCellState.FAILED,
    ShadowCellState.EXPIRED,
})

_PLAN_SOURCE_BY_SHAPE: Mapping[ShadowShape, PlanSource] = MappingProxyType({
    ShadowShape.IMPLEMENTER: PlanSource.PRODUCTION,
    ShadowShape.ARCHITECT_CONSEQUENCE: PlanSource.CANDIDATE,
})


def metered_spend(cells: Sequence[ShadowCell]) -> float:
    """Real dollars only: a subscription arm's imputed ``cost_usd`` is not spend."""
    return sum(cell.cost_usd for cell in cells if cell.cost_basis is CostBasis.METERED)


def _task_id_of(task: Mapping[str, Any]) -> str:
    task_id = str(task['id']).strip()
    if not task_id:
        raise ValueError(f'shadow coordinator: task record has a blank id ({task!r})')
    return task_id


def _base_sha_of(task: Mapping[str, Any]) -> str | None:
    sha = (task.get('metadata') or {}).get('branch_base_sha')
    return sha.strip() if isinstance(sha, str) and sha.strip() else None


@dataclass(frozen=True)
class _StormSnapshot:
    """The store's view a storm evaluation reads: the pause row and the window's cells."""

    storm: ShadowStormRecord
    window: tuple[ShadowCell, ...]
    run_ends_newest_first: tuple[ShadowCell, ...]


@dataclass(frozen=True)
class ShadowCoordinatorDeps:
    """Every effect boundary of the coordinator.

    ``isolation`` has no default: until the isolation probe lands, a default
    would claim an isolation nobody checked.
    """

    store: RunStore
    events: EventStore
    costs: CostStore
    runner: ShadowRunner
    escalate: Callable[[StormTrip], Awaitable[None]]
    clock: Callable[[], datetime]
    isolation: Callable[[], bool]
    read_host_sample: Callable[[], PsiSample]
    psi_config: PsiAdmissionConfig
    resolve_candidate: Callable[[str], EvalConfig | None] = get_config_by_name


class ShadowCoordinator:
    """Opens, drives, settles and pauses one project's shadow cells.

    Holds no cell state in memory: every question is asked of the store, and
    each cell's run is a background task the coordinator owns until it ends.
    """

    def __init__(
        self,
        config: ShadowEvalConfig,
        deps: ShadowCoordinatorDeps,
        *,
        project_id: str,
        run_id: str,
        incumbents: Mapping[ShadowShape, str],
    ) -> None:
        missing = [shape.value for shape in ShadowShape if shape not in incumbents]
        if missing:
            raise ValueError(f'ShadowCoordinator: no incumbent named for shape(s) {missing}')
        repo_of_project(project_id)
        self._config = config
        self._deps = deps
        self.project_id = project_id
        self.run_id = run_id
        self._incumbents = MappingProxyType(dict(incumbents))
        self._drives: set[asyncio.Task[None]] = set()
        self._storm = _UNPAUSED

    # -- open ---------------------------------------------------------------

    async def should_open(
        self, task: dict, shape: ShadowShape, candidate: EvalConfig,
    ) -> OpenDecision:
        """Whether a cell for this triple would open now; writes nothing."""
        return decide_open(await asyncio.to_thread(self._open_inputs, task, shape, candidate))

    async def on_phase(self, task: dict, phase: WorkflowState, entering: bool) -> list[str]:
        """The harness hook for every phase enter/exit: the cell ids it opened. Never raises."""
        if not self._config.enabled:
            return []
        try:
            return await self._open_triggered(task, phase, entering)
        except Exception:
            logger.exception('shadow coordinator: on_phase failed for task %r', task.get('id'))
            return []

    async def _open_triggered(
        self, task: dict, phase: WorkflowState, entering: bool,
    ) -> list[str]:
        task_id = _task_id_of(task)
        enabled = frozenset(self._config.shapes)
        opened: list[str] = []
        for shape in ShadowShape:
            if shape in shapes_triggered_by(phase, entering) and shape_enabled(shape, enabled):
                for name in self._candidates_for(shape):
                    cell_id = await self._try_open(task, task_id, shape, name)
                    if cell_id is not None:
                        opened.append(cell_id)
        return opened

    def _candidates_for(self, shape: ShadowShape) -> list[str]:
        if shape is ShadowShape.END_TO_END:
            return list(self._config.end_to_end_candidates)
        return list(self._config.candidates)

    async def _try_open(
        self, task: dict, task_id: str, shape: ShadowShape, name: str,
    ) -> str | None:
        """One (shape, candidate) attempt; a failure here never stops the others."""
        try:
            candidate = await self._resolve(task_id, shape, name)
            if candidate is None:
                return None
            return await self._open_cell(task, shape, candidate)
        except Exception:
            logger.exception(
                'shadow coordinator: opening %s/%s for task %s failed', shape.value, name, task_id,
            )
            return None

    async def _resolve(self, task_id: str, shape: ShadowShape, name: str) -> EvalConfig | None:
        """The named candidate, or ``None`` after an ``unknown_candidate`` skip.

        A raising resolver is logged at ERROR and its exception class rides on
        the skip, so a broken resolver is told apart from a misnamed candidate.
        """
        try:
            candidate = self._deps.resolve_candidate(name)
        except Exception as exc:
            logger.exception('shadow coordinator: resolving candidate %r raised', name)
            await self._emit_skip(
                task_id, shape, name, ShadowSkipReason.UNKNOWN_CANDIDATE,
                resolver_error=type(exc).__name__,
            )
            return None
        if candidate is None:
            await self._emit_skip(task_id, shape, name, ShadowSkipReason.UNKNOWN_CANDIDATE)
        return candidate

    async def _open_cell(
        self,
        task: dict,
        shape: ShadowShape,
        candidate: EvalConfig,
        *,
        parent_cell_id: str | None = None,
        plan: dict | None = None,
    ) -> str | None:
        """Decide, then insert, announce and drive the cell; the one open path."""
        inputs = await asyncio.to_thread(self._open_inputs, task, shape, candidate)
        decision = decide_open(inputs)
        if decision.reason is not None:
            await self._emit_skip(inputs.task_id, shape, candidate.name, decision.reason)
            return None
        cell = self._sampled_cell(inputs, candidate, parent_cell_id)
        try:
            await asyncio.to_thread(self._deps.store.insert_shadow_cell, cell)
        except sqlite3.IntegrityError:
            await self._emit_skip(inputs.task_id, shape, candidate.name, ShadowSkipReason.ALREADY_OPEN)
            return None
        await self._emit(EventType.shadow_cell_opened, cell, host=decision.host.value)
        self._spawn(CellRequest(cell=cell, task=task, candidate=candidate, plan=plan))
        return cell.cell_id

    def _open_inputs(self, task: dict, shape: ShadowShape, candidate: EvalConfig) -> OpenInputs:
        config, store, now = self._config, self._deps.store, self._deps.clock()
        task_id = _task_id_of(task)
        storm = store.get_shadow_storm(self.project_id)
        since = storm_window_start(now, storm.resumed_at)
        return OpenInputs(
            task_id=task_id,
            shape=shape,
            candidate=candidate.name,
            stratum=live_stratum(task, project=self.project_id),
            base_sha=_base_sha_of(task),
            seed=config.seed,
            sample_rate=config.sample_rate_for(candidate.name),
            enabled=config.enabled,
            enabled_shapes=frozenset(config.shapes),
            end_to_end_candidates=frozenset(config.end_to_end_candidates),
            paused=storm.paused,
            already_open=bool(store.find_shadow_cells(
                self._query(task_id=task_id, shape=shape, candidate=candidate.name, limit=1),
            )),
            has_credentials=candidate_has_credentials(candidate),
            isolation_available=self._deps.isolation(),
            credits_exhausted_recently=self._credits_exhausted_since(
                candidate.name, now - CREDIT_WINDOW,
            ),
            spent_usd=metered_spend(store.find_shadow_cells(self._query(opened_since=since))),
            daily_budget_usd=config.daily_budget_usd,
            held_cells=len(store.find_shadow_cells(self._query(states=_HELD_STATES))),
            max_concurrent=config.max_concurrent,
            host=self._host(),
        )

    def _query(self, **filters: Any) -> ShadowCellQuery:
        return ShadowCellQuery(project_id=self.project_id, **filters)

    def _credits_exhausted_since(self, candidate: str, since: datetime) -> bool:
        """Whether a cell of *candidate* opened since *since* ended with the allowance spent."""
        capped = self._deps.store.find_shadow_cells(self._query(
            states=frozenset({ShadowCellState.CAP_EXCLUDED}),
            candidate=candidate,
            opened_since=since,
        ))
        return any(cell.reason == CellEndReason.CREDITS_EXHAUSTED.value for cell in capped)

    def _host(self) -> HostVerdict:
        cfg = self._deps.psi_config
        if not cfg.enabled:
            return HostVerdict.CLEAR
        return host_verdict(self._deps.read_host_sample(), cfg)

    def _sampled_cell(
        self, inputs: OpenInputs, candidate: EvalConfig, parent_cell_id: str | None,
    ) -> ShadowCell:
        now = self._deps.clock()
        assert inputs.base_sha is not None  # decide_open opened only with a base sha
        return ShadowCell(
            cell_id=uuid.uuid4().hex,
            parent_cell_id=parent_cell_id,
            project_id=self.project_id,
            task_id=inputs.task_id,
            shape=inputs.shape,
            candidate=candidate.name,
            incumbent=self._incumbents[inputs.shape],
            stratum=inputs.stratum,
            base_sha=inputs.base_sha,
            plan_source=_PLAN_SOURCE_BY_SHAPE.get(inputs.shape),
            state=ShadowCellState.SAMPLED,
            cost_basis=cost_basis_of(candidate),
            run_id=self.run_id,
            opened_at=now,
            settle_deadline=now + timedelta(hours=self._config.settle_deadline_hours),
        )

    # -- drive --------------------------------------------------------------

    def _spawn(self, request: CellRequest) -> None:
        drive = asyncio.create_task(
            self._drive(request), name=f'shadow-cell-{request.cell.cell_id}',
        )
        self._drives.add(drive)
        drive.add_done_callback(self._drives.discard)

    async def _drive(self, request: CellRequest) -> None:
        """Run one cell through its lifecycle; every exit writes the cell's successor."""
        cell_id = request.cell.cell_id
        try:
            await self._run_cell(request)
        except asyncio.CancelledError:
            await self._fail_held(cell_id, CellEndReason.CANCELLED_AT_SHUTDOWN.value)
            raise
        except Exception as exc:
            logger.warning('shadow cell %s: runner raised', cell_id, exc_info=True)
            await self._fail_held(cell_id, runner_raised_reason(exc))
        await self._evaluate_storm()

    async def _run_cell(self, request: CellRequest) -> None:
        runner, cell_id = self._deps.runner, request.cell.cell_id
        worktree = await runner.prepare(request)
        opened = await self._advance(
            cell_id, ShadowCellState.SAMPLED,
            CellChange(ShadowCellState.OPENED, self._deps.clock(), worktree_path=worktree),
        )
        if opened is None:
            return
        running = await self._advance(
            cell_id, ShadowCellState.OPENED, CellChange(ShadowCellState.RUNNING, self._deps.clock()),
        )
        if running is None:
            return
        outcome = await runner.run(request, worktree)
        ended = await self._advance(
            cell_id, ShadowCellState.RUNNING, run_end_change(outcome, self._deps.clock()),
        )
        if ended is None:
            return
        if ended.state is ShadowCellState.CAP_EXCLUDED:
            await self._emit_ended(ended)
            return
        await self._catch_up(ended)
        if ended.shape is ShadowShape.ARCHITECT:
            await self._maybe_open_leg_two(request, outcome)

    async def _maybe_open_leg_two(self, leg_one: CellRequest, outcome: CellRunOutcome) -> None:
        """Open consequence leg 2 on the candidate's plan when leg 1 produced one to run."""
        if ShadowShape.ARCHITECT_CONSEQUENCE not in self._config.shapes:
            return
        if outcome.metrics is None or not consequence_leg_warranted(outcome.metrics):
            return
        try:
            await self._open_cell(
                leg_one.task,
                ShadowShape.ARCHITECT_CONSEQUENCE,
                leg_one.candidate,
                parent_cell_id=leg_one.cell.cell_id,
                plan=outcome.plan,
            )
        except Exception:
            logger.exception('shadow cell %s: opening consequence leg 2 failed', leg_one.cell.cell_id)

    async def _advance(
        self, cell_id: str, expected: ShadowCellState, change: CellChange,
    ) -> ShadowCell | None:
        cell = await asyncio.to_thread(
            self._deps.store.advance_shadow_cell, cell_id, expected=expected, change=change,
        )
        if cell is None:
            logger.warning(
                'shadow cell %s: no longer %r, so this run stops driving it',
                cell_id, expected.value,
            )
        return cell

    async def _fail_held(self, cell_id: str, reason: str) -> None:
        """Fail a cell this run still holds, from whatever held state it reached."""
        failed = await asyncio.to_thread(self._fail_held_sync, cell_id, reason)
        if failed is not None:
            await self._emit_ended(failed)

    def _fail_held_sync(self, cell_id: str, reason: str) -> ShadowCell | None:
        store = self._deps.store
        cell = store.get_shadow_cell(cell_id)
        if cell is None or not cell.state.held_by_run:
            return None
        return store.advance_shadow_cell(
            cell_id,
            expected=cell.state,
            change=CellChange(ShadowCellState.FAILED, self._deps.clock(), reason=reason),
        )

    # -- settle -------------------------------------------------------------

    async def on_merge_finalized(self, payload: Mapping[str, Any]) -> list[str]:
        """Settle the task's awaiting cells against a landing; the cell ids settled."""
        try:
            finalized = MergeFinalized.from_payload(payload)
        except ValueError:
            logger.warning('shadow coordinator: unreadable merge_finalized payload %r', payload)
            return []
        reference = finalized.reference()
        if reference is None:
            return []
        return await self._settle_task(finalized.task_id, reference)

    async def on_production_terminal(self, task_id: str, outcome: WorkflowOutcome) -> list[str]:
        """Settle the task's awaiting cells unreferenced when production will not land."""
        if outcome not in NOT_LANDED_TERMINALS:
            return []
        return await self._settle_task(task_id, ShadowReference.none())

    async def _settle_task(self, task_id: str, reference: ShadowReference) -> list[str]:
        """The one settle path of both entry points; like on_phase, it never raises."""
        try:
            awaiting = await asyncio.to_thread(
                self._deps.store.find_shadow_cells,
                self._query(task_id=task_id, states=frozenset({ShadowCellState.AWAITING_REFERENCE})),
            )
            settled: list[str] = []
            for cell in awaiting:
                if await self._settle(cell, reference):
                    settled.append(cell.cell_id)
            if settled:
                await self._evaluate_storm()
            return settled
        except Exception:
            logger.exception('shadow coordinator: settling task %s failed', task_id)
            return []

    async def _settle(self, cell: ShadowCell, reference: ShadowReference) -> bool:
        """Settle one awaiting cell against *reference*."""
        change = CellChange(
            ShadowCellState.SETTLED,
            self._deps.clock(),
            reference_kind=reference.kind,
            reference_sha=reference.sha,
        )
        return await self._close_awaiting(
            cell, change, reference_kind=reference.kind.value, reference_sha=reference.sha,
        )

    async def _close_awaiting(self, cell: ShadowCell, change: CellChange, **event: Any) -> bool:
        """Move an awaiting cell to its ``settled_at``-stamping end; False if another won it."""
        closed = await asyncio.to_thread(
            self._deps.store.advance_shadow_cell,
            cell.cell_id,
            expected=ShadowCellState.AWAITING_REFERENCE,
            change=change,
        )
        if closed is None:
            return False
        await self._emit(EventType.shadow_cell_settled, closed, **event)
        return True

    async def _catch_up(self, cell: ShadowCell) -> None:
        """Settle a just-completed cell whose production outcome is already recorded."""
        try:
            reference = await asyncio.to_thread(self._recorded_reference, cell)
            if reference is not None:
                await self._settle(cell, reference)
        except Exception:
            logger.warning(
                'shadow cell %s: late-reference catch-up failed; settle or expire still owns it',
                cell.cell_id, exc_info=True,
            )

    def _recorded_reference(self, cell: ShadowCell) -> ShadowReference | None:
        """Production's outcome for the cell's task, if recorded since the cell opened."""
        finalized = self._deps.events.latest_merge_finalized(task_id=cell.task_id, cross_run=True)
        if finalized is not None:
            reference = MergeFinalized.from_payload(finalized).reference()
            finished_at = datetime.fromisoformat(finalized['finished_at'])
            if reference is not None and finished_at >= cell.opened_at:
                return reference
        recorded = self._deps.store.get_task_outcome(self.project_id, cell.task_id)
        if (
            recorded is not None
            and recorded.outcome in NOT_LANDED_TERMINALS
            and recorded.completed_at is not None
            and recorded.completed_at >= cell.opened_at
        ):
            return ShadowReference.none()
        return None

    # -- expiry and restart -------------------------------------------------

    async def expire(self, now: datetime) -> ExpireSweep:
        """Expire the oldest awaiting cells past their deadline, at most EXPIRE_BATCH_LIMIT."""
        overdue = await asyncio.to_thread(
            self._deps.store.find_shadow_cells,
            self._query(states=frozenset({ShadowCellState.AWAITING_REFERENCE}), deadline_before=now),
        )
        batch, rest = overdue[:EXPIRE_BATCH_LIMIT], overdue[EXPIRE_BATCH_LIMIT:]
        reason = CellEndReason.SETTLE_DEADLINE.value
        change = CellChange(ShadowCellState.EXPIRED, now, reason=reason)
        expired: list[str] = []
        for cell in batch:
            if await self._close_awaiting(
                cell, change, state=ShadowCellState.EXPIRED.value, reason=reason,
            ):
                expired.append(cell.cell_id)
        if rest:
            logger.warning(
                'shadow expire: %d overdue awaiting cell(s) deferred past the batch limit of %d',
                len(rest), EXPIRE_BATCH_LIMIT,
            )
        await self._evaluate_storm()
        return ExpireSweep(expired=tuple(expired), deferred=tuple(cell.cell_id for cell in rest))

    async def reconcile_on_start(self) -> list[str]:
        """Fail every held cell whose run is not this one: its run died with its process.

        Uncapped, because every orphan must get its successor (INV-6); each
        start reconciles the previous run, which held at most max_concurrent.
        An awaiting cell is left alone: settle or expire owns it, not its run.
        """
        orphans = await asyncio.to_thread(self._fail_orphans)
        for cell in orphans:
            await self._emit_ended(cell)
        await self._evaluate_storm()
        return [cell.cell_id for cell in orphans]

    def _fail_orphans(self) -> list[ShadowCell]:
        store = self._deps.store
        change = CellChange(
            ShadowCellState.FAILED,
            self._deps.clock(),
            reason=CellEndReason.ORPHANED_BY_RESTART.value,
        )
        failed: list[ShadowCell] = []
        for cell in store.find_shadow_cells(self._query(states=_HELD_STATES)):
            if cell.run_id == self.run_id:
                continue
            orphan = store.advance_shadow_cell(cell.cell_id, expected=cell.state, change=change)
            if orphan is not None:
                failed.append(orphan)
        return failed

    # -- storm escape -------------------------------------------------------

    def storm_state(self) -> StormState:
        """The pause this coordinator last observed or wrote; advisory, never the open gate."""
        return self._storm

    async def resume(self) -> bool:
        """Lift the pause, restarting every storm window now; False when none stood."""
        resumed = await asyncio.to_thread(
            self._deps.store.resume_shadow_storm, self.project_id, self._deps.clock(),
        )
        self._storm = _UNPAUSED
        return resumed

    async def _evaluate_storm(self) -> None:
        """Pause on a tripped storm guard; run after every cell end, expire and start. Never raises."""
        try:
            await self._check_storm()
        except Exception:
            logger.exception('shadow coordinator: storm evaluation failed')

    async def _check_storm(self) -> None:
        threshold = self._config.failure_streak_pause
        snapshot = await asyncio.to_thread(self._storm_snapshot, threshold)
        if snapshot.storm.paused:
            self._storm = StormState.of(snapshot.storm)
            return
        settled = [cell for cell in snapshot.window if cell.state is ShadowCellState.SETTLED]
        facts = StormFacts(
            spent_usd=metered_spend(snapshot.window),
            daily_budget_usd=self._config.daily_budget_usd,
            run_ends_newest_first=tuple(
                RunEndedCell(cell.cell_id, cell.state) for cell in snapshot.run_ends_newest_first
            ),
            failure_streak_pause=threshold,
            ratios=await self._measured_ratios(settled),
        )
        trip = evaluate_storm(facts)
        if trip is None:
            self._storm = StormState.of(snapshot.storm)
            return
        await self._pause(trip)

    def _storm_snapshot(self, threshold: int) -> _StormSnapshot:
        store = self._deps.store
        storm = store.get_shadow_storm(self.project_id)
        if storm.paused:
            return _StormSnapshot(storm, (), ())
        since = storm_window_start(self._deps.clock(), storm.resumed_at)
        return _StormSnapshot(
            storm=storm,
            window=tuple(store.find_shadow_cells(self._query(opened_since=since))),
            run_ends_newest_first=tuple(store.find_shadow_cells(self._query(
                states=_RUN_ENDED_STATES,
                opened_since=storm.resumed_at,
                newest_first=True,
                limit=threshold,
            ))),
        )

    async def _measured_ratios(self, settled: Sequence[ShadowCell]) -> dict[str, CostRatioVerdict]:
        """The cost ratios, or none this round when production's cost is unreadable.

        An unreadable cost store must not disarm the budget and streak guards too.
        """
        try:
            return await self._cost_ratios(settled)
        except Exception:
            logger.warning(
                'shadow storm: production cost unreadable; no cost ratio is measured this round',
                exc_info=True,
            )
            return {}

    async def _cost_ratios(self, settled: Sequence[ShadowCell]) -> dict[str, CostRatioVerdict]:
        """Each candidate's settled cost beside production's: one cost read per candidate and basis."""
        groups: dict[tuple[str, CostBasis], list[ShadowCell]] = {}
        for cell in settled:
            groups.setdefault((cell.candidate, cell.cost_basis), []).append(cell)
        pairs: dict[str, list[CostPair]] = {}
        for (candidate, basis), cells in sorted(groups.items()):
            production = await self._deps.costs.tasks_invocation_cost(
                self.project_id, sorted({cell.task_id for cell in cells}),
            )
            pairs.setdefault(candidate, []).append(CostPair(
                cell_cost_usd=sum(cell.cost_usd for cell in cells),
                cell_basis=basis,
                production_cost_usd=production,
                production_basis=PRODUCTION_COST_BASIS,
            ))
        ceiling = self._config.cost_ratio_ceiling
        return {candidate: cost_ratio_verdict(found, ceiling) for candidate, found in pairs.items()}

    async def _pause(self, trip: StormTrip) -> None:
        """Write the durable pause; only the call that created it files the escalation."""
        store = self._deps.store
        created = await asyncio.to_thread(
            store.pause_shadow_storm, self.project_id, trip.cause, trip.detail, self._deps.clock(),
        )
        if not created:
            self._storm = StormState.of(
                await asyncio.to_thread(store.get_shadow_storm, self.project_id),
            )
            return
        self._storm = StormState(paused=True, cause=trip.cause, detail=trip.detail)
        logger.warning('shadow coordinator: paused on %s: %s', trip.cause.value, trip.detail)
        try:
            await self._deps.escalate(trip)
        except Exception:
            logger.exception(
                'shadow coordinator: the %s storm escalation failed; the pause stands',
                trip.cause.value,
            )

    # -- lifetime -----------------------------------------------------------

    async def join(self) -> None:
        """Wait for every live drive, including any a finishing drive spawned."""
        while self._drives:
            await asyncio.gather(*list(self._drives), return_exceptions=True)

    async def aclose(self) -> None:
        """Cancel every live drive; each writes its cell's successor before it ends."""
        for drive in list(self._drives):
            drive.cancel()
        await self.join()

    # -- events -------------------------------------------------------------

    async def _emit(self, event_type: EventType, cell: ShadowCell, **data: Any) -> None:
        payload = {
            'cell_id': cell.cell_id,
            'shape': cell.shape.value,
            'candidate': cell.candidate,
            **data,
        }
        await asyncio.to_thread(
            self._deps.events.emit, event_type, task_id=cell.task_id, data=payload,
        )

    async def _emit_ended(self, cell: ShadowCell) -> None:
        await self._emit(
            EventType.shadow_cell_ended, cell, state=cell.state.value, reason=cell.reason,
        )

    async def _emit_skip(
        self,
        task_id: str,
        shape: ShadowShape,
        candidate: str,
        reason: ShadowSkipReason,
        **detail: str,
    ) -> None:
        await asyncio.to_thread(
            self._deps.events.emit,
            EventType.shadow_cell_skipped,
            task_id=task_id,
            data={'reason': reason.value, 'shape': shape.value, 'candidate': candidate, **detail},
        )

