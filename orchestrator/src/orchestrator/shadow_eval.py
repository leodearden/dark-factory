"""The live shadow-eval coordinator: which cells open, how they run, settle and pause.

Each orchestrator process owns one coordinator. It turns production's phase
events into shadow cells, drives each cell's run, settles the cell against
production's own outcome, and pauses itself on a storm. The pure policy
(sampling, the open-gate walk, the storm guards) is module functions over
typed snapshots; the coordinator only gathers those snapshots and records
their consequences. Design: plans/live-shadow-eval-prd.md C2.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType

from shared.psi import PsiSample

from orchestrator.config import PsiAdmissionConfig
from orchestrator.evals.configs import CANDIDATE_ENDPOINT_PRICES, EvalConfig
from orchestrator.event_store import ShadowSkipReason
from orchestrator.shadow_types import CostBasis, ShadowCellState, ShadowShape, ShadowStormCause
from orchestrator.workflow_types import WorkflowState

__all__ = [
    'PRODUCTION_COST_BASIS',
    'SHAPE_TRIGGERS',
    'TRAILING_WINDOW',
    'CostPair',
    'CostRatioVerdict',
    'HostVerdict',
    'OpenDecision',
    'OpenInputs',
    'RatioSkip',
    'RunEndedCell',
    'StormFacts',
    'StormTrip',
    'budget_exhausted',
    'candidate_has_credentials',
    'cost_basis_of',
    'cost_ratio_verdict',
    'decide_open',
    'evaluate_storm',
    'failure_streak_tripped',
    'host_verdict',
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
    """One settled cell's cost beside production's cost for the same task."""

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
