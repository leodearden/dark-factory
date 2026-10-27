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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from shared.psi import PsiSample

from orchestrator.config import PsiAdmissionConfig
from orchestrator.evals.configs import CANDIDATE_ENDPOINT_PRICES, EvalConfig
from orchestrator.event_store import ShadowSkipReason
from orchestrator.shadow_types import CostBasis, ShadowShape
from orchestrator.workflow_types import WorkflowState

__all__ = [
    'SHAPE_TRIGGERS',
    'HostVerdict',
    'OpenDecision',
    'OpenInputs',
    'budget_exhausted',
    'candidate_has_credentials',
    'cost_basis_of',
    'decide_open',
    'host_verdict',
    'sample_point',
    'sampled_in',
    'shape_enabled',
    'shapes_triggered_by',
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
