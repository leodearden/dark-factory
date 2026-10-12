"""Classifies an LLM invocation as scored or unmeasured and names why, as typed values.

Why each cause is unmeasurable, the classifier's precedence and the persisted
record shape: docs/plan-scoring-and-judge.md, "Unmeasured cells: typed causes
(task 6734)".
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar

from shared.invocation_outcome import (
    OK,
    AuthFailed,
    CapHit,
    ModelNotFound,
    ServerError,
    ZeroOutputWedge,
    classify_invocation,
)

if TYPE_CHECKING:
    from shared.cli_invoke import AgentResult

logger = logging.getLogger(__name__)

T = TypeVar('T')

_REASON_CHARS = 80


class MeasurementStage(StrEnum):
    ARCHITECT = 'architect'
    JUDGE = 'judge'
    PAIRWISE_JUDGE = 'pairwise_judge'
    MATCHER = 'matcher'
    ASSESSOR = 'assessor'
    SYNTHESIZER = 'synthesizer'


class UnmeasuredCause(StrEnum):
    CAP_HIT = 'cap_hit'
    AUTH_FAILED = 'auth_failed'
    MODEL_NOT_FOUND = 'model_not_found'
    API_ERROR = 'api_error'
    SERVER_ERROR = 'server_error'
    WEDGE = 'wedge'
    TIMEOUT = 'timeout'
    CAP_EXHAUSTED = 'cap_exhausted'
    HARNESS_ERROR = 'harness_error'
    RAISED = 'raised'
    UNPARSEABLE = 'unparseable'
    LEGACY_UNCLASSIFIED = 'legacy_unclassified'


class UnmeasuredRecord(TypedDict):
    """The persisted, JSON-safe form of an :class:`Unmeasured`."""

    stage: str
    cause: str
    reason: str


def _require_type(owner: str, field: str, value: object, expected: type) -> None:
    if not isinstance(value, expected):
        raise TypeError(
            f'{owner}.{field} must be a {expected.__name__}, got {value!r}'
        )


def _require_spend(owner: str, value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f'{owner}.cost_usd must be a finite, non-negative number, got {value!r}'
        )


@dataclass(frozen=True)
class Unmeasured:
    """An invocation that produced no measurement: where, why, and what it spent."""

    stage: MeasurementStage
    cause: UnmeasuredCause
    reason: str
    cost_usd: float

    def __post_init__(self) -> None:
        _require_type('Unmeasured', 'stage', self.stage, MeasurementStage)
        _require_type('Unmeasured', 'cause', self.cause, UnmeasuredCause)
        _require_type('Unmeasured', 'reason', self.reason, str)
        _require_spend('Unmeasured', self.cost_usd)

    def to_record(self) -> UnmeasuredRecord:
        return {
            'stage': self.stage.value,
            'cause': self.cause.value,
            'reason': self.reason,
        }


@dataclass(frozen=True)
class Scored(Generic[T]):
    """An invocation that produced a measurement, and what it spent."""

    value: T
    cost_usd: float

    def __post_init__(self) -> None:
        _require_spend('Scored', self.cost_usd)


def clip_reason(text: str) -> str:
    """*text* on one line with whitespace collapsed, clipped to a short reason."""
    return ' '.join(text.split())[:_REASON_CHARS]


def exception_reason(exc: BaseException) -> str:
    return f'{type(exc).__name__}: {clip_reason(str(exc))}'


def coerce_cost_usd(value: object) -> float:
    """*value* as a finite, non-negative float; ``0.0`` for anything else.

    A non-finite or negative number also logs one WARNING. Gated on
    ``isinstance`` rather than ``float()``, because a bare ``MagicMock``'s
    ``float()`` is ``1.0``.
    """
    if not isinstance(value, (int, float)):
        return 0.0
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        logger.warning(
            f'cost_usd coercion received a non-finite or negative value '
            f'({v!r}) — degraded to 0.0 rather than poisoning a downstream '
            f'sum/mean or persisting invalid JSON.'
        )
        return 0.0
    return v


def classify_unmeasured(
    result: AgentResult, *, stage: MeasurementStage, backend: str,
) -> Unmeasured | None:
    """The transport refusal behind *result*, or ``None`` if the model answered.

    Built on ``classify_invocation(strict_confirm=True)``. Precedence: CapHit,
    AuthFailed, ModelNotFound; an OK result is never a refusal; then a
    body-less HTTP 429, ServerError, ZeroOutputWedge. Every other outcome is
    ``None``.
    """
    refusal = _transport_refusal(result, backend)
    if refusal is None:
        return None
    cause, reason = refusal
    return Unmeasured(stage, cause, reason, cost_usd=coerce_cost_usd(result.cost_usd))


def _transport_refusal(
    result: AgentResult, backend: str,
) -> tuple[UnmeasuredCause, str] | None:
    outcome = classify_invocation(result, strict_confirm=True, backend=backend)
    if isinstance(outcome, CapHit):
        return UnmeasuredCause.CAP_HIT, clip_reason(outcome.reason)
    if isinstance(outcome, AuthFailed):
        return UnmeasuredCause.AUTH_FAILED, f'HTTP {outcome.status}'
    if isinstance(outcome, ModelNotFound):
        return UnmeasuredCause.MODEL_NOT_FOUND, clip_reason(outcome.reason)
    if isinstance(outcome, OK):
        return None
    if result.api_error_status == 429:
        return UnmeasuredCause.API_ERROR, 'HTTP 429'
    if isinstance(outcome, ServerError):
        return UnmeasuredCause.SERVER_ERROR, f'HTTP {outcome.status}'
    if isinstance(outcome, ZeroOutputWedge):
        return UnmeasuredCause.WEDGE, 'zero-output timeout (no transcript turns)'
    return None
