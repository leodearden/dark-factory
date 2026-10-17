"""Classifies an LLM invocation as scored or unmeasured and names why, as typed values.

Why each cause is unmeasurable, the classifier's precedence and the persisted
record shape: docs/plan-scoring-and-judge.md, "Unmeasured cells: typed causes
(task 6734)".
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

from shared.cli_invoke import AllAccountsCappedException, invoke_with_cap_retry
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
    from shared.usage_gate import UsageGate

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


def require_answer_iff_measured(
    owner: str, field: str, answer: object, unmeasured: Unmeasured | None,
) -> None:
    """A verdict either answered or says why it did not — never both, never neither."""
    if (answer is None) == (unmeasured is None):
        raise ValueError(
            f'{owner}.{field} must be None exactly when unmeasured is set; '
            f'got {field}={answer!r}, unmeasured={unmeasured!r}'
        )


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
    return Unmeasured(
        stage, cause, reason, cost_usd=coerce_cost_usd(result.cost_usd),
    )


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


def json_answer(result: AgentResult) -> dict[str, Any]:
    """The JSON object *result* answered: ``structured_output``, else its output text.

    Raises ``ValueError`` (``json.JSONDecodeError`` included) when there is none,
    or ``TypeError`` when the output text is not a string.
    """
    payload = result.structured_output or json.loads(result.output)
    if not isinstance(payload, dict):
        raise ValueError(f'answer is not a JSON object but {type(payload).__name__}')
    return payload


async def measured_call(
    invoke: Callable[..., Awaitable[AgentResult]],
    parse: Callable[[AgentResult], T],
    *,
    stage: MeasurementStage,
    backend: str,
    usage_gate: UsageGate | None = None,
    label: str | None = None,
    **invoke_kwargs: Any,
) -> Scored[T] | Unmeasured:
    """Invoke, classify and parse one LLM call.

    ``Scored`` iff *parse* returned; otherwise ``Unmeasured`` naming why — a
    transport refusal, cap patience exhausted, or an answer *parse* rejected
    with ``ValueError``/``TypeError``/``KeyError``. Never fabricates a value.
    ``cost_usd`` is the spend summed over every attempt. With *usage_gate* the
    call runs through ``invoke_with_cap_retry``, which injects each account's
    credentials into *invoke_kwargs*. Any other exception propagates.
    """
    name = label or str(stage)
    spent = 0.0

    async def _attempt(**dispatch: Any) -> AgentResult:
        nonlocal spent
        attempt = await invoke(**dispatch)
        spent += coerce_cost_usd(attempt.cost_usd)
        return attempt

    try:
        if usage_gate is None:
            result = await _attempt(backend=backend, **invoke_kwargs)
        else:
            result = await invoke_with_cap_retry(
                usage_gate, name, invoke_fn=_attempt, backend=backend,
                **invoke_kwargs,
            )
    except AllAccountsCappedException as e:
        logger.warning(f'{name}: unmeasured, cap patience exhausted: {e}')
        return Unmeasured(
            stage, UnmeasuredCause.CAP_EXHAUSTED, clip_reason(str(e)), cost_usd=spent,
        )

    refused = classify_unmeasured(result, stage=stage, backend=backend)
    if refused is not None:
        logger.warning(
            f'{name}: unmeasured, refused at the transport layer: '
            f'{refused.cause}: {refused.reason}'
        )
        return dataclasses.replace(refused, cost_usd=spent)
    try:
        value = parse(result)
    except (ValueError, TypeError, KeyError) as e:
        reason = exception_reason(e)
        logger.warning(f'{name}: unmeasured, unparseable answer: {reason}')
        return Unmeasured(stage, UnmeasuredCause.UNPARSEABLE, reason, cost_usd=spent)
    return Scored(value, cost_usd=spent)
