"""The conformance audit: a hard per-attempt validator installed on an arm's LLM client.

The audit wraps the client's ``_generate_response``, the seam upstream's
``generate_response`` retry loop calls once per attempt. Every received response
is validated against its ``response_model``. A mismatch is counted and raised, so
upstream re-prompts and finally fails, and off-schema JSON never reaches the
caller. Transport failures are tallied outside the conformance denominator.
Upstream has two per-attempt return shapes, chosen by client family at install:
``BaseOpenAIClient``'s ``(payload, input_tokens, output_tokens)`` and
``OpenAIGenericClient``'s bare payload; a client of any other family is refused.
"""

import inspect
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.openai_base_client import BaseOpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from pydantic import BaseModel, TypeAdapter, ValidationError
from shared.memory_eval_metrics import Metric

from fused_memory.arm_harness.metrics_record import LlmMetricId

_JSON_OBJECT: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])
_SEAM = '_generate_response'

ResponseValidator = Callable[[object, type[BaseModel] | None], None]
"""Raises ``ValidationError`` (or ``JSONDecodeError``) when a response is off-schema."""


def _payload_of_token_tuple(result: Any) -> object:
    payload, _input_tokens, _output_tokens = result
    return payload


def _payload_itself(result: Any) -> object:
    return result


@dataclass(frozen=True)
class _AttemptShape:
    family: type[LLMClient]
    payload: Callable[[Any], object]


_ATTEMPT_SHAPES: tuple[_AttemptShape, ...] = (
    _AttemptShape(BaseOpenAIClient, _payload_of_token_tuple),
    _AttemptShape(OpenAIGenericClient, _payload_itself),
)


@dataclass(frozen=True)
class ConformanceCounts:
    schema_valid: int
    transport_errors: int
    invalid_by_error_class: Mapping[str, int]

    @property
    def schema_invalid(self) -> int:
        return sum(self.invalid_by_error_class.values())

    @property
    def calls(self) -> int:
        return self.schema_valid + self.schema_invalid + self.transport_errors


class ConformanceLedger:
    """One arm run's mutable tally of audited attempts; ``snapshot()`` is its frozen view."""

    def __init__(self) -> None:
        self._schema_valid = 0
        self._transport_errors = 0
        self._invalid: Counter[str] = Counter()

    def record_valid(self) -> None:
        self._schema_valid += 1

    def record_invalid(self, error: BaseException) -> None:
        self._invalid[type(error).__name__] += 1

    def record_transport_error(self) -> None:
        self._transport_errors += 1

    def snapshot(self) -> ConformanceCounts:
        return ConformanceCounts(
            schema_valid=self._schema_valid,
            transport_errors=self._transport_errors,
            invalid_by_error_class=MappingProxyType(dict(self._invalid)),
        )


def validate_response(result: object, response_model: type[BaseModel] | None) -> None:
    """The audit's default validator: ``response_model`` when given, else any JSON object."""
    if response_model is None:
        _JSON_OBJECT.validate_python(result)
    else:
        response_model.model_validate(result)


def install_conformance_audit(
    client: LLMClient,
    ledger: ConformanceLedger,
    *,
    validator: ResponseValidator = validate_response,
) -> None:
    attempt = getattr(client, _SEAM, None)
    if not callable(attempt):
        raise TypeError(
            f'{type(client).__name__} has no {_SEAM}: the conformance audit wraps the '
            'per-attempt seam of graphiti_core llm_client/client.py::LLMClient.generate_response, '
            'and auditing without it would be silently inert'
        )
    shape = _attempt_shape_of(client)
    signature = inspect.signature(attempt)
    if 'response_model' not in signature.parameters:
        raise TypeError(
            f'{type(client).__name__}.{_SEAM}{signature} takes no response_model: '
            'the conformance audit cannot validate an attempt without it'
        )
    # Instance attribute shadows the class method: upstream's retry loop calls self._generate_response.
    setattr(client, _SEAM, _audited(attempt, signature, shape, ledger, validator))


def _attempt_shape_of(client: LLMClient) -> _AttemptShape:
    for shape in _ATTEMPT_SHAPES:
        if isinstance(client, shape.family):
            return shape
    families = ', '.join(shape.family.__name__ for shape in _ATTEMPT_SHAPES)
    raise TypeError(
        f'{type(client).__name__} is in none of the attempt-shape families the conformance '
        f'audit supports ({families}): it refuses rather than run inert on a {_SEAM} '
        'return value whose shape it does not know'
    )


def _audited(
    attempt: Callable[..., Any],
    signature: inspect.Signature,
    shape: _AttemptShape,
    ledger: ConformanceLedger,
    validator: ResponseValidator,
) -> Callable[..., Awaitable[Any]]:
    async def audited_attempt(*args: Any, **kwargs: Any) -> Any:
        call = signature.bind(*args, **kwargs)
        call.apply_defaults()
        response_model = call.arguments['response_model']
        try:
            result = await attempt(*args, **kwargs)
            validator(shape.payload(result), response_model)
        except (json.JSONDecodeError, ValidationError) as error:
            ledger.record_invalid(error)
            raise
        except Exception:
            ledger.record_transport_error()
            raise
        ledger.record_valid()
        return result

    return audited_attempt


def conformance_rate_metric(counts: ConformanceCounts) -> Metric | None:
    received = counts.schema_valid + counts.schema_invalid
    if received == 0:
        return None
    return Metric(
        metric_id=LlmMetricId.CONFORMANCE_RATE,
        kind='proportion',
        value=counts.schema_valid / received,
        n=received,
        denominator=received,
        direction='lower_is_worse',
    )
