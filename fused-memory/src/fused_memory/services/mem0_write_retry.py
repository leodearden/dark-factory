"""Durable server-side retry of Mem0 writes that failed on a transient provider error, driven by the mem0 intent journal."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

import openai


class TransientProviderFailure(StrEnum):
    RATE_LIMITED = 'rate_limited'
    SERVER_ERROR = 'server_error'
    CONNECTION = 'connection'


def classify_transient_provider_failure(
    exc: BaseException,
) -> TransientProviderFailure | None:
    """Name the transient provider failure ``exc`` is, or ``None`` if it is not one.

    Matched by openai SDK type only, never by message text. Re-issuing such a
    write cannot duplicate it: ``Mem0Backend.add`` pins ``infer=False``, and
    mem0's ``_add_to_vector_store`` embeds BEFORE ``vector_store.insert``, so an
    openai error proves nothing was persisted.

    The write-budget ``TimeoutError`` from ``await_within`` is deliberately not
    transient: it wraps the whole add and may fire while the insert is in
    flight. Ambiguous timeouts are task 4872's domain.
    """
    if isinstance(exc, openai.RateLimitError):
        return TransientProviderFailure.RATE_LIMITED
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return TransientProviderFailure.SERVER_ERROR
    if isinstance(exc, openai.APIConnectionError):
        return TransientProviderFailure.CONNECTION
    return None


def retry_reason(failure: TransientProviderFailure, exc: BaseException) -> str:
    return f'{failure.value}: {type(exc).__name__}: {exc}'


@dataclass(frozen=True)
class Mem0RetryPolicy:
    initial_delay_seconds: float
    max_delay_seconds: float
    poll_interval_seconds: float
    batch_size: int

    def __post_init__(self) -> None:
        for name in (
            'initial_delay_seconds',
            'max_delay_seconds',
            'poll_interval_seconds',
            'batch_size',
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f'Mem0RetryPolicy.{name} must be positive, got {value}')
        if self.initial_delay_seconds > self.max_delay_seconds:
            raise ValueError(
                f'Mem0RetryPolicy.initial_delay_seconds ({self.initial_delay_seconds}) '
                f'exceeds max_delay_seconds ({self.max_delay_seconds})'
            )

    def retry_at(self, *, attempts: int, now: datetime) -> datetime:
        """When to re-issue a write that has failed ``attempts`` times."""
        delay = min(
            self.max_delay_seconds,
            self.initial_delay_seconds * 2 ** (attempts - 1),
        )
        return now + timedelta(seconds=delay)


DEFAULT_MEM0_RETRY_POLICY = Mem0RetryPolicy(
    initial_delay_seconds=30.0,
    max_delay_seconds=900.0,
    poll_interval_seconds=30.0,
    batch_size=50,
)
