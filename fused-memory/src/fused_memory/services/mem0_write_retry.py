"""Durable server-side retry of Mem0 writes that failed on a transient provider error, driven by the mem0 intent journal."""

from __future__ import annotations

import asyncio
import collections
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

import openai

from fused_memory.services.write_journal import Mem0IntentStatus, WriteJournal

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True)
class Mem0RetryPassSummary:
    due: int = 0
    landed: int = 0
    reconciled: int = 0
    rescheduled: int = 0
    dead_lettered: int = 0


class _RetryOutcome(StrEnum):
    LANDED = 'landed'
    RECONCILED = 'reconciled'
    RESCHEDULED = 'rescheduled'
    DEAD_LETTERED = 'dead_lettered'


def mem0_write_op_landed(backend_ops: Sequence[Mapping]) -> bool:
    """Whether a write's backend_ops prove its Mem0 leg landed: any SUCCESS mem0 row."""
    return any(op.get('backend') == 'mem0' and op.get('success') for op in backend_ops)


async def retry_due_mem0_intents(
    journal: WriteJournal,
    reissue: Callable[[dict], Awaitable[object]],
    *,
    policy: Mem0RetryPolicy,
    now: datetime,
) -> Mem0RetryPassSummary:
    """Re-issue up to one batch of ``retrying`` intents due by ``now``."""
    due = await journal.get_due_mem0_retries(now=now, limit=policy.batch_size)
    outcomes: collections.Counter[_RetryOutcome] = collections.Counter()
    for intent in due:
        outcomes[await _retry_one(journal, reissue, intent, policy=policy, now=now)] += 1
    return Mem0RetryPassSummary(
        due=len(due),
        landed=outcomes[_RetryOutcome.LANDED],
        reconciled=outcomes[_RetryOutcome.RECONCILED],
        rescheduled=outcomes[_RetryOutcome.RESCHEDULED],
        dead_lettered=outcomes[_RetryOutcome.DEAD_LETTERED],
    )


async def _retry_one(
    journal: WriteJournal,
    reissue: Callable[[dict], Awaitable[object]],
    intent: dict,
    *,
    policy: Mem0RetryPolicy,
    now: datetime,
) -> _RetryOutcome:
    attempt = intent['attempts'] + 1
    if mem0_write_op_landed(await journal.get_backend_ops_for_write_op(intent['write_op_id'])):
        await journal.resolve_mem0_intent(
            intent['id'],
            Mem0IntentStatus.COMPLETED,
            reason='reconciled: a success mem0 backend_op shows an earlier retry landed',
            expected_status=Mem0IntentStatus.RETRYING,
        )
        return _RetryOutcome.RECONCILED
    try:
        await reissue(intent)
    except Exception as exc:
        return await _record_failed_attempt(
            journal, intent, exc, attempt=attempt, policy=policy, now=now
        )
    await journal.resolve_mem0_intent(
        intent['id'],
        Mem0IntentStatus.COMPLETED,
        reason=f'landed on retry attempt {attempt}',
        expected_status=Mem0IntentStatus.RETRYING,
    )
    logger.info(
        'Mem0 retry landed intent %s on attempt %d (write_op_id=%s)',
        intent['id'],
        attempt,
        intent['write_op_id'],
    )
    return _RetryOutcome.LANDED


async def _record_failed_attempt(
    journal: WriteJournal,
    intent: dict,
    exc: Exception,
    *,
    attempt: int,
    policy: Mem0RetryPolicy,
    now: datetime,
) -> _RetryOutcome:
    failure = classify_transient_provider_failure(exc)
    if failure is not None:
        retry_at = policy.retry_at(attempts=attempt, now=now)
        await journal.schedule_mem0_retry(
            intent['id'], reason=retry_reason(failure, exc), retry_at=retry_at
        )
        logger.warning(
            'Mem0 retry attempt %d of intent %s failed transiently (%s); next at %s '
            '(write_op_id=%s)',
            attempt,
            intent['id'],
            failure.value,
            retry_at.isoformat(),
            intent['write_op_id'],
        )
        return _RetryOutcome.RESCHEDULED
    reason = (
        f'retry attempt {attempt} raised a non-transient {type(exc).__name__}: {exc} '
        f'(write_op_id={intent["write_op_id"]})'
    )
    await journal.resolve_mem0_intent(
        intent['id'],
        Mem0IntentStatus.DEAD,
        reason=reason,
        expected_status=Mem0IntentStatus.RETRYING,
    )
    logger.error('Mem0 retry dead-lettered intent %s: %s', intent['id'], reason)
    return _RetryOutcome.DEAD_LETTERED


async def run_periodic_mem0_retry(
    pass_fn: Callable[[], Awaitable[object]], *, interval_seconds: float
) -> None:
    """Run ``pass_fn`` every ``interval_seconds``, sleeping first, until cancelled."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await pass_fn()
        except Exception:
            logger.exception('Mem0 periodic retry pass failed; the loop continues')
