"""The vocabulary and policy for a task submission the curator could not judge.

The curator defers a submission when its account pool, its LLM call or its
corpus is unavailable, when its zero-output breaker is open, or when it raised
instead of deciding. Why it was deferred (:class:`DeferReason`), how loudly a
backlog of such deferrals escalates and what restores curation for it
(:func:`backlog_severity`, :func:`backlog_remediation`), when it is retried
(:class:`CurationRetryPolicy`; a deferral has no terminal bound, it waits for a
verdict), and how a create that no curator verdict judged is marked
(:func:`dedupe_skipped_marker`). The bottom layer of the
curator's defer seam: the curator, the ticket store and the interceptor
import it; it imports none of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal, assert_never

from shared.cli_invoke import AllAccountsCappedException
from shared.usage_gate import PoolFrozen

if TYPE_CHECKING:
    from fused_memory.config.schema import CuratorConfig


class DeferReason(StrEnum):
    ALL_ACCOUNTS_CAPPED = 'all-accounts-capped'
    ACCOUNTS_AUTH_FAILED = 'accounts-auth-failed'
    LLM_FAILED = 'llm-failed'
    CORPUS_FAILED = 'corpus-failed'
    ZERO_OUTPUT_BREAKER_OPEN = 'zero-output-breaker-open'
    CURATOR_RAISED = 'curator-raised'


def backlog_severity(reason: DeferReason) -> Literal['info', 'blocking']:
    match reason:
        case DeferReason.ALL_ACCOUNTS_CAPPED | DeferReason.ZERO_OUTPUT_BREAKER_OPEN:
            return 'info'
        case (
            DeferReason.ACCOUNTS_AUTH_FAILED
            | DeferReason.LLM_FAILED
            | DeferReason.CORPUS_FAILED
            | DeferReason.CURATOR_RAISED
        ):
            return 'blocking'
        case _:
            assert_never(reason)


def backlog_remediation(reason: DeferReason) -> str:
    match reason:
        case DeferReason.ALL_ACCOUNTS_CAPPED:
            return (
                'Every curator account is usage-capped; curation resumes by itself '
                'once an account cap resets.'
            )
        case DeferReason.ACCOUNTS_AUTH_FAILED:
            return (
                'Every curator account failed authentication, which will not clear by '
                "itself; restore the accounts' credentials to resume curation."
            )
        case DeferReason.LLM_FAILED:
            return (
                "The curator's LLM call is failing and is not known to recover by itself; "
                'read the curator_failure escalations and the fused-memory log for the '
                'failing subtype or exception (for example the claude binary missing from '
                'the service PATH). A submission the curator can never judge may be '
                'withdrawn with cancel_ticket.'
            )
        case DeferReason.CORPUS_FAILED:
            return (
                'The curator could not assemble its candidate corpus and is not known to '
                'recover by itself; Qdrant, the embedder or the task backend is failing.'
            )
        case DeferReason.ZERO_OUTPUT_BREAKER_OPEN:
            return (
                "The curator's zero-output breaker is open and closes by itself after "
                'zero_output_breaker_cooldown_seconds; the hangs that opened it fold into '
                'the curator_zero_output_hang record.'
            )
        case DeferReason.CURATOR_RAISED:
            return (
                'The curator raised instead of deciding, a code defect that will not clear '
                'by itself; read the traceback logged by the fused-memory ticket worker.'
            )
        case _:
            assert_never(reason)


def deferral_justification(reason: DeferReason) -> str:
    return f'curation-deferred: {reason}'


POOL_UNAVAILABLE_ERRORS: Final = (
    AllAccountsCappedException,
    PoolFrozen,
)


def pool_defer_reason(exc: AllAccountsCappedException | PoolFrozen) -> DeferReason:
    if (
        isinstance(exc, PoolFrozen)
        and exc.scope is None
        and not exc.capped_account_names
        and exc.auth_failed_account_names
    ):
        return DeferReason.ACCOUNTS_AUTH_FAILED
    return DeferReason.ALL_ACCOUNTS_CAPPED


def _require_aware(name: str, value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f'{name} must be timezone-aware, got naive {value.isoformat()}')


@dataclass(frozen=True)
class CurationRetryPolicy:
    backoff_base_seconds: float
    backoff_max_seconds: float

    def __post_init__(self) -> None:
        bounds = {
            'backoff_base_seconds': self.backoff_base_seconds,
            'backoff_max_seconds': self.backoff_max_seconds,
        }
        non_positive = {name: value for name, value in bounds.items() if not value > 0}
        if non_positive:
            raise ValueError(f'CurationRetryPolicy bounds must be > 0, got {non_positive}')
        if self.backoff_max_seconds < self.backoff_base_seconds:
            raise ValueError(
                f'CurationRetryPolicy backoff_max_seconds ({self.backoff_max_seconds}) must be '
                f'>= backoff_base_seconds ({self.backoff_base_seconds})'
            )

    @classmethod
    def from_config(cls, curator: CuratorConfig) -> CurationRetryPolicy:
        return cls(
            backoff_base_seconds=curator.deferral_backoff_base_seconds,
            backoff_max_seconds=curator.deferral_backoff_max_seconds,
        )

    def next_attempt_at(self, *, attempts: int, now: datetime) -> datetime:
        if attempts < 1:
            raise ValueError(f'attempts counts the deferral being recorded, so must be >= 1, got {attempts}')
        _require_aware('now', now)
        return now + timedelta(seconds=self._backoff_seconds(attempts))

    def _backoff_seconds(self, attempts: int) -> float:
        doublings = attempts - 1
        if doublings >= math.log2(self.backoff_max_seconds / self.backoff_base_seconds):
            return self.backoff_max_seconds
        return self.backoff_base_seconds * 2**doublings


DEDUPE_SKIPPED_KEY: Final = 'dedupe_skipped'
_MARKER_REASON_MAX_CHARS: Final = 500


def dedupe_skipped_marker(reason: str, *, at: datetime) -> dict[str, dict[str, str]]:
    _require_aware('at', at)
    return {
        DEDUPE_SKIPPED_KEY: {
            'reason': reason[:_MARKER_REASON_MAX_CHARS],
            'at': at.astimezone(UTC).isoformat(),
        },
    }
