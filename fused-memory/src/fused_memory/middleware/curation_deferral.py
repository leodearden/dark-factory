"""The vocabulary and policy for a task submission the curator could not judge.

Why it was deferred (:class:`DeferReason`), when it is retried
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


def backlog_severity(reason: DeferReason) -> Literal['info', 'blocking']:
    match reason:
        case DeferReason.ALL_ACCOUNTS_CAPPED:
            return 'info'
        case DeferReason.ACCOUNTS_AUTH_FAILED:
            return 'blocking'
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
