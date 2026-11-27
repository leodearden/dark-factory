"""How many attempts an openai SDK client makes per request, how long each may take, and their combined ceiling.

Pydantic and the stdlib only: config loading must not import openai or httpx.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Mirrors of openai/_constants.py INITIAL_RETRY_DELAY and MAX_RETRY_DELAY, pinned by tests/test_request_attempts_config.py.
SDK_INITIAL_RETRY_DELAY_SECONDS = 0.5
SDK_MAX_RETRY_DELAY_SECONDS = 8.0


class RequestAttemptsConfig(BaseModel):
    """A per-attempt deadline and an attempt count for one openai SDK client."""

    per_attempt_timeout_seconds: float = Field(
        default=8.0,
        gt=0,
        description=(
            'Caps every SDK timeout phase of one attempt (read, write, pool; '
            "connect keeps the SDK's 5 s when that is smaller)."
        ),
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            'Attempts per request, the first included. The stalled-schedule ceiling '
            'assumes each attempt stalls in a single phase. Sleeps the SDK takes on '
            'a Retry-After header (at most 60 s, on 429/5xx) and stalls slower than '
            "the ceiling are not counted; the caller's await_within budget still "
            'ends them.'
        ),
    )

    @property
    def stalled_schedule_seconds(self) -> float:
        """The longest a request takes when every attempt stalls to its deadline."""
        backoff = sum(
            min(SDK_INITIAL_RETRY_DELAY_SECONDS * 2**retry, SDK_MAX_RETRY_DELAY_SECONDS)
            for retry in range(self.max_attempts - 1)
        )
        return self.max_attempts * self.per_attempt_timeout_seconds + backoff

    def check_fits_within(self, budget_s: float, *, owner: str, budget_name: str) -> None:
        """Raise ValueError unless a fully stalled schedule ends strictly inside *budget_s*."""
        ceiling = self.stalled_schedule_seconds
        if ceiling < budget_s:
            return
        raise ValueError(
            f'{owner} (max_attempts={self.max_attempts}, '
            f'per_attempt_timeout_seconds={self.per_attempt_timeout_seconds}) stalls for '
            f'up to {ceiling}s including SDK retry backoff, which must be less than '
            f'{budget_name} ({budget_s}s): otherwise a stalled request outlives its '
            'budget instead of failing on its own retries.'
        )
