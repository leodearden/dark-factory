"""The ``merge_verify_cancelled`` row: why the lane cancelled a running verify, and how far it got.

Rationale: ``plans/merge-lane-policy-prd.md`` D10.
"""
from __future__ import annotations

import dataclasses
import statistics
from collections import deque
from enum import StrEnum


class VerifyCancelCause(StrEnum):
    """Why the lane cancelled a still-running verify."""

    #: A head-failure cascade cancelled a downstream verify.
    CASCADE = 'cascade'
    #: The request's sole waiter gave up.
    ABANDONED = 'abandoned'
    #: An operator merge-queue halt requeued the request.
    OPERATOR_HALT = 'operator_halt'
    #: The no-progress (dead-verify) abort fired.
    NO_PROGRESS = 'no_progress'
    #: A green deep tip superseded the head's verify.
    TIP_ADOPTED = 'tip_adopted'
    #: ``stop()`` or a detach drained the lane.
    SHUTDOWN = 'shutdown'
    #: A passthrough finalize failed and tore down a parked gate.
    FINALIZE_ERROR = 'finalize_error'


@dataclasses.dataclass(frozen=True)
class VerifyCancel:
    """One deliberate cancel of a running verify.

    ``started_at`` is on the worker clock's wall ``now()`` scale, the same
    scale as ``InflightEntry.started_at``; None when the canceller cannot
    see the verify's start.
    """

    cause: VerifyCancelCause
    started_at: float | None
    host: str | None


class VerifyDurationWindow:
    """Per-host rolling window of completed-verify durations, in milliseconds."""

    def __init__(self, *, maxlen: int) -> None:
        if maxlen < 1:
            raise ValueError(f'VerifyDurationWindow maxlen must be >= 1, got maxlen={maxlen}')
        self._maxlen = maxlen
        self._durations: dict[str, deque[float]] = {}

    def record(self, host: str, duration_ms: float) -> None:
        if duration_ms <= 0:
            return
        self._durations.setdefault(host, deque(maxlen=self._maxlen)).append(duration_ms)

    def fraction(self, host: str, elapsed_ms: float | None) -> float | None:
        """*elapsed_ms* over *host*'s median completed verify; None until it has one."""
        durations = self._durations.get(host)
        if elapsed_ms is None or not durations:
            return None
        return round(elapsed_ms / statistics.median(durations), 3)


def merge_verify_cancelled_data(
    cancel: VerifyCancel, *, now: float, window: VerifyDurationWindow,
) -> dict[str, object]:
    elapsed_ms = (
        None if cancel.started_at is None
        else max(0, round((now - cancel.started_at) * 1000))
    )
    return {
        'elapsed_ms': elapsed_ms,
        'fraction': None if cancel.host is None else window.fraction(cancel.host, elapsed_ms),
        'cause': cancel.cause.value,
        'host': cancel.host,
    }
