"""The ``merge_verify_cancelled`` row: why the lane cancelled a running verify, and how far it got.

Rationale: ``plans/merge-lane-policy-prd.md`` D10.
"""
from __future__ import annotations

import dataclasses
import statistics
from collections import deque
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.merge_lane.types import InflightEntry
    from orchestrator.verify_runner import HostLease


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
    #: A head's gate re-verify became a dispatch, so its stacked successor's verify was cancelled.
    GATE_REANCHOR = 'gate_reanchor'


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

    @classmethod
    def of(
        cls,
        cause: VerifyCancelCause,
        entry: InflightEntry | None,
        lease: HostLease | None = None,
    ) -> VerifyCancel:
        """The cancel of *entry*'s verify, on *lease* or else the entry's own lease."""
        holder = lease if lease is not None else (entry.lease if entry is not None else None)
        return cls(
            cause,
            started_at=entry.started_at if entry is not None else None,
            host=holder.name if holder is not None else None,
        )


class RecordedDispatches:
    """The verify dispatches whose cancel has a row, so each gets exactly one.

    A dispatch is its task id and ``started_at``.  A second canceller can
    reach a verify an abort trigger already cancelled, e.g. ``stop()``
    draining the entry while the trigger still cleans up.  Past *cap* the
    oldest dispatch is forgotten.  A cancel with no ``started_at`` names no
    dispatch and is always claimed.
    """

    def __init__(self, *, cap: int) -> None:
        if cap < 1:
            raise ValueError(f'RecordedDispatches cap must be >= 1, got cap={cap}')
        self._cap = cap
        self._claimed: dict[tuple[str, float], None] = {}

    def claim(self, task_id: str, cancel: VerifyCancel) -> bool:
        """True the first time *task_id*'s dispatch is claimed."""
        if cancel.started_at is None:
            return True
        key = (task_id, cancel.started_at)
        if key in self._claimed:
            return False
        self._claimed[key] = None
        while len(self._claimed) > self._cap:
            del self._claimed[next(iter(self._claimed))]
        return True


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
