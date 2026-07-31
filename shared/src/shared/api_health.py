"""ApiHealthGate — the provider-degradation circuit breaker (contract C4).

Source contract: ``plans/server-side-api-error-handling-prd.md`` C4 (task mu).

The gate answers one question for the whole fleet: *is the provider itself
degraded right now?*  It is fed from the invocation choke point via
:meth:`ApiHealthGate.report` and read by the scheduler (dispatch throttle),
the workflow (steward suppression), the harness (watcher-rotation deferral)
and the escalation lifecycle (``api_degraded`` file/resolve/L2-promote).

Why a breaker at all rather than per-invocation retry: PRD decision 4 —
the 2026-07-29 incident showed the FRESHEST account carrying the highest
failure rate, i.e. the provider was degraded, not any one account's quota.
Failing over on a 5xx only multiplies load on an already-degraded provider.

Why the trip needs three conditions and not just a count: decision 10 —
*throttle, not halt*.  A provider degradation that is only partial must not
strand the healthy fraction of the fleet, so the gate demands failure
**count**, failure **breadth** (distinct tasks) and failure **rate** all at
once before it declares the provider down.  Any one alone is satisfiable by
a single pathological task or by a locally-sick fleet.

State machine::

    Closed  --(count & breadth & rate all breached)-->  Open
    Open    --OK-->  Probing(1)  --OK-->  Closed        (hysteresis)
    Probing --5xx--> Open  (original `since` preserved, streak reset)

``Probing`` is a sub-state of open: the provider is not healthy until the
gate closes.  Consumers should read :attr:`ApiHealthGate.is_degraded` rather
than testing ``isinstance(s, Open)``, which silently un-throttles mid-outage.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from shared.invocation_outcome import InvocationOutcome, ServerError

if TYPE_CHECKING:
    from shared.cost_store import CostStore

logger = logging.getLogger(__name__)

__all__ = [
    'GateStats',
    'GateState',
    'Closed',
    'Open',
    'Probing',
    'ApiHealthGate',
]

# PRD C4 `api_health.*` config-block defaults.  These are the documented
# values; xi's config block maps 1:1 onto the constructor keywords with no
# translation layer, so the PRD table and this module cannot drift.
DEFAULT_WINDOW_SECS = 600.0
DEFAULT_MIN_FAILURES = 8
DEFAULT_MIN_DISTINCT_TASKS = 3
DEFAULT_FAILURE_RATE_THRESHOLD = 0.5
DEFAULT_CLOSE_AFTER_SUCCESSES = 2


@dataclass(frozen=True)
class GateStats:
    """The rolling-window evidence behind a state, captured at transition time.

    Carried on ``Open``/``Probing`` so the escalation narrative and the
    dashboard strip can say *why* the gate tripped without re-deriving it.
    """

    failures: int
    """5xx (``ServerError``) reports in the window."""

    completions: int
    """All completed invocations in the window — the rate denominator."""

    distinct_tasks: int
    """Distinct non-None task_ids among the FAILURE reports (the breadth signal)."""

    failure_rate: float
    """``failures / completions``; 0.0 when the window is empty."""

    window_secs: float
    """The window width these counts were taken over."""


class GateState:
    """Base class for the gate-state tagged union.

    Every concrete variant is a frozen dataclass subclass — see ``__all__``.
    Use ``isinstance(state, <Variant>)`` to discriminate, exactly as callers
    already do for ``shared.invocation_outcome.InvocationOutcome``.
    """


@dataclass(frozen=True)
class Closed(GateState):
    """The provider is healthy; dispatch proceeds normally."""


@dataclass(frozen=True)
class Open(GateState):
    """The provider is degraded; consumers throttle."""

    since: datetime
    """Wall-clock instant of the ORIGINAL trip.

    Preserved across probe regressions on purpose — it measures the true
    continuous outage, which is what omicron's ``max_open_before_l2_hours``
    promotion clock reads.
    """

    stats: GateStats


@dataclass(frozen=True)
class Probing(GateState):
    """Recovery is being probed: successes are accruing but the gate is not yet closed.

    Still degraded — consumers must keep throttling.
    """

    since: datetime
    """Carried over from :class:`Open` — see its note."""

    stats: GateStats
    consecutive_successes: int


@dataclass(frozen=True)
class _Sample:
    """One reported invocation in the rolling window."""

    ts: float
    """Monotonic timestamp of the report."""

    is_failure: bool
    """True iff the outcome was a ``ServerError`` (5xx)."""

    task_id: str | None
    """Reporting task, or None for task-less invocations (watcher/steward)."""


class ApiHealthGate:
    """In-memory C4 state machine over a rolling window of invocation outcomes.

    The in-memory state is authoritative; the ``api_error`` rows written to
    :class:`~shared.cost_store.CostStore` are forensics only.

    Thresholds are constructor keywords defaulting to PRD C4's values.  Note
    there is deliberately no ``enabled`` flag: enablement is xi's wiring
    concern (it simply does not construct/report when disabled), and carrying
    a disabled branch here would be a dead path this task cannot test.
    """

    def __init__(
        self,
        *,
        window_secs: float = DEFAULT_WINDOW_SECS,
        min_failures: int = DEFAULT_MIN_FAILURES,
        min_distinct_tasks: int = DEFAULT_MIN_DISTINCT_TASKS,
        failure_rate_threshold: float = DEFAULT_FAILURE_RATE_THRESHOLD,
        close_after_successes: int = DEFAULT_CLOSE_AFTER_SUCCESSES,
        cost_store: CostStore | None = None,
        project_id: str | None = None,
        run_id: str | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        # Validate eagerly and fail LOUD rather than clamping to something
        # plausible.  A clamped-to-zero threshold is not a degraded gate, it
        # is an inverted one: `min_failures=0` or `failure_rate_threshold=0`
        # trips on the empty window and stays tripped, throttling the entire
        # fleet forever with no signal anywhere that a config typo caused it.
        # Refusing to construct surfaces the typo at startup, next to the
        # config that caused it.
        if window_secs <= 0:
            raise ValueError(f'window_secs must be > 0, got {window_secs!r}')
        if min_failures < 1:
            raise ValueError(f'min_failures must be >= 1, got {min_failures!r}')
        if min_distinct_tasks < 1:
            raise ValueError(f'min_distinct_tasks must be >= 1, got {min_distinct_tasks!r}')
        if not 0 < failure_rate_threshold <= 1:
            raise ValueError(
                f'failure_rate_threshold must be in (0, 1], got {failure_rate_threshold!r}'
            )
        if close_after_successes < 1:
            raise ValueError(f'close_after_successes must be >= 1, got {close_after_successes!r}')

        self.window_secs = window_secs
        self.min_failures = min_failures
        self.min_distinct_tasks = min_distinct_tasks
        self.failure_rate_threshold = failure_rate_threshold
        self.close_after_successes = close_after_successes

        self._cost_store = cost_store
        self._project_id = project_id
        self._run_id = run_id
        # Two clocks on purpose: monotonic for window/hysteresis arithmetic
        # (survives NTP steps), wall for `since` and the row `created_at`
        # (read by humans and by omicron's promotion clock).
        self._monotonic_clock = monotonic_clock
        self._wall_clock = wall_clock

        self._state: GateState = Closed()
        # Rolling window of reported invocations, oldest first.  Same shape as
        # Scheduler._blocked_transitions (scheduler.py:1551): a deque on the
        # monotonic clock, evicted from the left on every report.
        self._samples: deque[_Sample] = deque()

    async def report(
        self,
        outcome: InvocationOutcome,
        *,
        task_id: str | None,
        account: str,
        role: str,
    ) -> GateState:
        """Feed one completed invocation to the gate; return the resulting state.

        Called from the invocation choke point.  The returned value IS the
        post-report state, so the caller needs no follow-up :meth:`state` call.

        Args:
            outcome: The classified invocation outcome.  ``ServerError`` counts
                as a provider failure; ``OK`` as a provider success; every
                other variant is neutral (see :meth:`_classify`).
            task_id: The reporting task, or None for task-less invocations.
            account: Account name, recorded on the forensics row.
            role: Agent role, recorded on the forensics row.
        """
        now = self._monotonic_clock()
        self._evict(now)

        is_failure = isinstance(outcome, ServerError)
        self._samples.append(_Sample(ts=now, is_failure=is_failure, task_id=task_id))

        stats = self._stats()
        # State mutation happens here, before any await: on a single-threaded
        # event loop that makes the transition atomic, so two concurrent
        # reporters cannot interleave mid-transition and no lock is needed.
        self._state = self._next_state(outcome, stats)
        return self._state

    def _next_state(self, outcome: InvocationOutcome, stats: GateStats) -> GateState:
        """Pure transition function: (current state, outcome, window stats) -> next state."""
        if isinstance(self._state, Closed):
            if self._should_trip(stats):
                logger.warning(
                    'ApiHealthGate TRIPPED (provider degraded): %d/%d failures (%.0f%%) '
                    'across %d distinct tasks in %.0fs — thresholds %d/%d/%.2f',
                    stats.failures,
                    stats.completions,
                    stats.failure_rate * 100,
                    stats.distinct_tasks,
                    stats.window_secs,
                    self.min_failures,
                    self.min_distinct_tasks,
                    self.failure_rate_threshold,
                )
                return Open(since=self._wall_clock(), stats=stats)
            return self._state
        return self._state

    def _should_trip(self, stats: GateStats) -> bool:
        """All three C4 conditions must hold — each is independently necessary.

        Count alone trips on one pathological task retrying; breadth alone
        trips on three tasks that each failed once; rate alone trips on a
        single failure in an otherwise-idle window.  Requiring all three is
        what makes a trip mean "the provider is down" (decision 10:
        throttle, not halt — never strand the healthy fraction).
        """
        return (
            stats.failures >= self.min_failures
            and stats.distinct_tasks >= self.min_distinct_tasks
            and stats.failure_rate >= self.failure_rate_threshold
        )

    def _evict(self, now: float) -> None:
        """Drop samples older than ``window_secs`` from the left of the deque."""
        cutoff = now - self.window_secs
        while self._samples and self._samples[0].ts < cutoff:
            self._samples.popleft()

    def _stats(self) -> GateStats:
        """Summarise the current window."""
        completions = len(self._samples)
        failures = sum(1 for s in self._samples if s.is_failure)
        distinct = {s.task_id for s in self._samples if s.is_failure and s.task_id is not None}
        return GateStats(
            failures=failures,
            completions=completions,
            distinct_tasks=len(distinct),
            failure_rate=(failures / completions) if completions else 0.0,
            window_secs=self.window_secs,
        )

    def state(self) -> GateState:
        """Return the current state as an immutable snapshot.

        Sync on purpose: the scheduler's dispatch-eligibility hot path reads
        the gate without an await.
        """
        return self._state

    @property
    def is_degraded(self) -> bool:
        """True while the provider is Open OR Probing.

        Every C4 open-state consumer must keep throttling while probing.
        Exposing one boolean here makes "forgot to handle Probing" — a silent
        un-throttle mid-outage — unrepresentable at the call sites.
        """
        return not isinstance(self._state, Closed)
