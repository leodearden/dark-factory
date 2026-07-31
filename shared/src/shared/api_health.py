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

**The gate is report-driven, and a degraded gate can outlive its evidence.**
There is no timer: the window ages out only when a report arrives, and the
only route from ``Open`` back to ``Closed`` is a successful probe.  So a gate
that trips and then sees no traffic — an idle overnight queue, dispatch
throttled to a single probe, a paused fleet — stays ``Open`` indefinitely on
evidence that has fully aged out.  That is deliberate: closing on silence
would declare recovery with no successful invocation to show for it, which
is exactly the claim the probe-close contract exists to require.

Any consumer whose behaviour escalates with TIME SPENT open — notably the
``max_open_before_l2_hours`` promotion clock reading :attr:`Open.since` —
must therefore weigh :attr:`ApiHealthGate.evidence_is_stale` (or
:meth:`ApiHealthGate.seconds_since_last_report`) alongside the state.  "Open,
last 5xx four seconds ago" and "Open, no traffic at all since the trip four
hours ago" are the same state and very different facts; promoting the second
one to L2 pages a human about a provider nothing has spoken to since.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING

from shared.invocation_outcome import OK, InvocationOutcome, ServerError

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


class _Signal(Enum):
    """What one outcome says about the provider — see :meth:`ApiHealthGate._classify`."""

    FAILURE = 'failure'
    SUCCESS = 'success'
    NEUTRAL = 'neutral'


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
        # Monotonic stamp of the last report, kept OUTSIDE the deque so it
        # survives eviction and the close-time clear.  It is what tells a
        # time-sensitive consumer whether a degraded state still has live
        # evidence behind it — see `seconds_since_last_report`.
        self._last_report_monotonic: float | None = None

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

        Under concurrent reporters the return is the state as of RETURN time,
        not a snapshot taken at this report's own transition: a reporter parked
        in a slow forensics write while another reporter's 5xx trips the gate
        gets back the trip.  That direction is deliberate — handing a caller a
        superseded ``Closed`` during a known outage is the silent un-throttle
        this module exists to prevent.

        Args:
            outcome: The classified invocation outcome.  ``ServerError`` counts
                as a provider failure; ``OK`` as a provider success; every
                other variant is neutral (see :meth:`_classify`).
            task_id: The reporting task, or None for task-less invocations.
            account: Account name, recorded on the forensics row.
            role: Agent role, recorded on the forensics row.
        """
        now = self._monotonic_clock()
        self._last_report_monotonic = now
        self._evict(now)

        signal = self._classify(outcome)
        self._samples.append(_Sample(ts=now, is_failure=signal is _Signal.FAILURE, task_id=task_id))

        stats = self._stats()
        # State mutation happens here, before any await: on a single-threaded
        # event loop that makes the transition atomic, so two concurrent
        # reporters cannot interleave mid-transition and no lock is needed.
        self._state = self._next_state(signal, stats)

        if isinstance(outcome, ServerError):
            # Strictly AFTER the transition: `state_after` must be accurate,
            # and the in-memory state stays authoritative regardless of how
            # long this I/O takes or whether it succeeds at all.
            await self._write_api_error_row(
                outcome, task_id=task_id, account=account, role=role, stats=stats
            )

        return self._state

    async def _write_api_error_row(
        self,
        outcome: ServerError,
        *,
        task_id: str | None,
        account: str,
        role: str,
        stats: GateStats,
    ) -> None:
        """Record one ``api_error`` account event for forensics.

        Best-effort by contract.  These rows are evidence, never control
        state: the in-memory state machine is authoritative, so a full disk or
        a locked database must never fail the invocation that reported the 5xx
        and must never suppress a trip — which would hide the very outage the
        gate exists to surface.  Mirrors ``UsageGate._write_cost_event``
        (usage_gate.py:1663-1680).

        The failure is swallowed but LOGGED at WARNING: losing forensics is
        worth an operator's attention even though it is not worth an
        exception.  (Logging at WARN+ is also what keeps this handler off the
        silent-fallthrough gate's signature (b).)

        A gate with no cost store is fully functional — xi can wire the state
        machine before the store, and the tests drive it without one.
        """
        if self._cost_store is None:
            return
        details = json.dumps(
            {
                'status': outcome.status,
                'task_id': task_id,
                'role': role,
                'state_after': type(self._state).__name__,
                'failures': stats.failures,
                'completions': stats.completions,
                'distinct_tasks': stats.distinct_tasks,
                'failure_rate': stats.failure_rate,
            }
        )
        try:
            await self._cost_store.save_api_error_event(
                account_name=account,
                project_id=self._project_id,
                run_id=self._run_id,
                details=details,
                created_at=self._wall_clock().isoformat(),
            )
        except Exception as exc:
            logger.warning(
                'ApiHealthGate forensics write failed for %s (status %d, task %s) — '
                'gate state is unaffected: %s',
                account,
                outcome.status,
                task_id,
                exc,
            )

    @staticmethod
    def _classify(outcome: InvocationOutcome) -> _Signal:
        """Reduce an invocation outcome to what it says about the PROVIDER.

        Three-way, and single-sourced here so a variant added to
        ``InvocationOutcome`` later lands in exactly one place:

        * ``ServerError`` -> FAILURE.  The only outcome that is evidence the
          provider is degraded; every other failure mode has its own handler.
        * ``OK`` -> SUCCESS.  Evidence the provider is serving.
        * everything else -> NEUTRAL (CapHit, NearCap, AuthFailed,
          CliLocalError, ModelNotFound, ZeroOutputWedge, Failure).

        Neutral is not "ignored".  A neutral outcome still counts in the rate
        DENOMINATOR, because C4's rate is 5xx per *completed invocation* and a
        cap hit or a local wedge is a completed invocation.  Dropping them
        would inflate the measured 5xx rate until a fleet whose problem is
        entirely local tripped the provider breaker and halted itself.

        But a neutral outcome must not touch the probe streak in either
        direction: a CapHit is no evidence the provider recovered (so it must
        not advance the streak toward closing) and none that it is still down
        (so it must not reset one).
        """
        if isinstance(outcome, ServerError):
            return _Signal.FAILURE
        if isinstance(outcome, OK):
            return _Signal.SUCCESS
        return _Signal.NEUTRAL

    def _next_state(self, signal: _Signal, stats: GateStats) -> GateState:
        """Pure transition function: (current state, signal, window stats) -> next state."""
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

        # --- Degraded (Open or Probing): watch for recovery ---------------
        if signal is _Signal.SUCCESS:
            streak = 1
            if isinstance(self._state, Probing):
                streak = self._state.consecutive_successes + 1
            if streak >= self.close_after_successes:
                return self._close(stats)
            return Probing(
                since=self._degraded_since(),
                stats=self._degraded_stats(),
                consecutive_successes=streak,
            )

        if signal is _Signal.FAILURE:
            # A failed probe means the outage never ended: back to Open, streak
            # discarded (dropping Probing IS the reset — there is no counter to
            # zero), evidence refreshed from the live window.
            #
            # `since` is deliberately NOT restamped.  It measures the true
            # CONTINUOUS outage, which is what omicron's
            # `max_open_before_l2_hours` promotion clock reads.  Restamping on
            # every flap would let a provider that fails one probe every 90
            # minutes sit Open forever and never promote to L2 — exactly the
            # storm-escape decision 12 exists to prevent.
            return Open(since=self._degraded_since(), stats=stats)

        # Neutral outcome while degraded — see _classify's note: no evidence
        # either way about the provider, so the state (and any probe streak
        # inside it) is left exactly as it was.
        return self._state

    def _close(self, stats: GateStats) -> GateState:
        """Transition to Closed, clearing the window.

        Clearing is load-bearing, not tidiness.  At the moment the closing
        probe lands, the failures that CAUSED the trip are still inside
        ``window_secs`` — so without the clear the very next 5xx would find a
        full burst waiting for it and instantly re-trip, and the breaker would
        oscillate for a whole window after every recovery.  Same trap
        ``Scheduler.resume()`` documents when it clears ``_blocked_transitions``
        (scheduler.py:1862-1867).

        Requiring fresh post-close evidence makes a re-trip mean "the provider
        is down again" rather than "the provider was down a minute ago".

        The cost, stated plainly because it is real: under a SUSTAINED partial
        outage (say 60% failing), two lucky consecutive successes close the
        gate, un-throttle the fleet AND discard the accumulated evidence, so
        re-tripping needs a fresh full burst — ``min_failures`` across
        ``min_distinct_tasks`` at ``failure_rate_threshold`` — during which the
        fleet runs unthrottled against a provider that never recovered.
        ``test_a_still_degraded_provider_can_flap_closed_on_two_lucky_probes``
        pins that behaviour so it stays a visible trade-off rather than an
        emergent surprise.

        It is accepted here because the failure modes are not symmetric:
        oscillation after a REAL recovery is guaranteed and self-sustaining
        (the trip evidence is always still in window at close time), while a
        flap-closed under partial degradation costs one burst of throughput
        and then re-trips.  The knobs to reach for first if that flap is
        observed in practice are ``close_after_successes`` (PRD-mandated at 2,
        so a PRD change) and, failing that, carrying failure samples across a
        close at reduced weight instead of clearing.
        """
        since = self._degraded_since()
        logger.info(
            'ApiHealthGate CLOSED (provider recovered) after %s — '
            'closing window held %d/%d failures',
            self._wall_clock() - since,
            stats.failures,
            stats.completions,
        )
        self._samples.clear()
        return Closed()

    def _degraded_since(self) -> datetime:
        """The current degraded state's original trip instant."""
        state = self._state
        if isinstance(state, Open | Probing):
            return state.since
        # Unreachable: only called from the degraded branch.  Fall back to now
        # rather than crash an invocation over a state-machine bug.
        logger.warning('_degraded_since called in state %r — falling back to now', state)
        return self._wall_clock()

    def _degraded_stats(self) -> GateStats:
        """The evidence captured at the original trip, carried across probes."""
        state = self._state
        if isinstance(state, Open | Probing):
            return state.stats
        logger.warning('_degraded_stats called in state %r — falling back to live window', state)
        return self._stats()

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
        """Drop samples older than ``window_secs`` from the left of the deque.

        Mirrors ``Scheduler._blocked_transitions`` (scheduler.py:1551): a deque
        ordered by monotonic timestamp, trimmed from the left.  Monotonic
        rather than wall-clock so an NTP step cannot retroactively empty or
        inflate the window mid-outage.

        Eviction is driven by REPORTS, not by a timer: an idle gate keeps
        stale samples until the next report arrives.  No decision taken INSIDE
        this class is affected, because :meth:`report` always evicts before it
        evaluates — but decisions taken OUTSIDE it, off :meth:`state`, are:
        a long-idle gate reflects the last report, not the passage of time,
        and a degraded one stays degraded on evidence that has fully aged out.
        :attr:`evidence_is_stale` is what exposes that to those callers; see
        the module docstring.

        The comparison is strict (``<``), so a sample exactly ``window_secs``
        old is still in window; ``<=`` would silently shorten every window by
        one sample at the boundary.
        """
        cutoff = now - self.window_secs
        while self._samples and self._samples[0].ts < cutoff:
            self._samples.popleft()

    def _stats(self) -> GateStats:
        """Summarise the current window.

        Three accounting choices, each with a consequence worth stating:

        * **Denominator = every sample**, not just failures and successes.
          C4's rate is "5xx / completed invocations", and a CapHit or a local
          wedge IS a completed invocation.  Excluding them would inflate the
          measured 5xx rate, so a fleet that is sick LOCALLY would trip the
          provider breaker and halt itself.  Counting them makes the rate
          conservative: local failures dilute it rather than amplify it.

        * **Numerator = ServerError only.**  The gate exists to detect
          provider degradation; every other failure mode has its own handler.

        * **Breadth = distinct non-None task_ids among FAILURES only.**  It
          answers "is this provider-wide or one bad task", so it must not
          count tasks that merely succeeded.  A None task_id (watcher/steward
          invocations) cannot attest breadth and is excluded here — but note
          it is NOT excluded from ``failures`` above: dropping it would
          under-report a real outage, while folding all Nones into one
          synthetic task would let a burst of watcher failures fake breadth.
        """
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

        Deliberately does NOT evict or otherwise advance the machine: a read
        must not mutate, and there is no state this could reach that a
        successful probe would not reach more honestly.  The consequence is
        that a long-idle gate reports the last REPORTED state, not the passage
        of time — pair this with :attr:`evidence_is_stale` if the decision you
        are taking depends on how fresh that state is (module docstring).
        """
        return self._state

    @property
    def is_degraded(self) -> bool:
        """True while the provider is Open OR Probing.

        Every C4 open-state consumer must keep throttling while probing.
        Exposing one boolean here makes "forgot to handle Probing" — a silent
        un-throttle mid-outage — unrepresentable at the call sites.

        Throttling on stale evidence is the safe direction (it costs a little
        throughput until the first probe lands and closes the gate), so this
        flag is intentionally NOT staleness-aware.  Decisions that are not
        safe in that direction — escalating, paging, promoting to L2 — must
        consult :attr:`evidence_is_stale` too.
        """
        return not isinstance(self._state, Closed)

    def seconds_since_last_report(self) -> float | None:
        """Monotonic seconds since the last :meth:`report`, or None if never reported.

        The age of the freshest evidence behind :meth:`state`.  Monotonic, so
        an NTP step cannot make a degraded gate look freshly-fed or
        hours-abandoned.
        """
        if self._last_report_monotonic is None:
            return None
        return self._monotonic_clock() - self._last_report_monotonic

    @property
    def evidence_is_stale(self) -> bool:
        """True when no report has landed within the last ``window_secs``.

        The gate is report-driven: nothing ages out and nothing closes while
        the fleet is silent, and a tripped gate CAUSES silence (dispatch
        throttles to a single probe; the queue may be idle anyway).  So
        ``Open`` on its own does not distinguish "the provider is failing us
        right now" from "the provider failed us hours ago and nothing has
        asked it since".

        Consult this before any escalation that measures time-spent-open —
        chiefly the ``max_open_before_l2_hours`` promotion clock over
        :attr:`Open.since`.  Promoting a stale-evidence gate to L2 pages a
        human about a provider that may have recovered unobserved.

        Meaningful while :attr:`is_degraded`; a Closed gate is making no claim
        that needs backing, and a never-reported gate reads True because it
        has no evidence at all.
        """
        idle = self.seconds_since_last_report()
        return idle is None or idle >= self.window_secs
