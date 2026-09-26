"""The burst escape and the injected-sink discipline a boundary guard owes (INV-4).

A guard that sits on every tool call and quietly ABSORBS a defect owes its
operator two things, and they are the same two for every such guard: a way to
learn that the absorption is happening at scale, and a channel to say so on
that never turns a working guard into an outage of its own. Both are mechanism
rather than policy — WHICH outcomes are counted, and what a burst is called,
are the guard's decisions and stay declared in the guard.

WHY THIS IS A MODULE AND NOT A BASE CLASS. The two guards that need it
(``mcp_markup_middleware``, ``uuid_prefix_guard``) share no other behaviour —
different detectors, different matrices, different refusal shapes — so a shared
superclass would be an inheritance relationship carrying one collaborator's
worth of state. Composition keeps the coupling to exactly what is shared, and
keeps this testable without a FastMCP server (heuristic 9).

WHY IT EXISTS AT ALL. ``uuid_prefix_guard`` shipped with ~120 lines that were
near-verbatim copies of ``mcp_markup_middleware``'s, including one 34-line
contiguous identical block covering the whole of ``_record_storm``. That is the
lock-step duplication INV-5 forbids and the classic drift shape: a fix to the
dormant-counter sweep or to the summary's key set would have had two homes and
only one of them would have been edited. This module is the ONE home.
``uuid_prefix_guard`` composes it today; migrating ``mcp_markup_middleware``
onto it is the one obvious follow-up, and until that lands the two bodies still
differ (that module is outside the extracting task's declared file scope).

``StormCounter`` itself is untouched and gains no new mode. This is a
CONSUMER of it — the per-key dict, the dormant sweep and the summary shape a
boundary guard needs — not a fourth counting policy.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from shared.storm_counter import StormCounter

__all__ = [
    'BoundaryStormEscape',
    'Sink',
    'call_sink',
]

logger = logging.getLogger(__name__)


#: An injected emitter. Either channel may be a plain function OR an ``async
#: def``: the queue and escalation machinery a registration site wires these to
#: is largely async in this repo, so both shapes are legitimate things to be
#: handed and :func:`call_sink` accepts either.
Sink = Callable[[dict[str, Any]], Any | Awaitable[Any]]


async def call_sink(sink: Sink, record: dict[str, Any], channel: str, *, owner: str) -> Any:
    """Invoke one injected sink, AWAITING an async emitter, and never raise.

    Calling an ``async def`` emitter without awaiting it queues NOTHING while
    handing back a coroutine that looks like a result — an escalation that
    reports no id, a fact stream that silently holds nothing — and the sole
    trace would be a bare ``coroutine was never awaited`` RuntimeWarning. That
    is precisely the silent fail-soft these guards exist to end, committed by
    the guard itself, so an awaitable is awaited rather than trusted to be a
    value.

    Never raises. A sink runs AFTER its call's outcome is already decided, so
    both channels are purely ADDITIVE: a sink outage costs an operator
    visibility rather than turning a working guard into an outage of its own.
    Logged via ``logger.exception``, never swallowed.

    *owner* names the guard in that log line, because this module is shared and
    "the fact sink failed" is not actionable while "uuid prefix guard: the fact
    sink failed" is.
    """
    try:
        result = sink(record)
        if inspect.isawaitable(result):
            result = await result
    except Exception:
        logger.exception(
            '%s: the %s sink failed for %r; the outcome stands',
            owner, channel, record.get('error_type') or record.get('fact'),
        )
        return None
    return result


class BoundaryStormEscape:
    """Count one guard's absorbed outcomes per ``(project, outcome)``; escalate a burst.

    *owner* names the guard in this module's log lines. *error_type* is how a
    fired burst names itself to the escalation sink, and *log_event* is the
    greppable prefix of the operator-facing ERROR — both are the composing
    guard's vocabulary, which is why they are supplied rather than derived.

    *threshold* and *window_seconds* are PUBLIC and read at every
    :meth:`record`, never captured into the counters. That is ``StormCounter``'s
    reload-safety contract: a registration site whose numbers come from a
    green-tier config leaf can rebind them live.

    The counters are held PER INSTANCE, so no burst state bleeds between
    servers, or between tests in one process.
    """

    def __init__(
        self,
        *,
        owner: str,
        error_type: str,
        log_event: str,
        escalation_sink: Sink | None = None,
        threshold: int = 3,
        window_seconds: float = 3600.0,
        time_provider: Callable[[], float] = time.time,
    ) -> None:
        self.threshold = threshold
        self.window_seconds = window_seconds
        self._owner = owner
        self._error_type = error_type
        self._log_event = log_event
        self._escalation_sink = escalation_sink
        self._time_provider = time_provider
        # ONE COUNTER PER KEY, not one counter with a composed label. MEASURED
        # (``mcp_markup_middleware``): StormCounter holds a single deque and a
        # single _last_fire_ts, so its count spans EVERY event in the window
        # regardless of label — a label buys per-key ATTRIBUTION, never a
        # per-key THRESHOLD. Pooling would fire an alarm naming a project or an
        # outcome that never burst, and an operator sent chasing a burst that
        # did not happen learns to ignore the alarm.
        self._counters: dict[str, StormCounter] = {}

    @property
    def tracked_keys(self) -> frozenset[str]:
        """The ``(project, outcome)`` keys currently holding a counter.

        Public because ``project`` is CALLER-SUPPLIED: one counter per key
        means one object per key ever seen, so "dormant keys are evicted" is a
        real bound on this object's memory and belongs in its interface rather
        than being inferred from a private dict. Read-only by construction — a
        frozenset, so a reader cannot evict a live counter by accident.
        """
        return frozenset(self._counters)

    async def record(self, outcome: str, project: str | None) -> dict[str, Any] | None:
        """Count one absorbed *outcome*; escalate and return a summary iff a burst fired.

        Keyed by ``(project, outcome)`` as a single string, which is both the
        key the counter dict is indexed by and the label the summary is
        attributed with — one spelling, so the two cannot disagree.

        Returns ``None`` on the overwhelmingly common non-firing path, so a
        caller folds the result into its response with a single ``is not None``
        rather than having to know what a quiet window looks like.
        """
        key = f'{project}\x1f{outcome}'
        counter = self._counters.get(key)
        if counter is None:
            counter = StormCounter(time_provider=self._time_provider)
            self._counters[key] = counter

        summary = counter.record(
            threshold=self.threshold,
            window_seconds=self.window_seconds,
            label=key,
        )

        # `project` is caller-supplied and one counter per key means one object
        # per key ever seen — so sweep the dormant ones, exactly as the
        # MemoryService consumer StormCounter.prune() was written for.
        for other, dormant in list(self._counters.items()):
            if other != key and dormant.prune(self.window_seconds) == 0:
                del self._counters[other]

        if summary is None:
            return None

        storm = {
            'count': summary['count'],
            'threshold': summary['threshold'],
            'window_seconds': summary['window_seconds'],
            'outcome': outcome,
            'project': project,
        }
        # ERROR and greppable. The summary a guard folds into its response
        # reaches ONLY the caller, which is the one party that already knows
        # something happened, so the operator-facing half cannot ride on it.
        logger.error(
            '%s: %d %s outcome(s) in %ss for project=%r',
            self._log_event, storm['count'], outcome, storm['window_seconds'], project,
        )
        await self._file_escalation(storm)
        return storm

    async def _file_escalation(self, storm: dict[str, Any]) -> None:
        """Hand the burst to the injected sink; never change an outcome.

        No dedup here. Dedup is against an OPEN escalation in the target queue,
        which is knowledge this layer does not have and must not guess at — the
        sink owns it.
        """
        if self._escalation_sink is None:
            return
        await call_sink(
            self._escalation_sink,
            {'error_type': self._error_type, **storm},
            'escalation',
            owner=self._owner,
        )
