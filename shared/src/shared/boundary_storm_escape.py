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
only one of them would have been edited. This module is the ONE home, and
both guards compose it (``mcp_markup_middleware`` since task 5102).

Counting is delegated to ``shared.storm_counter.KeyedStormCounters``, the one
keyed registry. This module owns only the guard-facing half: the storm record's
shape, the operator-facing ERROR line and the sink filing. It is not a fourth
counting policy.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from shared.storm_counter import KeyedStormCounters

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

#: The keys a storm record (or its sink record) carries of its own. A per-call
#: ``crossing`` fact may not reuse one: it would silently overwrite the burst's
#: own numbers, or its attribution, in the record an operator triages.
_RESERVED_RECORD_KEYS = frozenset({
    'error_type', 'count', 'threshold', 'window_seconds', 'outcome', 'project',
    'callers',
})


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

    *threshold*, *window_seconds* and *time_provider* are PUBLIC and read at
    every :meth:`record`, never captured into the counters. That is
    ``StormCounter``'s reload-safety contract: a registration site whose numbers
    come from a green-tier config leaf can rebind them live, and a test can tune
    a REGISTERED guard without reaching private state.

    *names_callers* is structural. When set, every storm carries ``callers``,
    the distinct per-call ``caller=`` labels of its window. When unset, the
    record has no such slot and a ``caller=`` is a wiring bug. *log_advice* is a
    sentence appended to the ERROR line, telling the operator where to route it.

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
        names_callers: bool = False,
        log_advice: str = '',
    ) -> None:
        self.threshold = threshold
        self.window_seconds = window_seconds
        self.time_provider = time_provider
        self._owner = owner
        self._error_type = error_type
        self._log_event = log_event
        self._log_advice = log_advice
        self._names_callers = names_callers
        self._escalation_sink = escalation_sink
        self._counters: KeyedStormCounters[str] = KeyedStormCounters()

    @property
    def names_callers(self) -> bool:
        """Whether storms carry ``callers``; fixed at construction."""
        return self._names_callers

    @property
    def tracked_keys(self) -> frozenset[str]:
        """The ``(project, outcome)`` keys currently holding a counter.

        Public because ``project`` is CALLER-SUPPLIED: one counter per key
        means one object per key ever seen, so "dormant keys are evicted" is a
        real bound on this object's memory and belongs in its interface rather
        than being inferred from a private dict. Read-only by construction — a
        frozenset, so a reader cannot evict a live counter by accident.
        """
        return self._counters.tracked_keys

    async def record(
        self,
        outcome: str,
        project: str | None,
        *,
        caller: str | None = None,
        crossing: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Count one absorbed *outcome*; escalate and return a summary iff a burst fired.

        Keyed by ``(project, outcome)`` as a single string, so neither a second
        project nor a second outcome can pool into a premature fire. *caller*
        is the event's ATTRIBUTION label, never part of the key: it names who
        was in the window without changing when the window fires.

        *crossing* holds facts about THIS call, which matter only if it is the
        one that crosses the threshold. They are merged into the record after
        its base keys, and none may reuse one of them.

        Returns ``None`` on the overwhelmingly common non-firing path, so a
        caller folds the result into its response with a single ``is not None``
        rather than having to know what a quiet window looks like.

        :raises ValueError: on a *caller* without ``names_callers``, or a
            *crossing* key that collides with the record's own.
        """
        crossing = crossing or {}
        self._require_wiring(caller, crossing)
        summary = self._counters.record(
            f'{project}\x1f{outcome}',
            threshold=self.threshold,
            window_seconds=self.window_seconds,
            label=caller,
            now=self.time_provider(),
        )
        if summary is None:
            return None

        storm: dict[str, Any] = {
            'count': summary['count'],
            'threshold': summary['threshold'],
            'window_seconds': summary['window_seconds'],
            'outcome': outcome,
            'project': project,
            **crossing,
        }
        if self._names_callers:
            storm['callers'] = summary['labels']
        self._log_burst(storm, crossing)
        await self._file_escalation(storm)
        return storm

    def _require_wiring(self, caller: str | None, crossing: Mapping[str, Any]) -> None:
        if caller is not None and not self._names_callers:
            raise ValueError(
                f'caller={caller!r} was passed to a {self._owner} storm escape '
                'built without names_callers=True; its storms carry no callers '
                'slot and would silently drop the attribution. Construct it '
                'with names_callers=True, or drop the caller.'
            )
        collisions = sorted(_RESERVED_RECORD_KEYS.intersection(crossing))
        if collisions:
            raise ValueError(
                f'crossing key(s) {", ".join(map(repr, collisions))} collide with '
                f"the {self._owner} storm record's own keys and would overwrite "
                'them; give the crossing fact a name of its own.'
            )

    def _log_burst(self, storm: dict[str, Any], crossing: Mapping[str, Any]) -> None:
        """ONE ERROR line per burst, greppable by *log_event*.

        The summary a guard folds into its response reaches ONLY the caller,
        which is the one party that already knows something happened, so the
        operator-facing half cannot ride on it. Crossing facts are rendered
        under the record's own key names, so a key read off an escalation
        record greps straight to its line.
        """
        advice = f' — {self._log_advice}' if self._log_advice else ''
        rendered = ' '.join(f'{key}={value!r}' for key, value in crossing.items())
        suffix = f'; crossing call {rendered}' if crossing else ''
        logger.error(
            '%s: %d %s outcome(s) in %ss for project=%r%s%s',
            self._log_event, storm['count'], storm['outcome'],
            storm['window_seconds'], storm['project'], advice, suffix,
        )

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
