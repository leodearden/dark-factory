"""Shared rolling-window burst detector (INV-4 storm counters, INV-5 single home).

The body below — append ``(now, label, key)``, prune to the window, count,
compare to the threshold, then rate-limit to one fire per window, and report the
DISTINCT labels seen in the window — is the established storm-counter pattern
first written as ``reconciliation/harness.py::_record_placeholder_finding_drop``
and since reproduced in ``harness._dead_owner_suppressions`` and
``server/markup_tripwire.MarkupStormCounter``. Task 3088 extracted it (into
``fused_memory.server.storm_counter``) so a fourth consumer reuses rather than
re-copies it (INV-5).

Task 3689 PROMOTED it to ``shared`` when that fourth consumer arrived:
``shared.mcp_markup_middleware`` keys a burst by ``(project, policy_outcome)``,
and ``shared`` is the base layer every other package imports, so it may not
import ``fused_memory``. The old module is now a re-export shim naming this one
as the single home; every existing importer (``server/markup_tripwire``,
``services/memory_service``) and fused-memory's own test suite keep working
unedited, which is what pins the shim honest.

Uses bulk_reset_guard's guard-side injectable-clock convention
(``time_provider`` stored as ``self._now``) so a 3600s window can be tested by
advancing a fake clock instead of sleeping.

RELOAD SAFETY — the one contract difference from ``MarkupStormCounter``, and
the reason this class takes them as arguments rather than storing them:
``threshold`` and ``window_seconds`` are supplied PER :meth:`record` CALL.
``config/reload.py``'s reload-safety rule states that a config value captured
by value at construction cannot observe an in-place reload and must therefore
stay restart-only. A consumer whose threshold comes from a green-tier config
leaf (``mem0_update.storm_threshold``) must read it live on every call, or the
leaf is restart-only in disguise — registered in ``RELOADABLE_FIELDS`` while
silently ignoring reloads. Callers whose thresholds are module constants
(``MarkupStormCounter``) simply pass their stored values through.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from typing import Any


class StormCounter:
    """Rolling-window burst detector over labelled events.

    One event is routine; a BURST within the window is the condition worth
    escalating rather than merely logging (INV-4).

    The label dimension is load-bearing, not decoration: a window that mixed
    two labels into a bare count would let the caller attribute the whole burst
    to whichever event happened to cross the threshold. Labels are opaque
    strings carrying no schema, so the same class serves per-``project_root``
    keying (``MarkupStormCounter``) and per-``agent_id`` keying
    (``MemoryService.update_memory``).

    A counter built with ``count_distinct=True`` gains a SECOND, orthogonal
    dimension: the per-call ``key``. The threshold is then compared against the
    number of DISTINCT non-``None`` keys in the window rather than the raw
    event count, while ``label`` keeps naming the burst independently. The
    motivating consumer is
    ``reconciliation/harness.py::_record_dead_owner_suppression``, which must
    threshold on distinct dead-owner ``instance_id`` values (task 2039 — every
    orphan recovered by ONE restart shares that one owner's instance_id, so a
    single multi-project restart contributes 1 no matter how many projects it
    touched) while still attributing the burst to the distinct ``project_id``
    values it spanned. Neither the single ``label`` nor the middleware's
    one-counter-per-key convention can express that: the former thresholds on
    raw events, and the latter would put the threshold on the NUMBER of counter
    objects, which no individual counter can see.

    ``count_distinct`` is deliberately a CONSTRUCTOR flag while ``threshold``
    and ``window_seconds`` stay per-call. It is a STRUCTURAL mode fixed by the
    call site, not a config leaf, so capturing it at construction cannot go
    stale — the RELOAD SAFETY rule in the module docstring constrains config
    VALUES only.

    State is PROCESS-LOCAL and resets on restart, like every other in-process
    storm counter in this codebase: the counter exists to catch a live burst,
    not to keep durable statistics. It is also per-instance, so no state bleeds
    between servers (or between tests).

    Not thread-safe by construction; the callers run on a single event loop and
    :meth:`record` never awaits.
    """

    def __init__(
        self,
        time_provider: Callable[[], float] = time.time,
        *,
        count_distinct: bool = False,
    ) -> None:
        self._now = time_provider
        self._count_distinct = count_distinct
        self._events: deque[tuple[float, str | None, str | None]] = deque()
        self._last_fire_ts: float | None = None

    def _prune(self, now: float, window_seconds: float) -> int:
        """Drop events older than the window as of *now*; return how many remain.

        The window is half-open: an event aged exactly *window_seconds* is
        already out.

        Always the RAW remaining-event count, even in ``count_distinct`` mode —
        :meth:`record` derives the distinct-key count from the pruned deque
        itself, and :meth:`prune`'s public contract is remaining STATE.
        """
        cutoff = now - window_seconds
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()
        return len(self._events)

    def prune(self, window_seconds: float, now: float | None = None) -> int:
        """Age out stale events without recording one; return the live count.

        The sweep hook for a caller that keys counters by an UNBOUNDED label
        (``MemoryService`` keys them by caller-supplied ``agent_id``): each
        counter self-prunes its own deque, but nothing evicts the counter
        OBJECT, so a long-lived server would accumulate one per label it ever
        saw. A caller can sweep with this and drop whatever returns ``0``.

        Dropping an empty counter is behaviour-preserving, not merely cheap.
        The only other state is ``_last_fire_ts``, stamped while its own event
        was still in the deque — so an empty window implies that fire has
        already aged past the rate limit, and a freshly constructed counter
        would decide identically on the next event.

        The returned count is the number of remaining EVENTS in every mode,
        never the distinct-key count: this is an emptiness probe for sweepers
        that drop whatever returns ``0``, so it must answer "how much state is
        left". A ``count_distinct`` counter holding only ``key=None`` events
        has zero distinct keys but is not empty, and evicting it would discard
        live window state.

        *now* is the optional PER-CALL clock override described on
        :meth:`record`; omitting it reads the constructor-injected
        ``time_provider``, which is the default every existing consumer uses.
        """
        return self._prune(now if now is not None else self._now(), window_seconds)

    def record(
        self,
        *,
        threshold: int,
        window_seconds: float,
        label: str | None = None,
        key: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Record one event; return a storm summary iff a burst just fired.

        *threshold* and *window_seconds* are read fresh on every call so a
        consumer backed by a green-tier config leaf can pass live values (see
        the module docstring's RELOAD SAFETY note).

        *label* is what the event should be attributed to, or ``None`` when the
        caller could not resolve one. Unlabelled events still count toward the
        burst — there is simply nothing to name them against.

        *key* is the DISTINCT-COUNT dimension, and is read only when the
        counter was built with ``count_distinct=True`` (see the class
        docstring). It is orthogonal to *label*: the burst is thresholded on
        distinct keys and NAMED by distinct labels. ``key=None`` is excluded
        from the distinct set entirely — such an event neither counts toward
        the threshold nor blocks it.

        *now* is an optional PER-CALL clock override, as an epoch float. The
        constructor-injected *time_provider* remains the default and is what
        every consumer holding its counter for the process lifetime uses
        (``MarkupStormCounter``, ``MemoryService``). ``now=`` exists for a
        caller that already carries a per-call injected timestamp of its own:
        ``reconciliation/harness.py``'s three storm counters take
        ``now: datetime | None`` on every recording method (the
        ``_finding_recently_resolved`` convention at harness.py:1592) and
        resolve it against ``datetime.now(UTC)`` before calling in. Without
        this door they could only delegate through a mutable clock-holder
        mutated around each call, which is the hand-rolled state INV-5 exists
        to delete. It threads through the window, the pruning and the
        rate-limit arithmetic alike, so an injected instant behaves exactly as
        a provider-stamped one would.

        The count compared to *threshold* is the number of events in the
        window, or — in ``count_distinct`` mode — the number of distinct
        non-``None`` keys among them.

        Returns ``None`` when the count within the window is below *threshold*,
        AND when the threshold is met but a previous fire is still inside the
        window (the rate limit — without it, a runaway emitting hundreds of
        events would escalate hundreds of times for one incident).

        Otherwise stamps the rate-limit timestamp and returns a
        JSON-serializable summary with ``count``, ``threshold``,
        ``window_seconds`` and ``labels`` — the sorted DISTINCT non-``None``
        labels seen in the window, so the caller can attribute the burst
        instead of blaming whichever event crossed the threshold.
        """
        effective_now = now if now is not None else self._now()

        # Append, then prune.
        self._events.append((effective_now, label, key))
        count = self._prune(effective_now, window_seconds)
        if self._count_distinct:
            count = len({k for _, _, k in self._events if k is not None})
        if count < threshold:
            return None

        # Threshold crossed — apply the per-window rate limit.
        if (
            self._last_fire_ts is not None
            and (effective_now - self._last_fire_ts) < window_seconds
        ):
            return None

        self._last_fire_ts = effective_now
        return {
            'count': count,
            'threshold': threshold,
            'window_seconds': window_seconds,
            # Unlabelled events still count toward the burst, but there is
            # nothing to escalate them against, so they are simply not named.
            'labels': sorted({lbl for _, lbl, _ in self._events if lbl is not None}),
        }
