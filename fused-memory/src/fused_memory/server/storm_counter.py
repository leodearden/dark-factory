"""Shared rolling-window burst detector (INV-4 storm counters, INV-5 single home).

The body below — append ``(now, label)``, prune to the window, count, compare
to the threshold, then rate-limit to one fire per window, and report the
DISTINCT labels seen in the window — is the established storm-counter pattern
first written as ``reconciliation/harness.py::_record_placeholder_finding_drop``
and since reproduced in ``harness._dead_owner_suppressions`` and
``server/markup_tripwire.MarkupStormCounter``. Task 3088 extracted it here so a
fourth consumer reuses rather than re-copies it (INV-5).

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

    State is PROCESS-LOCAL and resets on restart, like every other in-process
    storm counter in this codebase: the counter exists to catch a live burst,
    not to keep durable statistics. It is also per-instance, so no state bleeds
    between servers (or between tests).

    Not thread-safe by construction; the callers run on a single event loop and
    :meth:`record` never awaits.
    """

    def __init__(self, time_provider: Callable[[], float] = time.time) -> None:
        self._now = time_provider
        self._events: deque[tuple[float, str | None]] = deque()
        self._last_fire_ts: float | None = None

    def record(
        self,
        *,
        threshold: int,
        window_seconds: float,
        label: str | None = None,
    ) -> dict[str, Any] | None:
        """Record one event; return a storm summary iff a burst just fired.

        *threshold* and *window_seconds* are read fresh on every call so a
        consumer backed by a green-tier config leaf can pass live values (see
        the module docstring's RELOAD SAFETY note).

        *label* is what the event should be attributed to, or ``None`` when the
        caller could not resolve one. Unlabelled events still count toward the
        burst — there is simply nothing to name them against.

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
        now = self._now()

        # Append, then prune. The window is half-open: an event aged exactly
        # window_seconds is already out.
        self._events.append((now, label))
        cutoff = now - window_seconds
        while self._events and self._events[0][0] <= cutoff:
            self._events.popleft()

        count = len(self._events)
        if count < threshold:
            return None

        # Threshold crossed — apply the per-window rate limit.
        if self._last_fire_ts is not None and (now - self._last_fire_ts) < window_seconds:
            return None

        self._last_fire_ts = now
        return {
            'count': count,
            'threshold': threshold,
            'window_seconds': window_seconds,
            # Unlabelled events still count toward the burst, but there is
            # nothing to escalate them against, so they are simply not named.
            'labels': sorted({label for _, label in self._events if label is not None}),
        }
