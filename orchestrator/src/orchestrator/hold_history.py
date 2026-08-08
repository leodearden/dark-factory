"""HoldHistory — per-module lock-hold duration history and predictor (task 3822 / plans/scheduler-dispatch-scoring-and-lock-layer-prd.md task ζ).

The measured evidence (plans/evidence/scheduler-scoring-2026-08-06/
PARKING_MODEL_REPORT.md:116-126) is blunt: static task attributes predict
hold duration *worse than the test-set mean* (global median R² -0.22 DF /
-0.67 reify; tier and tier+width no better).  The **only** signal that works
is per-module hold history — median of the last 10 holds on the task's
modules, R² 0.26 (DF) / 0.68 (reify).  This module is that one predictor.

Two public surfaces:

- :func:`iter_hold_spans` — THE one shared acquire→release pairing helper
  (design invariant INV-5).  Both the durable seed and the in-process feed go
  through it, so their span semantics cannot drift apart.
- :class:`HoldHistory` — a rolling per-module window of observed durations
  with :meth:`HoldHistory.predicted_hold` /
  :meth:`HoldHistory.predicted_remaining` on top.

REFUSAL, NOT FABRICATION
Below ``min_samples`` observations ``predicted_hold`` returns ``None`` — not
0.0, not a global default.  PRD :459-461: "An empty history must refuse, not
admit — a predicate that accepts the empty case certifies structure, not
capability."
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_ACQUIRED = 'lock_acquired'
_RELEASED = 'lock_released'
_SERVICE_RESTART = 'service_restart'

#: Event types :func:`iter_hold_spans` reads.  Everything else in the stream is
#: skipped without touching the open-span map.
_INTERESTING = frozenset({_ACQUIRED, _RELEASED, _SERVICE_RESTART})


@dataclass(frozen=True)
class HoldSpan:
    """One observed module hold: ``task_id`` held ``module`` from ``start`` to ``end``.

    ``start``/``end`` are POSIX seconds.  ``truncated`` marks a span whose end
    was *imposed* rather than observed — a double-acquire force-close or an
    era boundary — as opposed to a clean ``lock_released``.
    """

    task_id: str
    module: str
    start: float
    end: float
    truncated: bool = False

    @property
    def duration(self) -> float:
        """Hold length in seconds, clamped at 0.0.

        ``start``/``end`` stay faithful to the rows they came from; the clamp
        lives here so a clock step (NTP, VM skew) that puts a release before
        its acquire can never contribute a negative sample.
        """
        return max(0.0, self.end - self.start)


def _parse_ts(raw: Any) -> float | None:
    """Parse an event row's ``timestamp`` column into POSIX seconds.

    ``EventStore.emit`` writes ``datetime.now(UTC).isoformat()`` (tz-aware),
    so ``fromisoformat().timestamp()`` is exact.  Returns None — and the
    caller drops the row — for anything unparseable.
    """
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)
    try:
        return datetime.fromisoformat(str(raw)).timestamp()
    except (TypeError, ValueError):
        return None


def _modules_of(row: dict) -> list[str]:
    """Module list from a lock event's ``data`` payload.

    Both lock event payloads carry ``data['modules']`` as a list; the
    single-string coercion mirrors analyze_modules.py:135-136, which has to
    tolerate the same field defensively.
    """
    data = row.get('data') or {}
    if not isinstance(data, dict):
        return []
    modules = data.get('modules') or []
    if isinstance(modules, str):
        modules = [modules]
    if not isinstance(modules, list):
        return []
    return [m for m in modules if isinstance(m, str) and m]


def iter_hold_spans(
    rows: Iterable[dict],
    *,
    era_boundaries: Iterable[float] = (),
) -> Iterator[HoldSpan]:
    """Pair ``lock_acquired`` → ``lock_released`` rows into :class:`HoldSpan` s.

    *rows* must be ONE stream in the store's own ``id`` order (that is what
    ``EventStore.fetch_events_by_type_all_runs`` returns), so a release can
    never be applied before its acquire.  Rows of other event types are
    ignored here.

    Open spans are keyed by **(task_id, module)**, not by task_id alone:
    ``release_subset`` emits a PARTIAL ``lock_released`` naming only the
    modules a plan refinement narrowed away (scheduler.py:6954-6959) while the
    task keeps holding the rest.  Per-(task, module) keying makes that correct
    with no special case, and is the natural granularity for a per-module
    predictor anyway.

    A release with no open span is an ORPHAN-RELEASE (3,594 DF / 5,780 reify
    in the measured trace) and is silently ignored — the
    ``if task_id in open_acquires`` guard from analyze_modules.py:145 carried
    over verbatim.

    ERA BOUNDARIES close every still-open span at the boundary timestamp and
    mark it ``truncated``.  The span is *counted*, not discarded:
    PARKING_MODEL_REPORT.md:102 is explicit that "the lock did block others
    until then", so the observed prefix is a real lower bound.  A ``truncated``
    span is a hold whose end was IMPOSED; a dropped span is a hold whose end was
    never observed at all — the second would have to be fabricated, so it isn't.

    TWO boundary sources, ONE code path (INV-5):

    1. a ``service_restart`` row, closing at its own timestamp;
    2. a **run_id transition** between consecutive rows, closing at the
       PREVIOUS row's timestamp — the last instant the stream shows the lock
       still held.

    (2) is not redundant with (1).  ``service_restart`` has a single emit site
    scoped to a *backing service* (service_restart.py:747-759), so an
    orchestrator crash or systemd restart emits no such row at all — while
    ``fetch_events_by_type_all_runs`` spans runs by construction.  Without the
    run-id boundary, a span stranded at the end of one run stays open until some
    ``service_restart`` days later and reports a multi-day hold that never
    happened: the infinite-hold failure D11 exists to prevent.

    Spans still open when the stream ENDS are dropped, not closed at "now".

    *era_boundaries* accepts extra boundary timestamps (POSIX seconds) beyond
    the ones this helper derives from the stream itself.  They are interleaved
    into the stream in timestamp order (a boundary at or before a row's
    timestamp fires first), and any left over after the last row still fire —
    they are observed instants, exactly like a ``service_restart`` row.
    """
    open_spans: dict[tuple[str, str], float] = {}
    pending_boundaries = sorted(float(b) for b in era_boundaries)
    boundary_i = 0

    def _close_all(at: float) -> Iterator[HoldSpan]:
        """Close EVERY open span at *at*, truncated, and empty the map.

        The one era-boundary closure (INV-5): ``service_restart`` rows, caller-
        supplied boundaries and — from task ζ step-6 — run-id transitions all
        funnel through here, so no boundary source can drift from another.
        """
        for (task_id, module), start in list(open_spans.items()):
            yield HoldSpan(task_id, module, start, at, truncated=True)
        open_spans.clear()

    def _drain_boundaries_through(at: float) -> Iterator[HoldSpan]:
        """Fire every caller-supplied boundary at or before *at*."""
        nonlocal boundary_i
        while boundary_i < len(pending_boundaries) and pending_boundaries[boundary_i] <= at:
            yield from _close_all(pending_boundaries[boundary_i])
            boundary_i += 1

    prev_run_id = ''
    prev_at = 0.0

    for r in rows:
        event_type = str(r.get('event_type') or '')
        if event_type not in _INTERESTING:
            continue
        at = _parse_ts(r.get('timestamp'))
        if at is None:
            logger.debug('hold_history: unparseable timestamp %r — row dropped', r.get('timestamp'))
            continue

        run_id = str(r.get('run_id') or '')
        if run_id and prev_run_id and run_id != prev_run_id:
            # ERA BOUNDARY (run transition).  Close at the PREVIOUS row's
            # timestamp, not this one: the prior run ended somewhere in the gap
            # and `prev_at` is the last instant we have evidence the lock was
            # still held.  Closing at `at` would charge the whole inter-run gap
            # — however many days of it — to a hold nobody observed.
            yield from _close_all(prev_at)
        # A row with no run_id carries the current era forward rather than
        # opening an "unknown" one: firing a boundary on missing data would
        # truncate real spans.
        prev_run_id = run_id or prev_run_id
        prev_at = at

        yield from _drain_boundaries_through(at)

        if event_type == _SERVICE_RESTART:
            # ERA BOUNDARY.  This row's task_id is the *trigger* task that
            # provoked the restart (service_restart.py:747-759), NOT a lock
            # holder — reading it as one would open a phantom span.  Only the
            # timestamp is load-bearing here.
            yield from _close_all(at)
            continue

        raw_task_id = r.get('task_id')
        if not raw_task_id:
            continue
        task_id = str(raw_task_id)

        if event_type == _ACQUIRED:
            for module in _modules_of(r):
                key = (task_id, module)
                prior_start = open_spans.get(key)
                if prior_start is not None:
                    # DOUBLE-ACQUIRE (PARKING_MODEL_REPORT.md:101): force-close
                    # the previous span here.  Re-opening at `at` rather than
                    # keeping `prior_start` matters — keeping it would make the
                    # eventual release re-count the prefix just yielded.
                    yield HoldSpan(task_id, module, prior_start, at, truncated=True)
                open_spans[key] = at
        else:
            for module in _modules_of(r):
                start = open_spans.pop((task_id, module), None)
                if start is None:
                    continue  # ORPHAN-RELEASE — span never opened.
                yield HoldSpan(task_id, module, start, at)

    yield from _drain_boundaries_through(float('inf'))

    if open_spans:
        # END OF STREAM.  Unlike an era boundary, nothing here observed an end
        # — closing these at "now" would invent a duration for a hold whose
        # termination was never recorded, so they are dropped outright.
        logger.debug(
            'hold_history: %d span(s) still open at end of stream — dropped (no observed end)',
            len(open_spans),
        )
