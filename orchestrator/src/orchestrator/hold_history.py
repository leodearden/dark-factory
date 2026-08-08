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

    *era_boundaries* accepts extra boundary timestamps (POSIX seconds) beyond
    the ones this helper derives from the stream itself.
    """
    open_spans: dict[tuple[str, str], float] = {}

    for r in rows:
        event_type = str(r.get('event_type') or '')
        if event_type not in (_ACQUIRED, _RELEASED):
            continue
        at = _parse_ts(r.get('timestamp'))
        if at is None:
            logger.debug('hold_history: unparseable timestamp %r — row dropped', r.get('timestamp'))
            continue
        raw_task_id = r.get('task_id')
        if not raw_task_id:
            continue
        task_id = str(raw_task_id)

        if event_type == _ACQUIRED:
            for module in _modules_of(r):
                open_spans[(task_id, module)] = at
        else:
            for module in _modules_of(r):
                start = open_spans.pop((task_id, module), None)
                if start is None:
                    continue  # ORPHAN-RELEASE — span never opened.
                yield HoldSpan(task_id, module, start, at)
