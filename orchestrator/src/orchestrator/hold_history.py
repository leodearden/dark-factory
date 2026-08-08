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
import statistics
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cycle guard (rebase_cost_readout.py idiom)
    from orchestrator.event_store import EventStore

logger = logging.getLogger(__name__)

_ACQUIRED = 'lock_acquired'
_RELEASED = 'lock_released'
_SERVICE_RESTART = 'service_restart'

#: Event types :func:`iter_hold_spans` reads.  Everything else in the stream is
#: skipped without touching the open-span map.
_INTERESTING = frozenset({_ACQUIRED, _RELEASED, _SERVICE_RESTART})

#: Holds per module kept in the rolling window.  PARKING_MODEL_REPORT.md:116-126
#: measures the *last 10* holds; a longer window drags in holds that stopped
#: being representative, a shorter one is noise.
DEFAULT_WINDOW = 10

#: Fewest pooled samples that will produce a prediction at all.  Below this the
#: predictor REFUSES (returns None) rather than answer from one or two holds.
DEFAULT_MIN_SAMPLES = 3


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


def _clean_modules(modules: Iterable[str]) -> list[str]:
    """Non-empty string module keys, in order — the live feed's input guard."""
    return [m for m in modules if isinstance(m, str) and m]


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


#: One open hold per (task_id, module) -> its start, in POSIX seconds.  Both the
#: durable seed and the live feed keep one of these; the two ``_apply_*``
#: helpers below are the ONLY things that mutate it (INV-5), so the seed and the
#: feed cannot drift on what an acquire or a release means.
OpenSpans = dict[tuple[str, str], float]


def _apply_acquire(
    open_spans: OpenSpans, task_id: str, modules: Iterable[str], at: float,
) -> Iterator[HoldSpan]:
    """Open a span per module, force-closing a DOUBLE-ACQUIRE at *at*.

    PARKING_MODEL_REPORT.md:101.  The prior span is yielded, not discarded — it
    really did block other tasks — and the new span starts at *at* rather than
    keeping the old start, which would make the eventual release re-count the
    prefix just yielded.
    """
    for module in modules:
        key = (task_id, module)
        prior_start = open_spans.get(key)
        if prior_start is not None:
            yield HoldSpan(task_id, module, prior_start, at, truncated=True)
        open_spans[key] = at


def _apply_release(
    open_spans: OpenSpans, task_id: str, modules: Iterable[str], at: float,
) -> Iterator[HoldSpan]:
    """Close the named modules' spans, ignoring ORPHAN-RELEASEs.

    Only the modules NAMED are closed: ``release_subset`` emits a partial
    release for the modules a plan refinement narrowed away while the task keeps
    holding the rest (scheduler.py:6954-6959).  A release with no open span is
    an orphan (3,594 DF / 5,780 reify in the measured trace) and is skipped —
    the ``if task_id in open_acquires`` guard from analyze_modules.py:145.
    """
    for module in modules:
        start = open_spans.pop((task_id, module), None)
        if start is None:
            continue
        yield HoldSpan(task_id, module, start, at)


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

        # The acquire/release rules themselves live in the two module-level
        # helpers, shared verbatim with HoldHistory's live feed (INV-5).
        if event_type == _ACQUIRED:
            yield from _apply_acquire(open_spans, task_id, _modules_of(r), at)
        else:
            yield from _apply_release(open_spans, task_id, _modules_of(r), at)

    yield from _drain_boundaries_through(float('inf'))

    if open_spans:
        # END OF STREAM.  Unlike an era boundary, nothing here observed an end
        # — closing these at "now" would invent a duration for a hold whose
        # termination was never recorded, so they are dropped outright.
        logger.debug(
            'hold_history: %d span(s) still open at end of stream — dropped (no observed end)',
            len(open_spans),
        )


class HoldHistory:
    """Rolling per-module lock-hold durations, and the median predictor over them.

    One bounded :class:`collections.deque` per module key, ``maxlen=window``, so
    eviction is a property of the container rather than something every read
    path has to remember to slice.  Module keys are the depth-coarsened path
    prefixes ``Scheduler._get_modules`` produces (scheduler.py:7418-7455) and are
    matched EXACTLY — no prefix or parent rollup, because the lock layer already
    treats those as distinct keys.

    The window is keyed by module alone, deliberately not by (task, module): the
    measured signal (PARKING_MODEL_REPORT.md:116-126) is "how long do holds on
    THIS module tend to run", and per-task keying would shatter it into windows
    of one or two samples that could never clear ``min_samples``.
    """

    def __init__(
        self,
        *,
        window: int = DEFAULT_WINDOW,
        min_samples: int = DEFAULT_MIN_SAMPLES,
    ) -> None:
        self._window = int(window)
        self._min_samples = int(min_samples)
        self._samples: dict[str, deque[float]] = {}
        self._seeded = False
        #: Holds open RIGHT NOW, fed by :meth:`observe_acquired` /
        #: :meth:`observe_released`.  Separate from the seed's own transient map:
        #: the seed replays history that has already ended, this tracks holds
        #: still running, which is what :meth:`predicted_remaining` reads.
        self._open: OpenSpans = {}

    # --- seeding from durable history -------------------------------------

    def seed_from_event_store(self, event_store: EventStore) -> int:
        """Seed the windows from the durable event log.  Returns spans recorded.

        Reads the three event types :func:`iter_hold_spans` cares about through
        ``fetch_events_by_type_all_runs`` — the run-AGNOSTIC fetch, so a restart
        does not amnesia away the history the predictor exists to use — and
        merges them back into ONE stream sorted by row ``id``.

        The merge is the load-bearing part.  Each type comes back as its own
        list, so concatenating them would place every release after every
        acquire: T4's double-acquire and the ``service_restart`` boundary would
        then resolve against the wrong neighbours and yield durations that never
        happened.  ``id`` is the store's own insertion order and is what
        ``fetch_events_by_type_all_runs`` already sorts each list by, so a merge
        on it restores the original stream exactly.

        SEED-ONCE.  ``finish_startup`` is not guaranteed to run exactly once
        across a reconfigure, and a second seed would replay the same spans into
        bounded windows — halving each module's effective history while leaving
        the medians looking plausible.  A silent corruption is worse than a
        missed refresh, so the second call is a no-op returning 0.

        FAIL-SOFT.  Any exception is logged and swallowed, returning 0, matching
        event_store.py's own read-path convention.  This runs inside
        ``finish_startup``: propagating would turn an unreadable history — a
        degradation :meth:`predicted_hold`'s refusal already handles correctly —
        into a failure to start the scheduler at all.

        The seed-once flag is set BEFORE the read, deliberately: spans are
        recorded as they stream, so a mid-stream failure leaves a PARTIAL seed
        that a retry would double-count.  One partial history that refuses until
        the live feed refills it beats a silently doubled one.
        """
        if self._seeded:
            return 0
        self._seeded = True
        try:
            rows: list[dict] = []
            for event_type in (_ACQUIRED, _RELEASED, _SERVICE_RESTART):
                rows.extend(event_store.fetch_events_by_type_all_runs(event_type))
            rows.sort(key=lambda r: r.get('id') or 0)

            recorded = 0
            for span in iter_hold_spans(rows):
                self.record_span(span)
                recorded += 1
        except Exception:
            logger.warning('hold_history: seed from event store failed', exc_info=True)
            return 0
        logger.info(
            'hold_history: seeded %d span(s) across %d module(s) from %d event row(s)',
            recorded, len(self._samples), len(rows),
        )
        return recorded

    # --- feeding ----------------------------------------------------------

    def record(self, module: str, duration: float) -> None:
        """File one observed hold duration (seconds) against *module*."""
        if not module:
            return
        window = self._samples.get(module)
        if window is None:
            window = self._samples[module] = deque(maxlen=self._window)
        window.append(float(duration))

    def record_span(self, span: HoldSpan) -> None:
        """File a :class:`HoldSpan` — the shape :func:`iter_hold_spans` yields.

        ``truncated`` spans ARE recorded.  Their end was imposed, not fabricated,
        so the duration is a real lower bound on a real hold
        (PARKING_MODEL_REPORT.md:102: "the lock did block others until then").
        Dropping them would bias the window toward short, tidy holds — precisely
        the ones least worth predicting.
        """
        self.record(span.module, span.duration)

    # --- the live in-process feed -----------------------------------------

    def observe_acquired(self, task_id: str, modules: Iterable[str], *, at: float) -> None:
        """Note that *task_id* took the lock on *modules* at *at* (POSIX seconds).

        Routed through the same ``_apply_acquire`` the durable seed uses, so a
        double-acquire is force-closed and recorded here exactly as it would be
        when replayed off the event log (INV-5).
        """
        for span in _apply_acquire(self._open, str(task_id), _clean_modules(modules), float(at)):
            self.record_span(span)

    def observe_released(self, task_id: str, modules: Iterable[str], *, at: float) -> None:
        """Note that *task_id* released *modules* at *at* — records the durations.

        Only the modules NAMED are closed (partial releases are real), and a
        release with no open span is an orphan and records nothing.  Both rules
        come from the shared ``_apply_release``.
        """
        for span in _apply_release(self._open, str(task_id), _clean_modules(modules), float(at)):
            self.record_span(span)

    # --- reading ----------------------------------------------------------

    def _pooled(self, modules: Iterable[str]) -> list[float]:
        """Every sample from every named module, deduplicating the module list.

        A task whose paths coarsen to the same module twice must not have that
        module's window counted twice — that would silently double its weight in
        the pooled median.
        """
        pooled: list[float] = []
        for module in dict.fromkeys(modules):
            pooled.extend(self._samples.get(module, ()))
        return pooled

    def sample_count(self, modules: Iterable[str]) -> int:
        """How many samples back a :meth:`predicted_hold` over *modules*.

        Exposed for tests and for task η's diagnostics, which need to say WHY a
        prediction was refused rather than just that it was.
        """
        return len(self._pooled(modules))

    def predicted_hold(self, modules: Iterable[str]) -> float | None:
        """Median hold (seconds) across the pooled windows of *modules*.

        POOLED, not a median-of-medians: a module with 10 samples should weigh
        more than one with 3, because it is the better-evidenced key.  The gate
        below counts the pooled samples for the same reason — it counts what the
        answer would actually be made from, so three modules with one sample
        each admit while two refuse.

        **None means NO PREDICTION, and callers must refuse in turn — never
        substitute a default.**  Returning 0.0 (or a global mean, or a
        configured "typical" hold) would read to every caller as a confident
        "this hold is instant", which is strictly worse than an admitted
        absence: a wrong number propagates silently, a None cannot.  PRD
        :459-461: "An empty history must refuse, not admit — a predicate that
        accepts the empty case certifies structure, not capability."

        Note the asymmetry this protects: a 0.0 *sample* is an observation (a
        module whose holds really are instant), while a 0.0 *prediction* from no
        samples is a fabrication.  The gate is on the sample COUNT, never on the
        value, so the two can never be confused.

        ``min_samples`` is a constructor parameter, deliberately NOT read from
        config here — task η owns the ``backfill_min_samples`` leaf and this
        module must stand alone without it.
        """
        pooled = self._pooled(modules)
        if len(pooled) < self._min_samples:
            return None
        return float(statistics.median(pooled))

    def open_modules(self, task_id: str) -> list[str]:
        """Modules *task_id* is holding right now, in acquire order."""
        wanted = str(task_id)
        return [module for (tid, module) in self._open if tid == wanted]

    def predicted_remaining(self, task_id: str, *, now: float) -> float | None:
        """Seconds *task_id* is predicted to keep holding its locks, or None.

        ``max(0.0, predicted_hold(open modules) - elapsed)``, where *elapsed*
        runs from the task's EARLIEST open hold — it has been blocking others
        since its first acquire, and measuring from the latest would under-count
        the elapsed time and over-state how much is left.  That is the
        optimistic direction, and the one that hurts: it keeps a waiting task
        waiting.

        0.0 and None are DIFFERENT answers and callers must keep them apart.
        0.0 means "predicted to have finished already, and hasn't" — a live,
        actionable fact about an overdue holder.  None means there is no
        prediction at all: the task holds nothing, or its modules are below
        ``min_samples``.  Collapsing the overdue case into None would erase the
        one signal a waiting caller most needs; returning a negative number
        would read as time credit that does not exist, hence the floor.
        """
        wanted = str(task_id)
        starts = [start for (tid, _module), start in self._open.items() if tid == wanted]
        if not starts:
            return None
        predicted = self.predicted_hold(self.open_modules(wanted))
        if predicted is None:
            return None
        return max(0.0, predicted - (float(now) - min(starts)))
