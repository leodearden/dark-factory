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
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator
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

#: Age at which an open hold in the LIVE map is treated as bookkeeping residue
#: rather than a running hold (24h).  Deliberately far above any real hold — the
#: orchestrator's own per-role timeouts are tens of minutes — so this can only
#: ever catch a lost release, never truncate the genuinely-overdue holder whose
#: 0.0 :meth:`HoldHistory.predicted_remaining` exists to report.
DEFAULT_STALE_OPEN_SECS = 86400.0

#: Seed wall time above which :meth:`HoldHistory.seed_from_event_store` warns.
#: The scan is unbounded in the age of ``runs.db`` and blocks its caller, so the
#: growth must be visible in the log rather than only in startup latency.
SLOW_SEED_WARN_SECS = 2.0


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
    """Non-empty string module keys, DE-DUPLICATED, in first-seen order.

    The one element-level guard for both feeds.  De-duplication is not cosmetic:
    ``_apply_acquire`` treats a module it has already opened as a DOUBLE-ACQUIRE
    and force-closes the prior span, so a payload naming the same module twice
    would yield a phantom zero-duration ``truncated`` span — fabricating a
    censoring signal on what is actually a clean pair.  ``files_to_modules``
    de-duplicates upstream today, so this is defence against a payload
    reconstructed from JSON rather than a live defect.

    Filtering precedes hashing deliberately: ``modules`` comes off a JSON
    payload and may carry unhashable junk (a nested list), which would make
    ``dict.fromkeys`` raise instead of dropping it.
    """
    return list(dict.fromkeys(m for m in modules if isinstance(m, str) and m))


def modules_of(row: dict) -> list[str]:
    """THE rule for "which modules does this lock/skip event name" (INV-5).

    Public because there is more than one consumer: :func:`iter_hold_spans`
    reads it for span pairing, and ``analyze_modules`` reads it directly for
    its per-module dispatch/skip counters, which spans do not supply.  One
    shared helper rather than a second copy of the coercion in the CLI —
    duplicating it is what INV-5 exists to forbid.

    ``data['modules']`` is written by ``Scheduler._emit_lock_event``
    (scheduler.py:8169) and the skip path (scheduler.py:4932-4941) as a list of
    already-depth-coarsened module keys.  Everything below that is defence
    against a payload reconstructed from JSON: a bare string is COERCED (it is
    unambiguous about which single module it names, so dropping it would throw
    away a real hold), while an unusable shape yields ``[]`` rather than a
    phantom key.

    The element-level filtering and de-duplication are delegated to
    :func:`_clean_modules` rather than re-inlined, so the durable seed, the live
    feed and the CLI cannot disagree about what a module list contains — the
    same INV-5 reason this function is public at all.
    """
    data = row.get('data') or {}
    if not isinstance(data, dict):
        return []
    modules = data.get('modules') or []
    if isinstance(modules, str):
        modules = [modules]
    if not isinstance(modules, list):
        return []
    return _clean_modules(modules)


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
    a rule inherited from ``analyze_modules``' own pairing loop, which task 3869
    then deleted by converging that CLI onto this helper.
    """
    for module in modules:
        start = open_spans.pop((task_id, module), None)
        if start is None:
            continue
        yield HoldSpan(task_id, module, start, at)


def iter_hold_spans(rows: Iterable[dict]) -> Iterator[HoldSpan]:
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
    in the measured trace) and is silently ignored.  ``analyze_modules`` used
    to guard this in its own pairing loop; task 3869 pointed that CLI here
    instead, so the rule now lives in one place.

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

    Both boundary sources are derived from the stream ITSELF; there is
    deliberately no caller-supplied boundary parameter.  An earlier draft had
    one, unused by any caller and therefore untested, and it carried an unvalidated
    ordering subtlety (a boundary falling between two rows fired *after* the
    run-transition closure and silently became a no-op).  Task η re-adds it —
    with tests — if and when a caller materialises.
    """
    open_spans: dict[tuple[str, str], float] = {}

    def _close_all(at: float) -> Iterator[HoldSpan]:
        """Close EVERY open span at *at*, truncated, and empty the map.

        The one era-boundary closure (INV-5): ``service_restart`` rows and
        run-id transitions both funnel through here, so no boundary source can
        drift from another.
        """
        for (task_id, module), start in list(open_spans.items()):
            yield HoldSpan(task_id, module, start, at, truncated=True)
        open_spans.clear()

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
            yield from _apply_acquire(open_spans, task_id, modules_of(r), at)
        else:
            yield from _apply_release(open_spans, task_id, modules_of(r), at)

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
        min_samples: int | Callable[[], int] = DEFAULT_MIN_SAMPLES,
        stale_open_secs: float = DEFAULT_STALE_OPEN_SECS,
    ) -> None:
        self._window = int(window)
        #: The sample floor, as a PROVIDER resolved on every read.  A plain int
        #: becomes a constant closure (and keeps its eager coercion, so a bad
        #: literal still fails loudly here rather than deep inside
        #: :meth:`predicted_hold`); a callable is stored as-is so an owner
        #: whose floor can change under it — e.g. a hot-reloadable config leaf
        #: — stays live without this module ever seeing a config object.
        self._min_samples_source: Callable[[], int]
        if callable(min_samples):
            self._min_samples_source = min_samples
        else:
            fixed = int(min_samples)
            self._min_samples_source = lambda: fixed
        self._stale_open_secs = float(stale_open_secs)
        #: module -> bounded window of ``(duration, truncated)``.  The flag is
        #: carried alongside the duration rather than dropped at the window
        #: boundary: a truncated sample is a right-censored LOWER BOUND, and on
        #: a long-lived event log most seeded samples are censored, so a
        #: prediction made entirely from them deserves to be recognisable as
        #: such.  :meth:`truncated_fraction` is what exposes it.
        self._samples: dict[str, deque[tuple[float, bool]]] = {}
        self._seeded = False
        #: Holds open RIGHT NOW, fed by :meth:`observe_acquired` /
        #: :meth:`observe_released`.  Separate from the seed's own transient map:
        #: the seed replays history that has already ended, this tracks holds
        #: still running, which is what :meth:`predicted_remaining` reads.
        self._open: OpenSpans = {}

    @property
    def min_samples(self) -> int:
        """The floor in force RIGHT NOW — re-resolved on every access.

        Read by :meth:`predicted_hold` per call rather than captured once, so a
        caller whose floor comes from a hot-reloadable leaf does not have to
        rebuild the predictor (or remember to push) after a reload.
        """
        return int(self._min_samples_source())

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

        COST, and why it is NOT capped here.  The scan is the whole lifetime of
        ``runs.db``, unbounded by age or run count, and it runs synchronously
        inside ``finish_startup``.  Measured on the 136 MB production DB (254
        runs, 2026-08-09): 26,169 rows fetched, 17,883 spans replayed, ~4.3 s
        fetch + ~1.2 s replay, 3,113 module keys retained.  Bounding the replay
        to a recent tail was measured and REJECTED: at a 12,000-row tail only
        523 of 1,011 module keys still clear ``min_samples`` — half the
        predictor's coverage — because most of the older rows are unpaired
        acquires that only become samples when an era boundary later in the
        stream force-closes them.  Trading half the coverage of the one
        predictor that works for a few seconds of one-time startup latency is
        the wrong way round.  The seed is instead INSTRUMENTED (rows, spans,
        modules and elapsed are logged, and a slow seed warns) so the growth is
        visible rather than silent.  Bounding it properly needs an age-based
        ``since`` predicate pushed into the SQL — coverage then degrades
        gracefully instead of falling off a cliff — plus an
        ``asyncio.to_thread`` at the ``finish_startup`` call site so the event
        loop is not blocked; both live in files this task does not own.
        """
        if self._seeded:
            return 0
        self._seeded = True
        started = time.perf_counter()
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
        elapsed = time.perf_counter() - started
        logger.info(
            'hold_history: seeded %d span(s) across %d module(s) from %d event row(s) in %.2fs',
            recorded, len(self._samples), len(rows), elapsed,
        )
        if elapsed > SLOW_SEED_WARN_SECS:
            # The scan grows monotonically with the age of runs.db and blocks
            # the caller while it runs.  Say so out loud once per start rather
            # than letting a startup stall accumulate invisibly.
            logger.warning(
                'hold_history: seed scanned %d event row(s) in %.2fs (> %.1fs) — '
                'the scan covers the whole lifetime of the event store and grows '
                'with it; it needs an age-bounded fetch',
                len(rows), elapsed, SLOW_SEED_WARN_SECS,
            )
        return recorded

    # --- feeding ----------------------------------------------------------

    def record(self, module: str, duration: float, *, truncated: bool = False) -> None:
        """File one observed hold duration (seconds) against *module*.

        *truncated* marks the sample as a right-censored lower bound — an end
        that was imposed rather than observed.  It does not change how the
        sample is used by :meth:`predicted_hold`; it is retained so
        :meth:`truncated_fraction` can report how much of the evidence is
        censored.
        """
        if not module:
            return
        window = self._samples.get(module)
        if window is None:
            window = self._samples[module] = deque(maxlen=self._window)
        window.append((float(duration), bool(truncated)))

    def record_span(self, span: HoldSpan) -> None:
        """File a :class:`HoldSpan` — the shape :func:`iter_hold_spans` yields.

        ``truncated`` spans ARE recorded.  Their end was imposed, not fabricated,
        so the duration is a real lower bound on a real hold
        (PARKING_MODEL_REPORT.md:102: "the lock did block others until then").
        Dropping them would bias the window toward short, tidy holds — precisely
        the ones least worth predicting.

        The flag travels WITH the duration into the window rather than being
        discarded here.  Recording a censored lower bound and then making it
        indistinguishable from a clean observation would leave the pooled median
        looking equally well-evidenced either way, with no way for a caller to
        tell — see :meth:`truncated_fraction`.
        """
        self.record(span.module, span.duration, truncated=span.truncated)

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

    def forget(self, task_id: str) -> int:
        """Drop every open hold still recorded for *task_id*.  Returns how many.

        The reconciliation half of the live feed, called from
        ``Scheduler.release`` — the single writer that drops ALL of a task's
        locks — so the open map cannot outlive the lock table it mirrors.
        ``observe_released`` closes only the modules it is NAMED (partial
        releases are real), which means any module the lock table lost track of
        would otherwise stay open for the life of the process.

        That residue is not benign.  :meth:`predicted_remaining` would keep
        answering for the phantom holder, and — its elapsed time only growing —
        would answer a floored **0.0** forever: the "overdue holder" reading,
        which is an actionable live claim, made about a task that released its
        locks hours ago.  A wrong 0.0 is the worst-shaped output this API has,
        because the whole contract turns on callers telling 0.0 from None.

        The residual spans are DROPPED, not recorded.  An entry that survived to
        here means the acquire/release bookkeeping for that key is already
        broken, so its ``start`` may be arbitrarily stale; charging a duration
        against it would file a fabricated sample — exactly what
        :func:`iter_hold_spans` refuses to do for a span still open at
        end-of-stream.  The modules genuinely held were closed for real by the
        ``lock_released`` that precedes this call.
        """
        wanted = str(task_id)
        dropped = [key for key in self._open if key[0] == wanted]
        for key in dropped:
            del self._open[key]
        if dropped:
            logger.debug(
                'hold_history: dropped %d unreleased open hold(s) for task %s: %s',
                len(dropped), wanted, [module for _tid, module in dropped],
            )
        return len(dropped)

    # --- reading ----------------------------------------------------------

    def _pooled_samples(self, modules: Iterable[str]) -> list[tuple[float, bool]]:
        """Every ``(duration, truncated)`` from every named module, deduplicated.

        A task whose paths coarsen to the same module twice must not have that
        module's window counted twice — that would silently double its weight in
        the pooled median.
        """
        pooled: list[tuple[float, bool]] = []
        for module in dict.fromkeys(modules):
            pooled.extend(self._samples.get(module, ()))
        return pooled

    def _pooled(self, modules: Iterable[str]) -> list[float]:
        """Just the durations from :meth:`_pooled_samples` — what the median uses."""
        return [duration for duration, _truncated in self._pooled_samples(modules)]

    def sample_count(self, modules: Iterable[str]) -> int:
        """How many samples back a :meth:`predicted_hold` over *modules*.

        Exposed for tests and for task η's diagnostics, which need to say WHY a
        prediction was refused rather than just that it was.
        """
        return len(self._pooled_samples(modules))

    def truncated_fraction(self, modules: Iterable[str]) -> float | None:
        """Share (0.0-1.0) of the pooled samples for *modules* that are censored.

        A truncated sample's end was IMPOSED — an era boundary or a
        double-acquire force-close — so its duration is a lower bound on the
        real hold, not the hold.  Such samples are counted (see
        :meth:`record_span`), but a prediction made mostly from them is
        systematically biased and a caller deserves to know: on a long-lived
        event log the durable seed produces far more acquires than releases, so
        much of a freshly-seeded window is censored.

        The companion to :meth:`sample_count`: together they let task η say
        whether a prediction is thin, censored, both, or trustworthy — rather
        than only whether one exists.  Returns None when there are no samples at
        all, because a fraction of nothing is not 0.0 (which would read as
        "clean evidence") but no answer, the same refusal
        :meth:`predicted_hold` makes.
        """
        pooled = self._pooled_samples(modules)
        if not pooled:
            return None
        return sum(1 for _duration, truncated in pooled if truncated) / len(pooled)

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
        module must stand alone without it.  It may however be a zero-argument
        PROVIDER rather than a fixed int, in which case it is resolved on every
        call: that is what lets the owner of a hot-reloadable leaf keep the
        floor live (a value captured at construction would freeze it for the
        process era, so an operator's reload would land in ``applied`` and
        change nothing) while this module still learns nothing about config.
        The pull is deliberate — a pushed setter would have to be re-issued at
        every entry point, and :meth:`predicted_remaining` calls this method
        internally, so a push discipline would silently miss that path.
        """
        pooled = self._pooled(modules)
        if len(pooled) < self.min_samples:
            return None
        return float(statistics.median(pooled))

    def open_modules(self, task_id: str) -> list[str]:
        """Modules *task_id* is holding right now, in acquire order."""
        wanted = str(task_id)
        return [module for (tid, module) in self._open if tid == wanted]

    def _sweep_stale_open(self, task_id: str, now: float) -> list[str]:
        """Drop *task_id*'s open holds whose OWN start is older than the ceiling.

        Returns the module names dropped, OLDEST FIRST.  Per-KEY, deliberately
        not per-task: the ceiling asks "is THIS hold bookkeeping residue?",
        which is a fact about one hold, and generalising it to the whole task
        took live sibling holds down with the phantom.  :meth:`forget` answers
        the different question ("drop everything this task holds") for its one
        caller, ``Scheduler.release``, and stays all-or-nothing.

        Staleness is monotone in ``start``, so the swept set is always exactly
        the oldest prefix of the task's holds — the sweep cannot strand a
        newer-but-stale entry.  Sorted rather than left in dict order so the
        operator-facing log names them in a meaningful order.
        """
        stale = sorted(
            (start, key) for key, start in self._open.items()
            if key[0] == task_id and (now - start) > self._stale_open_secs
        )
        for _start, key in stale:
            del self._open[key]
        return [module for _start, (_tid, module) in stale]

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

        STALENESS BACKSTOP.  An open hold older than ``stale_open_secs`` (24h by
        default, far beyond any real hold) is residue from a release that never
        reached the live feed, not a running hold.  Such holds are dropped —
        because the alternative is the permanent 0.0 described above, an
        actionable claim about a task that is not holding anything.

        The sweep is PER-HOLD (:meth:`_sweep_stale_open`): only the entries past
        the ceiling go, a live sibling hold SURVIVES and is still answered for,
        and the answer is None only when nothing survives.  Dropping the whole
        task instead cost twice over — no prediction for a task that really was
        holding, and, with the live hold's open span gone, its real
        ``lock_released`` reached ``_apply_release`` as an orphan and recorded
        nothing, so a hold the feed observed end to end never became a sample.

        The deterministic fix is :meth:`forget` on release; this catches the
        case where that call never happens either.  Deliberately an ABSOLUTE
        ceiling and not a multiple of the prediction: a hold running 10x its
        module's median is exactly the overdue holder this method exists to
        surface, and a relative ceiling would silence it.
        """
        wanted = str(task_id)
        now_f = float(now)
        dropped = self._sweep_stale_open(wanted, now_f)
        starts = [start for (tid, _module), start in self._open.items() if tid == wanted]
        if dropped:
            # Names ONLY the swept modules and counts the survivors: an operator
            # must not mistake a live hold for a consumed one, and the outcome is
            # no longer knowable in advance, so the message asserts neither.
            # Still a WARNING even when the prediction survives — a hold that
            # reached the ceiling is a missed release however the call resolves.
            logger.warning(
                'hold_history: task %s — dropped %d open hold(s) older than %.0fs as missed '
                'release(s): %s; %d open hold(s) remain',
                wanted, len(dropped), self._stale_open_secs, dropped, len(starts),
            )
        if not starts:
            return None
        elapsed = now_f - min(starts)
        predicted = self.predicted_hold(self.open_modules(wanted))
        if predicted is None:
            return None
        return max(0.0, predicted - elapsed)
