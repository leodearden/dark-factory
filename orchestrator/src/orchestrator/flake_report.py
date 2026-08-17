"""Flake ledger operator report (plans/flake-ledger-prd.md, task ι) — the READ path.

This is PRD §1 surface 3: the one surface in the flake subsystem an operator can run
without wondering what it changed.  It aggregates the ledger written by task α
(:mod:`orchestrator.flake_ledger`) into three outputs — open debt, per-test recurrence
chains, and the three §5.6 health counters — and renders them as text.

BINDING CONTRACT — READ ONLY.  Nothing in this module opens debt, files a task,
resolves anything, or escalates.  It reaches only α's public READ API
(``list_open_debt`` / ``read_occurrences`` / ``read_debt``) and it writes no SQL of its
own.  One consequence is not obvious and is enforced by :func:`build_report`'s first
guard: α's readers PROVISION on read (``_open`` does ``parent.mkdir`` → ``connect`` →
``executescript(_SCHEMA)``), so calling one against a project that has no ledger would
create ``data/orchestrator/runs.db`` plus its WAL sidecars as a side effect of PRINTING
a report.  Absence is therefore detected before any read and reported honestly, rather
than papered over by a freshly-provisioned empty DB.

THRESHOLDS ARE PARAMETERS, not hardcoded comparisons.  The ``DEFAULT_*`` constants below
mirror PRD §10's defaults, and every counter function takes them as keyword arguments.
Task θ's ``FlakeLedgerConfig`` (§10) can then pass its configured values into these same
functions instead of re-deriving any counter at a second site — INV-5
(no-lockstep-duplication) is the failure mode this PRD exists to fix, and two sites
computing one rate slightly differently is exactly how it recurs.

LAYERING.  Aggregation and rendering are not storage, so they do not live in α's module.
:func:`render_report` returns a string and imports no ``click``; the CLI command only
echoes it — the same split as ``eval-list-fixtures`` →
``evals.task_sampler.format_stratification_table``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from orchestrator.flake_ledger import DebtRow, FlakeOccurrenceRow, FlakeVerdict

# --- PRD §10 defaults --------------------------------------------------------
#
# Mirrored here as module constants so ι has no dependency on task θ's config model
# (which does not exist yet, and whose dependency edge runs θ → η, i.e. DOWNSTREAM of
# this task).  Every one is a keyword parameter at the function that uses it.

# §5.6 class 2 BACKSTOP: an open debt row older than this is over age.
DEFAULT_DEBT_AGE_ESCALATE_DAYS = 3

# §5.6 class 1: unconfirmable / (unconfirmable + confirmed) above this rate is
# "the gate has gone blind".
DEFAULT_GATE_BLIND_RATE_THRESHOLD = 0.25

# ... but only once the window carries at least this many observations, so the class
# cannot fire on 1-of-2.
DEFAULT_GATE_BLIND_MIN_OBSERVATIONS = 8

# The window the class-1 rate is computed over (7 days).
DEFAULT_GATE_BLIND_WINDOW_HOURS = 168

# §5.6 class 3: this many DISTINCT tests suppressed inside one window is systemic
# host pressure, not N independent flaky tests.
DEFAULT_SYSTEMIC_DISTINCT_TESTS = 4
DEFAULT_SYSTEMIC_WINDOW_MINUTES = 60

# The occurrence read is bounded (``flake_occurrence`` is append-only and unpruned by
# design — α's module docstring, §11 Q1).  A read that fills this limit is surfaced as
# truncated rather than silently yielding a rate over an unknown window.
DEFAULT_OCCURRENCE_READ_LIMIT = 20000


def _parse_stamp(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 ledger stamp to an aware UTC datetime, or ``None``.

    A naive value is assumed UTC and gets the tzinfo ATTACHED, never
    ``.astimezone()`` — mirroring :func:`flake_ledger._canonicalize_utc`, where
    ``.astimezone()`` on a naive datetime would apply the HOST's local offset and
    silently shift the stamp by the dispatcher's timezone.

    Returns ``None`` (never raises) on a malformed or missing stamp.  α's own
    ``_normalize_observed_at`` raises by design and is private, so ι cannot reuse it;
    this is the single spelling of the naive-means-UTC rule for the whole report.
    Degrading to ``None`` rather than to zero is the safe direction: an unparseable
    ``opened_at`` rendered as ``0d`` would make a stale debt row look brand new and
    silently suppress it from the age backstop, whereas ``unknown`` is visibly wrong.
    One bad row must also not take down the report — a read path an operator cannot
    rely on is worse than a caveated one.
    """
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_age(delta: timedelta | None) -> str:
    """Render an age as a compact ``'Nd Nh'`` string; ``None`` renders as unknown.

    ``None`` must NOT render as ``'0d 0h'``: those two states mean opposite things to
    an operator (a brand-new row vs. a row whose clock could not be read at all).
    """
    if delta is None:
        return '(unknown age)'
    total_hours = int(delta.total_seconds() // 3600)
    return f'{total_hours // 24}d {total_hours % 24}h'


# --- §5.6 class 1: the gate has gone blind ----------------------------------


@dataclass(frozen=True)
class GateBlindCounter:
    """The unconfirmable RATE over a window, plus everything needed to read it honestly.

    ``rate`` is ``None`` — never ``0.0`` — when ``total`` is zero, and ``sufficient``
    is separate from ``exceeds_threshold`` so the renderer can say "insufficient
    observations" rather than "below threshold".  Those two states mean opposite things
    to an operator: one is "the gate looks fine", the other is "we cannot tell".
    """

    unconfirmable: int
    confirmed: int
    total: int
    rate: float | None
    threshold: float
    min_observations: int
    sufficient: bool
    exceeds_threshold: bool
    window_hours: int
    truncated: bool


def compute_gate_blind(
    occurrences: Sequence[FlakeOccurrenceRow],
    *,
    threshold: float = DEFAULT_GATE_BLIND_RATE_THRESHOLD,
    min_observations: int = DEFAULT_GATE_BLIND_MIN_OBSERVATIONS,
    window_hours: int = DEFAULT_GATE_BLIND_WINDOW_HOURS,
    truncated: bool = False,
) -> GateBlindCounter:
    """Count ``unconfirmable`` against confirmed verdicts over *occurrences*.

    Rows carrying the ``UNKNOWN_TEST_ID`` sentinel are COUNTED, deliberately: an
    unconfirmable observation that resolved no node-ids is the entire class-1 signal
    (§5.6 — "6 unconfirmable lines sat at INFO for a month"), so filtering it out
    because it names no test would delete the measurement.

    Verdicts are bucketed on :class:`FlakeVerdict` members, never on bare string
    literals, so a typo cannot silently split a bucket.  An unrecognised verdict string
    should be unreachable (``record_flake_occurrence`` coerces on the way in) but is
    excluded from BOTH buckets and from ``total`` if it ever appears — inflating the
    denominator with rows of unknown meaning would understate the rate, which is the
    dangerous direction for a blindness counter.
    """
    unconfirmable = 0
    confirmed = 0
    for row in occurrences:
        if row.verdict == FlakeVerdict.unconfirmable:
            unconfirmable += 1
        elif row.verdict in (FlakeVerdict.passes_in_isolation, FlakeVerdict.fails_in_isolation):
            confirmed += 1
    total = unconfirmable + confirmed
    rate = (unconfirmable / total) if total else None
    sufficient = total >= min_observations
    return GateBlindCounter(
        unconfirmable=unconfirmable,
        confirmed=confirmed,
        total=total,
        rate=rate,
        threshold=threshold,
        min_observations=min_observations,
        sufficient=sufficient,
        exceeds_threshold=sufficient and rate is not None and rate > threshold,
        window_hours=window_hours,
        truncated=truncated,
    )


# --- §5.6 class 2: the de-flake cycle is not converging ----------------------


@dataclass(frozen=True)
class NonConvergenceCounter:
    """Open debt that is not resolving: recurrence, age, and unowned rows."""

    open_tests: int
    recurrent_tests: int
    over_age_tests: int
    unowned_tests: int
    unparseable_opened_at: int
    oldest_age: timedelta | None
    age_threshold_days: int


def compute_non_convergence(
    open_debt_rows: Sequence[DebtRow],
    now: datetime,
    *,
    age_days: int = DEFAULT_DEBT_AGE_ESCALATE_DAYS,
) -> NonConvergenceCounter:
    """Summarise open debt against §5.6's class-2 triggers.

    READING THE TWO TRIGGERS (§5.6): ``recurrent_tests`` (``open_count > 1``) is the
    PRIMARY signal — a test that has been in debt more than once has already had a fix
    that did not hold.  ``over_age_tests`` is only the BACKSTOP.  The PRD's "0 of 35
    de-flake tasks ever exceeded 3 days" figure must NOT be read as evidence the
    backstop is inert: §2 records that it is survivorship-filtered over CLOSED tasks,
    so the rows it would have caught are precisely the ones missing from the sample.

    ``unowned_tests`` counts open rows with no ``owner_task_id`` — a live §5.9 invariant
    breach, and the single most consequential state this report can encounter, because
    §5.7's suppression LANDS the merge: once a test is suppressed, nothing except the
    debt invariant keeps it visible at all.  Counting it (and rendering it loudly) is a
    READ of the ledger; ι files nothing.

    An unparseable ``opened_at`` is counted in its own bucket and contributes to neither
    ``over_age_tests`` nor ``oldest_age``, so a row whose clock cannot be read is
    visibly unknown rather than silently brand-new.
    """
    threshold = timedelta(days=age_days)
    recurrent = 0
    over_age = 0
    unowned = 0
    unparseable = 0
    oldest: timedelta | None = None
    for row in open_debt_rows:
        if row.open_count > 1:
            recurrent += 1
        if row.owner_task_id is None:
            unowned += 1
        opened = _parse_stamp(row.opened_at)
        if opened is None:
            unparseable += 1
            continue
        age = now - opened
        # Strictly greater: a row opened exactly `age_days` ago has not yet EXCEEDED
        # the bound.
        if age > threshold:
            over_age += 1
        if oldest is None or age > oldest:
            oldest = age
    return NonConvergenceCounter(
        open_tests=len(open_debt_rows),
        recurrent_tests=recurrent,
        over_age_tests=over_age,
        unowned_tests=unowned,
        unparseable_opened_at=unparseable,
        oldest_age=oldest,
        age_threshold_days=age_days,
    )


# --- §5.6 class 3: systemic host pressure -----------------------------------


@dataclass(frozen=True)
class SystemicCounter:
    """The peak number of DISTINCT tests suppressed inside any one window."""

    peak_distinct_tests: int
    peak_window_start: str | None
    peak_window_psi: float | None
    exceeds_threshold: bool
    threshold: int
    window_minutes: int


def compute_systemic(
    occurrences: Sequence[FlakeOccurrenceRow],
    *,
    distinct_tests: int = DEFAULT_SYSTEMIC_DISTINCT_TESTS,
    window_minutes: int = DEFAULT_SYSTEMIC_WINDOW_MINUTES,
) -> SystemicCounter:
    """Find the busiest suppression window and report its distinct-test count.

    §5.6's discriminator between class 2 and class 3 is DISTINCTNESS, not volume: one
    test suppressed six times in an hour is a single non-converging test, while six
    different tests suppressed in the same hour is the host, not the tests.  Counting
    distinct ``test_id``s per window is what keeps those two apart.

    Only ``passes_in_isolation`` rows count — a ``fails_in_isolation`` verdict is a real
    red, and folding those in would manufacture a pressure signal out of ordinary
    breakage.  Rows whose ``observed_at`` will not parse are dropped rather than being
    given a default position on the timeline.

    ``peak_window_psi`` is the max NON-NULL ``psi_cpu_some10`` in the peak window, and
    stays ``None`` when no row in it carried a reading.  α binds a missing PSI to SQL
    NULL specifically so it never becomes ``0.0``; rendering ``0.0`` here would say "the
    host was idle" when the truth is "the host was not measured".

    COMPLEXITY: this is a quadratic scan over the occurrence list, which is acceptable
    ONLY because the caller passes a ``since``-filtered, ``limit``-capped read (see
    :func:`build_report`).  Do not "optimise" it into an unbounded streaming form
    without first re-establishing that bound — and do not remove the bound on the
    grounds that the scan got cheaper.
    """
    window = timedelta(minutes=window_minutes)
    events: list[tuple[datetime, str, float | None]] = []
    for row in occurrences:
        if row.verdict != FlakeVerdict.passes_in_isolation:
            continue
        stamp = _parse_stamp(row.observed_at)
        if stamp is None:
            continue
        events.append((stamp, row.test_id, row.psi_cpu_some10))
    events.sort(key=lambda e: (e[0], e[1]))

    peak_count = 0
    peak_start: str | None = None
    peak_psi: float | None = None
    for start, _, _ in events:
        end = start + window
        in_window = [e for e in events if start <= e[0] <= end]
        distinct = {e[1] for e in in_window}
        if len(distinct) > peak_count:
            peak_count = len(distinct)
            peak_start = start.isoformat()
            psis = [e[2] for e in in_window if e[2] is not None]
            peak_psi = max(psis) if psis else None

    return SystemicCounter(
        peak_distinct_tests=peak_count,
        peak_window_start=peak_start,
        peak_window_psi=peak_psi,
        exceeds_threshold=peak_count >= distinct_tests,
        threshold=distinct_tests,
        window_minutes=window_minutes,
    )
