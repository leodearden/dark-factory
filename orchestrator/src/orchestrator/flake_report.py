"""Flake ledger operator report (plans/flake-ledger-prd.md, task ι) — the READ path.

This is PRD §1 surface 3: the one surface in the flake subsystem an operator can run
without wondering what it changed.  It aggregates the ledger written by task α
(:mod:`orchestrator.flake_ledger`) into three outputs — open debt, per-test recurrence
chains, and the three §5.6 health counters — and renders them as text.

BINDING CONTRACT — READ ONLY.  Nothing in this module opens debt, files a task,
resolves anything, or escalates.  All ledger DATA comes from α's public READ API
(``list_open_debt`` / ``read_occurrences`` / ``read_debt``); the only SQL of its own is
:func:`probe_ledger`'s single ``sqlite_master`` SELECT, which mutates strictly less than
those readers do (no DDL, no journal-mode pragma).  One consequence is not obvious and
is enforced by :func:`build_report`'s guards: α's readers PROVISION on read (``_open``
does ``parent.mkdir`` → ``connect`` → ``executescript(_SCHEMA)``), so calling one
against a project that has no ledger would create ``data/orchestrator/runs.db`` plus its
WAL sidecars as a side effect of PRINTING a report — and calling one against a
``runs.db`` that has no flake tables yet would CREATE those tables in it.  Absence and
unreadability are therefore both established BEFORE any read, and reported honestly
rather than papered over by a freshly-provisioned empty result.

NOT-A-MEASUREMENT IS ITS OWN STATE.  α's readers are B12 entry points: they swallow
every exception, warn, and return ``[]``.  A corrupt, truncated, or lock-contended
ledger therefore hands this module the same empty lists a healthy quiet one does, and a
report that could not tell them apart would print "(no open debt) … status: ok" over a
database it never actually read — the exact silent degradation the absent-DB guard
exists to prevent, one layer down.  :func:`probe_ledger` classifies the file first and
:func:`render_report` says loudly when the counters are not a measurement.

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

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from orchestrator.flake_ledger import (
    UNKNOWN_TEST_ID,
    DebtRow,
    FlakeCallSite,
    FlakeOccurrenceRow,
    FlakeVerdict,
    list_open_debt,
    read_debt,
    read_occurrences,
)

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


# --- ledger reachability -----------------------------------------------------
#
# Four states, because collapsing them is how a report starts lying.  α's readers
# collapse the last three into "[]" all on their own (B12: swallow, warn, return empty),
# and the CLI prints only the rendered string, so that warning reaches nobody.

# No runs.db at this path at all.
LEDGER_ABSENT = 'absent'
# A readable SQLite database carrying both flake tables — the only state in which the
# counters below are an actual measurement.
LEDGER_OK = 'ok'
# A readable SQLite database with no flake tables: the orchestrator's own runs.db exists
# (run_store.py owns that file too) but nothing has ever written the flake ledger into it.
LEDGER_NO_TABLES = 'no_flake_tables'
# The file would not open, or would not answer a trivial query: corrupt, truncated,
# mid-write, or locked beyond the connection timeout.
LEDGER_UNREADABLE = 'unreadable'


def probe_ledger(db_path: Path) -> str:
    """Classify the ledger file at *db_path* — one of the ``LEDGER_*`` states above.

    Call only for a path that EXISTS: this opens the file read-write (as α's readers do),
    so a probe of an absent path would create it.

    The probe is one ``sqlite_master`` SELECT.  It deliberately does NOT run
    ``PRAGMA quick_check``/``integrity_check``, which walk the whole database — this is
    an interactive operator command and ``runs.db`` reaches hundreds of megabytes in
    production (``analyze_modules.py`` measures 136MB).  A header-level corruption or a
    lock the connection cannot get past both surface on the first query anyway, which is
    the failure this needs to catch: not "is every page sound" but "did we actually read
    this database, or are the empty lists downstream a lie".

    Distinguishing :data:`LEDGER_NO_TABLES` from :data:`LEDGER_UNREADABLE` matters twice
    over.  It is the common benign case (a project whose orchestrator has run but whose
    flake ledger has never been written), so folding it into "unreadable" would cry wolf
    — and routing it AWAY from α's readers is what stops the read-only report from
    running ``executescript(_SCHEMA)`` against a live ``runs.db`` and creating the flake
    tables in it.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('flake_debt', 'flake_occurrence')"
            ).fetchall()
        finally:
            conn.close()
    except Exception:
        # Deliberately broad: sqlite3 raises DatabaseError for a corrupt header,
        # OperationalError for a lock or a permission problem, and the honest answer to
        # every one of them is the same — this is not a measurement.
        return LEDGER_UNREADABLE
    if {'flake_debt', 'flake_occurrence'} <= {row[0] for row in rows}:
        return LEDGER_OK
    return LEDGER_NO_TABLES


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

    COMPLEXITY: O(n log n) — one sort, then a TWO-POINTER sweep in which each of the two
    indices only ever advances, so the window scan itself is O(n).  It was previously a
    per-event re-scan of the whole list, i.e. O(n²), justified by the caller's
    ``since``/``limit`` bound; that justification did not hold, because the module's own
    cap (:data:`DEFAULT_OCCURRENCE_READ_LIMIT`, 20000) admits a list large enough to
    stall an interactive CLI for tens of seconds with no output.  The ``since``/``limit``
    bound is still required — a count over an unbounded read is a count over an unknown
    window — but it is a HONESTY bound on the counters, never a licence for a quadratic
    scan.
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

    # `lo` is the first event at or after the candidate window start, `hi` one past the
    # last event at or before its (INCLUSIVE) end.  Candidate starts are taken in sorted
    # order, so both bounds are monotonically non-decreasing and neither pointer ever
    # rewinds.  `counts` is the live multiset of test_ids inside [lo, hi), maintained
    # incrementally, so `distinct` is available in O(1) per candidate window.
    counts: dict[str, int] = {}
    distinct = 0
    lo = 0
    hi = 0
    peak_count = 0
    peak_start: str | None = None
    peak_slice = (0, 0)
    for i in range(len(events)):
        start = events[i][0]
        end = start + window
        while hi < len(events) and events[hi][0] <= end:
            tid = events[hi][1]
            counts[tid] = counts.get(tid, 0) + 1
            if counts[tid] == 1:
                distinct += 1
            hi += 1
        # Strictly `<`, so events sharing this exact start stamp stay INSIDE the window —
        # the same inclusive-both-ends bound the per-event re-scan had.
        while lo < i and events[lo][0] < start:
            tid = events[lo][1]
            counts[tid] -= 1
            if counts[tid] == 0:
                del counts[tid]
                distinct -= 1
            lo += 1
        if distinct > peak_count:
            peak_count = distinct
            peak_start = start.isoformat()
            peak_slice = (lo, hi)

    # Deferred to here rather than maintained incrementally: a max is not removable from
    # a sliding window in O(1), and the peak window is needed exactly once.
    psis = [e[2] for e in events[peak_slice[0]:peak_slice[1]] if e[2] is not None]
    peak_psi = max(psis) if psis else None

    return SystemicCounter(
        peak_distinct_tests=peak_count,
        peak_window_start=peak_start,
        peak_window_psi=peak_psi,
        exceeds_threshold=peak_count >= distinct_tests,
        threshold=distinct_tests,
        window_minutes=window_minutes,
    )


# --- recurrence chains -------------------------------------------------------


@dataclass(frozen=True)
class ChainRow:
    """One test's recurrence chain: its debt cycle plus its windowed occurrences."""

    test_id: str
    debt: DebtRow | None
    occurrence_count: int
    call_site_counts: dict[str, int]
    last_observed_at: str | None


def build_chains(
    occurrences: Sequence[FlakeOccurrenceRow],
    open_debt_rows: Sequence[DebtRow],
    debt_lookup: Callable[[str], DebtRow | None],
) -> list[ChainRow]:
    """Per-test chains over the UNION of open debt and windowed occurrences.

    The union is what makes the chain readable across cycles.  ``list_open_debt``
    filters on ``resolved_at IS NULL``, so a chain built from it alone goes blank
    exactly when a test is BETWEEN cycles — hiding the PRD's motivating case
    (``test_spawn_claude.py``, 7 de-flake tasks in 7 weeks) at the very moment each fix
    appears to have worked.  ``debt_lookup`` therefore reaches ``read_debt``, which
    returns RESOLVED rows too (§5.2 retains them deliberately, because the recurrence
    trigger reads them).

    Including tests that have occurrences but NO debt row is also what makes task κ's
    signal real: κ's acceptance check is that a ``CHRONIC-FLAKY`` marker with no JSONL
    present produces an occurrence that appears in THIS REPORT and no second de-flake
    task, which requires the report to show occurrences carrying no debt.

    ``UNKNOWN_TEST_ID`` is excluded: a sentinel names no test, so it can own no chain —
    ``open_debt`` itself refuses it for exactly this reason.  It still counts toward the
    gate-blind rate, where it is the entire point.

    ``call_site_counts`` is keyed on :class:`FlakeCallSite` values, so ``chronic_marker``
    stays visible beside ``merge_gate`` and ``main_probe`` rather than hiding inside a
    single per-test total — which is what would leave κ with no way to demonstrate its
    signal, and would let θ's future per-site rates be derived a second, divergent way.

    Ordering is deterministic — highest ``open_count`` first, then ``test_id`` — because
    the rendered report must be byte-stable.

    COMPLEXITY: occurrences are grouped by ``test_id`` in ONE pass and then indexed.  The
    earlier form re-filtered the whole occurrence list once per test in the universe,
    which is O(tests × occurrences) — quietly the dominant cost on exactly the ledger
    this report is for, since a chronically flaky suite grows in BOTH factors at once.
    """
    by_test: dict[str, list[FlakeOccurrenceRow]] = {}
    for row in occurrences:
        by_test.setdefault(row.test_id, []).append(row)

    universe: set[str] = {row.test_id for row in open_debt_rows}
    universe |= {test_id for test_id in by_test if test_id != UNKNOWN_TEST_ID}

    chains: list[ChainRow] = []
    for test_id in universe:
        mine = by_test.get(test_id, ())
        counts: dict[str, int] = {}
        for row in mine:
            # Bucket on the enum's value where the string is a known member, so an
            # unrecognised call_site stays visible under its own key instead of being
            # dropped or folded into a neighbour.
            try:
                key = FlakeCallSite(row.call_site).value
            except ValueError:
                key = row.call_site
            counts[key] = counts.get(key, 0) + 1
        stamps = [row.observed_at for row in mine]
        chains.append(
            ChainRow(
                test_id=test_id,
                debt=debt_lookup(test_id),
                occurrence_count=len(mine),
                call_site_counts=counts,
                last_observed_at=max(stamps) if stamps else None,
            )
        )
    chains.sort(key=lambda c: (-(c.debt.open_count if c.debt else 0), c.test_id))
    return chains


# --- the assembled report ----------------------------------------------------


@dataclass(frozen=True)
class DebtReportRow:
    """One open debt row plus its computed age (``None`` when ``opened_at`` won't parse)."""

    debt: DebtRow
    age: timedelta | None


@dataclass(frozen=True)
class FlakeLedgerReport:
    """Everything the operator report prints, already aggregated.

    ``db_present`` is carried explicitly because "no ledger here" and "an empty ledger"
    are different facts, and α's readers cannot distinguish them: they PROVISION on
    read, so reading an absent DB would hand back an empty result set from a database
    this report had just created.

    ``db_status`` carries the same distinction one layer down, for a file that IS present
    (see :func:`probe_ledger`).  :attr:`measured` is the single predicate every consumer
    should ask — including task θ, which must not act on counters that were never read.
    It defaults to :data:`LEDGER_OK` so a hand-built report (the render tests build them
    directly) means what it looks like it means.
    """

    db_path: Path
    db_present: bool
    generated_at: datetime
    window_since: str | None
    open_debt: list[DebtReportRow]
    chains: list[ChainRow]
    gate_blind: GateBlindCounter
    non_convergence: NonConvergenceCounter
    systemic: SystemicCounter
    db_status: str = LEDGER_OK

    @property
    def measured(self) -> bool:
        """True only when the counters are an actual reading of an actual ledger."""
        return self.db_present and self.db_status == LEDGER_OK


def _unmeasured_report(
    db_path: Path,
    *,
    db_present: bool,
    db_status: str,
    now: datetime,
    window_hours: int,
    age_days: int,
    gate_blind_threshold: float,
    gate_blind_min_observations: int,
    systemic_distinct_tests: int,
    systemic_window_minutes: int,
) -> FlakeLedgerReport:
    """An empty report for a ledger that was never read — absent, tableless, unreadable.

    The counters are still built (from empty input) rather than left ``None``: every
    threshold the operator is being measured against stays visible, and
    :attr:`FlakeLedgerReport.measured` is what says the numbers are not a reading.
    """
    return FlakeLedgerReport(
        db_path=db_path,
        db_present=db_present,
        db_status=db_status,
        generated_at=now,
        window_since=None,
        open_debt=[],
        chains=[],
        gate_blind=compute_gate_blind(
            [],
            threshold=gate_blind_threshold,
            min_observations=gate_blind_min_observations,
            window_hours=window_hours,
        ),
        non_convergence=compute_non_convergence([], now, age_days=age_days),
        systemic=compute_systemic(
            [],
            distinct_tests=systemic_distinct_tests,
            window_minutes=systemic_window_minutes,
        ),
    )


def build_report(
    db_path: Path,
    *,
    now: datetime | None = None,
    window_hours: int = DEFAULT_GATE_BLIND_WINDOW_HOURS,
    age_days: int = DEFAULT_DEBT_AGE_ESCALATE_DAYS,
    gate_blind_threshold: float = DEFAULT_GATE_BLIND_RATE_THRESHOLD,
    gate_blind_min_observations: int = DEFAULT_GATE_BLIND_MIN_OBSERVATIONS,
    systemic_distinct_tests: int = DEFAULT_SYSTEMIC_DISTINCT_TESTS,
    systemic_window_minutes: int = DEFAULT_SYSTEMIC_WINDOW_MINUTES,
    occurrence_limit: int = DEFAULT_OCCURRENCE_READ_LIMIT,
) -> FlakeLedgerReport:
    """Read the ledger at *db_path* and aggregate it into a :class:`FlakeLedgerReport`.

    All ledger DATA comes from α's public read API — ``list_open_debt`` /
    ``read_occurrences`` / ``read_debt``; the only SQL of this module's own is
    :func:`probe_ledger`'s ``sqlite_master`` SELECT, which decides whether those readers
    are safe to call at all.  *now* defaults to the wall clock and is injectable so the
    whole report is deterministic under test.

    The occurrence read is bounded on BOTH axes: ``since`` (so the counters divide by a
    known window) and ``limit`` (because ``flake_occurrence`` is append-only and
    unpruned by design).  α's docstring is explicit that a count over a ``limit``-capped
    read is a count over an unknown window and dividing by it is meaningless, so a read
    that fills the limit is surfaced rather than silently yielding a rate.
    """
    now = now or datetime.now(UTC)

    # Shared by both not-a-measurement exits below, so the thresholds an operator is
    # being shown stay identical whichever guard fired.
    def _unmeasured(*, db_present: bool, db_status: str) -> FlakeLedgerReport:
        return _unmeasured_report(
            db_path,
            db_present=db_present,
            db_status=db_status,
            now=now,
            window_hours=window_hours,
            age_days=age_days,
            gate_blind_threshold=gate_blind_threshold,
            gate_blind_min_observations=gate_blind_min_observations,
            systemic_distinct_tests=systemic_distinct_tests,
            systemic_window_minutes=systemic_window_minutes,
        )

    # READ-ONLY GUARD — load-bearing, not defensive.  Every one of α's readers routes
    # through `_open`, which does `db_path.parent.mkdir(parents=True, exist_ok=True)`,
    # `sqlite3.connect()` (which CREATES the file) and `executescript(_SCHEMA)`, plus
    # `-wal`/`-shm` sidecars from the `journal_mode=WAL` pragma.  So calling ANY read
    # here against a project that has no ledger would provision
    # `data/orchestrator/runs.db` and two sidecar files as a side effect of printing a
    # report.  Reporting absence is also strictly more informative than reporting an
    # empty ledger, since α's readers cannot tell "no data" from "freshly provisioned".
    if not db_path.exists():
        return _unmeasured(db_present=False, db_status=LEDGER_ABSENT)

    # SECOND GUARD — an unreadable ledger is not an empty one.  α's readers swallow every
    # exception and return `[]` (B12), so a corrupt, truncated or lock-contended runs.db
    # would otherwise render as "(no open debt) … status: ok" with the only trace being a
    # logger.warning the CLI never surfaces.  The tableless case is routed out here too,
    # because α's `_open` would CREATE the flake tables in a runs.db that lacks them —
    # the read-only report writing DDL into a live project database.
    db_status = probe_ledger(db_path)
    if db_status != LEDGER_OK:
        return _unmeasured(db_present=True, db_status=db_status)

    since = (now - timedelta(hours=window_hours)).isoformat()
    open_rows = list_open_debt(db_path)
    occurrences = read_occurrences(db_path, since=since, limit=occurrence_limit)
    truncated = len(occurrences) >= occurrence_limit

    # Every OPEN debt row is already in hand from the single `list_open_debt` scan above,
    # so the chain build must not re-read it per test: each `read_debt` goes through α's
    # `_open`, which opens a fresh connection, applies the durability pragmas (including
    # a `journal_mode=WAL` switch) and runs `executescript(_SCHEMA)` against the LIVE
    # runs.db the merge lane is also using.  N+1 of those turns a nominally read-only
    # report into N schema executions contending on that database.  What remains is only
    # the BETWEEN-CYCLES remainder — a test with occurrences whose debt is currently
    # RESOLVED, and so absent from `list_open_debt` — which is the case the chain exists
    # to show and which α exposes no batched reader for (a `read_debt_many` belongs in α,
    # not here; ι writes no SQL of its own).
    open_by_test = {row.test_id: row for row in open_rows}

    def _debt_lookup(test_id: str) -> DebtRow | None:
        hit = open_by_test.get(test_id)
        return hit if hit is not None else read_debt(db_path, test_id)

    # `list_open_debt`'s `opened_at, test_id` ordering is contractual (its docstring
    # names ι as the reason), so it is relied on directly and never re-sorted here.
    open_debt = [
        DebtReportRow(
            debt=row,
            age=(now - opened) if (opened := _parse_stamp(row.opened_at)) is not None else None,
        )
        for row in open_rows
    ]

    return FlakeLedgerReport(
        db_path=db_path,
        db_present=True,
        generated_at=now,
        window_since=since,
        open_debt=open_debt,
        chains=build_chains(occurrences, open_rows, _debt_lookup),
        gate_blind=compute_gate_blind(
            occurrences,
            threshold=gate_blind_threshold,
            min_observations=gate_blind_min_observations,
            window_hours=window_hours,
            truncated=truncated,
        ),
        non_convergence=compute_non_convergence(open_rows, now, age_days=age_days),
        systemic=compute_systemic(
            occurrences,
            distinct_tests=systemic_distinct_tests,
            window_minutes=systemic_window_minutes,
        ),
    )


# --- rendering ---------------------------------------------------------------


def render_report(report: FlakeLedgerReport) -> str:
    """Render *report* as text.

    Pure and byte-deterministic — the same :class:`FlakeLedgerReport` always renders
    identically (fixed section order, every section rendered even when empty, no
    wall-clock or dict-iteration-order dependence), following
    ``format_stratification_table``'s determinism property.  No ``click`` import: this
    returns a string and the CLI only echoes it.
    """
    gb, nc, sy = report.gate_blind, report.non_convergence, report.systemic
    lines: list[str] = [
        f'flake ledger: {report.db_path}',
        f'generated_at: {report.generated_at.isoformat()}',
    ]
    if not report.db_present:
        # Absence is a DIFFERENT fact from an empty ledger, and α's readers cannot tell
        # them apart (they provision on read).  Say which one this is.
        lines.append('  NO LEDGER: this project has no runs.db — nothing has been recorded yet.')
    elif report.db_status == LEDGER_UNREADABLE:
        # The loudest line in the report, because it is the one state in which every
        # number below is fiction: α's readers degrade a corrupt/locked database to `[]`
        # and warn to a logger the CLI never prints.
        lines.append(
            '  *** LEDGER UNREADABLE: this runs.db did not answer a trivial query '
            '(corrupt, truncated, or locked) — the counters below are NOT a '
            'measurement. ***'
        )
    elif report.db_status == LEDGER_NO_TABLES:
        lines.append(
            '  NO FLAKE TABLES: this runs.db carries no flake ledger tables — nothing '
            'has been recorded yet, and the counters below are NOT a measurement.'
        )
    else:
        lines.append(f'window since: {report.window_since}')
    if gb.truncated:
        # α's read_occurrences docstring: a count over a limit-capped read is a count
        # over an unknown window, and dividing by it is meaningless.  The counters are
        # still printed — a bounded read is better than an unbounded one — but they are
        # labelled so an operator cannot mistake them for a full-window measurement.
        lines.append(
            '  WARNING: the occurrence read was TRUNCATED at its limit — the counters '
            'below cover a PARTIAL window, not the full one.'
        )

    # --- section 1: open debt ---
    lines += ['', 'OPEN DEBT', '-' * 60]
    if not report.open_debt:
        lines.append('  (no open debt)')
    for row in report.open_debt:
        d = row.debt
        # A blank owner cell would render a §5.9 breach as MISSING DATA.  It is instead
        # the single most consequential state this report can encounter: suppression
        # LANDS the merge (§5.7), so an unowned debt row means a flaky test is silently
        # no longer blocking anything and nobody owns fixing it.  Naming it loudly is a
        # READ of the ledger — ι files nothing.
        owner = f'owner={d.owner_task_id}' if d.owner_task_id else '*** NO OWNER (invariant breach) ***'
        over = ' OVER-AGE' if (row.age is not None and row.age > timedelta(days=nc.age_threshold_days)) else ''
        lines.append(
            f'  {d.test_id}  age={format_age(row.age)}  {owner}  '
            f'open_count={d.open_count}{over}'
        )

    # --- section 2: recurrence chains ---
    lines += ['', 'RECURRENCE CHAINS', '-' * 60]
    if not report.chains:
        lines.append('  (no occurrences in window and no open debt)')
    for chain in report.chains:
        d = chain.debt
        lines.append(f'  {chain.test_id}')
        if d is None:
            # κ's case: an occurrence with no debt row at all must still be visible.
            lines.append('      debt: (none — occurrences only, no debt row)')
        else:
            state = 'resolved' if d.resolved_at else 'open'
            lines.append(
                f'      debt: {state}  open_count={d.open_count}  '
                f'prior_resolved_at={d.prior_resolved_at or "-"}  '
                f'prior_resolving_commit={d.prior_resolving_commit or "-"}'
            )
        # Sorted so the split is byte-stable, and every site is spelled out rather than
        # summed, so κ's chronic_marker cannot hide inside a merge_gate total.
        sites = '  '.join(f'{k}={v}' for k, v in sorted(chain.call_site_counts.items()))
        lines.append(
            f'      occurrences: {chain.occurrence_count}  [{sites or "none in window"}]  '
            f'last={chain.last_observed_at or "-"}'
        )

    # --- section 3: health counters ---
    lines += ['', 'HEALTH COUNTERS', '-' * 60]
    lines.append(f'  gate blind (class 1, {gb.window_hours}h window):')
    if not gb.total:
        lines.append('      no occurrences in window')
    # `n/a`, never `0.00`: α warns that a zero-row answer "reads as healthy rather than
    # as a broken query", and "we cannot tell" must not print as "the gate is fine".
    rate_txt = 'n/a' if gb.rate is None else f'{gb.rate:.2f}'
    lines.append(
        f'      unconfirmable rate: {rate_txt}  '
        f'({gb.unconfirmable} unconfirmable / {gb.total} observations)  '
        f'threshold={gb.threshold:.2f}'
    )
    if not report.measured:
        # `ok` is a claim about the ledger, and there is no ledger reading to make it
        # from.  Saying "not measured" is the whole point of the guards in build_report.
        lines.append('      status: not measured (no ledger reading)')
    elif not gb.sufficient:
        # DISTINCT from "below threshold": those two states mean opposite things.
        lines.append(
            f'      status: insufficient observations '
            f'(need {gb.min_observations}, have {gb.total})'
        )
    else:
        lines.append(f'      status: {"OVER THRESHOLD" if gb.exceeds_threshold else "ok"}')

    lines.append(f'  non-convergence (class 2, age backstop {nc.age_threshold_days}d):')
    lines.append(
        f'      open={nc.open_tests}  recurrent={nc.recurrent_tests}  '
        f'over_age={nc.over_age_tests}  unowned={nc.unowned_tests}  '
        f'oldest={format_age(nc.oldest_age)}'
    )
    if nc.unparseable_opened_at:
        lines.append(
            f'      WARNING: {nc.unparseable_opened_at} row(s) have an unreadable '
            'opened_at and are excluded from the age figures above'
        )

    lines.append(f'  systemic (class 3, {sy.window_minutes}m window):')
    psi_txt = 'n/a' if sy.peak_window_psi is None else f'{sy.peak_window_psi:.1f}'
    lines.append(
        f'      peak distinct tests: {sy.peak_distinct_tests}  threshold={sy.threshold}  '
        f'peak_window_start={sy.peak_window_start or "-"}  peak_psi_cpu_some10={psi_txt}'
    )
    if not report.measured:
        lines.append('      status: not measured (no ledger reading)')
    else:
        lines.append(f'      status: {"OVER THRESHOLD" if sy.exceeds_threshold else "ok"}')

    return '\n'.join(lines)
