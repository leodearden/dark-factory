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
    """
    universe: set[str] = {row.test_id for row in open_debt_rows}
    universe |= {row.test_id for row in occurrences if row.test_id != UNKNOWN_TEST_ID}

    chains: list[ChainRow] = []
    for test_id in universe:
        mine = [row for row in occurrences if row.test_id == test_id]
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

    Reaches ONLY α's public read API — ``list_open_debt`` / ``read_occurrences`` /
    ``read_debt`` — and writes no SQL of its own.  *now* defaults to the wall clock and
    is injectable so the whole report is deterministic under test.

    The occurrence read is bounded on BOTH axes: ``since`` (so the counters divide by a
    known window) and ``limit`` (because ``flake_occurrence`` is append-only and
    unpruned by design).  α's docstring is explicit that a count over a ``limit``-capped
    read is a count over an unknown window and dividing by it is meaningless, so a read
    that fills the limit is surfaced rather than silently yielding a rate.
    """
    now = now or datetime.now(UTC)

    # READ-ONLY GUARD — load-bearing, not defensive.  Every one of α's readers routes
    # through `_open`, which does `db_path.parent.mkdir(parents=True, exist_ok=True)`,
    # `sqlite3.connect()` (which CREATES the file) and `executescript(_SCHEMA)`, plus
    # `-wal`/`-shm` sidecars from the `journal_mode=WAL` pragma.  So calling ANY read
    # here against a project that has no ledger would provision
    # `data/orchestrator/runs.db` and two sidecar files as a side effect of printing a
    # report.  Reporting absence is also strictly more informative than reporting an
    # empty ledger, since α's readers cannot tell "no data" from "freshly provisioned".
    if not db_path.exists():
        return FlakeLedgerReport(
            db_path=db_path,
            db_present=False,
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

    since = (now - timedelta(hours=window_hours)).isoformat()
    open_rows = list_open_debt(db_path)
    occurrences = read_occurrences(db_path, since=since, limit=occurrence_limit)
    truncated = len(occurrences) >= occurrence_limit

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
        chains=build_chains(occurrences, open_rows, lambda t: read_debt(db_path, t)),
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
    if not gb.sufficient:
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
    lines.append(f'      status: {"OVER THRESHOLD" if sy.exceeds_threshold else "ok"}')

    return '\n'.join(lines)
