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

from datetime import UTC, datetime, timedelta

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
