"""Derivation of the absolute outer bound on a recovered sidecar's age.

Task 3730 / PRD leaf δ (``plans/session-resume-eligibility-seam-prd.md`` D3):
``session_resume.absolute_resume_age_secs`` is the backstop that keeps
"a durable archive outranks age" from degenerating into "no age limit at
all". It is a DERIVED bound, not a chosen number — this module holds the
sampler that measures its two inputs against runs.db and the arithmetic that
turns them into a requirement, so the live guard
(``orchestrator/tests/test_resume_age_bound.py``) and its host-independent
falsifiability companion consume ONE implementation rather than two
re-derivations that can drift apart.

Shape copied from ``scripts/gc_agent_transcripts.py`` (task 3621), the
existing derived-bound precedent in this repo: a sampler returning a
dataclass-or-``None``, a pure ``required_*`` function, and a safety-factor
constant carrying its measurement provenance inline. Only the SOURCE differs —
3621 samples the filesystem archive, this samples runs.db.

Sited under ``orchestrator/src`` rather than ``scripts/`` because it derives an
orchestrator CONFIG knob and is read by orchestrator tests; nothing here runs
on the dispatch path.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Overridable via env var for portability across checkouts -- runs.db is
# offline, gitignored data that lives in the MAIN repo, not a task worktree
# (mirrors prompt_opt.__main__'s PROMPT_OPT_RUNS_DB and reviewer_trial's
# REVIEWER_TRIAL_RUNS_DB idiom). A project_root-relative spelling — the one
# shared.flake_ledger.ledger_db_path uses — is deliberately NOT reused: in a
# task worktree that composes against the WRONG tree, so the live derived-bound
# guard would skip on every verify and the bound would be decorative.
RUNS_DB_ENV_VAR = 'RESUME_AGE_RUNS_DB'
_DEFAULT_RUNS_DB = '/home/leo/src/dark-factory/data/orchestrator/runs.db'

# Trailing window BOTH terms are sampled over. Matches the 90-day retention
# window the transcript archive keeps, and bounds the derivation to the fleet's
# CURRENT behaviour: without it a single ancient row would manufacture a phantom
# multi-month gap between itself and the recent corpus, and a single retired
# 40-hour invocation would stay in the in-flight maximum forever — either way
# the bound would ratchet up instead of re-deriving.
#
# ONE window for both terms, deliberately. The two are reported side by side
# (:class:`ResumeAgeSample` carries a single ``span_days`` beside both maxima,
# and the live guard's failure message prints them together), so a term drawn
# from outside that span would describe a corpus the sample does not claim to
# have measured — and would ratchet the bound off history the fleet has
# already left behind.
SAMPLE_WINDOW_DAYS = 90

# Smallest number of legitimate in-flight samples :func:`observed_resume_age_inputs`
# will report from. Below it the sampler returns None rather than a weak
# number, and the distinction is load-bearing: absence must never read as a
# ZERO, because a zero term satisfies any bound trivially and would leave the
# derived-bound guard permanently green while measuring nothing. Same rationale
# 3621 records for MIN_RATE_SAMPLE_DAYS at gc_agent_transcripts.py. Sized well
# under the live corpus (n=4730 legitimate completions on 2026-09-07) so it
# gates only genuinely sparse sources — a fresh checkout, a truncated db, a
# corpus holding nothing but cancellations.
MIN_SAMPLE_TASKS = 50

# THE MEASUREMENT the shipped default was derived from, as CONSTANTS rather
# than prose. Both terms are load-bearing code, not commentary: the
# anti-inflation clamp in test_resume_age_bound.py doubles them to prove the
# live guard's subject can still trip. Re-typing them into the test made two
# copies of one measurement, so a re-derivation could update this block and
# leave the clamp calibrated off stale numbers — SPOT, with the drift silent.
#
# MEASURED on this host against data/orchestrator/runs.db, read-only, by the
# sampler below (a re-measurement of the 2026-09-04 planning figures; both
# terms moved under 1% in three days, which is itself evidence the derivation
# is stable rather than noise-driven).
MEASUREMENT_DATE = '2026-09-07'

# T1 — the longest LEGITIMATE invocation: 8.90 h, from n=4,730 task_completed
# rows with duration_ms > 0 and outcome NOT IN ('cancelled','soft-cancelled').
# RE-VERIFIED 2026-09-10 under the trailing-window scoping that now applies to
# this term too: the MAXIMUM is unchanged (the longest invocation is recent) and
# only the population narrows, to n=2,993 rows inside the window.
MEASURED_INFLIGHT_SECS = 32_052.087

# T2 — the longest stretch with no event of any kind: 56.99 h
# (2026-06-12 -> 2026-06-14), from n=303,040 inter-event gaps spanning 90.58
# days. Next four: 42.42 h, 37.17 h, 25.42 h, 19.84 h.
#
# RE-VERIFIED 2026-09-10: that outage has now aged OUT of the trailing window,
# so a fresh sample reads 42.42 h and the requirement falls to 277,169 s. The
# shipped default is deliberately NOT lowered to follow it — the window sliding
# past a real outage is not evidence the outage cannot recur, and a backstop
# that shrank below a worst case the fleet has actually produced would start
# rejecting sessions that were still legitimately in flight. The direction that
# must track the fleet is UPWARD, and that is the one the live guard checks.
MEASURED_DOWNTIME_SECS = 205_149.466

# Headroom multiplier over the plain (max in-flight + max downtime) projection,
# applied to the two terms above:
#
#   requirement ..... ceil((MEASURED_INFLIGHT_SECS + MEASURED_DOWNTIME_SECS)
#                     x 1.5) = 355,803 s = 4.12 days.
#   shipped default .. 432,000 s (5 days) — the requirement rounded UP to the
#                     next whole day, which is the whole of the rule; no
#                     multiplier was tuned to reach it.
#   margin .......... 1.21x over the requirement.
#   trip point ...... the guard goes red once T1 + T2 exceeds 288,000 s (80 h)
#                     — about 1.21x the 65.89 h measured above, i.e. roughly a
#                     3.3-day outage. Rare enough not to flap, and when it does
#                     fire it is telling the truth: sidecars really can be that
#                     old, and the bound has to be re-derived rather than
#                     assumed.
#
# Must be >= 1: below 1 the bound would sit UNDER the worst case it is derived
# from and would reject sessions that are still legitimately in flight.
#
# The factor is deliberately modest (1.5, against 3621's 3) because the two
# terms are already MAXIMA over a 90-day window rather than a peak rate — the
# conservatism is in the inputs, not the multiplier. The anti-inflation clamp
# in test_resume_age_bound.py caps the practical headroom below 2x anyway: a
# sample with both terms doubled must still exceed the shipped default, or the
# constant has been raised until no realistic fleet could trip the guard.
RESUME_AGE_SAFETY_FACTOR = 1.5


def default_runs_db_path() -> Path:
    """Return the runs.db the live derived-bound guard measures.

    ``$RESUME_AGE_RUNS_DB`` when set, else the main checkout's absolute path.
    See :data:`RUNS_DB_ENV_VAR` for why this is not project_root-relative.
    """
    return Path(os.environ.get(RUNS_DB_ENV_VAR, _DEFAULT_RUNS_DB))


@dataclass(frozen=True)
class ResumeAgeSample:
    """What :func:`observed_resume_age_inputs` measured, terms and sizes both.

    ``inflight_max_secs`` is the longest LEGITIMATE invocation observed;
    ``downtime_max_secs`` the longest stretch in which the orchestrator emitted
    no event at all. The remaining fields travel with them so a skip or failure
    message can report what was actually measured rather than only what was
    concluded — the live guard's re-derivation recipe is built out of them.
    """

    inflight_max_secs: float
    downtime_max_secs: float
    inflight_rows: int
    gap_rows: int
    span_days: float


# Outcomes whose recorded duration reflects OPERATOR ACTION rather than how
# long a task can legitimately be in flight. Cancelling a task stops the clock
# at an arbitrary point, so including them would measure how long someone took
# to notice, which is not the quantity the bound is about.
_ILLEGITIMATE_OUTCOMES = ('cancelled', 'soft-cancelled')


def parsed_utc(value: str) -> datetime | None:
    """One ``events.timestamp`` as an aware UTC datetime, or None if unusable.

    PUBLIC because ``orchestrator.storm_window_bound`` imports it (task 3733):
    the two derived-bound samplers read the same column out of the same
    database, so they must not be able to drift in how they parse it.

    Annotated ``str`` because the column is declared TEXT NOT NULL, and the
    TypeError is still caught because sqlite DECLARES types rather than
    enforcing them: an unparseable row is not evidence of an outage, so it is
    skipped rather than raising out of a sampler whose callers treat an
    exception as a crash and a None as "too sparse to measure".
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def observed_resume_age_inputs(db_path: Path | str) -> ResumeAgeSample | None:
    """Measure both terms of the absolute resume bound from *db_path*.

    Opens the database READ-ONLY (``file:...?mode=ro``): this module derives a
    bound from production data and must never be able to write it.

    Returns ``None`` — never a zero-valued sample — when the database is
    absent, the ``events`` table is missing or empty, or fewer than
    :data:`MIN_SAMPLE_TASKS` legitimate in-flight rows survive filtering. A
    caller has to be able to tell "too sparse to measure" from "measured and
    fine"; a zero term would satisfy any bound and read as the latter.

    BOTH terms are scoped to the trailing :data:`SAMPLE_WINDOW_DAYS`, and
    ``span_days`` reports the span they were drawn from.

    The IN-FLIGHT term is ``max(duration_ms)`` over in-window ``task_completed``
    rows with a positive duration whose ``data.outcome`` is not in
    :data:`_ILLEGITIMATE_OUTCOMES`. A NULL duration is an UNMEASURED
    invocation rather than a zero-length one, and a non-positive duration is a
    clock artefact; neither is evidence and both are dropped.

    The DOWNTIME term is the largest gap between consecutive in-window event
    timestamps, over EVERY event type — any event at all proves the
    orchestrator was alive, so sampling only completions would read a
    quiet-but-running stretch as an outage and inflate the bound.

    Timestamps are parsed in Python rather than compared as strings: the SQL
    bound is only a prefilter, and a row whose timestamp does not parse is
    skipped rather than raising. The gap scan consumes the cursor in ONE pass,
    keeping only the running maximum and the two endpoints — the live db holds
    ~300k in-window rows, and materialising them (plus a second list of their
    gaps) cost seconds and tens of MB per call for numbers nothing reads. It
    relies on the query's ``ORDER BY timestamp``, which is sound for the same
    reason the ``timestamp >= ?`` prefilter is: every row is written by one
    code path as ``datetime.now(UTC).isoformat()``
    (``event_store.py::EventStore.emit``), a fixed-offset UTC spelling whose
    lexicographic order IS its chronological order.
    """
    path = Path(db_path)
    if not path.is_file():
        return None

    try:
        con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    except sqlite3.Error:
        logger.debug('resume_age_bound: cannot open %s read-only', path, exc_info=True)
        return None

    threshold = datetime.now(UTC) - timedelta(days=SAMPLE_WINDOW_DAYS)
    window_start = threshold.isoformat()

    first: datetime | None = None
    previous: datetime | None = None
    downtime_max_secs = 0.0
    gap_rows = 0
    try:
        placeholders = ', '.join('?' for _ in _ILLEGITIMATE_OUTCOMES)
        inflight_rows, inflight_max_ms = con.execute(
            'SELECT COUNT(*), MAX(duration_ms) FROM events '
            "WHERE event_type = 'task_completed' AND duration_ms > 0 "
            'AND timestamp >= ? '
            "AND (json_extract(data, '$.outcome') IS NULL "
            f"     OR json_extract(data, '$.outcome') NOT IN ({placeholders}))",
            (window_start, *_ILLEGITIMATE_OUTCOMES),
        ).fetchone()

        for (value,) in con.execute(
            'SELECT timestamp FROM events WHERE timestamp >= ? ORDER BY timestamp',
            (window_start,),
        ):
            parsed = parsed_utc(value)
            if parsed is None or parsed < threshold:
                continue
            if previous is None:
                first = parsed
            else:
                downtime_max_secs = max(
                    downtime_max_secs, (parsed - previous).total_seconds()
                )
                gap_rows += 1
            previous = parsed
    except sqlite3.Error:
        # A schema drift or a truncated db is "unmeasurable", not "zero".
        logger.debug('resume_age_bound: cannot sample %s', path, exc_info=True)
        return None
    finally:
        con.close()

    if not inflight_rows or inflight_rows < MIN_SAMPLE_TASKS or not inflight_max_ms:
        return None
    if first is None or previous is None or not gap_rows:
        return None  # no gap can be measured from fewer than two events
    if downtime_max_secs <= 0:
        return None  # degenerate: every event shares one instant

    return ResumeAgeSample(
        inflight_max_secs=inflight_max_ms / 1000,
        downtime_max_secs=downtime_max_secs,
        inflight_rows=inflight_rows,
        gap_rows=gap_rows,
        span_days=(previous - first).total_seconds() / 86400,
    )


def required_absolute_resume_age_secs(
    inflight_max_secs: float,
    downtime_max_secs: float,
    safety_factor: float,
) -> int:
    """Smallest ``absolute_resume_age_secs`` that cannot reject a live session.

    *inflight_max_secs* is the longest LEGITIMATE invocation observed and
    *downtime_max_secs* the longest stretch the orchestrator emitted nothing
    (see :func:`observed_resume_age_inputs`); *safety_factor* is the headroom
    multiplier over that plain sum (see :data:`RESUME_AGE_SAFETY_FACTOR`).

    WHY TWO TERMS, and why the second is not optional. A sidecar's
    ``started_at`` is stamped per INVOCATION
    (``workflow.py::TaskWorkflow._invoke``, at the ``write_agent_session``
    call), so its age when the ``_run_slot`` guard finally evaluates it is
    in-flight-time-at-crash PLUS however long the orchestrator was down before
    re-dispatching. The sidecar sits untouched across an outage, accruing age
    while nothing runs. Omitting the downtime term is not conservative — it is
    wrong for the quantity being bounded, and it would reject sessions from a
    task that never stopped being legitimate.

    The downtime term is also what makes the derivation HONEST rather than
    reverse-engineered. The in-flight term ALONE measured 8.90 h on
    2026-09-07, which cannot clear the 86,400 s ``freshness_window_secs`` at
    any safety factor a reviewer would accept — so a single-term derivation
    would force someone to pick a multiplier BECAUSE it cleared the
    constraint, which is the "magic number wearing a formula" that
    ``docs/legibility/design-invariants.md`` G6 and task 3621 both reject.
    T2's measured 56.99 h supplies a PHYSICAL reason the absolute bound
    exceeds freshness.

    Rounds UP: a fractional requirement must never round down into headroom
    the measured worst case does not leave.

    To RE-DERIVE the bound, measure the live db and read the answer here::

        sample = observed_resume_age_inputs(default_runs_db_path())
        required_absolute_resume_age_secs(
            sample.inflight_max_secs,
            sample.downtime_max_secs,
            RESUME_AGE_SAFETY_FACTOR,
        )
    """
    return math.ceil((inflight_max_secs + downtime_max_secs) * safety_factor)
