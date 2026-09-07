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

# Trailing window the DOWNTIME term is sampled over. Matches the 90-day
# retention window the transcript archive keeps, and bounds the derivation to
# the fleet's CURRENT behaviour: without it a single ancient row would
# manufacture a phantom multi-month gap between itself and the recent corpus,
# and the bound would ratchet up forever instead of re-deriving.
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


def observed_resume_age_inputs(db_path: Path | str) -> ResumeAgeSample | None:
    """Measure both terms of the absolute resume bound from *db_path*.

    Opens the database READ-ONLY (``file:...?mode=ro``): this module derives a
    bound from production data and must never be able to write it.

    Returns ``None`` — never a zero-valued sample — when the database is
    absent, the ``events`` table is missing or empty, or fewer than
    :data:`MIN_SAMPLE_TASKS` legitimate in-flight rows survive filtering. A
    caller has to be able to tell "too sparse to measure" from "measured and
    fine"; a zero term would satisfy any bound and read as the latter.

    The IN-FLIGHT term is ``max(duration_ms)`` over ``task_completed`` rows
    with a positive duration whose ``data.outcome`` is not in
    :data:`_ILLEGITIMATE_OUTCOMES`. A NULL duration is an UNMEASURED
    invocation rather than a zero-length one, and a non-positive duration is a
    clock artefact; neither is evidence and both are dropped.

    The DOWNTIME term is the largest gap between consecutive event timestamps
    inside the trailing :data:`SAMPLE_WINDOW_DAYS`, over EVERY event type —
    any event at all proves the orchestrator was alive, so sampling only
    completions would read a quiet-but-running stretch as an outage and
    inflate the bound.

    Timestamps are parsed in Python rather than compared as strings: the SQL
    bound is only a prefilter, and a row whose timestamp does not parse is
    skipped rather than raising.
    """
    path = Path(db_path)
    if not path.is_file():
        return None

    try:
        con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    except sqlite3.Error:
        logger.debug('resume_age_bound: cannot open %s read-only', path, exc_info=True)
        return None

    try:
        placeholders = ', '.join('?' for _ in _ILLEGITIMATE_OUTCOMES)
        inflight_rows, inflight_max_ms = con.execute(
            'SELECT COUNT(*), MAX(duration_ms) FROM events '
            "WHERE event_type = 'task_completed' AND duration_ms > 0 "
            "AND (json_extract(data, '$.outcome') IS NULL "
            f"     OR json_extract(data, '$.outcome') NOT IN ({placeholders}))",
            _ILLEGITIMATE_OUTCOMES,
        ).fetchone()

        threshold = datetime.now(UTC) - timedelta(days=SAMPLE_WINDOW_DAYS)
        raw = con.execute(
            'SELECT timestamp FROM events WHERE timestamp >= ? ORDER BY timestamp',
            (threshold.isoformat(),),
        ).fetchall()
    except sqlite3.Error:
        # A schema drift or a truncated db is "unmeasurable", not "zero".
        logger.debug('resume_age_bound: cannot sample %s', path, exc_info=True)
        return None
    finally:
        con.close()

    if not inflight_rows or inflight_rows < MIN_SAMPLE_TASKS or not inflight_max_ms:
        return None

    stamps: list[datetime] = []
    for (value,) in raw:
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError):
            continue  # an unparseable row is not evidence of an outage
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        if parsed >= threshold:
            stamps.append(parsed)
    if len(stamps) < 2:
        return None  # no gap can be measured from fewer than two events

    stamps.sort()
    gaps = [
        (stamps[i + 1] - stamps[i]).total_seconds() for i in range(len(stamps) - 1)
    ]
    downtime_max_secs = max(gaps)
    if downtime_max_secs <= 0:
        return None  # degenerate: every event shares one instant

    return ResumeAgeSample(
        inflight_max_secs=inflight_max_ms / 1000,
        downtime_max_secs=downtime_max_secs,
        inflight_rows=inflight_rows,
        gap_rows=len(gaps),
        span_days=(stamps[-1] - stamps[0]).total_seconds() / 86400,
    )
