"""Derivation of the rolling window a run of failed resumes must arrive inside.

Task ε/3733: ``session_resume.storm_window_secs`` is the width of the chain the
INV-4 storm escape counts over — two eligible-but-FAILED resumes belong to the
same run only if they arrive within it. It is a DERIVED bound, not a chosen
number: this module holds the sampler that measures its input against runs.db
and the pure function that turns a gap series into the longest run it admits,
so the live guard (``orchestrator/tests/test_storm_window_bound.py``) and its
host-independent falsifiability companion consume ONE implementation rather
than two re-derivations that can drift apart.

WHY IT HAD TO BE RE-DERIVED. The shipped 3600 s came from the
``session_resume_fallback`` burst signature ("real bursts land ~17 fallbacks
inside one hour") — a population task 3728 entirely carved out of the streak.
It was a stale inheritance describing a feeder that no longer exists, and
freezing a replacement in a config comment would reproduce exactly the rot this
task exists to fix. Task 3730 landed the remedy shape in this same config
submodel (``resume_age_bound.py`` plus a guard that re-derives every run), so
this is its focused sibling.

WHY THE BOUND IS TWO-SIDED, AND WHY NEITHER SIDE IS A MULTIPLIER.
``docs/legibility/design-invariants.md`` G6 rejects "a magic number wearing a
formula", so neither side here is a safety factor anyone chose:

  NOT INERT — ``min observed inter-arrival <= storm_window_secs``. Below the
  smallest gap the population actually produces, no two failures can EVER
  chain, so the escape is unfireable at ANY threshold. That is a strict
  NECESSARY CONDITION derived from the data, not a ratio; it is also why the
  WINDOW rather than the threshold was the binding constraint on 3600 s.

  NO FALSE ALARM — ``longest_chained_run(gaps, window) < threshold``. At the
  shipped window the longest run the measured NULL produces must stay strictly
  below the threshold, or ordinary background failures page an operator. That
  is a direct COUNT over the measured series, not a projection.

Together they bracket an interval, and the live guard asserts INTERVAL
MEMBERSHIP rather than equality to a point — so an ordinary fleet drift does
not turn the suite red while a real one does.

WHY THE SAMPLER MEASURES ``session_resume_failed``. That event is exactly the
population the streak now feeds on, at both stages, and sampling it is
CONSERVATIVE: after task 3733 the feeder additionally admits pre_flight restore
faults (0 rows at derivation time), which can only shorten gaps and make
chaining easier, never harder.

Sited as a focused SIBLING of ``resume_age_bound.py`` rather than folded into
it: that module declares a single purpose in its own first line — the outer
bound on a recovered sidecar's AGE — and a second, unrelated bound would dilute
it. The scaffolding is imported rather than duplicated, so every derived-bound
guard in this submodel resolves ONE runs.db over ONE trailing window and parses
``events.timestamp`` ONE way.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from orchestrator.resume_age_bound import (
    RUNS_DB_ENV_VAR,
    SAMPLE_WINDOW_DAYS,
    default_runs_db_path,
    parsed_utc,
)

logger = logging.getLogger(__name__)

# Re-exported for the guard's own skip message, which names the env var a
# reader would have to set. Imported rather than redeclared: a second locator
# would be a second answer to "which database is the fleet?" (SPOT). The name
# reads narrower than its two consumers now are — recorded rather than hidden;
# renaming a just-landed module's env var is out of scope here, and a second
# var would be strictly worse.
__all__ = [
    'MEASUREMENT_DATE',
    'MEASURED_LONGEST_RUN_BY_WINDOW',
    'MEASURED_MIN_GAP_SECS',
    'MEASURED_ROWS',
    'MEASURED_RUN_REACHES_FIVE_ABOVE_SECS',
    'MEASURED_SPAN',
    'MIN_GAP_ROWS',
    'RUNS_DB_ENV_VAR',
    'SAMPLE_WINDOW_DAYS',
    'StormWindowSample',
    'default_runs_db_path',
    'longest_chained_run',
    'observed_storm_window_inputs',
    'parsed_utc',
]

# Smallest number of inter-arrivals :func:`observed_storm_window_inputs` will
# report from. Below it the sampler returns None rather than a weak number, and
# the distinction is load-bearing: ABSENCE MUST NEVER READ AS A ZERO, because a
# zero minimum gap satisfies the not-inert side of the bound trivially and would
# leave the live guard permanently green while measuring nothing.
#
# Deliberately NOT ``resume_age_bound.MIN_SAMPLE_TASKS`` (50). This feeder is
# genuinely sparse — 10 rows in 22 days at derivation time — so inheriting that
# floor would make the guard skip on every run, i.e. decorative. Five gaps (six
# failures) is enough to refuse a corpus of one or two isolated rows while
# staying live on the population that actually exists.
MIN_GAP_ROWS = 5

# THE MEASUREMENT the shipped defaults were derived from, as CONSTANTS rather
# than prose — the whole point, given the value they replace went stale exactly
# by being a comment. Read by the tests (and by
# test_crash_recovery.py's shipped-defaults row, which spaces its reports at
# MEASURED_MIN_GAP_SECS rather than re-typing the number), so one re-derivation
# updates one place.
#
# MEASURED on this host against data/orchestrator/runs.db, read-only, by the
# sampler below.
MEASUREMENT_DATE = '2026-09-16'

# The population: every ``session_resume_failed`` row in runs.db, all-time.
# ALL stage='cli', ALL role='implementer'; stage='pre_flight' had 0 rows.
MEASURED_ROWS = 10
MEASURED_SPAN = '2026-08-24..2026-09-15'

# NOT-INERT side. The smallest observed inter-arrival: 5.82 h. The nine gaps,
# sorted (hours): 5.82, 8.73, 30.58, 34.15, 44.66, 53.00, 55.12, 62.55, 218.01.
# At the pre-ε default of 3600 s the longest run the population admits is 1 —
# the escape was unfireable at any threshold.
MEASURED_MIN_GAP_SECS = 20_946.880

# NO-FALSE-ALARM side. The longest chained run the measured NULL produces, by
# window, from :func:`longest_chained_run` over those gaps. A mapping rather
# than prose so a test can check the recorded table against the live function
# and the table cannot quietly drift from the code that produced it.
MEASURED_LONGEST_RUN_BY_WINDOW = {
    3_600: 1,     # 1 h — the pre-ε default
    21_600: 2,    # 6 h
    43_200: 3,    # 12 h
    86_400: 3,    # 24 h — the shipped default
    172_800: 4,   # 48 h
}

# The null first chains a run of 5 — the shipped ``fallback_storm_threshold`` —
# only at a window strictly ABOVE 53.00 h, so that is the open upper end of the
# admissible interval.
#
# Hence: admissible interval [5.82 h, 53.00 h) = [20,947 s, 190,807 s). The
# shipped 86,400 s (24 h) sits inside it with 4.12x margin above the floor and
# 2.21x below the ceiling, and exceeds any plausible orchestrator boot lifetime
# — so a BOOT bounds a run rather than an arbitrary sub-boot clock.
# ``fallback_storm_threshold`` STAYS 5: the window, not the threshold, was the
# binding constraint, and 5 sits two above the null's measured maximum of 3.
MEASURED_RUN_REACHES_FIVE_ABOVE_SECS = 190_806.542


@dataclass(frozen=True)
class StormWindowSample:
    """What :func:`observed_storm_window_inputs` measured.

    ``gaps_secs`` is the whole inter-arrival series, oldest first, because the
    no-false-alarm side of the bound is a run count over the SERIES and cannot
    be computed from a summary. ``min_gap_secs`` and ``gap_rows`` are derived
    from it rather than stored beside it, so the number a skip message reports
    and the series the guard chains over cannot disagree.
    """

    gaps_secs: tuple[float, ...]
    span_days: float

    @property
    def min_gap_secs(self) -> float:
        return min(self.gaps_secs)

    @property
    def gap_rows(self) -> int:
        return len(self.gaps_secs)


def longest_chained_run(gaps_secs: Sequence[float] | Iterable[float],
                        window_secs: float) -> int:
    """Longest run of failures that chain at *window_secs*.

    *gaps_secs* is the series of intervals between consecutive failures, so a
    run of N failures shows up as N-1 consecutive gaps under the window. An
    EMPTY series is a run of 1 — one failure that nothing chained to — never 0,
    which would be the different claim that nothing failed at all and would
    make the no-false-alarm side vacuously satisfiable.

    STRICTLY under, matching the production decay exactly:
    ``Harness.note_resume_failed`` retires a run once the elapsed time is
    ``>= storm_window_secs``, so a gap equal to the window ENDS the run. A
    ``<=`` here would derive the bound from a chaining rule the code does not
    implement.

    Monotonic non-decreasing in *window_secs* — a wider window can never chain
    fewer failures — which is what makes the admissible interval an interval.
    """
    best = run = 1
    for gap in gaps_secs:
        run = run + 1 if gap < window_secs else 1
        best = max(best, run)
    return best


def observed_storm_window_inputs(db_path: Path | str) -> StormWindowSample | None:
    """Measure the failed-resume inter-arrival series in *db_path*.

    Opens the database READ-ONLY (``file:...?mode=ro``): this module derives a
    bound from production data and must never be able to write it.

    Returns ``None`` — never a zero-valued sample — when the database is
    absent, the ``events`` table is missing or empty, or fewer than
    :data:`MIN_GAP_ROWS` inter-arrivals survive. A caller has to be able to
    tell "too sparse to measure" from "measured and fine"; a zero minimum gap
    would satisfy the not-inert side of the bound and read as the latter.

    Scoped to the trailing :data:`SAMPLE_WINDOW_DAYS` so the bound describes the
    fleet's CURRENT behaviour: without it a single ancient row would manufacture
    a phantom multi-month gap between itself and the recent corpus, and the
    not-inert side would ratchet the window up instead of re-deriving it.

    Only ``session_resume_failed`` rows are sampled — exactly the population the
    streak feeds on. Sampling anything else would measure how busy the fleet is
    rather than how tightly this population arrives.

    Timestamps are parsed in Python rather than compared as strings: the SQL
    bound is only a prefilter, and a row whose timestamp does not parse is
    skipped rather than raising. The ``ORDER BY timestamp`` is sound because
    every row is written by one code path as ``datetime.now(UTC).isoformat()``
    (``event_store.py::EventStore.emit``), a fixed-offset UTC spelling whose
    lexicographic order IS its chronological order.
    """
    path = Path(db_path)
    if not path.is_file():
        return None

    try:
        con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    except sqlite3.Error:
        logger.debug('storm_window_bound: cannot open %s read-only', path,
                     exc_info=True)
        return None

    threshold = datetime.now(UTC) - timedelta(days=SAMPLE_WINDOW_DAYS)
    stamps: list[datetime] = []
    try:
        for (value,) in con.execute(
            "SELECT timestamp FROM events WHERE event_type = 'session_resume_failed' "
            'AND timestamp >= ? ORDER BY timestamp',
            (threshold.isoformat(),),
        ):
            parsed = parsed_utc(value)
            if parsed is not None and parsed >= threshold:
                stamps.append(parsed)
    except sqlite3.Error:
        # A schema drift or a truncated db is "unmeasurable", not "zero".
        logger.debug('storm_window_bound: cannot sample %s', path, exc_info=True)
        return None
    finally:
        con.close()

    gaps = tuple(
        (later - earlier).total_seconds()
        for earlier, later in zip(stamps, stamps[1:], strict=False)
    )
    if len(gaps) < MIN_GAP_ROWS:
        return None
    if min(gaps) <= 0:
        return None  # degenerate: two failures share one instant

    return StormWindowSample(
        gaps_secs=gaps,
        span_days=(stamps[-1] - stamps[0]).total_seconds() / 86400,
    )
