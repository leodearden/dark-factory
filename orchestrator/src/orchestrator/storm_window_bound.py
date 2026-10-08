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

NECESSARY, NOT SUFFICIENT — WHAT THE NOT-INERT SIDE DOES NOT ESTABLISH. It is
derived from the FAILURE inter-arrival series alone, and production has a
second term this module cannot sample: ``Harness.note_resume_succeeded``
retires the whole run on ANY surviving resume, fleet-wide, so two failures
inside the window do NOT chain if one resume succeeded between them.

MEASURED 2026-09-18 against ``data/orchestrator/runs.db``: 12
``session_resume_failed`` rows, every one stage='cli' — i.e. rejections of
in-workflow progress-timeout re-arms — against 8 ``session_resume`` rows, all
from the harness eligibility predicate and none from the arm seam. The arm-seam
SUCCESS population emits no event at all, so its size is not merely unmeasured
but unmeasurable from runs.db, and the interleaved series cannot be
reconstructed after the fact.

So the inequality bounds the window from below and nothing more: clearing it
does not prove the escape will fire, only that a window below it could not.
``Harness.note_resume_succeeded`` now COUNTS the reset population
(``_session_resume_survivals``, reported on the storm L1 so an operator can
tell a quiet fleet from a constantly-reset streak) — the instrument a future
derivation of the sufficient side needs, and one that cannot be backfilled for
a population nothing ever recorded. Until that series exists, this module
states the limit rather than letting an inequality imply a fireability it does
not establish.

WHAT THE SAMPLER MEASURES, AND WHAT IT CARVES OUT. ``session_resume_failed`` is
the population the streak feeds on — but not all of it feeds:
``harness._BY_DESIGN_RESTORE_OUTCOMES`` ('disabled', the kill switch, and
'miss', the archive-COVERAGE signal) neither feed the streak nor reset it, so
rows carrying them are excluded here too. Sampling the superset instead would
NOT be "conservative"; that word is true of exactly one side:

  - NO FALSE ALARM: a superset can only chain LONGER runs, so it is strictly
    stricter. Safe either way.
  - NOT INERT: a superset can only make the minimum gap SMALLER, which makes
    the necessary condition EASIER to satisfy — the opposite of conservative,
    and the reason the filter is applied rather than waived.

At derivation time the two populations coincide (all 12 rows are stage='cli'
with ``restore`` absent), so the filter changes no number here. It is applied
because the design explicitly anticipates 'miss' rows arriving, at which point
the unfiltered not-inert side would start reading a floor the streak's real
feeder never produced. The pre_flight restore faults ε added (0 rows at
derivation time) DO feed, and can only shorten gaps — that direction is
genuinely conservative for the shipped window.

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
    'MEASURED_ARRIVALS_PER_DAY',
    'MEASURED_GAPS_SECS',
    'MEASURED_LONGEST_RUN_BY_WINDOW',
    'MEASURED_MIN_GAP_SECS',
    'MEASURED_ROWS',
    'MEASURED_RUN_REACHES_FIVE_ABOVE_SECS',
    'MEASURED_SPAN',
    'MEASURED_SPAN_DAYS',
    'MIN_GAP_ROWS',
    'RUNS_DB_ENV_VAR',
    'SAMPLE_WINDOW_DAYS',
    'StormWindowSample',
    'default_runs_db_path',
    'longest_chained_run',
    'observed_storm_window_inputs',
    'parsed_utc',
]

# Trailing window this bound is sampled over, in days. DELIBERATELY NARROWER
# than ``resume_age_bound.SAMPLE_WINDOW_DAYS`` (90, the archive retention) and
# strictly NESTED inside it, so both bounds still describe ONE fleet read out of
# ONE database over overlapping corpora — the SPOT that matters is the LOCATOR
# and the TIMESTAMP PARSER, which are imported above, not the width of the
# slice each bound is meaningful over.
#
# WHY NARROWER. The no-false-alarm side samples the fleet's ordinary failure
# arrivals as a NULL, and nothing excludes a genuine incident from that null.
# So the first real session-resume storm — precisely the event INV-4's L1 exists
# to page for — ALSO reddens the live guard for every task's verify across the
# fleet, and keeps it red until the incident ages out of the sample. A quarter
# of that is hostage-taking with no remedy available to a task agent; a month is
# survivable, and still spans the whole 22.8-day era the measured population
# occupies.
#
# WHY NOT NARROWER STILL: below MIN_GAP_ROWS + 1 arrivals the sampler returns
# None and the guard goes decorative. At MEASURED_ARRIVALS_PER_DAY that floor
# needs 11.4 days, so 30 carries 2.6x margin against going silent while capping
# incident contamination at a month. Both ends are measurements; neither is a
# chosen ratio (design-invariants G6).
SAMPLE_WINDOW_DAYS = 30

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
MEASUREMENT_DATE = '2026-09-18'

# The population: every ``session_resume_failed`` row in runs.db, all-time.
# ALL stage='cli', ALL role='implementer'; stage='pre_flight' had 0 rows, and
# no row carried a by-design ``restore``, so the feeder filter the sampler
# applies excluded nothing.
MEASURED_ROWS = 12
MEASURED_SPAN = '2026-08-24..2026-09-16'
MEASURED_SPAN_DAYS = 22.780

# Arrivals per day over that span. Load-bearing, not colour: it is what
# :data:`SAMPLE_WINDOW_DAYS` is derived from — a trailing window has to hold at
# least ``MIN_GAP_ROWS + 1`` arrivals at this rate or the sampler returns None
# and the live guard goes decorative.
MEASURED_ARRIVALS_PER_DAY = MEASURED_ROWS / MEASURED_SPAN_DAYS

# THE SERIES ITSELF, oldest first, in seconds — not prose. Every other measured
# constant below is a function OF this series, so recording it is what lets a
# test re-derive them with :func:`longest_chained_run` instead of trusting four
# hand-copied numbers. Full precision on purpose: the ceiling is one of these
# values exactly, and a rounded copy would sit on the wrong side of it.
#
# Sorted, in hours: 3.19, 5.82, 8.73, 30.58, 30.90, 34.15, 44.66, 53.00, 55.12,
# 62.55, 218.01.
MEASURED_GAPS_SECS = (
    225_195.852766,   # 62.55 h
    784_823.137243,   # 218.01 h
    31_432.159035,    # 8.73 h
    20_946.880065,    # 5.82 h
    122_929.320631,   # 34.15 h
    190_806.541744,   # 53.00 h
    160_793.874421,   # 44.66 h
    110_094.339358,   # 30.58 h
    198_425.969649,   # 55.12 h
    111_232.759746,   # 30.90 h
    11_496.166019,    # 3.19 h
)

# NOT-INERT side. The smallest observed inter-arrival: 3.19 h. At the pre-ε
# default of 3600 s the longest run this population admits is 1 — the escape was
# unfireable at any threshold.
MEASURED_MIN_GAP_SECS = 11_496.166019

# NO-FALSE-ALARM side. The longest chained run the measured NULL produces, by
# window, from :func:`longest_chained_run` over MEASURED_GAPS_SECS. A mapping
# rather than prose so the recorded table can be checked against the live
# function over the recorded series — which
# test_storm_window_bound.py::test_the_recorded_table_is_what_the_function_says
# does, for every entry, so no row of it is unverifiable prose wearing a dict.
MEASURED_LONGEST_RUN_BY_WINDOW = {
    3_600: 1,     # 1 h — the pre-ε default
    21_600: 2,    # 6 h
    43_200: 3,    # 12 h
    86_400: 3,    # 24 h — the shipped default
    172_800: 4,   # 48 h
}

# The null first chains a run of 5 — the shipped ``fallback_storm_threshold`` —
# only at a window strictly ABOVE 53.00 h, so that is the open upper end of the
# admissible interval. It is a MEMBER of MEASURED_GAPS_SECS, exactly: the run
# that reaches 5 is the one this gap was breaking, so the boundary is the gap
# value itself and rounding it up (as an earlier revision did, to 190_806.542)
# puts it on the wrong side — at the rounded value the null already chains 7.
#
# Hence: admissible interval [3.19 h, 53.00 h) = [11,496 s, 190,807 s). The
# shipped 86,400 s (24 h) sits inside it with 7.52x margin above the floor and
# 2.21x below the ceiling, and exceeds any plausible orchestrator boot lifetime
# — so a BOOT bounds a run rather than an arbitrary sub-boot clock.
# ``fallback_storm_threshold`` STAYS 5: the window, not the threshold, was the
# binding constraint, and 5 sits two above the null's measured maximum of 3.
MEASURED_RUN_REACHES_FIVE_ABOVE_SECS = 190_806.541744


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

    Only ``session_resume_failed`` rows are sampled — sampling anything else
    would measure how busy the fleet is rather than how tightly this population
    arrives — and within them only the rows that actually FEED the streak:
    ``harness._BY_DESIGN_RESTORE_OUTCOMES`` is carved out at the classifier, so
    a row carrying one of those outcomes never increments a streak and must not
    shorten a measured gap either. The carve-out set is imported rather than
    re-typed (SPOT); see the module docstring for why sampling the superset
    would be conservative on one side of the bound and ANTI-conservative on the
    other. The import is function-local so a 300-line derivation module does
    not drag the orchestrator's largest module into its import graph.

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

    from orchestrator.harness import _BY_DESIGN_RESTORE_OUTCOMES  # noqa: PLC0415

    carved = sorted(_BY_DESIGN_RESTORE_OUTCOMES)
    threshold = datetime.now(UTC) - timedelta(days=SAMPLE_WINDOW_DAYS)
    stamps: list[datetime] = []
    try:
        for (value,) in con.execute(
            "SELECT timestamp FROM events WHERE event_type = 'session_resume_failed' "
            'AND timestamp >= ? '
            # A cli-stage rejection carries no restore outcome at all, and IS a
            # feeder — so an absent/NULL restore is KEPT, and only the named
            # by-design outcomes are dropped. Bound as parameters, never
            # interpolated, so the carve-out set stays data.
            "AND coalesce(json_extract(data, '$.restore'), '') NOT IN "
            f'({", ".join("?" * len(carved))}) '
            'ORDER BY timestamp',
            (threshold.isoformat(), *carved),
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
