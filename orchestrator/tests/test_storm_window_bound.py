"""Tests for :mod:`orchestrator.storm_window_bound` (task ε/3733).

The module derives ``session_resume.storm_window_secs`` — the rolling window
inside which consecutive eligible-but-FAILED resumes must arrive to count as
one run — from the MEASURED inter-arrival signature of the population that
actually feeds the INV-4 storm streak, rather than from the
``session_resume_fallback`` burst signature the shipped 3600 s inherited and
that task 3728 entirely carved out.

Structure mirrors ``test_resume_age_bound.py`` (task 3730), the derived-bound
precedent in this very config submodel: a SYNTHETIC-corpus sampler suite and a
host-independent falsifiability proof that are meaningful in any checkout, plus
a LIVE guard that re-derives against the real runs.db and SKIPS (never fails)
where that data is absent.

The bound is TWO-SIDED and both sides are measurements, not safety-factor
multipliers — see the module docstring for why neither is a ratio anyone chose.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestrator import storm_window_bound as swb
from orchestrator.config import SessionResumeConfig

# Mirrors the `events` table of orchestrator/src/orchestrator/event_store.py —
# copied rather than imported so a schema drift shows up as a red test here
# instead of the synthetic corpus silently tracking a changed production shape.
# Same rationale, same DDL as test_resume_age_bound.py.
_EVENTS_DDL = """
CREATE TABLE events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    run_id      TEXT    NOT NULL,
    task_id     TEXT,
    event_type  TEXT    NOT NULL,
    phase       TEXT,
    role        TEXT,
    data        TEXT    DEFAULT '{}',
    cost_usd    REAL,
    duration_ms INTEGER
)
"""


def _make_db(path: Path, rows: list[dict]) -> Path:
    """Write a synthetic runs.db holding exactly *rows* and return its path."""
    con = sqlite3.connect(path)
    try:
        con.execute(_EVENTS_DDL)
        con.executemany(
            'INSERT INTO events (timestamp, run_id, task_id, event_type, data) '
            'VALUES (?, ?, ?, ?, ?)',
            [
                (
                    row['timestamp'],
                    row.get('run_id', 'run-1'),
                    row.get('task_id', '1'),
                    row['event_type'],
                    json.dumps(row.get('data', {})),
                )
                for row in rows
            ],
        )
        con.commit()
    finally:
        con.close()
    return path


def _failures(starts: list[datetime]) -> list[dict]:
    """``session_resume_failed`` rows at exactly *starts* — the sampled
    population."""
    return [
        {'timestamp': when.isoformat(), 'event_type': 'session_resume_failed'}
        for when in starts
    ]


def _spaced(anchor: datetime, gaps: list[timedelta]) -> list[datetime]:
    """Instants whose consecutive differences are exactly *gaps*."""
    out = [anchor]
    for gap in gaps:
        out.append(out[-1] + gap)
    return out


def _enough(anchor: datetime, gap: timedelta = timedelta(hours=6)) -> list[timedelta]:
    """Just enough evenly-spaced gaps to clear the sampler's minimum."""
    return [gap] * swb.MIN_GAP_ROWS


@pytest.fixture
def anchor() -> datetime:
    """A recent instant, comfortably inside the sampler's trailing window.

    Ten days rather than thirty: the trailing window is now this module's own
    narrower one (see ``swb.SAMPLE_WINDOW_DAYS``), and a corpus anchored at its
    exact edge would be sampled or not depending on microseconds.
    """
    return datetime.now(UTC) - timedelta(days=10)


# ---------------------------------------------------------------------------
# (i) HOST-INDEPENDENT unit rows: the pure function and the sampler.
#
# ABSENCE MUST NEVER READ AS A ZERO. Every "not enough data" arm returns None,
# not a sample carrying 0, because a zero min-gap would satisfy the not-inert
# side of the bound trivially and leave the live guard permanently green while
# measuring nothing (the vacuity trap 3730 and 3621 both record).
# ---------------------------------------------------------------------------


def test_longest_run_is_one_for_an_empty_gap_list():
    """No gaps means no chain: a single failure is a run of one, never zero.

    Zero would read as "nothing failed", which is a different claim from "one
    thing failed and nothing chained to it" — and it is the claim that would
    make the false-alarm side of the bound vacuously satisfiable.
    """
    assert swb.longest_chained_run((), 3600) == 1


def test_longest_run_counts_only_gaps_strictly_under_the_window():
    """The comparison must match the production decay EXACTLY.

    ``Harness.note_resume_failed`` retires a run once the elapsed time is
    ``>= storm_window_secs``, so a gap exactly equal to the window ENDS the
    run. A ``<=`` here would derive the bound from a chaining rule the code
    does not implement, and the guard would be measuring a different system.
    """
    gaps = (100.0, 100.0)
    assert swb.longest_chained_run(gaps, 101) == 3
    assert swb.longest_chained_run(gaps, 100) == 1


def test_longest_run_is_monotonic_non_decreasing_in_the_window():
    """A wider window can never chain FEWER failures.

    This is what makes the admissible interval an interval at all: the
    false-alarm side is an upper bound precisely because widening the window
    can only lengthen the null's longest run.
    """
    gaps = (60.0, 3600.0, 120.0, 7200.0, 90.0)
    runs = [swb.longest_chained_run(gaps, w) for w in (1, 100, 1000, 5000, 10_000)]
    assert runs == sorted(runs)


def test_longest_run_resets_on_a_gap_at_or_over_the_window():
    """A chain broken in the middle is two runs, not one long one."""
    gaps = (10.0, 10.0, 1_000_000.0, 10.0)
    assert swb.longest_chained_run(gaps, 60) == 3


def test_absent_db_returns_none(tmp_path):
    assert swb.observed_storm_window_inputs(tmp_path / 'nope.db') is None


def test_empty_events_table_returns_none(tmp_path):
    assert swb.observed_storm_window_inputs(_make_db(tmp_path / 'e.db', [])) is None


def test_missing_events_table_returns_none(tmp_path):
    """A schema drift or a truncated db is UNMEASURABLE, not zero."""
    db = tmp_path / 'noschema.db'
    sqlite3.connect(db).close()
    assert swb.observed_storm_window_inputs(db) is None


def test_sample_below_the_minimum_gap_count_returns_none(tmp_path, anchor):
    """Too sparse to measure must be distinguishable from measured-and-fine."""
    starts = _spaced(anchor, [timedelta(hours=6)] * (swb.MIN_GAP_ROWS - 1))
    db = _make_db(tmp_path / 'sparse.db', _failures(starts))
    assert swb.observed_storm_window_inputs(db) is None


def test_sampler_measures_only_the_feeder_population(tmp_path, anchor):
    """Only ``session_resume_failed`` rows are sampled.

    The streak's feeder is eligible-but-FAILED resumes. Sampling any other
    event would measure how busy the fleet is rather than how tightly this
    population arrives, and would shrink the measured inter-arrival toward
    zero — making the not-inert side trivially satisfiable.
    """
    starts = _spaced(anchor, _enough(anchor))
    noise = [
        {'timestamp': (anchor + timedelta(minutes=i)).isoformat(),
         'event_type': 'session_resume_fallback'}
        for i in range(50)
    ]
    db = _make_db(tmp_path / 'mixed.db', _failures(starts) + noise)

    sample = swb.observed_storm_window_inputs(db)
    assert sample is not None
    assert sample.gap_rows == swb.MIN_GAP_ROWS
    assert sample.min_gap_secs == timedelta(hours=6).total_seconds()


def test_sampler_reports_what_it_measured(tmp_path, anchor):
    """The sample carries its own provenance, so a skip or failure message can
    say what it saw rather than only what it concluded."""
    gaps = [timedelta(hours=9)] * (swb.MIN_GAP_ROWS - 1) + [timedelta(hours=4)]
    starts = _spaced(anchor, gaps)
    db = _make_db(tmp_path / 'report.db', _failures(starts))

    sample = swb.observed_storm_window_inputs(db)
    assert sample is not None
    assert sample.gaps_secs == tuple(g.total_seconds() for g in gaps)
    assert sample.min_gap_secs == timedelta(hours=4).total_seconds()
    assert sample.gap_rows == len(gaps)
    assert sample.span_days == pytest.approx(
        sum(gaps, timedelta()).total_seconds() / 86400
    )


def test_rows_outside_the_trailing_window_are_not_sampled(tmp_path, anchor):
    """The bound describes the fleet's CURRENT behaviour.

    Without the trailing-window scoping a single ancient row would manufacture
    a phantom multi-month gap between itself and the recent corpus, and the
    not-inert side would ratchet the window up instead of re-deriving it.
    """
    ancient = datetime.now(UTC) - timedelta(days=swb.SAMPLE_WINDOW_DAYS + 30)
    starts = _spaced(anchor, _enough(anchor))
    db = _make_db(
        tmp_path / 'aged.db', _failures([ancient, *starts]),
    )

    sample = swb.observed_storm_window_inputs(db)
    assert sample is not None
    assert sample.gap_rows == swb.MIN_GAP_ROWS
    assert sample.min_gap_secs == timedelta(hours=6).total_seconds()


def test_sampler_opens_the_db_read_only(tmp_path, anchor):
    """The sampler must never be able to write the production runs.db.

    Pinned by observation rather than by reading the connect string: a
    read-only connection cannot create the journal/WAL sidecars a write would,
    and the corpus is byte-identical after sampling.
    """
    starts = _spaced(anchor, _enough(anchor))
    db = _make_db(tmp_path / 'runs.db', _failures(starts))
    before = db.read_bytes()

    assert swb.observed_storm_window_inputs(db) is not None

    assert db.read_bytes() == before
    assert not (tmp_path / 'runs.db-journal').exists()
    assert not (tmp_path / 'runs.db-wal').exists()


def test_the_sampler_excludes_the_outcomes_the_streak_carves_out(
    tmp_path, anchor,
):
    """A row the CLASSIFIER ignores must not shorten a measured gap.

    ``harness._BY_DESIGN_RESTORE_OUTCOMES`` never increments a streak, so
    sampling those rows would measure a population the bound is not about. The
    error is one-sided and easy to mistake for caution: a superset chains
    LONGER runs (stricter on the no-false-alarm side) but yields a SMALLER
    minimum gap, which makes the not-inert necessary condition easier to
    satisfy — anti-conservative exactly where the derivation leans on it.

    A cli-stage rejection carries no ``restore`` at all and IS a feeder, so an
    absent key must be KEPT; that is the arm the coalesce in the SELECT exists
    for, and dropping it would sample nothing at all on today's live corpus.
    """
    from orchestrator.harness import (  # noqa: PLC0415
        _BY_DESIGN_RESTORE_OUTCOMES,
    )

    feeders = _spaced(anchor, _enough(anchor))
    rows: list[dict] = _failures(feeders)
    # One carved-out row wedged a minute after each feeder: were it sampled,
    # the minimum gap would collapse from six hours to one minute.
    for outcome in sorted(_BY_DESIGN_RESTORE_OUTCOMES):
        rows += [
            {'timestamp': (when + timedelta(minutes=1)).isoformat(),
             'event_type': 'session_resume_failed',
             'data': {'stage': 'pre_flight', 'restore': outcome}}
            for when in feeders
        ]
    # ...and a genuine pre_flight fault, which DOES feed and is kept.
    genuine = feeders[-1] + timedelta(hours=6)
    rows.append({'timestamp': genuine.isoformat(),
                 'event_type': 'session_resume_failed',
                 'data': {'stage': 'pre_flight', 'restore': 'fault'}})

    sample = swb.observed_storm_window_inputs(_make_db(tmp_path / 'c.db', rows))
    assert sample is not None
    assert sample.gap_rows == swb.MIN_GAP_ROWS + 1
    assert sample.min_gap_secs == timedelta(hours=6).total_seconds()


def test_the_runs_db_locator_is_shared_with_resume_age_bound():
    """ONE runs.db locator and ONE timestamp parser for every derived-bound
    guard in this submodel — but not necessarily one slice width.

    Two locators would be two answers to "which database is the fleet?" and
    two parsers two answers to "what does a timestamp mean", so those are
    imported, never redeclared.

    The TRAILING WINDOW is deliberately this module's own and NARROWER, because
    a genuine storm contaminates this bound's null corpus and would otherwise
    hold the whole fleet's verify red for a quarter. The original rationale for
    sharing it — that two bounds must not be derived from corpora that do not
    overlap while both claiming to describe one fleet — is preserved by
    NESTING: a strictly narrower window is a subset of the wider one by
    construction, so the corpora always overlap.
    """
    from orchestrator import resume_age_bound as rab  # noqa: PLC0415

    assert swb.default_runs_db_path is rab.default_runs_db_path
    assert swb.parsed_utc is rab.parsed_utc
    assert 0 < swb.SAMPLE_WINDOW_DAYS < rab.SAMPLE_WINDOW_DAYS


def test_the_trailing_window_is_wide_enough_to_stay_measurable():
    """...and narrowing it must not quietly make the live guard decorative.

    Below ``MIN_GAP_ROWS + 1`` arrivals the sampler returns None and the guard
    SKIPS, which is correct behaviour and useless signal. The floor is not a
    chosen ratio: it is the sampler's own minimum divided by the measured
    arrival rate, so this row goes red if either end of that derivation moves.
    """
    needed_days = (swb.MIN_GAP_ROWS + 1) / swb.MEASURED_ARRIVALS_PER_DAY
    assert needed_days <= swb.SAMPLE_WINDOW_DAYS, (
        f'a {swb.SAMPLE_WINDOW_DAYS}-day trailing window holds fewer than '
        f'{swb.MIN_GAP_ROWS + 1} arrivals at the measured '
        f'{swb.MEASURED_ARRIVALS_PER_DAY:.2f}/day, so the live guard would '
        'skip rather than measure'
    )


# ---------------------------------------------------------------------------
# (ii) FALSIFIABILITY — the property 3730's test_required_bound_comparison_can_fail
# exists for. The live guard below only runs where runs.db exists, so the proof
# that its two inequalities can actually FAIL has to be made HERE, over corpora
# whose terms are known BY CONSTRUCTION. Without it the guard could be
# vacuously green on every host.
# ---------------------------------------------------------------------------


def _sample_of(tmp_path: Path, anchor: datetime, name: str,
               gaps: list[timedelta]) -> swb.StormWindowSample:
    """Sample a corpus whose gaps are known by construction."""
    db = _make_db(tmp_path / name, _failures(_spaced(anchor, gaps)))
    sample = swb.observed_storm_window_inputs(db)
    assert sample is not None
    assert sample.gaps_secs == tuple(g.total_seconds() for g in gaps)
    return sample


def test_the_not_inert_side_can_fail(tmp_path, anchor):
    """``min_gap <= window`` must be able to go RED.

    Below the smallest observed inter-arrival no two failures can EVER chain,
    so the alarm is unfireable at ANY threshold — a strict necessary condition
    on the window, and the reason the WINDOW rather than the threshold was the
    binding constraint on the shipped 3600 s.
    """
    sample = _sample_of(
        tmp_path, anchor, 'inert.db', [timedelta(hours=6)] * swb.MIN_GAP_ROWS,
    )

    too_narrow = sample.min_gap_secs - 1
    assert not (sample.min_gap_secs <= too_narrow), (
        'the not-inert comparison must be able to FAIL — a guard that cannot '
        'go red is measuring nothing'
    )
    assert sample.min_gap_secs <= sample.min_gap_secs


def test_the_no_false_alarm_side_can_fail(tmp_path, anchor):
    """``longest_chained_run(gaps, window) < threshold`` must be able to go RED.

    A corpus of tightly-spaced failures chains a run as long as the threshold,
    which is the shape that would page an operator for the fleet's ordinary
    background rate. Shown failing at a window wide enough to chain it and
    clearing at one narrow enough not to.
    """
    threshold = SessionResumeConfig().fallback_storm_threshold
    tight = [timedelta(minutes=1)] * max(swb.MIN_GAP_ROWS, threshold)
    sample = _sample_of(tmp_path, anchor, 'noisy.db', tight)

    chaining = timedelta(minutes=5).total_seconds()
    assert not (swb.longest_chained_run(sample.gaps_secs, chaining) < threshold), (
        'the no-false-alarm comparison must be able to FAIL — otherwise the '
        'shipped window could be any size at all'
    )
    narrow = timedelta(seconds=30).total_seconds()
    assert swb.longest_chained_run(sample.gaps_secs, narrow) < threshold


def test_the_recorded_table_is_what_the_function_says():
    """Every recorded number is RE-DERIVED from the recorded series.

    The provenance constants used to claim this and not do it: the table was
    only ever indexed, never regenerated, so four of its five entries — and
    the floor, and the ceiling — were unverifiable prose wearing a dict. What
    made the check impossible was that the gap SERIES itself lived in a
    comment; recording it as ``MEASURED_GAPS_SECS`` is what turns every other
    constant into a function of something checkable.

    It bites: the recorded ceiling was 190_806.542, rounded UP off the gap it
    is derived from, and at that window the null already chains 7 — so the
    "open upper end" was on the wrong side of itself. This row is what catches
    that class of copy error.
    """
    gaps = swb.MEASURED_GAPS_SECS
    assert len(gaps) == swb.MEASURED_ROWS - 1, (
        'N arrivals produce exactly N-1 inter-arrivals'
    )
    assert min(gaps) == swb.MEASURED_MIN_GAP_SECS

    rederived = {
        window: swb.longest_chained_run(gaps, window)
        for window in swb.MEASURED_LONGEST_RUN_BY_WINDOW
    }
    assert rederived == swb.MEASURED_LONGEST_RUN_BY_WINDOW

    # The ceiling is the exact boundary, pinned from BOTH sides: at it the null
    # stays under the threshold, and one float above it does not.
    threshold = SessionResumeConfig().fallback_storm_threshold
    ceiling = swb.MEASURED_RUN_REACHES_FIVE_ABOVE_SECS
    assert swb.longest_chained_run(gaps, ceiling) < threshold
    assert swb.longest_chained_run(
        gaps, math.nextafter(ceiling, math.inf)
    ) >= threshold


def test_the_shipped_pair_sits_inside_the_recorded_admissible_interval():
    """The provenance constants and the shipped defaults agree.

    ``MEASURED_*`` record the derivation as CODE rather than prose, so a
    re-derivation updates one place and this row re-checks the conclusion the
    config comment states. Arithmetic over module constants, host-free: the
    LIVE guard below is what re-measures.
    """
    config = SessionResumeConfig()
    floor, ceiling = (
        swb.MEASURED_MIN_GAP_SECS, swb.MEASURED_RUN_REACHES_FIVE_ABOVE_SECS,
    )
    assert floor <= config.storm_window_secs < ceiling
    # ...and the recorded run-by-window table agrees at the shipped window.
    # Read with .get and an explicit message: retuning storm_window_secs to a
    # value the table does not cover is a real and expected edit, and it should
    # say which constant to extend rather than raise a bare KeyError.
    recorded_run = swb.MEASURED_LONGEST_RUN_BY_WINDOW.get(
        config.storm_window_secs
    )
    assert recorded_run is not None, (
        f'session_resume.storm_window_secs is {config.storm_window_secs}s, '
        'which storm_window_bound.MEASURED_LONGEST_RUN_BY_WINDOW does not '
        'record a null run for. Re-derive the table at the new window and '
        'add the entry in the same commit as the retune.'
    )
    assert recorded_run < config.fallback_storm_threshold


# ---------------------------------------------------------------------------
# (iii) The bound re-derived against the LIVE runs.db, mirroring
# test_resume_age_bound.py's test_absolute_resume_age_is_derived_from_live_runs_db.
#
# Meaningful on a live host, inert in a fresh checkout: with no runs.db (or too
# sparse a one) it SKIPS rather than passing, so silence stays legible in the
# pytest output as "not measured" instead of masquerading as a green check.
# ---------------------------------------------------------------------------


def test_storm_window_is_derived_from_live_runs_db():
    """DERIVED BOUND, TWO-SIDED: the shipped (threshold, window) pair must sit
    inside the interval the fleet's MEASURED failure arrivals admit.

    NOT INERT — ``min_gap <= storm_window_secs``. Below the smallest observed
    inter-arrival no two failures can ever chain, so INV-4's escape cannot fire
    at any threshold: green, and measuring nothing.

    NO FALSE ALARM — ``longest_chained_run(gaps, window) < threshold``. At the
    shipped window the longest run the measured NULL produces must stay
    strictly below the threshold, or ordinary background failures page an
    operator.

    Neither side is a safety-factor multiplier: one is a strict necessary
    condition and the other a direct count (design-invariants G6 rejects "a
    magic number wearing a formula"). Re-measured every run, so a fleet whose
    failures start arriving in tighter bursts trips this test instead of
    quietly filing a false storm.

    NOTHING EXCLUDES A GENUINE INCIDENT from the null corpus, so a real
    session-resume storm reddens this row for every task until it ages out of
    the trailing window. That is why the window is this module's own narrower
    one, and why the no-false-alarm message below names both causes and the
    action for each rather than asserting a mis-tuned default.
    """
    db = swb.default_runs_db_path()
    if not db.is_file():
        pytest.skip(
            f'no live runs.db at {db} — nothing to derive the bound from '
            f'(set ${swb.RUNS_DB_ENV_VAR} to point at one; host-independent '
            'falsifiability for both inequalities is covered by '
            'test_the_not_inert_side_can_fail and '
            'test_the_no_false_alarm_side_can_fail)'
        )

    sample = swb.observed_storm_window_inputs(db)
    if sample is None:
        pytest.skip(
            f'runs.db at {db} holds fewer than {swb.MIN_GAP_ROWS} '
            'session_resume_failed inter-arrivals inside the trailing '
            f'{swb.SAMPLE_WINDOW_DAYS} days — too sparse to derive a bound '
            'from, and absence must never read as a zero gap'
        )

    # NON-DEGENERATE before it is consumed: a zero min gap would satisfy the
    # not-inert side trivially and leave this guard green while measuring
    # nothing — the exact vacuity trap it exists to avoid.
    assert sample.gap_rows >= swb.MIN_GAP_ROWS
    assert sample.min_gap_secs > 0, 'degenerate sample: a zero inter-arrival'

    config = SessionResumeConfig()
    window = config.storm_window_secs
    threshold = config.fallback_storm_threshold
    longest = swb.longest_chained_run(sample.gaps_secs, window)
    tightest = ', '.join(
        f'{gap / 3600:.2f}h' for gap in sorted(sample.gaps_secs)[:5]
    )
    recipe = (
        f'  runs.db .......... {db}\n'
        f'  sample ........... {sample.gap_rows} session_resume_failed '
        f'inter-arrivals over {sample.span_days:.1f} days, trailing '
        f'{swb.SAMPLE_WINDOW_DAYS} days\n'
        f'  min inter-arrival  {sample.min_gap_secs:.0f}s '
        f'({sample.min_gap_secs / 3600:.2f}h)\n'
        f'  tightest 5 gaps .. {tightest}\n'
        f'  shipped window ... {window}s ({window / 3600:.2f}h)\n'
        f'  shipped threshold  {threshold}\n'
        f'  longest null run . {longest} at the shipped window\n'
        f'  recorded floor ... {swb.MEASURED_MIN_GAP_SECS:.0f}s from '
        f'{swb.MEASUREMENT_DATE} ({swb.MEASURED_ROWS} rows)\n'
        'Provenance for the numbers the defaults were derived from is in the '
        'MEASURED_* constants of '
        'orchestrator/src/orchestrator/storm_window_bound.py — re-measure '
        'before trusting them, and update them in the SAME commit as any '
        'change to SessionResumeConfig.storm_window_secs / '
        'fallback_storm_threshold and test_config.py::test_defaults.'
    )

    assert sample.min_gap_secs <= window, (
        'session_resume.storm_window_secs is BELOW the smallest observed '
        'interval between eligible-but-FAILED resumes, so no two of them can '
        'ever chain and the INV-4 storm escape cannot fire at ANY threshold — '
        'green, and measuring nothing.\n' + recipe
    )
    assert longest < threshold, (
        'at the shipped window the SAMPLED failure arrivals already chain a '
        'run as long as fallback_storm_threshold. TWO causes, and they need '
        'opposite actions:\n'
        '  (1) A GENUINE SESSION-RESUME STORM inside the sampled trailing '
        f'{swb.SAMPLE_WINDOW_DAYS} days — i.e. exactly the event INV-4\'s L1 '
        'exists to page for. Nothing excludes an incident from this NULL '
        'corpus, so a real one reddens this row for every task until it ages '
        'out. ACTION: look for an open session-resume storm L1 (sentinel '
        'harness.py::_SESSION_RESUME_STORM_SENTINEL) and fix the '
        'archive-restore cause it names; this row goes green on its own once '
        'the incident leaves the window. It is NOT a reason to widen the '
        'threshold or narrow the window — that would disable the alarm '
        'because it fired.\n'
        '  (2) A MIS-TUNED DEFAULT — the fleet\'s ORDINARY background rate '
        'now chains that far, so the L1 would page for the background rather '
        'than for a systematic failure. ACTION: re-derive the pair and update '
        'SessionResumeConfig plus the MEASURED_* constants in one commit.\n'
        'Tell them apart from the gap series below: an incident is a tight '
        'cluster inside an otherwise sparse series, a mis-tuned default is a '
        'uniformly tighter one.\n' + recipe
    )

    # PROVENANCE, re-checked by the same run that re-derives the bound. The
    # MEASURED_* constants are a DATED RECORD of one derivation, not a live
    # mirror, so "recorded == measured" is the wrong check: a trailing window
    # moves the recorded floor every time a row ages in or out (it halved
    # between two derivations two days apart), and a tolerance band on that
    # would redden every task's verify for no defect, no incident and no
    # mis-tuned default — the failure mode the message above exists to avoid.
    #
    # What must never drift is the VERDICT the record is CONSUMED for:
    # test_the_shipped_pair_sits_inside_the_recorded_admissible_interval
    # validates the shipped window against the RECORDED interval, so the record
    # and the live fleet have to agree about whether that window clears the
    # not-inert floor. When they disagree, the record is what is wrong — the
    # live measurement is re-taken above.
    #
    # REACHABLE, and it is the case the host-free row CANNOT diagnose. The
    # not-inert assert above already fires when the LIVE floor exceeds the
    # window, so the only state left for this one is: live clears, recorded does
    # not. That is a stale record — and it is what the host-free row would
    # report as a mis-tuned default, because with no runs.db it cannot tell the
    # two apart. This row can, and says so.
    assert (sample.min_gap_secs <= window) == (
        window >= swb.MEASURED_MIN_GAP_SECS
    ), (
        'the RECORDED derivation and the LIVE fleet now disagree about '
        'whether session_resume.storm_window_secs clears the not-inert floor, '
        'so the MEASURED_* provenance constants have gone stale exactly the '
        'way the 3600s config comment they replaced did. RE-DERIVE THEM AND '
        'UPDATE THEM IN THIS COMMIT (the sampler in storm_window_bound.py is '
        'what produced them).\n' + recipe
    )
