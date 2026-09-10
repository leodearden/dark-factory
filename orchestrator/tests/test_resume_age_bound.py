"""Tests for :mod:`orchestrator.resume_age_bound` (task 3730 / PRD leaf δ).

The module derives the ABSOLUTE outer bound on a recovered sidecar's age —
``session_resume.absolute_resume_age_secs`` — from two terms MEASURED against
runs.db rather than chosen: the longest legitimate in-flight invocation, and
the longest observed orchestrator downtime. See D3 of
``plans/session-resume-eligibility-seam-prd.md``.

Structure mirrors ``scripts/tests/test_gc_agent_transcripts.py`` (task 3621),
the existing derived-bound precedent in this repo: a SYNTHETIC-corpus sampler
suite and a host-independent falsifiability proof that are meaningful in any
checkout, plus a LIVE guard that re-derives against the real runs.db and SKIPS
(never fails) where that data is absent.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestrator import resume_age_bound as rab
from orchestrator.config import SessionResumeConfig

# Mirrors the `events` table of orchestrator/src/orchestrator/event_store.py —
# copied rather than imported so a schema drift shows up as a red test here
# instead of the synthetic corpus silently tracking a changed production shape.
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
            'INSERT INTO events (timestamp, run_id, task_id, event_type, data, '
            'duration_ms) VALUES (?, ?, ?, ?, ?, ?)',
            [
                (
                    row['timestamp'],
                    row.get('run_id', 'run-1'),
                    row.get('task_id', '1'),
                    row['event_type'],
                    json.dumps(row.get('data', {})),
                    row.get('duration_ms'),
                )
                for row in rows
            ],
        )
        con.commit()
    finally:
        con.close()
    return path


def _completed(
    when: datetime,
    duration_ms: int | None,
    outcome: str = 'done',
) -> dict:
    """One ``task_completed`` row, the population the in-flight term samples."""
    return {
        'timestamp': when.isoformat(),
        'event_type': 'task_completed',
        'data': {'outcome': outcome},
        'duration_ms': duration_ms,
    }


def _filler(count: int, *, start: datetime, step: timedelta, duration_ms: int = 1000) -> list[dict]:
    """*count* short, evenly-spaced legitimate completions.

    Evenly spaced so a test that wants a specific max GAP can create exactly
    one, and short so a test that wants a specific max DURATION can too.
    """
    return [_completed(start + step * i, duration_ms) for i in range(count)]


@pytest.fixture
def anchor() -> datetime:
    """A recent instant, comfortably inside the sampler's trailing window."""
    return datetime.now(UTC) - timedelta(days=30)


# ---------------------------------------------------------------------------
# step-1 (task 3730): observed_resume_age_inputs(db_path) — the sampler.
#
# ABSENCE MUST NEVER READ AS A ZERO. Every "not enough data" arm returns None,
# not a sample carrying 0, because a zero input satisfies any bound trivially
# and would leave the live derived-bound guard permanently green while
# measuring nothing (the vacuity trap 3621 records at
# scripts/gc_agent_transcripts.py::MIN_RATE_SAMPLE_DAYS).
# ---------------------------------------------------------------------------


def test_absent_db_returns_none(tmp_path):
    """(a) A missing runs.db is unmeasurable, not zero — and never raises."""
    assert rab.observed_resume_age_inputs(tmp_path / 'nope.db') is None


def test_empty_events_table_returns_none(tmp_path):
    """(a) A schema-present, row-absent db is unmeasurable, not zero."""
    db = _make_db(tmp_path / 'runs.db', [])
    assert rab.observed_resume_age_inputs(db) is None


def test_sample_below_min_returns_none(tmp_path, anchor):
    """(a) A sample under MIN_SAMPLE_TASKS is None — never a zero-valued sample.

    Asserted as ``is None`` rather than falsiness: a ``ResumeAgeSample`` whose
    terms were both 0 would also be reportable, and it is exactly that shape —
    a measured-looking zero — the None contract exists to make impossible.
    """
    rows = _filler(
        rab.MIN_SAMPLE_TASKS - 1, start=anchor, step=timedelta(minutes=5)
    )
    db = _make_db(tmp_path / 'runs.db', rows)
    assert rab.observed_resume_age_inputs(db) is None


def test_inflight_term_ignores_cancelled_outcomes(tmp_path, anchor):
    """(b) A long CANCELLED row must not inflate the term; a long DONE row must.

    Cancellation durations reflect operator action, not how long a task can
    legitimately be in flight — the quantity the bound is about.
    """
    rows = _filler(rab.MIN_SAMPLE_TASKS, start=anchor, step=timedelta(minutes=5))
    long_legit_ms = 7 * 3600 * 1000
    rows.append(_completed(anchor + timedelta(hours=20), long_legit_ms, 'done'))
    # Both cancellation spellings, each an order of magnitude longer.
    rows.append(_completed(anchor + timedelta(hours=21), 90 * 3600 * 1000, 'cancelled'))
    rows.append(
        _completed(anchor + timedelta(hours=22), 90 * 3600 * 1000, 'soft-cancelled')
    )
    db = _make_db(tmp_path / 'runs.db', rows)

    sample = rab.observed_resume_age_inputs(db)
    assert sample is not None
    assert sample.inflight_max_secs == long_legit_ms / 1000


def test_inflight_term_ignores_null_and_nonpositive_durations(tmp_path, anchor):
    """(c) NULL / zero / negative duration_ms rows are excluded outright.

    A NULL duration is an unmeasured invocation, not a zero-length one, and a
    non-positive one is a clock artefact; neither is evidence about how long a
    task can be in flight.
    """
    rows = _filler(rab.MIN_SAMPLE_TASKS, start=anchor, step=timedelta(minutes=5))
    legit_ms = 4 * 3600 * 1000
    rows.append(_completed(anchor + timedelta(hours=20), legit_ms, 'done'))
    rows.append(_completed(anchor + timedelta(hours=21), None, 'done'))
    rows.append(_completed(anchor + timedelta(hours=22), 0, 'done'))
    rows.append(_completed(anchor + timedelta(hours=23), -5, 'done'))
    db = _make_db(tmp_path / 'runs.db', rows)

    sample = rab.observed_resume_age_inputs(db)
    assert sample is not None
    assert sample.inflight_max_secs == legit_ms / 1000
    # The three excluded rows are not counted as measured in-flight samples.
    assert sample.inflight_rows == rab.MIN_SAMPLE_TASKS + 1


def test_downtime_term_is_the_largest_gap_between_consecutive_events(tmp_path, anchor):
    """(d) The downtime term is the largest inter-event gap in the window."""
    step = timedelta(minutes=5)
    rows = _filler(rab.MIN_SAMPLE_TASKS, start=anchor, step=step)
    last = anchor + step * (rab.MIN_SAMPLE_TASKS - 1)
    gap = timedelta(hours=11)
    rows.append(_completed(last + gap, 1000, 'done'))
    db = _make_db(tmp_path / 'runs.db', rows)

    sample = rab.observed_resume_age_inputs(db)
    assert sample is not None
    assert sample.downtime_max_secs == gap.total_seconds()


def test_downtime_term_counts_every_event_type_as_liveness(tmp_path, anchor):
    """(d) ANY event proves the orchestrator was alive, not just task_completed.

    The sharp form: a single unrelated event dropped inside what would
    otherwise be a long task_completed-to-task_completed gap must SPLIT that
    gap. Sampling only completions would read an idle-but-alive stretch as
    downtime and inflate the bound.
    """
    step = timedelta(minutes=5)
    rows = _filler(rab.MIN_SAMPLE_TASKS, start=anchor, step=step)
    last = anchor + step * (rab.MIN_SAMPLE_TASKS - 1)
    # A 12h stretch with no completion in it — but the orchestrator was alive
    # 4h in, so the true maximum downtime across it is 8h, not 12h.
    rows.append({'timestamp': (last + timedelta(hours=4)).isoformat(),
                 'event_type': 'task_started'})
    rows.append(_completed(last + timedelta(hours=12), 1000, 'done'))
    db = _make_db(tmp_path / 'runs.db', rows)

    sample = rab.observed_resume_age_inputs(db)
    assert sample is not None
    assert sample.downtime_max_secs == timedelta(hours=8).total_seconds()


def test_events_outside_the_trailing_window_are_not_sampled(tmp_path, anchor):
    """(d) The downtime term is scoped to the trailing window.

    A row far older than the window would manufacture an enormous phantom gap
    between it and the recent corpus; the bound must be derived from the
    fleet's CURRENT behaviour, which is what makes it re-derive rather than
    ratchet.
    """
    step = timedelta(minutes=5)
    rows = _filler(rab.MIN_SAMPLE_TASKS, start=anchor, step=step)
    ancient = datetime.now(UTC) - timedelta(days=rab.SAMPLE_WINDOW_DAYS + 30)
    rows.append(_completed(ancient, 1000, 'done'))
    db = _make_db(tmp_path / 'runs.db', rows)

    sample = rab.observed_resume_age_inputs(db)
    assert sample is not None
    assert sample.downtime_max_secs == step.total_seconds()
    assert sample.span_days < rab.SAMPLE_WINDOW_DAYS


def test_an_ancient_long_invocation_does_not_inflate_the_inflight_term(
    tmp_path, anchor
):
    """(d) The IN-FLIGHT term is scoped to the SAME trailing window.

    The sibling above seeds its ancient row with a 1-second duration, so it
    pins only the downtime side and would stay green with an unbounded
    ``MAX(duration_ms)``. Here the ancient row is a 40-HOUR completion — far
    longer than anything in the recent corpus — and the term must ignore it:
    one long-retired invocation would otherwise sit in the maximum forever,
    ratcheting the bound up instead of letting it re-derive, and the sample
    would report a term drawn from outside the ``span_days`` it advertises.
    """
    step = timedelta(minutes=5)
    rows = _filler(rab.MIN_SAMPLE_TASKS, start=anchor, step=step)
    recent_max_ms = 3 * 3600 * 1000
    rows.append(_completed(anchor + timedelta(hours=20), recent_max_ms, 'done'))
    ancient = datetime.now(UTC) - timedelta(days=rab.SAMPLE_WINDOW_DAYS + 30)
    rows.append(_completed(ancient, 40 * 3600 * 1000, 'done'))
    db = _make_db(tmp_path / 'runs.db', rows)

    sample = rab.observed_resume_age_inputs(db)
    assert sample is not None
    assert sample.inflight_max_secs == recent_max_ms / 1000
    # ...and the out-of-window row is not counted as a measured sample either.
    assert sample.inflight_rows == rab.MIN_SAMPLE_TASKS + 1


def test_sample_reports_what_was_actually_measured(tmp_path, anchor):
    """(e) Both maxima travel with their sample sizes and span.

    A skip or failure message has to be able to say what it measured, not just
    what it concluded — the re-derivation recipe in the live guard below is
    built entirely out of these fields.
    """
    step = timedelta(minutes=5)
    rows = _filler(rab.MIN_SAMPLE_TASKS, start=anchor, step=step)
    db = _make_db(tmp_path / 'runs.db', rows)

    sample = rab.observed_resume_age_inputs(db)
    assert sample is not None
    assert sample.inflight_rows == rab.MIN_SAMPLE_TASKS
    assert sample.gap_rows == rab.MIN_SAMPLE_TASKS - 1
    assert sample.inflight_max_secs > 0
    assert sample.downtime_max_secs > 0
    expected_span = (step * (rab.MIN_SAMPLE_TASKS - 1)).total_seconds() / 86400
    assert sample.span_days == pytest.approx(expected_span)


def test_sampler_opens_the_db_read_only(tmp_path, anchor):
    """The sampler must never be able to write the production runs.db.

    Pinned by observation rather than by reading the connect string: a
    read-only connection cannot create the journal/WAL sidecars a write would,
    and the corpus is byte-identical after sampling.
    """
    rows = _filler(rab.MIN_SAMPLE_TASKS, start=anchor, step=timedelta(minutes=5))
    db = _make_db(tmp_path / 'runs.db', rows)
    before = db.read_bytes()

    assert rab.observed_resume_age_inputs(db) is not None

    assert db.read_bytes() == before
    assert not (tmp_path / 'runs.db-journal').exists()
    assert not (tmp_path / 'runs.db-wal').exists()


def test_default_runs_db_path_is_env_overridable(tmp_path, monkeypatch):
    """The live guard must be able to reach runs.db from a task worktree.

    ``data/`` is gitignored and absent from a worktree, so the default is an
    ABSOLUTE main-checkout path with an env override — the idiom
    ``evals/prompt_opt/__main__.py`` and ``evals/reviewer_trial/__main__.py``
    already use for exactly this class of offline data.
    """
    monkeypatch.delenv(rab.RUNS_DB_ENV_VAR, raising=False)
    default = rab.default_runs_db_path()
    assert default.is_absolute()
    assert default.name == 'runs.db'

    monkeypatch.setenv(rab.RUNS_DB_ENV_VAR, str(tmp_path / 'elsewhere.db'))
    assert rab.default_runs_db_path() == tmp_path / 'elsewhere.db'


# ---------------------------------------------------------------------------
# step-3 (task 3730): required_absolute_resume_age_secs(...) — the NON-VACUITY
# proof, mirroring test_gc_agent_transcripts.py's
# test_required_max_task_dirs_is_falsifiable_against_a_known_peak.
#
# The live derived-bound guard below is inert in a fresh checkout by design:
# with no runs.db to measure it skips. So the guarantee that the bound is a
# REAL check — and not arithmetic on literals that can never fail — has to be
# made HERE, host-independently: over a corpus whose two terms are known by
# CONSTRUCTION, the exact comparison the live guard makes is shown to fail for
# a too-small bound and clear for an adequate one.
# ---------------------------------------------------------------------------

def _known_corpus(tmp_path: Path, anchor: datetime, *, inflight: timedelta,
                  downtime: timedelta) -> rab.ResumeAgeSample:
    """Sample a corpus built so both terms are known BY CONSTRUCTION."""
    step = timedelta(minutes=5)
    rows = _filler(rab.MIN_SAMPLE_TASKS, start=anchor, step=step)
    last = anchor + step * (rab.MIN_SAMPLE_TASKS - 1)
    rows.append(_completed(last + timedelta(minutes=5),
                           int(inflight.total_seconds() * 1000), 'done'))
    rows.append(_completed(last + timedelta(minutes=5) + downtime, 1000, 'done'))
    sample = rab.observed_resume_age_inputs(_make_db(tmp_path / 'known.db', rows))
    assert sample is not None
    # Known by construction — asserted before anything is derived from them.
    assert sample.inflight_max_secs == inflight.total_seconds()
    assert sample.downtime_max_secs == downtime.total_seconds()
    return sample


def test_required_bound_is_the_two_terms_times_the_factor(tmp_path, anchor):
    """(a) The requirement is ceil((in-flight + downtime) x safety).

    Nothing entering this arithmetic is a literal the test controls on both
    sides: both terms are read back off a SAMPLE of a corpus the test built,
    and the factor is the shipped constant.
    """
    sample = _known_corpus(
        tmp_path, anchor, inflight=timedelta(hours=6), downtime=timedelta(hours=10)
    )

    required = rab.required_absolute_resume_age_secs(
        sample.inflight_max_secs,
        sample.downtime_max_secs,
        rab.RESUME_AGE_SAFETY_FACTOR,
    )
    assert required == math.ceil(
        (sample.inflight_max_secs + sample.downtime_max_secs)
        * rab.RESUME_AGE_SAFETY_FACTOR
    )


def test_required_bound_comparison_can_fail(tmp_path, anchor):
    """(b) FALSIFIABILITY: the live guard's own comparison evaluates both ways.

    ``required <= candidate`` is the exact expression
    test_absolute_resume_age_is_derived_from_live_runs_db asserts. Shown here
    failing for a bound one second short and clearing for the shipped one.
    """
    sample = _known_corpus(
        tmp_path, anchor, inflight=timedelta(hours=6), downtime=timedelta(hours=10)
    )
    required = rab.required_absolute_resume_age_secs(
        sample.inflight_max_secs,
        sample.downtime_max_secs,
        rab.RESUME_AGE_SAFETY_FACTOR,
    )

    too_small = required - 1
    assert not (required <= too_small), (
        'the derived-bound comparison must be able to FAIL — a guard that '
        'cannot go red is measuring nothing'
    )
    # ...and it CLEARS for the bound we actually ship, against a corpus whose
    # terms are far below the live fleet's.
    assert required <= SessionResumeConfig().absolute_resume_age_secs


def test_required_bound_is_strictly_increasing_in_every_input():
    """(c) Neither term can be dropped, and the factor cannot be ignored.

    A derivation that ignored an input would still satisfy (a) on a corpus
    where that input happened not to matter. Varying each of the three
    separately is what makes "two terms" a checked claim rather than a
    docstring.
    """
    base = rab.required_absolute_resume_age_secs(1000.0, 2000.0, 1.5)
    assert rab.required_absolute_resume_age_secs(5000.0, 2000.0, 1.5) > base
    assert rab.required_absolute_resume_age_secs(1000.0, 9000.0, 1.5) > base
    assert rab.required_absolute_resume_age_secs(1000.0, 2000.0, 3.0) > base


def test_required_bound_rounds_up():
    """(d) A fractional requirement must never round DOWN into headroom that
    the measured worst case does not actually leave."""
    required = rab.required_absolute_resume_age_secs(1.0, 0.0, 1.5)
    assert required == 2  # ceil(1.5), not 1
    assert isinstance(required, int)


def test_shipped_safety_factor_is_at_least_one():
    """(e) Below 1 the bound would sit UNDER the worst case it is derived from
    and would reject sessions that are still legitimately in flight."""
    assert rab.RESUME_AGE_SAFETY_FACTOR >= 1


def test_shipped_default_is_reachable_by_a_plausible_fleet():
    """(f) THE ANTI-INFLATION CLAMP — the live guard's subject can still trip.

    (b) shows the COMPARISON can go either way; this shows its actual SUBJECT
    can. A fleet twice as slow to finish and twice as long to be down really
    does exceed the bound we ship. Without this, raising
    absolute_resume_age_secs to a week or a month would leave every test in
    this file green while the live guard became permanently vacuous — the
    exact failure mode this section exists to rule out, and the one D3 names:
    "archive outranks age" must not quietly become "no age limit".
    """
    shipped = SessionResumeConfig().absolute_resume_age_secs
    required_at_doubled = rab.required_absolute_resume_age_secs(
        2 * rab.MEASURED_INFLIGHT_SECS,
        2 * rab.MEASURED_DOWNTIME_SECS,
        rab.RESUME_AGE_SAFETY_FACTOR,
    )
    assert required_at_doubled > shipped, (
        f'a fleet with a {2 * rab.MEASURED_INFLIGHT_SECS / 3600:.1f} h max '
        f'in-flight duration and {2 * rab.MEASURED_DOWNTIME_SECS / 3600:.1f} h '
        f'max downtime requires {required_at_doubled} s but '
        f'absolute_resume_age_secs is {shipped} s — the shipped bound is now '
        'so large that no realistic fleet can trip the live derived-bound '
        'guard, which makes that guard vacuous. Either the bound was raised '
        'far beyond its derivation, or the reference terms need re-deriving '
        'from a fresh measurement — they are '
        'resume_age_bound.MEASURED_INFLIGHT_SECS / MEASURED_DOWNTIME_SECS '
        f'(measured {rab.MEASUREMENT_DATE}), read from the module rather than '
        'copied here so one re-derivation updates one place.'
    )


# ---------------------------------------------------------------------------
# step-7 (task 3730): the bound re-derived against the LIVE runs.db,
# mirroring test_gc_agent_transcripts.py's
# test_max_task_dirs_is_derived_from_live_archive_rate.
#
# Meaningful on a live host, inert in a fresh checkout: with no runs.db (or too
# sparse a one) it SKIPS rather than passing, so silence stays legible in the
# pytest output as "not measured" instead of masquerading as a green check.
# ---------------------------------------------------------------------------


def test_absolute_resume_age_is_derived_from_live_runs_db():
    """DERIVED BOUND: absolute_resume_age_secs must cover the fleet's MEASURED
    worst case — the longest legitimate invocation plus the longest outage.

    A sidecar's started_at is stamped per invocation, so its age when the
    _run_slot guard evaluates it is in-flight-time-at-crash plus orchestrator
    downtime. If the bound sits below that sum, the backstop starts rejecting
    sessions from tasks that never stopped being legitimately in flight — and
    it does so silently, since 'aged_out' is by-design and files no L1. This
    guard re-measures every run, so a fleet that gets slower or suffers longer
    outages trips the test instead of quietly losing resumable sessions.
    """
    db = rab.default_runs_db_path()
    if not db.is_file():
        pytest.skip(
            f'no live runs.db at {db} — nothing to derive the bound from '
            f'(set ${rab.RUNS_DB_ENV_VAR} to point at one; host-independent '
            'falsifiability for the same comparison is covered by '
            'test_required_bound_comparison_can_fail and '
            'test_shipped_default_is_reachable_by_a_plausible_fleet)'
        )

    sample = rab.observed_resume_age_inputs(db)
    if sample is None:
        pytest.skip(
            f'runs.db at {db} holds fewer than {rab.MIN_SAMPLE_TASKS} '
            'legitimate task_completed rows, or fewer than two parseable '
            f'timestamps inside the trailing {rab.SAMPLE_WINDOW_DAYS} days — '
            'too sparse to derive a bound from'
        )

    # NON-DEGENERATE before it is consumed. A zero term would make the bound
    # below satisfiable by ANY value, leaving this guard permanently green
    # while measuring nothing — the exact vacuity trap it exists to avoid.
    assert sample.inflight_rows > 0, 'degenerate sample: no completions measured'
    assert sample.gap_rows > 0, 'degenerate sample: no inter-event gaps measured'
    assert sample.inflight_max_secs > 0, 'degenerate sample: zero in-flight max'
    assert sample.downtime_max_secs > 0, 'degenerate sample: zero downtime max'

    required = rab.required_absolute_resume_age_secs(
        sample.inflight_max_secs,
        sample.downtime_max_secs,
        rab.RESUME_AGE_SAFETY_FACTOR,
    )
    shipped = SessionResumeConfig().absolute_resume_age_secs

    assert required <= shipped, (
        'session_resume.absolute_resume_age_secs is too small for the fleet\'s '
        'MEASURED behaviour: the backstop will reject recovered sessions from '
        'tasks that were still legitimately in flight, silently, because '
        "'aged_out' is by-design and files no L1.\n"
        f'  runs.db .......... {db}\n'
        f'  in-flight sample . {sample.inflight_rows} legitimate completions, '
        f'max {sample.inflight_max_secs:.0f}s ({sample.inflight_max_secs / 3600:.2f}h)\n'
        f'  downtime sample .. {sample.gap_rows} inter-event gaps over '
        f'{sample.span_days:.1f} days, max {sample.downtime_max_secs:.0f}s '
        f'({sample.downtime_max_secs / 3600:.2f}h)\n'
        f'  safety factor .... {rab.RESUME_AGE_SAFETY_FACTOR} (provenance and '
        'the measurement it was derived from are in the '
        'RESUME_AGE_SAFETY_FACTOR comment in '
        'orchestrator/src/orchestrator/resume_age_bound.py — re-measure before '
        'trusting it)\n'
        f'  REQUIRED ......... {required}s = ceil(('
        f'{sample.inflight_max_secs:.0f}s + {sample.downtime_max_secs:.0f}s) '
        f'x {rab.RESUME_AGE_SAFETY_FACTOR})\n'
        f'  current default .. {shipped}s ({shipped / 86400:.2f} days)\n'
        'TO FIX: re-derive from the numbers above (ruling: '
        'plans/session-resume-eligibility-seam-prd.md D3) and raise ALL FOUR '
        'lock-step sites in ONE commit, or any single commit is red:\n'
        '  1. orchestrator/src/orchestrator/config.py  '
        'SessionResumeConfig.absolute_resume_age_secs default (+ its '
        'description, which quotes the derived value)\n'
        '  2. orchestrator/src/orchestrator/resume_age_bound.py  '
        'MEASUREMENT_DATE, MEASURED_INFLIGHT_SECS and MEASURED_DOWNTIME_SECS '
        '(the anti-inflation clamp doubles those two, so a re-derivation that '
        'skips them leaves the clamp calibrated off the old fleet), plus the '
        'RESUME_AGE_SAFETY_FACTOR block that reads off them (requirement, '
        'margin, trip point)\n'
        '  3. orchestrator/tests/test_config.py  '
        'TestSessionResumeConfig::test_defaults\n'
        '  4. OPERATIONS.md  the session-resume subsection of §14\n'
        'The authoritative list is whatever '
        "`grep -rn 'absolute_resume_age_secs' --include='*.py' --include='*.md' "
        "--include='*.yaml'` returns outside plans/ and .worktrees/ — RE-RUN "
        'IT; do not trust this list to have stayed complete (3621 shipped a '
        'four-site list that was missing a fifth, and the commit raising the '
        'other four left the tree red — esc-3621-2).\n'
        'And CHECK THE RELATION while you are there: the bound must stay above '
        'session_resume.freshness_window_secs '
        '(test_config.py::test_absolute_bound_is_looser_than_the_freshness_window) '
        'and below the ~2x ceiling the anti-inflation clamp enforces '
        '(test_shipped_default_is_reachable_by_a_plausible_fleet).'
    )
