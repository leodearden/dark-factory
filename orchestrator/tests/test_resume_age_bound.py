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
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestrator import resume_age_bound as rab

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
