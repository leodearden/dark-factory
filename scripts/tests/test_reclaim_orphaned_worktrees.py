"""Tests for scripts/reclaim_orphaned_worktrees.py — the periodic verified
reclaim sweep over the ``.worktrees-orphaned/`` quarantine base (task 2980).

Models scripts/tests/test_gc_agent_transcripts.py 1:1 (pure select_* tests with
synthetic records + a fixed NOW; real-filesystem fixtures for the impure
scan/prune; a subprocess CLI driver with JSON on stdout / LOUD logs on stderr;
caplog LOUD-prefix assertions) — extended to build REAL temp git repos
(``git init`` + ``git worktree add``) laid out as
``<tmp>/.worktrees-orphaned/<id>-<ts>`` so list/park-commit/remove/prune run
against genuine git state.

``import reclaim_orphaned_worktrees`` resolves via scripts/tests/conftest.py's
sys.path insertion (no new conftest, no package __init__).

step-1: the pure ``parse_parking_dir_name(name)`` parser — trailing
``-<YYYYMMDDTHHMMSSZ>`` stamp -> tz-aware UTC datetime; None otherwise.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import reclaim_orphaned_worktrees as row
from reclaim_orphaned_worktrees import parse_parking_dir_name

LOG_PREFIX = "reclaim_orphaned_worktrees:"

NOW = 1_000_000_000.0
HOUR = 3600.0


# ---------------------------------------------------------------------------
# step-1: pure parse_parking_dir_name(name)
# ---------------------------------------------------------------------------

def test_parse_simple_id_parking_name():
    """A numeric-id parking basename parses its trailing stamp into a tz-aware
    UTC datetime."""
    assert parse_parking_dir_name("2920-20260722T153045Z") == datetime(
        2026, 7, 22, 15, 30, 45, tzinfo=UTC
    )


def test_parse_lane_parking_name_with_embedded_hyphens():
    """A lane parking id itself contains hyphens (``_lane-0``); the regex must
    anchor the stamp at the END of the basename, not at the first hyphen."""
    assert parse_parking_dir_name("_lane-0-20260706T000000Z") == datetime(
        2026, 7, 6, 0, 0, 0, tzinfo=UTC
    )


def test_parse_returns_utc_aware_datetime():
    """The returned datetime is timezone-aware and anchored to UTC (not naive)."""
    parsed = parse_parking_dir_name("2920-20260722T153045Z")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == UTC.utcoffset(None)


def test_parse_returns_none_without_trailing_stamp():
    """Names with no valid trailing ``-<YYYYMMDDTHHMMSSZ>`` stamp -> None."""
    assert parse_parking_dir_name("2920") is None
    assert parse_parking_dir_name("2920-2026-07-22") is None
    assert parse_parking_dir_name("no-timestamp") is None


def test_parse_returns_none_on_calendar_invalid_stamp():
    """A syntactically-shaped but calendar-invalid stamp (month 13) -> None,
    not a raised ValueError (parse is total)."""
    assert parse_parking_dir_name("2920-20261399T000000Z") is None


def test_module_exposes_loud_prefix():
    """The module carries the stable greppable LOUD log prefix."""
    assert row._LOG_PREFIX == LOG_PREFIX


# ---------------------------------------------------------------------------
# step-3: pure select_reclaimable(records, now, min_age_hours)
# ---------------------------------------------------------------------------

def _rec(name: str, age_seconds: float) -> row.ParkedWorktree:
    """A synthetic ParkedWorktree whose parked_at is *age_seconds* before NOW."""
    return row.ParkedWorktree(
        path=Path("/parkings") / name,
        branch=f"task/{name}",
        parked_at=datetime.fromtimestamp(NOW - age_seconds, tz=UTC),
    )


def test_select_reclaims_record_older_than_floor():
    """A parking older than the floor -> reclaim with reason 'age'; a young
    parking -> keep."""
    old = _rec("old", 49 * HOUR)
    young = _rec("young", 1 * HOUR)

    decision = row.select_reclaimable([old, young], NOW, min_age_hours=48)

    assert decision.reclaim_paths == {old.path}
    assert decision.keep_paths == {young.path}
    assert decision.reasons[old.path] == "age"


def test_select_age_boundary_exact_is_kept_one_second_older_reclaimed():
    """now - parked_at == min_age_hours*3600 exactly is KEPT (strict >, not >=);
    one second older is reclaimed."""
    boundary = _rec("boundary", 48 * HOUR)          # age == floor exactly
    just_old = _rec("just_old", 48 * HOUR + 1)      # one second older

    decision = row.select_reclaimable([boundary, just_old], NOW, min_age_hours=48)

    assert decision.keep_paths == {boundary.path}
    assert decision.reclaim_paths == {just_old.path}
    assert decision.reasons[just_old.path] == "age"


def test_select_non_positive_floor_reclaims_nothing():
    """A non-positive floor reclaims NOTHING — even an ancient parking is kept
    (fail-safe protecting fresh parkings; the OPPOSITE of gc_agent_transcripts'
    'non-positive disables the axis')."""
    ancient = _rec("ancient", 10_000 * HOUR)
    young = _rec("young", 1 * HOUR)

    for disabled in (0, -1):
        decision = row.select_reclaimable([ancient, young], NOW, min_age_hours=disabled)
        assert decision.keep_paths == {ancient.path, young.path}
        assert decision.reclaim_paths == set()


def test_select_reclaim_ordering_is_deterministic_oldest_first():
    """The reclaim list is deterministically ordered oldest-first, independent
    of input order."""
    a = _rec("a", 100 * HOUR)   # oldest
    b = _rec("b", 80 * HOUR)
    c = _rec("c", 60 * HOUR)    # newest of the three (still > floor)

    decision = row.select_reclaimable([c, a, b], NOW, min_age_hours=48)

    assert [r.path for r, _reason in decision.reclaim] == [a.path, b.path, c.path]
