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

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

import reclaim_orphaned_worktrees as row
from reclaim_orphaned_worktrees import parse_parking_dir_name

LOG_PREFIX = "reclaim_orphaned_worktrees:"

NOW = 1_000_000_000.0
HOUR = 3600.0

# Deterministic parking stamps used by the real-git-repo fixtures below.
TS_OLD = "20260722T153045Z"
TS_LANE = "20260706T000000Z"
PARKED_AT_OLD = datetime(2026, 7, 22, 15, 30, 45, tzinfo=UTC)
PARKED_AT_LANE = datetime(2026, 7, 6, 0, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Real temp git-repo fixtures (shared by the impure list/park/remove/reclaim
# tests). Build a genuine repo via `git init` + `git worktree add` laid out as
# <repo>/.worktrees-orphaned/<id>-<ts> so every git operation runs against real
# git state — no mocking of the porcelain.
# ---------------------------------------------------------------------------

def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run ``git -C <repo> <args>`` (test helper, distinct from the module's
    own ``_run_git``)."""
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {args} failed (rc={result.returncode}): {result.stderr}")
    return result


def _init_repo(tmp_path: Path) -> Path:
    """`git init` a repo with a single initial commit on ``main``; return it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Reclaim Test")
    (repo / "README.md").write_text("hello\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "--no-verify", "-m", "initial")
    return repo


def _parking_root(repo: Path) -> Path:
    """The quarantine base sibling of ``.worktrees`` (matches the producer)."""
    return repo / ".worktrees-orphaned"


def _add_parking(
    repo: Path,
    name: str,
    *,
    dirty: bool = False,
    detached: bool = False,
    modify_tracked: bool = False,
) -> Path:
    """`git worktree add` a parking at ``<repo>/.worktrees-orphaned/<name>``.

    Non-detached parkings check out a fresh ``task/<name>`` branch (mirroring
    the producer's renamed parking branch). ``dirty`` drops an untracked file;
    ``modify_tracked`` also edits the tracked README so both change classes are
    present.
    """
    dest = _parking_root(repo) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if detached:
        _git(repo, "worktree", "add", "-q", "--detach", str(dest))
    else:
        _git(repo, "worktree", "add", "-q", "-b", f"task/{name}", str(dest))
    if modify_tracked:
        (dest / "README.md").write_text("modified in parking\n")
    if dirty:
        (dest / "wip.txt").write_text("uncommitted work\n")
    return dest


def _add_normal_worktree(repo: Path, name: str) -> Path:
    """`git worktree add` a NON-parking worktree under ``<repo>/.worktrees/``
    (must be excluded by the parking-root band filter)."""
    dest = repo / ".worktrees" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "-q", "-b", f"task/{name}", str(dest))
    return dest


def _branch_resolves(repo: Path, branch: str) -> bool:
    return _git(
        repo, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}", check=False
    ).returncode == 0


def _worktree_paths(repo: Path) -> set[Path]:
    """Resolved paths currently registered in ``git worktree list --porcelain``."""
    out = _git(repo, "worktree", "list", "--porcelain").stdout
    return {
        Path(line[len("worktree ") :]).resolve()
        for line in out.splitlines()
        if line.startswith("worktree ")
    }


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


# ---------------------------------------------------------------------------
# step-5: list_parked_worktrees(repo, parking_root) against a real git repo
# ---------------------------------------------------------------------------

def test_list_returns_only_parkings_under_root_with_parsed_fields(tmp_path):
    """Only registered worktrees UNDER the parking root are returned — the main
    worktree and a normal ``.worktrees/`` worktree are excluded — each carrying
    its resolved path, porcelain branch, and parsed parked_at."""
    repo = _init_repo(tmp_path)
    _add_normal_worktree(repo, "3001")  # a live task worktree — must be excluded
    p_id = _add_parking(repo, f"2920-{TS_OLD}")
    p_lane = _add_parking(repo, f"_lane-0-{TS_LANE}")

    records = row.list_parked_worktrees(repo, _parking_root(repo))

    by_path = {r.path: r for r in records}
    assert set(by_path) == {p_id.resolve(), p_lane.resolve()}

    rec_id = by_path[p_id.resolve()]
    assert rec_id.branch == f"task/2920-{TS_OLD}"
    assert rec_id.parked_at == PARKED_AT_OLD

    rec_lane = by_path[p_lane.resolve()]
    assert rec_lane.branch == f"task/_lane-0-{TS_LANE}"
    assert rec_lane.parked_at == PARKED_AT_LANE


def test_list_skips_unparseable_basename_with_loud_warning(tmp_path, caplog):
    """A parking dir whose basename lacks a valid trailing stamp is SKIPPED with
    a LOUD WARNING and never returned (never guess an age)."""
    caplog.set_level(logging.WARNING, logger="reclaim_orphaned_worktrees")
    repo = _init_repo(tmp_path)
    good = _add_parking(repo, f"2920-{TS_OLD}")
    bad = _add_parking(repo, "no-valid-stamp")  # basename has no trailing stamp

    records = row.list_parked_worktrees(repo, _parking_root(repo))

    assert {r.path for r in records} == {good.resolve()}
    warn_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        LOG_PREFIX in m and str(bad.resolve()) in m for m in warn_msgs
    ), f"missing LOUD WARNING for the unparseable parking; got {warn_msgs}"


def test_list_empty_parking_root_returns_empty(tmp_path):
    """An existing-but-empty parking root (no parkings registered) yields []."""
    repo = _init_repo(tmp_path)
    _parking_root(repo).mkdir()
    assert row.list_parked_worktrees(repo, _parking_root(repo)) == []


def test_list_absent_parking_root_returns_empty(tmp_path):
    """An ABSENT parking root yields [] (no parkings to scan)."""
    repo = _init_repo(tmp_path)
    assert row.list_parked_worktrees(repo, _parking_root(repo)) == []
