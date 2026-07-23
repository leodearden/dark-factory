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

import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import reclaim_orphaned_worktrees as row
from reclaim_orphaned_worktrees import parse_parking_dir_name

LOG_PREFIX = "reclaim_orphaned_worktrees:"
SCRIPT = Path(__file__).parent.parent / "reclaim_orphaned_worktrees.py"

NOW = 1_000_000_000.0
HOUR = 3600.0

# Deterministic parking stamps used by the real-git-repo fixtures below.
TS_OLD = "20260722T153045Z"
TS_LANE = "20260706T000000Z"
TS_YOUNG = "20260724T120000Z"
PARKED_AT_OLD = datetime(2026, 7, 22, 15, 30, 45, tzinfo=UTC)
PARKED_AT_LANE = datetime(2026, 7, 6, 0, 0, 0, tzinfo=UTC)

# CLI reference clock: 2026-07-25 00:00 UTC. Against a 48h floor, TS_OLD
# (~2.35d) and TS_LANE (~19d) are reclaimable; TS_YOUNG (12h) is kept.
CLI_NOW = datetime(2026, 7, 25, 0, 0, 0, tzinfo=UTC).timestamp()


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


# ---------------------------------------------------------------------------
# step-7: is_worktree_dirty(path) / branch_ref_resolves(repo, branch)
# ---------------------------------------------------------------------------

def test_is_worktree_dirty_false_on_clean_checkout(tmp_path):
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}")
    assert row.is_worktree_dirty(parking) is False


def test_is_worktree_dirty_true_on_untracked_file(tmp_path):
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}")
    (parking / "scratch.txt").write_text("new\n")
    assert row.is_worktree_dirty(parking) is True


def test_is_worktree_dirty_true_on_modified_tracked_file(tmp_path):
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}")
    (parking / "README.md").write_text("changed\n")
    assert row.is_worktree_dirty(parking) is True


def test_is_worktree_dirty_fail_safe_true_when_status_errors(tmp_path, caplog):
    """A ``git status`` that errors (path is not a worktree) is treated
    fail-safe as DIRTY (True) and never raises — mirrors
    GitOps._worktree_dirty's fail-safe True."""
    caplog.set_level(logging.WARNING, logger="reclaim_orphaned_worktrees")
    not_a_worktree = tmp_path / "not-a-worktree"
    not_a_worktree.mkdir()
    assert row.is_worktree_dirty(not_a_worktree) is True


def test_branch_ref_resolves_true_for_existing_branch(tmp_path):
    repo = _init_repo(tmp_path)
    _add_parking(repo, f"2920-{TS_OLD}")
    assert row.branch_ref_resolves(repo, f"task/2920-{TS_OLD}") is True


def test_branch_ref_resolves_false_for_missing_branch(tmp_path):
    repo = _init_repo(tmp_path)
    assert row.branch_ref_resolves(repo, "task/does-not-exist") is False


# ---------------------------------------------------------------------------
# step-9: park_commit(worktree, reason) — zero content lost
# ---------------------------------------------------------------------------

def _install_failing_precommit_hook(repo: Path) -> None:
    """Install an always-failing pre-commit hook in the repo's COMMON git dir
    (shared by linked worktrees), so a plain ``git commit`` would fail unless
    ``--no-verify`` bypasses it."""
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text("#!/bin/sh\necho 'pre-commit rejects' >&2\nexit 1\n")
    hook.chmod(0o755)


def test_park_commit_commits_dirty_and_preserves_all_content(tmp_path):
    """A dirty parking (untracked file + modified tracked file) is park-committed
    onto its branch: a NEW commit lands, ``git status`` is EMPTY afterwards, and
    BOTH files' content is retrievable from the branch — proving zero content
    lost."""
    repo = _init_repo(tmp_path)
    name = f"2920-{TS_OLD}"
    branch = f"task/{name}"
    parking = _add_parking(repo, name, dirty=True, modify_tracked=True)
    head_before = _git(parking, "rev-parse", "HEAD").stdout.strip()

    sha = row.park_commit(parking, "age")

    assert sha is not None
    head_after = _git(parking, "rev-parse", "HEAD").stdout.strip()
    assert head_after == sha
    assert head_after != head_before  # a new commit exists on the branch
    assert _git(parking, "status", "--porcelain").stdout.strip() == ""
    # Content provably recoverable from the branch ref (independent of the tree).
    assert _git(parking, "show", f"{branch}:wip.txt").stdout == "uncommitted work\n"
    assert _git(parking, "show", f"{branch}:README.md").stdout == "modified in parking\n"


def test_park_commit_noop_on_clean_worktree(tmp_path):
    """A CLEAN worktree -> park_commit is a no-op: returns None, no new commit."""
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}")
    head_before = _git(parking, "rev-parse", "HEAD").stdout.strip()

    assert row.park_commit(parking, "age") is None

    assert _git(parking, "rev-parse", "HEAD").stdout.strip() == head_before


def test_park_commit_uses_no_verify_bypassing_failing_hook(tmp_path):
    """A repo with an always-failing pre-commit hook still park-commits — proves
    the ``--no-verify`` bypass (a parking branch must accept the snapshot
    unconditionally)."""
    repo = _init_repo(tmp_path)
    _install_failing_precommit_hook(repo)
    parking = _add_parking(repo, f"2920-{TS_OLD}", dirty=True)

    sha = row.park_commit(parking, "age")

    assert sha is not None, "park_commit must bypass the failing hook via --no-verify"
    assert _git(parking, "status", "--porcelain").stdout.strip() == ""


# ---------------------------------------------------------------------------
# step-11: remove_worktree(repo, path) — BRANCH-INTACT invariant
# ---------------------------------------------------------------------------

def test_remove_worktree_removes_registered_parking_branch_intact(tmp_path):
    """Removing a registered, clean parking: (a) the dir is gone; (b) it is no
    longer listed by ``git worktree list``; and CRUCIALLY (c) the parking branch
    STILL resolves — remove must NEVER delete the branch."""
    repo = _init_repo(tmp_path)
    name = f"2920-{TS_OLD}"
    branch = f"task/{name}"
    parking = _add_parking(repo, name)
    assert parking.resolve() in _worktree_paths(repo)

    assert row.remove_worktree(repo, parking) is True

    assert not parking.exists()  # (a)
    assert parking.resolve() not in _worktree_paths(repo)  # (b)
    assert _branch_resolves(repo, branch)  # (c) branch survives — content safe


def test_remove_worktree_bogus_path_returns_false_no_raise(tmp_path):
    """remove_worktree on a nonexistent/bogus path returns False without
    raising."""
    repo = _init_repo(tmp_path)
    bogus = _parking_root(repo) / "does-not-exist"
    assert row.remove_worktree(repo, bogus) is False


# ---------------------------------------------------------------------------
# step-13: reclaim_worktrees(repo, records, *, dry_run) + ReclaimOutcome
# ---------------------------------------------------------------------------

def _resolved(paths) -> list[Path]:
    return [Path(p).resolve() for p in paths]


def test_reclaim_clean_eligible_removed_branch_intact(tmp_path):
    """(a) A CLEAN eligible parking is removed, branch intact, in reclaimed."""
    repo = _init_repo(tmp_path)
    name = f"2920-{TS_OLD}"
    branch = f"task/{name}"
    parking = _add_parking(repo, name)
    records = row.list_parked_worktrees(repo, _parking_root(repo))

    outcome = row.reclaim_worktrees(repo, records, dry_run=False)

    assert _resolved(outcome.reclaimed) == [parking.resolve()]
    assert outcome.park_committed == []
    assert outcome.skipped == []
    assert outcome.failed == []
    assert not parking.exists()
    assert _branch_resolves(repo, branch)


def test_reclaim_dirty_eligible_park_committed_then_removed(tmp_path):
    """(b) A DIRTY eligible parking is park-committed FIRST (content recoverable
    from the branch) THEN removed — in reclaimed AND park_committed."""
    repo = _init_repo(tmp_path)
    name = f"2920-{TS_OLD}"
    branch = f"task/{name}"
    parking = _add_parking(repo, name, dirty=True, modify_tracked=True)
    records = row.list_parked_worktrees(repo, _parking_root(repo))

    outcome = row.reclaim_worktrees(repo, records, dry_run=False)

    assert _resolved(outcome.reclaimed) == [parking.resolve()]
    assert _resolved(outcome.park_committed) == [parking.resolve()]
    assert not parking.exists()
    assert _branch_resolves(repo, branch)
    # Content provably recoverable from the branch even though the tree is gone.
    assert _git(repo, "show", f"{branch}:wip.txt").stdout == "uncommitted work\n"
    assert _git(repo, "show", f"{branch}:README.md").stdout == "modified in parking\n"


def test_reclaim_detached_parking_skipped_not_removed(tmp_path, caplog):
    """(c) A DETACHED parking (branch=None) is SKIPPED + LOUD, NOT removed —
    the data-loss guard (content not provably on a ref)."""
    caplog.set_level(logging.WARNING, logger="reclaim_orphaned_worktrees")
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}", detached=True)
    records = row.list_parked_worktrees(repo, _parking_root(repo))
    assert records and records[0].branch is None

    outcome = row.reclaim_worktrees(repo, records, dry_run=False)

    assert _resolved(outcome.skipped) == [parking.resolve()]
    assert outcome.reclaimed == []
    assert parking.exists()  # never removed
    warn = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(LOG_PREFIX in m and str(parking.resolve()) in m for m in warn)


def test_reclaim_unresolvable_branch_skipped_not_removed(tmp_path):
    """(c') A record naming a branch that does NOT resolve is SKIPPED, NOT
    removed (guard fires even when the path is a real registered parking)."""
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}")
    bogus = row.ParkedWorktree(
        path=parking.resolve(), branch="task/no-such-branch", parked_at=PARKED_AT_OLD
    )

    outcome = row.reclaim_worktrees(repo, [bogus], dry_run=False)

    assert _resolved(outcome.skipped) == [parking.resolve()]
    assert outcome.reclaimed == []
    assert parking.exists()


def test_reclaim_dry_run_removes_and_commits_nothing(tmp_path, caplog):
    """(d) dry_run=True removes/commits NOTHING and logs 'would reclaim'."""
    caplog.set_level(logging.INFO, logger="reclaim_orphaned_worktrees")
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}", dirty=True)
    records = row.list_parked_worktrees(repo, _parking_root(repo))
    head_before = _git(parking, "rev-parse", "HEAD").stdout.strip()

    outcome = row.reclaim_worktrees(repo, records, dry_run=True)

    assert parking.exists()  # nothing removed
    assert _git(parking, "rev-parse", "HEAD").stdout.strip() == head_before  # nothing committed
    assert outcome.reclaimed == []
    assert outcome.park_committed == []
    msgs = [r.getMessage() for r in caplog.records]
    assert any(
        LOG_PREFIX in m and "would reclaim" in m and str(parking.resolve()) in m
        for m in msgs
    ), f"missing 'would reclaim' line; got {msgs}"


def test_reclaim_remove_failure_counted_siblings_continue(tmp_path, monkeypatch):
    """(e) One worktree whose remove fails is counted in failed; its sibling is
    still reclaimed and the call never raises."""
    repo = _init_repo(tmp_path)
    good = _add_parking(repo, f"2920-{TS_OLD}")
    bad = _add_parking(repo, f"_lane-0-{TS_LANE}")
    records = row.list_parked_worktrees(repo, _parking_root(repo))

    real_remove = row.remove_worktree

    def fake_remove(repo_arg, path):
        if Path(path).resolve() == bad.resolve():
            return False  # simulate `git worktree remove` failing
        return real_remove(repo_arg, path)

    monkeypatch.setattr(row, "remove_worktree", fake_remove)

    outcome = row.reclaim_worktrees(repo, records, dry_run=False)  # must not raise

    assert good.resolve() in _resolved(outcome.reclaimed)
    assert bad.resolve() in _resolved(outcome.failed)
    assert not good.exists()  # sibling still reclaimed
    assert bad.exists()  # the failed one is left in place


# ---------------------------------------------------------------------------
# step-15: end-to-end CLI via subprocess (JSON on stdout / LOUD logs on stderr)
# ---------------------------------------------------------------------------

def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Drive the reclaim CLI as a real subprocess. stdout carries the JSON
    report; stderr carries the LOUD human log lines."""
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _cli_repo_with_three_parkings(tmp_path):
    """A repo with a CLEAN+OLD, a DIRTY+OLD, and a YOUNG parking."""
    repo = _init_repo(tmp_path)
    clean_old = _add_parking(repo, f"2920-{TS_OLD}")
    dirty_old = _add_parking(repo, f"_lane-0-{TS_LANE}", dirty=True)
    young = _add_parking(repo, f"3050-{TS_YOUNG}")
    return repo, clean_old, dirty_old, young


def test_cli_reclaims_old_park_commits_dirty_keeps_young(tmp_path):
    """Old-clean removed; old-dirty park-committed then removed (content on the
    branch); young untouched; prune ran (no stale admin entries); JSON report on
    stdout with correct counts + check=false; exit 0; LOUD prefix on stderr."""
    repo, clean_old, dirty_old, young = _cli_repo_with_three_parkings(tmp_path)

    result = _run_cli("--repo", str(repo), "--now", str(CLI_NOW), "--min-age-hours", "48")

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert not clean_old.exists()
    assert not dirty_old.exists()
    assert young.exists()
    # Dirty content provably preserved on the parking branch.
    assert _git(repo, "show", f"task/_lane-0-{TS_LANE}:wip.txt").stdout == "uncommitted work\n"
    # Prune ran: no stale/prunable admin entries; only main + young remain.
    porcelain = _git(repo, "worktree", "list", "--porcelain").stdout
    assert "prunable" not in porcelain
    assert _worktree_paths(repo) == {repo.resolve(), young.resolve()}

    report = json.loads(result.stdout)
    assert report["scanned"] == 3
    assert report["kept"] == 1
    assert report["reclaimed"] == 2
    assert report["park_committed"] == 1
    assert report["skipped"] == 0
    assert report["failed"] == 0
    assert report["check"] is False
    assert LOG_PREFIX in result.stderr


def test_cli_check_is_dry_run(tmp_path):
    """--check removes/commits NOTHING, reports check=true & reclaimed=0, logs
    'would reclaim', exits 0."""
    repo, clean_old, dirty_old, young = _cli_repo_with_three_parkings(tmp_path)
    head_before = _git(dirty_old, "rev-parse", "HEAD").stdout.strip()

    result = _run_cli(
        "--repo", str(repo), "--now", str(CLI_NOW), "--min-age-hours", "48", "--check"
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert clean_old.exists()
    assert dirty_old.exists()
    assert young.exists()
    assert _git(dirty_old, "rev-parse", "HEAD").stdout.strip() == head_before  # nothing committed
    report = json.loads(result.stdout)
    assert report["check"] is True
    assert report["reclaimed"] == 0
    assert report["park_committed"] == 0
    assert "would reclaim" in result.stderr


def test_cli_default_parking_root_derived_from_repo(tmp_path):
    """--parking-root omitted resolves to <repo>/.worktrees-orphaned."""
    repo = _init_repo(tmp_path)

    result = _run_cli("--repo", str(repo), "--now", str(CLI_NOW), "--check")

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    report = json.loads(result.stdout)
    assert report["root"] == str(repo / ".worktrees-orphaned")
