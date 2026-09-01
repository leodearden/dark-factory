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
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import reclaim_orphaned_worktrees as row
from reclaim_orphaned_worktrees import parse_parking_dir_name

from df_pytest_isolation import (
    _GIT_REDIRECT_ENV,
    _GIT_REDIRECT_ENV_PREFIXES,
    git_redirect_env,
)

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
# _run_git: the shared subprocess chokepoint is fail-safe under a git timeout
# ---------------------------------------------------------------------------

def test_run_git_timeout_returns_fail_safe_nonzero_without_raising(tmp_path, monkeypatch):
    """A hung git invocation that trips the subprocess ``timeout`` must be mapped
    to a fail-safe ``(1, '', <msg>)`` — NOT propagated. ``subprocess.TimeoutExpired``
    is a ``SubprocessError``, not an ``OSError``, so a bare ``except OSError`` would
    let it escape and blow up the UNGUARDED chokepoints (``git worktree list`` in
    list_parked_worktrees, ``git worktree prune`` in main), breaking the
    never-raise / always-exit-0 contract the nightly timer relies on when the
    concurrently-running orchestrator holds an index.lock on the same repo."""
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["git", "status"], timeout=120)

    monkeypatch.setattr(row.subprocess, "run", fake_run)

    rc, out, err = row._run_git(["status", "--porcelain"], cwd=tmp_path)  # must not raise

    assert rc == 1
    assert out == ""
    assert err  # carries the TimeoutExpired message for the LOUD log


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

def _run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Drive the reclaim CLI as a real subprocess. stdout carries the JSON
    report; stderr carries the LOUD human log lines.

    ``env`` hands the child a full replacement environment — used by the
    leaked-GIT_DIR regression below to poison the child WITHOUT mutating this
    process's ``os.environ`` (which the suite's own git-hermeticity fixtures
    own). ``None`` inherits, exactly as before.
    """
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
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


# ---------------------------------------------------------------------------
# step-16: ambient GIT_* redirection is fail-closed
# ---------------------------------------------------------------------------

# An environment poisoned with every name that retargets git away from the path
# it is given, plus one indexed `git -c` pair. `-C <path>` / `cwd=` only change
# DIRECTORY; GIT_DIR and its siblings SKIP repository discovery outright, so an
# ambient GIT_DIR redirects EVERY git call this module makes regardless of
# --repo / --parking-root / cwd. Measured against the pre-guard script: under
# `GIT_DIR=<decoy>/.git`, `git worktree remove --force` destroyed a DECOY
# repository's worktree while the sandbox named by --repo was never touched.
POISONED_GIT_ENV = {
    "GIT_DIR": "/decoy/.git",
    "GIT_WORK_TREE": "/decoy",
    "GIT_INDEX_FILE": "/decoy/.git/index",
    "GIT_COMMON_DIR": "/decoy/.git",
    "GIT_OBJECT_DIRECTORY": "/decoy/.git/objects",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/decoy/.git/objects",
    "GIT_NAMESPACE": "decoy",
    "GIT_CONFIG_GLOBAL": "/decoy/gitconfig",
    "GIT_CONFIG_SYSTEM": "/decoy/gitconfig",
    "GIT_CONFIG_COUNT": "1",
    "GIT_CONFIG_KEY_0": "core.hooksPath",
    "GIT_CONFIG_VALUE_0": "/decoy/hooks",
}

# Names the scrub must LEAVE ALONE. The ceiling is a DIFFERENT defence's own
# mechanism (df_pytest_isolation._df_git_ceiling_at_basetemp, incident
# esc-3072-3), so scrubbing it here would disarm a sibling guard whenever this
# script runs under the suite; the identity vars change what a commit SAYS,
# never where it lands.
PRESERVED_GIT_ENV = {
    "GIT_CEILING_DIRECTORIES": "/tmp/pytest-basetemp",
    "GIT_AUTHOR_NAME": "Reclaim Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
}


def test_scrubbed_git_env_drops_every_redirecting_name():
    """No redirecting name — exact or indexed-prefix — survives the scrub."""
    scrubbed = row.scrubbed_git_env(dict(POISONED_GIT_ENV))

    for name in POISONED_GIT_ENV:
        assert name not in scrubbed, f"{name} survived the scrub"


def test_scrubbed_git_env_forces_lc_all_c_over_ambient_locale():
    """LC_ALL=C is FORCED (porcelain stability), overriding an ambient locale —
    the one behaviour carried over verbatim from the pre-guard env build."""
    scrubbed = row.scrubbed_git_env({"LC_ALL": "en_US.UTF-8", "GIT_DIR": "/decoy/.git"})

    assert scrubbed["LC_ALL"] == "C"


def test_scrubbed_git_env_preserves_unrelated_vars():
    """Everything that is not a git-redirecting name passes through verbatim —
    the scrub is a targeted removal, not a hermetic env rebuild (the systemd
    unit's PATH/HOME must survive)."""
    scrubbed = row.scrubbed_git_env(
        {"PATH": "/usr/bin:/bin", "HOME": "/home/leo", "GIT_DIR": "/decoy/.git"}
    )

    assert scrubbed["PATH"] == "/usr/bin:/bin"
    assert scrubbed["HOME"] == "/home/leo"


def test_scrubbed_git_env_preserves_ceiling_and_identity_vars():
    """GIT_CEILING_DIRECTORIES and the identity vars are DELIBERATELY kept."""
    scrubbed = row.scrubbed_git_env({**POISONED_GIT_ENV, **PRESERVED_GIT_ENV})

    for name, value in PRESERVED_GIT_ENV.items():
        assert scrubbed[name] == value, f"{name} must survive the scrub"


def test_scrubbed_git_env_does_not_mutate_input():
    """Pure: the caller's mapping (in production, ``os.environ``) is untouched."""
    environ = dict(POISONED_GIT_ENV)
    before = dict(environ)

    row.scrubbed_git_env(environ)

    assert environ == before


def test_scrubbed_git_env_matches_shared_redirect_classifier():
    """DRIFT PIN. This module is STDLIB-ONLY by design (the systemd wrapper runs
    plain `python3`, and df_pytest_isolation imports pytest), so the redirecting
    names are DUPLICATED into the script rather than imported. This test is the
    only legal place to import the shared definition, and it pins the copy
    against it in BOTH directions: a name added to df_pytest_isolation's list
    that the script does not scrub, or a name the script scrubs that the shared
    classifier does not consider redirecting, fails here."""
    assert row._GIT_REDIRECT_ENV == _GIT_REDIRECT_ENV
    assert row._GIT_REDIRECT_ENV_PREFIXES == _GIT_REDIRECT_ENV_PREFIXES

    # ...and behaviourally, over a mapping derived from the SHARED list (so a
    # widening there poisons this mapping too): every name the shared classifier
    # calls redirecting is absent from the scrub's output.
    derived = {name: "/decoy" for name in _GIT_REDIRECT_ENV}
    derived.update({f"{prefix}0": "/decoy" for prefix in _GIT_REDIRECT_ENV_PREFIXES})
    redirecting = git_redirect_env(derived)
    assert redirecting, "the shared classifier must classify the derived mapping"

    scrubbed = row.scrubbed_git_env(derived)
    assert [name for name in redirecting if name in scrubbed] == []


def test_run_git_chokepoint_scrubs_ambient_redirection(tmp_path, monkeypatch):
    """The scrub is reached at the SINGLE chokepoint: the env handed to
    ``subprocess.run`` carries no redirecting name, whatever the ambient one
    holds. Asserted at ``_run_git`` (not per call site) because the defect class
    is "a call site that forgets the guard" — fixing today's seven leaves the
    hole open for the eighth."""
    recorded: dict[str, dict[str, str]] = {}

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(*_args, **kwargs):
        recorded["env"] = kwargs["env"]
        return _Completed()

    for name, value in {**POISONED_GIT_ENV, **PRESERVED_GIT_ENV}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LC_ALL", "en_US.UTF-8")
    monkeypatch.setattr(row.subprocess, "run", fake_run)

    row._run_git(["status", "--porcelain"], cwd=tmp_path)

    env = recorded["env"]
    for name in POISONED_GIT_ENV:
        assert name not in env, f"{name} reached git through the chokepoint"
    for name, value in PRESERVED_GIT_ENV.items():
        assert env[name] == value, f"{name} must survive to git"
    assert env["LC_ALL"] == "C"


def _decoy_and_sandbox(tmp_path):
    """A DECOY repo owning a DIRTY worktree registered under the SANDBOX's
    parking root, plus the sandbox's own eligible parking.

    Reproduces the measured incident layout: ``git worktree list`` under a
    leaked ``GIT_DIR`` enumerates the DECOY's worktrees, and the parking-root
    band guard passes the decoy-owned one because it physically sits under the
    sandbox's quarantine base.
    """
    decoy_base = tmp_path / "decoy"
    decoy_base.mkdir()
    decoy = _init_repo(decoy_base)

    sandbox_base = tmp_path / "sandbox"
    sandbox_base.mkdir()
    sandbox = _init_repo(sandbox_base)

    own = _add_parking(sandbox, f"2920-{TS_OLD}")

    decoy_parking = _parking_root(sandbox) / f"d1-{TS_OLD}"
    _git(decoy, "worktree", "add", "-q", "-b", f"task/d1-{TS_OLD}", str(decoy_parking))
    (decoy_parking / "wip.txt").write_text("decoy work\n")

    return decoy, sandbox, own, decoy_parking


def _ref_snapshot(repo: Path) -> str:
    """Every ref and its sha — a whole-repo mutation detector."""
    return _git(repo, "for-each-ref", "--format=%(refname) %(objectname)").stdout


def test_cli_leaked_git_dir_never_touches_the_decoy_repository(tmp_path):
    """END-TO-END REGRESSION (incident 2026-08-31). Under an ambient
    ``GIT_DIR`` naming a DECOY repository, the sweep must act ONLY on the repo
    ``--repo`` names.

    Measured on the pre-guard script, BOTH halves failed: the decoy's registered
    worktree was park-committed into the decoy and then destroyed by
    ``git worktree remove --force``, while the sandbox's own eligible parking
    was never swept — the JSON report nonetheless claiming ``reclaimed=1``.
    """
    decoy, sandbox, own, decoy_parking = _decoy_and_sandbox(tmp_path)
    decoy_refs_before = _ref_snapshot(decoy)

    result = _run_cli(
        "--repo", str(sandbox),
        "--now", str(CLI_NOW),
        "--min-age-hours", "48",
        env=dict(os.environ, GIT_DIR=str(decoy / ".git")),
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"

    # THE DECOY IS UNTOUCHED: no redirected commit landed, nothing was removed.
    assert _ref_snapshot(decoy) == decoy_refs_before
    assert decoy_parking.exists()
    assert decoy_parking.resolve() in _worktree_paths(decoy)

    # POSITIVE CONTROL: the repository --repo actually named WAS swept, so the
    # guard refuses the wrong target rather than refusing everything.
    assert not own.exists()
    assert _branch_resolves(sandbox, f"task/2920-{TS_OLD}")
    report = json.loads(result.stdout)
    assert report["reclaimed"] == 1
    assert report["reclaimed_paths"] == [str(own.resolve())]


# ---------------------------------------------------------------------------
# step-17: residual fail-closed target verify — is_git_toplevel(path)
# ---------------------------------------------------------------------------

def test_is_git_toplevel_true_for_repo_root(tmp_path):
    repo = _init_repo(tmp_path)
    assert row.is_git_toplevel(repo) is True


def test_is_git_toplevel_true_for_registered_parking(tmp_path):
    """A LINKED worktree resolves its OWN path as toplevel (measured), so a
    per-parking verify is sound — park_commit can self-guard the worktree it is
    about to `add -A` from."""
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}")

    assert row.is_git_toplevel(parking) is True


def test_is_git_toplevel_true_through_a_symlink(tmp_path):
    """A repo reached via a SYMLINK passes. ``--show-toplevel`` reports the REAL
    path, so BOTH sides must be realpath-resolved before comparison — comparing
    the literal strings would false-refuse a legitimately symlinked checkout,
    turning the guard into an outage."""
    repo = _init_repo(tmp_path)
    link = tmp_path / "repo-link"
    link.symlink_to(repo)

    assert row.is_git_toplevel(link) is True


def test_is_git_toplevel_false_for_subdirectory_of_repo(tmp_path):
    """The misconfiguration this guard exists to catch: git happily discovers
    the ENCLOSING repo from a subdirectory, so the sweep would act on a
    repository the caller only half-named."""
    repo = _init_repo(tmp_path)
    subdir = repo / "subdir"
    subdir.mkdir()

    assert row.is_git_toplevel(subdir) is False


def test_is_git_toplevel_false_for_subdirectory_inside_parking(tmp_path):
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}")
    subdir = parking / "sub"
    subdir.mkdir()

    assert row.is_git_toplevel(subdir) is False


def test_is_git_toplevel_false_for_non_repository(tmp_path):
    """``rev-parse`` exits 128 outside a repository — fail CLOSED."""
    plain = tmp_path / "not-a-repo"
    plain.mkdir()

    assert row.is_git_toplevel(plain) is False


def test_is_git_toplevel_false_for_missing_path_without_raising(tmp_path):
    """A vanished target reaches ``_run_git``'s OSError path, which the module's
    never-raise contract maps to a non-zero rc. The predicate must inherit that
    fail-closed and NOT propagate — the sweep never raises."""
    assert row.is_git_toplevel(tmp_path / "does-not-exist") is False


# ---------------------------------------------------------------------------
# step-18: reclaim_worktrees refuses an unverified target
# ---------------------------------------------------------------------------

def _repo_with_clean_and_dirty_parkings(tmp_path):
    """A repo with one CLEAN-old and one DIRTY-old parking, plus its records."""
    repo = _init_repo(tmp_path)
    clean = _add_parking(repo, f"2920-{TS_OLD}")
    dirty = _add_parking(repo, f"_lane-0-{TS_LANE}", dirty=True)
    records = row.list_parked_worktrees(repo, _parking_root(repo))
    assert len(records) == 2
    return repo, clean, dirty, records


def test_reclaim_refuses_whole_sweep_when_repo_is_not_toplevel(tmp_path, caplog):
    """A ``repo`` that is not the ROOT of the repository git resolves for it
    refuses the ENTIRE sweep before any branch check, park-commit or removal.

    git discovers the enclosing repo from a subdirectory, so without this gate
    the sweep would happily act on a repository the caller only half-named."""
    caplog.set_level(logging.WARNING, logger="reclaim_orphaned_worktrees")
    repo, clean, dirty, records = _repo_with_clean_and_dirty_parkings(tmp_path)
    head_before = _git(dirty, "rev-parse", "HEAD").stdout.strip()
    subdir = repo / "subdir"
    subdir.mkdir()

    outcome = row.reclaim_worktrees(subdir, records, dry_run=False)  # must not raise

    assert outcome.refused is True
    assert _resolved(outcome.skipped) == [r.path.resolve() for r in records]
    assert outcome.reclaimed == []
    assert outcome.park_committed == []
    assert outcome.failed == []
    # Nothing removed, nothing committed.
    assert clean.exists() and dirty.exists()
    assert {clean.resolve(), dirty.resolve()} <= _worktree_paths(repo)
    assert _git(dirty, "rev-parse", "HEAD").stdout.strip() == head_before
    warn = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(LOG_PREFIX in m and "REFUSING" in m and str(subdir) in m for m in warn)


def test_reclaim_refusal_also_applies_in_dry_run(tmp_path, caplog):
    """--check exists so an operator can see what the real run WOULD do. A dry
    run that printed 'would reclaim' while the real run would refuse is an
    actively misleading answer — and this misconfiguration is exactly the one a
    dry run most needs to surface."""
    # INFO, not WARNING: the "would reclaim" line this test asserts is ABSENT is
    # logged at INFO, so a WARNING floor would make that half vacuously true.
    caplog.set_level(logging.INFO, logger="reclaim_orphaned_worktrees")
    repo, clean, dirty, records = _repo_with_clean_and_dirty_parkings(tmp_path)
    subdir = repo / "subdir"
    subdir.mkdir()

    outcome = row.reclaim_worktrees(subdir, records, dry_run=True)

    assert outcome.refused is True
    assert _resolved(outcome.skipped) == [r.path.resolve() for r in records]
    assert outcome.reclaimed == []
    warn = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(LOG_PREFIX in m and "REFUSING" in m for m in warn)
    assert not any("would reclaim" in r.getMessage() for r in caplog.records)


def test_reclaim_proceeds_normally_through_a_symlinked_repo(tmp_path):
    """NON-REGRESSION: a repo reached via a SYMLINK is a legitimate target and
    must still be swept — the guard compares realpaths, not literal strings."""
    repo, clean, dirty, _records = _repo_with_clean_and_dirty_parkings(tmp_path)
    link = tmp_path / "repo-link"
    link.symlink_to(repo)
    records = row.list_parked_worktrees(link, _parking_root(link))

    outcome = row.reclaim_worktrees(link, records, dry_run=False)

    assert outcome.refused is False
    assert set(_resolved(outcome.reclaimed)) == {clean.resolve(), dirty.resolve()}
    assert not clean.exists()
    assert not dirty.exists()
    assert _branch_resolves(repo, f"task/_lane-0-{TS_LANE}")


# ---------------------------------------------------------------------------
# step-19: park_commit self-guards its own destructive verbs
# ---------------------------------------------------------------------------

def test_park_commit_refuses_a_worktree_it_cannot_verify(tmp_path, caplog):
    """``git add -A`` stages from the WORKTREE ROOT regardless of cwd, so an
    unverified target does not merely mis-scope the snapshot — it commits the
    whole tree of whatever repository git resolved. park_commit self-guards
    rather than trusting its caller to have checked."""
    caplog.set_level(logging.WARNING, logger="reclaim_orphaned_worktrees")
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}", dirty=True)
    subdir = parking / "sub"
    subdir.mkdir()
    (subdir / "more-wip.txt").write_text("nested uncommitted work\n")
    head_before = _git(parking, "rev-parse", "HEAD").stdout.strip()

    assert row.park_commit(subdir, "reclaim") is None  # must not raise

    assert _git(parking, "rev-parse", "HEAD").stdout.strip() == head_before
    assert _git(parking, "status", "--porcelain").stdout.strip()  # still dirty
    warn = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(LOG_PREFIX in m and "REFUSING" in m and str(subdir) in m for m in warn)


def test_park_commit_still_commits_a_verified_dirty_parking(tmp_path):
    """NON-REGRESSION: the guard must not disarm the ordinary happy path."""
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}", dirty=True)
    head_before = _git(parking, "rev-parse", "HEAD").stdout.strip()

    sha = row.park_commit(parking, "reclaim")

    assert sha and sha != head_before
    assert _git(parking, "rev-parse", "HEAD").stdout.strip() == sha
    assert not _git(parking, "status", "--porcelain").stdout.strip()


def test_park_commit_still_commits_through_a_symlinked_parking(tmp_path):
    """NON-REGRESSION: a parking reached via a SYMLINK is a legitimate target —
    the guard realpath-compares, so it must not false-refuse here."""
    repo = _init_repo(tmp_path)
    parking = _add_parking(repo, f"2920-{TS_OLD}", dirty=True)
    link = tmp_path / "parking-link"
    link.symlink_to(parking)

    sha = row.park_commit(link, "reclaim")

    assert sha
    assert _git(parking, "rev-parse", "HEAD").stdout.strip() == sha


# ---------------------------------------------------------------------------
# step-20: end-to-end CLI refusal — refused_target in the JSON report, no prune
# ---------------------------------------------------------------------------

def _cli_repo_with_live_and_prunable_parkings(tmp_path):
    """A repo with one LIVE old parking and one whose directory was deleted, so
    ``git worktree list --porcelain`` reports a ``prunable`` admin entry.

    The prunable entry is the observable for the FINAL ``git worktree prune``:
    it is the only phase whose effect survives after the sweep, so "did the
    prune run against this repo?" is answerable from the porcelain alone.
    """
    repo = _init_repo(tmp_path)
    live = _add_parking(repo, f"2920-{TS_OLD}")
    stale = _add_parking(repo, f"_lane-0-{TS_LANE}")
    shutil.rmtree(stale)
    assert "prunable" in _git(repo, "worktree", "list", "--porcelain").stdout
    return repo, live, stale


def test_cli_refuses_unverified_repo_target_and_skips_the_prune(tmp_path):
    """A ``--repo`` that is not the repository root refuses every phase — the
    sweep AND the final prune — while still exiting 0 so the nightly timer is
    never wedged. The refusal is carried by the JSON report's ``refused_target``
    key, because the always-0 exit code cannot carry it and ``skipped`` alone
    cannot distinguish 'detached branch' from 'refused target' (INV-4)."""
    repo, live, _stale = _cli_repo_with_live_and_prunable_parkings(tmp_path)
    subdir = repo / "subdir"
    subdir.mkdir()

    result = _run_cli(
        "--repo", str(subdir),
        "--parking-root", str(_parking_root(repo)),
        "--now", str(CLI_NOW),
        "--min-age-hours", "48",
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert live.exists()
    assert live.resolve() in _worktree_paths(repo)

    report = json.loads(result.stdout)
    assert report["refused_target"] is True
    assert report["reclaimed"] == 0
    assert report["park_committed"] == 0
    assert report["scanned"] > 0
    assert report["skipped"] == report["scanned"]
    assert LOG_PREFIX in result.stderr
    assert "REFUSING" in result.stderr

    # The final prune did NOT run against the unverified target.
    assert "prunable" in _git(repo, "worktree", "list", "--porcelain").stdout


def test_prune_gate_reuses_the_sweep_verdict_instead_of_re_probing(tmp_path, capsys, monkeypatch):
    """The report and the prune read ONE verdict, so they can never disagree.

    ``_run_git`` never raises — a 120s timeout or a transient ``OSError`` is
    mapped to rc=1 — so a SECOND ``is_git_toplevel(repo)`` probe in ``main``
    could answer differently from the one ``reclaim_worktrees`` already recorded.
    Simulated here with a probe that answers False once and truthfully after:
    against a re-probing prune gate the report would claim ``refused_target:
    true`` while the prune actually RAN (clearing the prunable entry) — exactly
    the report/behaviour ambiguity ``refused_target`` exists to remove.

    Driven in-process (not via ``_run_cli``) because monkeypatching cannot cross
    the subprocess boundary.
    """
    repo, live, _stale = _cli_repo_with_live_and_prunable_parkings(tmp_path)
    truthful = row.is_git_toplevel
    repo_probes: list[Path] = []

    def flaky(path):
        if Path(path).resolve() == repo.resolve():
            repo_probes.append(Path(path))
            return len(repo_probes) > 1  # first answer False, then truthful
        return truthful(path)

    monkeypatch.setattr(row, "is_git_toplevel", flaky)

    rc = row.main(
        ["--repo", str(repo), "--now", str(CLI_NOW), "--min-age-hours", "48"]
    )

    assert rc == 0  # a refusal is a missed sweep, never a wedged timer
    report = json.loads(capsys.readouterr().out)
    assert report["refused_target"] is True
    # ...and the prune AGREES with that report: the stale entry survives.
    assert "prunable" in _git(repo, "worktree", "list", "--porcelain").stdout
    assert live.exists()
    # One decision, one probe — the second source of truth is gone.
    assert len(repo_probes) == 1


def test_cli_verified_run_reports_not_refused_and_still_prunes(tmp_path):
    """NON-REGRESSION: an ordinary run reports ``refused_target`` false, still
    reclaims, and still clears the stale admin entry."""
    repo, live, _stale = _cli_repo_with_live_and_prunable_parkings(tmp_path)

    result = _run_cli(
        "--repo", str(repo), "--now", str(CLI_NOW), "--min-age-hours", "48"
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    report = json.loads(result.stdout)
    assert report["refused_target"] is False
    assert report["reclaimed"] == 1
    assert not live.exists()
    assert "prunable" not in _git(repo, "worktree", "list", "--porcelain").stdout


def test_cli_symlinked_repo_behaves_exactly_like_the_direct_path_run(tmp_path):
    """NON-REGRESSION: a ``--repo`` that is a SYMLINK to the repo root is a
    legitimate target — reclaims, prunes, ``refused_target`` false. A guard that
    compared literal paths would turn every symlinked deployment into a silently
    skipped nightly."""
    repo, live, _stale = _cli_repo_with_live_and_prunable_parkings(tmp_path)
    link = tmp_path / "repo-link"
    link.symlink_to(repo)

    result = _run_cli(
        "--repo", str(link), "--now", str(CLI_NOW), "--min-age-hours", "48"
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    report = json.loads(result.stdout)
    assert report["refused_target"] is False
    assert report["reclaimed"] == 1
    assert not live.exists()
    assert "prunable" not in _git(repo, "worktree", "list", "--porcelain").stdout
