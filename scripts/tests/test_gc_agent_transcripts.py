"""Tests for scripts/gc_agent_transcripts.py — the retention GC sweep over the
gzipped agent-transcript archive (task 2731, δ of
plans/agent-transcript-archival-prd.md).

step-1: the pure age-cap arm of select_prunable(task_dirs, now, max_age_days,
max_task_dirs) exercised with synthetic (Path, mtime) tuples — NO filesystem,
fixed NOW. A dir older than now - max_age_days*86400 prunes with reason 'age';
a fresh dir is kept; the exact boundary (now - mtime == max_age_days*86400) is
KEPT (strict >, not >=); max_age_days <= 0 disables the age axis.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

import gc_agent_transcripts as gct
import pytest
from gc_agent_transcripts import select_prunable

LOG_PREFIX = "gc_agent_transcripts:"

DAY = 86_400
NOW = 1_000_000_000.0

# A count cap far larger than any dir count used in the age tests, so the
# count arm (once implemented in step-4) never perturbs a pure-age assertion.
HIGH_COUNT_CAP = 10_000


def _dir(name: str) -> Path:
    return Path("/archive") / name


# ---------------------------------------------------------------------------
# step-1: pure age-cap arm
# ---------------------------------------------------------------------------

def test_age_arm_prunes_dir_older_than_cutoff():
    cutoff = NOW - 90 * DAY
    old = _dir("old")
    fresh = _dir("fresh")
    task_dirs = [(old, cutoff - DAY), (fresh, NOW)]

    decision = select_prunable(task_dirs, NOW, max_age_days=90, max_task_dirs=HIGH_COUNT_CAP)

    assert decision.prune_paths == {old}
    assert decision.keep_paths == {fresh}
    assert decision.reasons[old] == "age"


def test_age_arm_keeps_fresh_dir():
    fresh = _dir("fresh")
    decision = select_prunable([(fresh, NOW)], NOW, max_age_days=90, max_task_dirs=HIGH_COUNT_CAP)

    assert decision.keep_paths == {fresh}
    assert decision.prune_paths == set()


def test_age_boundary_exact_is_kept_one_second_older_is_pruned():
    """now - mtime == max_age_days*86400 exactly is KEPT (strict >, not >=);
    one second older is pruned."""
    cutoff = NOW - 90 * DAY
    boundary = _dir("boundary")  # mtime == cutoff -> now - mtime == 90*DAY exactly
    just_old = _dir("just_old")  # one second older than the boundary
    task_dirs = [(boundary, cutoff), (just_old, cutoff - 1)]

    decision = select_prunable(task_dirs, NOW, max_age_days=90, max_task_dirs=HIGH_COUNT_CAP)

    assert decision.keep_paths == {boundary}
    assert decision.prune_paths == {just_old}
    assert decision.reasons[just_old] == "age"


def test_non_positive_max_age_disables_age_pruning():
    """max_age_days <= 0 imposes no age bound: even an ancient dir is kept
    (count also disabled here, so nothing prunes at all)."""
    ancient = _dir("ancient")
    task_dirs = [(ancient, NOW - 10_000 * DAY)]

    for disabled in (0, -1):
        decision = select_prunable(
            task_dirs, NOW, max_age_days=disabled, max_task_dirs=0,
        )
        assert decision.keep_paths == {ancient}
        assert decision.prune_paths == set()


# ---------------------------------------------------------------------------
# step-3: pure count-cap arm + union / reason tagging
# ---------------------------------------------------------------------------

def _fresh_dirs(n: int) -> list[tuple[Path, float]]:
    """n fresh dirs (well within any age window), strictly descending mtime:
    t0 is newest, t{n-1} the oldest."""
    return [(_dir(f"t{i}"), NOW - i * DAY) for i in range(n)]


def test_count_arm_keeps_newest_cap_prunes_older_tail():
    dirs = _fresh_dirs(4)  # t0 newest .. t3 oldest
    decision = select_prunable(dirs, NOW, max_age_days=0, max_task_dirs=2)

    assert decision.keep_paths == {_dir("t0"), _dir("t1")}
    assert decision.prune_paths == {_dir("t2"), _dir("t3")}
    assert decision.reasons[_dir("t2")] == "count"
    assert decision.reasons[_dir("t3")] == "count"


def test_count_exactly_at_cap_keeps_all():
    dirs = _fresh_dirs(2)
    decision = select_prunable(dirs, NOW, max_age_days=0, max_task_dirs=2)

    assert decision.keep_paths == {_dir("t0"), _dir("t1")}
    assert decision.prune_paths == set()


def test_count_cap_plus_one_prunes_single_oldest():
    dirs = _fresh_dirs(3)  # t0 newest .. t2 oldest
    decision = select_prunable(dirs, NOW, max_age_days=0, max_task_dirs=2)

    assert decision.prune_paths == {_dir("t2")}
    assert decision.reasons[_dir("t2")] == "count"
    assert decision.keep_paths == {_dir("t0"), _dir("t1")}


def test_non_positive_max_task_dirs_disables_count_pruning():
    dirs = _fresh_dirs(5)
    for disabled in (0, -1):
        decision = select_prunable(dirs, NOW, max_age_days=0, max_task_dirs=disabled)
        assert decision.prune_paths == set()
        assert decision.keep_paths == {path for path, _mtime in dirs}


def test_age_reason_alone_when_within_count_cap():
    """An old dir that still sits within the count cap is tagged 'age' only —
    the union logic must not spuriously add 'count'."""
    cutoff = NOW - 90 * DAY
    young = _dir("young")
    old = _dir("old")
    task_dirs = [(young, NOW), (old, cutoff - DAY)]

    decision = select_prunable(task_dirs, NOW, max_age_days=90, max_task_dirs=5)

    assert decision.keep_paths == {young}
    assert decision.reasons[old] == "age"


def test_union_age_and_count_reasons():
    """A dir failing BOTH bounds is 'age+count'; a dir failing only the count
    bound is 'count'."""
    cutoff = NOW - 90 * DAY
    young_a = _dir("young_a")   # newest — within count + age -> keep
    young_b = _dir("young_b")   # 2nd newest — within count + age -> keep
    young_c = _dir("young_c")   # beyond count cap but fresh -> 'count'
    old_d = _dir("old_d")       # beyond count cap AND older than cutoff -> 'age+count'
    task_dirs = [
        (young_a, NOW),
        (young_b, NOW - DAY),
        (young_c, NOW - 2 * DAY),
        (old_d, cutoff - DAY),
    ]

    decision = select_prunable(task_dirs, NOW, max_age_days=90, max_task_dirs=2)

    assert decision.keep_paths == {young_a, young_b}
    assert decision.prune_paths == {young_c, old_d}
    assert decision.reasons[young_c] == "count"
    assert decision.reasons[old_d] == "age+count"


# ---------------------------------------------------------------------------
# step-5: scan_task_dirs(root) against a real filesystem archive
# ---------------------------------------------------------------------------

def _touch(path: Path, mtime: float) -> None:
    """Create *path* (and parents) as a small file with the given mtime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    os.utime(path, (mtime, mtime))


def test_scan_reports_newest_descendant_mtime_per_task_dir(tmp_path):
    root = tmp_path / "agent-transcripts"
    # task dir "100": a single main-session .gz.
    _touch(root / "100" / "enc" / "sid1.jsonl.gz", NOW - 10 * DAY)
    # task dir "200": two files at different mtimes -> newest wins.
    _touch(root / "200" / "enc" / "sid2.jsonl.gz", NOW - 50 * DAY)
    _touch(root / "200" / "enc" / "sid3.jsonl.gz", NOW - 5 * DAY)
    # task dir "300": nested subagent transcript only.
    _touch(root / "300" / "enc" / "sid4" / "subagents" / "agent-1.jsonl.gz", NOW - 3 * DAY)
    # a stray non-directory file directly under root -> ignored.
    _touch(root / "loose.jsonl.gz", NOW)

    result = dict(gct.scan_task_dirs(root))

    assert set(result.keys()) == {root / "100", root / "200", root / "300"}
    assert result[root / "100"] == pytest.approx(NOW - 10 * DAY, abs=1)
    assert result[root / "200"] == pytest.approx(NOW - 5 * DAY, abs=1)  # max of the two
    assert result[root / "300"] == pytest.approx(NOW - 3 * DAY, abs=1)


def test_scan_empty_task_dir_falls_back_to_own_mtime(tmp_path):
    root = tmp_path / "agent-transcripts"
    empty = root / "999"
    empty.mkdir(parents=True)
    os.utime(empty, (NOW - 7 * DAY, NOW - 7 * DAY))

    result = dict(gct.scan_task_dirs(root))

    assert set(result.keys()) == {empty}
    assert result[empty] == pytest.approx(NOW - 7 * DAY, abs=1)


def test_scan_missing_root_returns_empty(tmp_path):
    assert gct.scan_task_dirs(tmp_path / "does-not-exist") == []


def test_scan_existing_but_empty_root_returns_empty(tmp_path):
    root = tmp_path / "agent-transcripts"
    root.mkdir(parents=True)
    assert gct.scan_task_dirs(root) == []


# ---------------------------------------------------------------------------
# amend (review): scan_task_dirs must be best-effort + never-raise under a
# stat() failure mid-scan. The archive is written CONCURRENTLY by the α
# producer hook, so a descendant .gz / task dir can vanish or become
# unstattable between the rglob walk and the stat() call (a classic TOCTOU
# race). The module docstring loudly promises "never-raise / always exit 0";
# an unguarded stat() OSError would propagate scan_task_dirs -> main ->
# traceback -> exit 1 before any pruning even starts, breaking that contract
# (unlike prune_task_dirs, which is already tested for OSError resilience).
#
# PermissionError/EACCES stands in for the whole OSError class the guard must
# survive: a bare FileNotFoundError/ENOENT is swallowed by Path.is_file() /
# Path.is_dir() (which ignore ENOENT and return False) BEFORE the guarded
# stat runs, so it cannot exercise — or crash — this code path at all.
# ---------------------------------------------------------------------------

def test_scan_survives_stat_failure_on_descendant_file(tmp_path, caplog, monkeypatch):
    """A descendant .gz that becomes unstattable mid-scan is skipped LOUDLY;
    the remaining readable files still set the dir's age and sibling task dirs
    are still scanned — the sweep never raises."""
    caplog.set_level(logging.WARNING, logger="gc_agent_transcripts")
    root = tmp_path / "agent-transcripts"
    good = root / "100" / "enc" / "good.jsonl.gz"
    vanished = root / "100" / "enc" / "vanished.jsonl.gz"
    _touch(good, NOW - 20 * DAY)
    _touch(vanished, NOW - 1 * DAY)  # the newest file — but it fails to stat
    _touch(root / "200" / "enc" / "sib.jsonl.gz", NOW - 8 * DAY)

    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self == vanished:
            raise PermissionError(13, "Permission denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)

    result = dict(gct.scan_task_dirs(root))  # must NOT raise

    # Sibling dir unaffected; the failing dir is still reported off its one
    # readable file (the unstattable newest file is skipped, not fatal).
    assert set(result.keys()) == {root / "100", root / "200"}
    assert result[root / "100"] == pytest.approx(NOW - 20 * DAY, abs=1)
    assert result[root / "200"] == pytest.approx(NOW - 8 * DAY, abs=1)

    warn_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        LOG_PREFIX in m and str(vanished) in m for m in warn_msgs
    ), f"missing LOUD WARNING for the unstattable descendant; got {warn_msgs}"


def test_scan_survives_stat_failure_on_task_dir(tmp_path, caplog, monkeypatch):
    """A task dir that becomes unstattable mid-scan is skipped LOUDLY and the
    sibling dirs are still scanned — the sweep never raises."""
    caplog.set_level(logging.WARNING, logger="gc_agent_transcripts")
    root = tmp_path / "agent-transcripts"
    bad = root / "100"
    bad.mkdir(parents=True)
    _touch(root / "200" / "enc" / "sib.jsonl.gz", NOW - 8 * DAY)

    real_stat = Path.stat

    def fake_stat(self, *args, **kwargs):
        if self == bad:
            raise PermissionError(13, "Permission denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)

    result = dict(gct.scan_task_dirs(root))  # must NOT raise

    assert set(result.keys()) == {root / "200"}  # the unstattable dir dropped
    assert result[root / "200"] == pytest.approx(NOW - 8 * DAY, abs=1)

    warn_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        LOG_PREFIX in m and str(bad) in m for m in warn_msgs
    ), f"missing LOUD WARNING for the unstattable task dir; got {warn_msgs}"


# ---------------------------------------------------------------------------
# step-7: prune_task_dirs(prune_records, *, dry_run) — best-effort + LOUD
# ---------------------------------------------------------------------------

def _make_task_dirs(root: Path, records: list[tuple[str, str]]) -> list[tuple[Path, str]]:
    """Materialise ``(name, reason)`` specs as populated task dirs under *root*;
    return the ``(task_dir_path, reason)`` prune-record list."""
    out: list[tuple[Path, str]] = []
    for name, reason in records:
        d = root / name
        (d / "enc").mkdir(parents=True)
        (d / "enc" / "x.jsonl.gz").write_bytes(b"x")
        out.append((d, reason))
    return out


def test_prune_removes_dirs_and_logs_loud_per_removal(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="gc_agent_transcripts")
    records = _make_task_dirs(tmp_path, [("100", "age"), ("200", "count")])

    outcome = gct.prune_task_dirs(records, dry_run=False)

    for d, _reason in records:
        assert not d.exists()
    assert set(outcome.removed) == {d for d, _ in records}
    assert outcome.failed == []

    info_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    for d, reason in records:
        assert any(
            LOG_PREFIX in m and str(d) in m and reason in m for m in info_msgs
        ), f"missing LOUD removal line for {d} (reason={reason}); got {info_msgs}"


def test_prune_dry_run_removes_nothing_but_logs_would_prune(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="gc_agent_transcripts")
    records = _make_task_dirs(tmp_path, [("100", "age"), ("200", "count")])

    outcome = gct.prune_task_dirs(records, dry_run=True)

    for d, _reason in records:
        assert d.exists()  # dry-run deletes nothing
    assert outcome.removed == []
    assert outcome.failed == []

    msgs = [r.getMessage() for r in caplog.records]
    for d, _reason in records:
        assert any(
            LOG_PREFIX in m and "would prune" in m and str(d) in m for m in msgs
        ), f"missing 'would prune' line for {d}; got {msgs}"


def test_prune_best_effort_on_rmtree_oserror(tmp_path, caplog, monkeypatch):
    """A per-dir rmtree OSError is logged at WARNING + counted in `failed`, the
    sweep never raises, and the sibling dirs are still removed."""
    caplog.set_level(logging.INFO, logger="gc_agent_transcripts")
    records = _make_task_dirs(
        tmp_path, [("100", "age"), ("200", "count"), ("300", "age+count")]
    )
    fail_dir = records[1][0]  # the "200" dir

    real_rmtree = shutil.rmtree

    def fake_rmtree(path, *args, **kwargs):
        if Path(path) == fail_dir:
            raise OSError(28, "No space left on device")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", fake_rmtree)

    outcome = gct.prune_task_dirs(records, dry_run=False)  # must not raise

    assert fail_dir.exists()  # the failing dir survives
    assert fail_dir in outcome.failed
    assert set(outcome.removed) == {records[0][0], records[2][0]}
    assert not records[0][0].exists()
    assert not records[2][0].exists()

    warn_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        LOG_PREFIX in m and str(fail_dir) in m for m in warn_msgs
    ), f"missing WARNING line for failed dir {fail_dir}; got {warn_msgs}"


# ---------------------------------------------------------------------------
# step-9: drift guard — the stdlib mirror must equal α's canonical config
# ---------------------------------------------------------------------------

def test_default_constants_match_orchestrator_config():
    """DRIFT GUARD: the GC is stdlib-only and mirrors α's retention config as
    module constants instead of importing OrchestratorConfig. Those constants
    must never silently diverge from the canonical
    orchestrator.config.{TranscriptArchiveConfig,RetentionConfig} defaults — a
    divergence here would prune against the wrong root/caps. Value equality
    (not docstring/introspection), mirroring
    test_drain_check.test_default_fleet_dir_matches_orchestrator_fleet_heartbeat.
    """
    from orchestrator.config import RetentionConfig, TranscriptArchiveConfig

    assert TranscriptArchiveConfig().root == gct.ARCHIVE_ROOT_RELATIVE
    assert gct.ARCHIVE_ROOT_RELATIVE == "data/orchestrator/agent-transcripts"
    assert gct.DEFAULT_MAX_AGE_DAYS == RetentionConfig().max_age_days == 90
    assert gct.DEFAULT_MAX_TASK_DIRS == RetentionConfig().max_task_dirs == 5000


# ---------------------------------------------------------------------------
# step-11: CLI end-to-end via subprocess — build_parser() / main()
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).parent.parent / "gc_agent_transcripts.py"


def _run_cli(*args):
    """Drive the GC CLI as a real subprocess (inherits the parent env / PATH).

    stdout carries the machine-readable JSON report; stderr carries the LOUD
    human log lines (basicConfig logs to stderr), so ``json.loads(stdout)`` sees
    pure JSON.
    """
    return subprocess.run(
        ["python3", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _make_cli_archive(root: Path, specs: list[tuple[str, float]]) -> None:
    """Build ``<root>/<name>/enc/session.jsonl.gz`` for each ``(name, mtime)``,
    stamping the ``.gz`` mtime via ``os.utime`` so ``scan_task_dirs`` reads it
    as that task dir's retention age."""
    root.mkdir(parents=True, exist_ok=True)
    for name, mtime in specs:
        gz = root / name / "enc" / "session.jsonl.gz"
        gz.parent.mkdir(parents=True)
        gz.write_bytes(b"x")
        os.utime(gz, (mtime, mtime))


# 5 fresh dirs (all within the age cap), distinct mtimes: "100" newest ...
# "104" oldest. Newest-first order is 100, 101, 102, 103, 104.
def _five_fresh_specs() -> list[tuple[str, float]]:
    return [(str(100 + i), NOW - i * DAY) for i in range(5)]


def test_cli_count_cap_prunes_oldest_over_cap(tmp_path):
    """(a) A low --max-task-dirs over an N>cap archive removes the oldest
    (N-cap) dirs from disk, prints a LOUD log to stderr, exits 0, and emits a
    JSON report with removed>0."""
    root = tmp_path / "agent-transcripts"
    _make_cli_archive(root, _five_fresh_specs())

    result = _run_cli(
        "--root", str(root),
        "--now", str(NOW),
        "--max-task-dirs", "2",
        "--max-age-days", "0",  # disable the age axis — isolate the count cap
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    # cap=2 keeps the 2 newest (100, 101); prunes the 3 oldest (102, 103, 104).
    assert (root / "100").exists()
    assert (root / "101").exists()
    for name in ("102", "103", "104"):
        assert not (root / name).exists(), f"{name} should have been pruned"

    report = json.loads(result.stdout)
    assert report["removed"] == 3
    assert report["check"] is False
    # LOUD: the greppable prefix reaches stderr; real-run, not a dry-run.
    assert LOG_PREFIX in result.stderr
    assert "would prune" not in result.stderr


def test_cli_default_caps_keep_everything(tmp_path):
    """(b) With the default (large) caps, nothing is removed and exit is 0."""
    root = tmp_path / "agent-transcripts"
    _make_cli_archive(root, _five_fresh_specs())

    result = _run_cli("--root", str(root), "--now", str(NOW))

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    for i in range(5):
        assert (root / str(100 + i)).exists()
    report = json.loads(result.stdout)
    assert report["removed"] == 0


def test_cli_check_is_dry_run(tmp_path):
    """(c) --check over the same over-cap archive exits 0, deletes NOTHING,
    logs 'would prune', and reports check=true / removed=0."""
    root = tmp_path / "agent-transcripts"
    _make_cli_archive(root, _five_fresh_specs())

    result = _run_cli(
        "--root", str(root),
        "--now", str(NOW),
        "--max-task-dirs", "2",
        "--max-age-days", "0",
        "--check",
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    for i in range(5):
        assert (root / str(100 + i)).exists()  # dry-run deletes nothing
    report = json.loads(result.stdout)
    assert report["check"] is True
    assert report["removed"] == 0
    assert "would prune" in result.stderr


def test_cli_empty_root_is_noop(tmp_path):
    """(d) An existing-but-empty archive root is a no-op, exit 0."""
    root = tmp_path / "agent-transcripts"
    root.mkdir(parents=True)

    result = _run_cli("--root", str(root), "--now", str(NOW))

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    report = json.loads(result.stdout)
    assert report["removed"] == 0


def test_cli_absent_root_is_noop(tmp_path):
    """(e) An ABSENT archive root is a no-op, exit 0."""
    root = tmp_path / "does-not-exist"

    result = _run_cli("--root", str(root), "--now", str(NOW))

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    report = json.loads(result.stdout)
    assert report["removed"] == 0
