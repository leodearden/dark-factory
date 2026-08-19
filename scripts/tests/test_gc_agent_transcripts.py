"""Tests for scripts/gc_agent_transcripts.py — the retention GC sweep over the
agent-transcript archive (task 2731, δ of
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
    # task dir "100": a single main-session transcript.
    _touch(root / "100" / "enc" / "sid1.jsonl", NOW - 10 * DAY)
    # task dir "200": two files at different mtimes -> newest wins.
    _touch(root / "200" / "enc" / "sid2.jsonl", NOW - 50 * DAY)
    _touch(root / "200" / "enc" / "sid3.jsonl", NOW - 5 * DAY)
    # task dir "300": nested subagent transcript only.
    _touch(root / "300" / "enc" / "sid4" / "subagents" / "agent-1.jsonl", NOW - 3 * DAY)
    # a stray non-directory file directly under root -> ignored.
    _touch(root / "loose.jsonl", NOW)

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
# producer hook, so a descendant transcript / task dir can vanish or become
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
    """A descendant transcript that becomes unstattable mid-scan is skipped
    LOUDLY; the remaining readable files still set the dir's age and sibling
    task dirs are still scanned — the sweep never raises."""
    caplog.set_level(logging.WARNING, logger="gc_agent_transcripts")
    root = tmp_path / "agent-transcripts"
    good = root / "100" / "enc" / "good.jsonl"
    vanished = root / "100" / "enc" / "vanished.jsonl"
    _touch(good, NOW - 20 * DAY)
    _touch(vanished, NOW - 1 * DAY)  # the newest file — but it fails to stat
    _touch(root / "200" / "enc" / "sib.jsonl", NOW - 8 * DAY)

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
    _touch(root / "200" / "enc" / "sib.jsonl", NOW - 8 * DAY)

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
        (d / "enc" / "x.jsonl").write_bytes(b"x")
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
    # max_task_dirs is a DERIVED bound, re-derived against the live archive by
    # test_max_task_dirs_is_derived_from_live_archive_rate below. Kept as exact
    # equality on both sides: this guard exists to catch one site moving
    # without the others, which an inequality would let through.
    assert gct.DEFAULT_MAX_TASK_DIRS == RetentionConfig().max_task_dirs == 50000


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
    """Build ``<root>/<name>/enc/session.jsonl`` for each ``(name, mtime)``,
    stamping the transcript mtime via ``os.utime`` so ``scan_task_dirs`` reads
    it as that task dir's retention age."""
    root.mkdir(parents=True, exist_ok=True)
    for name, mtime in specs:
        transcript = root / name / "enc" / "session.jsonl"
        transcript.parent.mkdir(parents=True)
        transcript.write_bytes(b"x")
        os.utime(transcript, (mtime, mtime))


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


# ---------------------------------------------------------------------------
# step-1 (task 3621): observed_daily_rate(task_dirs) — the DERIVED task-dir
# arrival-rate sampler that sizes the count cap.
#
# The cap must be a DERIVED bound that re-derives itself against real archive
# throughput, not a magic number that silently binds when the fleet speeds up.
# These tests pin the sampler against a SYNTHETIC archive with a per-day
# arrival pattern known by construction, fed through the production
# scan_task_dirs so the sampler consumes exactly the (path, mtime) shape the
# GC already prunes on.
# ---------------------------------------------------------------------------

# The UTC day bucket holding NOW, so a synthetic archive can be stamped into
# known consecutive day buckets exactly the way scan_task_dirs' mtimes fall.
NOW_DAY = int(NOW // DAY)


def _archive_with_daily_arrivals(root: Path, counts: list[int]) -> None:
    """Materialise an archive whose task dirs land in KNOWN consecutive UTC day
    buckets: ``counts[i]`` task dirs in the i-th day, OLDEST day first, with the
    last entry landing in NOW's own day bucket.

    Each dir holds one transcript stamped MID-day, so the bucket a dir falls in
    is unambiguous regardless of rounding. ``counts[0]`` and ``counts[-1]`` must
    be > 0: they define the observed span's boundary buckets, which
    observed_daily_rate drops as partial by construction.
    """
    first_day = NOW_DAY - (len(counts) - 1)
    for offset, count in enumerate(counts):
        mtime = (first_day + offset) * DAY + DAY / 2
        for n in range(count):
            _touch(root / f"d{offset}_{n}" / "enc" / "session.jsonl", mtime)


def _rate_for(root: Path):
    """Derive the observed rate the same way the live guard does: scan the
    archive with the production scanner, then sample its mtimes."""
    return gct.observed_daily_rate(gct.scan_task_dirs(root))


def test_observed_rate_peak_is_the_busiest_complete_day(tmp_path):
    """(a) peak_per_day IS the injected burst-day dir count, and the other
    fields report the sample honestly rather than only the buckets that
    happened to drive the peak."""
    root = tmp_path / "agent-transcripts"
    #        boundary  <------------ interior (8 days) ------------>  boundary
    counts = [3,        1, 1, 5, 1, 1, 1, 1, 1,                       2]
    _archive_with_daily_arrivals(root, counts)

    rate = _rate_for(root)

    assert rate is not None
    assert rate.peak_per_day == 5          # the injected burst day
    assert rate.complete_days == 8         # span minus the two partial ends
    assert rate.span_days == 10            # every observed bucket, ends included
    assert rate.sample_dirs == sum(counts) == 17   # the WHOLE sample, honestly
    assert rate.mean_per_day == pytest.approx(12 / 8)  # interior dirs / interior days


def test_observed_rate_counts_idle_interior_days_as_days(tmp_path):
    """(b) A zero-arrival interior day IS a day: it lowers the mean and does
    NOT shrink complete_days. The live archive has genuinely idle days, and
    dropping them would inflate the mean and overstate steady-state throughput.
    """
    root = tmp_path / "agent-transcripts"
    #        bnd  <-------- interior: 3 of the 8 days are idle -------->  bnd
    counts = [1,   2, 0, 0, 2, 2, 2, 0, 2,                                1]
    _archive_with_daily_arrivals(root, counts)

    rate = _rate_for(root)

    assert rate is not None
    # Idle days are still days. Were they dropped instead, this would read
    # complete_days == 5 and mean_per_day == 2.0.
    assert rate.complete_days == 8
    assert rate.mean_per_day == pytest.approx(10 / 8)
    assert rate.peak_per_day == 2
    assert rate.sample_dirs == sum(counts) == 12


def test_observed_rate_drops_partial_boundary_buckets(tmp_path):
    """(c) The FIRST and LAST day buckets are partial BY CONSTRUCTION (the
    sample starts and ends mid-day), so neither may set the peak — even when it
    holds more dirs than any complete interior day.

    Direction matters: a partial trailing bucket UNDERSTATES the rate (the live
    archive's trailing bucket held a handful of dirs against a ~90-dir peak), and
    silently weakening the derived bound is the exact failure the derived cap
    exists to prevent.
    """
    root = tmp_path / "agent-transcripts"
    #        BIG boundary  <---- interior: every day holds 1 ---->  BIG boundary
    counts = [12,           1, 1, 1, 1, 1, 1, 1, 1,                 9]
    _archive_with_daily_arrivals(root, counts)

    rate = _rate_for(root)

    assert rate is not None
    assert rate.peak_per_day == 1, (
        "a partial boundary bucket must never set the peak"
    )
    assert rate.complete_days == 8
    assert rate.span_days == 10
    # The dropped dirs are still reported in the raw sample size — dropping them
    # from the RATE is a methodology choice, not a reason to under-report.
    assert rate.sample_dirs == sum(counts) == 29


def test_observed_rate_is_none_for_absent_root(tmp_path):
    """(d) An absent archive root yields no sample — None, never a zero rate.

    A zero rate would satisfy any cap trivially, so absence MUST be
    distinguishable from "measured and fine".
    """
    assert _rate_for(tmp_path / "does-not-exist") is None


def test_observed_rate_is_none_for_empty_root(tmp_path):
    """(d) An existing-but-empty archive root likewise yields None."""
    root = tmp_path / "agent-transcripts"
    root.mkdir(parents=True)

    assert _rate_for(root) is None


def test_observed_rate_is_none_below_min_sample_days(tmp_path):
    """(d) A sample with fewer than MIN_RATE_SAMPLE_DAYS COMPLETE interior days
    is too sparse to read a peak from, and returns None rather than a weak
    number. Pinned either side of the threshold, driven off the constant."""
    min_days = gct.MIN_RATE_SAMPLE_DAYS

    # span = min_days + 1  ->  complete interior days = min_days - 1: too sparse.
    sparse = tmp_path / "sparse"
    _archive_with_daily_arrivals(sparse, [1] * (min_days + 1))
    assert _rate_for(sparse) is None

    # One more day of span clears the bar: complete interior days == min_days.
    enough = tmp_path / "enough"
    _archive_with_daily_arrivals(enough, [1] * (min_days + 2))
    rate = _rate_for(enough)
    assert rate is not None
    assert rate.complete_days == min_days


def test_observed_rate_is_none_for_single_day_sample(tmp_path):
    """(d) A one-day span has NO complete interior bucket at all (both ends are
    partial), so it must return None rather than a degenerate zero-day mean."""
    root = tmp_path / "agent-transcripts"
    _archive_with_daily_arrivals(root, [5])

    assert _rate_for(root) is None


# ---------------------------------------------------------------------------
# step-3 (task 3621): required_max_task_dirs(...) — the NON-VACUITY proof.
#
# The live derived-bound guard (below) is inert in a fresh checkout by design:
# with no archive to measure it skips. So the guarantee that the bound is a
# REAL check and not arithmetic-on-literals-that-can-never-fail has to be made
# here, host-independently: over a synthetic archive whose peak is known by
# construction, the exact comparison the live guard makes is shown to FAIL for
# a too-small cap and PASS for an adequate one.
# ---------------------------------------------------------------------------

def test_required_max_task_dirs_is_falsifiable_against_a_known_peak(tmp_path):
    """(a)+(b) The live guard's comparison CAN fail.

    Over an archive whose derived peak is 5/day by construction, a
    deliberately-small cap sits BELOW the requirement and an adequate one
    clears it — the same `cap >= required_max_task_dirs(peak, age, factor)`
    expression, evaluating both ways.
    """
    root = tmp_path / "agent-transcripts"
    #        bnd  <---- interior: 8 days, busiest holds 5 ---->  bnd
    _archive_with_daily_arrivals(root, [1, 1, 5, 1, 1, 1, 1, 1, 1, 1])

    rate = _rate_for(root)
    assert rate is not None
    assert rate.peak_per_day == 5  # known by construction

    required = gct.required_max_task_dirs(
        rate.peak_per_day, gct.DEFAULT_MAX_AGE_DAYS, gct.RETENTION_SAFETY_FACTOR
    )
    # 5/day x 90 days x factor — nothing here is a literal the test controls.
    assert required == 5 * gct.DEFAULT_MAX_AGE_DAYS * gct.RETENTION_SAFETY_FACTOR

    too_small = required - 1
    assert not (too_small >= required), (
        "the derived-bound comparison must be able to FAIL — a guard that "
        "cannot go red is measuring nothing"
    )
    assert required >= required  # and it clears when the cap is adequate
    assert 10 * required >= required


def test_required_max_task_dirs_strictly_increases_with_peak_rate(tmp_path):
    """(c) A throughput RISE genuinely raises the bar; it is not absorbed.

    This is the property that makes the cap re-derive rather than sit green
    forever — the live peak already moved 71 -> ~90/day between the PRD's
    measurement and this task's.
    """
    requirements = [
        gct.required_max_task_dirs(peak, gct.DEFAULT_MAX_AGE_DAYS, gct.RETENTION_SAFETY_FACTOR)
        for peak in (1, 2, 50, 90, 91, 200)
    ]
    assert requirements == sorted(requirements)
    assert len(set(requirements)) == len(requirements)  # STRICTLY increasing


def test_required_max_task_dirs_strictly_increases_with_safety_factor():
    """(c) A larger safety factor demands a larger cap, monotonically."""
    requirements = [
        gct.required_max_task_dirs(90, gct.DEFAULT_MAX_AGE_DAYS, factor)
        for factor in (1, 1.5, 2, 3, 5)
    ]
    assert requirements == sorted(requirements)
    assert len(set(requirements)) == len(requirements)  # STRICTLY increasing


def test_required_max_task_dirs_rounds_up():
    """(d) A fractional requirement rounds UP, never down into false headroom.

    Rounding 1.5 down to 1 would report the cap as adequate when it is half a
    dir short — small per-day, but it is the direction that hides truncation.
    """
    assert gct.required_max_task_dirs(1, 1, 1.5) == 2
    assert gct.required_max_task_dirs(2.5, 1, 1) == 3
    assert gct.required_max_task_dirs(1, 3, 1.1) == 4  # 3.3 -> 4
    # An exact integer is NOT inflated by the rounding.
    assert gct.required_max_task_dirs(2, 3, 2) == 12


def test_retention_safety_factor_is_at_least_one():
    """(e) A factor below 1 would size the cap UNDER the plain 90-day
    projection of the observed rate — quietly re-admitting the very truncation
    the derived bound exists to prevent."""
    assert gct.RETENTION_SAFETY_FACTOR >= 1


# ---------------------------------------------------------------------------
# step-5 (task 3621): the count cap re-derived against the LIVE archive.
#
# Meaningful on a live host, inert in a fresh checkout: with no archive (or too
# sparse a one) it SKIPS rather than passing, so silence stays legible in the
# pytest output as "not measured" instead of masquerading as a green check.
# ---------------------------------------------------------------------------

def test_max_task_dirs_is_derived_from_live_archive_rate():
    """DERIVED BOUND: max_task_dirs must hold a FULL max_age_days window of the
    archive's real peak throughput, with headroom.

    The count cap prunes OLDEST-FIRST, so if it binds it truncates the 90-day
    retention window from the forensic end while the sweep still reports a
    90-day policy. This guard re-measures the live archive every run, so a
    fleet that speeds up trips the test instead of silently losing history.
    """
    root = gct.default_archive_root()
    if not root.is_dir():
        pytest.skip(
            f"no live archive at {root} — nothing to derive the cap from "
            "(host-independent falsifiability is covered by "
            "test_required_max_task_dirs_is_falsifiable_against_a_known_peak)"
        )

    scanned = gct.scan_task_dirs(root)
    rate = gct.observed_daily_rate(scanned)
    if rate is None:
        pytest.skip(
            f"live archive at {root} holds {len(scanned)} task dirs spanning "
            f"fewer than {gct.MIN_RATE_SAMPLE_DAYS} complete days — too sparse "
            "to derive a rate from"
        )

    # NON-DEGENERATE before it is consumed. A zero rate would make the bound
    # below satisfiable by ANY cap, leaving this guard permanently green while
    # measuring nothing — the exact vacuity trap it exists to avoid.
    assert rate.sample_dirs > 0, "degenerate sample: no task dirs measured"
    assert rate.peak_per_day > 0, "degenerate sample: zero peak arrival rate"

    required = gct.required_max_task_dirs(
        rate.peak_per_day, gct.DEFAULT_MAX_AGE_DAYS, gct.RETENTION_SAFETY_FACTOR
    )

    assert gct.DEFAULT_MAX_TASK_DIRS >= required, (
        "retention count cap is too small for the archive's MEASURED "
        "throughput: it will prune oldest-first and silently truncate the "
        f"{gct.DEFAULT_MAX_AGE_DAYS}-day retention window.\n"
        f"  archive root ..... {root}\n"
        f"  sample ........... {rate.sample_dirs} task dirs over "
        f"{rate.span_days} days ({rate.complete_days} complete)\n"
        f"  observed peak .... {rate.peak_per_day}/day "
        f"(mean {rate.mean_per_day:.1f}/day)\n"
        f"  safety factor .... {gct.RETENTION_SAFETY_FACTOR}\n"
        f"  REQUIRED cap ..... {required} = ceil({rate.peak_per_day}/day"
        f" x {gct.DEFAULT_MAX_AGE_DAYS} days x {gct.RETENTION_SAFETY_FACTOR})\n"
        f"  current cap ...... {gct.DEFAULT_MAX_TASK_DIRS}\n"
        "TO FIX: re-derive the cap from the numbers above (ruling: "
        "plans/transcript-preservation-seam-prd.md D8) and raise ALL FOUR "
        "lock-step sites in ONE commit, or any single commit is red:\n"
        "  1. orchestrator/src/orchestrator/config.py  RetentionConfig.max_task_dirs\n"
        "  2. scripts/gc_agent_transcripts.py  DEFAULT_MAX_TASK_DIRS (+ the "
        "module docstring's --max-task-dirs usage example)\n"
        "  3. orchestrator/tests/test_transcript_archive_config.py  defaults test\n"
        "  4. orchestrator/tests/test_transcript_archive_config.py  whole-submodel "
        "reload rider assertion\n"
        "  ...plus this file's test_default_constants_match_orchestrator_config "
        "drift guard."
    )
