"""Tests for orchestrator.merge_lane.liveness: extracted startup liveness-margin
guard, verify-host-unreachable alarms, and persistent-worktree operational
guards (MQ-refactor task γ).

These tests encode the behavior-preserving contracts of the module split,
mirroring task β's test_merge_gates.py:

1. Module-existence — ``orchestrator.merge_lane.liveness`` exists and exports the
   full closure of moved symbols (liveness-margin types/functions,
   verify-host-unreachable alarm helpers, and persistent-worktree guards).
2. Logger-name — the module logs under the lane's one logger name,
   ``orchestrator.merge_lane`` (task 5037), so ``caplog`` assertions
   filtered to that name keep capturing the moved
   guards' WARNING-level messages.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar
from unittest.mock import patch

import pytest
from shared.proc_group import cpu_ticks_under_path

from orchestrator.merge_lane.liveness import VerifyProgressWatch, verify_tree_cpu_ticks


def test_merge_liveness_exports_moved_public_symbols() -> None:
    from orchestrator.merge_lane.liveness import (
        _MERGE_WORKER_LOOP_DIED_SENTINEL,
        _VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX,
        MergeLivenessAssessment,
        MergeLivenessConfigError,
        PersistentWorktreeConfigError,
        _acquire_warm_verify_worktree,
        _alarm_verify_host_unreachable,
        _clear_verify_host_unreachable,
        _safety_valve_due,
        _verify_host_unreachable_sentinel,
        check_merge_liveness_margin,
        enforce_merge_liveness_margin,
        enforce_persistent_worktree_serial_lane,
    )

    for name, obj in {
        'MergeLivenessAssessment': MergeLivenessAssessment,
        'check_merge_liveness_margin': check_merge_liveness_margin,
        'MergeLivenessConfigError': MergeLivenessConfigError,
        'enforce_merge_liveness_margin': enforce_merge_liveness_margin,
        'PersistentWorktreeConfigError': PersistentWorktreeConfigError,
        '_safety_valve_due': _safety_valve_due,
        '_VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX': _VERIFY_HOST_UNREACHABLE_SENTINEL_PREFIX,
        '_MERGE_WORKER_LOOP_DIED_SENTINEL': _MERGE_WORKER_LOOP_DIED_SENTINEL,
        '_verify_host_unreachable_sentinel': _verify_host_unreachable_sentinel,
        '_alarm_verify_host_unreachable': _alarm_verify_host_unreachable,
        '_clear_verify_host_unreachable': _clear_verify_host_unreachable,
        '_acquire_warm_verify_worktree': _acquire_warm_verify_worktree,
        'enforce_persistent_worktree_serial_lane': enforce_persistent_worktree_serial_lane,
    }.items():
        assert obj is not None, f'{name} must not be None'


class TestNewestContentMtime:
    """newest_content_mtime(root): pure content-mtime helper (task 2420 step-1).

    Signal for the merge-worktree no-progress budget (DEFECT 1, split from
    2357; extends #1728): the newest mtime among files/dirs STRICTLY BELOW
    *root*, excluding the root inode itself and any ``.git`` subtree.
    Heartbeat-immune by construction — the #1728 alpha owner-heartbeat
    (``_touch_owned_merge_worktrees``) only ``os.utime``s the ROOT inode, so
    a helper that never stats the root itself cannot be fooled by it.

    RED (pre-impl): ``orchestrator.merge_lane.liveness.newest_content_mtime`` does
    not exist yet.
    """

    def test_returns_newest_mtime_among_content_strictly_below_root(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_lane.liveness import newest_content_mtime

        older = tmp_path / 'older.txt'
        newer = tmp_path / 'newer.txt'
        older.write_text('a')
        newer.write_text('b')
        os.utime(older, (1000.0, 1000.0))
        os.utime(newer, (2000.0, 2000.0))

        result = newest_content_mtime(tmp_path)

        assert result == 2000.0, f'expected the newer child mtime, got {result!r}'

    def test_subdirectory_mtime_also_counts_toward_the_max(
        self, tmp_path: Path,
    ) -> None:
        """A sub-directory's own mtime (not just file mtimes) must be scanned."""
        from orchestrator.merge_lane.liveness import newest_content_mtime

        sub = tmp_path / 'subdir'
        sub.mkdir()
        f = sub / 'file.txt'
        f.write_text('a')
        os.utime(f, (100.0, 100.0))
        # Sub-directory mtime is newer than any file mtime.
        os.utime(sub, (5000.0, 5000.0))

        result = newest_content_mtime(tmp_path)

        assert result == 5000.0, f'subdirectory mtime must count toward the max, got {result!r}'

    def test_root_heartbeat_touch_does_not_change_result(self, tmp_path: Path) -> None:
        """HEARTBEAT-IMMUNITY: os.utime(root) alone must not move the result.

        The #1728 alpha owner-heartbeat touches only the worktree ROOT
        inode every _HEARTBEAT_POLL_S seconds — this is exactly why the
        existing 3h root-mtime reaper never fires for a dead-verify-but-
        live-worker.  newest_content_mtime must never stat root itself.
        """
        from orchestrator.merge_lane.liveness import newest_content_mtime

        child = tmp_path / 'child.txt'
        child.write_text('x')
        os.utime(child, (1000.0, 1000.0))

        before = newest_content_mtime(tmp_path)
        assert before == 1000.0

        far_future = 4_102_444_800.0  # year 2100 — well past any real content mtime
        os.utime(tmp_path, (far_future, far_future))

        after = newest_content_mtime(tmp_path)
        assert after == before, (
            f'root-only utime (simulated alpha heartbeat) must not affect the '
            f'content-mtime result: before={before!r} after={after!r}'
        )

    def test_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        from orchestrator.merge_lane.liveness import newest_content_mtime

        assert newest_content_mtime(tmp_path) is None

    def test_returns_none_for_missing_path(self, tmp_path: Path) -> None:
        from orchestrator.merge_lane.liveness import newest_content_mtime

        missing = tmp_path / 'does-not-exist'
        assert newest_content_mtime(missing) is None

    def test_git_subtree_is_skipped(self, tmp_path: Path) -> None:
        from orchestrator.merge_lane.liveness import newest_content_mtime

        git_dir = tmp_path / '.git'
        git_dir.mkdir()
        git_file = git_dir / 'HEAD'
        git_file.write_text('ref: refs/heads/main')
        os.utime(git_file, (9999.0, 9999.0))

        content = tmp_path / 'target.txt'
        content.write_text('y')
        os.utime(content, (500.0, 500.0))

        result = newest_content_mtime(tmp_path)

        assert result == 500.0, f'.git content must be excluded from the scan, got {result!r}'

    def test_unreadable_or_vanished_entry_does_not_raise(self, tmp_path: Path) -> None:
        """Best-effort: a broken symlink (stat-raises OSError) is skipped, not fatal."""
        from orchestrator.merge_lane.liveness import newest_content_mtime

        good = tmp_path / 'good.txt'
        good.write_text('z')
        os.utime(good, (100.0, 100.0))

        broken_link = tmp_path / 'broken-symlink'
        broken_link.symlink_to(tmp_path / 'does-not-exist-target')

        result = newest_content_mtime(tmp_path)

        assert result == 100.0, f'a broken symlink must be skipped, not raised; got {result!r}'


def test_merge_liveness_logger_name_is_merge_lane() -> None:
    """The liveness module logs under the lane's one name, 'orchestrator.merge_lane'.

    Every merge-lane module shares it (task 5037), so a
    ``caplog.at_level(..., logger='orchestrator.merge_lane')`` filter
    captures this module's records too.
    """
    import orchestrator.merge_lane.liveness as merge_liveness

    assert merge_liveness.logger.name == 'orchestrator.merge_lane'



class TestEngineConstantsAreReadAtCallTime:
    """The liveness guards read their engine constants when CALLED, not when defined.

    Each test patches one constant on ``orchestrator.merge_lane.liveness``, the
    module that defines it and reads it, and calls the guard with the
    corresponding argument omitted. A def-time default would have frozen the
    original value and ignored the patch (see the module docstring of
    ``orchestrator/src/orchestrator/merge_lane/liveness.py``).
    """

    def test_check_merge_liveness_margin_defaults_liveness_secs_at_call_time(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.config import OrchestratorConfig
        from orchestrator.merge_lane.liveness import check_merge_liveness_margin

        cfg = OrchestratorConfig(project_root=tmp_path)
        with patch('orchestrator.merge_lane.liveness.INFLIGHT_MERGE_WORKTREE_LIVENESS_SECS', 600.0):
            result = check_merge_liveness_margin(cfg)

        assert result.liveness_secs == 600.0, result

    def test_check_merge_liveness_margin_reads_heartbeat_poll_at_call_time(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.config import OrchestratorConfig
        from orchestrator.merge_lane.liveness import (
            TOUCH_MISS_TOLERANCE,
            check_merge_liveness_margin,
        )

        cfg = OrchestratorConfig(project_root=tmp_path)
        with patch('orchestrator.merge_lane.liveness._HEARTBEAT_POLL_S', 1000.0):
            result = check_merge_liveness_margin(cfg, liveness_secs=10800.0)

        assert result.worst_case_secs == 1000.0 * TOUCH_MISS_TOLERANCE, result
        assert result.safe is False, result

    def test_enforce_persistent_worktree_serial_lane_defaults_merge_ahead_bound_at_call_time(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.config import GitConfig, OrchestratorConfig
        from orchestrator.merge_lane.liveness import (
            PersistentWorktreeConfigError,
            enforce_persistent_worktree_serial_lane,
        )

        cfg = OrchestratorConfig(
            project_root=tmp_path,
            git=GitConfig(persistent_merge_worktree=True),
        )
        # merge_ahead_bound omitted; num_hosts=1, so per-host ceil(5/1)=5 > 1 raises.
        with (
            patch('orchestrator.merge_lane.liveness._MERGE_AHEAD_BOUND', 5),
            pytest.raises(PersistentWorktreeConfigError),
        ):
            enforce_persistent_worktree_serial_lane(cfg)


_T = TypeVar('_T')
_PROBE_SECS = 30.0


class _Probe(Generic[_T]):
    """A zero-argument progress probe returning a fixed reading, counting its calls."""

    def __init__(self, reading: _T) -> None:
        self.reading = reading
        self.calls = 0

    def __call__(self) -> _T:
        self.calls += 1
        return self.reading


def _observe(
    watch: VerifyProgressWatch,
    now: float,
    *,
    cpu: _Probe[int | None],
    mtime: _Probe[float | None],
    dispatch_live: bool = False,
) -> VerifyProgressWatch:
    return watch.observe(
        now, dispatch_live=dispatch_live, probe_every_secs=_PROBE_SECS,
        cpu_ticks=cpu, content_mtime=mtime,
    )


class TestVerifyProgressWatch:
    """The in-flight verify's progress evidence, unioned (task 5290).

    A live remote dispatch short-circuits; otherwise each throttled probe
    asks the verify tree's CPU fingerprint first and walks merge_wt's mtime
    only when that is silent.
    """

    def test_a_live_dispatch_is_progress_and_probes_nothing(self) -> None:
        cpu, mtime = _Probe[int | None](100), _Probe[float | None](50.0)
        watch = VerifyProgressWatch.start(0.0, cpu_ticks=100, content_mtime=50.0)

        watch = _observe(watch, 1000.0, cpu=cpu, mtime=mtime, dispatch_live=True)

        assert watch.no_progress_secs(1000.0) == 0.0
        assert watch.dispatch_seen_in_flight is True
        assert (cpu.calls, mtime.calls) == (0, 0)

    def test_the_throttle_probes_nothing_inside_the_interval(self) -> None:
        cpu, mtime = _Probe[int | None](100), _Probe[float | None](50.0)
        start = VerifyProgressWatch.start(0.0, cpu_ticks=100, content_mtime=50.0)

        early = _observe(start, _PROBE_SECS - 1.0, cpu=cpu, mtime=mtime)
        assert early == start
        assert (cpu.calls, mtime.calls) == (0, 0)

        probed = _observe(start, _PROBE_SECS, cpu=cpu, mtime=mtime)
        assert (cpu.calls, mtime.calls) == (1, 1)
        _observe(probed, 2 * _PROBE_SECS - 1.0, cpu=cpu, mtime=mtime)
        assert (cpu.calls, mtime.calls) == (1, 1)

    @pytest.mark.parametrize('ticks', [107, 93], ids=['increase', 'decrease-a-process-exited'])
    def test_a_cpu_change_is_progress_without_the_mtime_walk(self, ticks: int) -> None:
        cpu, mtime = _Probe[int | None](ticks), _Probe[float | None](50.0)
        watch = VerifyProgressWatch.start(0.0, cpu_ticks=100, content_mtime=50.0)

        watch = _observe(watch, _PROBE_SECS, cpu=cpu, mtime=mtime)

        assert watch.no_progress_secs(_PROBE_SECS) == 0.0
        assert (cpu.calls, mtime.calls) == (1, 0)

    @pytest.mark.parametrize('ticks', [100, None], ids=['unchanged', 'none-both-times'])
    def test_silent_cpu_falls_through_to_a_newer_mtime(self, ticks: int | None) -> None:
        cpu, mtime = _Probe[int | None](ticks), _Probe[float | None](60.0)
        watch = VerifyProgressWatch.start(0.0, cpu_ticks=ticks, content_mtime=50.0)

        watch = _observe(watch, _PROBE_SECS, cpu=cpu, mtime=mtime)

        assert watch.no_progress_secs(_PROBE_SECS) == 0.0
        assert (cpu.calls, mtime.calls) == (1, 1)

    @pytest.mark.parametrize('reading', [50.0, None], ids=['equal-mtime', 'no-mtime'])
    def test_a_wedged_tree_accrues_no_progress(self, reading: float | None) -> None:
        """Alive, zero CPU delta, nothing written: no_progress_secs keeps growing."""
        cpu, mtime = _Probe[int | None](100), _Probe[float | None](reading)
        watch = VerifyProgressWatch.start(0.0, cpu_ticks=100, content_mtime=50.0)

        for sweep in range(1, 5):
            watch = _observe(watch, sweep * _PROBE_SECS, cpu=cpu, mtime=mtime)

        assert watch.no_progress_secs(4 * _PROBE_SECS) == 4 * _PROBE_SECS
        assert (cpu.calls, mtime.calls) == (4, 4)

    def test_a_process_appearing_is_progress(self) -> None:
        cpu, mtime = _Probe[int | None](5), _Probe[float | None](None)
        watch = VerifyProgressWatch.start(0.0, cpu_ticks=None, content_mtime=None)

        watch = _observe(watch, _PROBE_SECS, cpu=cpu, mtime=mtime)

        assert watch.no_progress_secs(_PROBE_SECS) == 0.0
        assert mtime.calls == 0

    def test_observe_returns_a_new_value_and_leaves_the_original(self) -> None:
        cpu, mtime = _Probe[int | None](100), _Probe[float | None](50.0)
        start = VerifyProgressWatch.start(0.0, cpu_ticks=100, content_mtime=50.0)

        after = _observe(start, 1000.0, cpu=cpu, mtime=mtime, dispatch_live=True)

        assert after is not start
        assert start.dispatch_seen_in_flight is False
        assert start.no_progress_secs(1000.0) == 1000.0


def _ticks_rise(read: Callable[[], int | None], *, within_secs: float) -> bool:
    """Whether *read* returns an int that later strictly increases, within the deadline."""
    first: int | None = None
    deadline = time.monotonic() + within_secs
    while time.monotonic() < deadline:
        ticks = read()
        if ticks is not None and first is None:
            first = ticks
        elif ticks is not None and first is not None and ticks > first:
            return True
        time.sleep(0.05)
    return False


def _busy_child(cwd: Path, *, own_session: bool) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [sys.executable, '-c', 'while True: pass'], cwd=cwd, start_new_session=own_session,
    )


class TestVerifyTreeCpuTicks:
    """The verify tree's CPU fingerprint on the real /proc, bounded."""

    @pytest.mark.timeout(60)
    def test_a_busy_verify_subprocess_in_its_own_session_is_counted(self, tmp_path: Path) -> None:
        """Any child in its own session counts, a verify's or not: only the caller's pgrp is excluded."""
        root = tmp_path.resolve()
        child = _busy_child(root, own_session=True)
        try:
            assert _ticks_rise(lambda: verify_tree_cpu_ticks(root), within_secs=30.0)
        finally:
            child.kill()
            child.wait()

    @pytest.mark.timeout(60)
    def test_the_orchestrators_own_process_group_is_never_counted(self, tmp_path: Path) -> None:
        """Anti-masking (#1728): work in the caller's own process group cannot refresh it."""
        root = tmp_path.resolve()
        child = _busy_child(root, own_session=False)
        try:
            assert _ticks_rise(lambda: cpu_ticks_under_path(root), within_secs=30.0), (
                'precondition: the unfiltered fingerprint sees the busy child'
            )
            assert verify_tree_cpu_ticks(root) is None
        finally:
            child.kill()
            child.wait()
