"""Tests for orchestrator.merge_liveness: extracted startup liveness-margin
guard, verify-host-unreachable alarms, and persistent-worktree operational
guards (MQ-refactor task γ).

These tests encode the behavior-preserving contracts of the module split,
mirroring task β's test_merge_gates.py:

1. Module-existence — ``orchestrator.merge_liveness`` exists and exports the
   full closure of moved symbols (liveness-margin types/functions,
   verify-host-unreachable alarm helpers, and persistent-worktree guards).
2. Logger-name — the module logs under the ``orchestrator.merge_queue``
   logger name (not ``orchestrator.merge_liveness``) so existing ``caplog``
   assertions filtered to the merge_queue logger keep capturing the moved
   guards' WARNING-level messages.
"""

from __future__ import annotations

import os
from pathlib import Path


def test_merge_liveness_exports_moved_public_symbols() -> None:
    from orchestrator.merge_liveness import (
        _MERGE_WORKER_LOOP_DIED_SENTINEL,
        _VERIFY_HOST_RECOVERED_SENTINEL_PREFIX,
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
        '_VERIFY_HOST_RECOVERED_SENTINEL_PREFIX': _VERIFY_HOST_RECOVERED_SENTINEL_PREFIX,
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

    RED (pre-impl): ``orchestrator.merge_liveness.newest_content_mtime`` does
    not exist yet.
    """

    def test_returns_newest_mtime_among_content_strictly_below_root(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.merge_liveness import newest_content_mtime

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
        from orchestrator.merge_liveness import newest_content_mtime

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
        from orchestrator.merge_liveness import newest_content_mtime

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
        from orchestrator.merge_liveness import newest_content_mtime

        assert newest_content_mtime(tmp_path) is None

    def test_returns_none_for_missing_path(self, tmp_path: Path) -> None:
        from orchestrator.merge_liveness import newest_content_mtime

        missing = tmp_path / 'does-not-exist'
        assert newest_content_mtime(missing) is None

    def test_git_subtree_is_skipped(self, tmp_path: Path) -> None:
        from orchestrator.merge_liveness import newest_content_mtime

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
        from orchestrator.merge_liveness import newest_content_mtime

        good = tmp_path / 'good.txt'
        good.write_text('z')
        os.utime(good, (100.0, 100.0))

        broken_link = tmp_path / 'broken-symlink'
        broken_link.symlink_to(tmp_path / 'does-not-exist-target')

        result = newest_content_mtime(tmp_path)

        assert result == 100.0, f'a broken symlink must be skipped, not raised; got {result!r}'


def test_merge_liveness_logger_name_is_merge_queue() -> None:
    """merge_liveness emits under the 'orchestrator.merge_queue' logger name.

    RED (pre-module): ``orchestrator.merge_liveness`` does not exist yet.

    Required so existing ``caplog.at_level(..., logger='orchestrator.merge_queue')``
    assertions in test_merge_queue.py keep capturing the moved guards'
    WARNING-level messages after relocation.
    """
    import orchestrator.merge_liveness as merge_liveness

    assert merge_liveness.logger.name == 'orchestrator.merge_queue'

