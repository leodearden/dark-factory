"""Tests for the wait helpers in _verify_merge_cli_fixtures.py."""

import subprocess
import sys

import pytest
from _verify_merge_cli_fixtures import _wait_for_pgid_file_or_report_crash, _wait_pgid_gone

# ---------------------------------------------------------------------------
# Task 2733 step-1/2 — _wait_pgid_gone: bounded poll-for-ESRCH helper (leg c)
# ---------------------------------------------------------------------------
#
# De-flakes the pgid-liveness assertion in
# test_cli_verify_merge.py::test_verify_merge_cancel_end_to_end: SIGKILL
# delivery + reaping is asynchronous, so a one-shot os.killpg(pgid, 0)
# immediately after child.wait() can observe a not-yet-reaped group member.
# _wait_pgid_gone asserts the real contract (eventual group death) via a
# bounded poll instead.


def test_wait_pgid_gone_returns_for_dead_group():
    """A pgid with no live processes (ESRCH) returns without raising.

    2 ** 22 is above /proc/sys/kernel/pid_max on most systems, so no
    process/group can plausibly exist at this pgid: os.killpg(pgid, 0)
    inside _wait_pgid_gone raises ProcessLookupError/ESRCH on the first poll
    and the helper returns cleanly.
    """
    NONEXISTENT_PID = 2 ** 22

    _wait_pgid_gone(NONEXISTENT_PID, timeout=1.0, interval=0.05)


def test_wait_pgid_gone_raises_for_live_group():
    """A still-live process group causes _wait_pgid_gone to raise after the deadline."""
    child = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(30)'],
        start_new_session=True,
    )
    try:
        with pytest.raises(AssertionError, match='still alive'):
            _wait_pgid_gone(child.pid, timeout=1.0, interval=0.05)
    finally:
        child.kill()
        child.wait(timeout=5)


# ---------------------------------------------------------------------------
# Task 4279 — _wait_for_pgid_file_or_report_crash: crash vs. timeout unit tests
# ---------------------------------------------------------------------------
#
# test_cli_verify_merge.py::test_verify_merge_cancel_end_to_end only ever
# exercises this helper's happy path (the pgid file appears on the first
# poll), leaving the exit_code != 0 crash branch and the still-running
# deadline branch dead in CI. These mirror the _wait_pgid_gone pair above,
# using cheap synthetic children instead of a real verify-merge subprocess.


def test_wait_for_pgid_file_or_report_crash_reports_crash_not_timeout(tmp_path):
    """A child that exits nonzero before writing the pgid file fails
    immediately with a crash-shaped message, not the generic timeout one.
    """
    never_created = tmp_path / 'does-not-exist.pgid'
    child = subprocess.Popen(
        [sys.executable, '-c', 'import sys; sys.exit(3)'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with pytest.raises(pytest.fail.Exception, match='crash, not a timeout'):
            _wait_for_pgid_file_or_report_crash(never_created, child, timeout=5, interval=0.05)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_wait_for_pgid_file_or_report_crash_reports_timeout_for_live_child(tmp_path):
    """A child still running at the deadline is reported as a genuine
    timeout, not a crash -- even though it has not written the pgid file.
    """
    never_created = tmp_path / 'does-not-exist.pgid'
    child = subprocess.Popen(
        ['sleep', '30'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        with pytest.raises(pytest.fail.Exception, match='did not write pgid file'):
            _wait_for_pgid_file_or_report_crash(never_created, child, timeout=0.5, interval=0.05)
    finally:
        child.kill()
        child.wait(timeout=5)
