"""Unit tests for ``df_pytest_isolation``'s shared process-exit probes (task 5908).

:class:`~df_pytest_isolation.ProcState`, :func:`~df_pytest_isolation.read_proc_state`,
:func:`~df_pytest_isolation.wait_pids_exited` and
:func:`~df_pytest_isolation.wait_pid_gone` are the test tree's one definition of
"has this pid exited": a zombie counts as exited, and only a pid with no
``/proc`` entry at all reads as None.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from df_pytest_isolation import (
    ProcState,
    read_proc_state,
    wait_pid_gone,
    wait_pids_exited,
)


@contextlib.contextmanager
def _unreaped_zombie() -> Iterator[int]:
    """Yield the pid of a child that has terminated but is not yet reaped; reap it on exit.

    ``os.waitid(..., WNOWAIT)`` blocks until the child has exited without
    reaping it, so the zombie is produced by a condition wait, never a sleep.
    """
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
    try:
        child.kill()
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        try:
            os.kill(child.pid, 0)
        except ProcessLookupError:
            pytest.fail(
                'harness bug: an unreaped zombie must still answer signal 0, or '
                'this test is not reproducing the state a signal-0 probe counts '
                'as alive'
            )
        yield child.pid
    finally:
        child.kill()
        child.wait()


class TestProcessExitProbes:
    """The shared "has this pid exited" probes count a zombie as exited (task 5908)."""

    def test_wait_pid_gone_counts_an_unreaped_zombie_as_gone(self) -> None:
        """A terminated-but-unreaped child is gone, although signal 0 still answers."""
        with _unreaped_zombie() as pid:
            assert wait_pid_gone(pid, timeout=0) is True, (
                f'pid {pid} is a zombie: it has already exited and only awaits '
                "its parent's reap. Waiting for it to vanish measures how fast "
                'systemd --user reaps a reparented orphan, not the kill under test.'
            )

    def test_wait_pid_gone_reports_a_live_process_as_not_gone(self) -> None:
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
        try:
            assert wait_pid_gone(child.pid, timeout=0) is False
        finally:
            child.kill()
            child.wait()

    def test_wait_pid_gone_reports_a_reaped_pid_as_gone(self) -> None:
        child = subprocess.Popen([sys.executable, '-c', ''])
        child.wait()

        assert wait_pid_gone(child.pid, timeout=0) is True

    def test_read_proc_state_reports_an_unreaped_zombie_as_exited(self) -> None:
        """A terminated-but-unreaped child reads as EXITED, and as None once reaped."""
        with _unreaped_zombie() as pid:
            state = read_proc_state(pid)
            assert state == ProcState(state='Z', ppid=os.getpid())
            assert state is not None and state.exited is True
            assert wait_pids_exited({pid}, timeout=0) == {}

        assert read_proc_state(pid) is None

    def test_read_proc_state_reports_a_live_child_as_running(self) -> None:
        """A live child reads as RUNNING, parented to this process, and survives the wait."""
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
        try:
            state = read_proc_state(child.pid)
            assert state is not None
            assert state.ppid == os.getpid()
            assert state.exited is False

            survivors = wait_pids_exited({child.pid}, timeout=0)
            assert set(survivors) == {child.pid}
            assert survivors[child.pid].exited is False
        finally:
            child.kill()
            child.wait()

    def test_wait_pids_exited_decides_on_a_probe_taken_after_its_last_sleep(self) -> None:
        """A deadline crossed DURING the sleep still gets one fresh probe before the verdict.

        The scripted clock jumps past the deadline while the loop sleeps.  A loop
        that re-checks the deadline before re-probing returns the stale ``R``
        observation; the verdict must come from the second probe, which sees ``Z``.
        """
        reads: list[int] = []

        def read_state(pid: int) -> ProcState:
            reads.append(pid)
            return ProcState('R', 1) if len(reads) == 1 else ProcState('Z', 1)

        ticks = iter([0.0, 0.5])

        survivors = wait_pids_exited(
            {7}, timeout=1.0, interval=0,
            _read_state=read_state, _clock=lambda: next(ticks, 99.0),
        )

        assert survivors == {}
        assert reads == [7, 7]

    def test_wait_pids_exited_reports_each_survivor_with_its_last_observed_state(self) -> None:
        """timeout=0 is exactly one probe; a pid with no /proc entry is not a survivor."""
        reads: list[int] = []

        def sleeping(pid: int) -> ProcState:
            reads.append(pid)
            return ProcState('S', 4242)

        assert wait_pids_exited({7}, timeout=0, _read_state=sleeping) == {7: ProcState('S', 4242)}
        assert reads == [7]

        states = {7: ProcState('S', 4242), 8: None}
        assert wait_pids_exited({7, 8}, timeout=0, _read_state=states.__getitem__) == {
            7: ProcState('S', 4242)
        }

    @staticmethod
    def _fake_stat(root: Path, pid: int, text: str) -> Path:
        stat = root / str(pid) / 'stat'
        stat.parent.mkdir(parents=True)
        stat.write_text(text)
        return stat

    def test_read_proc_state_parses_after_the_last_paren_of_a_spoofed_comm(
        self, tmp_path: Path,
    ) -> None:
        pid = 31337
        self._fake_stat(tmp_path, pid, f'{pid} (x) Z 9) S 4242 {pid} 0 0')

        state = read_proc_state(pid, _proc_root=tmp_path)

        assert state == ProcState('S', 4242), (
            f'got {state!r}: a parser splitting on the FIRST ") " reads the '
            "comm's embedded Z as the state and reports a live process as exited."
        )
        assert state is not None and state.exited is False

    def test_read_proc_state_returns_none_for_an_absent_entry(self, tmp_path: Path) -> None:
        assert read_proc_state(31337, _proc_root=tmp_path) is None

    def test_read_proc_state_raises_on_a_malformed_stat(self, tmp_path: Path) -> None:
        pid = 31337
        self._fake_stat(tmp_path, pid, 'garbage-without-a-paren')

        with pytest.raises(ValueError, match=str(pid)):
            read_proc_state(pid, _proc_root=tmp_path)

    @pytest.mark.skipif(os.geteuid() == 0, reason='root bypasses file modes')
    def test_read_proc_state_raises_rather_than_reading_an_unreadable_entry_as_gone(
        self, tmp_path: Path,
    ) -> None:
        """An unreadable entry is a process that EXISTS; None would make it read as exited."""
        pid = 31337
        stat = self._fake_stat(tmp_path, pid, f'{pid} (sleep) S 4242 {pid} 0 0')
        stat.chmod(0o000)
        try:
            with pytest.raises(PermissionError):
                read_proc_state(pid, _proc_root=tmp_path)
        finally:
            stat.chmod(0o644)
