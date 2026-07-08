"""Tests for orchestrator.verify_cancel — pgid-file path/lifecycle, descendant walk, cancel_request.

Steps covered:
  1 (test)  pgid-file path & lifecycle
  2 (impl)  verify_cancel module created
  3 (test)  collect_descendants — pure BFS with ppid_map injection
  4 (impl)  read_ppid_map + collect_descendants
  5 (test)  cancel_request — injected spies, all contract cases
  6 (impl)  cancel_request
  7 (test)  start_own_process_group setsid + fallback
  8 (impl)  start_own_process_group
  13 (test) real-process capstone: setsid + start_new_session escape tree reaped
"""

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Step-1: pgid-file path & lifecycle
# ---------------------------------------------------------------------------


class TestPgidFilePath:
    """pgid_file resolves under <worktree_base>/.merge_verify_pgids/ and sanitizes request_id."""

    def test_normal_request_id_resolves_inside_pgid_dir(self, tmp_path: Path):
        from orchestrator.verify_cancel import PGID_DIR_NAME, pgid_file

        wt_base = tmp_path / 'wt'
        path = pgid_file(wt_base, 'abc-123')
        assert path.parent == wt_base / PGID_DIR_NAME
        assert path.name == 'abc-123'

    def test_path_traversal_sanitized(self, tmp_path: Path):
        """A request_id like '../escape' must NOT produce a path outside pgid_dir."""
        from orchestrator.verify_cancel import PGID_DIR_NAME, pgid_file

        wt_base = tmp_path / 'wt'
        path = pgid_file(wt_base, '../escape')
        pgid_dir = wt_base / PGID_DIR_NAME
        # Must stay inside the pgid dir
        assert path.parent == pgid_dir or path.is_relative_to(pgid_dir)
        # The final filename must not be 'escape' at the wrong level
        assert path.parent == pgid_dir

    def test_slash_in_request_id_sanitized(self, tmp_path: Path):
        """A request_id like 'a/b' must not create a sub-directory escape."""
        from orchestrator.verify_cancel import PGID_DIR_NAME, pgid_file

        wt_base = tmp_path / 'wt'
        path = pgid_file(wt_base, 'a/b')
        pgid_dir = wt_base / PGID_DIR_NAME
        assert path.parent == pgid_dir

    def test_uuid_style_request_id(self, tmp_path: Path):
        from orchestrator.verify_cancel import PGID_DIR_NAME, pgid_file

        wt_base = tmp_path / 'wt'
        rid = 'f47ac10b-58cc-4372-a567-0e02b2c3d479'
        path = pgid_file(wt_base, rid)
        assert path.parent == wt_base / PGID_DIR_NAME
        assert path.name == rid


class TestPgidFileLifecycle:
    """write_pgid_file creates parent dirs + writes pid; remove_pgid_file is idempotent."""

    def test_write_creates_dirs_and_readable(self, tmp_path: Path):
        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'my-req')
        assert not path.exists()
        write_pgid_file(path, 4242)
        assert path.read_text().strip() == '4242'

    def test_write_overwrites_existing(self, tmp_path: Path):
        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'my-req')
        write_pgid_file(path, 1111)
        write_pgid_file(path, 2222)
        assert path.read_text().strip() == '2222'

    def test_remove_absent_is_idempotent(self, tmp_path: Path):
        from orchestrator.verify_cancel import pgid_file, remove_pgid_file

        path = pgid_file(tmp_path / 'wt', 'nonexistent')
        # Must not raise even though the file does not exist
        remove_pgid_file(path)

    def test_remove_deletes_existing(self, tmp_path: Path):
        from orchestrator.verify_cancel import pgid_file, remove_pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'my-req')
        write_pgid_file(path, 9999)
        assert path.exists()
        remove_pgid_file(path)
        assert not path.exists()


# ---------------------------------------------------------------------------
# Step-3: collect_descendants — pure BFS over an injected ppid_map
# ---------------------------------------------------------------------------


class TestCollectDescendants:
    """Pure collect_descendants(root, ppid_map) with injected ppid maps."""

    def _cd(self, root, ppid_map):
        from orchestrator.verify_cancel import collect_descendants
        return collect_descendants(root, ppid_map)

    def test_simple_chain(self):
        # root=100 -> 200 -> 300
        ppid_map = {200: 100, 300: 200}
        assert self._cd(100, ppid_map) == {200, 300}

    def test_tree_with_sibling(self):
        # root=100 -> {200, 201}; 200 -> 300; unrelated 999->1
        ppid_map = {200: 100, 201: 100, 300: 200, 999: 1}
        result = self._cd(100, ppid_map)
        assert result == {200, 201, 300}
        assert 999 not in result
        assert 100 not in result  # root excluded

    def test_missing_root_returns_empty(self):
        ppid_map = {200: 100, 300: 200}
        # root=999 has no children in the map
        assert self._cd(999, ppid_map) == set()

    def test_cyclic_map_does_not_loop(self):
        # Synthetic cycle: 200->100, 100->200 (nonsensical but must terminate)
        ppid_map = {100: 200, 200: 100, 300: 100}
        # Should terminate and return descendants without infinite loop
        result = self._cd(100, ppid_map)
        assert isinstance(result, set)
        assert 100 not in result  # root never included

    def test_empty_map_returns_empty(self):
        assert self._cd(42, {}) == set()

    def test_root_not_in_result(self):
        ppid_map = {200: 100}
        result = self._cd(100, ppid_map)
        assert 100 not in result
        assert 200 in result


# ---------------------------------------------------------------------------
# Step-5: cancel_request — injected spies, all contract cases
# ---------------------------------------------------------------------------


class TestCancelRequest:
    """cancel_request contract with injected ppid_map_provider, kill, killpg spies."""

    def _run(self, path, ppid_map, kill_side_effects=None, killpg_raises=None):
        """Helper: run cancel_request with deterministic injections.

        *kill_side_effects*: dict mapping pid -> exception to raise (or None to succeed).
        *killpg_raises*: exception to raise from killpg (or None to succeed).
        """

        from orchestrator.verify_cancel import cancel_request

        kill_calls = []
        killpg_calls = []

        def _kill(pid, sig):
            kill_calls.append((pid, sig))
            if kill_side_effects and pid in kill_side_effects:
                exc = kill_side_effects[pid]
                if exc is not None:
                    raise exc

        def _killpg(pgid, sig):
            killpg_calls.append((pgid, sig))
            if killpg_raises is not None:
                raise killpg_raises

        rc = cancel_request(
            path,
            ppid_map_provider=lambda: ppid_map,
            kill=_kill,
            killpg=_killpg,
        )
        return rc, kill_calls, killpg_calls

    def test_no_file_returns_0_no_kill(self, tmp_path):
        """(a) No pgid file -> return 0, no kill calls."""
        from orchestrator.verify_cancel import pgid_file
        path = pgid_file(tmp_path / 'wt', 'req-a')
        rc, kill_calls, killpg_calls = self._run(path, {})
        assert rc == 0
        assert kill_calls == []
        assert killpg_calls == []

    def test_live_descendants_killed_file_removed(self, tmp_path):
        """(b) File with pgid + live descendants -> SIGKILL to all, killpg backstop, file removed, rc=0."""
        import signal

        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'req-b')
        write_pgid_file(path, 100)

        # ppid_map: 100 -> {200, 201}
        ppid_map = {200: 100, 201: 100}

        rc, kill_calls, killpg_calls = self._run(path, ppid_map)

        assert rc == 0
        killed_pids = {pid for pid, sig in kill_calls}
        assert {100, 200, 201} == killed_pids
        for _, sig in kill_calls:
            assert sig == signal.SIGKILL
        assert len(killpg_calls) == 1
        assert killpg_calls[0] == (100, signal.SIGKILL)
        assert not path.exists()

    def test_already_dead_targets_count_as_success(self, tmp_path):
        """(c) Every target raises ProcessLookupError -> rc=0, file removed."""
        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'req-c')
        write_pgid_file(path, 100)
        ppid_map = {200: 100}

        side_effects = {100: ProcessLookupError(), 200: ProcessLookupError()}
        rc, kill_calls, _ = self._run(path, ppid_map, kill_side_effects=side_effects)

        assert rc == 0
        assert not path.exists()

    def test_permission_error_returns_nonzero_file_retained(self, tmp_path):
        """(d) A live target raises PermissionError -> rc=1, file retained."""
        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'req-d')
        write_pgid_file(path, 100)
        ppid_map = {}

        side_effects = {100: PermissionError()}
        rc, kill_calls, _ = self._run(path, ppid_map, kill_side_effects=side_effects)

        assert rc == 1
        assert path.exists()  # file retained on failure

    def test_corrupt_file_removed_returns_0(self, tmp_path):
        """(e) Corrupt/empty file (non-int) -> remove file, return 0, no kills."""
        from orchestrator.verify_cancel import pgid_file

        path = pgid_file(tmp_path / 'wt', 'req-e')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('not-an-int')

        rc, kill_calls, killpg_calls = self._run(path, {})

        assert rc == 0
        assert kill_calls == []
        assert not path.exists()

    def test_killpg_oserror_suppressed(self, tmp_path):
        """killpg OSError is suppressed; rc still 0 if no PermissionError from kill."""
        from orchestrator.verify_cancel import pgid_file, write_pgid_file

        path = pgid_file(tmp_path / 'wt', 'req-f')
        write_pgid_file(path, 555)
        ppid_map = {}

        rc, _, killpg_calls = self._run(
            path, ppid_map, killpg_raises=OSError('group gone')
        )
        assert rc == 0
        assert not path.exists()
        assert len(killpg_calls) == 1


# ---------------------------------------------------------------------------
# Step-7: start_own_process_group — setsid + fallback
# ---------------------------------------------------------------------------


class TestStartOwnProcessGroup:
    """start_own_process_group: normal setsid path and OSError fallback."""

    def test_setsid_called_and_pgid_returned(self, monkeypatch):
        """Normal path: setsid() succeeds, getpgrp() returns the new group."""
        import os

        import orchestrator.verify_cancel as vc_mod

        setsid_calls = []

        def fake_setsid():
            setsid_calls.append(True)

        monkeypatch.setattr(os, 'setsid', fake_setsid)
        monkeypatch.setattr(os, 'getpgrp', lambda: 12345)

        result = vc_mod.start_own_process_group()

        assert len(setsid_calls) == 1
        assert result == 12345

    def test_oserror_falls_back_to_getpgrp(self, monkeypatch):
        """EPERM path: setsid() raises OSError, fallback returns os.getpgrp()."""
        import errno
        import os

        import orchestrator.verify_cancel as vc_mod

        def fake_setsid():
            raise OSError(errno.EPERM, 'Operation not permitted')

        monkeypatch.setattr(os, 'setsid', fake_setsid)
        monkeypatch.setattr(os, 'getpgrp', lambda: 99999)

        result = vc_mod.start_own_process_group()

        # Must not raise; must return the current group
        assert result == 99999


# ---------------------------------------------------------------------------
# Task 2306 step-5: acquire_merge_verify_flock / release_merge_verify_flock —
# bounded-wait fcntl.flock guarding the laptop persistent merge-verify worktree.
# ---------------------------------------------------------------------------


class TestMergeVerifyLockPath:
    """merge_verify_lock_path resolves to a fixed filename under worktree_base."""

    def test_fixed_path(self, tmp_path: Path):
        from orchestrator.verify_cancel import merge_verify_lock_path

        path = merge_verify_lock_path(tmp_path)
        assert path == tmp_path / '.merge_verify.lock'


class TestAcquireReleaseMergeVerifyFlock:
    """acquire_merge_verify_flock / release_merge_verify_flock against a real temp file."""

    def test_acquire_when_free_returns_fd(self, tmp_path: Path):
        from orchestrator.verify_cancel import (
            acquire_merge_verify_flock,
            release_merge_verify_flock,
        )

        path = tmp_path / '.merge_verify.lock'
        fd = acquire_merge_verify_flock(path, timeout_secs=1.0)
        assert isinstance(fd, int)
        release_merge_verify_flock(fd)

    def test_release_then_reacquire_succeeds(self, tmp_path: Path):
        """release_merge_verify_flock frees the lock so a subsequent acquire succeeds."""
        from orchestrator.verify_cancel import (
            acquire_merge_verify_flock,
            release_merge_verify_flock,
        )

        path = tmp_path / '.merge_verify.lock'
        fd1 = acquire_merge_verify_flock(path, timeout_secs=1.0)
        assert fd1 is not None
        release_merge_verify_flock(fd1)

        fd2 = acquire_merge_verify_flock(path, timeout_secs=1.0)
        assert isinstance(fd2, int)
        release_merge_verify_flock(fd2)

    def test_contention_returns_none_within_timeout(self, tmp_path: Path):
        """A separate fd already holding LOCK_EX on the same path -> bounded-wait timeout.

        flock conflicts across independent fds even within one process, so a
        second os.open + fcntl.flock(LOCK_EX) on the same path from this same
        test process is sufficient to simulate a concurrent holder.
        """
        import fcntl
        import os
        import time

        from orchestrator.verify_cancel import acquire_merge_verify_flock

        path = tmp_path / '.merge_verify.lock'
        holder_fd = os.open(path, os.O_RDWR | os.O_CREAT)
        fcntl.flock(holder_fd, fcntl.LOCK_EX)
        try:
            start = time.monotonic()
            fd = acquire_merge_verify_flock(path, timeout_secs=0.3)
            elapsed = time.monotonic() - start
            assert fd is None
            assert elapsed >= 0.25
        finally:
            os.close(holder_fd)


# ---------------------------------------------------------------------------
# Task 2306 step-7: LOCK_HOLDER_PGID_KEY + write/read/remove_lock_holder_pgid —
# fixed-key holder-pgid rendezvous.  A waiter cannot know the holder's
# per-dispatch --request-id, so this uses a request-id-independent fixed key
# built directly on pgid_file/write_pgid_file/remove_pgid_file.
# ---------------------------------------------------------------------------


class TestLockHolderPgidRendezvous:
    """write/read/remove_lock_holder_pgid built on the existing pgid-file helpers."""

    def test_key_is_stable_nonempty_string(self):
        from orchestrator.verify_cancel import LOCK_HOLDER_PGID_KEY

        assert isinstance(LOCK_HOLDER_PGID_KEY, str)
        assert LOCK_HOLDER_PGID_KEY != ''

    def test_write_lands_at_fixed_key_pgid_file(self, tmp_path: Path):
        from orchestrator.verify_cancel import (
            LOCK_HOLDER_PGID_KEY,
            pgid_file,
            write_lock_holder_pgid,
        )

        write_lock_holder_pgid(tmp_path, 5150)
        expected_path = pgid_file(tmp_path, LOCK_HOLDER_PGID_KEY)
        assert expected_path.exists()
        assert expected_path.read_text().strip() == '5150'

    def test_read_returns_written_value(self, tmp_path: Path):
        from orchestrator.verify_cancel import read_lock_holder_pgid, write_lock_holder_pgid

        write_lock_holder_pgid(tmp_path, 5150)
        assert read_lock_holder_pgid(tmp_path) == 5150

    def test_remove_deletes_file_and_read_returns_none(self, tmp_path: Path):
        from orchestrator.verify_cancel import (
            LOCK_HOLDER_PGID_KEY,
            pgid_file,
            read_lock_holder_pgid,
            remove_lock_holder_pgid,
            write_lock_holder_pgid,
        )

        write_lock_holder_pgid(tmp_path, 5150)
        remove_lock_holder_pgid(tmp_path)
        assert not pgid_file(tmp_path, LOCK_HOLDER_PGID_KEY).exists()
        assert read_lock_holder_pgid(tmp_path) is None

    def test_read_returns_none_when_absent(self, tmp_path: Path):
        from orchestrator.verify_cancel import read_lock_holder_pgid

        assert read_lock_holder_pgid(tmp_path) is None

    def test_read_returns_none_when_corrupt(self, tmp_path: Path):
        from orchestrator.verify_cancel import (
            LOCK_HOLDER_PGID_KEY,
            pgid_file,
            read_lock_holder_pgid,
        )

        path = pgid_file(tmp_path, LOCK_HOLDER_PGID_KEY)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('not-an-int')
        assert read_lock_holder_pgid(tmp_path) is None


# ---------------------------------------------------------------------------
# Step-13 (part 1): real-process capstone — setsid + start_new_session escapes
# ---------------------------------------------------------------------------

# Script that runs as the root process:
#   1. os.setsid() to join a new session
#   2. Write its pgid (== its pid after setsid) to argv[1]
#   3. Spawn a child with start_new_session=True (escapes root's group)
#   4. The child spawns a grandchild sleeper also with start_new_session=True
#   5. Both root and child sleep 300s (simulating a long build)
_ESCAPE_TREE_SCRIPT = '''\
import os, sys, subprocess, time

pgid_file = sys.argv[1]
os.setsid()
pgid = os.getpgrp()  # == os.getpid() after setsid

with open(pgid_file, 'w') as f:
    f.write(str(pgid))

# Spawn child with start_new_session=True (escapes our process group)
child_script = """
import subprocess, time, sys
# Grandchild sleeper — also start_new_session=True (another escape)
gc = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(300)'],
    start_new_session=True,
)
# Write grandchild pid so parent can track it
with open(sys.argv[1], 'w') as f:
    f.write(str(gc.pid))
time.sleep(300)
"""
child_pid_file = pgid_file + '.child'
child = subprocess.Popen(
    [sys.executable, '-c', child_script, child_pid_file],
    start_new_session=True,
)
# Give child time to spawn the grandchild and write its pid file
time.sleep(300)
'''


def _is_running(pid: int) -> bool:
    """Return True if *pid* is still alive (os.kill(pid, 0) succeeds)."""
    import os
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _wait_for_file(path, timeout=10.0, interval=0.1):
    """Poll until *path* exists or *timeout* expires. Return True if found."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(interval)
    return False


@pytest.mark.timeout(30)
def test_cancel_request_reaps_start_new_session_escapes(tmp_path):
    """cancel_request reaps root AND start_new_session-escaped descendants.

    Spawns a real process tree:
      root (setsid) -> child (start_new_session) -> grandchild (start_new_session)

    All three are in DIFFERENT process groups (start_new_session creates a new
    group for each child), so a naive killpg(root_group) would only kill root.
    cancel_request must reap all three via the /proc PPID walk.
    """
    import sys
    import time

    from orchestrator.verify_cancel import cancel_request

    pgid_file_path = tmp_path / 'root.pgid'

    # Spawn root process
    root_proc = __import__('subprocess').Popen(
        [sys.executable, '-c', _ESCAPE_TREE_SCRIPT, str(pgid_file_path)],
    )

    # Wait for root to write its pgid file (means setsid + child spawned)
    assert _wait_for_file(pgid_file_path, timeout=10), (
        'root process did not write pgid file within 10s'
    )

    # Give the child a moment to spawn the grandchild
    time.sleep(0.5)

    # Take a PPID snapshot to find escaped descendants BEFORE killing
    from orchestrator.verify_cancel import collect_descendants, read_ppid_map
    ppid_map = read_ppid_map()
    pgid = int(pgid_file_path.read_text().strip())
    all_pids = collect_descendants(pgid, ppid_map) | {pgid}

    # Verify the tree is alive before cancelling
    assert _is_running(pgid), f'root pid {pgid} not running before cancel'

    # Run cancel_request (uses real /proc walk internally)
    rc = cancel_request(pgid_file_path)

    # Allow a brief window for processes to be reaped
    time.sleep(0.5)

    assert rc == 0, f'cancel_request returned {rc}, expected 0'
    assert not pgid_file_path.exists(), 'pgid file must be removed on success'

    # Reap root_proc zombie BEFORE checking alive status.
    # cancel_request kills root with SIGKILL, which makes it a zombie (state 'Z')
    # until its parent (this test process) calls wait().  os.kill(pid, 0) returns
    # success for zombies — they are still in /proc — so the alive check below
    # would incorrectly report root as alive if we don't wait first.
    root_proc.wait(timeout=5)

    # All tracked pids (including start_new_session escapes) must be dead
    still_alive = [pid for pid in all_pids if _is_running(pid)]
    assert still_alive == [], (
        f'These pids survived cancel_request (start_new_session escapes not reaped?): '
        f'{still_alive}'
    )


# ---------------------------------------------------------------------------
# Task 2308 γ step-1: run_stdin_watchdog — fd-0 trigger loop
#
# Connection-death heartbeat-watchdog (PRD: plans/laptop-warm-verify-flock-
# orphan-prd.md, Change B, §4 B1/B2 + §8.1). run_stdin_watchdog(read_fd,
# on_fire, *, heartbeat_timeout, select_fn, read_fn) fires on_fire() exactly
# once when either: (a) no heartbeat arrives within heartbeat_timeout
# (starvation — empty select() ready set), or (b) read_fd hits EOF (a read
# returns b'' — the writing end closed the channel).  A heartbeat (any
# non-empty read) resets the window and the loop continues.  select_fn/
# read_fn are injected so this is testable without a real pipe or wall-clock
# waits.
# ---------------------------------------------------------------------------


class TestRunStdinWatchdog:
    """run_stdin_watchdog(read_fd, on_fire, *, heartbeat_timeout, select_fn, read_fn)."""

    def test_fires_on_eof(self):
        """select_fn reports ready, read_fn returns b'' (EOF) -> on_fire fires once, loop returns."""
        from orchestrator.verify_cancel import run_stdin_watchdog

        select_calls = []
        read_calls = []
        fire_calls = []

        def fake_select(rlist, wlist, xlist, timeout):
            select_calls.append((rlist, wlist, xlist, timeout))
            return (rlist, [], [])  # fd ready

        def fake_read(fd, size):
            read_calls.append((fd, size))
            return b''  # EOF

        run_stdin_watchdog(
            7,
            lambda: fire_calls.append(True),
            heartbeat_timeout=5.0,
            select_fn=fake_select,
            read_fn=fake_read,
        )

        assert fire_calls == [True]
        assert len(select_calls) == 1
        assert select_calls[0] == ([7], [], [], 5.0)
        assert len(read_calls) == 1

    def test_fires_on_heartbeat_timeout(self):
        """select_fn reports an empty ready set (timeout) -> on_fire fires once, without reading."""
        from orchestrator.verify_cancel import run_stdin_watchdog

        select_calls = []
        read_calls = []
        fire_calls = []

        def fake_select(rlist, wlist, xlist, timeout):
            select_calls.append((rlist, wlist, xlist, timeout))
            return ([], [], [])  # timeout -- nothing ready

        def fake_read(fd, size):
            read_calls.append((fd, size))
            return b'\n'  # must never be reached

        run_stdin_watchdog(
            7,
            lambda: fire_calls.append(True),
            heartbeat_timeout=5.0,
            select_fn=fake_select,
            read_fn=fake_read,
        )

        assert fire_calls == [True]
        assert len(select_calls) == 1
        assert read_calls == []  # starvation fires without ever reading

    def test_survives_heartbeats_then_fires_on_eof(self):
        """K heartbeats (non-empty reads) reset the window and do not fire; the (K+1)th EOF does."""
        from orchestrator.verify_cancel import run_stdin_watchdog

        K = 3
        select_calls = []
        read_calls = []
        fire_calls = []

        def fake_select(rlist, wlist, xlist, timeout):
            select_calls.append((rlist, wlist, xlist, timeout))
            return (rlist, [], [])  # always "ready"

        def fake_read(fd, size):
            read_calls.append((fd, size))
            if len(read_calls) <= K:
                return b'\n'  # heartbeat -- resets the window, must not fire
            return b''  # EOF on the (K+1)th read

        run_stdin_watchdog(
            7,
            lambda: fire_calls.append(True),
            heartbeat_timeout=5.0,
            select_fn=fake_select,
            read_fn=fake_read,
        )

        # on_fire called exactly once, only after read_fn was called K+1 times
        assert fire_calls == [True]
        assert len(read_calls) == K + 1
        assert len(select_calls) == K + 1


# ---------------------------------------------------------------------------
# Task 2308 γ step-3: fire_watchdog_kill — SIGTERM descendants, grace, SIGKILL
# survivors, unconditional self-exit
# ---------------------------------------------------------------------------


class TestFireWatchdogKill:
    """fire_watchdog_kill(pgid, ...) — SIGTERM descendants, grace, SIGKILL survivors, exit."""

    def test_signals_only_descendants_grace_then_sigkill_then_exit(self):
        """(a)-(d): descendants only, grace between passes, SIGKILL survivors, exit is final."""
        import signal

        from orchestrator.verify_cancel import fire_watchdog_kill

        # ppid_map: P(100) -> A(200) -> B(300); unrelated 999 (ppid 1, not a descendant of P)
        ppid_map = {200: 100, 300: 200, 999: 1}

        events = []
        term_pids = []
        kill_pids = []
        killpg_calls = []

        def fake_kill(pid, sig):
            if sig == signal.SIGTERM:
                term_pids.append(pid)
                events.append(('term', pid))
            elif sig == signal.SIGKILL:
                kill_pids.append(pid)
                events.append(('kill', pid))

        def fake_killpg(pgid, sig):
            killpg_calls.append((pgid, sig))

        def fake_sleep(secs):
            events.append(('sleep', secs))

        def fake_exit(code):
            events.append(('exit', code))

        fire_watchdog_kill(
            100,
            grace_secs=5.0,
            ppid_map_provider=lambda: ppid_map,
            kill=fake_kill,
            killpg=fake_killpg,
            sleep=fake_sleep,
            exit_fn=fake_exit,
        )

        # (a) SIGTERM sent to exactly the descendants {200, 300} -- never root/self, never unrelated
        assert set(term_pids) == {200, 300}

        # (b) sleep(grace_secs) called between the SIGTERM pass and the SIGKILL pass
        sleep_idx = events.index(('sleep', 5.0))
        assert all(events.index(('term', pid)) < sleep_idx for pid in (200, 300))

        # (c) SIGKILL sent to surviving descendants {200, 300}
        assert set(kill_pids) == {200, 300}
        assert all(events.index(('kill', pid)) > sleep_idx for pid in (200, 300))

        # (d) exit_fn called once with a non-zero code, as the final action
        exit_events = [e for e in events if e[0] == 'exit']
        assert len(exit_events) == 1
        assert exit_events[0][1] != 0
        assert events[-1] == exit_events[0]

        # Design decision: the watchdog signals only descendants and never
        # killpg's its own group (it runs inside the target group, unlike
        # cancel_request) -- killpg is accepted for signature symmetry only.
        assert killpg_calls == []

    def test_dead_descendant_process_lookup_error_tolerated(self):
        """(e): a descendant already dead (ProcessLookupError) is tolerated; exit_fn still fires."""
        from orchestrator.verify_cancel import fire_watchdog_kill

        ppid_map = {200: 100}  # single descendant, already gone
        exit_calls = []

        def fake_kill(pid, sig):
            raise ProcessLookupError()

        def fake_killpg(pgid, sig):
            raise AssertionError('fire_watchdog_kill must not killpg its own group')

        def fake_sleep(secs):
            pass

        def fake_exit(code):
            exit_calls.append(code)

        # Must not raise despite kill() always raising ProcessLookupError.
        fire_watchdog_kill(
            100,
            grace_secs=0.0,
            ppid_map_provider=lambda: ppid_map,
            kill=fake_kill,
            killpg=fake_killpg,
            sleep=fake_sleep,
            exit_fn=fake_exit,
        )

        assert exit_calls == [1]


# ---------------------------------------------------------------------------
# Task 2308 γ step-5: start_stdin_watchdog — spawns a started daemon thread
# running run_stdin_watchdog, wired to fire on stdin EOF/starvation
# ---------------------------------------------------------------------------


class TestStartStdinWatchdog:
    """start_stdin_watchdog spawns a daemon thread running run_stdin_watchdog wired to fire."""

    def test_spawns_started_daemon_thread_wired_to_fire(self):
        """Returns a started daemon Thread; the injected fire callback runs exactly once."""
        import threading

        from orchestrator.verify_cancel import start_stdin_watchdog

        fire_calls = []

        def fake_select(rlist, wlist, xlist, timeout):
            return (rlist, [], [])  # immediately "ready"

        def fake_read(fd, size):
            return b''  # EOF

        thread = start_stdin_watchdog(
            12345,
            heartbeat_timeout=5.0,
            grace_secs=1.0,
            read_fd=0,
            select_fn=fake_select,
            read_fn=fake_read,
            fire=lambda: fire_calls.append(True),
        )

        # (a) return value is a Thread with .daemon True; join() only succeeds if started
        assert isinstance(thread, threading.Thread)
        assert thread.daemon is True
        thread.join(timeout=5.0)
        assert not thread.is_alive()

        # (b) the injected fire was invoked exactly once (loop wired to on_fire)
        assert fire_calls == [True]
