"""Tests for shared.verify_admission — flock N-slot task-verify semaphore +
role nice tiers.

PRD ``plans/verify-oversubscription-control-prd.md`` task T1 (foundation
substrate; T2 wires this into ``orchestrator/verify.py``). See that PRD's
§Contract for the exact seam signatures and the C-* contract clauses
(C-merge-priority, C-untimed-acquire, C-fail-open, C-no-FD-inheritance)
referenced throughout this suite.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time


class TestNicePrefix:
    def test_merge_role(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('merge') == ['nice', '-n', '5']

    def test_task_role(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('task') == ['nice', '-n', '15', 'ionice', '-c2', '-n7']

    def test_background_role(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('background') == ['nice', '-n', '19', 'ionice', '-c3']

    def test_offline_role_returns_empty(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('offline') == []

    def test_unknown_role_returns_empty(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('some-unknown-role') == []

    def test_empty_string_role_returns_empty(self):
        from shared.verify_admission import nice_prefix

        assert nice_prefix('') == []

    def test_returns_fresh_list_not_shared_mutable_state(self):
        """Mutating a returned list must not leak into subsequent calls."""
        from shared.verify_admission import nice_prefix

        result = nice_prefix('task')
        result.append('MUTATED')

        result_again = nice_prefix('task')

        assert result_again == ['nice', '-n', '15', 'ionice', '-c2', '-n7']
        assert 'MUTATED' not in result_again


class TestModuleExports:
    def test_all_is_exactly_the_public_seam(self):
        import shared.verify_admission as va

        assert set(va.__all__) == {'acquire_task_slot', 'nice_prefix'}


class TestAcquireInProcess:
    """In-process flock semaphore behavior (wait=False unless noted).

    flock locks bind to the open FILE DESCRIPTION, not the process, so two
    separate os.open() + flock() calls contend even within a single process
    — mutual exclusion and N-slot counting are fully testable in-process.
    """

    def test_merge_role_never_holds_and_does_not_block_when_slot_saturated(self, tmp_path):
        """(a) merge is the anti-livelock bypass: held=False, and it must not
        block even when asked to wait against an already-saturated N=1 slot.
        """
        from shared.verify_admission import acquire_task_slot

        with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=False) as task_held:
            assert task_held is True
            with acquire_task_slot('merge', slots_dir=tmp_path, n=1, wait=True) as merge_held:
                assert merge_held is False

    def test_offline_role_never_holds(self, tmp_path):
        """(b) offline is un-gated like merge — always held=False."""
        from shared.verify_admission import acquire_task_slot

        with acquire_task_slot('offline', slots_dir=tmp_path, n=1, wait=False) as held:
            assert held is False

    def test_task_role_holds_when_slot_free(self, tmp_path):
        """(c) task acquires when a slot is free."""
        from shared.verify_admission import acquire_task_slot

        with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=False) as held:
            assert held is True

    def test_background_role_holds_when_slot_free(self, tmp_path):
        """(c) background acquires when a slot is free."""
        from shared.verify_admission import acquire_task_slot

        with acquire_task_slot('background', slots_dir=tmp_path, n=1, wait=False) as held:
            assert held is True

    def test_n1_nested_acquire_blocked_while_outer_holds(self, tmp_path):
        """(d) N=1: a nested wait=False acquire finds no free slot while the
        outer acquire still holds it.
        """
        from shared.verify_admission import acquire_task_slot

        with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=False) as h1:
            assert h1 is True
            with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=False) as h2:
                assert h2 is False

    def test_release_frees_the_slot_for_a_fresh_acquire(self, tmp_path):
        """(e) once the outer `with` exits, the slot is free again."""
        from shared.verify_admission import acquire_task_slot

        with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=False) as h1:
            assert h1 is True

        with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=False) as h2:
            assert h2 is True

    def test_n2_allows_two_concurrent_holders_then_saturates(self, tmp_path):
        """(f) N=2: two nested acquires both succeed; a 3rd finds nothing free."""
        from shared.verify_admission import acquire_task_slot

        with acquire_task_slot('task', slots_dir=tmp_path, n=2, wait=False) as h1:
            assert h1 is True
            with acquire_task_slot('task', slots_dir=tmp_path, n=2, wait=False) as h2:
                assert h2 is True
                with acquire_task_slot('task', slots_dir=tmp_path, n=2, wait=False) as h3:
                    assert h3 is False


def _wait_for_marker(marker_path, timeout: float = 5.0) -> bool:
    """Poll for *marker_path* to appear; return False on timeout.

    Test-side synchronization only (bounded, unlike the production module's
    untimed wait) so a child process that fails to start doesn't hang the
    suite forever.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker_path.exists():
            return True
        time.sleep(0.02)
    return False


# Inline stdlib child: flocks slots_dir/slot-1 (matching the N=1 slot-file
# naming the module contract specifies), signals readiness via a marker
# file, then sleeps so the parent can observe it holding the slot.
_SELF_HEAL_CHILD_SRC = (
    'import fcntl, os, sys, time\n'
    "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o644)\n"
    'fcntl.flock(fd, fcntl.LOCK_EX)\n'
    "open(sys.argv[2], 'w').write('ready')\n"
    'time.sleep(60)\n'
)


class TestRealProcessSignals:
    """Signals that need actual subprocesses: flock self-heal is a kernel
    behavior triggered by process death, and FD non-inheritance is only
    observable across a real exec boundary.
    """

    def test_self_heal_slot_released_when_holder_is_sigkilled(self, tmp_path):
        """(g) A holder that is SIGKILLed (no clean shutdown, no atexit)
        still releases its slot — the kernel frees the flock on process
        death. No canary/daemon is involved; this is self-heal via the OS.
        """
        from shared.verify_admission import acquire_task_slot

        marker = tmp_path / 'ready.marker'
        slot_path = tmp_path / 'slot-1'
        proc = subprocess.Popen(
            [sys.executable, '-c', _SELF_HEAL_CHILD_SRC, str(slot_path), str(marker)],
        )
        try:
            assert _wait_for_marker(marker), 'child never signaled readiness'

            with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=False) as held_while_alive:
                assert held_while_alive is False

            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
            proc = None

            with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=False) as held_after_death:
                assert held_after_death is True
        finally:
            if proc is not None:
                proc.kill()
                proc.wait(timeout=5)

    def test_fd_non_inheritance_close_fds_false_child_does_not_pin_slot(self, tmp_path):
        """(h) A close_fds=False descendant spawned WHILE a slot is held must
        not pin that slot after the holder releases — the slot FD is marked
        non-inheritable (C-no-FD-inheritance), so it does not survive the
        child's fork+exec regardless of the parent's close_fds setting.

        The child MUST be spawned after the slot fd is open in this process
        (i.e. inside the `with`, before release) — a child can only ever
        inherit fds that exist in the parent's table at its fork time, so
        spawning it earlier could never observe a leak either way and would
        make this test pass vacuously even without
        ``os.set_inheritable(fd, False)``.
        """
        from shared.verify_admission import acquire_task_slot

        sleeper = None
        try:
            with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=False) as held:
                assert held is True
                # Spawned while the slot fd is open in this process, so a
                # fork would copy it into the child's fd table if it were
                # inheritable; close_fds=False disables Python's own
                # fd-closing safety net, isolating the assertion to the
                # module's explicit os.set_inheritable(fd, False) call.
                sleeper = subprocess.Popen(
                    [sys.executable, '-c', 'import time; time.sleep(60)'],
                    close_fds=False,
                )
                # Give the child a moment to complete its exec (any
                # CLOEXEC-marked fd is closed at exec time, not at fork time).
                time.sleep(0.2)

            # The outer `with` released the slot. If the slot fd had leaked
            # into the still-running sleeper, this fresh acquire would fail.
            with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=False) as held_after:
                assert held_after is True
        finally:
            if sleeper is not None:
                sleeper.kill()
                sleeper.wait(timeout=5)


class TestBlockingWait:
    """N=1 blocking handoff via real thread concurrency.

    The untimed poll loop's ``time.sleep`` + syscalls release the GIL, so a
    worker thread genuinely blocks behind the main thread's held slot until
    it releases -- this is the wait=True serialize signal.
    """

    def test_second_acquirer_blocks_until_first_releases(self, tmp_path):
        from shared.verify_admission import acquire_task_slot

        acquired_event = threading.Event()

        def worker():
            with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=True) as held:
                if held:
                    acquired_event.set()

        with acquire_task_slot('task', slots_dir=tmp_path, n=1, wait=False) as main_held:
            assert main_held is True

            t = threading.Thread(target=worker)
            t.start()

            # Worker should still be blocked while main holds the only slot.
            assert not acquired_event.wait(timeout=0.3)

        # Main released -> the worker's poll loop should now pick up the slot.
        t.join(timeout=5)
        assert not t.is_alive()
        assert acquired_event.is_set()


class TestFailOpen:
    """C-fail-open: missing dir / n<1 / any OSError -> held=False, never
    blocks, never raises, never creates the directory.
    """

    def test_missing_slots_dir_yields_false_immediately_and_does_not_create_it(self, tmp_path):
        """(a) A missing slots_dir fails open even with wait=True (no block,
        no mkdir).
        """
        from shared.verify_admission import acquire_task_slot

        missing = tmp_path / 'does-not-exist'
        assert not missing.exists()

        with acquire_task_slot('task', slots_dir=missing, n=1, wait=True) as held:
            assert held is False

        assert not missing.exists()

    def test_n_zero_yields_false(self, tmp_path):
        """(b) n=0 -> no slots to acquire -> held=False."""
        from shared.verify_admission import acquire_task_slot

        with acquire_task_slot('task', slots_dir=tmp_path, n=0, wait=False) as held:
            assert held is False

    def test_n_negative_yields_false(self, tmp_path):
        """(b) n<0 -> held=False."""
        from shared.verify_admission import acquire_task_slot

        with acquire_task_slot('task', slots_dir=tmp_path, n=-1, wait=False) as held:
            assert held is False

    def test_slots_dir_is_a_regular_file_yields_false_never_raises(self, tmp_path):
        """(c) slots_dir pointing at a regular file (not a directory) fails
        open rather than propagating an OSError out of the context manager.
        """
        from shared.verify_admission import acquire_task_slot

        not_a_dir = tmp_path / 'i-am-a-file'
        not_a_dir.write_text('not a directory')

        with acquire_task_slot('task', slots_dir=not_a_dir, n=1, wait=True) as held:
            assert held is False
