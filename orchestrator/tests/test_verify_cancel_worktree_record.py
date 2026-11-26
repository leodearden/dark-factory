"""The worktree-path record a remote verify keeps beside its pgid file.

A cancel SIGKILLs the verify tree, so the subcommand's own ``finally`` never
removes its worktree; the record is what lets ``cancel_request`` reap it
(PRD plans/remote-task-leg-verify-prd.md §4.2).
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from orchestrator.verify_cancel import (
    KILLED_RUN_LOCK_RELEASE_SECS,
    acquire_merge_verify_flock,
    cancel_request,
    lane_lock_path,
    pgid_file,
    read_worktree_path,
    release_merge_verify_flock,
    remove_worktree_path,
    worktree_path_file,
    write_pgid_file,
    write_worktree_path,
)


class TestWorktreePathRecord:
    def test_record_is_a_distinct_sibling_of_the_pgid_file(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-1')
        record = worktree_path_file(pgf)
        assert record.parent == pgf.parent
        assert record != pgf

    def test_distinct_request_ids_get_distinct_records(self, tmp_path: Path) -> None:
        base = tmp_path / 'wt'
        first = worktree_path_file(pgid_file(base, 'req-1'))
        second = worktree_path_file(pgid_file(base, 'req-2'))
        assert first != second

    def test_write_then_read_round_trips(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-rt')
        wt = tmp_path / 'wt' / '_merge-abc'
        write_worktree_path(pgf, wt)
        assert read_worktree_path(pgf) == Path(wt)

    def test_write_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'absent' / 'wt', 'req-parents')
        assert not pgf.parent.exists()
        write_worktree_path(pgf, tmp_path / 'some-wt')
        assert read_worktree_path(pgf) == tmp_path / 'some-wt'

    def test_overwrite_replaces_content_and_leaves_no_tmp_file(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req.1.2')
        write_worktree_path(pgf, tmp_path / 'first')
        write_worktree_path(pgf, tmp_path / 'second')
        assert read_worktree_path(pgf) == tmp_path / 'second'
        assert list(pgf.parent.glob('*.tmp')) == []

    def test_read_returns_none_when_record_absent(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-absent')
        assert read_worktree_path(pgf) is None

    def test_read_returns_none_when_record_empty(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-empty')
        record = worktree_path_file(pgf)
        record.parent.mkdir(parents=True)
        record.write_text('')
        assert read_worktree_path(pgf) is None

    def test_remove_is_idempotent(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-rm')
        remove_worktree_path(pgf)
        write_worktree_path(pgf, tmp_path / 'wt-x')
        remove_worktree_path(pgf)
        assert not worktree_path_file(pgf).exists()
        remove_worktree_path(pgf)

    def test_record_lifecycle_never_touches_the_pgid_file(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-pgid')
        write_pgid_file(pgf, 4242)
        write_worktree_path(pgf, tmp_path / 'wt-y')
        assert pgf.read_text() == '4242'
        remove_worktree_path(pgf)
        assert pgf.read_text() == '4242'


class _RecordingRemover:
    """A ``remove_worktree`` double that notes whether the pgid file was gone."""

    def __init__(self, pgid_path: Path) -> None:
        self.pgid_path = pgid_path
        self.calls: list[Path] = []
        self.pgid_file_existed_at_call: list[bool] = []

    def __call__(self, worktree: Path) -> None:
        self.calls.append(worktree)
        self.pgid_file_existed_at_call.append(self.pgid_path.exists())


def _refuse_kill(pid: int, sig: int) -> None:
    raise PermissionError(pid)


def _ignore_signal(target: int, sig: int) -> None:
    return None


def _cancel(path: Path, remover: _RecordingRemover, *, kill=_ignore_signal, out=None) -> int:
    """cancel_request with declawed kill/killpg and an empty process tree."""
    return cancel_request(
        path,
        ppid_map_provider=dict,
        kill=kill,
        killpg=_ignore_signal,
        killed_pgid_out=out,
        remove_worktree=remover,
    )


class TestCancelRequestReapsRecordedWorktree:
    def test_success_path_reaps_record_after_pgid_file_removed(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-a')
        write_pgid_file(pgf, 4242)
        wt = tmp_path / 'wt' / '_merge-a'
        write_worktree_path(pgf, wt)
        remover = _RecordingRemover(pgf)
        out: list[int] = []

        rc = _cancel(pgf, remover, out=out)

        assert rc == 0
        assert remover.calls == [wt]
        assert remover.pgid_file_existed_at_call == [False]
        assert not pgf.exists()
        assert not worktree_path_file(pgf).exists()
        assert out == [4242]

    def test_success_path_without_record_never_calls_remover(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-b')
        write_pgid_file(pgf, 4242)
        remover = _RecordingRemover(pgf)

        assert _cancel(pgf, remover) == 0
        assert remover.calls == []

    def test_absent_pgid_file_retains_record(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-c')
        write_worktree_path(pgf, tmp_path / 'wt' / '_merge-c')
        remover = _RecordingRemover(pgf)

        assert _cancel(pgf, remover) == 0
        assert remover.calls == []
        assert read_worktree_path(pgf) == tmp_path / 'wt' / '_merge-c'

    def test_permission_error_retains_pgid_file_and_record(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-d')
        write_pgid_file(pgf, 4242)
        write_worktree_path(pgf, tmp_path / 'wt' / '_merge-d')
        remover = _RecordingRemover(pgf)

        assert _cancel(pgf, remover, kill=_refuse_kill) == 1
        assert remover.calls == []
        assert pgf.exists()
        assert read_worktree_path(pgf) == tmp_path / 'wt' / '_merge-d'

    def test_corrupt_pgid_content_retains_record(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-e')
        pgf.parent.mkdir(parents=True)
        pgf.write_text('not-a-pgid')
        write_worktree_path(pgf, tmp_path / 'wt' / '_merge-e')
        remover = _RecordingRemover(pgf)

        assert _cancel(pgf, remover) == 0
        assert remover.calls == []
        assert read_worktree_path(pgf) == tmp_path / 'wt' / '_merge-e'

    def test_success_path_without_remover_retains_record(self, tmp_path: Path) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-f')
        write_pgid_file(pgf, 4242)
        write_worktree_path(pgf, tmp_path / 'wt' / '_merge-f')

        rc = cancel_request(
            pgf, ppid_map_provider=dict, kill=_ignore_signal, killpg=_ignore_signal,
        )

        assert rc == 0
        assert read_worktree_path(pgf) == tmp_path / 'wt' / '_merge-f'


_HOLDER_SCRIPT = (
    'import fcntl, os, sys, time\n'
    'fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)\n'
    'fcntl.flock(fd, fcntl.LOCK_EX)\n'
    "print('locked', flush=True)\n"
    'time.sleep(300)\n'
)


@contextlib.contextmanager
def _foreign_lane_holder(lock_path: Path) -> Iterator[subprocess.Popen[bytes]]:
    """A child process that holds an exclusive flock on *lock_path* until killed."""
    child = subprocess.Popen(
        [sys.executable, '-c', _HOLDER_SCRIPT, str(lock_path)], stdout=subprocess.PIPE,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline() == b'locked\n', 'the holder child never took the lock'
        yield child
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        if child.stdout is not None:
            child.stdout.close()


class _VirtualClock:
    """Injected ``monotonic``/``sleep``: each sleep advances ``now`` and is recorded."""

    def __init__(self, on_first_sleep: Callable[[], None] | None = None) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self._on_first_sleep = on_first_sleep

    def monotonic(self) -> float:
        return self.now

    def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self.now += secs
        callback, self._on_first_sleep = self._on_first_sleep, None
        if callback is not None:
            callback()


class _LockProbingRemover:
    """A ``remove_worktree`` double that notes whether the lane flock was free."""

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self.calls: list[Path] = []
        self.lock_free_at_call: list[bool] = []

    def __call__(self, worktree: Path) -> None:
        self.calls.append(worktree)
        fd = acquire_merge_verify_flock(self.lock_path, 0.0)
        self.lock_free_at_call.append(fd is not None)
        if fd is not None:
            release_merge_verify_flock(fd)


def _kill_and_reap(child: subprocess.Popen[bytes]) -> Callable[[], None]:
    def finish_the_kill() -> None:
        child.kill()
        child.wait()
    return finish_the_kill


class TestCancelRequestWaitsForTheKilledHolder:
    def test_reaps_only_after_the_killed_holder_releases_the_lane_lock(
        self, tmp_path: Path,
    ) -> None:
        wt = tmp_path / 'wt' / '_merge-held'
        wt.parent.mkdir(parents=True)
        pgf = pgid_file(tmp_path / 'wt', 'req-held')
        remover = _LockProbingRemover(lane_lock_path(wt))
        out: list[int] = []

        with _foreign_lane_holder(lane_lock_path(wt)) as child:
            write_pgid_file(pgf, child.pid)
            write_worktree_path(pgf, wt)
            clock = _VirtualClock(on_first_sleep=_kill_and_reap(child))

            rc = cancel_request(
                pgf,
                ppid_map_provider=dict,
                kill=_ignore_signal,
                killpg=_ignore_signal,
                killed_pgid_out=out,
                remove_worktree=remover,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

        assert rc == 0
        assert remover.calls == [wt]
        assert remover.lock_free_at_call == [True]
        assert clock.sleeps != []
        assert not worktree_path_file(pgf).exists()
        assert out == [child.pid]

    def test_no_wait_when_the_recorded_worktree_has_no_lane_lock(
        self, tmp_path: Path,
    ) -> None:
        pgf = pgid_file(tmp_path / 'wt', 'req-unlocked')
        write_pgid_file(pgf, 4242)
        wt = tmp_path / 'wt' / '_merge-unlocked'
        write_worktree_path(pgf, wt)
        remover = _RecordingRemover(pgf)
        clock = _VirtualClock()

        rc = cancel_request(
            pgf,
            ppid_map_provider=dict,
            kill=_ignore_signal,
            killpg=_ignore_signal,
            remove_worktree=remover,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

        assert rc == 0
        assert remover.calls == [wt]
        assert clock.sleeps == []

    def test_a_holder_that_outlives_the_bound_is_reported_and_still_handed_to_the_remover(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        wt = tmp_path / 'wt' / '_merge-stuck'
        wt.parent.mkdir(parents=True)
        pgf = pgid_file(tmp_path / 'wt', 'req-stuck')
        remover = _LockProbingRemover(lane_lock_path(wt))
        clock = _VirtualClock()

        with _foreign_lane_holder(lane_lock_path(wt)) as child:
            write_pgid_file(pgf, child.pid)
            write_worktree_path(pgf, wt)

            rc = cancel_request(
                pgf,
                ppid_map_provider=dict,
                kill=_ignore_signal,
                killpg=_ignore_signal,
                remove_worktree=remover,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )

        assert rc == 0
        assert remover.calls == [wt]
        assert remover.lock_free_at_call == [False]
        assert clock.now >= KILLED_RUN_LOCK_RELEASE_SECS
        assert not worktree_path_file(pgf).exists()
        err = capsys.readouterr().err
        assert str(child.pid) in err
        assert str(lane_lock_path(wt)) in err
