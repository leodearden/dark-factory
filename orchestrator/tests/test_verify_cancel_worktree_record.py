"""The worktree-path record a remote verify keeps beside its pgid file.

A cancel SIGKILLs the verify tree, so the subcommand's own ``finally`` never
removes its worktree; the record is what lets ``cancel_request`` reap it
(PRD plans/remote-task-leg-verify-prd.md §4.2).
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.verify_cancel import (
    cancel_request,
    pgid_file,
    read_worktree_path,
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
