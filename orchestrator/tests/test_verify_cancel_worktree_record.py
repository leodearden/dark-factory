"""The worktree-path record a remote verify keeps beside its pgid file.

A cancel SIGKILLs the verify tree, so the subcommand's own ``finally`` never
removes its worktree; the record is what lets ``cancel_request`` reap it
(PRD plans/remote-task-leg-verify-prd.md §4.2).
"""

from __future__ import annotations

from pathlib import Path

from orchestrator.verify_cancel import (
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
