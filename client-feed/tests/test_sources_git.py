"""Read-only git plumbing confined to one root, over throwaway repositories."""

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _git_repo import Repo, make_repo

from client_feed.contract import Branch
from client_feed.sources.git import (
    EntryKind,
    GitError,
    MergeStats,
    TreeEntry,
    blob_reader,
    generated_paths,
    is_ancestor,
    last_changes,
    main_head,
    merge_stats,
    repository_toplevel,
    task_branches,
    tree_entries,
)

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
T2 = datetime(2026, 1, 3, 12, 0, tzinfo=UTC)
T3 = datetime(2026, 1, 4, 12, 0, tzinfo=UTC)


@pytest.fixture
def repo(tmp_path: Path) -> Repo:
    return make_repo(tmp_path / 'repo')


def test_repository_toplevel_is_the_root_itself(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    repo.commit('one', when=T0)

    assert repository_toplevel(repo.root) == repo.root.resolve()


def test_repository_toplevel_of_a_subdirectory_is_the_enclosing_root(repo: Repo) -> None:
    repo.write('pkg/a.txt', 'a\n')
    repo.commit('one', when=T0)

    assert repository_toplevel(repo.root / 'pkg') == repo.root.resolve()


def test_repository_toplevel_outside_a_repository_raises_with_gits_own_reason(
    tmp_path: Path,
) -> None:
    plain = tmp_path / 'plain'
    plain.mkdir()

    with pytest.raises(GitError) as caught:
        repository_toplevel(plain)

    assert caught.value.returncode == 128
    assert caught.value.reason != str(caught.value)
    assert '\n' not in caught.value.reason
    assert str(plain) not in str(caught.value)


def test_repository_toplevel_of_a_missing_root_raises_with_the_os_reason(tmp_path: Path) -> None:
    with pytest.raises(GitError) as caught:
        repository_toplevel(tmp_path / 'missing')

    assert caught.value.returncode is None
    assert 'missing' in caught.value.reason
    assert str(tmp_path) not in str(caught.value)


def test_main_head_reads_main_even_while_another_branch_is_checked_out(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    main = repo.commit('one', when=T0)
    repo.branch('task/7')
    repo.checkout('task/7')
    repo.write('a.txt', 'b\n')
    repo.commit('two', when=T1)

    assert main_head(repo.root) == main


def test_main_head_is_none_before_the_first_commit(repo: Repo) -> None:
    assert main_head(repo.root) is None


def test_git_calls_are_pinned_to_the_root(
    repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo.write('a.txt', 'a\n')
    main = repo.commit('one', when=T0)
    other = make_repo(tmp_path / 'other')
    other.write('b.txt', 'b\n')
    other.commit('other', when=T1)
    monkeypatch.setenv('GIT_DIR', str(other.root / '.git'))
    monkeypatch.setenv('GIT_WORK_TREE', str(other.root))
    monkeypatch.chdir(other.root)

    assert main_head(repo.root) == main


def test_tree_entries_lists_files_symlinks_and_submodules(repo: Repo) -> None:
    repo.write('src/a.py', 'print(1)\n')
    script = repo.write('bin/run', '#!/bin/sh\n')
    script.chmod(0o755)
    repo.write('odd name/ünï.txt', 'x')
    repo.symlink('src/link', '/home/someone/secret')
    repo.gitlink('vendor/lib', 'b' * 40)
    commit = repo.commit('one', when=T0)

    entries = {entry.path: entry for entry in tree_entries(repo.root, commit)}

    assert entries['src/a.py'] == TreeEntry(
        path='src/a.py', kind=EntryKind.FILE, object_sha=repo.rev(f'{commit}:src/a.py'), size=9
    )
    assert entries['bin/run'].kind is EntryKind.FILE
    assert entries['odd name/ünï.txt'].size == 1
    assert entries['src/link'].kind is EntryKind.SYMLINK
    assert entries['vendor/lib'] == TreeEntry(
        path='vendor/lib', kind=EntryKind.SUBMODULE, object_sha='b' * 40, size=None
    )
    assert set(entries) == {'src/a.py', 'bin/run', 'odd name/ünï.txt', 'src/link', 'vendor/lib'}


def test_blob_reader_reads_committed_bytes_in_one_session(repo: Repo) -> None:
    repo.write('a.txt', b'alpha\n')
    repo.write('b.bin', b'\x00\x01\nbeta')
    repo.write('empty', b'')
    commit = repo.commit('one', when=T0)
    repo.write('a.txt', b'uncommitted edit\n')
    shas = {entry.path: entry.object_sha for entry in tree_entries(repo.root, commit)}

    with blob_reader(repo.root) as read:
        assert read(shas['a.txt']) == b'alpha\n'
        assert read(shas['b.bin']) == b'\x00\x01\nbeta'
        assert read(shas['empty']) == b''
        assert read(shas['a.txt']) == b'alpha\n'


def test_an_unknown_commit_raises_a_typed_error_without_a_path(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    repo.commit('one', when=T0)

    with pytest.raises(GitError) as caught:
        tree_entries(repo.root, '0' * 40)

    assert caught.value.subcommand == 'ls-tree'
    assert caught.value.returncode not in (0, None)
    assert str(repo.root) not in str(caught.value)
    assert '/' not in str(caught.value)


def test_reading_an_unknown_blob_raises_a_typed_error(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    repo.commit('one', when=T0)

    with pytest.raises(GitError) as caught, blob_reader(repo.root) as read:
        read('0' * 40)

    assert caught.value.subcommand == 'cat-file'
    assert str(repo.root) not in str(caught.value)


@pytest.mark.parametrize('commit', ['--output=x', 'HEAD', 'a' * 39, 'A' * 40, f'{"a" * 40}\n'])
def test_a_non_hex_commit_is_refused_before_git_runs(tmp_path: Path, commit: str) -> None:
    with pytest.raises(ValueError, match='sha'):
        tree_entries(tmp_path / 'missing', commit)


@pytest.mark.parametrize('since', [None, 'a' * 40], ids=['full-walk', 'incremental'])
def test_last_changes_refuses_a_non_hex_head_before_git_runs(
    tmp_path: Path, since: str | None
) -> None:
    with pytest.raises(ValueError, match='sha'):
        last_changes(tmp_path / 'missing', '--output=x', since=since)


def test_a_non_hex_blob_is_refused_before_it_is_requested(repo: Repo) -> None:
    with pytest.raises(ValueError, match='sha'), blob_reader(repo.root) as read:
        read(f'{"a" * 40}\n{"b" * 40}')


def test_task_branches_report_how_far_each_is_ahead_of_main(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    repo.commit('base', when=T0)
    for name in ('task/7', 'task/8', 'task/spike', 'feature/x'):
        repo.branch(name)
    repo.checkout('task/7')
    repo.write('seven.txt', '1\n')
    repo.commit('seven one', when=T1)
    repo.write('seven.txt', '2\n')
    repo.commit('seven two', when=T2)
    repo.checkout('task/8')
    repo.write('eight.txt', '8\n')
    repo.commit('eight', when=T1)
    repo.checkout('main')
    repo.merge_no_ff('task/8', 'Merge task/8 into main', when=T3)

    assert task_branches(repo.root) == (
        Branch(name='task/7', task_id=7, ahead=2, last_commit_at=T2),
        Branch(name='task/8', task_id=8, ahead=0, last_commit_at=T1),
        Branch(name='task/spike', task_id=None, ahead=0, last_commit_at=T0),
    )


def test_a_task_branch_whose_name_is_not_utf8_is_skipped(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    repo.commit('base', when=T0)
    repo.branch(os.fsdecode(b'task/\xff9'))
    repo.branch('task/9')

    assert task_branches(repo.root) == (
        Branch(name='task/9', task_id=9, ahead=0, last_commit_at=T0),
    )


def test_no_task_branches_is_empty(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    repo.commit('base', when=T0)
    repo.branch('feature/x')

    assert task_branches(repo.root) == ()


def test_merge_stats_count_what_the_merge_landed_on_main(repo: Repo) -> None:
    repo.write('a.txt', 'one\ntwo\n')
    repo.commit('base', when=T0)
    repo.branch('task/7')
    repo.checkout('task/7')
    repo.write('a.txt', 'one\nTWO\nthree\n')
    repo.write('b.txt', 'x\ny\nz\n')
    repo.commit('work', when=T1)
    repo.checkout('main')
    repo.write('c.txt', 'c\n')
    repo.commit('meanwhile on main', when=T1)
    merge = repo.merge_no_ff('task/7', 'Merge task/7 into main', when=T2)

    assert merge_stats(repo.root, merge) == MergeStats(files_changed=2, insertions=5, deletions=1)


def test_merge_stats_of_a_root_commit_count_its_files(repo: Repo) -> None:
    repo.write('a.txt', 'one\ntwo\n')
    repo.write('b.txt', 'b\n')
    root_commit = repo.commit('base', when=T0)

    assert merge_stats(repo.root, root_commit) == MergeStats(
        files_changed=2, insertions=3, deletions=0
    )


def test_merge_stats_count_a_pure_rename_as_one_unchanged_file(repo: Repo) -> None:
    repo.write('old name.txt', ''.join(f'line {n}\n' for n in range(20)))
    repo.commit('base', when=T0)
    (repo.root / 'old name.txt').rename(repo.root / 'new name.txt')
    rename = repo.commit('rename', when=T1)

    assert merge_stats(repo.root, rename) == MergeStats(files_changed=1, insertions=0, deletions=0)


def test_merge_stats_count_a_binary_file_without_lines(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    repo.commit('base', when=T0)
    repo.write('blob.bin', b'\x00\x01\x02\x00')
    binary = repo.commit('binary', when=T1)

    assert merge_stats(repo.root, binary) == MergeStats(files_changed=1, insertions=0, deletions=0)


def test_generated_paths_read_linguist_attributes_at_the_commit(repo: Repo) -> None:
    repo.write(
        '.gitattributes',
        'src/gen.py linguist-generated\n'
        'src/gen2.py linguist-generated=true\n'
        'src/x.py linguist-generated=false\n',
    )
    for name in ('gen', 'gen2', 'x', 'plain'):
        repo.write(f'src/{name}.py', f'{name} = 1\n')
    commit = repo.commit('base', when=T0)
    repo.write('.gitattributes', 'src/plain.py linguist-generated\n')

    generated = generated_paths(
        repo.root, commit, ['src/gen.py', 'src/gen2.py', 'src/x.py', 'src/plain.py']
    )

    assert generated == frozenset({'src/gen.py', 'src/gen2.py'})


def test_generated_paths_of_no_paths_spawns_no_git(tmp_path: Path) -> None:
    assert generated_paths(tmp_path / 'missing', 'a' * 40, []) == frozenset()


def test_last_changes_date_each_path_by_its_newest_first_parent_commit(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    repo.write('b.txt', 'b\n')
    repo.commit('base', when=T0)
    repo.write('b.txt', 'b2\n')
    second = repo.commit('b again', when=T1)
    repo.branch('task/7')
    repo.checkout('task/7')
    repo.write('c.txt', 'c\n')
    repo.commit('c on the branch', when=T2)
    repo.checkout('main')
    repo.write('d.txt', 'd\n')
    repo.commit('d on main while the branch is open', when=T2)
    head = repo.merge_no_ff('task/7', 'Merge task/7 into main', when=T3)

    assert last_changes(repo.root, head, since=None) == {
        'a.txt': T0,
        'b.txt': T1,
        'c.txt': T3,
        'd.txt': T2,
    }
    assert last_changes(repo.root, head, since=second) == {'c.txt': T3, 'd.txt': T2}
    assert last_changes(repo.root, head, since=head) == {}


def test_is_ancestor(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    first = repo.commit('base', when=T0)
    repo.write('a.txt', 'b\n')
    second = repo.commit('next', when=T1)

    assert is_ancestor(repo.root, first, second)
    assert is_ancestor(repo.root, first, first)
    assert not is_ancestor(repo.root, second, first)
    assert not is_ancestor(repo.root, '0' * 40, second)
