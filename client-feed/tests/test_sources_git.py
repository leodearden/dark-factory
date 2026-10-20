"""Read-only git plumbing confined to one root, over throwaway repositories."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from _git_repo import Repo, make_repo
from client_feed.sources.git import (
    EntryKind,
    GitError,
    TreeEntry,
    blob_reader,
    main_head,
    repository_toplevel,
    tree_entries,
)

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
T1 = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
T2 = datetime(2026, 1, 3, 12, 0, tzinfo=UTC)


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


def test_repository_toplevel_is_none_outside_a_repository(tmp_path: Path) -> None:
    plain = tmp_path / 'plain'
    plain.mkdir()

    assert repository_toplevel(plain) is None
    assert repository_toplevel(tmp_path / 'missing') is None


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


def test_a_non_hex_blob_is_refused_before_it_is_requested(repo: Repo) -> None:
    with pytest.raises(ValueError, match='sha'), blob_reader(repo.root) as read:
        read(f'{"a" * 40}\n{"b" * 40}')
