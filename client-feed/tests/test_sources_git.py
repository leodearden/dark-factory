"""Read-only git plumbing confined to one root, over throwaway repositories."""

import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest
from _git_repo import Repo, make_repo, path_with_git_answering, path_with_git_replaced_on

from client_feed.contract import Branch
from client_feed.sources.git import (
    ChangeStatus,
    EntryKind,
    GitError,
    MergeStats,
    PathChange,
    TreeEntry,
    added_lines,
    blob_reader,
    first_parent_changes,
    generated_paths,
    has_commit,
    is_ancestor,
    last_changes,
    main_head,
    merge_stats,
    repository_toplevel,
    task_branches,
    tree_changes,
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


def test_has_commit_holds_a_commit_on_main_or_on_another_branch_only(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    on_main = repo.commit('base', when=T0)
    repo.branch('task/7')
    repo.checkout('task/7')
    repo.write('a.txt', 'b\n')
    on_branch = repo.commit('seven', when=T1)

    assert has_commit(repo.root, on_main)
    assert has_commit(repo.root, on_branch)


def test_has_commit_is_false_for_a_sha_the_repository_does_not_hold(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    repo.commit('base', when=T0)

    assert not has_commit(repo.root, 'e' * 40)


def test_has_commit_is_false_for_an_object_that_is_not_a_commit(repo: Repo) -> None:
    repo.write('a.txt', 'a\n')
    commit = repo.commit('base', when=T0)

    assert not has_commit(repo.root, repo.rev(f'{commit}^{{tree}}'))


def test_has_commit_refuses_a_non_sha_before_git_runs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='sha'):
        has_commit(tmp_path / 'missing', '--output=x')


def test_has_commit_outside_a_repository_raises_rather_than_answering_false(
    tmp_path: Path,
) -> None:
    plain = tmp_path / 'plain'
    plain.mkdir()

    with pytest.raises(GitError) as caught:
        has_commit(plain, 'e' * 40)

    assert caught.value.returncode == 128


def test_has_commit_of_a_missing_root_raises_rather_than_answering_false(tmp_path: Path) -> None:
    with pytest.raises(GitError) as caught:
        has_commit(tmp_path / 'missing', 'e' * 40)

    assert caught.value.returncode is None


@dataclass(frozen=True)
class _History:
    c1: str
    c2: str
    c3: str


@pytest.fixture
def history(repo: Repo) -> _History:
    """c1 holds x, y and z; c2 modifies x, renames y to w unchanged and deletes z; c3 recreates z."""
    repo.write('x.py', 'x = 1\n')
    repo.write('y.py', 'why = "a line long enough to be its own file"\n')
    repo.write('z.py', 'zed = 3\n')
    c1 = repo.commit('c1', when=T0)
    repo.write('x.py', 'x = 2\n')
    repo.git('mv', 'y.py', 'w.py')
    repo.git('rm', '-q', 'z.py')
    c2 = repo.commit('c2', when=T1)
    repo.write('z.py', 'zed = "recreated"\n')
    c3 = repo.commit('c3', when=T2)
    return _History(c1=c1, c2=c2, c3=c3)


def _file_change(
    status: ChangeStatus, old: tuple[str, str] | None, new: tuple[str, str] | None
) -> PathChange:
    """A change between regular files, each side given as (path, blob) or None when absent."""
    return PathChange(
        status=status,
        old_path=None if old is None else old[0],
        new_path=None if new is None else new[0],
        old_blob=None if old is None else old[1],
        new_blob=None if new is None else new[1],
        old_kind=None if old is None else EntryKind.FILE,
        new_kind=None if new is None else EntryKind.FILE,
    )


def test_first_parent_changes_are_one_change_set_per_commit_oldest_first(
    repo: Repo, history: _History
) -> None:
    c1, c2, c3 = history.c1, history.c2, history.c3
    y_blob = repo.rev(f'{c1}:y.py')

    changes = first_parent_changes(repo.root, since=c1, head=c3)

    assert len(changes) == 2
    in_c2, in_c3 = changes
    assert set(in_c2) == {
        _file_change(
            ChangeStatus.MODIFIED,
            ('x.py', repo.rev(f'{c1}:x.py')),
            ('x.py', repo.rev(f'{c2}:x.py')),
        ),
        _file_change(ChangeStatus.RENAMED, ('y.py', y_blob), ('w.py', y_blob)),
        _file_change(ChangeStatus.DELETED, ('z.py', repo.rev(f'{c1}:z.py')), None),
    }
    assert repo.rev(f'{c1}:x.py') != repo.rev(f'{c2}:x.py')
    assert in_c3 == (_file_change(ChangeStatus.ADDED, None, ('z.py', repo.rev(f'{c3}:z.py'))),)


def test_first_parent_changes_keep_an_empty_change_set_for_a_commit_that_changes_nothing(
    repo: Repo, history: _History
) -> None:
    repo.git(
        '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', '-c', 'commit.gpgsign=false',
        'commit', '-q', '--allow-empty', '--no-verify', '-m', 'nothing',
    )
    empty = repo.rev('HEAD')

    changes = first_parent_changes(repo.root, since=history.c2, head=empty)

    assert len(changes) == 2
    assert changes[1] == ()


def test_a_merge_contributes_its_diff_against_its_first_parent_and_the_branch_commits_none(
    repo: Repo,
) -> None:
    repo.write('a.txt', 'a\n')
    repo.write('b.txt', 'b\n')
    base = repo.commit('base', when=T0)
    repo.branch('task/7')
    repo.checkout('task/7')
    repo.write('b.txt', 'b two\n')
    repo.commit('seven one', when=T1)
    repo.write('b.txt', 'b three\n')
    repo.write('c.txt', 'c\n')
    tip = repo.commit('seven two', when=T1)
    repo.checkout('main')
    repo.write('d.txt', 'd\n')
    meanwhile = repo.commit('meanwhile on main', when=T2)
    merge = repo.merge_no_ff('task/7', 'Merge task/7 into main', when=T3)

    changes = first_parent_changes(repo.root, since=base, head=merge)

    assert changes == (
        (_file_change(ChangeStatus.ADDED, None, ('d.txt', repo.rev(f'{meanwhile}:d.txt'))),),
        (
            _file_change(
                ChangeStatus.MODIFIED,
                ('b.txt', repo.rev(f'{base}:b.txt')),
                ('b.txt', repo.rev(f'{tip}:b.txt')),
            ),
            _file_change(ChangeStatus.ADDED, None, ('c.txt', repo.rev(f'{tip}:c.txt'))),
        ),
    )


def test_tree_changes_compare_two_trees_directly(repo: Repo, history: _History) -> None:
    c1, c3 = history.c1, history.c3
    y_blob = repo.rev(f'{c1}:y.py')

    changes = tree_changes(repo.root, c1, c3)

    assert set(changes) == {
        _file_change(
            ChangeStatus.MODIFIED,
            ('x.py', repo.rev(f'{c1}:x.py')),
            ('x.py', repo.rev(f'{c3}:x.py')),
        ),
        _file_change(ChangeStatus.RENAMED, ('y.py', y_blob), ('w.py', y_blob)),
        _file_change(
            ChangeStatus.MODIFIED,
            ('z.py', repo.rev(f'{c1}:z.py')),
            ('z.py', repo.rev(f'{c3}:z.py')),
        ),
    }
    assert len(changes) == 3


def test_a_regular_file_turned_symlink_is_a_type_change_to_a_symlink(repo: Repo) -> None:
    repo.write('l.py', 'l = 1\n')
    before = repo.commit('file', when=T0)
    (repo.root / 'l.py').unlink()
    repo.symlink('l.py', '/etc/passwd')
    after = repo.commit('symlink', when=T1)

    assert tree_changes(repo.root, before, after) == (
        PathChange(
            status=ChangeStatus.TYPE_CHANGED,
            old_path='l.py',
            new_path='l.py',
            old_blob=repo.rev(f'{before}:l.py'),
            new_blob=repo.rev(f'{after}:l.py'),
            old_kind=EntryKind.FILE,
            new_kind=EntryKind.SYMLINK,
        ),
    )


def _blob_pair(repo: Repo, path: str, old: str, new: str) -> tuple[str, str]:
    repo.write(path, old)
    before = repo.commit(f'{path} before', when=T0)
    repo.write(path, new)
    after = repo.commit(f'{path} after', when=T1)
    return repo.rev(f'{before}:{path}'), repo.rev(f'{after}:{path}')


def test_added_lines_are_the_added_or_changed_lines_only(repo: Repo) -> None:
    old, new = _blob_pair(repo, 'f.txt', 'a\nb\nc\n', 'a\nB\nc\n\nnew line\n')

    assert added_lines(repo.root, old, new) == b'B\n\nnew line'


def test_an_added_line_starting_with_plus_plus_loses_only_the_diffs_own_plus(repo: Repo) -> None:
    old, new = _blob_pair(repo, 'g.txt', 'a\n', 'a\n++x\n+++ b/fake\n')

    assert added_lines(repo.root, old, new) == b'++x\n+++ b/fake'


def test_added_lines_of_identical_blobs_are_empty(repo: Repo) -> None:
    old, _ = _blob_pair(repo, 'h.txt', 'same\n', 'other\n')

    assert added_lines(repo.root, old, old) == b''


@pytest.mark.parametrize('bad', ['--output=x', 'HEAD', 'a' * 39, 'A' * 40, f'{"a" * 40}\n'])
def test_change_readers_refuse_a_non_hex_sha_before_git_runs(tmp_path: Path, bad: str) -> None:
    root = tmp_path / 'missing'
    good = 'a' * 40
    calls = [
        lambda: tree_changes(root, bad, good),
        lambda: tree_changes(root, good, bad),
        lambda: first_parent_changes(root, since=bad, head=good),
        lambda: first_parent_changes(root, since=good, head=bad),
        lambda: added_lines(root, bad, good),
        lambda: added_lines(root, good, bad),
    ]
    for call in calls:
        with pytest.raises(ValueError, match='sha'):
            call()


def _answering(subcommand: str, stdout: bytes) -> Callable[[Path], str]:
    return functools.partial(path_with_git_answering, subcommand=subcommand, stdout=stdout)


def _read_one_blob(root: Path, sha: str) -> bytes:
    with blob_reader(root) as read:
        return read(sha)


PARSE_FAILURE = 'output not understood'
UNPARSABLE_OUTPUTS = [
    pytest.param(
        'ls-tree', _answering('ls-tree', b'garbage\0'), tree_entries, id='tree_entries'
    ),
    pytest.param(
        'for-each-ref',
        _answering('for-each-ref', b'refs/heads/task/7\x00not-a-time\x000 0\n'),
        lambda root, sha: task_branches(root),
        id='task_branches',
    ),
    pytest.param(
        'diff-tree', _answering('diff-tree', b'x\ty\tpath\0'), merge_stats, id='merge_stats'
    ),
    pytest.param(
        'log',
        _answering('log', b'\x01not-a-time\0'),
        lambda root, sha: last_changes(root, sha, since=None),
        id='last_changes',
    ),
    pytest.param(
        'diff-tree',
        _answering('diff-tree', b':bogus\0'),
        lambda root, sha: tree_changes(root, sha, sha),
        id='tree_changes',
    ),
    pytest.param(
        'log',
        _answering('log', b':bogus\0'),
        lambda root, sha: first_parent_changes(root, since=sha, head=sha),
        id='first_parent_changes',
    ),
    pytest.param(
        'cat-file',
        functools.partial(
            path_with_git_replaced_on,
            subcommand='cat-file',
            shell='read line; printf \'%s blob many\\n\' "$line"; exit 0',
        ),
        _read_one_blob,
        id='blob_reader',
    ),
]


@pytest.mark.parametrize(('subcommand', 'fake_git', 'reads'), UNPARSABLE_OUTPUTS)
def test_git_output_that_does_not_parse_is_a_git_error_of_its_command(
    repo: Repo,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    subcommand: str,
    fake_git: Callable[[Path], str],
    reads: Callable[[Path, str], object],
) -> None:
    repo.write('a.txt', 'a\n')
    commit = repo.commit('one', when=T0)
    monkeypatch.setenv('PATH', fake_git(tmp_path / 'bin'))

    with pytest.raises(GitError) as caught:
        reads(repo.root, commit)

    assert caught.value.subcommand == subcommand
    assert PARSE_FAILURE in caught.value.reason
    assert str(repo.root) not in str(caught.value)
