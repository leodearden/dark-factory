"""Tests for ``shared.delivered_check_scope`` — what a delivered_check's
``paths`` entries name at a commit (task 6480).

Every case runs against a REAL throwaway git repo on branch ``main``: the
classifier is a measurement of the repository's tree and mainline history,
so a mocked git would only test the mock.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from shared.delivered_check_scope import (
    SYS_MODULES_SHIM_PATTERN,
    PathState,
    ScopePath,
    classify_scope_paths,
    resolve_commit,
)

_SHIM_BODY = 'import sys\n\nfrom src import new_mod as target\n\nsys.modules[__name__] = target\n'


def _run_git(root: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(root), *args], check=True, capture_output=True, text=True
    ).stdout


def _commit(
    root: Path,
    message: str,
    files: dict[str, str] | None = None,
    delete: tuple[str, ...] = (),
) -> str:
    """Write *files*, ``git rm`` *delete*, commit everything; return the sha."""
    for rel, body in (files or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding='utf-8')
    for rel in delete:
        _run_git(root, 'rm', '-r', '-q', rel)
    _run_git(root, 'add', '-A')
    _run_git(root, 'commit', '--no-verify', '-q', '-m', message)
    return _run_git(root, 'rev-parse', 'HEAD').strip()


def _init_git_repo(root: Path, files: dict[str, str]) -> Path:
    """A real repo at *root* on branch main, seeded with *files* and committed."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['git', 'init', '-b', 'main', str(root)], check=True, capture_output=True, text=True
    )
    _run_git(root, 'config', 'user.email', 'scope-test@example.com')
    _run_git(root, 'config', 'user.name', 'Scope Test')
    _run_git(root, 'config', 'commit.gpgsign', 'false')
    _commit(root, 'seed the tree', files=files)
    return root


def _short(root: Path, sha: str) -> str:
    return _run_git(root, 'rev-parse', '--short', sha).strip()


def _states(result: dict[str, ScopePath] | None) -> dict[str, PathState]:
    assert result is not None
    return {path: scope.state for path, scope in result.items()}


class TestClassifyScopePaths:
    def test_tracked_file_and_directory_are_live(self, tmp_path):
        repo = _init_git_repo(tmp_path / 'repo', {'src/a.py': 'x = 1\n', 'src/b.py': 'y = 2\n'})

        result = classify_scope_paths(['src/a.py', 'src', 'src/'], repo_root=repo, ref='main')

        assert result == {
            'src/a.py': ScopePath('src/a.py', PathState.LIVE, None),
            'src': ScopePath('src', PathState.LIVE, None),
            'src/': ScopePath('src/', PathState.LIVE, None),
        }

    def test_module_level_sys_modules_rebind_is_a_shim(self, tmp_path):
        repo = _init_git_repo(
            tmp_path / 'repo',
            {
                'src/new_mod.py': 'def real():\n    return 1\n',
                'src/old_mod.py': _SHIM_BODY,
                'src/indented.py': (
                    'import sys\n\ndef swap(x):\n    sys.modules[__name__] = x\n'
                ),
                'docs/x.md': 'sys.modules[__name__] = y\n',
            },
        )

        result = classify_scope_paths(
            ['src/old_mod.py', 'src/indented.py', 'docs/x.md', 'src/new_mod.py'],
            repo_root=repo,
            ref='main',
        )

        assert _states(result) == {
            'src/old_mod.py': PathState.SYS_MODULES_SHIM,
            'src/indented.py': PathState.LIVE,
            'docs/x.md': PathState.LIVE,
            'src/new_mod.py': PathState.LIVE,
        }

    def test_shim_pattern_is_a_line_anchored_posix_ere(self):
        assert SYS_MODULES_SHIM_PATTERN.startswith('^sys')
        assert '[[:space:]]' in SYS_MODULES_SHIM_PATTERN

    def test_file_deleted_on_main_is_removed_naming_the_commit(self, tmp_path):
        repo = _init_git_repo(tmp_path / 'repo', {'src/gone.py': 'x = 1\n', 'src/keep.py': ''})
        sha = _commit(repo, 'delete the gone module', delete=('src/gone.py',))

        result = classify_scope_paths(['src/gone.py'], repo_root=repo, ref='main')

        assert result is not None
        scope = result['src/gone.py']
        assert scope.state is PathState.REMOVED
        assert scope.removed_in is not None
        assert scope.removed_in.startswith(_short(repo, sha))
        assert 'delete the gone module' in scope.removed_in

    def test_moved_file_is_removed_at_old_path_and_live_at_new(self, tmp_path):
        repo = _init_git_repo(tmp_path / 'repo', {'src/old_home.py': 'x = 1\n'})
        _run_git(repo, 'mv', 'src/old_home.py', 'src/new_home.py')
        _commit(repo, 'move the module')

        result = classify_scope_paths(
            ['src/old_home.py', 'src/new_home.py'], repo_root=repo, ref='main'
        )

        assert _states(result) == {
            'src/old_home.py': PathState.REMOVED,
            'src/new_home.py': PathState.LIVE,
        }

    def test_directory_whose_every_file_was_deleted_is_removed(self, tmp_path):
        repo = _init_git_repo(
            tmp_path / 'repo',
            {'olddir/a.py': 'a\n', 'olddir/sub/b.py': 'b\n', 'keep.py': ''},
        )
        _commit(repo, 'drop olddir', delete=('olddir',))

        result = classify_scope_paths(['olddir', 'olddir/'], repo_root=repo, ref='main')

        assert _states(result) == {
            'olddir': PathState.REMOVED,
            'olddir/': PathState.REMOVED,
        }

    def test_path_never_in_history_never_existed(self, tmp_path):
        repo = _init_git_repo(tmp_path / 'repo', {'src/a.py': ''})

        result = classify_scope_paths(['src/never_created.py'], repo_root=repo, ref='main')

        assert result == {
            'src/never_created.py': ScopePath(
                'src/never_created.py', PathState.NEVER_EXISTED, None
            ),
        }

    def test_file_only_ever_on_a_merged_side_branch_never_existed(self, tmp_path):
        """First-parent semantics: a path that lived and died on a side branch
        never existed on the mainline, so its absence is not staleness."""
        repo = _init_git_repo(tmp_path / 'repo', {'src/a.py': ''})
        _run_git(repo, 'checkout', '-q', '-b', 'side')
        _commit(repo, 'side adds a scratch file', files={'src/scratch.py': 'x\n'})
        _commit(repo, 'side drops the scratch file', delete=('src/scratch.py',))
        _run_git(repo, 'checkout', '-q', 'main')
        _run_git(repo, 'merge', '-q', '--no-ff', '--no-edit', 'side')

        result = classify_scope_paths(['src/scratch.py'], repo_root=repo, ref='main')

        assert _states(result) == {'src/scratch.py': PathState.NEVER_EXISTED}

    def test_classification_is_at_the_resolved_ref_not_a_cached_name(self, tmp_path):
        """The mainline deletion index is memoised; it must be keyed on the
        RESOLVED commit, so a later commit on the same branch name is seen."""
        repo = _init_git_repo(tmp_path / 'repo', {'src/a.py': 'x\n', 'src/keep.py': ''})

        before = classify_scope_paths(
            ['src/a.py', 'src/never.py'], repo_root=repo, ref='main'
        )
        _commit(repo, 'delete a', delete=('src/a.py',))
        after = classify_scope_paths(
            ['src/a.py', 'src/never.py'], repo_root=repo, ref='main'
        )

        assert _states(before) == {
            'src/a.py': PathState.LIVE,
            'src/never.py': PathState.NEVER_EXISTED,
        }
        assert _states(after) == {
            'src/a.py': PathState.REMOVED,
            'src/never.py': PathState.NEVER_EXISTED,
        }

    def test_glob_and_pathspec_magic_entries_are_not_classified(self, tmp_path):
        repo = _init_git_repo(tmp_path / 'repo', {'src/a.py': ''})

        result = classify_scope_paths(
            ['src/*.py', ':(glob)src/**', 'src/a?.py', 'src/[ab].py', 'src/a.py'],
            repo_root=repo,
            ref='main',
        )

        assert _states(result) == {'src/a.py': PathState.LIVE}

    def test_non_repo_root_returns_none(self, tmp_path):
        not_a_repo = tmp_path / 'plain'
        not_a_repo.mkdir()

        assert classify_scope_paths(['src/a.py'], repo_root=not_a_repo, ref='main') is None

    def test_unresolvable_ref_returns_none(self, tmp_path):
        repo = _init_git_repo(tmp_path / 'repo', {'src/a.py': ''})

        assert classify_scope_paths(['src/a.py'], repo_root=repo, ref='nope') is None

    def test_nul_byte_path_returns_none(self, tmp_path):
        repo = _init_git_repo(tmp_path / 'repo', {'src/a.py': ''})

        assert (
            classify_scope_paths(['src/a.py', 'bad\0path'], repo_root=repo, ref='main')
            is None
        )

    def test_empty_paths_cost_no_subprocess(self, tmp_path):
        with patch(
            'shared.delivered_check_scope.subprocess.run',
            side_effect=AssertionError('empty input must not shell out'),
        ):
            assert classify_scope_paths([], repo_root=tmp_path, ref='main') == {}
            assert classify_scope_paths(['src/*.py'], repo_root=tmp_path, ref='main') == {}


class TestResolveCommit:
    def test_branch_name_resolves_to_its_sha(self, tmp_path):
        repo = _init_git_repo(tmp_path / 'repo', {'src/a.py': ''})

        assert resolve_commit(repo, 'main') == _run_git(repo, 'rev-parse', 'main').strip()

    def test_unknown_ref_is_none(self, tmp_path):
        repo = _init_git_repo(tmp_path / 'repo', {'src/a.py': ''})

        assert resolve_commit(repo, 'no-such-branch') is None

    def test_non_repo_dir_is_none(self, tmp_path):
        assert resolve_commit(tmp_path, 'main') is None
