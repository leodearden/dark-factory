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

import pytest

from shared.delivered_check_polarity import lint_delivered_checks, polarity_error
from shared.delivered_check_scope import (
    STALE_PATH_CODES,
    STALE_PATH_REASONS,
    SYS_MODULES_SHIM_PATTERN,
    PathState,
    ScopePath,
    classify_scope_paths,
    is_literal_scope_path,
    resolve_commit,
    stale_scope_paths,
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
                'src/spaced.py': 'import sys\n\nsys.modules[__name__]  =  object()\n',
                'src/tabbed.py': 'import sys\n\nsys.modules[__name__]\t= object()\n',
            },
        )

        result = classify_scope_paths(
            [
                'src/old_mod.py',
                'src/indented.py',
                'docs/x.md',
                'src/new_mod.py',
                'src/spaced.py',
                'src/tabbed.py',
            ],
            repo_root=repo,
            ref='main',
        )

        assert _states(result) == {
            'src/old_mod.py': PathState.SYS_MODULES_SHIM,
            'src/indented.py': PathState.LIVE,
            'docs/x.md': PathState.LIVE,
            'src/new_mod.py': PathState.LIVE,
            'src/spaced.py': PathState.SYS_MODULES_SHIM,
            'src/tabbed.py': PathState.SYS_MODULES_SHIM,
        }

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

    def test_the_classified_keys_are_exactly_the_literal_entries(self, tmp_path):
        repo = _init_git_repo(tmp_path / 'repo', {'src/a.py': ''})
        entries = ['src/*.py', ':(glob)src/**', 'src/[ab].py', '/', '', 'src/a.py', 'src/']

        result = classify_scope_paths(entries, repo_root=repo, ref='main')

        assert result is not None
        assert set(result) == {p for p in entries if is_literal_scope_path(p)}
        assert set(result) == {'src/a.py', 'src/'}

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


_STALE_TABLE = [
    ('grep', PathState.LIVE, False),
    ('grep', PathState.SYS_MODULES_SHIM, True),
    ('grep', PathState.REMOVED, True),
    ('grep', PathState.NEVER_EXISTED, False),
    ('path', PathState.LIVE, False),
    ('path', PathState.SYS_MODULES_SHIM, False),
    ('path', PathState.REMOVED, True),
    ('path', PathState.NEVER_EXISTED, False),
]


class TestStaleScopePolicy:
    """The single staleness policy: only expect='present' checks are subject;
    a grep is stale on a shim or a removed path, a path check only on a removed
    one (a shim is a real file, and deleting one is a legitimate capability)."""

    @pytest.mark.parametrize(('kind', 'state', 'stale'), _STALE_TABLE)
    def test_present_checks_follow_the_per_kind_table(self, kind, state, stale):
        scope = ScopePath('src/x.py', state)

        assert stale_scope_paths(kind, 'present', [scope]) == ((scope,) if stale else ())

    @pytest.mark.parametrize('expect', ['absent', None, 'bogus'])
    @pytest.mark.parametrize(('kind', 'state', '_stale'), _STALE_TABLE)
    def test_non_present_expect_is_never_stale(self, kind, state, _stale, expect):
        assert stale_scope_paths(kind, expect, [ScopePath('src/x.py', state)]) == ()

    @pytest.mark.parametrize('kind', ['script', None, ['grep']])
    def test_other_kinds_are_never_stale_and_never_raise(self, kind):
        scope = ScopePath('src/x.py', PathState.REMOVED)

        assert stale_scope_paths(kind, 'present', [scope]) == ()

    def test_result_keeps_input_order_and_the_scope_objects_themselves(self):
        gone = ScopePath('src/gone.py', PathState.REMOVED, 'abc1234 drop it')
        live = ScopePath('src/live.py', PathState.LIVE)
        shim = ScopePath('src/old_mod.py', PathState.SYS_MODULES_SHIM)

        result = stale_scope_paths('grep', 'present', [shim, live, gone])

        assert result == (shim, gone)
        assert result[0] is shim
        assert result[1] is gone

    def test_stale_path_codes_name_exactly_the_two_stale_states(self):
        assert dict(STALE_PATH_CODES) == {
            PathState.SYS_MODULES_SHIM: 'shim_path',
            PathState.REMOVED: 'removed_path',
        }

    def test_every_stale_code_carries_one_reason(self):
        assert set(STALE_PATH_REASONS) == set(STALE_PATH_CODES.values())


@pytest.fixture
def stale_repo(tmp_path) -> tuple[Path, str]:
    """main carries a shim, a live module, and a module a later commit deleted.
    Returns the repo and the deleting commit's short sha."""
    repo = _init_git_repo(
        tmp_path / 'repo',
        {
            'src/new_mod.py': 'def real():\n    return 1\n',
            'src/old_mod.py': _SHIM_BODY,
            'src/gone.py': 'def legacy():\n    return 0\n',
        },
    )
    sha = _commit(repo, 'retire the gone module', delete=('src/gone.py',))
    return repo, _short(repo, sha)


def _check(**fields: object) -> dict[str, object]:
    check: dict[str, object] = {
        'name': 'cap',
        'kind': 'grep',
        'pattern': 'brand_new_symbol',
        'expect': 'present',
    }
    check.update(fields)
    return check


def _codes(findings) -> list[tuple[str, str]]:
    return [(f.check_name, f.code) for f in findings]


#: Argv elements only the scope axis's git probes carry: the shim probe's
#: pattern and the mainline deletion walk's ``--first-parent``.
_SCOPE_PROBE_MARKERS = (SYS_MODULES_SHIM_PATTERN, '--first-parent')


class TestLintFlagsStaleScope:
    def test_grep_scoped_to_a_shim_is_rejected_as_shim_path(self, stale_repo):
        repo, _ = stale_repo

        findings = lint_delivered_checks(
            [_check(paths=['src/old_mod.py'])], files=[], repo_root=repo
        )

        assert len(findings) == 1
        finding = findings[0]
        assert (finding.severity, finding.code, finding.detail) == (
            'reject',
            'shim_path',
            ('src/old_mod.py',),
        )
        assert 'src/old_mod.py' in finding.message
        assert 'sys.modules' in finding.message
        assert 'alias' in finding.message.lower()
        assert 'repath' in finding.message.lower()

    def test_grep_scoped_to_a_removed_file_names_the_removing_commit(self, stale_repo):
        repo, short_sha = stale_repo

        findings = lint_delivered_checks(
            [_check(paths=['src/gone.py'])], files=[], repo_root=repo
        )

        assert [(f.severity, f.code, f.detail) for f in findings] == [
            ('reject', 'removed_path', ('src/gone.py',)),
        ]
        assert short_sha in findings[0].message
        assert 'repath' in findings[0].message.lower()

    def test_path_check_naming_a_removed_file_is_rejected(self, stale_repo):
        repo, _ = stale_repo

        findings = lint_delivered_checks(
            [_check(kind='path', pattern=None, paths=['src/gone.py'])],
            files=[],
            repo_root=repo,
        )

        assert [(f.severity, f.code) for f in findings] == [('reject', 'removed_path')]

    def test_forward_looking_paths_get_no_finding(self, stale_repo):
        """A missing path is the normal pre-delivery state."""
        repo, _ = stale_repo

        findings = lint_delivered_checks(
            [
                _check(name='grep-new', paths=['src/never_created.py']),
                _check(
                    name='path-new', kind='path', pattern=None, paths=['src/never_created.py']
                ),
            ],
            files=[],
            repo_root=repo,
        )

        assert findings == []

    @pytest.mark.parametrize('scope', ['src/old_mod.py', 'src/gone.py'])
    def test_absent_checks_are_left_to_the_polarity_axis(self, stale_repo, scope):
        repo, _ = stale_repo

        findings = lint_delivered_checks(
            [_check(expect='absent', pattern='def ', paths=[scope])],
            files=[],
            repo_root=repo,
        )

        assert not {'shim_path', 'removed_path'} & {f.code for f in findings}

    def test_a_vacuous_shim_scoped_check_carries_both_axes(self, stale_repo):
        repo, _ = stale_repo

        findings = lint_delivered_checks(
            [_check(pattern='sys.modules', paths=['src/old_mod.py'])],
            files=[],
            repo_root=repo,
        )

        assert sorted(_codes(findings)) == [('cap', 'shim_path'), ('cap', 'vacuous_present')]

    def test_non_repo_root_reports_only_unevaluable(self, tmp_path):
        not_a_repo = tmp_path / 'plain'
        not_a_repo.mkdir()

        findings = lint_delivered_checks(
            [_check(paths=['src/old_mod.py'])], files=[], repo_root=not_a_repo
        )

        assert [(f.severity, f.code) for f in findings] == [('errored', 'unevaluable')]

    def test_unlinted_checks_cost_no_git_call(self, stale_repo):
        repo, _ = stale_repo

        with patch(
            'shared.delivered_check_scope.subprocess.run',
            side_effect=AssertionError('the scope axis must not shell out here'),
        ):
            findings = lint_delivered_checks(
                [
                    {'name': 'cap', 'kind': 'script', 'script': 'scripts/x.sh'},
                    _check(name=None, paths=['src/old_mod.py']),
                    _check(name='', kind='path', pattern=None, paths=['src/gone.py']),
                ],
                files=[],
                repo_root=repo,
            )

        assert findings == []

    def test_pathless_and_absent_checks_run_no_scope_probe(self, stale_repo):
        """The polarity axis still evaluates these; the scope axis must not."""
        repo, _ = stale_repo
        real_run = subprocess.run
        argvs: list[list[str]] = []

        def recording_run(argv, *args, **kwargs):
            argvs.append(list(argv))
            return real_run(argv, *args, **kwargs)

        with patch('shared.delivered_check_scope.subprocess.run', side_effect=recording_run):
            lint_delivered_checks(
                [
                    _check(name='pathless', paths=[]),
                    _check(
                        name='absent',
                        expect='absent',
                        pattern='def ',
                        paths=['src/new_mod.py', 'src/never_created.py'],
                    ),
                ],
                files=[],
                repo_root=repo,
            )

        assert argvs, 'the polarity axis should still have evaluated the checks'
        assert [
            argv for argv in argvs if any(m in argv for m in _SCOPE_PROBE_MARKERS)
        ] == []

    def test_runtime_diagnosis_call_shape_without_files_still_flags(self, stale_repo):
        repo, _ = stale_repo

        findings = lint_delivered_checks(
            [_check(paths=['src/old_mod.py'])], files=None, repo_root=repo
        )

        assert _codes(findings) == [('cap', 'shim_path')]

    def test_polarity_error_lists_the_stale_check_and_hints_at_repathing(self, stale_repo):
        repo, _ = stale_repo
        findings = lint_delivered_checks(
            [_check(paths=['src/old_mod.py'])], files=[], repo_root=repo
        )

        payload = polarity_error(findings, task_id='42')

        assert [(c['name'], c['code']) for c in payload['checks']] == [('cap', 'shim_path')]
        assert 'sys.modules' in payload['hint']
        assert 'repath' in payload['hint'].lower()
