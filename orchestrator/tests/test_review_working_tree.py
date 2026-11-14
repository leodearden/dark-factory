"""Tests for orchestrator.review_working_tree: the uncommitted state of a review's checkout."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import git_env_with_ceiling
from df_pytest_isolation import git_redirect_env

from orchestrator.review_working_tree import (
    SECTION_NAMED_PATH_CAP,
    NamedChange,
    Phase1WorkingTree,
    TreeChange,
    WorkingTreeState,
    attribute_phase1,
    qualified_phase1_summary,
    read_working_tree_state,
    render_working_tree_section,
)
from orchestrator.verify import VerifyResult


def _git(cwd: Path, *args: str) -> None:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    subprocess.run(['git', *args], cwd=cwd, env=env, check=True, capture_output=True, text=True)


def _write(root: Path, relpath: str, text: str) -> None:
    target = root / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


_MIXED_SEED = RepoSeed(
    files=(
        ('scripts/ok.py', 'x = 1\n'),
        ('keep.py', 'k = 1\n'),
        ('gone.py', 'g = 1\n'),
        ('.gitignore', 'build/\n'),
    ),
    message='init',
)


@pytest.mark.asyncio
class TestReadWorkingTreeState:
    async def test_clean_repo_is_clean(self, tmp_path: Path) -> None:
        repo = seed_repo(tmp_path / 'repo')

        state = await read_working_tree_state(repo)

        assert state is not None
        assert state.is_clean
        assert state.untracked == ()
        assert state.uncommitted == ()

    async def test_mixed_dirty_repo_lists_untracked_and_uncommitted_sorted(
        self, tmp_path: Path,
    ) -> None:
        repo = seed_repo(tmp_path / 'repo', _MIXED_SEED)
        _write(repo, 'scripts/bad.py', 'import os\n')
        _write(repo, 'build/out.py', 'ignored = True\n')
        _write(repo, 'notes dir/a b.py', 'spaced = True\n')
        _write(repo, 'keep.py', 'k = 2\n')
        _write(repo, 'staged.py', 's = 1\n')
        _git(repo, 'add', 'staged.py')
        (repo / 'gone.py').unlink()

        state = await read_working_tree_state(repo)

        assert state is not None
        assert not state.is_clean
        assert state.untracked == ('notes dir/a b.py', 'scripts/bad.py')
        assert state.uncommitted == ('gone.py', 'keep.py', 'staged.py')

    async def test_index_removed_path_is_uncommitted_not_untracked(
        self, tmp_path: Path,
    ) -> None:
        repo = seed_repo(tmp_path / 'repo', _MIXED_SEED)
        _git(repo, 'rm', '--cached', '-q', 'keep.py')

        state = await read_working_tree_state(repo)

        assert state is not None
        assert state.untracked == ()
        assert state.uncommitted == ('keep.py',)

    async def test_non_repo_directory_returns_none(self, tmp_path: Path) -> None:
        plain = tmp_path / 'plain'
        plain.mkdir()

        assert await read_working_tree_state(plain) is None

    async def test_missing_directory_returns_none(self, tmp_path: Path) -> None:
        assert await read_working_tree_state(tmp_path / 'does-not-exist') is None


class TestWorkingTreeStateInvariant:
    def test_path_in_both_tuples_is_rejected_by_name(self) -> None:
        with pytest.raises(ValueError, match='a.py'):
            WorkingTreeState(untracked=('a.py',), uncommitted=('a.py',))

    @pytest.mark.parametrize('field', ['untracked', 'uncommitted'])
    def test_absolute_path_is_rejected_by_name(self, field: str) -> None:
        with pytest.raises(ValueError, match='/x.py'):
            WorkingTreeState(**{'untracked': (), 'uncommitted': (), field: ('/x.py',)})


_RUFF_LINT_OUTPUT = (
    'UP032 [*] Use f-string instead of `format` call\n'
    '  --> scripts/bad.py:39:13\n'
    '   |\n'
    '39 |     msg = "{}".format(x)\n'
    '   |\n'
    'Found 9 errors.\n'
)


def _failed(*, lint: str = '', test: str = '', type_: str = '') -> VerifyResult:
    return VerifyResult(
        passed=False, test_output=test, lint_output=lint, type_output=type_,
        summary='Failures: lint issues',
    )


def _passed(*, lint: str = '', test: str = '', type_: str = '') -> VerifyResult:
    return VerifyResult(
        passed=True, test_output=test, lint_output=lint, type_output=type_, summary='OK',
    )


def _has_number(text: str, number: int) -> bool:
    return re.search(rf'\b{number}\b', text) is not None


_ABS_ROOT = Path('/abs/root')


def _attribute(
    phase1: VerifyResult,
    state: WorkingTreeState,
    *,
    run_dirs: Iterable[str] = (),
    root: Path = _ABS_ROOT,
) -> Phase1WorkingTree:
    return attribute_phase1(phase1, state, root=root, run_dirs=run_dirs)


class TestAttributePhase1:
    def test_ruff_shaped_lint_output_names_untracked_path(self) -> None:
        state = WorkingTreeState(untracked=('scripts/bad.py',), uncommitted=())

        tree = _attribute(_failed(lint=_RUFF_LINT_OUTPUT), state)

        assert tree.state is state
        assert tree.named_in_failure == (NamedChange('scripts/bad.py', TreeChange.UNTRACKED),)

    def test_subproject_relative_form_is_named_only_with_its_run_dir(self) -> None:
        state = WorkingTreeState(untracked=(), uncommitted=('pkg/src/pkg/x.py',))
        phase1 = _failed(lint='src/pkg/x.py:3:1: F401 `os` imported but unused\n')

        with_dir = _attribute(phase1, state, run_dirs=['pkg'])
        without_dir = _attribute(phase1, state)

        assert with_dir.named_in_failure == (
            NamedChange('pkg/src/pkg/x.py', TreeChange.UNCOMMITTED),
        )
        assert without_dir.named_in_failure == ()

    def test_absolute_path_in_type_output_is_named(self) -> None:
        state = WorkingTreeState(untracked=('scripts/bad.py',), uncommitted=())
        phase1 = _failed(type_='/abs/root/scripts/bad.py:1:1 - error: bad type\n')

        tree = _attribute(phase1, state)

        assert tree.named_in_failure == (NamedChange('scripts/bad.py', TreeChange.UNTRACKED),)

    def test_path_inside_a_longer_file_name_is_not_named(self) -> None:
        state = WorkingTreeState(untracked=('foo.py',), uncommitted=())
        phase1 = _failed(test='FAILED tests/test_foo.py::test_it - AssertionError\n')

        assert _attribute(phase1, state).named_in_failure == ()

    def test_output_naming_only_committed_paths_names_nothing(self) -> None:
        state = WorkingTreeState(untracked=('scripts/bad.py',), uncommitted=('keep.py',))
        phase1 = _failed(lint='  --> scripts/committed.py:1:1\nFound 1 error.\n')

        assert _attribute(phase1, state).named_in_failure == ()

    def test_passing_result_names_nothing_even_when_output_mentions_dirty_path(self) -> None:
        state = WorkingTreeState(untracked=('scripts/bad.py',), uncommitted=())

        tree = _attribute(_passed(lint=_RUFF_LINT_OUTPUT), state)

        assert tree.named_in_failure == ()


_BAD_UNTRACKED = WorkingTreeState(untracked=('scripts/bad.py',), uncommitted=())
_BAD_NAMED = (NamedChange('scripts/bad.py', TreeChange.UNTRACKED),)


class TestAttributionPrecision:
    def test_committed_path_ending_in_dirty_paths_names_neither(self) -> None:
        state = WorkingTreeState(
            untracked=('orchestrator/conftest.py', 'tests/conftest.py'), uncommitted=(),
        )
        phase1 = _failed(lint='  --> orchestrator/tests/conftest.py:1:1\n')

        tree = _attribute(phase1, state, run_dirs=['orchestrator'])

        assert tree.named_in_failure == ()

    def test_repo_relative_suffix_of_a_longer_path_is_not_named(self) -> None:
        phase1 = _failed(lint='  --> tools/scripts/bad.py:1:1\n')

        assert _attribute(phase1, _BAD_UNTRACKED).named_in_failure == ()

    @pytest.mark.parametrize('output', [
        'scripts/bad.py:1:1: F401 x\n',
        'header\nscripts/bad.py:1:1\n',
        'FAILED scripts/bad.py::test_x - E\n',
        "E   'scripts/bad.py'\n",
        'E   "scripts/bad.py"\n',
        '(scripts/bad.py)\n',
        '  --> ./scripts/bad.py:1:1\n',
    ])
    def test_delimited_relative_form_is_named(self, output: str) -> None:
        assert _attribute(_failed(lint=output), _BAD_UNTRACKED).named_in_failure == _BAD_NAMED

    @pytest.mark.parametrize('path', ['notes dir/a b.py', 'pkg/a+b.py', 'pkg/naïve.py'])
    def test_path_with_characters_beyond_the_common_alphabet_is_named(self, path: str) -> None:
        state = WorkingTreeState(untracked=(path,), uncommitted=())

        tree = _attribute(_failed(lint=f'  --> {path}:1:1\n'), state)

        assert tree.named_in_failure == (NamedChange(path, TreeChange.UNTRACKED),)

    def test_absolute_form_under_a_root_with_a_space_is_named(self, tmp_path: Path) -> None:
        root = tmp_path / 'my repo'
        root.mkdir()
        phase1 = _failed(type_=f'{root}/scripts/bad.py:1:1 - error\n')

        assert _attribute(phase1, _BAD_UNTRACKED, root=root).named_in_failure == _BAD_NAMED

    def test_path_continuing_past_the_dirty_path_is_not_named(self) -> None:
        state = WorkingTreeState(untracked=('docs/notes',), uncommitted=())

        assert _attribute(_failed(lint='docs/notes/x.py:1:1\n'), state).named_in_failure == ()

    @pytest.mark.parametrize(('template', 'named'), [
        ('{root}/scripts/bad.py:1:1 - error', True),
        ('File "{root}/scripts/bad.py", line 3', True),
        ('/elsewhere/scripts/bad.py:1:1 - error', False),
        ('/x{root}/scripts/bad.py:1:1', False),
    ])
    def test_absolute_form_matches_only_the_checkout_root(
        self, tmp_path: Path, template: str, named: bool,
    ) -> None:
        root = tmp_path / 'repo'
        root.mkdir()
        phase1 = _failed(type_=template.format(root=root))

        tree = _attribute(phase1, _BAD_UNTRACKED, root=root)

        assert tree.named_in_failure == (_BAD_NAMED if named else ())

    def test_absolute_form_under_the_resolved_root_is_named(self, tmp_path: Path) -> None:
        root = tmp_path / 'repo'
        root.mkdir()
        link = tmp_path / 'link'
        link.symlink_to(root, target_is_directory=True)
        phase1 = _failed(type_=f'{root.resolve()}/scripts/bad.py:1:1 - error\n')

        assert _attribute(phase1, _BAD_UNTRACKED, root=link).named_in_failure == _BAD_NAMED

    def test_module_relative_form_another_module_also_has_is_not_named(
        self, tmp_path: Path,
    ) -> None:
        _write(tmp_path, 'orchestrator/tests/conftest.py', 'a = 1\n')
        _write(tmp_path, 'fused-memory/tests/conftest.py', 'b = 1\n')
        state = WorkingTreeState(untracked=('orchestrator/tests/conftest.py',), uncommitted=())
        phase1 = _failed(lint='tests/conftest.py:1:1: F401 x\n')
        run_dirs = ['orchestrator', 'fused-memory']

        ambiguous = _attribute(phase1, state, run_dirs=run_dirs, root=tmp_path)
        (tmp_path / 'fused-memory' / 'tests' / 'conftest.py').unlink()
        unambiguous = _attribute(phase1, state, run_dirs=run_dirs, root=tmp_path)

        assert ambiguous.named_in_failure == ()
        assert unambiguous.named_in_failure == (
            NamedChange('orchestrator/tests/conftest.py', TreeChange.UNTRACKED),
        )

    def test_root_relative_form_a_module_also_has_is_not_named(self, tmp_path: Path) -> None:
        _write(tmp_path, 'tests/conftest.py', 'a = 1\n')
        _write(tmp_path, 'orchestrator/tests/conftest.py', 'b = 1\n')
        state = WorkingTreeState(untracked=('tests/conftest.py',), uncommitted=())
        phase1 = _failed(lint='tests/conftest.py:1:1\n')

        tree = _attribute(phase1, state, run_dirs=['orchestrator'], root=tmp_path)

        assert tree.named_in_failure == ()


class TestPhase1WorkingTreeInvariant:
    def test_named_path_absent_from_state_is_rejected_by_name(self) -> None:
        state = WorkingTreeState(untracked=('a.py',), uncommitted=())

        with pytest.raises(ValueError, match='nope.py'):
            Phase1WorkingTree(
                state, named_in_failure=(NamedChange('nope.py', TreeChange.UNTRACKED),),
            )

    def test_named_path_with_wrong_kind_is_rejected_by_name(self) -> None:
        state = WorkingTreeState(untracked=('a.py',), uncommitted=())

        with pytest.raises(ValueError, match='a.py'):
            Phase1WorkingTree(
                state, named_in_failure=(NamedChange('a.py', TreeChange.UNCOMMITTED),),
            )


def _unqualified_cases() -> list[tuple[VerifyResult, Phase1WorkingTree | None]]:
    clean = WorkingTreeState(untracked=(), uncommitted=())
    return [
        (_failed(lint=_RUFF_LINT_OUTPUT), None),
        (_failed(lint=_RUFF_LINT_OUTPUT), Phase1WorkingTree(clean, ())),
        (_passed(), None),
        (_passed(), Phase1WorkingTree(clean, ())),
        (_passed(), Phase1WorkingTree(_BAD_UNTRACKED, ())),
    ]


class TestQualifiedPhase1Summary:
    @pytest.mark.parametrize(('phase1', 'tree'), _unqualified_cases())
    def test_unqualified_cases_return_bare_summary(
        self, phase1: VerifyResult, tree: Phase1WorkingTree | None,
    ) -> None:
        assert qualified_phase1_summary(phase1, tree) == phase1.summary

    def test_failure_naming_untracked_path_is_qualified_with_it(self) -> None:
        phase1 = _failed(lint=_RUFF_LINT_OUTPUT)
        tree = _attribute(phase1, _BAD_UNTRACKED)

        summary = qualified_phase1_summary(phase1, tree)

        assert summary.startswith(phase1.summary)
        assert summary != phase1.summary
        assert 'scripts/bad.py' in summary
        assert 'untracked' in summary

    def test_failure_naming_nothing_dirty_states_the_counts(self) -> None:
        phase1 = _failed(lint='  --> scripts/committed.py:1:1\n')
        state = WorkingTreeState(
            untracked=tuple(f'u{i}.py' for i in range(7)),
            uncommitted=tuple(f'c{i}.py' for i in range(4)),
        )

        summary = qualified_phase1_summary(phase1, _attribute(phase1, state))

        assert summary != phase1.summary
        assert _has_number(summary, 7)
        assert _has_number(summary, 4)

    def test_pass_with_uncommitted_edits_states_their_count(self) -> None:
        phase1 = _passed()
        state = WorkingTreeState(untracked=(), uncommitted=('a.py', 'b.py', 'c.py'))

        summary = qualified_phase1_summary(phase1, _attribute(phase1, state))

        assert summary != phase1.summary
        assert _has_number(summary, 3)


class TestRenderWorkingTreeSection:
    @pytest.mark.parametrize(('phase1', 'tree'), _unqualified_cases())
    def test_unqualified_cases_render_nothing(
        self, phase1: VerifyResult, tree: Phase1WorkingTree | None,
    ) -> None:
        assert render_working_tree_section(phase1, tree) == ''

    def test_failure_lists_each_named_path_with_its_kind(self) -> None:
        state = WorkingTreeState(untracked=('scripts/bad.py',), uncommitted=('keep.py',))
        phase1 = _failed(lint=_RUFF_LINT_OUTPUT + '  --> keep.py:2:1\n')
        tree = _attribute(phase1, state)
        assert len(tree.named_in_failure) == 2

        lines = render_working_tree_section(phase1, tree).splitlines()

        for named in tree.named_in_failure:
            assert any(named.path in line and named.change.value in line for line in lines)

    def test_untracked_file_named_in_failure_is_not_to_be_reported(self) -> None:
        phase1 = _failed(lint=_RUFF_LINT_OUTPUT)

        section = render_working_tree_section(phase1, _attribute(phase1, _BAD_UNTRACKED))

        assert 'do NOT report' in section

    def test_uncommitted_file_named_in_failure_is_checked_against_head_not_suppressed(
        self,
    ) -> None:
        state = WorkingTreeState(untracked=(), uncommitted=('keep.py',))
        phase1 = _failed(lint='  --> keep.py:2:1\n')
        tree = _attribute(phase1, state)
        assert tree.named_in_failure == (NamedChange('keep.py', TreeChange.UNCOMMITTED),)

        section = render_working_tree_section(phase1, tree)

        assert 'do NOT report' not in section
        assert 'git show HEAD:' in section

    def test_named_paths_beyond_the_cap_are_counted_not_listed(self) -> None:
        paths = tuple(f'pkg/file_{i:02d}.py' for i in range(30))
        state = WorkingTreeState(untracked=paths, uncommitted=())
        phase1 = _failed(lint=''.join(f'  --> {p}:1:1\n' for p in paths))
        tree = _attribute(phase1, state)
        assert len(tree.named_in_failure) == 30

        section = render_working_tree_section(phase1, tree)

        assert sum(path in section for path in paths) == SECTION_NAMED_PATH_CAP
        assert _has_number(section, 30 - SECTION_NAMED_PATH_CAP)


class TestPhase1WorkingTreeRecord:
    def test_to_dict_is_json_native_counts_plus_named_paths(self) -> None:
        state = WorkingTreeState(untracked=('scripts/bad.py', 'u.py'), uncommitted=('keep.py',))
        phase1 = _failed(lint=_RUFF_LINT_OUTPUT)
        tree = _attribute(phase1, state)

        record = tree.to_dict()

        assert json.loads(json.dumps(record)) == record
        assert record == {
            'untracked_count': 2,
            'uncommitted_count': 1,
            'named_in_failure': [{'path': 'scripts/bad.py', 'change': 'untracked'}],
        }
