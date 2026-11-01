"""Contract of ``_tree_scan``: one streaming parse per file, facts not trees.

Every case runs over a ``tmp_path`` tree with synthetic analyses; none touches
the real repository, so this module is not a consumer of the shared pass.
"""
from __future__ import annotations

import ast
import enum
import gc
import re
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NamedTuple

import pytest
from _git_fixtures import RepoSeed, seed_repo
from _orch_helpers import git_env_with_ceiling
from _tree_scan import (
    TREE_PASS_FIXTURE,
    ParsedModule,
    TreeAnalysis,
    TreePass,
    analyses_requested_by,
    checkout_python_files,
    git_tracked_python_files,
    python_files,
)


def _write(root: Path, rel: str, text: str = 'x = 1\n') -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path.resolve()


def _analysis(name: str, files, extract) -> TreeAnalysis:
    return TreeAnalysis(name=name, files=files, extract=extract)


def _path_name(module: ParsedModule) -> str:
    return module.path.name


class TestOneParsePerFile:
    def test_overlapping_analyses_share_one_tree_per_file(self, tmp_path: Path) -> None:
        _write(tmp_path, 'a/one.py')
        _write(tmp_path, 'a/two.py')
        _write(tmp_path, 'top.py')
        seen: list[tuple[str, Path, object]] = []

        def recorder(name: str):
            def extract(module: ParsedModule) -> str:
                seen.append((name, module.path, module.tree))
                return name
            return extract

        narrow = _analysis('narrow', lambda: python_files(tmp_path / 'a'), recorder('narrow'))
        wide = _analysis('wide', lambda: python_files(tmp_path), recorder('wide'))

        pass_ = TreePass.run([narrow, wide])

        shared = set(python_files(tmp_path / 'a'))
        for path in shared:
            trees = [tree for _name, seen_path, tree in seen if seen_path == path]
            assert len(trees) == 2
            assert trees[0] is trees[1], f'{path} was parsed twice'
        assert set(pass_.facts(narrow)) == shared
        assert set(pass_.facts(wide)) == set(python_files(tmp_path))


class TestStreaming:
    def test_at_most_one_tree_is_ever_live(self, tmp_path: Path) -> None:
        for index in range(4):
            _write(tmp_path, f'm{index}.py')
        refs: list[weakref.ref] = []
        alive_at_each_call: list[int] = []

        def extract(module: ParsedModule) -> int:
            alive_at_each_call.append(sum(ref() is not None for ref in refs))
            refs.append(weakref.ref(module.tree))
            return len(module.tree.body)

        TreePass.run([_analysis('streaming', lambda: python_files(tmp_path), extract)])
        gc.collect()

        assert alive_at_each_call == [0, 0, 0, 0]
        assert [ref() for ref in refs] == [None, None, None, None]


class TestLazyParse:
    def test_a_file_no_analysis_touches_as_a_tree_is_never_parsed(self, tmp_path: Path) -> None:
        bad = _write(tmp_path, 'bad.py', 'def broken(:\n')
        source_only = _analysis(
            'source_only', lambda: python_files(tmp_path), lambda m: len(m.source),
        )

        facts = TreePass.run([source_only]).facts(source_only)

        assert dict(facts) == {bad: len('def broken(:\n')}

    def test_only_an_analysis_that_touches_the_tree_pays_for_a_syntax_error(
        self, tmp_path: Path,
    ) -> None:
        bad = _write(tmp_path, 'bad.py', 'def broken(:\n')
        good = _write(tmp_path, 'good.py')
        source_only = _analysis(
            'source_only', lambda: python_files(tmp_path), lambda m: len(m.source),
        )
        needs_tree = _analysis(
            'needs_tree', lambda: python_files(tmp_path), lambda m: len(m.tree.body),
        )

        pass_ = TreePass.run([source_only, needs_tree])

        assert set(pass_.facts(source_only)) == {bad, good}
        with pytest.raises(AssertionError) as raised:
            pass_.facts(needs_tree)
        assert 'needs_tree' in str(raised.value)
        assert str(bad) in str(raised.value)
        assert 'SyntaxError' in str(raised.value)
        assert str(good) not in str(raised.value)


class TestMayName:
    """``ParsedModule.may_name``: the sound substring prefilter an extractor runs before its walk."""

    def _module(self, tmp_path: Path, text: str) -> ParsedModule:
        return ParsedModule(_write(tmp_path, 'm.py', text))

    def test_a_fragment_of_an_identifier_is_found(self, tmp_path: Path) -> None:
        module = self._module(tmp_path, "files = root.rglob('*.py')\n")

        assert module.may_name('glob')
        assert module.may_name('absent', 'glob')
        assert not module.may_name('absent', 'missing')

    def test_a_compatibility_spelling_counts_because_the_parser_normalizes_it(
        self, tmp_path: Path,
    ) -> None:
        module = self._module(tmp_path, 'files = root.ｒglob()\n')

        assert 'rglob' in {node.attr for node in ast.walk(module.tree) if isinstance(node, ast.Attribute)}
        assert module.may_name('rglob')

    def test_ignore_case_matches_a_name_in_any_case(self, tmp_path: Path) -> None:
        module = self._module(tmp_path, "_Review_PROJECT_ROOT = '/tmp/x'\n")

        assert not module.may_name('project_root')
        assert module.may_name('project_root', ignore_case=True)

    def test_it_reads_the_source_and_never_parses(self, tmp_path: Path) -> None:
        module = self._module(tmp_path, 'def broken(:\n    glob\n')

        assert module.may_name('glob')
        assert not module.may_name('absent')
        with pytest.raises(SyntaxError):
            _ = module.tree


class TestFailureIsolation:
    def test_a_raising_extractor_fails_only_its_own_analysis(self, tmp_path: Path) -> None:
        target = _write(tmp_path, 'target.py')
        _write(tmp_path, 'other.py')

        def explode_on_target(module: ParsedModule) -> str:
            if module.path == target:
                raise ValueError('boom on purpose')
            return module.path.name

        raising = _analysis('raising', lambda: python_files(tmp_path), explode_on_target)
        bystander = _analysis('bystander', lambda: python_files(tmp_path), _path_name)

        pass_ = TreePass.run([raising, bystander])

        with pytest.raises(AssertionError) as raised:
            pass_.facts(raising)
        message = str(raised.value)
        for expected in ('raising', str(target), 'ValueError', 'boom on purpose'):
            assert expected in message
        assert len(pass_.facts(bystander)) == 2

    def test_an_analysis_that_cannot_list_its_files_fails_only_itself(
        self, tmp_path: Path,
    ) -> None:
        _write(tmp_path, 'present.py')

        def unlistable() -> tuple[Path, ...]:
            raise NotADirectoryError('no such root on purpose')

        broken = _analysis('broken', unlistable, _path_name)
        bystander = _analysis('bystander', lambda: python_files(tmp_path), _path_name)

        pass_ = TreePass.run([broken, bystander])

        with pytest.raises(AssertionError) as raised:
            pass_.facts(broken)
        message = str(raised.value)
        for expected in ('broken', 'NotADirectoryError', 'no such root on purpose'):
            assert expected in message
        assert list(pass_.facts(bystander).values()) == ['present.py']

    def test_a_non_utf8_source_fails_only_the_analysis_that_reads_it(
        self, tmp_path: Path,
    ) -> None:
        latin = tmp_path / 'latin.py'
        latin.write_bytes(b'name = "\xff\xfe"\n')
        reader = _analysis('reader', lambda: python_files(tmp_path), lambda m: m.source)
        namer = _analysis('namer', lambda: python_files(tmp_path), _path_name)

        pass_ = TreePass.run([reader, namer])

        with pytest.raises(AssertionError, match='UnicodeDecodeError'):
            pass_.facts(reader)
        assert dict(pass_.facts(namer)) == {latin.resolve(): 'latin.py'}


class _Point(NamedTuple):
    line: int
    text: str


class _Shade(enum.Enum):
    DARK = 'dark'


@dataclass(frozen=True)
class _FrozenFact:
    path: Path
    lines: tuple[int, ...]
    names: frozenset[str]


@dataclass
class _MutableFact:
    lines: tuple[int, ...]


class TestFactInvariant:
    @pytest.mark.parametrize(
        ('extract', 'offending_type'),
        [
            pytest.param(lambda m: m.tree, 'Module', id='the-tree'),
            pytest.param(lambda m: (1, m.tree.body[0]), 'Assign', id='tuple-holding-a-node'),
            pytest.param(lambda m: [1], 'list', id='list'),
            pytest.param(lambda m: {'a': 1}, 'dict', id='dict'),
            pytest.param(lambda m: {1}, 'set', id='set'),
            pytest.param(lambda m: _MutableFact((1,)), '_MutableFact', id='mutable-dataclass'),
            pytest.param(lambda m: _FrozenFact(m.path, (1,), frozenset({m.tree})), 'Module',
                         id='frozen-dataclass-holding-a-node'),
        ],
    )
    def test_a_mutable_or_ast_bearing_fact_is_a_failure(
        self, tmp_path: Path, extract, offending_type: str,
    ) -> None:
        _write(tmp_path, 'mod.py')
        analysis = _analysis('leaky', lambda: python_files(tmp_path), extract)

        pass_ = TreePass.run([analysis])

        with pytest.raises(AssertionError) as raised:
            pass_.facts(analysis)
        message = str(raised.value)
        assert 'leaky' in message
        assert f'holding a {offending_type};' in message

    @pytest.mark.parametrize(
        'extract',
        [
            pytest.param(lambda m: frozenset({1, 'a'}), id='frozenset'),
            pytest.param(lambda m: (1, 'a', (2.5, None)), id='nested-tuple'),
            pytest.param(lambda m: 'text', id='str'),
            pytest.param(lambda m: b'bytes', id='bytes'),
            pytest.param(lambda m: 7, id='int'),
            pytest.param(lambda m: 7.5, id='float'),
            pytest.param(lambda m: True, id='bool'),
            pytest.param(lambda m: None, id='none'),
            pytest.param(lambda m: m.path, id='path'),
            pytest.param(lambda m: _Shade.DARK, id='enum-member'),
            pytest.param(lambda m: (_Point(1, 'x'),), id='namedtuple'),
            pytest.param(lambda m: _FrozenFact(m.path, (1, 2), frozenset({'f'})),
                         id='frozen-dataclass'),
        ],
    )
    def test_an_immutable_plain_fact_is_kept(self, tmp_path: Path, extract) -> None:
        path = _write(tmp_path, 'mod.py')
        analysis = _analysis('plain', lambda: python_files(tmp_path), extract)

        facts = TreePass.run([analysis]).facts(analysis)

        assert list(facts) == [path]


class TestKeyingAndOrder:
    def test_facts_are_a_read_only_mapping_keyed_by_normalized_path_in_sorted_order(
        self, tmp_path: Path,
    ) -> None:
        zeta = _write(tmp_path, 'zeta.py')
        alpha = _write(tmp_path, 'alpha.py')
        unresolved = (tmp_path / 'sub' / '..' / 'zeta.py', tmp_path / 'alpha.py')
        (tmp_path / 'sub').mkdir()
        analysis = _analysis('keyed', lambda: unresolved, _path_name)

        facts = TreePass.run([analysis]).facts(analysis)

        assert list(facts) == [alpha, zeta]
        assert not hasattr(facts, '__setitem__')

    def test_a_symlinked_file_is_keyed_at_its_link_inside_its_root(self, tmp_path: Path) -> None:
        root = (tmp_path / 'root').resolve()
        target = _write(tmp_path, 'outside/target.py')
        link = root / 'link.py'
        root.mkdir()
        link.symlink_to(target)
        analysis = _analysis('linked', lambda: python_files(root), _path_name)

        facts = TreePass.run([analysis]).facts(analysis)

        assert python_files(root) == (link,)
        assert dict(facts) == {link: 'link.py'}
        assert [path.relative_to(root) for path in facts] == [Path('link.py')]


class TestDeclaration:
    def test_an_undeclared_analysis_names_tree_analyses(self, tmp_path: Path) -> None:
        _write(tmp_path, 'mod.py')
        declared = _analysis('declared', lambda: python_files(tmp_path), _path_name)
        undeclared = _analysis('undeclared', lambda: python_files(tmp_path), _path_name)

        pass_ = TreePass.run([declared])

        with pytest.raises(LookupError, match=r"'undeclared'.*TREE_ANALYSES"):
            pass_.facts(undeclared)

    def test_two_analyses_sharing_a_name_are_refused(self, tmp_path: Path) -> None:
        first = _analysis('same', lambda: python_files(tmp_path), _path_name)
        second = _analysis('same', lambda: python_files(tmp_path), _path_name)

        with pytest.raises(ValueError, match="'same'"):
            TreePass.run([first, second])


def _module_declaring(*analyses: TreeAnalysis) -> ModuleType:
    module = ModuleType('consumer')
    vars(module)['TREE_ANALYSES'] = analyses
    return module


class TestDiscovery:
    def test_only_modules_with_a_requesting_item_contribute(self, tmp_path: Path) -> None:
        wanted = _analysis('wanted', lambda: (), _path_name)
        ignored = _analysis('ignored', lambda: (), _path_name)
        items = [
            SimpleNamespace(module=_module_declaring(wanted), fixturenames=('tmp_path', TREE_PASS_FIXTURE)),
            SimpleNamespace(module=_module_declaring(ignored), fixturenames=('tmp_path',)),
        ]

        assert analyses_requested_by(items) == (wanted,)

    def test_dedupes_by_identity_and_keeps_first_seen_order(self) -> None:
        first = _analysis('first', lambda: (), _path_name)
        second = _analysis('second', lambda: (), _path_name)
        lookalike = _analysis('first', lambda: (), _path_name)
        module_a = _module_declaring(first, second)
        module_b = _module_declaring(second, lookalike)
        items = [
            SimpleNamespace(module=module_a, fixturenames=(TREE_PASS_FIXTURE,)),
            SimpleNamespace(module=module_a, fixturenames=(TREE_PASS_FIXTURE,)),
            SimpleNamespace(module=module_b, fixturenames=(TREE_PASS_FIXTURE,)),
        ]

        assert analyses_requested_by(items) == (first, second, lookalike)

    def test_items_without_a_module_are_ignored(self) -> None:
        items = [SimpleNamespace(fixturenames=(TREE_PASS_FIXTURE,))]

        assert analyses_requested_by(items) == ()


class TestPythonFiles:
    def test_returns_a_sorted_tuple_of_python_files_recursively(
        self, tmp_path: Path,
    ) -> None:
        deep = _write(tmp_path, 'pkg/deep/inner.py')
        top = _write(tmp_path, 'top.py')
        (tmp_path / 'notes.txt').write_text('not python\n')

        found = python_files(tmp_path)

        assert isinstance(found, tuple)
        assert found == tuple(sorted((deep, top)))

    def test_exclude_parts_are_matched_relative_to_each_root(self, tmp_path: Path) -> None:
        root = tmp_path / 'skip' / 'root'
        kept = _write(root, 'kept.py')
        _write(root, 'skip/hidden.py')
        _write(root, '.git/hooks/hook.py')

        assert python_files(root, exclude_parts=frozenset({'skip', '.git'})) == (kept,)

    def test_overlapping_roots_are_deduplicated(self, tmp_path: Path) -> None:
        inner = _write(tmp_path, 'a/inner.py')
        outer = _write(tmp_path, 'outer.py')

        assert python_files(tmp_path / 'a', tmp_path) == tuple(sorted((inner, outer)))

    @pytest.mark.parametrize('kind', ['missing', 'a-file'])
    def test_a_root_that_is_not_an_existing_directory_raises(
        self, tmp_path: Path, kind: str,
    ) -> None:
        root = tmp_path / 'nowhere'
        if kind == 'a-file':
            root = _write(tmp_path, 'plain.py')

        with pytest.raises(NotADirectoryError, match=re.escape(str(root))):
            python_files(root)


_CHECKOUT_SEED = RepoSeed(
    files=(
        ('README.md', '# checkout\n'),
        ('pkg/src/pkg/tracked.py', 'x = 1\n'),
        ('skills/tool/scripts/deep.py', 'x = 1\n'),
    ),
    message='seed a checkout',
)


def _seed_checkout(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    seed_repo(repo, _CHECKOUT_SEED)
    monkeypatch.setenv(
        'GIT_CEILING_DIRECTORIES', git_env_with_ceiling(repo)['GIT_CEILING_DIRECTORIES'],
    )
    return repo.resolve()


class TestCheckoutPythonFiles:
    def test_unions_git_tracked_files_with_on_disk_package_roots(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _seed_checkout(tmp_path / 'repo', monkeypatch)
        written = {
            _write(repo, rel)
            for rel in (
                'pkg/src/pkg/new.py',
                'pkg/tests/test_new.py',
                'scripts/sub/tool.py',
                'hooks/hook.py',
                'top_level.py',
            )
        }
        _write(repo, 'docs/untracked_elsewhere.py')

        found = checkout_python_files(repo)

        assert isinstance(found, tuple)
        assert found == tuple(sorted({
            repo / 'pkg/src/pkg/tracked.py',
            repo / 'skills/tool/scripts/deep.py',
            *written,
        }))

    def test_skip_parts_are_judged_relative_to_the_checkout_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _seed_checkout(tmp_path / '.worktrees' / 'task', monkeypatch)
        kept = _write(repo, 'pkg/src/pkg/kept.py')
        for rel in (
            'pkg/src/pkg/.venv/lib.py',
            'pkg/tests/node_modules/module.py',
            'scripts/__pycache__/cached.py',
            'hooks/.worktrees/copy.py',
        ):
            _write(repo, rel)

        found = checkout_python_files(repo)

        assert kept in found
        assert [
            path for path in found
            if {'.venv', 'node_modules', '__pycache__', '.worktrees'} & set(path.relative_to(repo).parts)
        ] == []

    def test_a_tracked_file_deleted_from_disk_is_not_returned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _seed_checkout(tmp_path / 'repo', monkeypatch)
        (repo / 'skills/tool/scripts/deep.py').unlink()

        assert checkout_python_files(repo) == (repo / 'pkg/src/pkg/tracked.py',)

    def test_without_a_git_binary_it_returns_the_on_disk_set_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _seed_checkout(tmp_path / 'repo', monkeypatch)
        empty_bin = tmp_path / 'empty-bin'
        empty_bin.mkdir()
        monkeypatch.setenv('PATH', str(empty_bin))

        assert checkout_python_files(repo) == (repo / 'pkg/src/pkg/tracked.py',)

    def test_outside_a_repository_it_returns_the_on_disk_set_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plain = (tmp_path / 'plain').resolve()
        on_disk = _write(plain, 'pkg/src/pkg/on_disk.py')
        _write(plain, 'skills/tool/scripts/deep.py')
        monkeypatch.setenv(
            'GIT_CEILING_DIRECTORIES', git_env_with_ceiling(plain)['GIT_CEILING_DIRECTORIES'],
        )

        assert checkout_python_files(plain) == (on_disk,)


class TestGitTrackedPythonFiles:
    def test_lists_only_the_tracked_python_files_under_the_resolved_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        repo = _seed_checkout(tmp_path / 'repo', monkeypatch)
        _write(repo, 'pkg/src/pkg/untracked.py')

        assert sorted(git_tracked_python_files(repo)) == [
            repo / 'pkg/src/pkg/tracked.py',
            repo / 'skills/tool/scripts/deep.py',
        ]

    def test_is_empty_outside_a_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plain = (tmp_path / 'plain').resolve()
        _write(plain, 'pkg/src/pkg/on_disk.py')
        monkeypatch.setenv(
            'GIT_CEILING_DIRECTORIES', git_env_with_ceiling(plain)['GIT_CEILING_DIRECTORIES'],
        )

        assert git_tracked_python_files(plain) == ()
