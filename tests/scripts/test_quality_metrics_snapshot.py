"""Tests of ``scripts/quality_metrics_snapshot.py``, the whole-repo metrics report.

Every fixture is a small REAL git repository whose root pyproject.toml names
two members, measured through the script's public entry points; no measure is
patched. The one real-tree test (cluster agreement) is the PRD's row 12.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest
import quality_metrics_snapshot as snapshot
import source_measures
from git_listing import git

REPO_ROOT = Path(__file__).parents[2]

_PYPROJECT = '[tool.uv.workspace]\nmembers = ["alpha", "beta"]\n'

#: The row-5 'before' source: complexipy 6.2 scores f at 2, and the file at 2.
_MOD_BEFORE = (
    '"""The row-5 fixture."""\n'
    '# f holds two sequential ifs.\n'
    'def f(a, b):\n'
    '    if a:\n'
    '        x = 1\n'
    '    if b:\n'
    '        x = 2\n'
    '    return 0\n'
)

_TRIVIAL_TEST = 'def test_x():\n    assert True\n'

_BASE: dict[str, str] = {
    'alpha/src/alpha/__init__.py': '',
    'alpha/src/alpha/mod.py': _MOD_BEFORE,
    'alpha/tests/test_mod.py': _TRIVIAL_TEST,
    'beta/src/beta/b.py': 'B = 1\n',
    'beta/tests/test_b.py': _TRIVIAL_TEST,
    'scripts/x.py': 'X = 1\n',
    'scripts/tests/test_x.py': _TRIVIAL_TEST,
    'tests/test_y.py': _TRIVIAL_TEST,
    'hooks/h.py': 'H = 1\n',
}

_BASE_DOMAIN = frozenset(path for path in _BASE if not path.startswith('hooks/'))

_IDENTITY = ('-c', 'user.name=fixture', '-c', 'user.email=fixture@example.invalid')


def _write(root: Path, files: dict[str, str]) -> None:
    for relpath, text in files.items():
        target = root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8')


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """A committed repo under *tmp_path*; a 'pyproject.toml' in *files* overrides the default."""
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    _write(root, {'pyproject.toml': _PYPROJECT, **files})
    git(root, 'add', '-A')
    git(root, *_IDENTITY, 'commit', '--no-verify', '-q', '-m', 'base')
    return root


def _commit(
    root: Path,
    message: str,
    *,
    write: dict[str, str] | None = None,
    remove: Iterable[str] = (),
    moves: Iterable[tuple[str, str]] = (),
) -> None:
    _write(root, write or {})
    for relpath in remove:
        git(root, 'rm', '-q', relpath)
    for source, destination in moves:
        git(root, 'mv', source, destination)
    git(root, 'add', '-A')
    git(root, *_IDENTITY, 'commit', '--no-verify', '--allow-empty', '-q', '-m', message)


def _run(root: Path, out: Path, *extra: str) -> int:
    return snapshot.main(['--run-id', 'run-1', '--out', str(out), '--root', str(root), *extra])


def _measured(tmp_path: Path, files: dict[str, str]) -> dict[str, Any]:
    root = _repo(tmp_path, files)
    out = tmp_path / 'out' / 'snapshot.json'
    assert _run(root, out) == 0
    return json.loads(out.read_text(encoding='utf-8'))


# ---------------------------------------------------------------------------
# The measuring run: domain, header and file records.


class TestTheDomain:
    def test_exactly_the_tracked_member_files_are_measured(self, tmp_path: Path) -> None:
        root = _repo(tmp_path, _BASE)
        _write(root, {'alpha/src/alpha/untracked.py': 'U = 1\n'})
        out = tmp_path / 'out' / 'snapshot.json'
        assert _run(root, out) == 0
        files = json.loads(out.read_text(encoding='utf-8'))['files']
        assert set(files) == _BASE_DOMAIN
        assert {path: (record['member'], record['kind']) for path, record in files.items()} == {
            'alpha/src/alpha/__init__.py': ('alpha', 'src'),
            'alpha/src/alpha/mod.py': ('alpha', 'src'),
            'alpha/tests/test_mod.py': ('alpha', 'tests'),
            'beta/src/beta/b.py': ('beta', 'src'),
            'beta/tests/test_b.py': ('beta', 'tests'),
            'scripts/x.py': ('scripts', 'src'),
            'scripts/tests/test_x.py': ('scripts', 'tests'),
            'tests/test_y.py': ('tests', 'tests'),
        }

    def test_members_are_reported_in_declared_order_with_their_file_counts(
        self, tmp_path: Path
    ) -> None:
        measured = _measured(tmp_path, _BASE)
        assert measured['evidence']['members'] == [
            {'name': 'alpha', 'pseudo': False, 'domain_files': 3},
            {'name': 'beta', 'pseudo': False, 'domain_files': 2},
            {'name': 'scripts', 'pseudo': True, 'domain_files': 2},
            {'name': 'tests', 'pseudo': True, 'domain_files': 1},
        ]


class TestTheHeader:
    def test_top_level_keys_in_contract_order(self, tmp_path: Path) -> None:
        assert list(_measured(tmp_path, _BASE)) == [
            'schema_version', 'instrument', 'run_id', 'as_of_sha', 'since',
            'evidence', 'cost', 'params', 'files', 'functions', 'import_graph',
        ]

    def test_header_values(self, tmp_path: Path) -> None:
        root = _repo(tmp_path, _BASE)
        out = tmp_path / 'out' / 'snapshot.json'
        assert _run(root, out) == 0
        measured = json.loads(out.read_text(encoding='utf-8'))
        assert measured['schema_version'] == 2
        assert measured['instrument'] == 'quality-metrics-snapshot'
        assert measured['run_id'] == 'run-1'
        assert measured['as_of_sha'] == git(root, 'rev-parse', 'HEAD').strip()
        assert measured['since'] == 'none'
        evidence = measured['evidence']
        assert evidence['domain_files'] == evidence['measured_files'] == 8
        assert evidence['unreadable'] == []
        assert evidence['complete'] is True
        wall_clock = measured['cost']['wall_clock_s']
        assert isinstance(wall_clock, float) and wall_clock >= 0
        assert measured['params'] == {
            'complexipy_version': source_measures.complexipy_version(),
            'h14_soft_ceiling_lines': 1500,
            'h14_alarm_lines': 2000,
        }


class TestFileRecords:
    def test_a_src_record_carries_the_shared_measures(self, tmp_path: Path) -> None:
        root = _repo(tmp_path, _BASE)
        out = tmp_path / 'out' / 'snapshot.json'
        assert _run(root, out) == 0
        files = json.loads(out.read_text(encoding='utf-8'))['files']
        path = 'alpha/src/alpha/mod.py'
        record = files[path]
        size = source_measures.file_size_measures(_MOD_BEFORE, path=path)
        assert (record['lines'], record['prose_lines']) == (size.lines, size.prose_lines)
        assert record['prose_ratio'] == round(size.prose_lines / size.lines, 4)
        assert (
            record['cognitive_total'],
            record['cognitive_max'],
            record['cognitive_max_function'],
            record['functions'],
        ) == (2, 2, 'f', 1)
        assert record['blob'] == git(root, 'hash-object', path).strip()
        assert (record['module'], record['package_init']) == ('alpha.mod', False)
        assert record['function_local_imports'] == source_measures.function_local_imports(
            _MOD_BEFORE, path=path
        )
        assert record['reexport_names'] == sorted(
            source_measures.reexport_names(_MOD_BEFORE, path=path)
        )
        package = files['alpha/src/alpha/__init__.py']
        assert (package['module'], package['package_init']) == ('alpha', True)

    def test_a_module_with_no_functions_has_an_empty_max(self, tmp_path: Path) -> None:
        record = _measured(tmp_path, _BASE)['files']['beta/src/beta/b.py']
        assert (
            record['cognitive_max'], record['cognitive_max_function'], record['functions']
        ) == (0, None, 0)

    def test_an_empty_file_has_no_prose_ratio(self, tmp_path: Path) -> None:
        files = _measured(tmp_path, {**_BASE, 'alpha/src/alpha/empty.py': ''})['files']
        assert files['alpha/src/alpha/empty.py']['lines'] == 0
        assert files['alpha/src/alpha/empty.py']['prose_ratio'] is None

    def test_a_tie_at_the_max_names_the_smallest_qualname(self, tmp_path: Path) -> None:
        tie = (
            'def b(x):\n    if x:\n        return 1\n    return 0\n\n\n'
            'def a(x):\n    if x:\n        return 1\n    return 0\n'
        )
        record = _measured(tmp_path, {**_BASE, 'alpha/src/alpha/tie.py': tie})['files'][
            'alpha/src/alpha/tie.py'
        ]
        assert (record['cognitive_max'], record['cognitive_max_function']) == (1, 'a')


class TestTheFunctionsMap:
    def test_only_src_functions_are_listed_by_path_and_qualname(self, tmp_path: Path) -> None:
        # Every test file in the fixture defines test_x; none of them is listed.
        assert _measured(tmp_path, _BASE)['functions'] == {'alpha/src/alpha/mod.py::f': 2}

    def test_files_and_functions_are_in_sorted_key_order(self, tmp_path: Path) -> None:
        extra = {'alpha/src/alpha/a_first.py': 'def g():\n    return 1\n'}
        measured = _measured(tmp_path, {**_BASE, **extra})
        assert list(measured['files']) == sorted(measured['files'])
        assert list(measured['functions']) == sorted(measured['functions'])
        assert len(measured['functions']) == 2


class TestRendering:
    def test_rendering_a_parsed_snapshot_reproduces_its_text(self, tmp_path: Path) -> None:
        root = _repo(tmp_path, _BASE)
        out = tmp_path / 'out' / 'snapshot.json'
        assert _run(root, out) == 0
        text = out.read_text(encoding='utf-8')
        assert snapshot.render_snapshot(json.loads(text)) == text

    def test_each_file_and_function_is_one_line(self, tmp_path: Path) -> None:
        root = _repo(tmp_path, _BASE)
        out = tmp_path / 'out' / 'snapshot.json'
        assert _run(root, out) == 0
        text = out.read_text(encoding='utf-8')
        measured = json.loads(text)
        lines = text.splitlines()
        for key in [*measured['files'], *measured['functions']]:
            prefix = f'    {json.dumps(key)}: '
            assert sum(line.startswith(prefix) for line in lines) == 1, key


class TestAnUnreadableFile:
    def test_it_is_named_and_skipped_and_the_run_is_incomplete(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path, {**_BASE, 'alpha/src/alpha/broken.py': 'def (:\n'})
        out = tmp_path / 'out' / 'snapshot.json'
        assert _run(root, out) == 0
        measured = json.loads(out.read_text(encoding='utf-8'))
        evidence = measured['evidence']
        assert evidence['unreadable'] == ['alpha/src/alpha/broken.py']
        assert evidence['complete'] is False
        assert evidence['measured_files'] == evidence['domain_files'] - 1
        assert 'alpha/src/alpha/broken.py' not in measured['files']
        assert 'alpha/src/alpha/broken.py' in capsys.readouterr().err


class TestTheCommandLine:
    def test_a_run_without_diff_ends_by_saying_there_was_no_previous(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path, _BASE)
        assert _run(root, tmp_path / 'out' / 'snapshot.json') == 0
        printed = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert printed[-1] == 'no previous snapshot given; since = none'

    def test_the_real_entry_point_measures_and_writes(self, tmp_path: Path) -> None:
        root = _repo(tmp_path, _BASE)
        out = tmp_path / 'out' / 'snapshot.json'
        env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
        proc = subprocess.run(
            [
                sys.executable, str(REPO_ROOT / 'scripts' / 'quality_metrics_snapshot.py'),
                '--run-id', 'r', '--out', str(out), '--root', str(root),
            ],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert out.is_file()

    @pytest.mark.parametrize(
        'argv',
        [
            pytest.param(['--run-id', 'r'], id='run-id-without-out'),
            pytest.param(['--out', 'x.json'], id='out-without-run-id'),
            pytest.param(['--current', 'a.json'], id='current-without-diff'),
            pytest.param(['--summary', 'a.json', '--run-id', 'r'], id='summary-with-run-id'),
            pytest.param([], id='no-mode'),
        ],
    )
    def test_misuse_is_an_argparse_exit_2(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as raised:
            snapshot.main(argv)
        assert raised.value.code == 2


# ---------------------------------------------------------------------------
# The import graph: nodes are module import names.

_GRAPH: dict[str, str] = {
    'alpha/src/pkg/__init__.py': 'THING = 1\n',
    'alpha/src/pkg/a.py': 'from pkg import THING\n',
    'alpha/src/pkg/b.py': 'from pkg import c\n',
    'alpha/src/pkg/c.py': 'X = 1\n',
    'alpha/src/a.py': 'import b\n',
    'alpha/src/b.py': 'import a\n',
    'alpha/src/c.py': 'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import d\n',
    'alpha/src/d.py': 'from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import c\n',
    'alpha/src/e.py': 'import os\n\n\ndef f():\n    import a\n    return a\n',
    'beta/src/beta/b.py': 'import pkg.c\n',
    'alpha/tests/test_pkg.py': 'from pkg import a\nimport pkg.c\n',
    'scripts/x.py': 'import legibility.y\n',
    'scripts/legibility/y.py': 'Y = 1\n',
}

_RELATIVE: dict[str, str] = {
    'alpha/src/rel/__init__.py': 'NAME = 1\n',
    'alpha/src/rel/other.py': 'y = 1\n',
    'alpha/src/rel/sub/__init__.py': 'from .. import NAME\n',
    'alpha/src/rel/sub/n.py': 'thing = 1\n',
    'alpha/src/rel/sub/m.py': (
        'import rel\nfrom . import n\nfrom .n import thing\nfrom ..other import y\n'
    ),
    'beta/src/beta/b.py': 'B = 1\n',
    'scripts/z.py': 'from . import q\n',
}

#: Six function-local imports, each in a different place inside a function; one class-level import.
_DEFERRED_SHAPES = (
    'from typing import TYPE_CHECKING\n'
    'def outer():\n'
    '    def inner():\n'
    '        import os\n'
    '    class Local:\n'
    '        from alpha import mod\n'
    '    try:\n'
    '        import json\n'
    '    except ImportError:\n'
    '        from alpha import mod as m2\n'
    '    if TYPE_CHECKING:\n'
    '        import typing\n'
    'class C:\n'
    '    import sys\n'
    '    def m(self):\n'
    '        import re\n'
)


class TestTheImportGraph:
    @pytest.fixture
    def measured(self, tmp_path: Path) -> dict[str, Any]:
        return _measured(tmp_path, _GRAPH)

    def test_it_has_exactly_the_six_sections(self, measured: dict[str, Any]) -> None:
        assert list(measured['import_graph']) == [
            'edges', 'reach_back', 'deferred', 'cycles', 'hidden_cycles', 'typing_cycles',
        ]

    def test_edges_are_explicit_first_party_src_imports(self, measured: dict[str, Any]) -> None:
        # Module-level, TYPE_CHECKING and function-local imports all count; no
        # implicit parent package, no stdlib, nothing from a tests file.
        assert measured['import_graph']['edges'] == [
            ['a', 'b'],
            ['b', 'a'],
            ['beta.b', 'pkg.c'],
            ['c', 'd'],
            ['d', 'c'],
            ['e', 'a'],
            ['pkg.a', 'pkg'],
            ['pkg.b', 'pkg.c'],
            ['x', 'legibility.y'],
        ]

    def test_cycles_are_over_runtime_module_level_edges_only(
        self, measured: dict[str, Any]
    ) -> None:
        # c <-> d exists only under TYPE_CHECKING, and e -> a is function-local.
        assert measured['import_graph']['cycles'] == [['a', 'b']]

    def test_the_wider_cycle_sets_add_function_local_then_type_checking_edges(
        self, measured: dict[str, Any]
    ) -> None:
        # e -> a is function-local, but a never reaches e; c <-> d is TYPE_CHECKING only.
        graph = measured['import_graph']
        assert graph['hidden_cycles'] == [['a', 'b']]
        assert graph['typing_cycles'] == [['a', 'b'], ['c', 'd']]

    def test_a_name_from_the_package_init_is_a_reach_back_and_a_submodule_is_not(
        self, measured: dict[str, Any]
    ) -> None:
        assert measured['import_graph']['reach_back'] == [
            {'from': 'pkg.a', 'to': 'pkg', 'names': ['THING'], 'line': 1},
        ]

    def test_deferred_imports_are_listed_by_site(self, measured: dict[str, Any]) -> None:
        deferred = measured['import_graph']['deferred']
        assert deferred == [{'from': 'e', 'line': 5, 'imports': ['a'], 'closes_cycle': False}]
        assert (
            len([entry for entry in deferred if entry['from'] == 'e'])
            == measured['files']['alpha/src/e.py']['function_local_imports']
        )

    def test_each_src_records_function_local_imports_are_its_deferred_entries(
        self, tmp_path: Path
    ) -> None:
        measured = _measured(tmp_path, {**_BASE, 'alpha/src/alpha/late.py': _DEFERRED_SHAPES})
        deferred_from = [entry['from'] for entry in measured['import_graph']['deferred']]
        src_records = [r for r in measured['files'].values() if r['kind'] == 'src']
        for record in src_records:
            assert record['function_local_imports'] == deferred_from.count(record['module'])
        assert measured['files']['alpha/src/alpha/late.py']['function_local_imports'] == 6

    def test_the_per_file_graph_fields(self, measured: dict[str, Any]) -> None:
        files = measured['files']
        assert files['alpha/src/pkg/a.py']['reach_back_imports'] == 1
        assert files['alpha/src/pkg/b.py']['reach_back_imports'] == 0
        assert files['alpha/src/pkg/b.py']['fan_out'] == 1
        assert files['alpha/src/e.py']['fan_out'] == 1
        assert files['alpha/src/pkg/c.py']['fan_in_src'] == 2
        assert files['alpha/src/a.py']['fan_in_src'] == 2
        assert files['alpha/src/pkg/c.py']['fan_in_tests'] == 1
        assert files['alpha/src/pkg/a.py']['fan_in_tests'] == 1
        assert files['alpha/src/b.py']['fan_in_tests'] == 0

    def test_a_src_record_has_exactly_the_contract_fields(
        self, measured: dict[str, Any]
    ) -> None:
        assert set(measured['files']['alpha/src/pkg/a.py']) == {
            'member', 'kind', 'blob', 'lines', 'prose_lines', 'prose_ratio',
            'cognitive_total', 'cognitive_max', 'cognitive_max_function', 'functions',
            'module', 'package_init', 'function_local_imports', 'reexport_names',
            'reach_back_imports', 'cycle_closing_imports', 'fan_out', 'fan_in_src', 'fan_in_tests',
        }

    def test_relative_imports_resolve(self, tmp_path: Path) -> None:
        graph = _measured(tmp_path, _RELATIVE)['import_graph']
        edges = graph['edges']
        for edge in (
            ['rel.sub', 'rel'],
            ['rel.sub.m', 'rel'],
            ['rel.sub.m', 'rel.other'],
            ['rel.sub.m', 'rel.sub.n'],
        ):
            assert edges.count(edge) == 1, edge
        # An __init__ importing a name from an ancestor package is a reach-back;
        # a bare `import rel` and a lateral `from ..other import y` are not, and
        # scripts/z.py's unresolvable relative import is neither edge nor error.
        assert graph['reach_back'] == [
            {'from': 'rel.sub', 'to': 'rel', 'names': ['NAME'], 'line': 1},
        ]
        assert not [edge for edge in edges if edge[0] == 'z']

    def test_an_unreadable_module_is_still_a_node(self, tmp_path: Path) -> None:
        measured = _measured(
            tmp_path,
            {**_BASE, 'alpha/src/broken.py': 'def (:\n', 'alpha/src/user.py': 'import broken\n'},
        )
        assert ['user', 'broken'] in measured['import_graph']['edges']
        assert 'alpha/src/broken.py' in measured['evidence']['unreadable']
        assert 'alpha/src/broken.py' not in measured['files']
        assert measured['files']['alpha/src/user.py']['fan_out'] == 1


#: Cycles that only function-local (j <-> m, f <-> g) or TYPE_CHECKING (t <-> v) imports close.
_CYCLE_SETS: dict[str, str] = {
    'alpha/src/f.py': 'import g\n',
    'alpha/src/g.py': 'def h():\n    import w, f\n    return f\n',
    'alpha/src/w.py': 'W = 1\n',
    'alpha/src/j.py': 'def k():\n    import m\n    return m\n',
    'alpha/src/m.py': 'def n():\n    import j\n    return j\n',
    'alpha/src/t.py': (
        'from typing import TYPE_CHECKING\n\n\ndef u():\n    if TYPE_CHECKING:\n        import v\n'
    ),
    'alpha/src/v.py': 'import t\n',
    'alpha/src/e.py': 'def late():\n    import f\n    return f\n',
    'beta/src/beta/b.py': 'B = 1\n',
}


class TestTheCycleSets:
    @pytest.fixture
    def measured(self, tmp_path: Path) -> dict[str, Any]:
        return _measured(tmp_path, _CYCLE_SETS)

    def test_each_set_adds_the_edges_of_a_later_moment(self, measured: dict[str, Any]) -> None:
        graph = measured['import_graph']
        assert graph['cycles'] == []
        assert graph['hidden_cycles'] == [['f', 'g'], ['j', 'm']]
        assert graph['typing_cycles'] == [['f', 'g'], ['j', 'm'], ['t', 'v']]

    def test_a_deferred_import_closes_a_cycle_when_it_lies_on_a_hidden_one(
        self, measured: dict[str, Any]
    ) -> None:
        # g: one target in the importer's hidden cycle suffices; j and m: both
        # function-local edges of one cycle are marked; t: a TYPE_CHECKING import
        # lies on a typing cycle only; e: its target never reaches back.
        assert measured['import_graph']['deferred'] == [
            {'from': 'e', 'line': 2, 'imports': ['f'], 'closes_cycle': False},
            {'from': 'g', 'line': 2, 'imports': ['f', 'w'], 'closes_cycle': True},
            {'from': 'j', 'line': 2, 'imports': ['m'], 'closes_cycle': True},
            {'from': 'm', 'line': 2, 'imports': ['j'], 'closes_cycle': True},
            {'from': 't', 'line': 6, 'imports': ['v'], 'closes_cycle': False},
        ]

    def test_each_src_record_counts_its_cycle_closing_imports(self, measured: dict[str, Any]) -> None:
        counts = {
            record['module']: record['cycle_closing_imports']
            for record in measured['files'].values()
            if record['kind'] == 'src'
        }
        assert counts == {
            'beta.b': 0, 'e': 0, 'f': 0, 'g': 1, 'j': 1, 'm': 1, 't': 0, 'v': 0, 'w': 0,
        }


# ---------------------------------------------------------------------------
# The tests-kind measures: what a test reaches inside first-party modules.

_TEST_MOD = (
    'from unittest.mock import patch\n'
    'from pkg import mod\n'
    '\n'
    'def test_it(monkeypatch):\n'
    "    with patch('pkg.mod._x'), patch('pkg.mod.public'), patch.object(mod, '_y'), "
    "patch('os._exit'):\n"
    '        assert mod._x is not None\n'
    '        assert mod.public\n'
)

_PATCHING: dict[str, str] = {
    'alpha/src/pkg/__init__.py': '',
    'alpha/src/pkg/mod.py': '_x = 1\n_y = 2\npublic = 3\n',
    'alpha/src/pkg/_impl.py': 'thing = 1\n',
    'alpha/tests/test_mod.py': _TEST_MOD,
    'alpha/tests/test_impl.py': (
        "from unittest.mock import patch\n\ndef test_impl():\n"
        "    with patch('pkg._impl.thing'):\n        pass\n"
    ),
    'alpha/tests/test_twice.py': (
        'from unittest.mock import patch\n\n'
        "def test_a():\n    with patch('pkg.mod._y'), patch('pkg.mod._x'):\n        pass\n\n"
        "def test_b():\n    with patch('pkg.mod._x'):\n        pass\n"
    ),
    'beta/src/beta/b.py': 'B = 1\n',
}

_SRC_ONLY_FIELDS = frozenset({
    'module', 'package_init', 'function_local_imports', 'reexport_names',
    'reach_back_imports', 'cycle_closing_imports', 'fan_out', 'fan_in_src', 'fan_in_tests',
})


class TestTheTestsKindMeasures:
    @pytest.fixture
    def files(self, tmp_path: Path) -> dict[str, Any]:
        return _measured(tmp_path, _PATCHING)['files']

    def test_private_names_patched_by_string_or_object_path(self, files: dict[str, Any]) -> None:
        # Row 10: the public leaf and the third-party os._exit do not count, and
        # patch.object(mod, '_y') resolves through `from pkg import mod`.
        assert files['alpha/tests/test_mod.py']['private_patch_targets'] == [
            'pkg.mod._x',
            'pkg.mod._y',
        ]

    def test_a_private_module_segment_counts(self, files: dict[str, Any]) -> None:
        assert files['alpha/tests/test_impl.py']['private_patch_targets'] == ['pkg._impl.thing']

    def test_private_reads_are_the_shared_measure(self, files: dict[str, Any]) -> None:
        reads = files['alpha/tests/test_mod.py']['private_reads']
        assert reads == source_measures.private_reads(_TEST_MOD, path='test_mod.py')
        assert reads >= 1

    def test_a_tests_record_has_exactly_the_contract_fields(self, files: dict[str, Any]) -> None:
        record = files['alpha/tests/test_mod.py']
        assert set(record) == {
            'member', 'kind', 'blob', 'lines', 'prose_lines', 'prose_ratio',
            'cognitive_total', 'cognitive_max', 'cognitive_max_function', 'functions',
            'private_patch_targets', 'private_reads',
        }
        assert not _SRC_ONLY_FIELDS & set(record)

    def test_targets_are_distinct_and_sorted(self, files: dict[str, Any]) -> None:
        assert files['alpha/tests/test_twice.py']['private_patch_targets'] == [
            'pkg.mod._x',
            'pkg.mod._y',
        ]


# ---------------------------------------------------------------------------
# --diff: what moved between two snapshots, in the contract's order.

_DIFF_SECTIONS = (
    'unreadable (measures unknown, never zero):',
    'files added:',
    'files removed:',
    'files renamed (same blob):',
    'complexity pair (max per function, module total):',
    'heuristic-14 crossings (measure, not fix):',
    'import graph:',
    'test files (private patch targets, private reads):',
)

#: The row-5 'after' source: g=1, h=1, f=0, file total 2 under complexipy 6.2.
_MOD_SPLIT = (
    '"""The row-5 fixture."""\n'
    '# f holds two sequential ifs.\n'
    'def g(a):\n'
    '    if a:\n'
    '        return 1\n'
    '    return 0\n'
    '\n\n'
    'def h(b):\n'
    '    if b:\n'
    '        return 2\n'
    '    return 0\n'
    '\n\n'
    'def f(a, b):\n'
    '    g(a)\n'
    '    h(b)\n'
    '    return 0\n'
)

_MOD_ONE_MORE_IF = _MOD_BEFORE.replace('    return 0\n', '    if a:\n        x = 3\n    return 0\n')

_MOVED = 'measure per heuristic 14, not a target'


def _assignments(count: int) -> str:
    return ''.join(f'x{number} = {number}\n' for number in range(count))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _chain(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    base_files: dict[str, str],
    *changes: Any,
) -> list[tuple[dict[str, Any], list[str]]]:
    """Measure the base commit, then each change in turn with --diff against the one before.

    Each entry is (snapshot, the diff lines its measuring run printed after the
    'wrote' line); the --current form is checked to print the same lines.
    """
    root = _repo(tmp_path, base_files)
    previous = tmp_path / 's1.json'
    assert snapshot.main(['--run-id', 'run-1', '--out', str(previous), '--root', str(root)]) == 0
    results: list[tuple[dict[str, Any], list[str]]] = [(_load(previous), [])]
    for number, change in enumerate(changes, start=2):
        change(root)
        current = tmp_path / f's{number}.json'
        capsys.readouterr()
        argv = ['--run-id', f'run-{number}', '--out', str(current), '--root', str(root)]
        assert snapshot.main([*argv, '--diff', str(previous)]) == 0
        printed = capsys.readouterr().out.splitlines()
        assert printed[0].startswith('wrote '), printed
        assert snapshot.main(['--current', str(current), '--diff', str(previous)]) == 0
        assert capsys.readouterr().out.splitlines() == printed[1:]
        results.append((_load(current), printed[1:]))
        previous = current
    return results


def _diff_of(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], base_files: dict[str, str], change: Any
) -> list[str]:
    return _chain(tmp_path, capsys, base_files, change)[1][1]


def _section(diff: list[str], header: str) -> list[str]:
    start = diff.index(header) + 1
    end = next(
        (index for index in range(start, len(diff)) if not diff[index].startswith('  ')),
        len(diff),
    )
    return diff[start:end]


def _rewrite(relpath: str, text: str) -> Any:
    return lambda root: _commit(root, f'rewrite {relpath}', write={relpath: text})


class TestDiffLayout:
    def test_the_sections_come_in_the_contract_order(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        diff = _diff_of(tmp_path, capsys, _BASE, _rewrite('alpha/src/alpha/mod.py', _MOD_SPLIT))
        headers = [line for line in diff if not line.startswith('  ')]
        assert headers[0].startswith('current: run-2 as_of ')
        assert headers[1].startswith('previous: run-1 as_of ')
        assert headers[2:] == list(_DIFF_SECTIONS)

    def test_a_snapshot_against_itself_states_every_absence(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / 'out' / 'snapshot.json'
        assert _run(_repo(tmp_path, _BASE), path) == 0
        sha = _load(path)['as_of_sha']
        capsys.readouterr()
        assert snapshot.main(['--current', str(path), '--diff', str(path)]) == 0
        assert capsys.readouterr().out.splitlines() == [
            f'current: run-1 as_of {sha} complete=true',
            f'previous: run-1 as_of {sha} complete=true',
            *(line for header in _DIFF_SECTIONS for line in (header, '  (none)')),
        ]


class TestTheComplexityPair:
    _PAIR = 'complexity pair (max per function, module total):'

    def test_a_split_moves_complexity(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        diff = _diff_of(tmp_path, capsys, _BASE, _rewrite('alpha/src/alpha/mod.py', _MOD_SPLIT))
        assert _section(diff, self._PAIR) == [
            '  alpha/src/alpha/mod.py: max 2 -> 1, total 2 -> 2; max down, total flat or '
            'down: complexity moved, per docs/code-quality.md §What to measure',
        ]

    def test_a_new_branch_adds_complexity(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        diff = _diff_of(
            tmp_path, capsys, _BASE, _rewrite('alpha/src/alpha/mod.py', _MOD_ONE_MORE_IF)
        )
        [line] = _section(diff, self._PAIR)
        assert line.startswith('  alpha/src/alpha/mod.py: max 2 -> ')
        assert line.endswith('; total up: complexity added')

    def test_a_merge_with_a_flat_total_carries_no_reading(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**_BASE, 'alpha/src/alpha/mod.py': _MOD_SPLIT}
        diff = _diff_of(tmp_path, capsys, base, _rewrite('alpha/src/alpha/mod.py', _MOD_BEFORE))
        assert _section(diff, self._PAIR) == ['  alpha/src/alpha/mod.py: max 1 -> 2, total 2 -> 2']


class TestHeuristic14Crossings:
    _CROSSINGS = 'heuristic-14 crossings (measure, not fix):'

    def test_both_marks_in_both_directions_in_path_order(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {
            **_BASE,
            'alpha/src/alpha/z_grow.py': _assignments(1490),
            'alpha/src/alpha/a_shrink.py': _assignments(2010),
        }

        def resize(root: Path) -> None:
            _commit(root, 'resize', write={
                'alpha/src/alpha/z_grow.py': _assignments(2010),
                'alpha/src/alpha/a_shrink.py': _assignments(1490),
            })

        assert _section(_diff_of(tmp_path, capsys, base, resize), self._CROSSINGS) == [
            f'  alpha/src/alpha/a_shrink.py: 2010 -> 1490 lines, crossed 1500 down; {_MOVED}',
            f'  alpha/src/alpha/a_shrink.py: 2010 -> 1490 lines, ALARM crossed 2000 down; {_MOVED}',
            f'  alpha/src/alpha/z_grow.py: 1490 -> 2010 lines, crossed 1500 up; {_MOVED}',
            f'  alpha/src/alpha/z_grow.py: 1490 -> 2010 lines, ALARM crossed 2000 up; {_MOVED}',
        ]

    def test_an_added_file_crosses_up_from_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        diff = _diff_of(tmp_path, capsys, _BASE, _rewrite('alpha/src/alpha/big.py', _assignments(2010)))
        assert _section(diff, self._CROSSINGS) == [
            f'  alpha/src/alpha/big.py: 0 -> 2010 lines, crossed 1500 up; {_MOVED}',
            f'  alpha/src/alpha/big.py: 0 -> 2010 lines, ALARM crossed 2000 up; {_MOVED}',
        ]


class TestRenames:
    def test_a_pure_move_is_a_rename_and_nothing_else(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def move(root: Path) -> None:
            _commit(root, 'move', moves=[('alpha/src/alpha/mod.py', 'alpha/src/alpha/moved.py')])

        diff = _diff_of(tmp_path, capsys, _BASE, move)
        assert _section(diff, 'files renamed (same blob):') == [
            '  alpha/src/alpha/mod.py -> alpha/src/alpha/moved.py',
        ]
        for header in _DIFF_SECTIONS:
            if header != 'files renamed (same blob):':
                assert _section(diff, header) == ['  (none)'], header

    @pytest.mark.parametrize(
        ('source', 'destination'),
        [
            ('scripts/x.py', 'scripts/tests/x.py'),
            ('scripts/tests/test_x.py', 'scripts/trivial.py'),
        ],
        ids=['src-to-tests', 'tests-to-src'],
    )
    def test_a_move_between_kinds_is_a_removal_and_an_addition(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str, destination: str
    ) -> None:
        def move(root: Path) -> None:
            _commit(root, 'move', moves=[(source, destination)])

        diff = _diff_of(tmp_path, capsys, _BASE, move)
        assert _section(diff, 'files renamed (same blob):') == ['  (none)']
        assert _section(diff, 'files added:') == [f'  {destination}']
        assert _section(diff, 'files removed:') == [f'  {source}']
        for header in _DIFF_SECTIONS:
            if header not in ('files renamed (same blob):', 'files added:', 'files removed:'):
                assert _section(diff, header) == ['  (none)'], header

    def test_an_empty_file_moved_between_a_src_and_a_tests_root_is_not_a_rename(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def swap(root: Path) -> None:
            _commit(
                root,
                'swap',
                remove=['alpha/src/alpha/__init__.py'],
                write={'alpha/tests/__init__.py': ''},
            )

        diff = _diff_of(tmp_path, capsys, _BASE, swap)
        assert _section(diff, 'files renamed (same blob):') == ['  (none)']
        assert _section(diff, 'files added:') == ['  alpha/tests/__init__.py']
        assert _section(diff, 'files removed:') == ['  alpha/src/alpha/__init__.py']

    def test_an_empty_file_is_never_a_rename(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def swap(root: Path) -> None:
            _commit(
                root,
                'swap',
                remove=['alpha/src/alpha/__init__.py'],
                write={'beta/src/beta/__init__.py': ''},
            )

        diff = _diff_of(tmp_path, capsys, _BASE, swap)
        assert _section(diff, 'files renamed (same blob):') == ['  (none)']
        assert _section(diff, 'files added:') == ['  beta/src/beta/__init__.py']
        assert _section(diff, 'files removed:') == ['  alpha/src/alpha/__init__.py']

    def test_a_blob_that_left_or_arrived_at_several_paths_is_not_paired(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**_BASE, 'alpha/src/alpha/c1.py': 'C = 1\n', 'alpha/src/alpha/c2.py': 'C = 1\n'}

        def move(root: Path) -> None:
            _commit(root, 'move', moves=[
                ('alpha/src/alpha/c1.py', 'alpha/src/alpha/z2.py'),
                ('alpha/src/alpha/c2.py', 'alpha/src/alpha/a9.py'),
            ])

        diff = _diff_of(tmp_path, capsys, base, move)
        assert _section(diff, 'files renamed (same blob):') == ['  (none)']
        assert _section(diff, 'files added:') == [
            '  alpha/src/alpha/a9.py',
            '  alpha/src/alpha/z2.py',
        ]
        assert _section(diff, 'files removed:') == [
            '  alpha/src/alpha/c1.py',
            '  alpha/src/alpha/c2.py',
        ]

    def test_a_path_whose_kind_changed_is_compared_within_one_kind(self, tmp_path: Path) -> None:
        path = tmp_path / 'out' / 'snapshot.json'
        assert _run(_repo(tmp_path, _BASE), path) == 0
        previous = _load(path)
        edited = copy.deepcopy(previous)
        edited['files']['scripts/x.py'] = {
            **previous['files']['scripts/tests/test_x.py'],
            'private_patch_targets': ['pkg._p'],
        }
        snapshot.validate_snapshot(edited, origin='edited')
        tests_header = 'test files (private patch targets, private reads):'

        forward = snapshot.diff_lines(edited, previous)
        assert _section(forward, tests_header) == ['  scripts/x.py: +pkg._p']
        assert _section(forward, 'import graph:') == ['  (none)']

        backward = snapshot.diff_lines(previous, edited)
        assert _section(backward, tests_header) == ['  scripts/x.py: -pkg._p']
        assert _section(backward, 'import graph:') == ['  (none)']


_GRAPH_BASE: dict[str, str] = {
    **_BASE,
    'alpha/src/a.py': 'A = 1\n',
    'alpha/src/b.py': 'B = 1\n',
    'alpha/src/e.py': 'def f():\n    return 1\n',
    'alpha/src/pkg/__init__.py': '',
    'alpha/src/pkg/a.py': 'Y = 1\n',
}

_GRAPH_CHANGED: dict[str, str] = {
    'alpha/src/a.py': 'import b\nA = 1\n',
    'alpha/src/b.py': 'import a\nB = 1\n',
    'alpha/src/e.py': 'def f():\n    import a\n    return a\n',
    'alpha/src/pkg/__init__.py': 'THING = 1\n',
    'alpha/src/pkg/a.py': 'from pkg import THING\nY = THING\n',
}


class TestImportGraphChanges:
    _GRAPH = 'import graph:'

    def test_additions_then_their_removals(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        reverted = {path: _GRAPH_BASE[path] for path in _GRAPH_CHANGED}
        chain = _chain(
            tmp_path,
            capsys,
            _GRAPH_BASE,
            lambda root: _commit(root, 'couple', write=_GRAPH_CHANGED),
            lambda root: _commit(root, 'decouple', write=reverted),
        )
        added = _section(chain[1][1], self._GRAPH)
        for line in (
            '  edge added: a -> b',
            '  edge added: b -> a',
            '  cycle added: a, b',
            '  hidden cycle added: a, b',
            '  typing cycle added: a, b',
            '  reach-back added: pkg.a -> pkg (THING)',
            '  deferred import added: e: a',
        ):
            assert line in added, (line, added)
        removed = _section(chain[2][1], self._GRAPH)
        for line in (
            '  edge removed: a -> b',
            '  edge removed: b -> a',
            '  cycle removed: a, b',
            '  hidden cycle removed: a, b',
            '  typing cycle removed: a, b',
            '  reach-back removed: pkg.a -> pkg (THING)',
            '  deferred import removed: e: a',
        ):
            assert line in removed, (line, removed)

    def test_line_numbers_are_not_compared(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        coupled = {**_GRAPH_BASE, **_GRAPH_CHANGED}
        shifted = '\n' + _GRAPH_CHANGED['alpha/src/pkg/a.py']
        diff = _diff_of(tmp_path, capsys, coupled, _rewrite('alpha/src/pkg/a.py', shifted))
        assert _section(diff, self._GRAPH) == ['  (none)']

    def test_a_function_local_import_closing_a_cycle_is_a_hidden_cycle_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**_BASE, 'alpha/src/f.py': 'import g\n', 'alpha/src/g.py': 'G = 1\n'}
        diff = _diff_of(
            tmp_path, capsys, base, _rewrite('alpha/src/g.py', 'def h():\n    import f\n    return f\n')
        )
        assert _section(diff, self._GRAPH) == [
            '  edge added: g -> f',
            '  deferred import added: g: f',
            '  cycle-closing deferred import added: g: f',
            '  hidden cycle added: f, g',
            '  typing cycle added: f, g',
        ]

    def test_a_deferred_import_that_starts_closing_a_cycle_is_a_cycle_closing_one_added(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {
            **_BASE,
            'alpha/src/f.py': 'F = 1\n',
            'alpha/src/g.py': 'def h():\n    import f\n    return f\n',
        }
        diff = _diff_of(tmp_path, capsys, base, _rewrite('alpha/src/f.py', 'import g\nF = 1\n'))
        assert _section(diff, self._GRAPH) == [
            '  edge added: f -> g',
            '  cycle-closing deferred import added: g: f',
            '  hidden cycle added: f, g',
            '  typing cycle added: f, g',
        ]

    def test_a_deferred_import_leaving_type_checking_onto_a_hidden_cycle_is_reported(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Edges, cycle sets and the deferred imports by (from, imports) all stay the same.
        guarded = (
            'from typing import TYPE_CHECKING\n\n\ndef h():\n    import f\n    return f\n\n\n'
            'def k():\n    if TYPE_CHECKING:\n        import f\n    return TYPE_CHECKING\n'
        )
        base = {**_BASE, 'alpha/src/f.py': 'import g\n', 'alpha/src/g.py': guarded}
        unguarded = guarded.replace('    if TYPE_CHECKING:\n        import f\n', '    import f\n')
        diff = _diff_of(tmp_path, capsys, base, _rewrite('alpha/src/g.py', unguarded))
        assert _section(diff, self._GRAPH) == ['  cycle-closing deferred import added: g: f']

    def test_a_type_checking_only_cycle_is_a_typing_cycle_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**_BASE, 'alpha/src/c.py': 'X = 1\n', 'alpha/src/d.py': 'X = 1\n'}

        def couple(root: Path) -> None:
            _commit(root, 'couple', write={
                'alpha/src/c.py': _GRAPH['alpha/src/c.py'],
                'alpha/src/d.py': _GRAPH['alpha/src/d.py'],
            })

        assert _section(_diff_of(tmp_path, capsys, base, couple), self._GRAPH) == [
            '  edge added: c -> d',
            '  edge added: d -> c',
            '  typing cycle added: c, d',
        ]


def _as_schema_1(taken: dict[str, Any]) -> dict[str, Any]:
    """What a pre-6612 measurement of the same tree recorded, e.g. plans/quality-metrics/review-all-dark_factory-20261008.json."""
    old = copy.deepcopy(taken)
    old['schema_version'] = 1
    graph = old['import_graph']
    del graph['hidden_cycles'], graph['typing_cycles']
    for entry in graph['deferred']:
        del entry['closes_cycle']
    for record in old['files'].values():
        record.pop('cycle_closing_imports', None)
    return old


def _unknown_in_schema_1(side: str) -> list[str]:
    return [
        f'  cycle-closing deferred imports unknown: the {side} snapshot is schema 1',
        f'  hidden cycles unknown: the {side} snapshot is schema 1',
        f'  typing cycles unknown: the {side} snapshot is schema 1',
    ]


class TestASchema1SnapshotIsStillRead:
    @pytest.fixture
    def taken(self, tmp_path: Path) -> dict[str, Any]:
        return _measured(tmp_path, _GRAPH)

    def test_it_is_valid(self, taken: dict[str, Any]) -> None:
        old = _as_schema_1(taken)
        assert snapshot.validate_snapshot(old, origin='v1') == old

    def test_its_absent_cycle_sets_are_unknown_on_either_side(self, taken: dict[str, Any]) -> None:
        # Nothing else in the section: deferred entries compare equal without closes_cycle.
        old = _as_schema_1(taken)
        assert _section(snapshot.diff_lines(taken, old), 'import graph:') == _unknown_in_schema_1(
            'previous'
        )
        assert _section(snapshot.diff_lines(old, taken), 'import graph:') == _unknown_in_schema_1(
            'current'
        )

    def test_a_measuring_run_diffs_against_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path, _GRAPH)
        first, second, previous = tmp_path / 's1.json', tmp_path / 's2.json', tmp_path / 'prev.json'
        assert _run(root, first) == 0
        previous.write_text(json.dumps(_as_schema_1(_load(first))), encoding='utf-8')
        capsys.readouterr()
        assert _run(root, second, '--diff', str(previous)) == 0
        printed = capsys.readouterr().out.splitlines()
        assert _section(printed, 'import graph:') == _unknown_in_schema_1('previous')
        assert snapshot.main(['--current', str(second), '--diff', str(previous)]) == 0
        printed = capsys.readouterr().out.splitlines()
        assert _section(printed, 'import graph:') == _unknown_in_schema_1('previous')

    @pytest.mark.parametrize(
        ('named', 'edit'),
        [
            pytest.param(
                'import_graph.deferred[0] keys',
                lambda old: old['import_graph']['deferred'][0].update(closes_cycle=False),
                id='closes_cycle',
            ),
            pytest.param(
                "files['alpha/src/a.py'] keys",
                lambda old: old['files']['alpha/src/a.py'].update(cycle_closing_imports=0),
                id='cycle_closing_imports',
            ),
        ],
    )
    def test_one_carrying_a_schema_2_field_is_refused(
        self, taken: dict[str, Any], named: str, edit: _Edit
    ) -> None:
        old = _as_schema_1(taken)
        edit(old)
        with pytest.raises(source_measures.MetricsError) as raised:
            snapshot.validate_snapshot(old, origin='v1')
        assert named in str(raised.value)

    def test_an_unread_version_is_refused_naming_the_read_ones(self, taken: dict[str, Any]) -> None:
        with pytest.raises(source_measures.MetricsError) as raised:
            snapshot.validate_snapshot({**taken, 'schema_version': 3}, origin='v3')
        assert 'schema_version is 3' in str(raised.value)
        assert 'one of [1, 2]' in str(raised.value)

    def test_the_instrument_is_checked_before_the_version(self) -> None:
        with pytest.raises(source_measures.MetricsError) as raised:
            snapshot.validate_snapshot({'schema_version': 7, 'instrument': 'other'}, origin='foreign')
        assert 'instrument' in str(raised.value)


_COUPLING_BEFORE = 'from pkg import mod\n\ndef test_it():\n    assert mod._y\n'
_COUPLING_AFTER = (
    'from unittest.mock import patch\nfrom pkg import mod\n\ndef test_it():\n'
    "    with patch('pkg.mod._x'):\n        assert mod._y\n        assert mod._y\n"
)


class TestTestFileCoupling:
    _TESTS = 'test files (private patch targets, private reads):'

    def test_targets_added_and_removed_and_the_read_delta(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {
            **_BASE,
            'alpha/src/pkg/__init__.py': '',
            'alpha/src/pkg/mod.py': '_x = 1\n_y = 2\n',
            'alpha/tests/test_mod.py': _COUPLING_BEFORE,
        }
        chain = _chain(
            tmp_path, capsys, base, _rewrite('alpha/tests/test_mod.py', _COUPLING_AFTER)
        )
        before = source_measures.private_reads(_COUPLING_BEFORE, path='before.py')
        after = source_measures.private_reads(_COUPLING_AFTER, path='after.py')
        assert _section(chain[1][1], self._TESTS) == [
            f'  alpha/tests/test_mod.py: +pkg.mod._x; private reads {before} -> {after}',
        ]
        reverse = snapshot.diff_lines(chain[0][0], chain[1][0])
        assert _section(reverse, self._TESTS) == [
            f'  alpha/tests/test_mod.py: -pkg.mod._x; private reads {after} -> {before}',
        ]


class TestCompleteness:
    def test_an_unreadable_previous_file_is_unknown_not_added(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**_BASE, 'alpha/src/alpha/broken.py': 'def (:\n'}
        diff = _diff_of(tmp_path, capsys, base, _rewrite('alpha/src/alpha/broken.py', 'X = 1\n'))
        previous = next(line for line in diff if line.startswith('previous: '))
        assert previous.endswith(' complete=false')
        assert _section(diff, 'unreadable (measures unknown, never zero):') == [
            '  previous: alpha/src/alpha/broken.py',
        ]
        assert _section(diff, 'files added:') == ['  (none)']
        pair = _section(diff, 'complexity pair (max per function, module total):')
        assert not [line for line in pair if 'broken.py' in line]

    def test_a_measured_diff_records_since_and_no_first_run_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        chain = _chain(tmp_path, capsys, _BASE, _rewrite('alpha/src/alpha/mod.py', _MOD_SPLIT))
        (first, _), (second, diff) = chain
        assert second['since'] == first['as_of_sha']
        assert not [line for line in diff if 'no previous snapshot given' in line]


# ---------------------------------------------------------------------------
# --summary: one row per member in name order, then the domain's totals.

_SUMMARY_HEADER = (
    '| member | pseudo | src files | src lines | src prose lines | src cognitive max / total '
    '| tests files | tests lines | tests cognitive total | files >= 1500 lines '
    '| files >= 2000 lines | function-local imports | re-export names | reach-back imports '
    '| private patch targets | private reads |'
)

_SUMMARY_FILES: dict[str, str] = {
    # Declared out of name order, so name order is what is pinned.
    'pyproject.toml': '[tool.uv.workspace]\nmembers = ["beta", "alpha"]\n',
    'alpha/src/alpha/__init__.py': 'THING = 1\n',
    'alpha/src/alpha/mod.py': _MOD_BEFORE,
    'alpha/src/alpha/big.py': _assignments(2010),
    'alpha/src/alpha/shim.py': 'from alpha.mod import f\n',
    'alpha/src/alpha/late.py': 'def g():\n    import alpha.mod\n    return alpha.mod\n',
    'alpha/src/alpha/rb.py': 'from alpha import THING\nY = THING\n',
    'alpha/tests/test_mod.py': (
        'from unittest.mock import patch\nfrom alpha import mod\n\ndef test_it():\n'
        "    with patch('alpha.mod._x'):\n        assert mod._y\n"
    ),
    'beta/src/beta/b.py': 'B = 1\n',
    'beta/tests/test_b.py': _TRIVIAL_TEST,
    'scripts/x.py': 'X = 1\n',
    'scripts/tests/test_x.py': _TRIVIAL_TEST,
    'tests/test_y.py': _TRIVIAL_TEST,
}


def _summary_of(path: Path, capsys: pytest.CaptureFixture[str]) -> list[str]:
    capsys.readouterr()
    assert snapshot.main(['--summary', str(path)]) == 0
    return capsys.readouterr().out.splitlines()


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip('|').split('|')]


def _table(summary: list[str]) -> tuple[list[str], list[list[str]]]:
    """The header cells and the data rows' cells, after the separator row."""
    start = summary.index(_SUMMARY_HEADER)
    rows = []
    for line in summary[start + 2:]:
        if not line.startswith('|'):
            break
        rows.append(_cells(line))
    return _cells(summary[start]), rows


def _expected_cells(records: list[dict[str, Any]]) -> list[str]:
    """One row's cells after member and pseudo, recomputed from the file records."""
    src = [record for record in records if record['kind'] == 'src']
    tests = [record for record in records if record['kind'] == 'tests']
    return [str(value) for value in (
        len(src),
        sum(record['lines'] for record in src),
        sum(record['prose_lines'] for record in src),
    )] + [
        f'{max((record["cognitive_max"] for record in src), default=0)} / '
        f'{sum(record["cognitive_total"] for record in src)}',
    ] + [str(value) for value in (
        len(tests),
        sum(record['lines'] for record in tests),
        sum(record['cognitive_total'] for record in tests),
        sum(record['lines'] >= 1500 for record in records),
        sum(record['lines'] >= 2000 for record in records),
        sum(record['function_local_imports'] for record in src),
        sum(len(record['reexport_names']) for record in src),
        sum(record['reach_back_imports'] for record in src),
        sum(len(record['private_patch_targets']) for record in tests),
        sum(record['private_reads'] for record in tests),
    )]


class TestTheSummary:
    @pytest.fixture
    def measured(self, tmp_path: Path) -> Path:
        path = tmp_path / 'out' / 'snapshot.json'
        assert _run(_repo(tmp_path, _SUMMARY_FILES), path) == 0
        return path

    def test_the_header_lines(self, measured: Path, capsys: pytest.CaptureFixture[str]) -> None:
        taken = _load(measured)
        graph = taken['import_graph']
        summary = _summary_of(measured, capsys)
        table_at = summary.index(_SUMMARY_HEADER)
        assert summary.index(f'run: run-1 as_of {taken["as_of_sha"]} since none') < table_at
        assert summary.index('files: 12/12 measured, complete=true') < table_at
        closing = sum(entry['closes_cycle'] for entry in graph['deferred'])
        assert summary.index(
            f'import graph: {len(graph["edges"])} edges, {len(graph["reach_back"])} reach-backs, '
            f'{len(graph["deferred"])} deferred imports, {closing} cycle-closing deferred imports, '
            f'{len(graph["cycles"])} cycles, {len(graph["hidden_cycles"])} hidden cycles, '
            f'{len(graph["typing_cycles"])} typing cycles'
        ) < table_at
        assert 'unreadable (measures unknown, never zero):' not in summary

    def test_a_schema_1_snapshot_names_its_cycle_sets_unknown(
        self, measured: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        taken = _load(measured)
        graph = taken['import_graph']
        old = tmp_path / 'schema-1.json'
        old.write_text(json.dumps(_as_schema_1(taken)), encoding='utf-8')
        summary = _summary_of(old, capsys)
        assert (
            f'import graph: {len(graph["edges"])} edges, {len(graph["reach_back"])} reach-backs, '
            f'{len(graph["deferred"])} deferred imports, {len(graph["cycles"])} cycles; '
            'cycle-closing deferred imports, hidden cycles, typing cycles unknown (schema 1)'
        ) in summary
        # The table has no column a schema-1 record lacks.
        assert _table(summary)[1] == _table(_summary_of(measured, capsys))[1]

    def test_an_incomplete_snapshot_names_its_unreadable_paths(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / 'out' / 'snapshot.json'
        assert _run(_repo(tmp_path, {**_BASE, 'alpha/src/alpha/broken.py': 'def (:\n'}), path) == 0
        summary = _summary_of(path, capsys)
        assert 'files: 8/9 measured, complete=false' in summary
        at = summary.index('unreadable (measures unknown, never zero):')
        assert summary[at + 1] == '  alpha/src/alpha/broken.py'

    def test_one_row_per_member_in_name_order_then_the_domain(
        self, measured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        header, rows = _table(_summary_of(measured, capsys))
        assert len(header) == 16
        names = [row[0] for row in rows]
        assert names == [*sorted(['alpha', 'beta', 'scripts', 'tests']), '(domain)']
        assert [row[1] for row in rows] == ['false', 'false', 'true', 'true', '']

    def test_each_cell_is_a_count_a_total_or_the_pair(
        self, measured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        taken = _load(measured)
        _header, rows = _table(_summary_of(measured, capsys))
        by_member: dict[str, list[dict[str, Any]]] = {}
        for record in taken['files'].values():
            by_member.setdefault(record['member'], []).append(record)
        for row in rows[:-1]:
            assert row[2:] == _expected_cells(by_member.get(row[0], [])), row[0]
        assert rows[-1][2:] == _expected_cells(list(taken['files'].values()))
        tests_row = next(row for row in rows if row[0] == 'tests')
        assert (tests_row[2], tests_row[5]) == ('0', '0 / 0')

    @pytest.mark.parametrize('content', [None, '{"instrument": "other"}'], ids=['missing', 'foreign'])
    def test_a_bad_snapshot_is_refused_naming_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None
    ) -> None:
        path = tmp_path / 'given.json'
        if content is not None:
            path.write_text(content, encoding='utf-8')
        assert snapshot.main(['--summary', str(path)]) == 2
        assert str(path) in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Refusals: exit 2, the cause named on stderr, nothing written.


def _refused(root: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> str:
    """Run against *root*, assert the refusal wrote nothing, and return stderr."""
    out = root.parent / 'never' / 'snapshot.json'
    assert _run(root, out, *extra) == 2
    assert not out.parent.exists()
    return capsys.readouterr().err


_FAKE_GIT = '''#!{python}
import os
import shutil
import subprocess
import sys
from pathlib import Path

real = os.environ['REAL_GIT']
marker = Path(os.environ['SHIM_MARKER'])
if 'status' in sys.argv[1:] and not marker.exists():
    subprocess.run(
        [real, '-C', os.environ['SHIM_ROOT'], '-c', 'user.name=f', '-c', 'user.email=f@e.invalid',
         'commit', '--allow-empty', '--no-verify', '-q', '-m', 'moved'],
        check=True,
    )
    marker.write_text('moved', encoding='utf-8')
completed = subprocess.run([real, *sys.argv[1:]], capture_output=True)
sys.stdout.buffer.write(completed.stdout)
sys.stderr.buffer.write(completed.stderr)
sys.exit(completed.returncode)
'''


class TestADirtyDomainIsRefused:
    def test_a_modified_member_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path, _BASE)
        _write(root, {'alpha/src/alpha/mod.py': 'CHANGED = 1\n'})
        assert 'alpha/src/alpha/mod.py' in _refused(root, capsys)

    def test_a_staged_deletion(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = _repo(tmp_path, _BASE)
        git(root, 'rm', '-q', 'beta/src/beta/b.py')
        assert 'beta/src/beta/b.py' in _refused(root, capsys)

    def test_untracked_files_and_files_outside_the_domain_are_not_dirt(
        self, tmp_path: Path
    ) -> None:
        root = _repo(tmp_path, _BASE)
        _write(root, {'alpha/src/alpha/untracked.py': 'U = 1\n', 'hooks/h.py': 'H = 2\n'})
        out = tmp_path / 'out' / 'snapshot.json'
        assert _run(root, out) == 0
        assert out.is_file()


class TestAMovedHeadIsRefused:
    def test_a_commit_landing_during_the_measurement(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A merge landing mid-run, simulated by a git that commits on its first
        # `status`: the dirty check precedes measuring, the HEAD check follows it.
        root = _repo(tmp_path, _BASE)
        before = git(root, 'rev-parse', 'HEAD').strip()
        real_git = shutil.which('git')
        assert real_git is not None
        shim_bin = tmp_path / 'bin'
        shim_bin.mkdir()
        shim = shim_bin / 'git'
        shim.write_text(_FAKE_GIT.format(python=sys.executable), encoding='utf-8')
        shim.chmod(0o755)
        monkeypatch.setenv('REAL_GIT', real_git)
        monkeypatch.setenv('SHIM_ROOT', str(root))
        monkeypatch.setenv('SHIM_MARKER', str(tmp_path / 'moved.marker'))
        monkeypatch.setenv('PATH', f'{shim_bin}{os.pathsep}{os.environ["PATH"]}')
        err = _refused(root, capsys)
        after = git(root, 'rev-parse', 'HEAD').strip()
        assert before != after
        assert before in err
        assert after in err


class TestToolFaultsAreRefused:
    def test_no_workspace_members(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path, {**_BASE, 'pyproject.toml': '[project]\nname = "x"\n'})
        assert 'tool.uv.workspace' in _refused(root, capsys)

    def test_git_absent(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = _repo(tmp_path, _BASE)
        empty_bin = tmp_path / 'empty-bin'
        empty_bin.mkdir()
        monkeypatch.setenv('PATH', str(empty_bin))
        assert 'could not run git' in _refused(root, capsys)

    def test_complexipy_outside_the_pin(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # complexipy_version is the seam require_complexipy documents for
        # seeding a version; no measure is patched.
        root = _repo(tmp_path, _BASE)
        monkeypatch.setattr(source_measures, 'complexipy_version', lambda: '7.0.1')
        err = _refused(root, capsys)
        assert '7.0.1' in err
        assert source_measures.COMPLEXIPY_REQUIRED in err

    def test_no_head_commit(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = tmp_path / 'repo'
        root.mkdir()
        git(root, 'init', '-q')
        _write(root, {'pyproject.toml': _PYPROJECT, **_BASE})
        git(root, 'add', '-A')
        assert 'HEAD' in _refused(root, capsys)


class TestABadPreviousSnapshotIsRefusedBeforeMeasuring:
    def test_a_missing_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = _repo(tmp_path, _BASE)
        previous = tmp_path / 'missing.json'
        assert str(previous) in _refused(root, capsys, '--diff', str(previous))

    def test_a_file_that_is_not_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path, _BASE)
        previous = tmp_path / 'previous.json'
        previous.write_text('not json\n', encoding='utf-8')
        assert str(previous) in _refused(root, capsys, '--diff', str(previous))

    def test_another_instruments_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path, _BASE)
        previous = tmp_path / 'previous.json'
        previous.write_text(
            json.dumps({'schema_version': 1, 'instrument': 'merge-lane-ratchet'}), encoding='utf-8'
        )
        err = _refused(root, capsys, '--diff', str(previous))
        assert str(previous) in err
        assert 'instrument' in err

    def test_another_schema_version(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = _repo(tmp_path, _BASE)
        measured = tmp_path / 'measured.json'
        assert _run(root, measured) == 0
        previous = tmp_path / 'previous.json'
        previous.write_text(
            json.dumps({**json.loads(measured.read_text(encoding='utf-8')), 'schema_version': 3}),
            encoding='utf-8',
        )
        capsys.readouterr()
        err = _refused(root, capsys, '--diff', str(previous))
        assert str(previous) in err
        assert 'schema_version' in err


_Edit = Callable[[dict[str, Any]], object]

#: A src and a tests record of the summary fixture.
_SRC_PATH = 'alpha/src/alpha/mod.py'
_TESTS_PATH = 'alpha/tests/test_mod.py'


def _setting(*keys: str | int, to: object) -> _Edit:
    """An edit that sets the value at *keys* inside a snapshot to *to*."""

    def edit(taken: dict[str, Any]) -> None:
        *parents, last = keys
        target: Any = taken
        for key in parents:
            target = target[key]
        target[last] = to

    return edit


def _record(path: str, field: str, to: object) -> Any:
    return pytest.param(
        f'files[{path!r}].{field}', _setting('files', path, field, to=to), id=f'{path}.{field}'
    )


#: (what stderr names, the edit): one field of the wrong shape per row.
_MISSHAPEN: list[Any] = [
    pytest.param('run_id', _setting('run_id', to=5), id='run_id'),
    pytest.param('as_of_sha', _setting('as_of_sha', to=None), id='as_of_sha'),
    pytest.param('since', _setting('since', to=3), id='since'),
    pytest.param('evidence.members', _setting('evidence', 'members', to={}), id='members'),
    pytest.param(
        'evidence.members[0].pseudo',
        _setting('evidence', 'members', 0, 'pseudo', to='false'),
        id='member.pseudo',
    ),
    pytest.param(
        'evidence.members[0] keys',
        lambda taken: taken['evidence']['members'][0].pop('domain_files'),
        id='member-keys',
    ),
    pytest.param('evidence.unreadable', _setting('evidence', 'unreadable', to=None), id='unreadable'),
    pytest.param(
        'evidence.unreadable[0]', _setting('evidence', 'unreadable', to=[3]), id='unreadable-item'
    ),
    pytest.param('evidence.complete', _setting('evidence', 'complete', to='yes'), id='complete'),
    pytest.param('cost', _setting('cost', to=None), id='cost'),
    pytest.param(
        'params.h14_alarm_lines', _setting('params', 'h14_alarm_lines', to='2000'), id='params'
    ),
    _record(_SRC_PATH, 'blob', None),
    _record(_SRC_PATH, 'prose_ratio', '0.5'),
    _record(_SRC_PATH, 'cognitive_max_function', 3),
    _record(_SRC_PATH, 'module', None),
    _record(_SRC_PATH, 'package_init', 'no'),
    _record(_SRC_PATH, 'reexport_names', None),
    _record(_SRC_PATH, 'fan_in_tests', 1.5),
    _record(_SRC_PATH, 'cycle_closing_imports', None),
    _record(_TESTS_PATH, 'private_patch_targets', 'alpha.mod._x'),
    _record(_TESTS_PATH, 'private_reads', None),
    pytest.param('functions', _setting('functions', to=[]), id='functions'),
    pytest.param(
        f"functions['{_SRC_PATH}::f']",
        _setting('functions', f'{_SRC_PATH}::f', to='2'),
        id='function-score',
    ),
    pytest.param('import_graph.edges', _setting('import_graph', 'edges', to=None), id='edges'),
    pytest.param(
        'import_graph.edges[0]',
        _setting('import_graph', 'edges', 0, to=['alpha.shim']),
        id='edge',
    ),
    pytest.param(
        'import_graph.reach_back[0].names',
        _setting('import_graph', 'reach_back', 0, 'names', to='THING'),
        id='reach-back-names',
    ),
    pytest.param(
        'import_graph.deferred[0].imports[0]',
        _setting('import_graph', 'deferred', 0, 'imports', to=[None]),
        id='deferred-imports',
    ),
    pytest.param(
        'import_graph.cycles[0][1]', _setting('import_graph', 'cycles', to=[['a', 1]]), id='cycles'
    ),
    pytest.param(
        'import_graph.deferred[0].closes_cycle',
        _setting('import_graph', 'deferred', 0, 'closes_cycle', to='no'),
        id='deferred-closes-cycle',
    ),
    pytest.param(
        'import_graph.hidden_cycles[0][1]',
        _setting('import_graph', 'hidden_cycles', to=[['a', 1]]),
        id='hidden-cycles',
    ),
    pytest.param(
        'import_graph.typing_cycles[0][1]',
        _setting('import_graph', 'typing_cycles', to=[['a', 1]]),
        id='typing-cycles',
    ),
    pytest.param(
        'import_graph keys',
        lambda taken: taken['import_graph'].pop('hidden_cycles'),
        id='import-graph-keys',
    ),
]


def _bump(field: str, by: int = 1) -> _Edit:
    return lambda taken: taken['evidence'].update({field: taken['evidence'][field] + by})


#: (what stderr names, the edit): evidence that no longer describes the records.
_DISAGREEING: list[Any] = [
    pytest.param(
        f'files[{_SRC_PATH!r}].member',
        _setting('files', _SRC_PATH, 'member', to='gamma'),
        id='member-not-in-evidence',
    ),
    pytest.param('evidence.domain_files', _bump('domain_files'), id='domain_files'),
    pytest.param('evidence.measured_files', _bump('measured_files'), id='measured_files'),
    pytest.param(
        'evidence.unreadable',
        lambda taken: taken['evidence']['unreadable'].append('alpha/src/alpha/gone.py'),
        id='unreadable-count',
    ),
    pytest.param('evidence.complete', _setting('evidence', 'complete', to=False), id='complete'),
]


class TestAMalformedSnapshotIsRefused:
    @pytest.fixture(scope='class')
    def taken(self, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
        tmp_path = tmp_path_factory.mktemp('malformed')
        path = tmp_path / 'out' / 'snapshot.json'
        assert _run(_repo(tmp_path, _SUMMARY_FILES), path) == 0
        return _load(path)

    def _compared(
        self, tmp_path: Path, current: dict[str, Any], previous: dict[str, Any]
    ) -> tuple[int, Path]:
        given, valid = tmp_path / 'given.json', tmp_path / 'valid.json'
        given.write_text(json.dumps(current), encoding='utf-8')
        valid.write_text(json.dumps(previous), encoding='utf-8')
        return snapshot.main(['--current', str(given), '--diff', str(valid)]), given

    def test_the_unedited_snapshot_is_accepted(
        self, taken: dict[str, Any], tmp_path: Path
    ) -> None:
        assert self._compared(tmp_path, taken, taken)[0] == 0

    def _refusal(
        self,
        taken: dict[str, Any],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        edit: _Edit,
    ) -> tuple[Path, str]:
        edited = copy.deepcopy(taken)
        edit(edited)
        code, given = self._compared(tmp_path, edited, taken)
        assert code == 2
        return given, capsys.readouterr().err

    @pytest.mark.parametrize(('where', 'edit'), _MISSHAPEN)
    def test_a_field_of_the_wrong_shape_is_named(
        self,
        taken: dict[str, Any],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        where: str,
        edit: _Edit,
    ) -> None:
        given, err = self._refusal(taken, tmp_path, capsys, edit)
        assert str(given) in err
        assert f'{where} is ' in err

    @pytest.mark.parametrize(('where', 'edit'), _DISAGREEING)
    def test_evidence_that_disagrees_with_the_records_is_named(
        self,
        taken: dict[str, Any],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        where: str,
        edit: _Edit,
    ) -> None:
        given, err = self._refusal(taken, tmp_path, capsys, edit)
        assert str(given) in err
        assert where in err


def test_two_src_files_importing_as_one_module_name_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The import graph is keyed by module name, so an ambiguous one would be a
    # wrong graph.
    root = _repo(tmp_path, {**_BASE, 'alpha/src/x.py': 'Y = 1\n'})
    err = _refused(root, capsys)
    assert 'alpha/src/x.py' in err
    assert 'scripts/x.py' in err


def test_importing_the_snapshot_never_loads_the_ratchet() -> None:
    # A subprocess, because xdist workers share sys.modules with whatever else
    # the worker already imported.
    code = (
        'import sys, quality_metrics_snapshot\n'
        "print(sorted({'merge_lane_metrics'} & set(sys.modules)))\n"
    )
    proc = subprocess.run(
        [sys.executable, '-c', code],
        cwd=REPO_ROOT / 'scripts',
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == '[]', proc.stdout


def test_the_cluster_numbers_agree_with_the_ratchet() -> None:
    # Row 12: both instruments read one set of measures. The TEST may import the
    # ratchet; the snapshot never does (see the layering test above).
    import merge_lane_metrics

    report: dict[str, Any] = merge_lane_metrics.build_report(REPO_ROOT)  # type: ignore[assignment]
    cluster = set(report['files'])
    domain = tuple(
        dataclasses.replace(member, files=tuple(f for f in member.files if f.path in cluster))
        for member in source_measures.workspace_domain(REPO_ROOT)
    )
    measured = snapshot.measure_domain(REPO_ROOT, domain)
    assert measured.unreadable == ()
    assert set(measured.files) == cluster
    for path in sorted(cluster):
        ours = measured.files[path]
        theirs = report['files'][path]
        assert (ours['lines'], ours['prose_lines'], ours['cognitive_total']) == (
            theirs['lines'], theirs['prose_lines'], theirs['cognitive'],
        ), path
