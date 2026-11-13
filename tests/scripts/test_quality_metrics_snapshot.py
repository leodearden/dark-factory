"""Tests of ``scripts/quality_metrics_snapshot.py``, the whole-repo metrics report.

Every fixture is a small REAL git repository whose root pyproject.toml names
two workspace members (or, for a single-package project, one package),
measured through the script's public entry points; no measure is patched.
The one real-tree test (cluster agreement) is the PRD's row 12.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import quality_metrics_snapshot as snapshot
import source_measures
from git_listing import git
from quality_snapshot_fixtures import (
    BASE_FILES,
    GRAPH_FILES,
    MOD_BEFORE,
    TRIVIAL_TEST,
    WORKSPACE_PYPROJECT,
    SnapshotEdit,
    as_schema_1,
    as_schema_2,
    commit,
    committed_repo,
    diff_chain,
    diff_of,
    diff_section,
    measured_snapshot,
    read_json,
    rewriting,
    run_snapshot,
    write_files,
)
from quality_snapshot_identity import below_run_id_and_cost
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).parents[2]

_BASE_DOMAIN = frozenset(path for path in BASE_FILES if not path.startswith('hooks/'))


def _run_in_a_process(root: Path, out: Path, run_id: str, *, hash_seed: str) -> None:
    """Measure *root* through the real entry point, in a process of its own with *hash_seed*."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith('GIT_') or key == 'GIT_CEILING_DIRECTORIES'
    }
    env['PYTHONHASHSEED'] = hash_seed
    proc = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / 'scripts' / 'quality_metrics_snapshot.py'),
            '--run-id', run_id, '--out', str(out), '--root', str(root),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# The measuring run: domain, header and file records.


class TestTheDomain:
    def test_exactly_the_tracked_member_files_are_measured(self, tmp_path: Path) -> None:
        root = committed_repo(tmp_path, BASE_FILES)
        write_files(root, {'alpha/src/alpha/untracked.py': 'U = 1\n'})
        out = tmp_path / 'out' / 'snapshot.json'
        assert run_snapshot(root, out) == 0
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
        measured = measured_snapshot(tmp_path, BASE_FILES)
        assert measured['evidence']['members'] == [
            {'name': 'alpha', 'pseudo': False, 'domain_files': 3},
            {'name': 'beta', 'pseudo': False, 'domain_files': 2},
            {'name': 'scripts', 'pseudo': True, 'domain_files': 2},
            {'name': 'tests', 'pseudo': True, 'domain_files': 1},
        ]


class TestTheHeader:
    def test_top_level_keys_in_contract_order(self, tmp_path: Path) -> None:
        assert list(measured_snapshot(tmp_path, BASE_FILES)) == [
            'schema_version', 'instrument', 'run_id', 'as_of_sha', 'since',
            'evidence', 'cost', 'params', 'files', 'functions', 'import_graph',
        ]

    def test_header_values(self, tmp_path: Path) -> None:
        root = committed_repo(tmp_path, BASE_FILES)
        out = tmp_path / 'out' / 'snapshot.json'
        assert run_snapshot(root, out) == 0
        measured = json.loads(out.read_text(encoding='utf-8'))
        assert measured['schema_version'] == 3
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
        assert list(measured['params']) == [
            'complexipy_version', 'python_version', 'instrument',
            'h14_soft_ceiling_lines', 'h14_alarm_lines',
        ]
        assert measured['params'] == {
            'complexipy_version': source_measures.complexipy_version(),
            'python_version': platform.python_version(),
            'instrument': {
                'quality_metrics_snapshot': _blob_on_disk('scripts/quality_metrics_snapshot.py'),
                'source_measures': _blob_on_disk('scripts/source_measures.py'),
            },
            'h14_soft_ceiling_lines': 1500,
            'h14_alarm_lines': 2000,
        }


def _blob_on_disk(relpath: str) -> str:
    """The running instrument file's blob id: hash-object, so an uncommitted edit still agrees."""
    return git(REPO_ROOT, 'hash-object', '--', relpath).strip()


class TestFileRecords:
    def test_a_src_record_carries_the_shared_measures(self, tmp_path: Path) -> None:
        root = committed_repo(tmp_path, BASE_FILES)
        out = tmp_path / 'out' / 'snapshot.json'
        assert run_snapshot(root, out) == 0
        files = json.loads(out.read_text(encoding='utf-8'))['files']
        path = 'alpha/src/alpha/mod.py'
        record = files[path]
        size = source_measures.file_size_measures(MOD_BEFORE, path=path)
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
            MOD_BEFORE, path=path
        )
        assert record['reexport_names'] == sorted(
            source_measures.reexport_names(MOD_BEFORE, path=path)
        )
        package = files['alpha/src/alpha/__init__.py']
        assert (package['module'], package['package_init']) == ('alpha', True)

    def test_a_module_with_no_functions_has_an_empty_max(self, tmp_path: Path) -> None:
        record = measured_snapshot(tmp_path, BASE_FILES)['files']['beta/src/beta/b.py']
        assert (
            record['cognitive_max'], record['cognitive_max_function'], record['functions']
        ) == (0, None, 0)

    def test_an_empty_file_has_no_prose_ratio(self, tmp_path: Path) -> None:
        files = measured_snapshot(tmp_path, {**BASE_FILES, 'alpha/src/alpha/empty.py': ''})['files']
        assert files['alpha/src/alpha/empty.py']['lines'] == 0
        assert files['alpha/src/alpha/empty.py']['prose_ratio'] is None

    def test_a_tie_at_the_max_names_the_smallest_qualname(self, tmp_path: Path) -> None:
        tie = (
            'def b(x):\n    if x:\n        return 1\n    return 0\n\n\n'
            'def a(x):\n    if x:\n        return 1\n    return 0\n'
        )
        record = measured_snapshot(tmp_path, {**BASE_FILES, 'alpha/src/alpha/tie.py': tie})['files'][
            'alpha/src/alpha/tie.py'
        ]
        assert (record['cognitive_max'], record['cognitive_max_function']) == (1, 'a')


class TestTheFunctionsMap:
    def test_only_src_functions_are_listed_by_path_and_qualname(self, tmp_path: Path) -> None:
        # Every test file in the fixture defines test_x; none of them is listed.
        assert measured_snapshot(tmp_path, BASE_FILES)['functions'] == {'alpha/src/alpha/mod.py::f': 2}

    def test_files_and_functions_are_in_sorted_key_order(self, tmp_path: Path) -> None:
        extra = {'alpha/src/alpha/a_first.py': 'def g():\n    return 1\n'}
        measured = measured_snapshot(tmp_path, {**BASE_FILES, **extra})
        assert list(measured['files']) == sorted(measured['files'])
        assert list(measured['functions']) == sorted(measured['functions'])
        assert len(measured['functions']) == 2


class TestRendering:
    def test_rendering_a_parsed_snapshot_reproduces_its_text(self, tmp_path: Path) -> None:
        root = committed_repo(tmp_path, BASE_FILES)
        out = tmp_path / 'out' / 'snapshot.json'
        assert run_snapshot(root, out) == 0
        text = out.read_text(encoding='utf-8')
        assert snapshot.render_snapshot(json.loads(text)) == text

    def test_each_file_and_function_is_one_line(self, tmp_path: Path) -> None:
        root = committed_repo(tmp_path, BASE_FILES)
        out = tmp_path / 'out' / 'snapshot.json'
        assert run_snapshot(root, out) == 0
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
        root = committed_repo(tmp_path, {**BASE_FILES, 'alpha/src/alpha/broken.py': 'def (:\n'})
        out = tmp_path / 'out' / 'snapshot.json'
        assert run_snapshot(root, out) == 0
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
        root = committed_repo(tmp_path, BASE_FILES)
        assert run_snapshot(root, tmp_path / 'out' / 'snapshot.json') == 0
        printed = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert printed[-1] == 'no previous snapshot given; since = none'

    def test_the_real_entry_point_measures_and_writes(self, tmp_path: Path) -> None:
        out = tmp_path / 'out' / 'snapshot.json'
        _run_in_a_process(committed_repo(tmp_path, BASE_FILES), out, 'r', hash_seed='0')
        assert out.is_file()

    @pytest.mark.parametrize(
        'argv',
        [
            pytest.param(['--run-id', 'r'], id='run-id-without-out'),
            pytest.param(['--out', 'x.json'], id='out-without-run-id'),
            pytest.param(['--current', 'a.json'], id='current-without-diff'),
            pytest.param(['--summary', 'a.json', '--run-id', 'r'], id='summary-with-run-id'),
            pytest.param(['--params', '--run-id', 'r'], id='params-with-run-id'),
            pytest.param([], id='no-mode'),
        ],
    )
    def test_misuse_is_an_argparse_exit_2(self, argv: list[str]) -> None:
        with pytest.raises(SystemExit) as raised:
            snapshot.main(argv)
        assert raised.value.code == 2


class TestTheParamsMode:
    def test_it_prints_the_params_a_measurement_records_as_one_json_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        recorded = measured_snapshot(tmp_path, BASE_FILES)['params']
        capsys.readouterr()
        assert snapshot.main(['--params']) == 0
        printed = capsys.readouterr().out.splitlines()
        assert len(printed) == 1
        assert json.loads(printed[0]) == recorded


# ---------------------------------------------------------------------------
# The import graph: nodes are module import names.

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
        return measured_snapshot(tmp_path, GRAPH_FILES)

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
        measured = measured_snapshot(tmp_path, {**BASE_FILES, 'alpha/src/alpha/late.py': _DEFERRED_SHAPES})
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
        graph = measured_snapshot(tmp_path, _RELATIVE)['import_graph']
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
        measured = measured_snapshot(
            tmp_path,
            {**BASE_FILES, 'alpha/src/broken.py': 'def (:\n', 'alpha/src/user.py': 'import broken\n'},
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
        return measured_snapshot(tmp_path, _CYCLE_SETS)

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
        return measured_snapshot(tmp_path, _PATCHING)['files']

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

_MOD_ONE_MORE_IF = MOD_BEFORE.replace('    return 0\n', '    if a:\n        x = 3\n    return 0\n')

_MOVED = 'measure per heuristic 14, not a target'


def _assignments(count: int) -> str:
    return ''.join(f'x{number} = {number}\n' for number in range(count))


class TestDiffLayout:
    def test_the_sections_come_in_the_contract_order(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        diff = diff_of(tmp_path, capsys, BASE_FILES, rewriting('alpha/src/alpha/mod.py', _MOD_SPLIT))
        headers = [line for line in diff if not line.startswith('  ')]
        assert headers[0].startswith('current: run-2 as_of ')
        assert headers[1].startswith('previous: run-1 as_of ')
        assert headers[2:] == list(_DIFF_SECTIONS)

    def test_a_snapshot_against_itself_states_every_absence(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / 'out' / 'snapshot.json'
        assert run_snapshot(committed_repo(tmp_path, BASE_FILES), path) == 0
        sha = read_json(path)['as_of_sha']
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
        diff = diff_of(tmp_path, capsys, BASE_FILES, rewriting('alpha/src/alpha/mod.py', _MOD_SPLIT))
        assert diff_section(diff, self._PAIR) == [
            '  alpha/src/alpha/mod.py: max 2 -> 1, total 2 -> 2; max down, total flat or '
            'down: complexity moved, per docs/code-quality.md §What to measure',
        ]

    def test_a_new_branch_adds_complexity(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        diff = diff_of(
            tmp_path, capsys, BASE_FILES, rewriting('alpha/src/alpha/mod.py', _MOD_ONE_MORE_IF)
        )
        [line] = diff_section(diff, self._PAIR)
        assert line.startswith('  alpha/src/alpha/mod.py: max 2 -> ')
        assert line.endswith('; total up: complexity added')

    def test_a_merge_with_a_flat_total_carries_no_reading(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**BASE_FILES, 'alpha/src/alpha/mod.py': _MOD_SPLIT}
        diff = diff_of(tmp_path, capsys, base, rewriting('alpha/src/alpha/mod.py', MOD_BEFORE))
        assert diff_section(diff, self._PAIR) == ['  alpha/src/alpha/mod.py: max 1 -> 2, total 2 -> 2']


class TestHeuristic14Crossings:
    _CROSSINGS = 'heuristic-14 crossings (measure, not fix):'

    def test_both_marks_in_both_directions_in_path_order(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {
            **BASE_FILES,
            'alpha/src/alpha/z_grow.py': _assignments(1490),
            'alpha/src/alpha/a_shrink.py': _assignments(2010),
        }

        def resize(root: Path) -> None:
            commit(root, 'resize', write={
                'alpha/src/alpha/z_grow.py': _assignments(2010),
                'alpha/src/alpha/a_shrink.py': _assignments(1490),
            })

        assert diff_section(diff_of(tmp_path, capsys, base, resize), self._CROSSINGS) == [
            f'  alpha/src/alpha/a_shrink.py: 2010 -> 1490 lines, crossed 1500 down; {_MOVED}',
            f'  alpha/src/alpha/a_shrink.py: 2010 -> 1490 lines, ALARM crossed 2000 down; {_MOVED}',
            f'  alpha/src/alpha/z_grow.py: 1490 -> 2010 lines, crossed 1500 up; {_MOVED}',
            f'  alpha/src/alpha/z_grow.py: 1490 -> 2010 lines, ALARM crossed 2000 up; {_MOVED}',
        ]

    def test_an_added_file_crosses_up_from_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        diff = diff_of(tmp_path, capsys, BASE_FILES, rewriting('alpha/src/alpha/big.py', _assignments(2010)))
        assert diff_section(diff, self._CROSSINGS) == [
            f'  alpha/src/alpha/big.py: 0 -> 2010 lines, crossed 1500 up; {_MOVED}',
            f'  alpha/src/alpha/big.py: 0 -> 2010 lines, ALARM crossed 2000 up; {_MOVED}',
        ]


class TestRenames:
    def test_a_pure_move_is_a_rename_and_nothing_else(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def move(root: Path) -> None:
            commit(root, 'move', moves=[('alpha/src/alpha/mod.py', 'alpha/src/alpha/moved.py')])

        diff = diff_of(tmp_path, capsys, BASE_FILES, move)
        assert diff_section(diff, 'files renamed (same blob):') == [
            '  alpha/src/alpha/mod.py -> alpha/src/alpha/moved.py',
        ]
        for header in _DIFF_SECTIONS:
            if header != 'files renamed (same blob):':
                assert diff_section(diff, header) == ['  (none)'], header

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
            commit(root, 'move', moves=[(source, destination)])

        diff = diff_of(tmp_path, capsys, BASE_FILES, move)
        assert diff_section(diff, 'files renamed (same blob):') == ['  (none)']
        assert diff_section(diff, 'files added:') == [f'  {destination}']
        assert diff_section(diff, 'files removed:') == [f'  {source}']
        for header in _DIFF_SECTIONS:
            if header not in ('files renamed (same blob):', 'files added:', 'files removed:'):
                assert diff_section(diff, header) == ['  (none)'], header

    def test_an_empty_file_moved_between_a_src_and_a_tests_root_is_not_a_rename(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def swap(root: Path) -> None:
            commit(
                root,
                'swap',
                remove=['alpha/src/alpha/__init__.py'],
                write={'alpha/tests/__init__.py': ''},
            )

        diff = diff_of(tmp_path, capsys, BASE_FILES, swap)
        assert diff_section(diff, 'files renamed (same blob):') == ['  (none)']
        assert diff_section(diff, 'files added:') == ['  alpha/tests/__init__.py']
        assert diff_section(diff, 'files removed:') == ['  alpha/src/alpha/__init__.py']

    def test_an_empty_file_is_never_a_rename(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def swap(root: Path) -> None:
            commit(
                root,
                'swap',
                remove=['alpha/src/alpha/__init__.py'],
                write={'beta/src/beta/__init__.py': ''},
            )

        diff = diff_of(tmp_path, capsys, BASE_FILES, swap)
        assert diff_section(diff, 'files renamed (same blob):') == ['  (none)']
        assert diff_section(diff, 'files added:') == ['  beta/src/beta/__init__.py']
        assert diff_section(diff, 'files removed:') == ['  alpha/src/alpha/__init__.py']

    def test_a_blob_that_left_or_arrived_at_several_paths_is_not_paired(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**BASE_FILES, 'alpha/src/alpha/c1.py': 'C = 1\n', 'alpha/src/alpha/c2.py': 'C = 1\n'}

        def move(root: Path) -> None:
            commit(root, 'move', moves=[
                ('alpha/src/alpha/c1.py', 'alpha/src/alpha/z2.py'),
                ('alpha/src/alpha/c2.py', 'alpha/src/alpha/a9.py'),
            ])

        diff = diff_of(tmp_path, capsys, base, move)
        assert diff_section(diff, 'files renamed (same blob):') == ['  (none)']
        assert diff_section(diff, 'files added:') == [
            '  alpha/src/alpha/a9.py',
            '  alpha/src/alpha/z2.py',
        ]
        assert diff_section(diff, 'files removed:') == [
            '  alpha/src/alpha/c1.py',
            '  alpha/src/alpha/c2.py',
        ]

    def test_a_path_whose_kind_changed_is_compared_within_one_kind(self, tmp_path: Path) -> None:
        path = tmp_path / 'out' / 'snapshot.json'
        assert run_snapshot(committed_repo(tmp_path, BASE_FILES), path) == 0
        previous = read_json(path)
        edited = copy.deepcopy(previous)
        edited['files']['scripts/x.py'] = {
            **previous['files']['scripts/tests/test_x.py'],
            'private_patch_targets': ['pkg._p'],
        }
        snapshot.validate_snapshot(edited, origin='edited')
        tests_header = 'test files (private patch targets, private reads):'

        forward = snapshot.diff_lines(edited, previous)
        assert diff_section(forward, tests_header) == ['  scripts/x.py: +pkg._p']
        assert diff_section(forward, 'import graph:') == ['  (none)']

        backward = snapshot.diff_lines(previous, edited)
        assert diff_section(backward, tests_header) == ['  scripts/x.py: -pkg._p']
        assert diff_section(backward, 'import graph:') == ['  (none)']


_GRAPH_BASE: dict[str, str] = {
    **BASE_FILES,
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
        chain = diff_chain(
            tmp_path,
            capsys,
            _GRAPH_BASE,
            lambda root: commit(root, 'couple', write=_GRAPH_CHANGED),
            lambda root: commit(root, 'decouple', write=reverted),
        )
        added = diff_section(chain[1][1], self._GRAPH)
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
        removed = diff_section(chain[2][1], self._GRAPH)
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
        diff = diff_of(tmp_path, capsys, coupled, rewriting('alpha/src/pkg/a.py', shifted))
        assert diff_section(diff, self._GRAPH) == ['  (none)']

    def test_a_function_local_import_closing_a_cycle_is_a_hidden_cycle_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**BASE_FILES, 'alpha/src/f.py': 'import g\n', 'alpha/src/g.py': 'G = 1\n'}
        diff = diff_of(
            tmp_path, capsys, base, rewriting('alpha/src/g.py', 'def h():\n    import f\n    return f\n')
        )
        assert diff_section(diff, self._GRAPH) == [
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
            **BASE_FILES,
            'alpha/src/f.py': 'F = 1\n',
            'alpha/src/g.py': 'def h():\n    import f\n    return f\n',
        }
        diff = diff_of(tmp_path, capsys, base, rewriting('alpha/src/f.py', 'import g\nF = 1\n'))
        assert diff_section(diff, self._GRAPH) == [
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
        base = {**BASE_FILES, 'alpha/src/f.py': 'import g\n', 'alpha/src/g.py': guarded}
        unguarded = guarded.replace('    if TYPE_CHECKING:\n        import f\n', '    import f\n')
        diff = diff_of(tmp_path, capsys, base, rewriting('alpha/src/g.py', unguarded))
        assert diff_section(diff, self._GRAPH) == ['  cycle-closing deferred import added: g: f']

    def test_a_type_checking_only_cycle_is_a_typing_cycle_only(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**BASE_FILES, 'alpha/src/c.py': 'X = 1\n', 'alpha/src/d.py': 'X = 1\n'}

        def couple(root: Path) -> None:
            commit(root, 'couple', write={
                'alpha/src/c.py': GRAPH_FILES['alpha/src/c.py'],
                'alpha/src/d.py': GRAPH_FILES['alpha/src/d.py'],
            })

        assert diff_section(diff_of(tmp_path, capsys, base, couple), self._GRAPH) == [
            '  edge added: c -> d',
            '  edge added: d -> c',
            '  typing cycle added: c, d',
        ]


def _unknown_in_schema_1(side: str) -> list[str]:
    return [
        f'  cycle-closing deferred imports unknown: the {side} snapshot is schema 1',
        f'  hidden cycles unknown: the {side} snapshot is schema 1',
        f'  typing cycles unknown: the {side} snapshot is schema 1',
    ]


class TestASchema1SnapshotIsStillRead:
    @pytest.fixture
    def taken(self, tmp_path: Path) -> dict[str, Any]:
        return measured_snapshot(tmp_path, GRAPH_FILES)

    def test_it_is_valid(self, taken: dict[str, Any]) -> None:
        old = as_schema_1(taken)
        assert snapshot.validate_snapshot(old, origin='v1') == old

    def test_its_absent_cycle_sets_are_unknown_on_either_side(self, taken: dict[str, Any]) -> None:
        # Nothing else in the section: deferred entries compare equal without closes_cycle.
        old = as_schema_1(taken)
        assert diff_section(snapshot.diff_lines(taken, old), 'import graph:') == _unknown_in_schema_1(
            'previous'
        )
        assert diff_section(snapshot.diff_lines(old, taken), 'import graph:') == _unknown_in_schema_1(
            'current'
        )

    def test_a_measuring_run_diffs_against_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = committed_repo(tmp_path, GRAPH_FILES)
        first, second, previous = tmp_path / 's1.json', tmp_path / 's2.json', tmp_path / 'prev.json'
        assert run_snapshot(root, first) == 0
        previous.write_text(json.dumps(as_schema_1(read_json(first))), encoding='utf-8')
        capsys.readouterr()
        assert run_snapshot(root, second, '--diff', str(previous)) == 0
        printed = capsys.readouterr().out.splitlines()
        assert diff_section(printed, 'import graph:') == _unknown_in_schema_1('previous')
        assert snapshot.main(['--current', str(second), '--diff', str(previous)]) == 0
        printed = capsys.readouterr().out.splitlines()
        assert diff_section(printed, 'import graph:') == _unknown_in_schema_1('previous')

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
        self, taken: dict[str, Any], named: str, edit: SnapshotEdit
    ) -> None:
        old = as_schema_1(taken)
        edit(old)
        with pytest.raises(source_measures.MetricsError) as raised:
            snapshot.validate_snapshot(old, origin='v1')
        assert named in str(raised.value)

    def test_an_unread_version_is_refused_naming_the_read_ones(self, taken: dict[str, Any]) -> None:
        with pytest.raises(source_measures.MetricsError) as raised:
            snapshot.validate_snapshot({**taken, 'schema_version': 4}, origin='v4')
        assert 'schema_version is 4' in str(raised.value)
        assert 'one of [1, 2, 3]' in str(raised.value)

    def test_the_instrument_is_checked_before_the_version(self) -> None:
        with pytest.raises(source_measures.MetricsError) as raised:
            snapshot.validate_snapshot({'schema_version': 7, 'instrument': 'other'}, origin='foreign')
        assert 'instrument' in str(raised.value)


class TestASchema2SnapshotIsStillRead:
    @pytest.fixture
    def taken(self, tmp_path: Path) -> dict[str, Any]:
        return measured_snapshot(tmp_path, GRAPH_FILES)

    def test_it_is_valid(self, taken: dict[str, Any]) -> None:
        old = as_schema_2(taken)
        assert snapshot.validate_snapshot(old, origin='v2') == old

    @pytest.mark.parametrize(
        ('version', 'dropped'),
        [
            pytest.param(2, 'instrument', id='schema-2-carrying-python_version'),
            pytest.param(2, 'python_version', id='schema-2-carrying-instrument'),
            pytest.param(3, 'instrument', id='schema-3-lacking-instrument'),
        ],
    )
    def test_params_of_another_schema_are_refused(
        self, taken: dict[str, Any], version: int, dropped: str
    ) -> None:
        edited = copy.deepcopy(taken)
        edited['schema_version'] = version
        del edited['params'][dropped]
        with pytest.raises(source_measures.MetricsError) as raised:
            snapshot.validate_snapshot(edited, origin='edited')
        assert 'params keys' in str(raised.value)


class TestMeasurementIdentity:
    @pytest.fixture
    def taken(self, tmp_path: Path) -> dict[str, Any]:
        return measured_snapshot(tmp_path, GRAPH_FILES)

    def test_it_is_the_recorded_sha_and_params_as_canonical_json(self, taken: dict[str, Any]) -> None:
        identity = snapshot.measurement_identity(taken)
        params_json = json.dumps(taken['params'], sort_keys=True)
        assert identity == snapshot.MeasurementIdentity(taken['as_of_sha'], params_json)
        assert identity == snapshot.MeasurementIdentity.of(taken['as_of_sha'], taken['params'])
        assert str(identity) == f"{taken['as_of_sha']} {params_json}"

    def test_it_is_a_hashable_value_its_snapshot_cannot_change(self, taken: dict[str, Any]) -> None:
        identity = snapshot.measurement_identity(taken)
        assert {identity: 'measured'}[snapshot.measurement_identity(copy.deepcopy(taken))] == 'measured'
        before = str(identity)
        taken['params']['complexipy_version'] = '0.0.0'
        assert str(identity) == before
        assert snapshot.measurement_identity(taken) != identity

    def test_two_runs_at_one_identity_are_byte_identical_below_run_id_and_cost(
        self, tmp_path: Path
    ) -> None:
        # Decision 17: identity is (as_of_sha, params), not the checkout path nor the str-hash
        # seed. Seeds 1 and 5 iterate this fixture's import-edge set in different orders.
        root = committed_repo(tmp_path, _SUMMARY_FILES)
        pinned = tmp_path / 'pinned'
        git(root, 'worktree', 'add', '--detach', str(pinned), 'HEAD')
        first, second = tmp_path / 'a.json', tmp_path / 'b.json'
        _run_in_a_process(root, first, 'run-a', hash_seed='1')
        _run_in_a_process(pinned, second, 'run-b', hash_seed='5')
        a, b = snapshot.load_snapshot(first), snapshot.load_snapshot(second)
        assert snapshot.measurement_identity(a) == snapshot.measurement_identity(b)
        assert below_run_id_and_cost(first) == below_run_id_and_cost(second)
        assert a['run_id'] != b['run_id']

    @pytest.mark.parametrize('as_older', [as_schema_2, as_schema_1], ids=['schema-2', 'schema-1'])
    def test_an_older_schema_has_no_identity(
        self, taken: dict[str, Any], as_older: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        old = as_older(taken)
        assert not snapshot.identity_known(old)
        with pytest.raises(source_measures.MetricsError) as raised:
            snapshot.measurement_identity(old)
        assert 'records no instrument, python_version' in str(raised.value)

    @pytest.mark.parametrize(
        'pair',
        [
            pytest.param(lambda t: (t, copy.deepcopy(t)), id='one-identity'),
            pytest.param(lambda t: (t, _with_complexipy(t, '0.0.0')), id='params-changed'),
            pytest.param(lambda t: (t, as_schema_2(t)), id='schema-3-and-2'),
            pytest.param(lambda t: (as_schema_2(t), as_schema_2(t)), id='both-schema-2'),
        ],
    )
    def test_a_shared_identity_is_exactly_a_diff_without_params_differ(
        self,
        taken: dict[str, Any],
        pair: Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]],
    ) -> None:
        current, previous = pair(taken)
        shared = (
            snapshot.identity_known(current)
            and snapshot.identity_known(previous)
            and snapshot.measurement_identity(current) == snapshot.measurement_identity(previous)
        )
        differs = snapshot.diff_lines(current, previous)[0].startswith('params differ: ')
        assert shared is not differs


def _with_complexipy(taken: dict[str, Any], version: str) -> dict[str, Any]:
    edited = copy.deepcopy(taken)
    edited['params']['complexipy_version'] = version
    return edited


_NOT_ATTRIBUTABLE = (
    '  not attributable to code: a delta below may come from the params change, not from the code'
)


class TestParamsDiffer:
    """A params key unknown on either side, or differing, opens the diff; unknown never equals unknown."""

    @pytest.fixture
    def taken(self, tmp_path: Path) -> dict[str, Any]:
        return measured_snapshot(tmp_path, GRAPH_FILES)

    @staticmethod
    def _new_params(taken: dict[str, Any]) -> tuple[str, str]:
        params = taken['params']
        return json.dumps(params['instrument'], sort_keys=True), json.dumps(params['python_version'])

    def test_a_schema_2_previous_reads_instrument_unknown(self, taken: dict[str, Any]) -> None:
        instrument, python = self._new_params(taken)
        diff = snapshot.diff_lines(taken, as_schema_2(taken))
        assert diff[0] == (
            f'params differ: instrument unknown -> {instrument}; python_version unknown -> {python}'
        )
        assert diff[1] == _NOT_ATTRIBUTABLE
        assert diff[2].startswith('current: ')

    def test_a_schema_1_previous_reads_the_same(self, taken: dict[str, Any]) -> None:
        assert (
            snapshot.diff_lines(taken, as_schema_1(taken))[:2]
            == snapshot.diff_lines(taken, as_schema_2(taken))[:2]
        )

    def test_the_reverse_direction(self, taken: dict[str, Any]) -> None:
        instrument, python = self._new_params(taken)
        assert snapshot.diff_lines(as_schema_2(taken), taken)[0] == (
            f'params differ: instrument {instrument} -> unknown; python_version {python} -> unknown'
        )

    def test_unknown_never_equals_unknown(self, taken: dict[str, Any]) -> None:
        old = as_schema_2(taken)
        assert snapshot.diff_lines(old, old)[0] == (
            'params differ: instrument unknown -> unknown; python_version unknown -> unknown'
        )

    def test_only_differing_keys_are_listed(self, taken: dict[str, Any]) -> None:
        older = copy.deepcopy(taken)
        older['params']['complexipy_version'] = '0.0.0'
        real = json.dumps(taken['params']['complexipy_version'])
        assert snapshot.diff_lines(taken, older)[0] == (
            f'params differ: complexipy_version "0.0.0" -> {real}'
        )

    def test_equal_params_open_with_the_headers(self, taken: dict[str, Any]) -> None:
        assert snapshot.diff_lines(taken, taken)[0].startswith('current: ')

    def test_the_command_line_diff_opens_with_it_and_exits_0(
        self, taken: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        current, previous = tmp_path / 'current.json', tmp_path / 'previous.json'
        current.write_text(snapshot.render_snapshot(taken), encoding='utf-8')
        previous.write_text(snapshot.render_snapshot(as_schema_2(taken)), encoding='utf-8')
        capsys.readouterr()
        assert snapshot.main(['--current', str(current), '--diff', str(previous)]) == 0
        printed = capsys.readouterr().out.splitlines()
        assert printed[:2] == snapshot.diff_lines(taken, as_schema_2(taken))[:2]
        assert printed[0].startswith('params differ: instrument unknown -> ')

    def test_the_committed_10_08_snapshot_reads_instrument_unknown(
        self, taken: dict[str, Any]
    ) -> None:
        # The PRD's sketch row 17, on the actual file its signal diffs.
        previous = snapshot.load_snapshot(
            REPO_ROOT / 'plans' / 'quality-metrics' / 'review-all-dark_factory-20261008.json'
        )
        diff = snapshot.diff_lines(taken, previous)
        assert diff[0].startswith('params differ: instrument unknown -> ')
        assert diff[1] == _NOT_ATTRIBUTABLE


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
            **BASE_FILES,
            'alpha/src/pkg/__init__.py': '',
            'alpha/src/pkg/mod.py': '_x = 1\n_y = 2\n',
            'alpha/tests/test_mod.py': _COUPLING_BEFORE,
        }
        chain = diff_chain(
            tmp_path, capsys, base, rewriting('alpha/tests/test_mod.py', _COUPLING_AFTER)
        )
        before = source_measures.private_reads(_COUPLING_BEFORE, path='before.py')
        after = source_measures.private_reads(_COUPLING_AFTER, path='after.py')
        assert diff_section(chain[1][1], self._TESTS) == [
            f'  alpha/tests/test_mod.py: +pkg.mod._x; private reads {before} -> {after}',
        ]
        reverse = snapshot.diff_lines(chain[0][0], chain[1][0])
        assert diff_section(reverse, self._TESTS) == [
            f'  alpha/tests/test_mod.py: -pkg.mod._x; private reads {after} -> {before}',
        ]


class TestCompleteness:
    def test_an_unreadable_previous_file_is_unknown_not_added(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        base = {**BASE_FILES, 'alpha/src/alpha/broken.py': 'def (:\n'}
        diff = diff_of(tmp_path, capsys, base, rewriting('alpha/src/alpha/broken.py', 'X = 1\n'))
        previous = next(line for line in diff if line.startswith('previous: '))
        assert previous.endswith(' complete=false')
        assert diff_section(diff, 'unreadable (measures unknown, never zero):') == [
            '  previous: alpha/src/alpha/broken.py',
        ]
        assert diff_section(diff, 'files added:') == ['  (none)']
        pair = diff_section(diff, 'complexity pair (max per function, module total):')
        assert not [line for line in pair if 'broken.py' in line]

    def test_a_measured_diff_records_since_and_no_first_run_line(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        chain = diff_chain(tmp_path, capsys, BASE_FILES, rewriting('alpha/src/alpha/mod.py', _MOD_SPLIT))
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
    'alpha/src/alpha/mod.py': MOD_BEFORE,
    'alpha/src/alpha/big.py': _assignments(2010),
    'alpha/src/alpha/shim.py': 'from alpha.mod import f\n',
    'alpha/src/alpha/late.py': 'def g():\n    import alpha.mod\n    return alpha.mod\n',
    'alpha/src/alpha/rb.py': 'from alpha import THING\nY = THING\n',
    'alpha/tests/test_mod.py': (
        'from unittest.mock import patch\nfrom alpha import mod\n\ndef test_it():\n'
        "    with patch('alpha.mod._x'):\n        assert mod._y\n"
    ),
    'beta/src/beta/b.py': 'B = 1\n',
    'beta/tests/test_b.py': TRIVIAL_TEST,
    'scripts/x.py': 'X = 1\n',
    'scripts/tests/test_x.py': TRIVIAL_TEST,
    'tests/test_y.py': TRIVIAL_TEST,
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
        assert run_snapshot(committed_repo(tmp_path, _SUMMARY_FILES), path) == 0
        return path

    def test_the_header_lines(self, measured: Path, capsys: pytest.CaptureFixture[str]) -> None:
        taken = read_json(measured)
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
        taken = read_json(measured)
        graph = taken['import_graph']
        old = tmp_path / 'schema-1.json'
        old.write_text(json.dumps(as_schema_1(taken)), encoding='utf-8')
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
        assert run_snapshot(committed_repo(tmp_path, {**BASE_FILES, 'alpha/src/alpha/broken.py': 'def (:\n'}), path) == 0
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
        taken = read_json(measured)
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
# A single-package project: no [tool.uv.workspace], one package in the src layout.

_SINGLE_PACKAGE_FILES: dict[str, str] = {
    'pyproject.toml': '[project]\nname = "solo"\n',
    'src/solo/__init__.py': '',
    'src/solo/mod.py': MOD_BEFORE,
    'tests/test_mod.py': (
        'from unittest.mock import patch\n\n\ndef test_it():\n'
        "    with patch('solo.mod._helper'):\n        assert True\n"
    ),
    'scripts/x.py': 'X = 1\n',
    'hooks/h.py': 'H = 1\n',
}


class TestASinglePackageProject:
    @pytest.fixture
    def measured(self, tmp_path: Path) -> Path:
        path = tmp_path / 'out' / 'snapshot.json'
        assert run_snapshot(committed_repo(tmp_path, _SINGLE_PACKAGE_FILES), path) == 0
        return path

    def test_the_package_and_the_scripts_pseudo_member_are_the_domain(
        self, measured: Path
    ) -> None:
        taken = read_json(measured)
        assert {
            path: (record['member'], record['kind']) for path, record in taken['files'].items()
        } == {
            'src/solo/__init__.py': ('solo', 'src'),
            'src/solo/mod.py': ('solo', 'src'),
            'tests/test_mod.py': ('solo', 'tests'),
            'scripts/x.py': ('scripts', 'src'),
        }
        assert taken['evidence']['members'] == [
            {'name': 'solo', 'pseudo': False, 'domain_files': 3},
            {'name': 'scripts', 'pseudo': True, 'domain_files': 1},
        ]
        assert taken['evidence']['complete'] is True

    def test_src_files_are_named_by_their_path_below_src(self, measured: Path) -> None:
        taken = read_json(measured)
        files = taken['files']
        package = files['src/solo/__init__.py']
        assert (package['module'], package['package_init']) == ('solo', True)
        assert files['src/solo/mod.py']['module'] == 'solo.mod'
        assert taken['functions']['src/solo/mod.py::f'] == 2

    def test_the_package_import_names_reach_the_patch_target_measure(
        self, measured: Path
    ) -> None:
        record = read_json(measured)['files']['tests/test_mod.py']
        assert record['private_patch_targets'] == ['solo.mod._helper']

    def test_the_summary_has_a_row_for_the_package(
        self, measured: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        summary = _summary_of(measured, capsys)
        assert [line for line in summary if line.startswith('| solo | false |')]


# ---------------------------------------------------------------------------
# Refusals: exit 2, the cause named on stderr, nothing written.


def _refused(root: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> str:
    """Run against *root*, assert the refusal wrote nothing, and return stderr."""
    out = root.parent / 'never' / 'snapshot.json'
    assert run_snapshot(root, out, *extra) == 2
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
        root = committed_repo(tmp_path, BASE_FILES)
        write_files(root, {'alpha/src/alpha/mod.py': 'CHANGED = 1\n'})
        assert 'alpha/src/alpha/mod.py' in _refused(root, capsys)

    def test_a_staged_deletion(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = committed_repo(tmp_path, BASE_FILES)
        git(root, 'rm', '-q', 'beta/src/beta/b.py')
        assert 'beta/src/beta/b.py' in _refused(root, capsys)

    def test_untracked_files_and_files_outside_the_domain_are_not_dirt(
        self, tmp_path: Path
    ) -> None:
        root = committed_repo(tmp_path, BASE_FILES)
        write_files(root, {'alpha/src/alpha/untracked.py': 'U = 1\n', 'hooks/h.py': 'H = 2\n'})
        out = tmp_path / 'out' / 'snapshot.json'
        assert run_snapshot(root, out) == 0
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
        root = committed_repo(tmp_path, BASE_FILES)
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
    def test_neither_a_workspace_nor_a_project_name(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = committed_repo(tmp_path, {**BASE_FILES, 'pyproject.toml': '[tool.ruff]\nline-length = 100\n'})
        err = _refused(root, capsys)
        assert '[project].name' in err
        assert 'tool.uv.workspace' in err

    def test_git_absent(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = committed_repo(tmp_path, BASE_FILES)
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
        root = committed_repo(tmp_path, BASE_FILES)
        monkeypatch.setattr(source_measures, 'complexipy_version', lambda: '7.0.1')
        err = _refused(root, capsys)
        assert '7.0.1' in err
        assert source_measures.COMPLEXIPY_REQUIRED in err

    def test_no_head_commit(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = tmp_path / 'repo'
        root.mkdir()
        git(root, 'init', '-q')
        write_files(root, {'pyproject.toml': WORKSPACE_PYPROJECT, **BASE_FILES})
        git(root, 'add', '-A')
        assert 'HEAD' in _refused(root, capsys)


class TestABadPreviousSnapshotIsRefusedBeforeMeasuring:
    def test_a_missing_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        root = committed_repo(tmp_path, BASE_FILES)
        previous = tmp_path / 'missing.json'
        assert str(previous) in _refused(root, capsys, '--diff', str(previous))

    def test_a_file_that_is_not_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = committed_repo(tmp_path, BASE_FILES)
        previous = tmp_path / 'previous.json'
        previous.write_text('not json\n', encoding='utf-8')
        assert str(previous) in _refused(root, capsys, '--diff', str(previous))

    def test_another_instruments_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = committed_repo(tmp_path, BASE_FILES)
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
        root = committed_repo(tmp_path, BASE_FILES)
        measured = tmp_path / 'measured.json'
        assert run_snapshot(root, measured) == 0
        previous = tmp_path / 'previous.json'
        previous.write_text(
            json.dumps({**json.loads(measured.read_text(encoding='utf-8')), 'schema_version': 4}),
            encoding='utf-8',
        )
        capsys.readouterr()
        err = _refused(root, capsys, '--diff', str(previous))
        assert str(previous) in err
        assert 'schema_version' in err


#: A src and a tests record of the summary fixture.
_SRC_PATH = 'alpha/src/alpha/mod.py'
_TESTS_PATH = 'alpha/tests/test_mod.py'


def _setting(*keys: str | int, to: object) -> SnapshotEdit:
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


def _bump(field: str, by: int = 1) -> SnapshotEdit:
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
        assert run_snapshot(committed_repo(tmp_path, _SUMMARY_FILES), path) == 0
        return read_json(path)

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
        edit: SnapshotEdit,
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
        edit: SnapshotEdit,
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
        edit: SnapshotEdit,
    ) -> None:
        given, err = self._refusal(taken, tmp_path, capsys, edit)
        assert str(given) in err
        assert where in err


def test_two_src_files_importing_as_one_module_name_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The import graph is keyed by module name, so an ambiguous one would be a
    # wrong graph.
    root = committed_repo(tmp_path, {**BASE_FILES, 'alpha/src/x.py': 'Y = 1\n'})
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
