"""Tests of ``scripts/quality_metrics_snapshot.py``, the whole-repo metrics report.

Every fixture is a small REAL git repository whose root pyproject.toml names
two members, measured through the script's public entry points; no measure is
patched. The one real-tree test (cluster agreement) is the PRD's row 12.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
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
        assert measured['schema_version'] == 1
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


class TestTheImportGraph:
    @pytest.fixture
    def measured(self, tmp_path: Path) -> dict[str, Any]:
        return _measured(tmp_path, _GRAPH)

    def test_it_has_exactly_the_four_sections(self, measured: dict[str, Any]) -> None:
        assert set(measured['import_graph']) == {'edges', 'reach_back', 'deferred', 'cycles'}

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

    def test_a_name_from_the_package_init_is_a_reach_back_and_a_submodule_is_not(
        self, measured: dict[str, Any]
    ) -> None:
        assert measured['import_graph']['reach_back'] == [
            {'from': 'pkg.a', 'to': 'pkg', 'names': ['THING'], 'line': 1},
        ]

    def test_deferred_imports_are_listed_by_site(self, measured: dict[str, Any]) -> None:
        deferred = measured['import_graph']['deferred']
        assert deferred == [{'from': 'e', 'line': 5, 'imports': ['a']}]
        assert (
            len([entry for entry in deferred if entry['from'] == 'e'])
            == measured['files']['alpha/src/e.py']['function_local_imports']
        )

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
            'reach_back_imports', 'fan_out', 'fan_in_src', 'fan_in_tests',
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
    'reach_back_imports', 'fan_out', 'fan_in_src', 'fan_in_tests',
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
            json.dumps({**json.loads(measured.read_text(encoding='utf-8')), 'schema_version': 2}),
            encoding='utf-8',
        )
        capsys.readouterr()
        err = _refused(root, capsys, '--diff', str(previous))
        assert str(previous) in err
        assert 'schema_version' in err


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
