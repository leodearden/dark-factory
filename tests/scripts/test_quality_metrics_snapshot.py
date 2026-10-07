"""Tests of ``scripts/quality_metrics_snapshot.py``, the whole-repo metrics report.

Every fixture is a small REAL git repository whose root pyproject.toml names
two members, measured through the script's public entry points; no measure is
patched. The one real-tree test (cluster agreement) is the PRD's row 12.
"""
from __future__ import annotations

import dataclasses
import json
import os
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
