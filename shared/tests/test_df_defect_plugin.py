"""The injected pytest plugin ``_df_defect_plugin``, observed through real pytest runs.

Every case runs a fixture project in a subprocess with ``-p _df_defect_plugin``
and the plugin directory on ``PYTHONPATH``, which is how verify injects it
(plans/defect-tests-prd.md decision 19). Nothing here imports the plugin: its
contract is the files it writes and the markers it registers.

Decision 19 also requires the plugin to load in a target that has no ``shared``
installed. The interpreter running these cases DOES have it, so every run puts a
``shared`` package that refuses to import AHEAD of the plugin directory.
``PYTHONPATH`` precedes site-packages and is inherited by xdist workers, so any
``shared`` import from the plugin, in any process, fails the run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from shared.defect_plugin_location import DEFECT_PLUGIN_DIR
from shared.defects import SideOutcome

pytestmark = pytest.mark.timeout(300)

_DEFECT_REF = 'human:20261119T000000Z:abcdef12'
_SECOND_REF = 'human:20261119T000000Z:12345678'
_CLASS_REF = 'review:7045:aaaaaaaaaaaa'

_PROJECT_A_INI = """\
[pytest]
markers =
    own: the project's own marker
filterwarnings = error
pythonpath = .
"""

_PROJECT_A_TESTS = f"""\
import pytest

import testing


@pytest.mark.own
def test_own_marker():
    assert testing.ORIGIN == 'project'


@pytest.mark.defect({_DEFECT_REF!r})
def test_defect_one():
    pass


@pytest.mark.defect({_DEFECT_REF!r})
@pytest.mark.defect({_SECOND_REF!r})
def test_two_refs():
    pass


@pytest.mark.criterion('C-1')
def test_criterion():
    pass


class TestScoped:
    pytestmark = pytest.mark.defect({_CLASS_REF!r})

    def test_method(self):
        pass


@pytest.mark.defect(ref='x')
def test_malformed():
    pass


def test_plain():
    pass
"""

_EXPECTED_MAP_A = {
    'tests/test_a.py::test_defect_one': {'defects': [_DEFECT_REF], 'criterion': None},
    'tests/test_a.py::test_two_refs': {
        'defects': sorted([_DEFECT_REF, _SECOND_REF]),
        'criterion': None,
    },
    'tests/test_a.py::test_criterion': {'defects': [], 'criterion': 'C-1'},
    'tests/test_a.py::TestScoped::test_method': {'defects': [_CLASS_REF], 'criterion': None},
}


def _write_project(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding='utf-8')
    return root


_TARGET_HAS_NO_SHARED = 'targets have no shared (plans/defect-tests-prd.md decision 19)'


def _unimportable_shared_beside(project: Path) -> Path:
    root = project.parent / f'{project.name}-no-shared'
    package = root / 'shared'
    package.mkdir(parents=True, exist_ok=True)
    (package / '__init__.py').write_text(f'raise ImportError({_TARGET_HAS_NO_SHARED!r})\n', encoding='utf-8')
    return root


def _run_pytest(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ('PYTEST_ADDOPTS', 'PYTHONPATH')}
    env['PYTHONPATH'] = os.pathsep.join([str(_unimportable_shared_beside(project)), str(DEFECT_PLUGIN_DIR)])
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-p', '_df_defect_plugin', '-p', 'no:cacheprovider', *args],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )


def _read_map(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    return {
        nodeid: {**entry, 'defects': sorted(entry['defects'])} for nodeid, entry in raw.items()
    }


@pytest.fixture(scope='module')
def project_a(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_project(
        tmp_path_factory.mktemp('project_a'),
        {
            'pytest.ini': _PROJECT_A_INI,
            'testing.py': "ORIGIN = 'project'\n",
            'tests/test_a.py': _PROJECT_A_TESTS,
        },
    )


@pytest.fixture(scope='module')
def project_a_runs(
    project_a: Path, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, tuple[subprocess.CompletedProcess[str], Path]]:
    runs = {}
    for label, extra in (('serial', ()), ('xdist', ('-n', '2'))):
        out = tmp_path_factory.mktemp(f'out_{label}')
        result = _run_pytest(project_a, '--strict-markers', f'--junitxml={out}/r.xml', *extra)
        runs[label] = (result, out)
    return runs


class TestMarkersAndCollectionMap:
    @pytest.mark.parametrize('label', ['serial', 'xdist'])
    def test_target_run_stays_green(self, project_a_runs, label):
        result, _ = project_a_runs[label]
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize('label', ['serial', 'xdist'])
    def test_map_lists_exactly_the_well_formed_marked_items(self, project_a_runs, label):
        _, out = project_a_runs[label]
        assert _read_map(out / 'r.xml.tests.json') == _EXPECTED_MAP_A

    def test_serial_and_xdist_maps_agree(self, project_a_runs):
        serial = _read_map(project_a_runs['serial'][1] / 'r.xml.tests.json')
        xdist = _read_map(project_a_runs['xdist'][1] / 'r.xml.tests.json')
        assert serial == xdist

    def test_no_junitxml_writes_no_map(self, project_a):
        result = _run_pytest(project_a, '--strict-markers')
        assert result.returncode == 0, result.stdout + result.stderr
        for directory in (project_a, project_a / 'tests'):
            assert list(directory.glob('*.tests.json')) == []

    def test_nothing_marked_still_writes_an_empty_map(self, tmp_path):
        project = _write_project(
            tmp_path / 'unmarked',
            {'pytest.ini': '[pytest]\n', 'tests/test_u.py': 'def test_u():\n    pass\n'},
        )
        result = _run_pytest(project, f'--junitxml={tmp_path}/out/r.xml')
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads((tmp_path / 'out' / 'r.xml.tests.json').read_text()) == {}

    def test_markers_listing_keeps_the_projects_own(self, project_a):
        result = _run_pytest(project_a, '--markers')
        assert result.returncode == 0, result.stdout + result.stderr
        for name in ('own', 'defect', 'criterion'):
            assert f'@pytest.mark.{name}' in result.stdout


class TestPluginDirectory:
    def test_holds_only_the_plugin_module(self):
        entries = {entry.name for entry in DEFECT_PLUGIN_DIR.iterdir()} - {'__pycache__'}
        assert entries == {'_df_defect_plugin.py'}


_CRASHING_TESTS = """\
import os


def test_before():
    pass


def test_crash():
    os._exit(1)


def test_after():
    pass
"""


class TestACrashedRunRecordsNothing:
    """A process that dies mid-run leaves no file, so absent keeps meaning "not recorded" under xdist too."""

    def _run_crashing(self, tmp_path: Path, xdist: tuple[str, ...]) -> tuple[subprocess.CompletedProcess[str], Path]:
        project = _write_project(
            tmp_path / 'crash', {'pytest.ini': '[pytest]\n', 'tests/test_crash.py': _CRASHING_TESTS},
        )
        out = tmp_path / 'out'
        result = _run_pytest(project, f'--df-defect-outcomes={out}/o.json', f'--junitxml={out}/r.xml', *xdist)
        assert result.returncode != 0, result.stdout + result.stderr
        return result, out

    @pytest.mark.parametrize('xdist', [(), ('-n', '2')], ids=['serial', 'xdist'])
    def test_a_crash_writes_neither_file(self, tmp_path, xdist):
        result, out = self._run_crashing(tmp_path, xdist)
        assert not (out / 'o.json').exists(), result.stdout + result.stderr
        assert not (out / 'r.xml.tests.json').exists(), result.stdout + result.stderr

    def test_an_xdist_crash_removes_a_stale_file(self, tmp_path):
        out = tmp_path / 'out'
        out.mkdir()
        for name in ('o.json', 'r.xml.tests.json'):
            (out / name).write_text('{"stale": {}}', encoding='utf-8')
        result, out = self._run_crashing(tmp_path, ('-n', '2'))
        assert 'crashed' in result.stdout + result.stderr
        assert sorted(p.name for p in out.iterdir() if p.name.endswith('.json')) == []


class TestTargetsWithoutShared:
    @pytest.mark.parametrize('xdist', [(), ('-n', '2')], ids=['serial', 'xdist'])
    def test_shared_is_unimportable_in_every_process_the_plugin_runs_in(self, tmp_path, xdist):
        project = _write_project(
            tmp_path / 'probe',
            {
                'pytest.ini': '[pytest]\n',
                'tests/test_probe.py': f"""\
                    import pytest


                    @pytest.mark.parametrize('worker', range(4))
                    def test_shared_refuses(worker):
                        with pytest.raises(ImportError, match={re.escape(_TARGET_HAS_NO_SHARED)!r}):
                            import shared
                    """,
            },
        )
        result = _run_pytest(project, *xdist)
        assert result.returncode == 0, result.stdout + result.stderr
        assert '4 passed' in result.stdout


_PROJECT_B_TESTS = """\
import json

import pytest


@pytest.fixture
def attribute_error():
    json.definitely_absent_attr


@pytest.fixture
def nameless_import_error():
    raise ImportError('cannot import', name='m')


@pytest.fixture
def runtime_error():
    raise RuntimeError('boom')


def test_pass():
    pass


def test_assert():
    assert False


def test_skip():
    pytest.skip('not here')


@pytest.mark.xfail
def test_xfail():
    assert False


def test_attribute_fixture(attribute_error):
    pass


def test_nameless_import_fixture(nameless_import_error):
    pass


def test_runtime_fixture(runtime_error):
    pass


def test_call_phase_import():
    from json import definitely_absent_fn
"""


def _record(side_outcome, exc_type=None, module=None, symbol=None, owner_type=None):
    return {
        'side_outcome': side_outcome,
        'exc_type': exc_type,
        'module': module,
        'symbol': symbol,
        'owner_type': owner_type,
    }


_EXPECTED_OUTCOMES_B = {
    'tests/test_b.py::test_pass': _record('passed'),
    'tests/test_b.py::test_assert': _record('failed', 'builtins.AssertionError'),
    'tests/test_b.py::test_skip': _record('skipped'),
    'tests/test_b.py::test_xfail': _record('skipped'),
    'tests/test_b.py::test_attribute_fixture': _record(
        'error_attributed',
        'builtins.AttributeError',
        module='json',
        symbol='definitely_absent_attr',
        owner_type='builtins.module',
    ),
    'tests/test_b.py::test_nameless_import_fixture': _record(
        'error_unattributed', 'builtins.ImportError', module='m'
    ),
    'tests/test_b.py::test_runtime_fixture': _record('error_unattributed', 'builtins.RuntimeError'),
    'tests/test_b.py::test_call_phase_import': _record(
        'failed', 'builtins.ImportError', module='json', symbol='definitely_absent_fn'
    ),
}

_EXPECTED_OUTCOMES_C = {
    'tests/test_new_symbol.py': _record(
        'error_attributed', 'builtins.ImportError', module='json', symbol='definitely_absent_fn'
    ),
    'tests/test_new_module.py': _record(
        'error_attributed',
        'builtins.ModuleNotFoundError',
        module='definitely_absent_module_7045',
    ),
    'tests/test_ok.py::test_ok': _record('passed'),
}


@pytest.fixture(scope='module')
def outcome_runs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[tuple[str, str], tuple[subprocess.CompletedProcess[str], Path]]:
    projects = {
        'B': (
            _write_project(
                tmp_path_factory.mktemp('project_b'),
                {'pytest.ini': '[pytest]\n', 'tests/test_b.py': _PROJECT_B_TESTS},
            ),
            (),
        ),
        'C': (
            _write_project(
                tmp_path_factory.mktemp('project_c'),
                {
                    'pytest.ini': '[pytest]\n',
                    'tests/test_new_symbol.py': 'from json import definitely_absent_fn\n',
                    'tests/test_new_module.py': 'import definitely_absent_module_7045\n',
                    'tests/test_ok.py': 'def test_ok():\n    pass\n',
                },
            ),
            ('--continue-on-collection-errors',),
        ),
    }
    runs = {}
    for name, (project, extra) in projects.items():
        for label, xdist in (('serial', ()), ('xdist', ('-n', '2'))):
            out = tmp_path_factory.mktemp(f'outcomes_{name}_{label}')
            result = _run_pytest(project, f'--df-defect-outcomes={out}/o.json', *extra, *xdist)
            runs[(name, label)] = (result, out / 'o.json')
    return runs


def _read_outcomes(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding='utf-8'))


class TestOutcomeRecorder:
    @pytest.mark.parametrize('label', ['serial', 'xdist'])
    def test_test_phase_outcomes(self, outcome_runs, label):
        result, path = outcome_runs[('B', label)]
        assert path.is_file(), result.stdout + result.stderr
        assert _read_outcomes(path) == _EXPECTED_OUTCOMES_B

    @pytest.mark.parametrize('label', ['serial', 'xdist'])
    def test_collection_errors_record_the_structural_cause(self, outcome_runs, label):
        result, path = outcome_runs[('C', label)]
        assert path.is_file(), result.stdout + result.stderr
        assert _read_outcomes(path) == _EXPECTED_OUTCOMES_C

    @pytest.mark.parametrize('name', ['B', 'C'])
    def test_serial_and_xdist_files_agree(self, outcome_runs, name):
        serial = _read_outcomes(outcome_runs[(name, 'serial')][1])
        xdist = _read_outcomes(outcome_runs[(name, 'xdist')][1])
        assert serial == xdist

    def test_no_option_writes_no_outcome_file(self, tmp_path):
        project = _write_project(
            tmp_path / 'project',
            {'pytest.ini': '[pytest]\n', 'tests/test_u.py': 'def test_u():\n    pass\n'},
        )
        result = _run_pytest(project, f'--junitxml={tmp_path}/out/r.xml')
        assert result.returncode == 0, result.stdout + result.stderr
        assert sorted(p.name for p in (tmp_path / 'out').iterdir()) == ['r.xml', 'r.xml.tests.json']

    def test_records_span_the_side_outcome_vocabulary(self, outcome_runs):
        recorded = {
            record['side_outcome']
            for _, path in outcome_runs.values()
            for record in _read_outcomes(path).values()
        }
        assert recorded <= {outcome.value for outcome in SideOutcome}
        assert recorded == {outcome.value for outcome in SideOutcome} - {SideOutcome.NOT_COLLECTED.value}
