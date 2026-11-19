"""The injected pytest plugin ``_df_defect_plugin``, observed through real pytest runs.

Every case runs a fixture project in a subprocess with ``-p _df_defect_plugin``
and the plugin directory as the ONLY ``PYTHONPATH`` entry, which is how verify
injects it (plans/defect-tests-prd.md decision 19). Nothing here imports the
plugin: its contract is the files it writes and the markers it registers.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import shared

pytestmark = pytest.mark.timeout(300)

_PLUGIN_DIR = Path(shared.__file__).parent / '_pytest_inject'

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


def _run_pytest(project: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in ('PYTEST_ADDOPTS', 'PYTHONPATH')}
    env['PYTHONPATH'] = str(_PLUGIN_DIR)
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
        entries = {entry.name for entry in _PLUGIN_DIR.iterdir()} - {'__pycache__'}
        assert entries == {'_df_defect_plugin.py'}
