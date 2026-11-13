"""Tiny on-disk projects for the offline lane's real-pytest confirm scenarios (task 6858).

``lane_project`` writes a pytest project whose ``tests/test_lane.py`` holds the
named tests of ``LANE_TESTS``; each test's behaviour across runs is fixed by a
marker file it writes on its first run.  A ``Sabotage`` armed in the project
breaks the next pytest session that collects its tests.  ``write_script`` puts
an executable stub of a legacy leg's runner script into a worktree.

Every session a lane project runs keeps its cache off and its basetemp inside
the project, never in the host-wide ``/tmp/pytest-of-<user>`` root, whose
exit-time cleanup of other sessions' dead dirs is unbounded; the contract is
``test_offline_lane_confirm.py::TestRealPytestGenericLeg::test_a_pooled_detection_leaves_the_hosts_default_pytest_temp_root_untouched``.
"""

from __future__ import annotations

import json
import stat
from enum import StrEnum
from pathlib import Path

LANE_FILE = 'tests/test_lane.py'

_BASE_ADDOPTS = '-p no:cacheprovider --basetemp=.lane-basetemp'

_TEMP_ROOT_RECORD = '.inherited-temproot'

_PRELUDE = '''\
import os
import pathlib

import pytest

_HERE = pathlib.Path(__file__).parent


def _first_run(name):
    mark = _HERE / f'.{name}.ran'
    first = not mark.exists()
    mark.touch()
    return first
'''

LANE_TESTS = {
    'test_steady_fail': 'def test_steady_fail():\n    assert 1 == 2\n',
    'test_flaky': "def test_flaky():\n    assert not _first_run('flaky')\n",
    'test_grouped_flaky': (
        "@pytest.mark.xdist_group('lane_group')\n"
        'def test_grouped_flaky():\n'
        "    assert not _first_run('grouped_flaky')\n"
    ),
    'test_red_on_rerun': "def test_red_on_rerun():\n    assert _first_run('red_on_rerun')\n",
    'test_skips_on_rerun': (
        'def test_skips_on_rerun():\n'
        "    if not _first_run('skips_on_rerun'):\n"
        "        pytest.skip('held')\n"
        '    assert 1 == 2\n'
    ),
    'test_uses_worker_id': "def test_uses_worker_id(worker_id):\n    assert worker_id.startswith('gw')\n",
    'test_flaky_uses_worker_id': (
        'def test_flaky_uses_worker_id(worker_id):\n'
        "    assert not _first_run('flaky_worker_id')\n"
    ),
    'test_param': (
        "@pytest.mark.parametrize('word', ['x y'])\n"
        'def test_param(word):\n'
        "    assert word == 'xy'\n"
    ),
    'test_ok': 'def test_ok():\n    pass\n',
    'test_records_its_temp_root': (
        'def test_records_its_temp_root():\n'
        f'    (_HERE / {_TEMP_ROOT_RECORD!r}).write_text('
        "os.environ.get('PYTEST_DEBUG_TEMPROOT', ''), encoding='utf-8')\n"
    ),
}


class Sabotage(StrEnum):
    CRASH_AT_IMPORT = '.crash-at-import'
    KILL_AFTER_FIRST_TEST = '.kill-after-first-test'


_CONFTEST = f'''\
import os
import pathlib
import signal

_HERE = pathlib.Path(__file__).parent

if (_HERE / {Sabotage.CRASH_AT_IMPORT.value!r}).exists():
    raise RuntimeError('the session crashed before collecting')


def pytest_runtest_logfinish(nodeid, location):
    if (_HERE / {Sabotage.KILL_AFTER_FIRST_TEST.value!r}).exists():
        os.kill(os.getpid(), signal.SIGKILL)
'''


def node_id(test: str) -> str:
    return f'{LANE_FILE}::{test}'


def inherited_temp_root(project: Path) -> str:
    """The ``PYTEST_DEBUG_TEMPROOT`` that *project*'s ``test_records_its_temp_root`` saw."""
    return (project / 'tests' / _TEMP_ROOT_RECORD).read_text(encoding='utf-8')


def write_pytest_config(project: Path, addopts: str = '') -> None:
    options = f'{_BASE_ADDOPTS} {addopts}'.strip()
    (project / 'pyproject.toml').write_text(
        f'[tool.pytest.ini_options]\naddopts = {json.dumps(options)}\n', encoding='utf-8',
    )


def lane_project(root: Path, *tests: str, addopts: str = '') -> Path:
    """A pytest project at *root* whose lane file holds *tests*, in that order;
    *addopts* extends the project's base pytest options."""
    tests_dir = root / 'tests'
    tests_dir.mkdir(parents=True)
    write_pytest_config(root, addopts)
    (tests_dir / 'conftest.py').write_text(_CONFTEST, encoding='utf-8')
    body = '\n\n'.join(LANE_TESTS[test] for test in tests)
    (root / LANE_FILE).write_text(f'{_PRELUDE}\n\n{body}', encoding='utf-8')
    return root


def arm(project: Path, sabotage: Sabotage) -> None:
    (project / 'tests' / sabotage.value).touch()


def disarm(project: Path, sabotage: Sabotage) -> None:
    (project / 'tests' / sabotage.value).unlink(missing_ok=True)


def write_script(worktree: Path, relative: str, body: str) -> None:
    """An executable ``sh`` script at *worktree*/*relative* running *body*."""
    script = worktree / relative
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f'#!/bin/sh\n{body}', encoding='utf-8')
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
