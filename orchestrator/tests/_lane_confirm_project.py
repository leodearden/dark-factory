"""Tiny on-disk projects for the offline lane's real-pytest confirm scenarios (task 6858).

``lane_project`` writes a pytest project whose ``tests/test_lane.py`` holds the
named tests of ``LANE_TESTS``; each test's behaviour across runs is fixed by a
marker file it writes on its first run.  A ``Sabotage`` armed in the project
breaks the next pytest session that collects its tests.  ``write_script`` puts
an executable stub of a legacy leg's runner script into a worktree.
"""

from __future__ import annotations

import stat
from enum import StrEnum
from pathlib import Path

LANE_FILE = 'tests/test_lane.py'

_PYPROJECT = '[tool.pytest.ini_options]\naddopts = "-p no:cacheprovider"\n'

_PRELUDE = '''\
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
    'test_red_on_rerun': "def test_red_on_rerun():\n    assert _first_run('red_on_rerun')\n",
    'test_skips_on_rerun': (
        'def test_skips_on_rerun():\n'
        "    if not _first_run('skips_on_rerun'):\n"
        "        pytest.skip('held')\n"
        '    assert 1 == 2\n'
    ),
    'test_uses_worker_id': "def test_uses_worker_id(worker_id):\n    assert worker_id.startswith('gw')\n",
    'test_param': (
        "@pytest.mark.parametrize('word', ['x y'])\n"
        'def test_param(word):\n'
        "    assert word == 'xy'\n"
    ),
    'test_ok': 'def test_ok():\n    pass\n',
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


def lane_project(root: Path, *tests: str) -> Path:
    """A pytest project at *root* whose lane file holds *tests*, in that order."""
    tests_dir = root / 'tests'
    tests_dir.mkdir(parents=True)
    (root / 'pyproject.toml').write_text(_PYPROJECT, encoding='utf-8')
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
