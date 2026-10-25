"""conftest.py's ``_drain_async_mock_coroutines`` fails the test that left an
AsyncMock call un-awaited, at that test's own teardown (task 5669).

An in-process test cannot observe its own teardown failing, so the probe runs
in a REAL nested pytest session under this subproject's inifile and conftest.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _probe_session import run_probe_session

_PROBE = Path(__file__).with_name('_async_mock_orphan_probe.py')
_KILL_SECS = 120


@pytest.fixture(scope='module')
def probe_run(tmp_path_factory: pytest.TempPathFactory) -> subprocess.CompletedProcess[str]:
    return run_probe_session(
        _PROBE, basetemp=tmp_path_factory.mktemp('async-mock-orphan-probe'),
        kill_secs=_KILL_SECS, backstop_secs=_KILL_SECS * 8,
    )


def _tails(run: subprocess.CompletedProcess[str]) -> str:
    return f'rc={run.returncode}\n{run.stdout[-3000:]}{run.stderr[-2000:]}'


def test_the_leaking_test_fails_at_its_own_teardown_naming_the_mock(
    probe_run: subprocess.CompletedProcess[str],
) -> None:
    assert probe_run.returncode == 1, _tails(probe_run)
    assert '2 passed, 1 error' in probe_run.stdout, _tails(probe_run)
    assert (
        'ERROR at teardown of test_leaves_an_async_mock_call_unawaited' in probe_run.stdout
    ), _tails(probe_run)
    assert 'planted_notifier' in probe_run.stdout, _tails(probe_run)


def test_a_sibling_that_awaits_its_mock_is_not_blamed(
    probe_run: subprocess.CompletedProcess[str],
) -> None:
    assert 'awaited_notifier' not in probe_run.stdout, _tails(probe_run)
    blamed = [
        line
        for line in probe_run.stdout.splitlines()
        if ('ERROR' in line or 'FAILED' in line) and 'test_awaits_its_async_mock_call' in line
    ]
    assert blamed == [], _tails(probe_run)
