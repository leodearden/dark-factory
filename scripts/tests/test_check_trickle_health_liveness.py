"""Tests for scripts/legibility/check_trickle_health_liveness.sh — did
legibility-trickle-health@<project>.service run within the window AND reach
a verdict (task 5862)?

The accepted exit statuses are read from check_trickle_health's own
constants, so renumbering a verdict code without updating the probe fails
here. Harness: scripts/tests/systemd_show_fakes.py.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from legibility import check_trickle_health
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT
from systemd_show_fakes import run_unit_probe, systemd_timestamp

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPT = (
    Path(__file__).parent.parent / 'legibility' / 'check_trickle_health_liveness.sh'
)
SERVICE = 'legibility-trickle-health@proj_a.service'


def _fields(result, exit_status, *, age_hours=2):
    exited_at = datetime.now(UTC) - timedelta(hours=age_hours)
    return {
        'Result': result,
        'ExecMainStatus': exit_status,
        'ExecMainExitTimestamp': systemd_timestamp(exited_at),
    }


def _probe(tmp_path, fields, args=('proj_a', '24')):
    run = run_unit_probe(SCRIPT, tmp_path, *args, fields=fields)
    assert not run.git_called, 'the liveness probe must never invoke git'
    return run


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), (
        f'Expected {SCRIPT} to be executable (os.X_OK); it is not. '
        f'Run: chmod +x {SCRIPT}'
    )


def test_a_fresh_healthy_run_is_alive(tmp_path):
    run = _probe(
        tmp_path, _fields('success', str(check_trickle_health.EXIT_HEALTHY)),
    )

    assert run.result.returncode == 0, (
        f'stdout={run.result.stdout!r} stderr={run.result.stderr!r}'
    )
    assert any(
        call[0] == 'show' and SERVICE in call for call in run.systemctl_calls
    ), f'Expected a `systemctl show ... {SERVICE}` probe; calls={run.systemctl_calls!r}'


def test_a_fresh_degraded_verdict_is_alive(tmp_path):
    """A DEGRADED verdict is the health unit WORKING: it ran every probe and
    is correctly reporting a broken trickle."""
    run = _probe(
        tmp_path, _fields('exit-code', str(check_trickle_health.EXIT_DEGRADED)),
    )

    assert run.result.returncode == 0, (
        f'stdout={run.result.stdout!r} stderr={run.result.stderr!r}'
    )


@pytest.mark.parametrize('result, exit_status', [
    pytest.param(
        'exit-code', str(check_trickle_health.EXIT_UNRESOLVED),
        id='crash-or-unresolved',
    ),
    pytest.param('exit-code', '2', id='usage-error'),
    pytest.param('exit-code', '203', id='exec-failure'),
    pytest.param('timeout', '0', id='timeout'),
    pytest.param('signal', '15', id='signal'),
    pytest.param('oom-kill', '9', id='oom-kill'),
])
def test_a_fresh_run_that_reached_no_verdict_is_not_alive(
    tmp_path, result, exit_status
):
    run = _probe(tmp_path, _fields(result, exit_status))

    assert run.result.returncode != 0, (
        f'{result}/{exit_status} is a broken health unit, not a verdict; '
        f'stdout={run.result.stdout!r} stderr={run.result.stderr!r}'
    )
    assert f"'{result}'" in run.result.stderr
    assert f"'{exit_status}'" in run.result.stderr


def test_a_stale_run_is_not_alive(tmp_path):
    run = _probe(
        tmp_path,
        _fields('success', str(check_trickle_health.EXIT_HEALTHY), age_hours=48),
    )

    assert run.result.returncode != 0, (
        f'stdout={run.result.stdout!r} stderr={run.result.stderr!r}'
    )


def test_a_unit_that_never_ran_is_not_alive(tmp_path):
    run = _probe(
        tmp_path,
        {'Result': '', 'ExecMainStatus': '', 'ExecMainExitTimestamp': ''},
    )

    assert run.result.returncode != 0, (
        f'stdout={run.result.stdout!r} stderr={run.result.stderr!r}'
    )


def test_an_unparseable_timestamp_is_not_alive(tmp_path):
    run = _probe(
        tmp_path,
        {
            'Result': 'success',
            'ExecMainStatus': str(check_trickle_health.EXIT_HEALTHY),
            'ExecMainExitTimestamp': 'not-a-date',
        },
    )

    assert run.result.returncode != 0, (
        f'stdout={run.result.stdout!r} stderr={run.result.stderr!r}'
    )


def test_a_wrong_argument_count_is_a_usage_error(tmp_path):
    run = _probe(
        tmp_path, _fields('success', str(check_trickle_health.EXIT_HEALTHY)),
        args=('proj_a',),
    )

    assert run.result.returncode != 0
    assert 'usage' in run.result.stderr
