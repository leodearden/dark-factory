"""Pins scripts/tests' directory-wide tempdir confinement (task 11128, esc-5296-7):
decision D15's mechanism in plans/review-all-s20-cli-invocation-and-usage-gate-prd.md.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from legibility import account_pool
from legibility.session_runner import SessionRunner
from shared.config_dir import CONFIG_DIR_PREFIX, reset_sweep_once_state

_SESSION_PREFIX = CONFIG_DIR_PREFIX + 'legibility-session-'


def test_every_test_gets_a_private_empty_tempdir_outside_tmp_path(
    private_tempdir, tmp_path, tmp_path_factory,
):
    assert Path(tempfile.gettempdir()) == private_tempdir, (
        f'tempfile.gettempdir() is {tempfile.gettempdir()}, not this test\'s '
        f'private tempdir {private_tempdir}'
    )
    assert private_tempdir.is_relative_to(tmp_path_factory.getbasetemp()), (
        f'{private_tempdir} lies outside pytest\'s basetemp '
        f'{tmp_path_factory.getbasetemp()}'
    )
    assert not private_tempdir.is_relative_to(tmp_path), (
        f'{private_tempdir} lies inside tmp_path {tmp_path}, adding an entry '
        f'to every test\'s tmp_path'
    )
    assert list(private_tempdir.iterdir()) == [], (
        f'{private_tempdir} is not empty: {sorted(private_tempdir.iterdir())}'
    )


def test_a_session_runner_sweeps_and_keeps_its_config_dir_inside_the_private_tempdir(
    private_tempdir, plant_dead_pid_dir, pool_roster, sentinel_login,
):
    assert Path(tempfile.gettempdir()) == private_tempdir, (
        f'tempfile.gettempdir() is {tempfile.gettempdir()}, not {private_tempdir}; '
        f'refusing to build a SessionRunner that would sweep it'
    )
    ghost = plant_dead_pid_dir(private_tempdir, _SESSION_PREFIX)
    reset_sweep_once_state(_SESSION_PREFIX)
    accounts_file, env_file = pool_roster('a')
    gate = account_pool.build_pool(accounts_file=accounts_file, env_file=env_file)

    with SessionRunner(gate, label='tempdir-confinement'):
        config_dir = private_tempdir / f'{_SESSION_PREFIX}{os.getpid()}'
        assert config_dir.is_dir(), (
            f'the SessionRunner\'s config dir {config_dir} is not in the private '
            f'tempdir; it holds {sorted(private_tempdir.iterdir())}'
        )

    assert not ghost.exists(), (
        f'the dead-PID ghost {ghost} survived: either the SessionRunner\'s sweep '
        f'did not run under the private tempdir, or _SESSION_PREFIX no longer '
        f'matches the prefix scripts/legibility/session_runner.py sweeps'
    )
