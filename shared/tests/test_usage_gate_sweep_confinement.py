"""Contract for ``conftest.py``'s suite-wide confinement of the UsageGate probe-dir sweep.

``UsageGate.__init__`` reclaims dead-PID probe dirs under the system tempdir, so
every gate a test builds must have that sweep re-based onto a private per-test
root unless the test opts out with ``real_probe_dir_sweep``. These tests pin the
autouse fixture's effect from inside the suite it guards, against a stand-in
system tempdir, so neither the hazard nor the check can ever reach the real one.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from _usage_gate_test_helpers import make_gate
from test_config_dir import PROBE_PREFIX, find_dead_pid, plant

from shared.config_dir import reset_sweep_once_state


@pytest.fixture(autouse=True)
def _fresh_usage_gate_sweep_mark():
    """Make each test's gate genuinely sweep, whichever test marked the prefix first."""
    reset_sweep_once_state(PROBE_PREFIX)
    yield
    reset_sweep_once_state(PROBE_PREFIX)


@pytest.fixture
def system_tempdir(tmp_path, monkeypatch) -> Path:
    """A private stand-in for the system tempdir, for the unconfined sweep and probe dirs alike."""
    stand_in = tmp_path / 'system-tmp'
    stand_in.mkdir()
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(stand_in))
    return stand_in


def test_a_gate_that_did_not_opt_in_never_sweeps_the_system_tempdir(system_tempdir):
    ghost = plant(system_tempdir, f'{PROBE_PREFIX}ghost-{find_dead_pid()}')

    make_gate(['work'])

    assert ghost.exists(), (
        f'the UsageGate probe-dir sweep reached the system tempdir and reclaimed '
        f'{ghost.name}; it must be confined for every test that did not opt out'
    )


def test_the_confined_sweep_still_reclaims_dead_pid_dirs(system_tempdir, usage_gate_sweep_root):
    ghost = plant(usage_gate_sweep_root, f'{PROBE_PREFIX}ghost-{find_dead_pid()}')
    live_sibling = plant(usage_gate_sweep_root, f'{PROBE_PREFIX}alive-{os.getpid()}')

    make_gate(['work'])

    assert not ghost.exists(), f'{ghost.name} should have been reclaimed by the confined sweep'
    assert live_sibling.exists()
    assert (live_sibling / '.credentials.json').exists()


def test_real_probe_dir_sweep_restores_the_unconfined_sweep(real_probe_dir_sweep, system_tempdir):
    ghost = plant(system_tempdir, f'{PROBE_PREFIX}ghost-{find_dead_pid()}')

    make_gate(['work'])

    assert not ghost.exists(), f'{ghost.name} should have been reclaimed by the real sweep'
