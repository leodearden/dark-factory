"""Pins scripts/tests' arming of the UsageGate probe-dir sweep confinement (task 6275).

It is armed by ``df_pytest_isolation.py::arm_usage_gate_sweep_confinement``; the
full contract (the opt-out, the still-reclaiming re-base) is
``shared/tests/test_usage_gate_sweep_confinement.py``.
"""

from __future__ import annotations

import tempfile

from shared.config_dir import CONFIG_DIR_PREFIX, reset_sweep_once_state
from shared.config_models import AccountConfig, UsageCapConfig
from shared.usage_gate import UsageGate

_PROBE_PREFIX = CONFIG_DIR_PREFIX + 'usage-gate-probe-'


def test_a_real_gate_sweeps_only_its_private_root(
    tmp_path, monkeypatch, usage_gate_sweep_root, plant_dead_pid_dir,
):
    system_tempdir = tmp_path / 'system-tmp'
    system_tempdir.mkdir()
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(system_tempdir))
    system_ghost = plant_dead_pid_dir(system_tempdir, _PROBE_PREFIX)
    confined_ghost = plant_dead_pid_dir(usage_gate_sweep_root, _PROBE_PREFIX)
    monkeypatch.setenv('DF_TASK_6275_TOKEN_WORK', 'fake-token-work')
    reset_sweep_once_state(_PROBE_PREFIX)

    UsageGate(
        UsageCapConfig(
            accounts=[AccountConfig(name='work', oauth_token_env='DF_TASK_6275_TOKEN_WORK')],
            wait_for_reset=False,
        )
    )

    assert system_ghost.exists(), (
        f'the UsageGate probe-dir sweep reached the system tempdir and reclaimed '
        f'{system_ghost.name}; it must be confined for every test that did not '
        f'request real_probe_dir_sweep'
    )
    assert not confined_ghost.exists(), (
        f'{confined_ghost.name} survived in usage_gate_sweep_root; the confined sweep '
        f'must still reclaim dead-PID dirs (re-base, not stub)'
    )
