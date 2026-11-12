"""Every orchestrator unit test reads this project's OPERATIONAL config file.

``conftest.py``'s autouse ``_isolate_orch_config`` provides that binding, and
task 3886 decided to keep it. The decision record is
``docs/config-retune-gate.md``. This test observes the binding where it
actually runs: in the environment the fixture leaves for each test.

One knob is masked inside that binding: ``verify_use_cgroup_scope`` reaches out
of the test sandbox into the operator's systemd user manager, so no test may
take it from the yaml.
"""

import os
from pathlib import Path

from orchestrator.config import OrchestratorConfig

OPERATIONAL_CONFIG = Path(__file__).resolve().parents[2] / 'dark-factory-orchestrator.yaml'


def test_the_suite_is_bound_to_the_operational_config_file():
    bound = os.environ.get('ORCH_CONFIG_PATH')

    assert bound is not None, (
        'ORCH_CONFIG_PATH is unset inside an orchestrator test, so a bare '
        'OrchestratorConfig() loads only the package defaults instead of '
        f'{OPERATIONAL_CONFIG}. See docs/config-retune-gate.md before changing '
        'the binding.'
    )
    assert Path(bound).resolve() == OPERATIONAL_CONFIG.resolve(), (
        f'ORCH_CONFIG_PATH is bound at {bound!r}, not the operational '
        f'{OPERATIONAL_CONFIG}. Unit tests would read values the factory does '
        'not run at. See docs/config-retune-gate.md before changing the binding.'
    )
    assert Path(bound).is_file(), (
        f'ORCH_CONFIG_PATH is bound at {bound!r}, which does not exist. The '
        'loader skips a missing file without raising, so every test would '
        'silently read the package defaults.'
    )


def test_a_bound_test_never_spawns_a_real_verify_scope(monkeypatch, tmp_path):
    scoped_config = tmp_path / 'scoped.yaml'
    scoped_config.write_text('verify_use_cgroup_scope: true\n')
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(scoped_config))

    assert OrchestratorConfig().verify_use_cgroup_scope is False, (
        'A bare OrchestratorConfig() took verify_use_cgroup_scope=true from the '
        'yaml, so verify.py::_run_cmd would launch every verify command through '
        "`systemd-run --user --scope`, creating real transient units in the "
        "operator's user manager. conftest.py::_isolate_orch_config pins it off."
    )
    assert OrchestratorConfig(verify_use_cgroup_scope=True).verify_use_cgroup_scope is True, (
        'An explicit verify_use_cgroup_scope=True kwarg must still beat the pin: '
        'test_verify_scope_cpu_weight.py opts into scopes that way.'
    )
