"""Every orchestrator unit test reads this project's OPERATIONAL config file.

``conftest.py``'s autouse ``_isolate_orch_config`` provides that binding, and
task 3886 decided to keep it. The decision record is
``docs/config-retune-gate.md``. This test observes the binding where it
actually runs: in the environment the fixture leaves for each test.
"""

import os
from pathlib import Path

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
