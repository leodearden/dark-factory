"""Keep scheduler PSI tests independent of the project yaml's ``psi_admission``.

A bare ``OrchestratorConfig(...)`` built in a test inherits whatever
``psi_admission`` block the project yaml carries, because conftest's autouse
``_isolate_orch_config`` pins ``ORCH_CONFIG_PATH`` at the live yaml. A scheduler
test that silently relies on a psi_admission CODE default therefore breaks the
day an operator tunes that block. Running such a file against the yaml written
here, whose block differs from code on every leaf, makes that dependency fail at
once instead (task 3590, PRD ``plans/load-throttle-harmonisation-prd.md``
boundary row 10). ``cpu_arm_cfg`` is the explicit config such a test passes.

A test module opts in by binding the fixture under a module-level name:
``_nondefault_psi_admission_yaml = _psi_admission_yaml.nondefault_psi_admission_yaml``.
Bind it through the module, not a ``from`` import: a fixture is registered under
every module-level name that holds it, and an autouse one runs once per name.

The block's values are valid under both the v1 and v2 ``PsiAdmissionConfig``
schemas; a schema without a leaf drops it silently.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from orchestrator.config import PsiAdmissionConfig


@pytest.fixture(autouse=True)
def nondefault_psi_admission_yaml(monkeypatch, tmp_path) -> Path:
    """Point ``ORCH_CONFIG_PATH`` at a copy of its yaml with a non-default psi_admission block.

    Every other value is kept, so the copy still carries the operational
    values (``lock_depth``, ``merge_verify_breadth``) scheduler tests run under.
    Returns the path of the copy, written under ``tmp_path``.
    """
    source = Path(os.environ['ORCH_CONFIG_PATH'])
    config = yaml.safe_load(source.read_text()) or {}
    config['psi_admission'] = {
        'enabled': False,
        'cpu_some_avg10': 70.0,
        'mem_some_avg10': 1.0,
        'mem_full_avg10': 0.5,
        'io_some_avg10': 2.0,
        'runqueue_ratio': 0.5,
        'own_cpu_some_avg10': 1.0,
        'min_inflight_floor': 3,
    }
    target = tmp_path / 'orchestrator-nondefault-psi-admission.yaml'
    target.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(target))
    return target


def cpu_arm_cfg(**overrides) -> PsiAdmissionConfig:
    """A fresh gate config whose CPU arm, which these tests drive, is explicit.

    Fresh per call, never a shared instance: apply_reload mutates submodels in
    place.
    """
    return PsiAdmissionConfig(cpu_some_avg10=85.0, **overrides)
