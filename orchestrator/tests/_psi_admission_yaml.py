"""Project yaml whose ``psi_admission`` block differs from code on every leaf.

A bare ``OrchestratorConfig(...)`` built in a test inherits whatever
``psi_admission`` block the project yaml carries, because conftest's autouse
``_isolate_orch_config`` pins ``ORCH_CONFIG_PATH`` at the live yaml. A scheduler
test that silently relies on a psi_admission CODE default therefore breaks the
day an operator tunes that block. Running such a file against this yaml makes
that dependency fail at once instead (task 3590, PRD
``plans/load-throttle-harmonisation-prd.md`` boundary row 10).

The block's values are valid under both the v1 and v2 ``PsiAdmissionConfig``
schemas; a schema without a leaf drops it silently.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def write_config_with_nondefault_psi_admission(tmp_path: Path) -> Path:
    """Copy the yaml at ``ORCH_CONFIG_PATH`` with a non-default psi_admission block.

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
    return target
