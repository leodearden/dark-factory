"""The lock states one task's lock can be in, and lockChipStateFor run over them.

lockChipStateFor (scheduler_utils.jsx) is the one lock classifier given a
task's view of one SCHEDULER.modules entry. test_lock_chip_state.py executes it
beside the Scheduler heatmap's cellStateFor; test_tab_scheduler.py checks that
styles.css colours every class it answers. Both take the matrix from here.

Executing it needs node: absent from PATH it skips, or fails when CI is set.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
from collections.abc import Iterable
from typing import Any

import pytest

SCHED_UTILS_PATH = str(
    pathlib.Path(__file__).parent.parent / 'src/dashboard/static/redux/scheduler_utils.jsx'
)

ROW_TASK = 'B'
ROW_PROJECT = 'P'
MODULE_PATH = 'm'

# Each lock state a module in the row's lock set can be in, as module fields.
CELL_MATRIX: dict[str, dict[str, Any]] = {
    'free': {},
    "held by the row's own task": {'holder': ROW_TASK, 'holder_project': ROW_PROJECT},
    "held by the row's task id in another project": {'holder': ROW_TASK, 'holder_project': 'Q'},
    'held by another task': {'holder': 'A', 'holder_project': ROW_PROJECT},
    'parked only': {'parked_by': 'C', 'parked_owner_live': True},
    "parked by the row's own task": {'parked_by': ROW_TASK, 'parked_owner_live': True},
    'parked and held': {
        'holder': 'A', 'holder_project': ROW_PROJECT, 'parked_by': ROW_TASK, 'parked_owner_live': True,
    },
    'parked by a dead owner': {'parked_by': 'C', 'parked_owner_live': False},
}


def node_path() -> str:
    path = shutil.which('node')
    if not path:
        if os.environ.get('CI'):
            pytest.fail('node is required in CI but not found on PATH')
        pytest.skip('node not available')
    return path


def heatmap_module(**fields: Any) -> dict[str, Any]:
    return {'project': ROW_PROJECT, 'path': MODULE_PATH, 'holder': None, 'holder_project': None,
            'parked_by': None, 'parked_owner_live': None, **fields}


def heatmap_row(module: dict[str, Any], lock_set: Iterable[str] = (MODULE_PATH,)) -> dict[str, Any]:
    """Task B's scheduler row; parked on the module exactly when the module says B parked it."""
    parked = [module['path']] if module.get('parked_by') == ROW_TASK else []
    return {'project': ROW_PROJECT, 'task_id': ROW_TASK, 'lock_set': list(lock_set),
            'park_state': {'modules': parked} if parked else None}


_CHIP_DRIVER = r"""
const vm = require('vm');
const fs = require('fs');
const [utilsPath, casesJson] = process.argv.slice(1);
const context = vm.createContext({ console, window: {} });
vm.runInContext(fs.readFileSync(utilsPath, 'utf8'), context, { filename: utilsPath });
context.__cases = JSON.parse(casesJson);
const out = vm.runInContext(
  '__cases.map(([module, taskId, project]) => window.DF_SCHED_UTILS.lockChipStateFor(module, taskId, project))',
  context,
);
process.stdout.write(JSON.stringify(out) + '\n');
"""


def lock_chip_states_for(cases: Iterable[tuple[Any, str, str]]) -> list[dict[str, Any]]:
    """lockChipStateFor(module, task_id, project) for each case, in one node process."""
    result = subprocess.run(
        [node_path(), '-e', _CHIP_DRIVER, SCHED_UTILS_PATH, json.dumps(list(cases))],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip())


def chip_classes_over_the_matrix() -> set[str]:
    """Every cls lockChipStateFor answers for task B across CELL_MATRIX."""
    cases = [(heatmap_module(**fields), ROW_TASK, ROW_PROJECT) for fields in CELL_MATRIX.values()]
    return {state['cls'] for state in lock_chip_states_for(cases)}
