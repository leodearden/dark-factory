"""The lock states one task's lock can be in, and lockChipStateFor run over them.

lockChipStateFor (scheduler_utils.jsx) is the one lock classifier given a
task's view of one SCHEDULER.modules entry. :func:`heatmap_cell_and_chip` runs
it beside the Scheduler heatmap's cellStateFor, which is EXECUTED, not grepped:
its declaration is sliced out of scheduler_heatmap.jsx (raising on a miss) and
run in one vm context with the two classic scripts it reaches at runtime,
scheduler_heatmap_bounds.js for rowTouchesModule and scheduler_utils.jsx for
window.DF_SCHED_UTILS. test_lock_chip_state.py drives both over the matrix,
test_boundary_js.py over a lock the real /scheduler route served, and
test_tab_scheduler.py checks that styles.css colours every class the chip answers.

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
from _dashboard_helpers import extract_function_body, find_function_params

_REDUX_DIR = pathlib.Path(__file__).parent.parent / 'src/dashboard/static/redux'
SCHED_UTILS_PATH = str(_REDUX_DIR / 'scheduler_utils.jsx')
HEATMAP_BOUNDS_PATH = str(_REDUX_DIR / 'scheduler_heatmap_bounds.js')
HEATMAP_PATH = _REDUX_DIR / 'scheduler_heatmap.jsx'

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


_CELL_DRIVER = r"""
const vm = require('vm');
const fs = require('fs');
const [boundsPath, utilsPath, cellStateForSrc, argsJson] = process.argv.slice(1);
const context = vm.createContext({ console, window: {} });
vm.runInContext(fs.readFileSync(boundsPath, 'utf8'), context, { filename: boundsPath });
vm.runInContext(fs.readFileSync(utilsPath, 'utf8'), context, { filename: utilsPath });
vm.runInContext(cellStateForSrc, context, { filename: 'scheduler_heatmap.jsx::cellStateFor' });
context.__args = JSON.parse(argsJson);
const out = vm.runInContext(
  '({ cell: cellStateFor(__args[0], __args[1]), chip: window.DF_SCHED_UTILS.lockChipStateFor(__args[1], __args[0].task_id, __args[0].project) })',
  context,
);
process.stdout.write(JSON.stringify(out) + '\n');
"""


def _cell_state_for_source() -> str:
    """``function cellStateFor(<params>) <body>`` exactly as scheduler_heatmap.jsx declares it."""
    source = HEATMAP_PATH.read_text()
    _masked, params_start, params_end = find_function_params(source, 'cellStateFor')
    body = extract_function_body(source, 'cellStateFor')
    return f'function cellStateFor({source[params_start:params_end]}) {body}'


def heatmap_cell_and_chip(row, module):
    """(cellStateFor(row, module), the row task's LockChip answer) from one vm context."""
    result = subprocess.run(
        [node_path(), '-e', _CELL_DRIVER, HEATMAP_BOUNDS_PATH, SCHED_UTILS_PATH,
         _cell_state_for_source(), json.dumps([row, module])],
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(result.stdout.strip())
    return out['cell'], out['chip']
