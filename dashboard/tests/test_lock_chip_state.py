"""Behavioral (node-vm) tests for lockChipState — the lock-chip precedence helper.

Executes lockChipState({holder, isMine, parkedBy, parkedOwnerLive}) from
scheduler_utils.jsx inside a node vm sandbox and asserts the returned
{cls, hint, ownerLabel} across the full precedence matrix:

  holder (mine/taken) > parked > free

Uses the same node-vm harness established in test_chip_label_disambiguation.py.
Tests skip when node is absent from PATH (CI requires node).

It is the ONE lock classifier: the Scheduler heatmap's cellStateFor is executed
below beside it, and must answer every lock state exactly as the task-row
LockChip does — membership (rowTouchesModule) is the only thing it adds.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import pytest
from _dashboard_helpers import extract_function_body, find_function_params

_REDUX_DIR = pathlib.Path(__file__).parent.parent / 'src/dashboard/static/redux'
SCHED_UTILS_PATH = str(_REDUX_DIR / 'scheduler_utils.jsx')
HEATMAP_BOUNDS_PATH = str(_REDUX_DIR / 'scheduler_heatmap_bounds.js')
HEATMAP_PATH = _REDUX_DIR / 'scheduler_heatmap.jsx'

# Node driver: extracts the pure-JS helper section from scheduler_utils.jsx
# (everything before the window.DF_SCHED_UTILS export line) and runs it in a
# vm sandbox.  Returns a plain object via JSON.stringify.
_DRIVER = r"""
const vm = require('vm');
const fs = require('fs');
const src = fs.readFileSync(process.argv[1], 'utf8');

// Extract everything before the window.DF_SCHED_UTILS export line.
const endIdx = src.lastIndexOf('\nwindow.DF_SCHED_UTILS');
if (endIdx < 0) throw new Error('window.DF_SCHED_UTILS not found — check scheduler_utils.jsx');
const helpersSrc = src.slice(0, endIdx);

const name    = process.argv[2];
const argsJson = process.argv[3];

const sandbox = { Math, console };
if (argsJson !== undefined) sandbox.__args = JSON.parse(argsJson);

const scriptBody = argsJson !== undefined
  ? 'var r = ' + name + '(...__args); if (r instanceof Map) r = Object.fromEntries(r); var __result = r;'
  : 'var r = ' + name + '; if (r instanceof Map) r = Object.fromEntries(r); var __result = r;';

vm.runInNewContext(helpersSrc + '\n' + scriptBody, sandbox);
process.stdout.write(JSON.stringify(sandbox.__result) + '\n');
"""


def _node():
    path = shutil.which('node')
    if not path:
        if os.environ.get('CI'):
            pytest.fail('node is required in CI but not found on PATH')
        pytest.skip('node not available')
    return path


def _eval_sched_utils_fn(fn_name, *args):
    """Call fn_name(*args) in a node vm sandbox and return the decoded result."""
    result = subprocess.run(
        [_node(), '-e', _DRIVER, SCHED_UTILS_PATH, fn_name, json.dumps(list(args))],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip())


def lock_chip_state(holder, isMine=False, parkedBy=None, parkedOwnerLive=None):
    """Invoke lockChipState({...}) in the vm sandbox and return the decoded dict."""
    return _eval_sched_utils_fn(
        'lockChipState',
        {
            'holder': holder,
            'isMine': isMine,
            'parkedBy': parkedBy,
            'parkedOwnerLive': parkedOwnerLive,
        },
    )


# ---------------------------------------------------------------------------
# Precedence-matrix tests (node-vm runtime)
# ---------------------------------------------------------------------------

class TestLockChipStatePrecedence:
    def test_held_by_self_is_lock_mine(self):
        """(a) held-by-self: holder present + isMine=True → cls='lock-mine'."""
        result = lock_chip_state(holder='me', isMine=True)
        assert result['cls'] == 'lock-mine', (
            f"held-by-self must yield cls='lock-mine', got {result!r}"
        )

    def test_held_by_other_is_lock_taken_with_owner_label(self):
        """(b) held-by-other: holder present + isMine=False → cls='lock-taken',
        ownerLabel='T-other'."""
        result = lock_chip_state(holder='other', isMine=False)
        assert result['cls'] == 'lock-taken', (
            f"held-by-other must yield cls='lock-taken', got {result!r}"
        )
        assert result['ownerLabel'] == 'T-other', (
            f"held-by-other ownerLabel must be 'T-other', got {result.get('ownerLabel')!r}"
        )

    def test_unheld_parked_live_is_lock_parked(self):
        """(c) [B6] unheld+parked-live → cls='lock-parked', ownerLabel='T-owner'
        (no warning glyph when parkedOwnerLive is True)."""
        result = lock_chip_state(holder=None, parkedBy='owner', parkedOwnerLive=True)
        assert result['cls'] == 'lock-parked', (
            f"unheld+parked-live must yield cls='lock-parked', got {result!r}"
        )
        assert result['ownerLabel'] == 'T-owner', (
            f"ownerLabel must be 'T-owner' (live), got {result.get('ownerLabel')!r}"
        )

    def test_unheld_parked_stale_has_warning_glyph(self):
        """(d) unheld+parked-stale (parkedOwnerLive=False) → ownerLabel includes ⚠."""
        result = lock_chip_state(holder=None, parkedBy='owner', parkedOwnerLive=False)
        assert result['cls'] == 'lock-parked', (
            f"unheld+parked-stale must yield cls='lock-parked', got {result!r}"
        )
        assert result['ownerLabel'] == 'T-owner ⚠', (
            f"stale ownerLabel must be 'T-owner ⚠', got {result.get('ownerLabel')!r}"
        )

    def test_unheld_unparked_is_lock_free(self):
        """(e) unheld+free → cls='lock-free'."""
        result = lock_chip_state(holder=None, parkedBy=None)
        assert result['cls'] == 'lock-free', (
            f"unheld+unparked must yield cls='lock-free', got {result!r}"
        )

    def test_held_and_parked_holder_wins(self):
        """(f) [B7] held+parked → holder beats parked, cls='lock-taken' not 'lock-parked'.

        Proves the holder > parked precedence by EXECUTING the helper — robust to any
        source-level refactor of the chip code that preserves this runtime contract.
        """
        result = lock_chip_state(holder='other', isMine=False, parkedBy='owner', parkedOwnerLive=True)
        assert result['cls'] == 'lock-taken', (
            f"held+parked must yield cls='lock-taken' (holder wins), got {result!r}"
        )
        assert result['cls'] != 'lock-parked', (
            "held+parked must NOT yield cls='lock-parked' — holder takes precedence"
        )


# ---------------------------------------------------------------------------
# The heatmap cell is classified by lockChipState too (PRD sketch #12, second
# half). cellStateFor is EXECUTED, not grepped: its declaration is sliced out
# of scheduler_heatmap.jsx (raising on a miss) and run in one vm context with
# the two classic scripts it reaches at runtime — scheduler_heatmap_bounds.js
# for rowTouchesModule, scheduler_utils.jsx for window.DF_SCHED_UTILS.
# ---------------------------------------------------------------------------

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
  '({ cell: cellStateFor(__args[0], __args[1]), chip: window.DF_SCHED_UTILS.lockChipState(__args[2]) })',
  context,
);
process.stdout.write(JSON.stringify(out) + '\n');
"""

ROW_TASK = 'B'
ROW_PROJECT = 'P'
MODULE_PATH = 'm'

# Each lock state a cell in the row's lock set can be in, as module fields.
CELL_MATRIX = {
    'free': {},
    "held by the row's own task": {'holder': ROW_TASK, 'holder_project': ROW_PROJECT},
    'held by another task': {'holder': 'A', 'holder_project': ROW_PROJECT},
    'parked only': {'parked_by': 'C', 'parked_owner_live': True},
    "parked by the row's own task": {'parked_by': ROW_TASK, 'parked_owner_live': True},
    'parked and held': {
        'holder': 'A', 'holder_project': ROW_PROJECT, 'parked_by': ROW_TASK, 'parked_owner_live': True,
    },
    'parked by a dead owner': {'parked_by': 'C', 'parked_owner_live': False},
}


def _cell_state_for_source() -> str:
    """``function cellStateFor(<params>) <body>`` exactly as scheduler_heatmap.jsx declares it."""
    source = HEATMAP_PATH.read_text()
    _masked, params_start, params_end = find_function_params(source, 'cellStateFor')
    body = extract_function_body(source, 'cellStateFor')
    return f'function cellStateFor({source[params_start:params_end]}) {body}'


def heatmap_module(**fields):
    return {'project': ROW_PROJECT, 'path': MODULE_PATH, 'holder': None, 'holder_project': None,
            'parked_by': None, 'parked_owner_live': None, **fields}


def heatmap_row(module, lock_set=(MODULE_PATH,)):
    """Task B's scheduler row; parked on the module exactly when the module says B parked it."""
    parked = [module['path']] if module.get('parked_by') == ROW_TASK else []
    return {'project': ROW_PROJECT, 'task_id': ROW_TASK, 'lock_set': list(lock_set),
            'park_state': {'modules': parked} if parked else None}


def lock_chip_inputs(row, module):
    """The lockChipState inputs tabs.jsx::LockChip builds for this task and module."""
    holder = module.get('holder')
    holder_project = module.get('holder_project') or row['project']
    return {
        'holder': holder,
        'isMine': bool(holder) and holder == row['task_id'] and holder_project == row['project'],
        'parkedBy': module.get('parked_by'),
        'parkedOwnerLive': module.get('parked_owner_live'),
    }


def heatmap_cell_and_chip(row, module):
    """(cellStateFor(row, module), lockChipState(LockChip's inputs)) from one vm context."""
    result = subprocess.run(
        [_node(), '-e', _CELL_DRIVER, HEATMAP_BOUNDS_PATH, SCHED_UTILS_PATH,
         _cell_state_for_source(), json.dumps([row, module, lock_chip_inputs(row, module)])],
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(result.stdout.strip())
    return out['cell'], out['chip']


def chip_classes_over_the_matrix() -> set[str]:
    """Every cls lockChipState returns across CELL_MATRIX."""
    classes = set()
    for fields in CELL_MATRIX.values():
        module = heatmap_module(**fields)
        classes.add(lock_chip_state(**lock_chip_inputs(heatmap_row(module), module))['cls'])
    return classes


class TestHeatmapCellIsTheLockChip:
    def test_parked_and_held_reads_as_taken_in_both(self):
        """A cell held by A and parked by this row's own task is lock-taken, as the chip is.

        The heatmap used to rank parked-by-me first and the chip ranks the
        holder first, so the same lock read 'parked' in one view and 'taken'
        in the other.
        """
        module = heatmap_module(holder='A', holder_project=ROW_PROJECT,
                                parked_by=ROW_TASK, parked_owner_live=True)
        row = heatmap_row(module)
        assert row['park_state'] == {'modules': [MODULE_PATH]}
        assert lock_chip_inputs(row, module) == {
            'holder': 'A', 'isMine': False, 'parkedBy': ROW_TASK, 'parkedOwnerLive': True,
        }
        cell, chip = heatmap_cell_and_chip(row, module)
        assert cell == chip, f'heatmap cell {cell!r} disagrees with the lock chip {chip!r}'
        assert chip['cls'] == 'lock-taken'

    @pytest.mark.parametrize('case', sorted(CELL_MATRIX))
    def test_every_lock_state_classifies_as_the_chip_does(self, case):
        module = heatmap_module(**CELL_MATRIX[case])
        cell, chip = heatmap_cell_and_chip(heatmap_row(module), module)
        assert cell == chip, f'{case}: heatmap cell {cell!r} disagrees with the lock chip {chip!r}'

    @pytest.mark.parametrize('row_project, lock_set', [
        (ROW_PROJECT, ()),
        ('Q', (MODULE_PATH,)),
    ], ids=['module outside the lock set', 'same path, another project'])
    def test_a_module_the_row_does_not_lock_is_not_in_set(self, row_project, lock_set):
        module = heatmap_module(holder='A', holder_project=ROW_PROJECT)
        row = {**heatmap_row(module, lock_set=lock_set), 'project': row_project}
        cell, _chip = heatmap_cell_and_chip(row, module)
        assert isinstance(cell, dict) and set(cell) == {'cls', 'hint', 'ownerLabel'}, (
            f'cellStateFor does not answer in lockChipState\'s shape: {cell!r}'
        )
        assert cell['cls'] == 'not-in-set'
