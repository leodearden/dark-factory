"""Behavioral (node-vm) tests for lockChipStateFor, the one lock classifier.

Executes lockChipStateFor(module, taskId, project) from scheduler_utils.jsx in a
node vm context and asserts the returned {cls, ownerLabel} across the full
precedence matrix:

  holder (mine/taken) > parked > free

It is the module's only classifier export: it derives "this lock is mine" from
a task's view of a SCHEDULER.modules entry and applies the precedence. The
Scheduler heatmap's cellStateFor is executed below beside it, and must answer
every lock state exactly as the task-row LockChip does; membership
(rowTouchesModule) is the only thing it adds.
"""

from __future__ import annotations

import json
import subprocess

import pytest
from _dashboard_helpers import node_path
from _lock_chip_matrix import (
    CELL_MATRIX,
    MODULE_PATH,
    ROW_PROJECT,
    ROW_TASK,
    SCHED_UTILS_PATH,
    heatmap_cell_and_chip,
    heatmap_module,
    heatmap_row,
    lock_chip_states_for,
)

# ---------------------------------------------------------------------------
# Precedence-matrix tests (node-vm runtime)
# ---------------------------------------------------------------------------

# Each case: the module's lock fields, and the (cls, ownerLabel) task ROW_TASK
# of ROW_PROJECT must read it as.
_PRECEDENCE = {
    'held by self': ({'holder': ROW_TASK, 'holder_project': ROW_PROJECT}, 'lock-mine', None),
    'held by other': ({'holder': 'other'}, 'lock-taken', 'T-other'),
    'parked, live owner': ({'parked_by': 'owner', 'parked_owner_live': True}, 'lock-parked', 'T-owner'),
    'parked, stale owner': ({'parked_by': 'owner', 'parked_owner_live': False}, 'lock-parked', 'T-owner ⚠'),
    'free': ({}, 'lock-free', None),
    'held and parked, the holder wins': (
        {'holder': 'other', 'parked_by': 'owner', 'parked_owner_live': True}, 'lock-taken', 'T-other',
    ),
}


@pytest.fixture(scope='module')
def precedence_states() -> dict[str, dict]:
    """lockChipStateFor's answer for every _PRECEDENCE case, from one node process."""
    cases = [(heatmap_module(**fields), ROW_TASK, ROW_PROJECT) for fields, _cls, _owner in _PRECEDENCE.values()]
    return dict(zip(_PRECEDENCE, lock_chip_states_for(cases), strict=True))


class TestLockChipStateForPrecedence:
    @pytest.mark.parametrize('case', list(_PRECEDENCE))
    def test_precedence(self, precedence_states, case):
        _fields, cls, owner_label = _PRECEDENCE[case]
        state = precedence_states[case]
        assert (state['cls'], state['ownerLabel']) == (cls, owner_label), f'{case}: {state!r}'


_EXPORTS_DRIVER = r"""
const vm = require('vm');
const fs = require('fs');
const [utilsPath] = process.argv.slice(1);
const context = vm.createContext({ console, window: {} });
vm.runInContext(fs.readFileSync(utilsPath, 'utf8'), context, { filename: utilsPath });
process.stdout.write(JSON.stringify(Object.keys(context.window.DF_SCHED_UTILS)) + '\n');
"""


def _exported_sched_utils_keys() -> list[str]:
    result = subprocess.run(
        [node_path(), '-e', _EXPORTS_DRIVER, SCHED_UTILS_PATH],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip())


def test_lock_chip_state_for_is_the_only_classifier_exported():
    """No caller can hand the precedence a mine-ness it decided itself."""
    exported = _exported_sched_utils_keys()
    assert 'lockChipStateFor' in exported, exported
    assert 'lockChipState' not in exported, (
        'window.DF_SCHED_UTILS still exports the bare lockChipState, which takes a '
        'caller-computed isMine; lockChipStateFor is the one place a lock is decided to be mine'
    )


class TestLockChipStateForDecidesMine:
    """lockChipStateFor(module, taskId, project) is the one place a lock is "mine".

    Mine means held by this task id IN this project; a module that names no
    holder_project is read as the task's own project.
    """

    @pytest.mark.parametrize('fields, cls, owner_label', [
        ({'holder': ROW_TASK, 'holder_project': ROW_PROJECT}, 'lock-mine', None),
        ({'holder': ROW_TASK}, 'lock-mine', None),
        ({'holder': ROW_TASK, 'holder_project': 'Q'}, 'lock-taken', f'T-{ROW_TASK}'),
        ({'holder': 'A', 'holder_project': ROW_PROJECT}, 'lock-taken', 'T-A'),
        ({'parked_by': 'C', 'parked_owner_live': False}, 'lock-parked', 'T-C ⚠'),
        ({}, 'lock-free', None),
    ], ids=['own task, own project', 'own task, no holder_project',
            'own task id, another project', 'another task', 'parked, dead owner', 'free'])
    def test_mine_is_this_task_in_this_project(self, fields, cls, owner_label):
        [state] = lock_chip_states_for([(heatmap_module(**fields), ROW_TASK, ROW_PROJECT)])
        assert (state['cls'], state['ownerLabel']) == (cls, owner_label), state

    def test_a_module_the_scheduler_does_not_list_is_free(self):
        [state] = lock_chip_states_for([(None, ROW_TASK, ROW_PROJECT)])
        assert state['cls'] == 'lock-free', state


# ---------------------------------------------------------------------------
# The heatmap cell is classified by lockChipStateFor too (PRD sketch #12,
# second half). cellStateFor is EXECUTED, not grepped, by
# _lock_chip_matrix.heatmap_cell_and_chip.
# ---------------------------------------------------------------------------


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
