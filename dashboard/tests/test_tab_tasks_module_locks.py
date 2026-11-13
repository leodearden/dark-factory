"""Wiring of tab_tasks.jsx::ModuleLocksSection, the Task-detail lock chips.

What the section draws is rendered by ``js/jsx_render.test.mjs``, which also
swaps ``window.DF_SCHED_UTILS.lockChipStateFor`` for a sentinel, so an inline
chip classifier shows there however it is spelled. locksCellState and
schedulerLocksDatum are bound when tab_tasks.jsx loads, so no render can swap
them, and an inline copy gives the same answer until the two drift: only these
probes see one. Every probe runs over comment-stripped code, and the extractor
raises on a miss, so an absence assert cannot pass on a deleted section.
"""

from __future__ import annotations

import re

import pytest
from _jsx_source_helpers import extract_function_body, strip_js_comments


@pytest.fixture(scope='module')
def module_locks_code(tab_tasks_jsx_body: str) -> str:
    return extract_function_body(strip_js_comments(tab_tasks_jsx_body), 'ModuleLocksSection')


def test_chips_are_classified_by_lock_chip_state_for(module_locks_code: str) -> None:
    assert re.search(r'\blockChipStateFor\s*\(', module_locks_code), (
        'ModuleLocksSection does not classify its chips with lockChipStateFor'
    )
    assert not re.search(r'\blockChipState\s*\(', module_locks_code), (
        'ModuleLocksSection calls the bare lockChipState with a mine-ness it decided itself'
    )
    for token in ('isMine', 'holder_project', 'parked_by', 'parked_owner_live'):
        assert token not in module_locks_code, (
            f'ModuleLocksSection still reads `{token}`. scheduler_utils.jsx::lockChipStateFor '
            'is the one place a lock is decided to be mine, taken, parked or free; hand it '
            'the module entry whole.'
        )
    assert not re.search(r'\.holder\b', module_locks_code), (
        'ModuleLocksSection reads the holder off a module entry itself. '
        'scheduler_utils.jsx::lockChipStateFor decides whether a held lock is mine.'
    )


def test_the_section_asks_locks_cell_state_whether_anything_is_known(module_locks_code: str) -> None:
    assert re.search(r'\blocksCellState\s*\(', module_locks_code), (
        'ModuleLocksSection does not ask task_row_cells.js::locksCellState what to draw'
    )
    assert re.search(r'\bschedulerLocksDatum\s*\(', module_locks_code), (
        'ModuleLocksSection does not build its datum with task_row_cells.js::schedulerLocksDatum'
    )
    assert 'offline_projects' not in module_locks_code, (
        'ModuleLocksSection reads offline_projects itself; schedulerLocksDatum decides '
        'which projects are holes'
    )
    assert not re.search(r"""state\s*===\s*['"]unknown['"]""", module_locks_code), (
        'ModuleLocksSection re-derives the unknown test; locksCellState decides it'
    )
