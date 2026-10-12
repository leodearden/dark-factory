"""Behavioural contract: the priority label on each Park Stacks entry.

parkEntryLabel (tab_scheduler.jsx) is EXECUTED, not grepped: its declaration is
sliced out of the served .jsx (raising on a miss) and run under node, the same
way _lock_chip_matrix.py runs cellStateFor.  The label names what the
orchestrator named (ModuleLockTable.snapshot_park_stacks): a pin's order, or a
fairness park's tier — never the raw pin rank (task 6144's 'tier -999999').
An orchestrator still running pre-6144 code names nothing, and its entry falls
back to the raw rank rather than an empty label.

Executing it needs node: absent from PATH it skips, or fails when CI is set.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

import pytest
from _dashboard_helpers import extract_function_body, find_function_params
from _lock_chip_matrix import node_path

_TAB_SCHEDULER = (
    pathlib.Path(__file__).parent.parent / 'src/dashboard/static/redux/tab_scheduler.jsx'
)

_DRIVER = r"""
const vm = require('vm');
const [labelSrc, entriesJson] = process.argv.slice(1);
const context = vm.createContext({});
vm.runInContext(labelSrc, context, { filename: 'tab_scheduler.jsx::parkEntryLabel' });
context.__entries = JSON.parse(entriesJson);
process.stdout.write(JSON.stringify(vm.runInContext('__entries.map(parkEntryLabel)', context)) + '\n');
"""


def _park_entry_label_source() -> str:
    source = _TAB_SCHEDULER.read_text()
    _masked, params_start, params_end = find_function_params(source, 'parkEntryLabel')
    body = extract_function_body(source, 'parkEntryLabel')
    return f'function parkEntryLabel({source[params_start:params_end]}) {body}'


def park_entry_labels(entries: list[dict]) -> list[str]:
    result = subprocess.run(
        [node_path(), '-e', _DRIVER, _park_entry_label_source(), json.dumps(entries)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout.strip())


@pytest.mark.parametrize(
    ('entry', 'label'),
    [
        ({'owner': '7', 'rank': -999999, 'source': 'pin', 'pin_order': 1}, 'pin #1'),
        ({'owner': '7', 'rank': -1000000, 'source': 'pin', 'pin_order': 0}, 'pin #0'),
        ({'owner': '8', 'rank': 1, 'source': 'fairness', 'tier': 'high'}, 'high'),
        ({'owner': '9', 'rank': -999999}, 'rank -999999'),
        ({'owner': '9', 'rank': 3}, 'rank 3'),
    ],
    ids=['pin', 'pin-order-zero', 'fairness', 'legacy-pin-rank', 'legacy-tier-rank'],
)
def test_park_entry_label(entry, label):
    assert park_entry_labels([entry]) == [label]
