"""Behavioral (node-vm) tests for lockChipState — the lock-chip precedence helper.

Executes lockChipState({holder, isMine, parkedBy, parkedOwnerLive}) from
scheduler_utils.jsx inside a node vm sandbox and asserts the returned
{cls, hint, ownerLabel} across the full precedence matrix:

  holder (mine/taken) > parked > free

Uses the same node-vm harness established in test_chip_label_disambiguation.py.
Tests skip when node is absent from PATH (CI requires node).
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import pytest

SCHED_UTILS_PATH = str(
    pathlib.Path(__file__).parent.parent / 'src/dashboard/static/redux/scheduler_utils.jsx'
)

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
