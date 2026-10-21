"""Behavioural tests for ``skills/review-all/scripts/program/state.py``.

The library is loaded by path (``review_all_program_fixture.load_state``) and
driven through its public functions over the committed fixture program; no
private name is patched.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from review_all_program_fixture import PROGRAM_DIR, load_state
from shared import task_statuses

state = load_state()

_HOST_PYTHON = Path('/usr/bin/python3')
_INTERPRETERS = [sys.executable] + (
    [str(_HOST_PYTHON)]
    if _HOST_PYTHON.exists()
    and os.path.realpath(_HOST_PYTHON) != os.path.realpath(sys.executable)
    else []
)
_ALLOWED_FIRST_PARTY = {
    '__main__',
    'state',
    'shared',
    'shared.task_statuses',
    'shared.mcp_post',
    'shared.mcp_envelope',
}
_LOADED_NON_STDLIB = """
import json, sys
sys.path.insert(0, {program_dir!r})
import state
print(json.dumps(sorted(
    name for name in sys.modules
    if name.partition('.')[0] not in sys.stdlib_module_names
)))
"""


@pytest.mark.parametrize('python', _INTERPRETERS)
def test_state_imports_under_isolated_interpreter_from_outside_the_repo(
    python: str, tmp_path: Path,
) -> None:
    code = _LOADED_NON_STDLIB.format(program_dir=str(PROGRAM_DIR))
    proc = subprocess.run(
        [python, '-I', '-S', '-c', code],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    loaded = set(json.loads(proc.stdout))
    assert loaded <= _ALLOWED_FIRST_PARTY, sorted(loaded - _ALLOWED_FIRST_PARTY)


def test_wire_constants() -> None:
    assert state.RETURN_SCHEMA == 'review-all-prd-return/1'
    assert state.PROGRAM_SCHEMA == 'review-all-program/1'
    assert state.SEAM_WAIT_KEYS.key == 'x_seam_wait'
    assert state.SEAM_WAIT_KEYS.fields == ('stream', 'contract', 'release_owner')
    assert state.SEAM_RELEASED_KEYS.key == 'x_seam_released'
    assert state.SEAM_RELEASED_KEYS.fields == (
        'by', 'at', 'onto', 'contract_as_delivered',
    )
    assert set(state.LEDGER_OPS) == {
        'stream_status', 'amend_stream', 'amend_program', 'xref', 'action',
        'action_result', 'scanned', 'note', 'commit',
    }
    assert state.NON_TERMINAL_STATUSES == tuple(
        sorted(str(s) for s in task_statuses.ACTIVE)
    )
