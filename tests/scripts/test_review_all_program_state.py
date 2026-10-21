"""Behavioural tests for ``skills/review-all/scripts/program/state.py``.

The library is loaded by path (``review_all_program_fixture.load_state``) and
driven through its public functions over the committed fixture program; no
private name is patched.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from review_all_program_fixture import FIXTURE_DIR, PROGRAM_DIR, load_state

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
    active = tuple(sorted(str(s) for s in task_statuses.ACTIVE))
    assert active == state.NON_TERMINAL_STATUSES


def _fixture_program_data() -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / 'program.json').read_text())


def _program_violation(mutate: Callable[[dict[str, Any]], None]) -> str:
    data = _fixture_program_data()
    mutate(data)
    with pytest.raises(state.ProgramInvalid) as caught:
        state.parse_program(data)
    return str(caught.value)


def test_parse_program_reads_the_fixture_into_frozen_streams() -> None:
    program = state.parse_program(_fixture_program_data())
    assert program.run_id == 'review-all-fixture-20261019'
    assert tuple(s.id for s in program.streams) == ('S1', 'S2', 'S3')
    assert program.stream_ids == ('S1', 'S2', 'S3')
    assert program.stream('S2').deps == ('S1',)
    assert program.stream('S1').keys == ('fk-0000000000a1', 'fk-0000000000a2')
    assert program.stream('S1').mode is state.StreamMode.AGENT
    with pytest.raises(dataclasses.FrozenInstanceError):
        program.run_id = 'other'


def test_parse_program_refuses_a_key_in_two_streams_naming_both() -> None:
    data = json.loads((FIXTURE_DIR / 'program-dupkey.json').read_text())
    with pytest.raises(state.ProgramInvalid) as caught:
        state.parse_program(data)
    message = str(caught.value)
    assert 'fk-0000000000a2' in message
    assert 'S1' in message and 'S3' in message


def _set_schema(data: dict[str, Any]) -> None:
    data['schema'] = 'review-all-program/9'


def _dep_on_missing_stream(data: dict[str, Any]) -> None:
    data['streams'][1]['deps'] = ['S9']


def _duplicate_stream_id(data: dict[str, Any]) -> None:
    data['streams'][2]['id'] = 'S1'


def _drop_doc_path(data: dict[str, Any]) -> None:
    del data['doc_path']


@pytest.mark.parametrize(
    ('mutate', 'named'),
    [
        (_set_schema, 'review-all-program/9'),
        (_dep_on_missing_stream, 'streams[1].deps[0]'),
        (_duplicate_stream_id, 'streams[2].id'),
        (_drop_doc_path, 'doc_path'),
    ],
    ids=['schema', 'dep-names-no-stream', 'duplicate-stream-id', 'missing-key'],
)
def test_parse_program_refuses_a_malformed_program_naming_the_value(
    mutate: Callable[[dict[str, Any]], None], named: str,
) -> None:
    assert named in _program_violation(mutate)


def test_parse_program_refuses_a_dependency_cycle_naming_its_streams() -> None:
    def cycle(data: dict[str, Any]) -> None:
        data['streams'][0]['deps'] = ['S2']

    message = _program_violation(cycle)
    assert 'cycle' in message
    assert 'S1' in message and 'S2' in message


@pytest.mark.parametrize('field', ['mode', 'size', 'risk'])
def test_parse_program_refuses_an_unknown_vocabulary_value(field: str) -> None:
    def unknown(data: dict[str, Any]) -> None:
        data['streams'][0][field] = 'bogus'

    message = _program_violation(unknown)
    assert f'streams[0].{field}' in message
    assert 'bogus' in message


def test_parse_program_reports_every_violation_together() -> None:
    data = copy.deepcopy(json.loads((FIXTURE_DIR / 'program-dupkey.json').read_text()))
    data['streams'][1]['risk'] = 'extreme'
    with pytest.raises(state.ProgramInvalid) as caught:
        state.parse_program(data)
    violations = caught.value.violations
    assert any('fk-0000000000a2' in v for v in violations)
    assert any('streams[1].risk' in v and 'extreme' in v for v in violations)
