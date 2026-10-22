"""``gate_check.py`` judged through its CLI, over a seeded home and a fake store copy.

Every case seeds S1 and S2 as authored and runs ``gate_check.py --run <run>`` as
an ``-I -S`` subprocess from a temp cwd. "After correction" is an in-place edit
of the store copy, which the fake store re-reads per request. Assertions name
the defect kind each line starts with and the task ids it names, not prose.
"""
from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from review_all_program_seeded_run import RUN_ID, SeededRun, seeded_run

SCRIPT = 'gate_check.py'
GATE = 110
CORRECT = 0
DEFECTIVE = 1
MISSING = 'missing dependency'
FOREIGN = 'foreign dependency'
ESCAPES = 'escapes the completion measure'
EMPTY = 'empty program'
OPEN_PROGRAM_IDS = (101, 102, 103, 104, 105, 106, 107, 108, 113)


@pytest.fixture
def interim(tmp_path: Path) -> Iterator[SeededRun]:
    with seeded_run(tmp_path, store='store.json') as run:
        yield run


def _check(run: SeededRun) -> subprocess.CompletedProcess[str]:
    return run.run(SCRIPT, '--run', RUN_ID)


def _lines(result: subprocess.CompletedProcess[str], kind: str) -> list[str]:
    return [line for line in result.stdout.splitlines() if line.startswith(f'{kind}:')]


def _task_ids(result: subprocess.CompletedProcess[str], kind: str) -> set[int]:
    ids = set()
    for line in _lines(result, kind):
        named = re.match(rf'{kind}: task (\d+)\b', line)
        assert named is not None, f'a {kind} line names no task: {line!r}'
        ids.add(int(named[1]))
    return ids


def _assert_correct(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == CORRECT, result.stdout + result.stderr
    for kind in (MISSING, FOREIGN, ESCAPES, EMPTY):
        assert _lines(result, kind) == [], result.stdout
    assert result.stdout.splitlines()[-1] == f'gate {GATE}: correct'


def test_interim_gate_missing_a_high_task_and_carrying_an_outsider_is_defective(
    interim: SeededRun,
) -> None:
    result = _check(interim)

    assert result.returncode == DEFECTIVE, result.stdout + result.stderr
    form = [line for line in result.stdout.splitlines() if 'interim form' in line]
    assert len(form) == 1 and re.search(rf'\b{GATE}\b', form[0]), result.stdout
    assert _task_ids(result, MISSING) == {106}
    assert _task_ids(result, FOREIGN) == {201}
    assert _lines(result, ESCAPES) == [], 'the interim form does not measure x_finding_run'


def test_interim_gate_on_exactly_the_open_high_tasks_is_correct(interim: SeededRun) -> None:
    interim.store.edit({GATE: {'dependencies': [101, 104, 106]}})

    _assert_correct(_check(interim))


def test_a_terminal_high_task_need_not_be_a_dependency(interim: SeededRun) -> None:
    interim.store.edit({106: {'status': 'done'}, GATE: {'dependencies': [101, 104]}})

    _assert_correct(_check(interim))


def test_a_lower_priority_program_task_is_a_foreign_dependency(interim: SeededRun) -> None:
    interim.store.edit({GATE: {'dependencies': [101, 104, 106, 102]}})

    result = _check(interim)

    assert result.returncode == DEFECTIVE, result.stdout + result.stderr
    assert _task_ids(result, FOREIGN) == {102}
    assert _task_ids(result, MISSING) == set()


def test_with_no_high_or_critical_task_every_open_program_task_is_expected(
    interim: SeededRun,
) -> None:
    interim.store.edit({task_id: {'priority': 'medium'} for task_id in (101, 104, 106)})

    result = _check(interim)

    assert result.returncode == DEFECTIVE, result.stdout + result.stderr
    assert _task_ids(result, MISSING) == {102, 103, 105, 106, 107, 108, 113}
    assert 109 not in _task_ids(result, MISSING), 'a cancelled task is never missing'
    assert _task_ids(result, FOREIGN) == {201}

    interim.store.edit({GATE: {'dependencies': list(OPEN_PROGRAM_IDS)}})

    _assert_correct(_check(interim))


def test_an_empty_program_is_defective(interim: SeededRun) -> None:
    interim.store.keep_only({GATE, 201})

    result = _check(interim)

    assert result.returncode == DEFECTIVE, result.stdout + result.stderr
    assert len(_lines(result, EMPTY)) == 1, result.stdout
