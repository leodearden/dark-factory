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
UNCHECKED = 2
NO_GATE = 3
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
    form = _lines(result, 'interim form')
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


def _assert_unjudged(result: subprocess.CompletedProcess[str]) -> None:
    for kind in (MISSING, FOREIGN, ESCAPES, EMPTY):
        assert _lines(result, kind) == [], result.stdout
    assert not any(line.endswith(': correct') for line in result.stdout.splitlines())


ANOTHER_RUNS_CHAIN = {'trigger_chain': {
    'role': 'human_gate', 'skill': 'review-all', 'run_id': 'review-all-other-20261019',
}}


@pytest.mark.parametrize('change', [None, {'metadata': ANOTHER_RUNS_CHAIN}],
                         ids=['gate-removed', 'gate-of-another-run'])
def test_no_human_gate_is_its_own_state_not_a_defect(
    interim: SeededRun, change: dict[str, object] | None,
) -> None:
    interim.store.edit({GATE: change})

    result = _check(interim)

    assert result.returncode == NO_GATE, result.stdout + result.stderr
    assert f'no human_gate task carries trigger_chain.run_id = {RUN_ID}' in (
        result.stdout.splitlines())
    _assert_unjudged(result)


def test_a_store_without_include_rows_is_unchecked(tmp_path: Path) -> None:
    with seeded_run(tmp_path, store_flags=('--no-include-rows',)) as run:
        result = _check(run)

    assert result.returncode == UNCHECKED, result.stdout + result.stderr
    assert 'include_rows' in result.stderr
    _assert_unjudged(result)


def test_a_run_with_no_home_is_unchecked(interim: SeededRun) -> None:
    result = interim.run(SCRIPT, '--run', 'review-all-absent-20261019')

    assert result.returncode == UNCHECKED, result.stdout + result.stderr
    assert 'program.json' in result.stderr
    _assert_unjudged(result)


def test_no_run_argument_is_unchecked(interim: SeededRun) -> None:
    assert interim.run(SCRIPT).returncode == UNCHECKED


COMPLETION_GATE = 111
ESCAPEE = 104
ESCAPEE_PRD_PATH = 'plans/review-all-fixture-s1-prd.md'


@pytest.fixture
def final(tmp_path: Path) -> Iterator[SeededRun]:
    with seeded_run(tmp_path, store='store-final.json') as run:
        yield run


def test_final_gate_over_a_program_with_an_untagged_open_task_is_defective(
    final: SeededRun,
) -> None:
    result = _check(final)

    assert result.returncode == DEFECTIVE, result.stdout + result.stderr
    form = _lines(result, 'final form')
    assert len(form) == 1, result.stdout
    assert re.search(rf'human gate {GATE}\b', form[0]), form
    assert re.search(rf'completion gate {COMPLETION_GATE}\b', form[0]), form
    assert _task_ids(result, ESCAPES) == {ESCAPEE}
    assert ESCAPEE_PRD_PATH in _lines(result, ESCAPES)[0]
    assert _lines(result, MISSING) == []
    assert _lines(result, FOREIGN) == []


def test_final_gate_is_correct_once_the_escapee_carries_the_run(final: SeededRun) -> None:
    final.store.edit({ESCAPEE: {'metadata': {'x_finding_run': RUN_ID}}})

    _assert_correct(_check(final))


def test_final_gate_must_depend_on_its_completion_gate_only(final: SeededRun) -> None:
    final.store.edit({GATE: {'dependencies': [101]}})

    result = _check(final)

    assert result.returncode == DEFECTIVE, result.stdout + result.stderr
    assert _task_ids(result, MISSING) == {COMPLETION_GATE}
    assert _task_ids(result, FOREIGN) == {101}


@pytest.mark.parametrize('status', ['cancelled', 'done'])
def test_a_terminal_escapee_is_not_reported(final: SeededRun, status: str) -> None:
    final.store.edit({ESCAPEE: {'status': status}})

    _assert_correct(_check(final))


def test_a_task_carrying_another_runs_tag_escapes(final: SeededRun) -> None:
    final.store.edit({ESCAPEE: {'metadata': {'x_finding_run': 'review-all-other-20261001'}}})

    result = _check(final)

    assert result.returncode == DEFECTIVE, result.stdout + result.stderr
    assert _task_ids(result, ESCAPES) == {ESCAPEE}


def test_an_empty_final_program_is_defective(final: SeededRun) -> None:
    final.store.keep_only({GATE, COMPLETION_GATE, 201})

    result = _check(final)

    assert result.returncode == DEFECTIVE, result.stdout + result.stderr
    assert len(_lines(result, EMPTY)) == 1, result.stdout
