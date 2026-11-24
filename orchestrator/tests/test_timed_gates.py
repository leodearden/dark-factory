"""Which tasks are timed gates, and in what order they dispatch (task 11330)."""

from __future__ import annotations

from typing import Any

import pytest

from orchestrator.timed_gates import is_timed_gate, timed_gates

PREDICATE = {'script': 's.py', 'timeout_secs': 60, 'kind': 'predicate'}
DATED = {'mode': 'dated', 'at': '2026-11-19T11:35:00+00:00'}


def _task(tid: str, metadata: Any) -> dict:
    return {'id': tid, 'status': 'pending', 'priority': 'low', 'metadata': metadata}


def _predicate_link(tid: str, **overrides: Any) -> dict:
    metadata = {
        'task_kind': 'deterministic',
        'before_done': dict(PREDICATE),
        'milestone': dict(DATED),
        'recurrence': {
            'key': 'claude-code-mode-headless-support',
            'interval_secs': 86400,
            'minted_from': '10900',
        },
        'source': 'recurrence-mint',
    }
    metadata.update(overrides)
    return _task(tid, metadata)


def _pure_gate(tid: str) -> dict:
    return _task(tid, {
        'task_kind': 'deterministic',
        'always_escalates': True,
        'milestone': {'mode': 'delayed', 'after_secs': 60},
    })


def _deploy(tid: str) -> dict:
    return _predicate_link(tid, before_done={**PREDICATE, 'kind': 'deploy'})


def test_a_recurrence_minted_predicate_link_with_a_milestone_is_a_timed_gate():
    assert is_timed_gate(_predicate_link('1'))


def test_a_pure_gate_with_a_milestone_is_a_timed_gate():
    assert is_timed_gate(_pure_gate('1'))


def test_a_deploy_is_not_a_timed_gate():
    assert not is_timed_gate(_deploy('1'))


def test_a_before_done_without_a_kind_is_a_deploy_and_not_a_timed_gate():
    kindless = {key: value for key, value in PREDICATE.items() if key != 'kind'}

    assert not is_timed_gate(_predicate_link('1', before_done=kindless))


def test_a_deterministic_predicate_without_a_milestone_is_not_a_timed_gate():
    task = _predicate_link('1')
    del task['metadata']['milestone']

    assert not is_timed_gate(task)


def test_a_normal_task_with_a_dated_milestone_is_not_a_timed_gate():
    assert not is_timed_gate(_predicate_link('1', task_kind='normal'))


@pytest.mark.parametrize('metadata', [None, 'deterministic', ['task_kind']])
def test_a_task_whose_metadata_is_not_a_dict_is_not_a_timed_gate(metadata):
    assert not is_timed_gate(_task('1', metadata))


def test_a_task_without_metadata_is_not_a_timed_gate():
    assert not is_timed_gate({'id': '1', 'status': 'pending'})


def test_a_deterministic_task_that_does_not_always_escalate_and_has_no_before_done_is_not_a_timed_gate():
    task = _task('1', {'task_kind': 'deterministic', 'always_escalates': False, 'milestone': dict(DATED)})

    assert not is_timed_gate(task)


def test_timed_gates_keeps_only_the_timed_gates_oldest_id_first():
    tasks = [
        _task('3', {'files': ['a.py']}),
        _predicate_link('12'),
        _deploy('5'),
        _pure_gate('9'),
    ]

    assert [task['id'] for task in timed_gates(tasks)] == ['9', '12']
