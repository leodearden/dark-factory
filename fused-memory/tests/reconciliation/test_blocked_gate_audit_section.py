"""Behaviour of ``reconciliation/blocked_gate_audit_section.py``.

The section is a COMPLETE enumeration of the project's blocked gate tasks, so
the selector is tested arm by arm: the declared shape (deterministic kind with
gate operational mode, absent meaning gate), the observed shape (a non-empty
``gate_escalated_at`` stamp), and their union.
"""

from __future__ import annotations

import copy

import pytest

from fused_memory.reconciliation.blocked_gate_audit_section import select_blocked_gate_tasks

STAMP = '2026-08-19T05:42:35Z'


def _task(task_id: int, metadata: object = None, *, status: str = 'blocked') -> dict:
    task: dict = {'id': task_id, 'title': f'task {task_id}', 'status': status}
    if metadata is not None:
        task['metadata'] = metadata
    return task


def _ids(tasks: list[dict]) -> list[int]:
    return [t['id'] for t in tasks]


class TestSelectBlockedGateTasks:
    @pytest.mark.parametrize(
        'metadata',
        [
            {'task_kind': 'deterministic', 'operational_mode': 'gate'},
            {'task_kind': 'deterministic'},
        ],
        ids=['explicit-gate-mode', 'absent-mode-means-gate'],
    )
    def test_declared_gate_shape_is_selected(self, metadata):
        assert _ids(select_blocked_gate_tasks([_task(1, metadata)])) == [1]

    @pytest.mark.parametrize(
        'metadata',
        [
            {'task_kind': 'deterministic', 'operational_mode': 'llm'},
            {'task_kind': 'normal', 'operational_mode': 'gate'},
            {'operational_mode': 'gate'},
        ],
        ids=['llm-mode', 'normal-kind', 'absent-kind-means-normal'],
    )
    def test_undeclared_unstamped_shape_is_not_selected(self, metadata):
        assert select_blocked_gate_tasks([_task(1, metadata)]) == []

    @pytest.mark.parametrize(
        'metadata',
        [
            {'gate_escalated_at': STAMP},
            {'task_kind': 'deterministic', 'operational_mode': 'llm', 'gate_escalated_at': STAMP},
        ],
        ids=['stamp-only', 'llm-mode-coerced-pure-gate'],
    )
    def test_observed_stamp_is_selected_without_declared_shape(self, metadata):
        assert _ids(select_blocked_gate_tasks([_task(1, metadata)])) == [1]

    @pytest.mark.parametrize('stamp', ['', None, 123], ids=['empty', 'none', 'int'])
    def test_non_string_or_empty_stamp_does_not_select(self, stamp):
        assert select_blocked_gate_tasks([_task(1, {'gate_escalated_at': stamp})]) == []

    def test_task_matching_both_arms_appears_once(self):
        metadata = {'task_kind': 'deterministic', 'operational_mode': 'gate', 'gate_escalated_at': STAMP}
        assert _ids(select_blocked_gate_tasks([_task(1, metadata)])) == [1]

    @pytest.mark.parametrize('status', ['pending', 'in-progress', 'review', 'done', 'cancelled'])
    @pytest.mark.parametrize(
        'metadata',
        [{'task_kind': 'deterministic', 'operational_mode': 'gate'}, {'gate_escalated_at': STAMP}],
        ids=['declared', 'observed'],
    )
    def test_non_blocked_status_is_excluded(self, status, metadata):
        assert select_blocked_gate_tasks([_task(1, metadata, status=status)]) == []

    @pytest.mark.parametrize(
        'metadata',
        [None, '{"task_kind": "deterministic"}', ['deterministic'], 7],
        ids=['none', 'json-string', 'list', 'int'],
    )
    def test_non_dict_metadata_is_excluded(self, metadata):
        task = _task(1)
        task['metadata'] = metadata
        assert select_blocked_gate_tasks([task]) == []

    def test_absent_metadata_is_excluded(self):
        assert select_blocked_gate_tasks([_task(1)]) == []

    def test_non_dict_elements_are_skipped(self):
        gate = _task(1, {'task_kind': 'deterministic'})
        assert _ids(select_blocked_gate_tasks([None, 'x', 3, gate])) == [1]

    def test_orders_oldest_stamp_first(self):
        tasks = [
            _task(1, {'gate_escalated_at': '2026-08-21T00:00:00Z'}),
            _task(2, {'gate_escalated_at': '2026-08-19T00:00:00Z'}),
            _task(3, {'gate_escalated_at': '2026-08-20T00:00:00Z'}),
        ]
        assert _ids(select_blocked_gate_tasks(tasks)) == [2, 3, 1]

    def test_unstamped_and_unparseable_sort_first(self):
        tasks = [
            _task(1, {'gate_escalated_at': '2026-08-19T00:00:00Z'}),
            _task(5, {'task_kind': 'deterministic', 'gate_escalated_at': 'not-a-date'}),
            _task(9, {'task_kind': 'deterministic'}),
        ]
        assert _ids(select_blocked_gate_tasks(tasks)) == [5, 9, 1]

    def test_ties_break_by_id_ascending(self):
        tasks = [
            _task(30, {'gate_escalated_at': STAMP}),
            _task(10, {'gate_escalated_at': STAMP}),
            _task(20, {'gate_escalated_at': STAMP}),
        ]
        assert _ids(select_blocked_gate_tasks(tasks)) == [10, 20, 30]

    def test_z_and_offset_spellings_order_against_each_other(self):
        tasks = [
            _task(1, {'gate_escalated_at': '2026-08-19T12:00:00+00:00'}),
            _task(2, {'gate_escalated_at': '2026-08-19T11:00:00Z'}),
            _task(3, {'gate_escalated_at': '2026-08-19T13:00:00Z'}),
        ]
        assert _ids(select_blocked_gate_tasks(tasks)) == [2, 1, 3]

    def test_empty_input_returns_empty_list(self):
        assert select_blocked_gate_tasks([]) == []

    def test_input_is_not_mutated(self):
        tasks = [
            _task(2, {'gate_escalated_at': '2026-08-21T00:00:00Z'}),
            _task(1, {'gate_escalated_at': '2026-08-19T00:00:00Z'}),
        ]
        before = copy.deepcopy(tasks)
        select_blocked_gate_tasks(tasks)
        assert tasks == before
