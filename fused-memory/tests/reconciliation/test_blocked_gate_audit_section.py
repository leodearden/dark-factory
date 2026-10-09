"""Behaviour of ``reconciliation/blocked_gate_audit_section.py``.

The section is a COMPLETE enumeration of the project's blocked gate tasks, so
the selector is tested arm by arm: the declared shape (deterministic kind with
gate operational mode, absent meaning gate), the observed shape (a non-empty
``gate_escalated_at`` stamp), and their union.
"""

from __future__ import annotations

import copy
import logging
from datetime import UTC, datetime, timedelta

import pytest

from fused_memory.reconciliation.blocked_gate_audit_section import (
    BLOCKED_GATE_AUDIT_HEADER,
    MAX_BLOCKED_GATE_AUDIT_RENDERED,
    render_blocked_gate_audit_section,
    select_blocked_gate_tasks,
)
from fused_memory.reconciliation.prompts.stage2 import build_stage2_system_prompt

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


_MODULE_LOGGER = 'fused_memory.reconciliation.blocked_gate_audit_section'
_CAP_EVENT = 'reconciliation.gate_task_audit_render_capped'


def _stamped_gates(count: int) -> list[dict]:
    """Blocked gates whose ids ascend with their ``gate_escalated_at``."""
    start = datetime(2026, 8, 1, tzinfo=UTC)
    return [
        _task(i, {'task_kind': 'deterministic', 'gate_escalated_at': (start + timedelta(hours=i)).isoformat()})
        for i in range(1, count + 1)
    ]


def _render(tasks: list[dict]) -> str:
    return render_blocked_gate_audit_section(tasks, project_id='p', run_id='r')


def _rendered_ids(section: str) -> list[int]:
    return [int(line[3 : line.index(']')]) for line in section.splitlines() if line.startswith('- [')]


def _cap_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == _CAP_EVENT]


class TestRenderBlockedGateAuditSection:
    def test_renders_every_gate_with_true_count(self):
        gates = _stamped_gates(12)
        non_gates = [
            _task(100, {'task_kind': 'normal'}),
            _task(101, {'task_kind': 'deterministic'}, status='pending'),
        ]
        section = _render(gates + non_gates)
        assert f'{BLOCKED_GATE_AUDIT_HEADER} (12 gate task(s) awaiting review)' in section.splitlines()
        assert _rendered_ids(section) == list(range(1, 13))

    def test_empty_state_still_renders_header(self):
        section = _render([_task(1, {'task_kind': 'normal'})])
        assert f'{BLOCKED_GATE_AUDIT_HEADER} (0 gate task(s) awaiting review)' in section.splitlines()
        assert 'No tasks.' in section

    def test_render_cap_keeps_oldest_drops_newest_with_overflow_note(self, caplog):
        total = MAX_BLOCKED_GATE_AUDIT_RENDERED + 5
        with caplog.at_level(logging.WARNING, logger=_MODULE_LOGGER):
            section = _render(_stamped_gates(total))
        assert _rendered_ids(section) == list(range(1, MAX_BLOCKED_GATE_AUDIT_RENDERED + 1))
        assert f'({total} gate task(s) awaiting review)' in section
        note = next(line for line in section.splitlines() if line.startswith('_NOTE:'))
        assert '5 additional' in note
        assert 'clipped' in note
        assert 'NOT complete this cycle' in note
        [record] = _cap_records(caplog)
        assert record.levelno == logging.WARNING
        assert record.__dict__['project_id'] == 'p'
        assert record.__dict__['run_id'] == 'r'
        assert record.__dict__['total_gate_tasks'] == total
        assert record.__dict__['rendered'] == MAX_BLOCKED_GATE_AUDIT_RENDERED
        assert record.__dict__['omitted'] == 5

    def test_unstamped_gate_survives_render_cap(self):
        gates = _stamped_gates(MAX_BLOCKED_GATE_AUDIT_RENDERED + 5)
        unstamped_id = 10_000
        section = _render(gates + [_task(unstamped_id, {'task_kind': 'deterministic'})])
        assert unstamped_id in _rendered_ids(section)

    def test_no_overflow_note_or_warning_under_cap(self, caplog):
        with caplog.at_level(logging.WARNING, logger=_MODULE_LOGGER):
            section = _render(_stamped_gates(MAX_BLOCKED_GATE_AUDIT_RENDERED))
        assert len(_rendered_ids(section)) == MAX_BLOCKED_GATE_AUDIT_RENDERED
        assert '_NOTE' not in section
        assert _cap_records(caplog) == []


class TestStage2PromptNamesTheAudit:
    """The payload section never ships without the system prompt telling the model to review it."""

    @pytest.mark.parametrize('project_id', ['dark_factory', 'autopilot_video'])
    def test_prompt_names_the_section_header(self, project_id):
        assert BLOCKED_GATE_AUDIT_HEADER in build_stage2_system_prompt(project_id)
