"""Behaviour of ``reconciliation/capped_task_list.py``.

The helper is the single home of the no-silent-caps render: a task list
clipped to its cap ALWAYS carries a ``_NOTE:`` line (the token the Stage-2
prompt keys on) and a WARNING log naming what was dropped.
"""

from __future__ import annotations

import logging

import pytest

from fused_memory.reconciliation.capped_task_list import render_capped_task_list

_EVENT = 'reconciliation.test_render_capped'


def _tasks(count: int) -> list[dict]:
    return [{'id': i, 'title': f'task {i}', 'status': 'blocked'} for i in range(1, count + 1)]


def _render(tasks: list[dict], *, cap: int = 3) -> str:
    return render_capped_task_list(
        tasks,
        cap=cap,
        cap_name='MAX_WIDGETS_RENDERED',
        omitted_noun='widget task(s)',
        dropped_first='the newest widgets',
        log_event=_EVENT,
        log_extra={'project_id': 'p', 'run_id': 'r', 'total_widgets': len(tasks)},
    )


def _rendered_ids(text: str) -> list[int]:
    return [int(line[3 : line.index(']')]) for line in text.splitlines() if line.startswith('- [')]


def _cap_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage() == _EVENT]


class TestRenderCappedTaskList:
    @pytest.mark.parametrize('count', [0, 2, 3])
    def test_at_or_under_cap_renders_all_without_note_or_warning(self, caplog, count):
        with caplog.at_level(logging.WARNING):
            text = _render(_tasks(count))
        assert _rendered_ids(text) == list(range(1, count + 1))
        assert '_NOTE' not in text
        assert _cap_records(caplog) == []

    def test_empty_list_renders_no_tasks_marker(self):
        assert 'No tasks.' in _render([])

    def test_over_cap_keeps_the_head_in_given_order(self):
        tasks = list(reversed(_tasks(5)))
        assert _rendered_ids(_render(tasks)) == [5, 4, 3]

    def test_over_cap_note_names_count_cap_noun_and_dropped_end(self):
        note = next(line for line in _render(_tasks(5)).splitlines() if line.startswith('_NOTE:'))
        assert '2 additional widget task(s)' in note
        assert 'MAX_WIDGETS_RENDERED=3' in note
        assert 'NOT complete this cycle' in note
        assert 'the newest widgets were dropped first' in note

    def test_over_cap_logs_one_warning_with_caller_extras_and_clip_counts(self, caplog):
        with caplog.at_level(logging.WARNING):
            _render(_tasks(5))
        [record] = _cap_records(caplog)
        assert record.levelno == logging.WARNING
        assert record.__dict__['project_id'] == 'p'
        assert record.__dict__['run_id'] == 'r'
        assert record.__dict__['total_widgets'] == 5
        assert record.__dict__['rendered'] == 3
        assert record.__dict__['omitted'] == 2
