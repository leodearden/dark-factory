"""The one rule for the task a stored task id names."""

from __future__ import annotations

import pytest

from dashboard.data.task_ids import parse_task_id


@pytest.mark.parametrize(('raw', 'expected'), [('3546', 3546), ('7', 7), ('0042', 42)])
def test_a_string_of_ascii_digits_names_that_task(raw, expected):
    assert parse_task_id(raw) == expected


@pytest.mark.parametrize(
    'raw',
    [None, 7, 7.0, '', ' 7', '7 ', '-7', '+7', '7a', 'task-7', '²', '٧', 'esc-7-1'],
    ids=[
        'null', 'int', 'float', 'empty', 'leading_space', 'trailing_space', 'signed_minus',
        'signed_plus', 'suffixed', 'label', 'superscript', 'arabic_indic', 'escalation_id',
    ],
)
def test_anything_else_names_no_task(raw):
    assert parse_task_id(raw) is None
