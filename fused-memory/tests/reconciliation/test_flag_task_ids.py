"""Tests for ``flag_task_ids.task_id_components``, the one flag task-id splitter.

Both flag-side consumers — ``flag_dedup._flag_candidate_task_ids`` and
``preservation_specimen_guard._flag_task_ids`` — decompose a finding's
task_id value through this function, so its contract is pinned once, here.

The splitter decomposes; it does not judge.  Whether a component could name
a real task is the consumer's call (the guard screens with
``_is_usable_task_id``), which is why ``-5`` and ``'0'`` survive below.
"""

from __future__ import annotations

from typing import Any

import pytest

from fused_memory.reconciliation.flag_task_ids import task_id_components


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('3105', ('3105',)),
        ('3105,4223', ('3105', '4223')),
        ('3105, 4223 ,5080', ('3105', '4223', '5080')),
        ('3105,4223,3105', ('3105', '4223')),
        (3105, ('3105',)),
        ('dark_factory:3105,3105.1', ('dark_factory:3105', '3105.1')),
    ],
)
def test_decomposes_str_and_int_values(raw: Any, expected: tuple[str, ...]) -> None:
    assert task_id_components(raw) == expected


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        (-5, ('-5',)),
        ('0,3105', ('0', '3105')),
    ],
)
def test_does_not_judge_usability(raw: Any, expected: tuple[str, ...]) -> None:
    assert task_id_components(raw) == expected


@pytest.mark.parametrize(
    'raw',
    [True, False, None, 3105.0, [], ['3105'], {'task_id': '3105'}, b'3105'],
)
def test_a_value_that_is_not_a_task_id_yields_nothing(raw: Any) -> None:
    assert task_id_components(raw) == ()


@pytest.mark.parametrize('raw', [',', ',,', ' , ', ''])
def test_a_separator_only_or_blank_value_yields_nothing(raw: str) -> None:
    assert task_id_components(raw) == ()
