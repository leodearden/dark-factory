"""The public seams of the task-snapshot collectors."""

from __future__ import annotations

from dataclasses import replace

import pytest

from dashboard.data.active_tasks import TASKS_BUDGET


@pytest.mark.parametrize(
    ('field', 'bad'),
    [('per_project', 0), ('total', -1.0), ('root_concurrency', 0)],
)
def test_tasks_budget_refuses_a_bound_that_cannot_serve_a_root(field, bad):
    with pytest.raises(ValueError, match=field) as refused:
        replace(TASKS_BUDGET, **{field: bad})

    assert str(bad) in str(refused.value), (
        f'the refusal must name the offending value {bad!r}, got {refused.value}'
    )
