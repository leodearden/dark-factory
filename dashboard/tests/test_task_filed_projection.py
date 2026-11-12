"""The ``created_at`` / ``created_at_source`` filing-time pair through the dashboard's row seams.

PRD ``plans/instruments-honesty-prd.md`` task α; the source vocabulary is
``shared/src/shared/task_created_at.py::CreatedAtSource``.
"""

from __future__ import annotations

import pytest

from dashboard.data.tasks import _shape_task

_FILED = '2026-11-12T09:10:17.253Z'


def _row(**overrides) -> dict:
    """A minimally-valid MCP ``get_tasks`` row."""
    row = {
        'id': '7',
        'title': 'a task',
        'status': 'in-progress',
        'priority': 'medium',
        'dependencies': [],
        'metadata': {},
        'updatedAt': '2026-11-12T10:00:00.000Z',
    }
    row.update(overrides)
    return row


class TestShapeTaskCarriesFilingTime:
    @pytest.mark.parametrize('source', ['stamped', 'first_event'])
    def test_preserves_created_at_and_its_source(self, source):
        shaped = _shape_task(_row(created_at=_FILED, created_at_source=source))

        assert shaped is not None
        assert shaped['created_at'] == _FILED
        assert shaped['created_at_source'] == source

    def test_absent_pair_surfaces_none_with_keys_present(self):
        shaped = _shape_task(_row())

        assert shaped is not None
        assert 'created_at' in shaped
        assert 'created_at_source' in shaped
        assert shaped['created_at'] is None
        assert shaped['created_at_source'] is None

    def test_null_pair_on_an_unbackfilled_row_stays_none(self):
        shaped = _shape_task(_row(created_at=None, created_at_source=None))

        assert shaped is not None
        assert shaped['created_at'] is None
        assert shaped['created_at_source'] is None
