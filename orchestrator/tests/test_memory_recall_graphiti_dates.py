"""Which date a Graphiti briefing row shows, through ``render_memory_results``.

The date tag means when a record entered memory, for every store. A Graphiti
edge's ``created_at`` is its birth, so it wins over ``temporal.valid_at`` (when
the fact became true), which only dates a hit that has no ``created_at``.
"""

from __future__ import annotations

import pytest
from _briefing_helpers import _result

from orchestrator.agents.memory_recall import render_memory_results

BORN = '2026-10-01T12:00:00+00:00'
VALID = '2024-03-15T00:00:00+00:00'


@pytest.mark.parametrize(
    ('created_at', 'valid_at', 'shown'),
    [
        (BORN, VALID, '2026-10-01'),
        (None, VALID, '2024-03-15'),
        (None, None, 'undated'),
    ],
)
def test_a_graphiti_row_is_dated_by_its_birth_before_its_valid_time(
    created_at, valid_at, shown,
):
    edge = {
        **_result('edge-1', 'Service A depends on Service B.'),
        'created_at': created_at,
        'temporal': None if valid_at is None else {'valid_at': valid_at, 'invalid_at': None},
    }

    assert render_memory_results([edge]) == (
        f'- [uncategorized · {shown} · graphiti] Service A depends on Service B.'
    )
