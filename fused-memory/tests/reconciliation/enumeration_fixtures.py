"""Shared fixtures for the enumeration liveness gate's tests (task 3056).

Imported as ``from reconciliation.enumeration_fixtures import ...`` by both
``tests/reconciliation/test_enumeration_guard.py`` (the gate's rule matrix) and
``tests/test_submit_task_enumeration_gate.py`` (the MCP tool boundary).

The ids are the real ones from reify gate 5537: three live entries plus
0d542614, deleted about eleven minutes before the gate shipped.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

LIVE_B3F = 'b3f55b47-0879-49e0-a26d-53bf6ea9689a'
LIVE_21F = '21f78de1-a5f3-4259-b907-16c8f2328e2f'
LIVE_23E = '23eff0cd-8515-40fb-9b3e-9a5be19cbc0c'
DELETED_0D5 = '0d542614-874b-4003-b136-76dd9d8b40f8'
LIVE_IDS = (LIVE_B3F, LIVE_21F, LIVE_23E)

# Canonical UUIDs that are NOT Mem0 ids: Stage-1 finding ids, which
# get_memory_by_id (a Mem0 point read) reports as absent.
FINDING_IDS = [
    'f11018d4-be96-456b-9ab9-cf6814155500',
    '00004132-cd91-41dc-a059-bcd698bdd73c',
]

VALID_SCOPE = {
    'stores': ['mem0'],
    'categories': ['observations_and_summaries', 'procedural_knowledge'],
    'exhaustive': False,
    'note': 'Stage-1 search over two categories only',
}


def fake_memory_service(
    *,
    live: tuple[str, ...] = (),
    raising: dict[str, Exception] | None = None,
) -> AsyncMock:
    """A memory service whose point read resolves *live* ids, raises for the
    ids in *raising*, and returns None (a genuine miss) for everything else."""
    failures = dict(raising or {})

    async def _get(_project_id: str, memory_id: str) -> dict[str, Any] | None:
        if memory_id in failures:
            raise failures[memory_id]
        if memory_id in live:
            return {'id': memory_id, 'content': 'x', 'metadata': {}}
        return None

    service = AsyncMock()
    service.get_memory_by_id = AsyncMock(side_effect=_get)
    return service
