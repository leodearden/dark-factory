"""Live-shaped preserved-specimen records and a fake for the reads that find them.

Task 5456's preserved-specimen gate guard is pinned at two levels: its unit
matrix (``test_preserved_specimen_gate_guard.py``) and the ``submit_task``
boundary (``test_recurring_gate_boundary.py``).  Both need the same live
evidence: task 3105's ``investigation_outcome`` row, its own Graphiti node and
edge, task 5080's reset-gate wording, and a memory service answering the two
corroboration reads.  One copy lives here, so a change to the
``get_memories_by_metadata`` row shape is made once.

``tests/conftest.py`` puts the tests dir on ``sys.path``, so
``from _preserved_specimen_fixtures import ...`` resolves unqualified, as the
other ``tests/_*.py`` helper modules do.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import AsyncMock

SPECIMEN_TASK_ID = '3105'
SPECIMEN_MEMORY_ID = '0397cf9d-0341-47f8-acdb-b78d945b9c8d'
SPECIMEN_EDGE_UUID = 'a8fd36a8-46db-4ca8-a21c-554c38a918ee'
SPECIMEN_NODE_UUID = '95719d7c-3778-4687-8b50-10ec5ad46e75'

#: One live ``MemoryService.get_memories_by_metadata`` row for task 3105, in
#: the ``{'id', 'created_at', 'metadata'}`` shape the scroll returns.
SPECIMEN_ROW: dict[str, Any] = {
    'id': SPECIMEN_MEMORY_ID,
    'created_at': '2026-09-13T08:41:07.512331+00:00',
    'metadata': {
        'kind': 'investigation_outcome',
        'task_id': SPECIMEN_TASK_ID,
        'actionable': False,
        'entity_uuid': SPECIMEN_NODE_UUID,
        'run_id': '5f2c1e0a-7d44-4b0e-9a51-0c6b8d2e9f13',
        'data': (
            "Task 3105's in-progress/null-claimant/null-heartbeat state remains the "
            'deliberately preserved SOLE live validation specimen for gate task 3546; '
            'resetting it would destroy the evidence the soak/flip checklist reads.'
        ),
    },
}

#: ``get_entity('Task 3105')`` answered by the exact-name branch: task 3105's
#: OWN node plus the edge stating its preservation.
OWN_NODE_ENTITY: dict[str, Any] = {
    'nodes': [{'name': 'Task 3105', 'uuid': SPECIMEN_NODE_UUID, 'summary': ''}],
    'edges': [
        {
            'uuid': SPECIMEN_EDGE_UUID,
            'fact': 'Task 3105 is the sole preserved live validation specimen for gate task 3546.',
        }
    ],
}


def reset_gate_title(subject: str = SPECIMEN_TASK_ID) -> str:
    """Task 5080's real title, naming *subject*."""
    return (
        f'Human/orchestrator action needed: reset stranded task {subject} out of '
        f'in-progress (recon Stage 2 is structurally blocked from this write)'
    )


def reset_gate_description(subject: str = SPECIMEN_TASK_ID) -> str:
    """A short 5080-style description: it states the status and asks for nothing."""
    return (
        f'Task {subject} has been status=in-progress with claimant_run_id=null and '
        f'heartbeat_at=null for over 5 weeks, and carries no Live-Workflow Signal '
        f'this cycle.'
    )


def reset_gate_metadata(subject: str = SPECIMEN_TASK_ID) -> dict[str, Any]:
    """Task 5080's real routing shape plus a declared subject.

    ``operational_mode`` is ABSENT, as on 5080.
    """
    return {'execution_class': 'operational', 'gate_subject': subject}


def corroboration_memory(
    rows_by_task: Mapping[str, list[dict[str, Any]]] | None = None,
    entity: dict[str, Any] | None = None,
) -> AsyncMock:
    """A fake memory service whose two corroboration reads are inspectable.

    ``get_memories_by_metadata`` answers ``rows_by_task[filters['task_id']]``,
    nothing by default; ``get_entity`` answers *entity*, an empty graph by
    default.
    """
    rows = dict(rows_by_task or {})
    memory = AsyncMock()
    memory.get_memories_by_metadata.side_effect = (
        lambda *, project_id, filters, **_: list(rows.get(filters['task_id'], []))
    )
    memory.get_entity.return_value = entity if entity is not None else {'nodes': [], 'edges': []}
    return memory


def specimen_memory(entity: dict[str, Any] | None = None) -> AsyncMock:
    """:func:`corroboration_memory` holding task 3105's preservation record."""
    return corroboration_memory({SPECIMEN_TASK_ID: [SPECIMEN_ROW]}, entity=entity)


def corroboration_reads(memory: AsyncMock) -> int:
    """How many corroboration reads *memory* has served."""
    return memory.get_memories_by_metadata.await_count + memory.get_entity.await_count
