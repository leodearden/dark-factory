"""The ``get_task_facts`` reply shape for a ``{id: status}`` map, for task-store doubles."""

from __future__ import annotations

from collections.abc import Mapping


def task_facts_rows(statuses: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """One row per id: the status as given (so ``None`` survives), no dependencies, no provenance."""
    return {
        task_id: {
            'status': status,
            'title': f'Task {task_id}',
            'dependencies': [],
            'done_provenance_kind': None,
            'done_provenance_commit': None,
        }
        for task_id, status in statuses.items()
    }
