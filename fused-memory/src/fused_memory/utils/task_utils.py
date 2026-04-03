"""Pure utility functions for task tree manipulation.

These functions are stateless and have no dependency on TaskInterceptor,
TaskmasterBackend, or any other class. They live in the utils layer so that
both the middleware (task_interceptor.py) and the backends layer can import
them without creating a circular dependency.

Layering rule: utils ← backends ← middleware (unidirectional).
"""

from __future__ import annotations

from typing import Any


def _filter_tasks_by_status(tasks: list, statuses: list[str]) -> list:
    """Filter a task list (with subtasks) to only include tasks matching statuses.

    Recursively filters subtasks as well.  A parent task is kept if it matches
    *or* if any of its subtasks match (to preserve the tree structure).
    """
    status_set = set(statuses)
    filtered: list = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        task_status = t.get('status', 'unknown')
        subtasks = t.get('subtasks', [])

        # Recursively filter subtasks
        if isinstance(subtasks, list) and subtasks:
            filtered_subtasks = _filter_tasks_by_status(subtasks, statuses)
        else:
            filtered_subtasks = []

        if task_status in status_set:
            # Include this task; attach filtered subtasks
            filtered.append({**t, 'subtasks': filtered_subtasks})
        elif filtered_subtasks:
            # Parent doesn't match but has matching subtasks — keep parent for tree context
            filtered.append({**t, 'subtasks': filtered_subtasks})
    return filtered


def _collect_all_tasks(tasks: list) -> list:
    """Recursively flatten a task tree into a flat list of task dicts.

    Includes top-level tasks and all subtasks at every nesting level.
    Non-dict elements are ignored.
    """
    flat: list = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        flat.append(t)
        subtasks = t.get('subtasks', [])
        if isinstance(subtasks, list) and subtasks:
            flat.extend(_collect_all_tasks(subtasks))
    return flat


def _compact_task(task: Any) -> Any:
    """Return a compact version of a task dict with verbose fields stripped.

    Keeps: id, status, title, dependencies, priority.
    Strips: description, details, and any other unlisted verbose fields.
    Subtasks are recursively compacted.

    Non-dict inputs are returned unchanged.
    """
    if not isinstance(task, dict):
        return task

    _VERBOSE_FIELDS = {'description', 'details'}
    result = {k: v for k, v in task.items() if k not in _VERBOSE_FIELDS}

    # Recursively compact subtasks
    if 'subtasks' in result and isinstance(result['subtasks'], list):
        result['subtasks'] = [_compact_task(st) for st in result['subtasks']]

    return result


def _compact_tasks(tasks: list) -> list:
    """Apply _compact_task to each element in a list."""
    return [_compact_task(t) for t in tasks]
