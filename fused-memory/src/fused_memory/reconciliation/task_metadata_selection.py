"""Select task ids by a predicate over each task's metadata.

Both gate selectors — ``curator_gate_resolution_sweep.py::extract_open_gate_task_ids``
and ``gate_owned_finding_phrasing.py::extract_human_gated_task_ids`` — walk the
task list through this one function and differ only in the metadata predicate
they pass, so a new malformed shape or id-normalisation change reaches both
(task 5710).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

__all__ = ['select_task_ids_by_metadata']


def select_task_ids_by_metadata(
    tasks: Iterable[object],
    metadata_predicate: Callable[[dict[str, Any]], bool],
) -> list[str]:
    """Return the sorted, deduped str ids of tasks whose metadata satisfies *metadata_predicate*.

    - A task contributes ``str(task['id'])`` when it is a ``dict``, its
      ``metadata`` is a ``dict`` satisfying *metadata_predicate*, and its id is
      not ``None``.  Every other shape is skipped rather than raised on.
    - The predicate is only ever called with a ``dict``, so it may ``.get()``
      freely.
    - Sorted and deduped: the ``int`` and ``str`` spellings of one id collapse.

    Pure: no I/O, no side effects.  Empty input returns ``[]``.
    """
    seen: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            continue
        metadata = task.get('metadata')
        if not isinstance(metadata, dict) or not metadata_predicate(metadata):
            continue
        raw_tid = task.get('id')
        if raw_tid is None:
            continue
        seen.add(str(raw_tid))
    return sorted(seen)
