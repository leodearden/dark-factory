"""Each recurrence chain's current link, read from one tag's task rows.

docs/prds/recurring-deterministic-tasks.md R-D5: a chain is broken exactly when
its current link is done.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from shared.task_statuses import TERMINAL

__all__ = ['current_links', 'recurrence_key']


def recurrence_key(task: Mapping[str, Any]) -> str | None:
    """The chain key a task row carries in ``metadata.recurrence.key``, else None."""
    metadata = task.get('metadata')
    recurrence = metadata.get('recurrence') if isinstance(metadata, Mapping) else None
    key = recurrence.get('key') if isinstance(recurrence, Mapping) else None
    return key if isinstance(key, str) and key else None


def current_links(tasks: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Map each chain key to its current link: the newest non-terminal link, else the newest link.

    Newest means the highest task id. A tag never reissues an id, so pass the
    rows of ONE tag.
    """
    ranked: dict[str, tuple[tuple[bool, int], Mapping[str, Any]]] = {}
    for task in tasks:
        key = recurrence_key(task)
        if key is None:
            continue
        rank = (task['status'] not in TERMINAL, int(task['id']))
        if key not in ranked or rank > ranked[key][0]:
            ranked[key] = (rank, task)
    return {key: task for key, (_, task) in ranked.items()}
