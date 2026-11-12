"""Corroborates that tasks exist, each in its OWN project, through one get_task path.

Taskmaster ids are per-project, so a task is looked up under the root of the
project that names it. Every caller shares :func:`corroborate_task_lookups`
and supplies only what counts as confirmation for one lookup.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

#: Keys whose presence on a get_task result dict identifies it as an actual
#: task record (as opposed to e.g. an unrelated empty dict).  Mirrors the
#: shape of both the MCP-wrapper task envelope and the raw sqlite backend row.
_TASK_IDENTITY_KEYS: frozenset[str] = frozenset({'id', 'task_id', 'title', 'status'})


def confirm_task_present(get_task_result: object) -> bool:
    """Fail-safe classifier: True ONLY when get_task POSITIVELY confirms presence.

    True iff the result is a plain task-record dict (no ``'error'`` /
    ``'error_type'`` key) carrying at least one task-identity key (``id``,
    ``task_id``, ``title`` or ``status``) as evidence it is an actual task
    rather than an unrelated empty dict.

    A not-found or inconclusive error dict, ``None``, an empty dict and a
    non-dict value are all False: an uncertain or absent result must never be
    treated as corroboration that a task exists.
    """
    if not isinstance(get_task_result, dict):
        return False
    if 'error' in get_task_result or 'error_type' in get_task_result:
        return False
    return any(key in get_task_result for key in _TASK_IDENTITY_KEYS)


async def safe_get_task(taskmaster: Any, task_id: Any, project_root: str) -> Any:
    """Fetch ONE task, normalising any exception to an error dict.

    A result the backend returns (task dict and error dict alike) passes
    through untouched, and ``(task_id, project_root)`` are forwarded
    positionally in that order. ANY exception becomes
    ``{'error': str(exc), 'error_type': type(exc).__name__}``, so a classifier
    reads the raised path exactly as it reads the MCP wrapper's own error-dict
    path, and callers can use a PLAIN ``asyncio.gather``.
    """
    try:
        return await taskmaster.get_task(task_id, project_root)
    except Exception as exc:
        return {'error': str(exc), 'error_type': type(exc).__name__}


class LookupVerdict(Enum):
    """What one :class:`TaskLookup` established.

    UNCONFIRMED means "we asked and the answer did not confirm"; UNRESOLVABLE
    means "we could not ask" (no root for the project, or no taskmaster).
    """

    CONFIRMED = 'confirmed'
    UNCONFIRMED = 'unconfirmed'
    UNRESOLVABLE = 'unresolvable'


@dataclass(frozen=True)
class TaskLookup:
    """One task to corroborate in its OWN project, and what counts as confirmation.

    *accept* classifies the raw (or exception-normalised) ``get_task`` result;
    :func:`confirm_task_present` is the plain "the task exists" rule.
    """

    project_id: str | None
    task_id: str
    accept: Callable[[object], bool]


def _lookup_root(known_projects: Mapping[str, str], project_id: str | None) -> str | None:
    if not isinstance(project_id, str) or not project_id.strip():
        return None
    return known_projects.get(project_id) or None


def _classify_lookup(lookup: TaskLookup, result: object) -> LookupVerdict:
    try:
        accepted = lookup.accept(result)
    except Exception:
        logger.debug(
            'reconciliation.corroborate_task_lookups: accept predicate raised for '
            'project_id=%s task_id=%s — treating as unconfirmed',
            lookup.project_id, lookup.task_id, exc_info=True,
        )
        return LookupVerdict.UNCONFIRMED
    return LookupVerdict.CONFIRMED if accepted else LookupVerdict.UNCONFIRMED


async def corroborate_task_lookups(
    taskmaster: Any,
    known_projects: Mapping[str, str] | None,
    lookups: Sequence[TaskLookup],
) -> tuple[LookupVerdict, ...]:
    """Corroborate each lookup against its own project's root; one verdict per lookup.

    Every resolvable lookup goes out in ONE plain ``asyncio.gather`` of
    :func:`safe_get_task` (which already normalises exceptions). Confirmation
    fails closed — a raising ``accept`` is UNCONFIRMED — and the pass never
    raises: an unexpected fault logs a WARNING and degrades to all-UNRESOLVABLE.
    """
    all_unresolvable = (LookupVerdict.UNRESOLVABLE,) * len(lookups)
    if not taskmaster or not known_projects or not lookups:
        return all_unresolvable
    try:
        resolvable: list[tuple[int, str]] = []
        for index, lookup in enumerate(lookups):
            root = _lookup_root(known_projects, lookup.project_id)
            if root:
                resolvable.append((index, root))
        results: list[Any] = await asyncio.gather(
            *[
                safe_get_task(taskmaster, lookups[index].task_id, root)
                for index, root in resolvable
            ]
        )
        verdicts: list[LookupVerdict] = list(all_unresolvable)
        for (index, _root), result in zip(resolvable, results, strict=True):
            verdicts[index] = _classify_lookup(lookups[index], result)
        return tuple(verdicts)
    except Exception:
        logger.warning(
            'reconciliation.corroborate_task_lookups: corroboration pass failed for '
            '%d lookup(s) — degrading to unresolvable.',
            len(lookups),
            exc_info=True,
        )
        return all_unresolvable
