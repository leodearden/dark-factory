"""The follow-up holder: which created tasks wait behind it, and what it refuses (plans/follow-up-filing-shape-prd.md §4.3).

curator_types.py imports this module, so it must keep importing only the stdlib and lower layers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final

from shared.followup_hold_records import (
    FOLLOWUP_HOLD_METADATA_KEY,
    FOLLOWUP_HOLDER_METADATA_KEY,
    FollowupHold,
    FollowupHolder,
)
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import ACTIVE, TaskStatus

from fused_memory.backends.task_backend_errors import (
    TASKMASTER_TOOL_ERROR,
    TaskmasterError,
    TaskNotFoundError,
)
from fused_memory.middleware.metadata_dict import raw_metadata_dict
from fused_memory.models.reconciliation import EventType
from fused_memory.models.scope import resolve_project_id

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol

logger = logging.getLogger(__name__)

__all__ = [
    'HELD_PRIORITIES',
    'HOLDER_GATE_KEYS',
    'PROTECTED_STATUSES',
    'BackfillEmit',
    'DuplicateFollowupHolderError',
    'FollowupHolderProtectedError',
    'FollowupHolders',
    'HolderProtection',
    'backfill',
    'held_count',
    'hold_at_create',
    'hold_scope',
    'is_holder',
    'protected_status',
    'protected_update',
    'protection_refusal',
]

HELD_PRIORITIES: Final = frozenset({'low', 'polish'})
PROTECTED_STATUSES: Final = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED, TaskStatus.DEFERRED})
HOLDER_GATE_KEYS: Final = frozenset({FOLLOWUP_HOLDER_METADATA_KEY, 'deferred_watch', 'trigger_met'})

_WAITING_STATUSES: Final = frozenset(ACTIVE - {TaskStatus.DEFERRED})

_FOLLOWUP_SOURCE: Final = 'agent-followup'
_REVIEW_SUGGESTIONS_CONTEXT: Final = 'review_suggestions'
_HOLD_EXCLUDING_KEYS: Final = ('execution_class', 'prd_path', 'x_finding_key', 'human_decomposed')
_HOLD_SCOPE_KEYS: Final = ('source', 'spawn_context', 'task_kind', *_HOLD_EXCLUDING_KEYS)
_BACKFILL_FIELDS: Final = (
    'priority',
    'dependencies',
    *(f'metadata.{key}' for key in (*_HOLD_SCOPE_KEYS, FOLLOWUP_HOLD_METADATA_KEY)),
)
_METADATA_SOURCE: Final = 'followup_hold'


def is_holder(task: Mapping[str, object]) -> bool:
    metadata = coerce_task_metadata(task.get('metadata'))
    return metadata is not None and metadata.get(FOLLOWUP_HOLDER_METADATA_KEY) is not None


def hold_scope(metadata: object, priority: object) -> bool:
    """Decision 9: a low follow-up or reviewer suggestion of the normal kind that nothing cites.

    ``human_decomposed`` is the mark every planning-mode birth carries, so a PRD decompose is
    never held at filing or by the backfill.
    """
    if not isinstance(priority, str) or priority not in HELD_PRIORITIES:
        return False
    fields = raw_metadata_dict(metadata, source=_METADATA_SOURCE)
    filed_as_followup = (
        fields.get('source') == _FOLLOWUP_SOURCE
        or fields.get('spawn_context') == _REVIEW_SUGGESTIONS_CONTEXT
    )
    return (
        filed_as_followup
        and fields.get('task_kind') in (None, 'normal')
        and not any(fields.get(key) for key in _HOLD_EXCLUDING_KEYS)
    )


def hold_at_create(
    kwargs: Mapping[str, object], metadata: object, holder_id: int, *, held_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The ``add_task`` arguments and metadata of a task born held: the holder edge and its record."""
    record = FollowupHold(holder=holder_id, held_at=held_at)
    held_metadata = {
        **raw_metadata_dict(metadata, source=_METADATA_SOURCE),
        FOLLOWUP_HOLD_METADATA_KEY: record.model_dump(mode='json', exclude_none=True),
    }
    held_kwargs = {**kwargs, 'dependencies': _with_holder_edge(kwargs.get('dependencies'), holder_id)}
    return held_kwargs, held_metadata


def _with_holder_edge(dependencies: object, holder_id: int) -> str:
    if dependencies is not None and not isinstance(dependencies, str):
        raise TypeError(
            f'add_task dependencies must be a comma-separated id string, got {type(dependencies).__name__}'
        )
    named = [part.strip() for part in (dependencies or '').split(',') if part.strip()]
    if str(holder_id) not in named:
        named.append(str(holder_id))
    return ','.join(named)


_HOLDER_WATCH = (
    'The follow-up holder never dispatches: it only parks held low follow-ups behind one '
    'dependency edge until each is promoted (plans/follow-up-filing-shape-prd.md §4.3).'
)
_HOLDER_DESCRIPTION = (
    "This project's follow-up holder. Low-priority agent follow-ups and reviewer suggestions are "
    'filed depending on it, so they wait here until a citation, a combine, a pack or a human '
    'promotes them.'
)
_HOLDER_DETAILS = (
    'Minted by fused-memory (plans/follow-up-filing-shape-prd.md §4.3). It is a deterministic pure '
    'gate with deferred_watch set, so it never dispatches. set_task_status refuses done, cancelled '
    'and deferred on it unless release_holder=True is passed, because that releases every held '
    'follow-up at once. See docs/task-authoring.md §3.4.'
)


class DuplicateFollowupHolderError(TaskmasterError):
    """More than one live task carries the holder marker: the one-holder invariant is already broken."""

    def __init__(self, project_id: str, holder_ids: Sequence[int]) -> None:
        self.project_id = project_id
        self.holder_ids = tuple(holder_ids)
        named = ', '.join(str(holder_id) for holder_id in self.holder_ids)
        super().__init__(
            TASKMASTER_TOOL_ERROR,
            f'project {project_id} has more than one live follow-up holder (tasks {named}); '
            f'nothing was held. Release all but one with release_holder=True.',
        )


class FollowupHolders:
    """Each project's holder id, memoised; the store's marker stays the authority for which task it is."""

    def __init__(self) -> None:
        self._by_root: dict[str, int] = {}

    async def ensure_holder(self, tm: TaskBackendProtocol, project_root: str) -> int:
        """The project's live holder, minted on first need.

        The caller holds the project's write lock, so lookup-then-mint is atomic. A memo hit is
        re-read, so a released or removed holder is never served.
        """
        remembered = self._by_root.get(project_root)
        if remembered is not None and await _is_live_holder(tm, project_root, remembered):
            return remembered
        holder_id = await _live_holder(tm, project_root)
        if holder_id is None:
            holder_id = await _mint_holder(tm, project_root)
        self._by_root[project_root] = holder_id
        return holder_id


async def _is_live_holder(tm: TaskBackendProtocol, project_root: str, holder_id: int) -> bool:
    try:
        row = await tm.get_task(str(holder_id), project_root)
    except TaskNotFoundError:
        return False
    return is_holder(row) and row.get('status') in ACTIVE


async def _live_holder(tm: TaskBackendProtocol, project_root: str) -> int | None:
    found = await tm.find_tasks_by_metadata(
        project_root, FOLLOWUP_HOLDER_METADATA_KEY, statuses=sorted(ACTIVE), limit=2,
    )
    holder_ids = [int(row['task_id']) for row in found['results']]
    if len(holder_ids) > 1:
        raise DuplicateFollowupHolderError(resolve_project_id(project_root), holder_ids)
    return holder_ids[0] if holder_ids else None


async def _mint_holder(tm: TaskBackendProtocol, project_root: str) -> int:
    project_id = resolve_project_id(project_root)
    marker = FollowupHolder(project_id=project_id, minted_at=datetime.now(UTC).isoformat())
    metadata = {
        'task_kind': 'deterministic',
        'always_escalates': True,
        'execution_class': 'decision',
        'deferred_watch': _HOLDER_WATCH,
        FOLLOWUP_HOLDER_METADATA_KEY: marker.model_dump(mode='json'),
    }
    created = await tm.add_task(
        project_root=project_root,
        title=f'Follow-up holder ({project_id})',
        description=_HOLDER_DESCRIPTION,
        details=_HOLDER_DETAILS,
        priority='low',
        metadata=json.dumps(metadata),
    )
    holder_id = int(created['id'])
    logger.info('followup_hold: minted follow-up holder task %s for project %s', holder_id, project_id)
    return holder_id


class HolderProtection(StrEnum):
    STATUS = 'protected_status'
    MARKER = 'marker_stripped'
    DEFERRED_WATCH = 'deferred_watch_stripped'
    TRIGGER_MET = 'trigger_met_set'


def protected_status(
    task: Mapping[str, object], status: str, *, release_holder: bool,
) -> HolderProtection | None:
    """Decision 13: a holder leaves ``pending`` for a status a dependent reads as settled only on release.

    Only a literal ``True`` releases: a whole-backlog release is a human act, never a truthy accident.
    """
    if release_holder is True or status not in PROTECTED_STATUSES or not is_holder(task):
        return None
    return HolderProtection.STATUS


def protected_update(
    stored_metadata: object, payload: object, *, replace: bool,
) -> HolderProtection | None:
    """Decision 13: an update may not strip a holder's marker or ``deferred_watch``, nor set ``trigger_met``.

    The judgement is on the metadata the write would leave, so a ``replace`` that echoes the
    stored gate passes and one that drops it does not.
    """
    stored = raw_metadata_dict(stored_metadata, source=_METADATA_SOURCE)
    marker = stored.get(FOLLOWUP_HOLDER_METADATA_KEY)
    if marker is None:
        return None
    written = raw_metadata_dict(payload, source=_METADATA_SOURCE)
    after = written if replace else {**stored, **written}
    if after.get(FOLLOWUP_HOLDER_METADATA_KEY) != marker:
        return HolderProtection.MARKER
    if stored.get('deferred_watch') and not after.get('deferred_watch'):
        return HolderProtection.DEFERRED_WATCH
    if after.get('trigger_met') and not stored.get('trigger_met'):
        return HolderProtection.TRIGGER_MET
    return None


_HOLDER_PROTECTED_HINT = (
    'This task is its project\'s follow-up holder: every held low follow-up depends on it, and a '
    'done, cancelled or deferred dependency counts as settled, so leaving pending would release '
    'the whole held backlog at once. That release is a human act: pass release_holder=True to '
    'set_task_status. update_task may not strip followup_holder or deferred_watch, nor set '
    'trigger_met. See docs/task-authoring.md §3.4.'
)


class FollowupHolderProtectedError(TaskmasterError):
    """A status or metadata write that would release or ungate the follow-up holder.

    ``attempted_status`` is None for an ``update_task`` refusal. ``held_count`` is how many active
    tasks depend on the holder. Call :meth:`to_error_dict` for the wire shape; callers branch on
    ``error == ERROR``.
    """

    ERROR: ClassVar[str] = 'followup_holder_protected'

    def __init__(
        self,
        *,
        holder_id: str,
        protection: HolderProtection,
        attempted_status: str | None,
        held_count: int,
    ) -> None:
        self.holder_id = holder_id
        self.protection = protection
        self.attempted_status = attempted_status
        self.held_count = held_count
        super().__init__(TASKMASTER_TOOL_ERROR, f'{protection.value}: {_HOLDER_PROTECTED_HINT}')

    def event_payload(self) -> dict[str, object]:
        return {
            'holder_id': self.holder_id,
            'attempted_status': self.attempted_status,
            'protection': self.protection.value,
            'held_count': self.held_count,
        }

    def to_error_dict(self) -> dict[str, Any]:
        return {
            'success': False,
            'error': self.ERROR,
            'error_type': type(self).__name__,
            **self.event_payload(),
            'hint': _HOLDER_PROTECTED_HINT,
        }


async def held_count(tm: TaskBackendProtocol, project_root: str, holder_id: int | str) -> int:
    """How many active tasks depend on the holder: the edge is the authority for held now."""
    holder = int(holder_id)
    edges = await tm.get_dependency_edges(project_root)
    dependents = [str(task_id) for task_id, dependencies in edges.items() if holder in dependencies]
    if not dependents:
        return 0
    statuses = await tm.get_statuses(project_root, ids=dependents)
    return sum(1 for status in statuses.values() if status in ACTIVE)


async def protection_refusal(
    tm: TaskBackendProtocol,
    project_root: str,
    holder_id: str,
    protection: HolderProtection,
    attempted_status: str | None,
) -> FollowupHolderProtectedError:
    return FollowupHolderProtectedError(
        holder_id=holder_id,
        protection=protection,
        attempted_status=attempted_status,
        held_count=await held_count(tm, project_root, holder_id),
    )


BackfillEmit = Callable[[EventType, Mapping[str, object]], Awaitable[None]]


async def backfill(
    tm: TaskBackendProtocol,
    project_root: str,
    *,
    holders: FollowupHolders,
    write_lock: asyncio.Lock,
    emit: BackfillEmit,
) -> int:
    """Decision 12: hold every pending task the hold rule matches that was never held, and report once.

    Each task's edge and record land in one backend write under the interceptor's write lock, with
    no journal row and no ``task_modified`` event. A task that already carries a ``followup_hold``
    record was promoted or released and is never re-held. A pending follow-up that an active,
    non-deferred task outside the stock waits on, directly or through other stock tasks, is left
    unheld, because nothing would promote it. One ``followup_hold_backfilled`` event is emitted per
    call, its count possibly 0, so a reader can tell the backfill ran.
    """
    async with write_lock:
        held = await _hold_stock(tm, project_root, holders)
    await emit(
        EventType.followup_hold_backfilled,
        {'project_id': resolve_project_id(project_root), 'count': held},
    )
    return held


async def _hold_stock(tm: TaskBackendProtocol, project_root: str, holders: FollowupHolders) -> int:
    candidates = await _unheld_stock(tm, project_root)
    waited_on = await _waited_on(tm, project_root, frozenset(int(task['id']) for task in candidates))
    stock = [task for task in candidates if int(task['id']) not in waited_on]
    if not stock:
        return 0
    holder_id = await holders.ensure_holder(tm, project_root)
    record = FollowupHold(holder=holder_id, held_at=datetime.now(UTC).isoformat())
    record_json = json.dumps({FOLLOWUP_HOLD_METADATA_KEY: record.model_dump(mode='json', exclude_none=True)})
    held = 0
    for task in stock:
        dependencies = [str(dependency) for dependency in task.get('dependencies') or []]
        if str(holder_id) in dependencies:
            continue
        await tm.update_task(
            str(task['id']), project_root,
            metadata=record_json, metadata_mode='merge', dependencies=[*dependencies, str(holder_id)],
        )
        held += 1
    return held


async def _unheld_stock(tm: TaskBackendProtocol, project_root: str) -> list[dict[str, Any]]:
    """The pending tasks the hold rule matches that were never held."""
    pending = await tm.get_tasks(project_root, statuses=[TaskStatus.PENDING], fields=_BACKFILL_FIELDS)
    return [
        task for task in pending['tasks']
        if hold_scope(task.get('metadata'), task.get('priority'))
        and FOLLOWUP_HOLD_METADATA_KEY not in raw_metadata_dict(task.get('metadata'), source=_METADATA_SOURCE)
    ]


async def _waited_on(
    tm: TaskBackendProtocol, project_root: str, candidate_ids: frozenset[int],
) -> frozenset[int]:
    """The candidates a task waiting now depends on, directly or through other candidates."""
    if not candidate_ids:
        return frozenset()
    edges = await tm.get_dependency_edges(project_root)
    seeds = await _cited_by_waiting_dependents(tm, project_root, candidate_ids, edges)
    return _dependency_closure(seeds, edges, within=candidate_ids)


async def _cited_by_waiting_dependents(
    tm: TaskBackendProtocol,
    project_root: str,
    candidate_ids: frozenset[int],
    edges: Mapping[int, Sequence[int]],
) -> frozenset[int]:
    cited_by = {
        dependent: candidate_ids.intersection(dependencies)
        for dependent, dependencies in edges.items()
        if dependent not in candidate_ids and not candidate_ids.isdisjoint(dependencies)
    }
    if not cited_by:
        return frozenset()
    statuses = await tm.get_statuses(project_root, ids=[str(dependent) for dependent in cited_by])
    return frozenset(
        candidate
        for dependent, cited in cited_by.items() if statuses.get(str(dependent)) in _WAITING_STATUSES
        for candidate in cited
    )


def _dependency_closure(
    seeds: Iterable[int], edges: Mapping[int, Sequence[int]], within: frozenset[int],
) -> frozenset[int]:
    """The seeds and everything they depend on, directly or transitively, without leaving ``within``."""
    reached: set[int] = set()
    frontier = [seed for seed in seeds if seed in within]
    while frontier:
        task_id = frontier.pop()
        if task_id not in reached:
            reached.add(task_id)
            frontier.extend(dependency for dependency in edges.get(task_id, ()) if dependency in within)
    return frozenset(reached)
