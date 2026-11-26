"""The follow-up holder: which created tasks wait behind it, and what it refuses (plans/follow-up-filing-shape-prd.md §4.3).

curator_types.py imports this module, so it must keep importing only the stdlib and lower layers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict, ValidationError
from shared.followup_hold_records import (
    FINDING_SET_HOLD_METADATA_KEY,
    FOLLOWUP_HOLD_METADATA_KEY,
    FOLLOWUP_HOLDER_METADATA_KEY,
    FindingSetHold,
    FollowupHold,
    FollowupHolder,
    PromotionCause,
    PromotionRecord,
)
from shared.project_id import normalize_project_id
from shared.task_metadata import ExternalDep
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import ACTIVE, TaskStatus
from shared.task_transitions import ActorClass, derive_actor_class

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
    'HOLDER_MINT_SOURCE',
    'PROTECTED_STATUSES',
    'DuplicateFollowupHolderError',
    'FollowupHolderProtectedError',
    'FollowupHolders',
    'HoldEmit',
    'HolderEdgeStripRefusedError',
    'HolderMinted',
    'HolderProtection',
    'PromotionDeferral',
    'PromotionDeferredError',
    'PromotionGuard',
    'UpdatePromotion',
    'backfill',
    'carry_hold_record',
    'citation_ref',
    'held_by',
    'held_count',
    'hold_at_create',
    'hold_scope',
    'is_holder',
    'judge_update',
    'may_strip_holder_edge',
    'protected_status',
    'protected_update',
    'promote',
    'promote_after_update',
    'promote_dependencies_of',
    'protection_refusal',
    'register_promotion_guard',
    'release_held',
    'releases_holder',
    'remove_holder_edge',
    'unregister_promotion_guard',
]

HELD_PRIORITIES: Final = frozenset({'low', 'polish'})
PROTECTED_STATUSES: Final = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED, TaskStatus.DEFERRED})
HOLDER_GATE_KEYS: Final = frozenset({FOLLOWUP_HOLDER_METADATA_KEY, 'deferred_watch', 'trigger_met'})
HOLDER_MINT_SOURCE: Final = 'followup-holder-mint'

_WAITING_STATUSES: Final = frozenset(ACTIVE - {TaskStatus.DEFERRED})

_FOLLOWUP_SOURCE: Final = 'agent-followup'
_REVIEW_SUGGESTIONS_CONTEXT: Final = 'review_suggestions'
_FINDING_KEY: Final = 'x_finding_key'
_PRD_PATH_KEY: Final = 'prd_path'
_CITATION_KEYS: Final = (_FINDING_KEY, _PRD_PATH_KEY)
_CITED_KEYS: Final = (*_CITATION_KEYS, 'human_decomposed')
_HOLD_SCOPE_KEYS: Final = (
    'source', 'spawn_context', 'task_kind', 'execution_class', *_CITED_KEYS, FINDING_SET_HOLD_METADATA_KEY,
)
_BACKFILL_FIELDS: Final = (
    'priority',
    'dependencies',
    *(f'metadata.{key}' for key in (*_HOLD_SCOPE_KEYS, FOLLOWUP_HOLD_METADATA_KEY)),
)
_EXTERNAL_DEPS_FIELDS: Final = ('id', 'metadata.external_deps')
_METADATA_SOURCE: Final = 'followup_hold'
_RELEASE_REF: Final = 'release_holder'
_REMOVE_REF: Final = 'remove_dependency'
_UPDATE_REF: Final = 'update_task'


def is_holder(task: Mapping[str, object]) -> bool:
    metadata = coerce_task_metadata(task.get('metadata'))
    return metadata is not None and metadata.get(FOLLOWUP_HOLDER_METADATA_KEY) is not None


def hold_scope(metadata: object, priority: object) -> bool:
    """Decision 9: a low task of the normal kind needing no human, in one of two arms.

    The uncited follow-up arm holds a follow-up or reviewer suggestion that nothing cites.
    ``human_decomposed`` is the mark every planning-mode birth carries, so this arm never holds a
    PRD decompose. The finding-set arm holds a task that cites a finding and carries a valid
    ``finding_set_hold`` marker, whatever its source, and even when it was born in planning mode.
    """
    if not isinstance(priority, str) or priority not in HELD_PRIORITIES:
        return False
    fields = raw_metadata_dict(metadata, source=_METADATA_SOURCE)
    if fields.get('task_kind') not in (None, 'normal') or fields.get('execution_class'):
        return False
    return _uncited_followup(fields) or _finding_set_member(fields)


def _uncited_followup(fields: Mapping[str, object]) -> bool:
    filed_as_followup = (
        fields.get('source') == _FOLLOWUP_SOURCE
        or fields.get('spawn_context') == _REVIEW_SUGGESTIONS_CONTEXT
    )
    return filed_as_followup and not any(fields.get(key) for key in _CITED_KEYS)


def _finding_set_member(fields: Mapping[str, object]) -> bool:
    return bool(fields.get(_FINDING_KEY)) and _finding_set_hold(fields) is not None


def _finding_set_hold(fields: Mapping[str, object], task_id: object = None) -> FindingSetHold | None:
    raw = fields.get(FINDING_SET_HOLD_METADATA_KEY)
    if raw is None:
        return None
    try:
        return FindingSetHold.model_validate(raw)
    except ValidationError as exc:
        logger.warning('followup_hold: task %s carries a malformed finding_set_hold marker: %s', task_id, exc)
        return None


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


def held_by(task: Mapping[str, object]) -> int | None:
    """The holder an active row waits behind now: its record names the edge, and the edge is the authority.

    A promoted row keeps its record but has lost the edge, so it is not held. A record that does
    not parse holds nothing and logs one WARNING naming the task.
    """
    record = _current_hold(task)
    return None if record is None else record.holder


def _current_hold(task: Mapping[str, object]) -> FollowupHold | None:
    if task.get('status') not in ACTIVE:
        return None
    raw = raw_metadata_dict(task.get('metadata'), source=_METADATA_SOURCE).get(FOLLOWUP_HOLD_METADATA_KEY)
    if raw is None:
        return None
    try:
        record = FollowupHold.model_validate(raw)
    except ValidationError as exc:
        logger.warning('followup_hold: task %s carries a malformed followup_hold record: %s', task.get('id'), exc)
        return None
    return record if str(record.holder) in _local_dependencies(task) else None


def _local_dependencies(task: Mapping[str, object]) -> list[str]:
    dependencies = task.get('dependencies') or ()
    return [str(dependency) for dependency in dependencies] if isinstance(dependencies, Iterable) else []


def citation_ref(metadata: object) -> str | None:
    """The citation a metadata payload carries: the first finding key, else the PRD path.

    The same keys exclude a filing from the hold, so a citation that would have kept a task unheld
    is the one that promotes it.
    """
    fields = raw_metadata_dict(metadata, source=_METADATA_SOURCE)
    finding_keys = fields.get(_FINDING_KEY)
    candidates = [*(finding_keys if isinstance(finding_keys, list) else [finding_keys]), fields.get(_PRD_PATH_KEY)]
    return next((value for value in candidates if isinstance(value, str) and value), None)


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
    'and deferred on it unless release_holder=True is passed, because each of them releases every '
    'held follow-up. See docs/task-authoring.md §3.4.'
)


class DuplicateFollowupHolderError(TaskmasterError):
    """More than one live task carries the holder marker: the one-holder invariant is already broken.

    Every filing the hold rule matches fails while this lasts, and so does the project's backfill.
    """

    def __init__(self, project_id: str, holder_ids: Sequence[int]) -> None:
        self.project_id = project_id
        self.holder_ids = tuple(holder_ids)
        named = ', '.join(str(holder_id) for holder_id in self.holder_ids)
        super().__init__(
            TASKMASTER_TOOL_ERROR,
            f'project {project_id} has more than one live follow-up holder (tasks {named}), so this '
            f'filing failed and created no task. Every low follow-up filed in this project fails '
            f'the same way until all but one holder is released with release_holder=True.',
        )


HolderMinted = Callable[[str, int], Awaitable[None]]


class FollowupHolders:
    """Each project's holder id, memoised; the store's marker stays the authority for which task it is.

    ``on_minted`` is awaited with the project root and the new holder's id after every mint, still
    under the caller's write lock, so no holder comes into being unannounced.
    """

    def __init__(self, *, on_minted: HolderMinted) -> None:
        self._by_root: dict[str, int] = {}
        self._on_minted = on_minted

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
            await self._on_minted(project_root, holder_id)
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
    """Decision 13: a holder goes ``done``, ``cancelled`` or ``deferred`` only on release.

    Each of the three releases every held follow-up at once: the release promotes them all at the
    act (:func:`releases_holder`, decision 14). Only a literal ``True`` releases: a whole-backlog
    release is a human act, never a truthy accident.
    """
    if release_holder is True or status not in PROTECTED_STATUSES or not is_holder(task):
        return None
    return HolderProtection.STATUS


def releases_holder(task: Mapping[str, object], status: str, *, release_holder: bool) -> bool:
    """The explicit release :func:`protected_status` lets through: every held follow-up is promoted."""
    return release_holder is True and status in PROTECTED_STATUSES and is_holder(task)


def may_strip_holder_edge(agent_id: str | None) -> bool:
    """Open question 2: only a human may drop a held task's holder edge through a dependency replacement."""
    return derive_actor_class(agent_id) is ActorClass.HUMAN


_HOLDER_EDGE_STRIP_HINT = (
    'This task is a held follow-up: it depends on its project\'s follow-up holder, and a machine '
    'caller may not drop that edge with a dependencies replacement. Keep the holder id in the '
    'dependencies list. A held follow-up leaves the hold by promotion: cite it with x_finding_key '
    'or prd_path, depend on it from a pending task, or have a human call '
    'remove_dependency(task, holder). See docs/task-authoring.md §3.4.'
)


class HolderEdgeStripRefusedError(TaskmasterError):
    """A machine-role ``update_task`` whose dependencies replacement drops a held task's holder edge."""

    ERROR: ClassVar[str] = 'followup_holder_edge_strip_refused'

    def __init__(self, *, task_id: str, holder_id: str, agent_id: str | None) -> None:
        self.task_id = task_id
        self.holder_id = holder_id
        self.agent_id = agent_id
        super().__init__(TASKMASTER_TOOL_ERROR, f'task {task_id} would lose its holder edge to {holder_id}')

    def to_error_dict(self) -> dict[str, Any]:
        return {
            'success': False,
            'error': self.ERROR,
            'error_type': type(self).__name__,
            'task_id': self.task_id,
            'holder_id': self.holder_id,
            'agent_id': self.agent_id,
            'hint': _HOLDER_EDGE_STRIP_HINT,
        }


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


def carry_hold_record(stored_metadata: object, payload: object, *, replace: bool) -> dict[str, Any] | None:
    """The metadata an update must write instead of *payload* so the stored ``followup_hold`` survives it.

    The record is machine-written, by this module's own backend writes only: every promotion
    trigger finds a held task through it, so a held task whose record an update dropped or
    rewrote would wait behind its holder with nothing left to promote it. None when the write
    leaves the stored record as it is, or the row has none.
    """
    stored = raw_metadata_dict(stored_metadata, source=_METADATA_SOURCE)
    record = stored.get(FOLLOWUP_HOLD_METADATA_KEY)
    if record is None:
        return None
    written = raw_metadata_dict(payload, source=_METADATA_SOURCE)
    after = written if replace else {**stored, **written}
    if after.get(FOLLOWUP_HOLD_METADATA_KEY) == record:
        return None
    return {**written, FOLLOWUP_HOLD_METADATA_KEY: record}


_HOLDER_PROTECTED_HINT = (
    'This task is its project\'s follow-up holder: every held low follow-up depends on it. Done, '
    'cancelled or deferred releases every held follow-up at once, promoting each by human. That '
    'release is a human act: pass release_holder=True to set_task_status. update_task may not '
    'strip followup_holder or deferred_watch, nor set trigger_met. See docs/task-authoring.md §3.4.'
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


class PromotionDeferral(BaseModel):
    """A registered guard's answer that a held task must not be promoted yet, and why."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    guard: str
    reason: str


PromotionGuard = Callable[[Mapping[str, object], str, str | None], PromotionDeferral | None]
"""Called with the held task's live row, the promotion cause and its ref; a deferral blocks the promotion."""

_PROMOTION_GUARDS: list[PromotionGuard] = []


def register_promotion_guard(guard: PromotionGuard) -> None:
    """Consult *guard* before every promotion, after the guards registered before it; idempotent."""
    if not any(registered is guard for registered in _PROMOTION_GUARDS):
        _PROMOTION_GUARDS.append(guard)


def unregister_promotion_guard(guard: PromotionGuard) -> None:
    _PROMOTION_GUARDS[:] = [registered for registered in _PROMOTION_GUARDS if registered is not guard]


HoldEmit = Callable[[EventType, Mapping[str, object]], Awaitable[None]]
"""Journals one hold event for the project; the backfill and every promotion emit through it.

A caller outside the interceptor takes the journaling one from ``TaskInterceptor.hold_emit``.
"""

_Seed = tuple[int, PromotionCause, str | None]


async def promote(
    tm: TaskBackendProtocol,
    project_root: str,
    held_id: int,
    *,
    by: PromotionCause | str,
    ref: str | None = None,
    emit: HoldEmit,
) -> tuple[int, ...]:
    """Promote a held task and, by closure, every held task it depends on (decision 14).

    The caller holds the project's interceptor write lock, so each task is read and promoted with
    no write in between. Each promotion is ONE backend write that drops the holder edge and stamps
    ``followup_hold.promoted``, then one ``followup_promoted`` event; ``pending_since`` is never
    written. A registered guard's deferral, or a guard that raises, leaves its task held, emits
    ``followup_promotion_deferred`` and promotes none of that task's dependencies. A task that is
    not held now promotes nothing. Returns the promoted ids in promotion order; an unknown *by*
    raises ``ValueError``.
    """
    walk = await _promote_frontier(tm, project_root, [(held_id, PromotionCause(by), ref)], emit)
    return walk.promoted


async def remove_holder_edge(
    tm: TaskBackendProtocol, project_root: str, task_id: str, depends_on: str, *, emit: HoldEmit,
) -> tuple[int, ...] | None:
    """A human removing a held task's holder edge promotes it by ``human`` (open question 2).

    None when *depends_on* is not the edge *task_id* is held behind, so the caller removes that
    edge plainly. Raises :class:`PromotionDeferredError`, with nothing written, when a guard defers
    the task itself. Same precondition as :func:`promote`.
    """
    row = await _live_row(tm, project_root, task_id)
    record = None if row is None else _current_hold(row)
    if record is None or str(record.holder) != depends_on.strip():
        return None
    walk = await _promote_frontier(tm, project_root, [(int(task_id), PromotionCause.HUMAN, _REMOVE_REF)], emit)
    deferred = walk.deferred.get(int(task_id))
    if deferred is not None:
        raise PromotionDeferredError(
            task_id=task_id, holder_id=str(record.holder), cause=PromotionCause.HUMAN, deferral=deferred,
        )
    return walk.promoted


@dataclass(frozen=True)
class UpdatePromotion:
    """What an ``update_task`` promotes once its write lands, judged by :func:`judge_update` before it."""

    task_id: str
    human_release: FollowupHold | None = None
    """The hold a human ``dependencies`` replacement ends; the write itself drops the holder edge."""
    citation: str | None = None
    dependencies_written: bool = False


async def judge_update(
    tm: TaskBackendProtocol,
    project_root: str,
    task_id: str,
    before: Mapping[str, object] | None,
    *,
    metadata: object,
    dependencies: Sequence[object] | None,
    agent_id: str | None,
    emit: HoldEmit,
) -> UpdatePromotion:
    """Decision 14 for an update, judged before its write; *before* is the row already read, if any.

    A ``dependencies`` replacement that leaves out a held task's holder edge is a human promotion
    (open question 2). A machine caller gets :class:`HolderEdgeStripRefusedError`, and a guard's
    deferral emits ``followup_promotion_deferred`` and raises :class:`PromotionDeferredError`;
    either way the caller must write nothing. The promotion is carried out only once the write
    lands (:func:`promote_after_update`), so a write the backend refuses leaves the task held.
    Same precondition as :func:`promote`.
    """
    human_release = None
    if dependencies is not None:
        row = before if before is not None else await _live_row(tm, project_root, task_id)
        if row is not None:
            human_release = await _replacement_release(row, dependencies, agent_id, emit)
    return UpdatePromotion(
        task_id=task_id,
        human_release=human_release,
        citation=citation_ref(metadata),
        dependencies_written=dependencies is not None,
    )


async def _replacement_release(
    row: Mapping[str, object], dependencies: Sequence[object], agent_id: str | None, emit: HoldEmit,
) -> FollowupHold | None:
    record = _current_hold(row)
    if record is None or str(record.holder) in {str(dependency).strip() for dependency in dependencies}:
        return None
    task_id = str(row['id'])
    if not may_strip_holder_edge(agent_id):
        raise HolderEdgeStripRefusedError(task_id=task_id, holder_id=str(record.holder), agent_id=agent_id)
    deferral = _first_deferral(row, PromotionCause.HUMAN, _UPDATE_REF)
    if deferral is not None:
        await _emit_deferred(emit, task_id, PromotionCause.HUMAN, _UPDATE_REF, deferral)
        raise PromotionDeferredError(
            task_id=task_id, holder_id=str(record.holder), cause=PromotionCause.HUMAN, deferral=deferral,
        )
    return record


async def promote_after_update(
    tm: TaskBackendProtocol, project_root: str, judged: UpdatePromotion, *, emit: HoldEmit,
) -> tuple[int, ...]:
    """Carry out what :func:`judge_update` judged, once the update landed; returns the promoted ids.

    A human release stamps the record the update left (the update dropped the edge), then
    promotes the task's held dependencies by ``dependent``; a citation promotes the task; a
    ``dependencies`` replacement is an edge insert. Same precondition as :func:`promote`.
    """
    promoted: list[int] = []
    if judged.human_release is not None:
        promoted += await _stamp_human_release(tm, project_root, judged.task_id, judged.human_release, emit)
    if judged.citation is not None:
        promoted += await promote(
            tm, project_root, int(judged.task_id), by=PromotionCause.CITATION, ref=judged.citation, emit=emit,
        )
    if judged.dependencies_written:
        promoted += await promote_dependencies_of(tm, project_root, int(judged.task_id), emit=emit)
    return tuple(promoted)


async def _stamp_human_release(
    tm: TaskBackendProtocol, project_root: str, task_id: str, record: FollowupHold, emit: HoldEmit,
) -> list[int]:
    await tm.update_task(
        task_id, project_root,
        metadata=_promoted_metadata(record, PromotionCause.HUMAN, _UPDATE_REF), metadata_mode='merge',
    )
    await _emit_promoted(emit, task_id, PromotionCause.HUMAN, _UPDATE_REF)
    row = await _live_row(tm, project_root, task_id)
    seeds = [] if row is None else _dependent_seeds(task_id, _local_dependencies(row))
    return [int(task_id), *(await _promote_frontier(tm, project_root, seeds, emit)).promoted]


async def promote_dependencies_of(
    tm: TaskBackendProtocol, project_root: str, task_id: int, *, emit: HoldEmit,
) -> tuple[int, ...]:
    """An edge onto a held task promotes it once its dependent is pending and not itself held.

    So an edge between two held tasks keeps both held. Same precondition as :func:`promote`.
    """
    row = await _live_row(tm, project_root, task_id)
    if row is None or row.get('status') != TaskStatus.PENDING or held_by(row) is not None:
        return ()
    seeds = _dependent_seeds(task_id, _local_dependencies(row))
    return (await _promote_frontier(tm, project_root, seeds, emit)).promoted


async def release_held(
    tm: TaskBackendProtocol, project_root: str, holder_id: int, *, emit: HoldEmit,
) -> tuple[int, ...]:
    """The explicit release: every task held behind the holder is promoted by a human.

    Every dependent is seeded before any closure step, so each is attributed to the release. A
    task already promoted is skipped, so a repeated release finishes one that stopped partway.
    Same precondition as :func:`promote`.
    """
    edges = await tm.get_dependency_edges(project_root)
    dependents = sorted(task_id for task_id, dependencies in edges.items() if holder_id in dependencies)
    walk = await _promote_frontier(
        tm, project_root, [(dependent, PromotionCause.HUMAN, _RELEASE_REF) for dependent in dependents], emit,
    )
    return walk.promoted


@dataclass(frozen=True)
class _Walk:
    promoted: tuple[int, ...]
    deferred: Mapping[int, PromotionDeferral]


async def _promote_frontier(
    tm: TaskBackendProtocol, project_root: str, seeds: Sequence[_Seed], emit: HoldEmit,
) -> _Walk:
    """Breadth-first over the seeds, then over each promoted task's other dependencies; each task once."""
    frontier = deque(seeds)
    seen: set[int] = set()
    promoted: list[int] = []
    deferred: dict[int, PromotionDeferral] = {}
    while frontier:
        task_id, cause, ref = frontier.popleft()
        if task_id in seen:
            continue
        seen.add(task_id)
        row = await _live_row(tm, project_root, task_id)
        record = None if row is None else _current_hold(row)
        if row is None or record is None:
            continue
        deferral = _first_deferral(row, cause, ref)
        if deferral is not None:
            deferred[task_id] = deferral
            await _emit_deferred(emit, task_id, cause, ref, deferral)
            continue
        remaining = await _write_promotion(tm, project_root, row, record, cause, ref)
        await _emit_promoted(emit, task_id, cause, ref)
        promoted.append(task_id)
        frontier.extend(_dependent_seeds(task_id, remaining))
    return _Walk(promoted=tuple(promoted), deferred=deferred)


def _dependent_seeds(task_id: int | str, dependencies: Iterable[object]) -> list[_Seed]:
    """Each dependency of a task just promoted, promoted in turn by ``dependent`` with the task as ref."""
    return [(int(str(dependency)), PromotionCause.DEPENDENT, str(task_id)) for dependency in dependencies]


async def _emit_promoted(emit: HoldEmit, task_id: int | str, cause: PromotionCause, ref: str | None) -> None:
    await emit(EventType.followup_promoted, {'task_id': str(task_id), 'by': cause.value, 'ref': ref})


async def _emit_deferred(
    emit: HoldEmit, task_id: int | str, cause: PromotionCause, ref: str | None, deferral: PromotionDeferral,
) -> None:
    await emit(
        EventType.followup_promotion_deferred,
        {'task_id': str(task_id), 'by': cause.value, 'ref': ref, **deferral.model_dump()},
    )


async def _live_row(tm: TaskBackendProtocol, project_root: str, task_id: int | str) -> dict[str, Any] | None:
    try:
        return await tm.get_task(str(task_id), project_root)
    except TaskNotFoundError:
        return None


async def _write_promotion(
    tm: TaskBackendProtocol,
    project_root: str,
    row: Mapping[str, object],
    record: FollowupHold,
    cause: PromotionCause,
    ref: str | None,
) -> list[int]:
    """ONE write: the holder edge dropped and the promotion stamped. Returns the dependencies left."""
    remaining = [dependency for dependency in _local_dependencies(row) if dependency != str(record.holder)]
    await tm.update_task(
        str(row['id']), project_root,
        metadata=_promoted_metadata(record, cause, ref), metadata_mode='merge', dependencies=remaining,
    )
    return [int(dependency) for dependency in remaining]


def _promoted_metadata(record: FollowupHold, cause: PromotionCause, ref: str | None) -> str:
    stamped = record.model_copy(
        update={'promoted': PromotionRecord(by=cause, at=datetime.now(UTC).isoformat(), ref=ref)},
    )
    return json.dumps({FOLLOWUP_HOLD_METADATA_KEY: stamped.model_dump(mode='json', exclude_none=True)})


_PROMOTION_DEFERRED_HINT = (
    'A registered promotion guard deferred this promotion, so the held task keeps its holder edge '
    'and nothing was written. guard and reason say why; the followup_promotion_deferred event '
    'records the same. Retry once the guard\'s condition clears. See docs/task-authoring.md §3.4.'
)


class PromotionDeferredError(TaskmasterError):
    """A human removal of a held task's holder edge that a registered promotion guard deferred."""

    ERROR: ClassVar[str] = 'followup_promotion_deferred'

    def __init__(self, *, task_id: str, holder_id: str, cause: PromotionCause, deferral: PromotionDeferral) -> None:
        self.task_id = task_id
        self.holder_id = holder_id
        self.cause = cause
        self.deferral = deferral
        super().__init__(
            TASKMASTER_TOOL_ERROR,
            f'promotion of task {task_id} deferred by guard {deferral.guard}: {deferral.reason}',
        )

    def to_error_dict(self) -> dict[str, Any]:
        return {
            'success': False,
            'error': self.ERROR,
            'error_type': type(self).__name__,
            'task_id': self.task_id,
            'holder_id': self.holder_id,
            'by': self.cause.value,
            **self.deferral.model_dump(),
            'hint': _PROMOTION_DEFERRED_HINT,
        }


def _first_deferral(row: Mapping[str, object], cause: PromotionCause, ref: str | None) -> PromotionDeferral | None:
    """The first registered guard's deferral; a guard that raises defers too, so a fault keeps the task held."""
    for guard in tuple(_PROMOTION_GUARDS):
        try:
            deferral = guard(row, cause.value, ref)
        except Exception as exc:
            name = getattr(guard, '__qualname__', type(guard).__qualname__)
            logger.warning('followup_hold: promotion guard %s raised for task %s', name, row.get('id'), exc_info=True)
            return PromotionDeferral(guard=name, reason=f'the guard raised {type(exc).__name__}: {exc}')
        if deferral is not None:
            return deferral
    return None


async def backfill(
    tm: TaskBackendProtocol,
    project_root: str,
    *,
    holders: FollowupHolders,
    write_lock: asyncio.Lock,
    emit: HoldEmit,
    known_roots: Sequence[str],
) -> int:
    """Decision 12: hold every pending task the hold rule matches that was never held, and report once.

    Each task's edge and record land in one backend write under the interceptor's write lock, with
    no journal row and no ``task_modified`` event. A task that already carries a ``followup_hold``
    record was promoted or released and is never re-held. A pending follow-up that an active,
    non-deferred task outside the stock waits on, directly or through other stock tasks, is left
    unheld, because nothing would promote it; ``known_roots`` are the projects whose
    ``external_deps`` can cite this one. One ``followup_hold_backfilled`` event is emitted per
    call, its count possibly 0, so a reader can tell the backfill ran.
    """
    async with write_lock:
        held = await _hold_stock(tm, project_root, holders, known_roots)
    await emit(
        EventType.followup_hold_backfilled,
        {'project_id': resolve_project_id(project_root), 'count': held},
    )
    return held


async def _hold_stock(
    tm: TaskBackendProtocol, project_root: str, holders: FollowupHolders, known_roots: Sequence[str],
) -> int:
    candidates = await _unheld_stock(tm, project_root)
    candidate_ids = frozenset(int(task['id']) for task in candidates)
    waited_on = await _waited_on(tm, project_root, candidate_ids, known_roots)
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
    tm: TaskBackendProtocol, project_root: str, candidate_ids: frozenset[int], known_roots: Sequence[str],
) -> frozenset[int]:
    """The candidates a task waiting now depends on, directly or through other candidates."""
    if not candidate_ids:
        return frozenset()
    edges = await tm.get_dependency_edges(project_root)
    seeds = (
        await _cited_by_waiting_dependents(tm, project_root, candidate_ids, edges)
        | await _cited_by_other_projects(tm, project_root, known_roots)
    )
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


async def _cited_by_other_projects(
    tm: TaskBackendProtocol, project_root: str, known_roots: Sequence[str],
) -> frozenset[int]:
    """This project's task ids that a waiting task in another known project names in ``external_deps``.

    A failed read of another project propagates, so the backfill never holds blind.
    """
    project_id = resolve_project_id(project_root)
    cited: set[int] = set()
    for root in known_roots:
        if resolve_project_id(root) == project_id:
            continue
        waiting = await tm.get_tasks(root, statuses=sorted(_WAITING_STATUSES), fields=_EXTERNAL_DEPS_FIELDS)
        cited.update(
            task_id for task in waiting['tasks'] for task_id in _external_citations(task, project_id)
        )
    return frozenset(cited)


def _external_citations(task: Mapping[str, object], project_id: str) -> Iterator[int]:
    """The ids of ``project_id`` tasks that *task* cites; an entry the scheduler cannot resolve cites nothing."""
    entries = raw_metadata_dict(task.get('metadata'), source=_METADATA_SOURCE).get('external_deps')
    for entry in entries if isinstance(entries, list) else ():
        if not isinstance(entry, str):
            continue
        try:
            dependency = ExternalDep.parse(entry)
            task_id = int(dependency.task_id)
        except ValueError:
            continue
        if normalize_project_id(dependency.project_id) == project_id:
            yield task_id


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
