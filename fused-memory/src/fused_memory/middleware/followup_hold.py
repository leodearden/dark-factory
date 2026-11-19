"""The follow-up holder: which created tasks wait behind it, and what it refuses (plans/follow-up-filing-shape-prd.md §4.3).

curator_types.py imports this module, so it must keep importing only the stdlib and lower layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, ClassVar, Final

from shared.followup_hold_records import (
    FOLLOWUP_HOLD_METADATA_KEY,
    FOLLOWUP_HOLDER_METADATA_KEY,
    FollowupHold,
)
from shared.task_metadata_wire import coerce_task_metadata
from shared.task_statuses import TaskStatus

from fused_memory.backends.task_backend_errors import TASKMASTER_TOOL_ERROR, TaskmasterError
from fused_memory.middleware.metadata_dict import raw_metadata_dict

__all__ = [
    'HELD_PRIORITIES',
    'PROTECTED_STATUSES',
    'FollowupHolderProtectedError',
    'HolderProtection',
    'hold_at_create',
    'hold_scope',
    'is_holder',
    'protected_status',
    'protected_update',
]

HELD_PRIORITIES: Final = frozenset({'low', 'polish'})
PROTECTED_STATUSES: Final = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED, TaskStatus.DEFERRED})

_FOLLOWUP_SOURCE: Final = 'agent-followup'
_REVIEW_SUGGESTIONS_CONTEXT: Final = 'review_suggestions'
_HOLD_EXCLUDING_KEYS: Final = ('execution_class', 'prd_path', 'x_finding_key', 'human_decomposed')
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
) -> tuple[dict[str, object], dict[str, object]]:
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
