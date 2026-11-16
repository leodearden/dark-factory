"""The one home of the cancel-with-children contract: shapes, validation and the decision table.

Contract: plans/cancel-with-children-prd.md §4 and §5.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Self, TypedDict, TypeVar, assert_never, get_args

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from pydantic_core import ErrorDetails, PydanticCustomError

from shared.json_safe import json_safe
from shared.project_id import normalize_project_id
from shared.task_statuses import TaskStatus

__all__ = [
    'CALLER_ACTIONS',
    'CANCEL_CHILDREN_SKILL',
    'CHILD_CANCELLING_ACTIONS',
    'DECISION_TABLE',
    'ENTRY_ONLY_ACTIONS',
    'GET_CANCEL_CHILDREN_TOOL',
    'MISSING_FILES_CAP',
    'PAYLOAD_ID_CAP',
    'REASON_MAX_CHARS',
    'CancelAction',
    'CancelChildrenInvalid',
    'CancelChildrenInvalidReason',
    'CancelChildrenRefusalCode',
    'CancelChildrenView',
    'CanonicalProjectId',
    'CancelDispositionEntry',
    'CancelDispositionEntryVariant',
    'ChildDisposition',
    'ChildDispositionVariant',
    'ChildKey',
    'ChildRelation',
    'ChildRow',
    'CoCancelledEntry',
    'DecisionRow',
    'DispositionSource',
    'ExternalChild',
    'FoldDisposition',
    'FoldEntry',
    'InvalidEntries',
    'KeepDisposition',
    'KeepEntry',
    'MootDisposition',
    'MootEntry',
    'QualifiedTaskRefText',
    'RepointDisposition',
    'RepointEntry',
    'capped',
    'check_targets',
    'children_to_json',
    'co_cancelled_entry',
    'entries_to_json',
    'entry_for_parent',
    'parse_children',
    'parse_entries',
    'parse_superseded_by',
    'qualified_task_ref',
    'render_cancel_children_rules',
    'render_decision_table_markdown',
    'rows_to_json',
    'sibling_lookup_values',
    'stamp_entry',
    'with_entry',
]


class CancelAction(StrEnum):
    KEEP = 'keep'
    REPOINT = 'repoint'
    FOLD = 'fold'
    MOOT = 'moot'
    CO_CANCELLED = 'co_cancelled'


CALLER_ACTIONS: Final = frozenset(
    {CancelAction.KEEP, CancelAction.REPOINT, CancelAction.FOLD, CancelAction.MOOT}
)
ENTRY_ONLY_ACTIONS: Final = frozenset({CancelAction.CO_CANCELLED})
CHILD_CANCELLING_ACTIONS: Final = frozenset({CancelAction.FOLD, CancelAction.MOOT})

REASON_MAX_CHARS: Final = 500
MISSING_FILES_CAP: Final = 20
PAYLOAD_ID_CAP: Final = 50


_Sortable = TypeVar('_Sortable', int, str)


def _repeated(items: Sequence[_Sortable]) -> list[_Sortable]:
    return sorted(item for item, count in Counter(items).items() if count > 1)


def _distinct(ids: tuple[int, ...]) -> tuple[int, ...]:
    repeated = _repeated(ids)
    if repeated:
        raise PydanticCustomError(
            'duplicate_target',
            'names {repeated} more than once',
            {'repeated': repeated[:PAYLOAD_ID_CAP]},
        )
    return ids


TaskId = Annotated[int, Field(strict=True, gt=0)]
ReasonText = Annotated[str, StringConstraints(max_length=REASON_MAX_CHARS)]
DistinctTaskIds = Annotated[tuple[TaskId, ...], AfterValidator(_distinct)]
NonEmptyDistinctTaskIds = Annotated[DistinctTaskIds, Field(min_length=1)]


class _DispositionBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    reason: ReasonText | None = None


class KeepDisposition(_DispositionBase):
    action: Literal[CancelAction.KEEP] = CancelAction.KEEP


class RepointDisposition(_DispositionBase):
    action: Literal[CancelAction.REPOINT] = CancelAction.REPOINT
    onto: NonEmptyDistinctTaskIds


class FoldDisposition(_DispositionBase):
    action: Literal[CancelAction.FOLD] = CancelAction.FOLD
    into: TaskId


class MootDisposition(_DispositionBase):
    action: Literal[CancelAction.MOOT] = CancelAction.MOOT


_DispositionUnion = KeepDisposition | RepointDisposition | FoldDisposition | MootDisposition
ChildDispositionVariant = Annotated[_DispositionUnion, Field(discriminator='action')]


class ChildDisposition(RootModel[ChildDispositionVariant]):
    model_config = ConfigDict(frozen=True)


class CancelChildrenRefusalCode(StrEnum):
    """PRD §4.4's `error` slugs: the wire vocabulary of every cancel-with-children refusal."""

    UNDISPOSITIONED = 'cancel_children_undispositioned'
    INVALID = 'cancel_children_invalid'
    UNKNOWN = 'cancel_children_unknown'
    NESTED = 'cancel_children_nested'
    EXECUTION_FAILED = 'cancel_children_execution_failed'
    PARENT_TERMINAL = 'cancel_children_parent_terminal'
    UNVERIFIED = 'cancel_children_unverified'

    @property
    def error_type(self) -> str:
        return ''.join(word.capitalize() for word in self.value.split('_'))


class CancelChildrenInvalidReason(StrEnum):
    """The `reason_code` of a cancel_children_invalid refusal; the offending field is separate."""

    UNKNOWN_ACTION = 'unknown_action'
    ENTRY_ONLY_ACTION = 'entry_only_action'
    MISSING_FIELD = 'missing_field'
    FIELD_NOT_ALLOWED = 'field_not_allowed'
    TEXT_TOO_LONG = 'text_too_long'
    INVALID_VALUE = 'invalid_value'
    INVALID_CHILD_KEY = 'invalid_child_key'
    CHILD_IS_PARENT = 'child_is_parent'
    CO_CANCELLED_CHILD = 'co_cancelled_child'
    EXTERNAL_NOT_REPOINT = 'external_not_repoint'
    TARGET_IS_PARENT = 'target_is_parent'
    TARGET_IS_CHILD = 'target_is_child'
    DUPLICATE_TARGET = 'duplicate_target'
    TARGET_NOT_FOUND = 'target_not_found'
    TARGET_CANCELLED = 'target_cancelled'
    REPOINT_CYCLE = 'repoint_cycle'
    CSV_NOT_ALLOWED = 'csv_not_allowed'


_INVALID_HINT: Final = (
    'Nothing was written: correct the field named by `field` (see `reason_code`) and re-send '
    'the whole cancel.'
)


class CancelChildrenInvalid(Exception):
    """A children map or superseded_by list the gate refuses, with the reason, field and value."""

    def __init__(self, reason: CancelChildrenInvalidReason, field: str, value: object) -> None:
        super().__init__(f'{CancelChildrenRefusalCode.INVALID.value} {reason.value}: {field}={value!r}')
        self.reason = reason
        self.field = field
        self.value = value

    def to_error_dict(self, *, task_id: str) -> dict[str, Any]:
        code = CancelChildrenRefusalCode.INVALID
        return {
            'success': False,
            'error': code.value,
            'error_type': code.error_type,
            'task_id': task_id,
            'reason_code': self.reason.value,
            'field': self.field,
            'value': json_safe(self.value),
            'hint': _INVALID_HINT,
        }


_PROJECT_ID: Final = r'[^:\s]+'
_PROJECT_ID_TEXT: Final = re.compile(_PROJECT_ID, re.ASCII)
_QUALIFIED_TASK_REF: Final = re.compile(rf'({_PROJECT_ID}):([1-9][0-9]*)', re.ASCII)


def _is_canonical_project_id(text: str) -> bool:
    return _PROJECT_ID_TEXT.fullmatch(text) is not None and normalize_project_id(text) == text


def _canonical_project_id(text: str) -> str:
    if not _is_canonical_project_id(text):
        raise PydanticCustomError(
            'canonical_project_id',
            'expected a project id spelled as normalize_project_id spells it, got {text!r}',
            {'text': text},
        )
    return text


CanonicalProjectId = Annotated[str, AfterValidator(_canonical_project_id)]


def qualified_task_ref(project_id: str, task_id: int) -> str:
    """The one spelling of a task named across projects: `<project_id>:<id>`, canonically."""
    return f'{normalize_project_id(project_id)}:{task_id}'


def _split_qualified_ref(text: str) -> tuple[str, int] | None:
    """The parts of a qualified_task_ref spelling, or None (also for a non-canonical project)."""
    match = _QUALIFIED_TASK_REF.fullmatch(text)
    if match is None or not _is_canonical_project_id(match[1]):
        return None
    return match[1], int(match[2])


def _qualified_ref_text(text: str) -> str:
    if _split_qualified_ref(text) is None:
        raise PydanticCustomError(
            'qualified_task_ref',
            'expected `<project_id>:<id>` with a canonical project id, got {text!r}',
            {'text': text},
        )
    return text


QualifiedTaskRefText = Annotated[str, AfterValidator(_qualified_ref_text)]
_QUALIFIED_REF_ADAPTER: Final = TypeAdapter(QualifiedTaskRefText)


def sibling_lookup_values(parent_id: str) -> tuple[str | int, ...]:
    """Every stored form of a task id, because the metadata lookup is type-strict."""
    if parent_id.isdecimal():
        return (parent_id, int(parent_id))
    return (parent_id,)


@dataclass(frozen=True)
class ExternalChild:
    """A dependent in another project, named by the `external_deps` it carries on the parent."""

    project_id: str
    task_id: int

    def __post_init__(self) -> None:
        if _split_qualified_ref(self.key) != (self.project_id, self.task_id):
            raise ValueError(
                'an external child is a canonical project id and a positive task id, got '
                f'{self.project_id!r} and {self.task_id!r}'
            )

    @property
    def key(self) -> str:
        return qualified_task_ref(self.project_id, self.task_id)


ChildKey = int | ExternalChild

_LOCAL_KEY: Final = re.compile(r'[1-9][0-9]*', re.ASCII)
_DISPOSITION_ADAPTER: Final = TypeAdapter(ChildDispositionVariant)
_TASK_IDS_ADAPTER: Final = TypeAdapter(NonEmptyDistinctTaskIds)

_ERROR_TYPE_REASONS: Final[Mapping[str, CancelChildrenInvalidReason]] = {
    'union_tag_invalid': CancelChildrenInvalidReason.UNKNOWN_ACTION,
    'union_tag_not_found': CancelChildrenInvalidReason.MISSING_FIELD,
    'missing': CancelChildrenInvalidReason.MISSING_FIELD,
    'extra_forbidden': CancelChildrenInvalidReason.FIELD_NOT_ALLOWED,
    'string_too_long': CancelChildrenInvalidReason.TEXT_TOO_LONG,
    **{reason.value: reason for reason in CancelChildrenInvalidReason},
}


def _reason_for(error: ErrorDetails) -> CancelChildrenInvalidReason:
    return _ERROR_TYPE_REASONS.get(error['type'], CancelChildrenInvalidReason.INVALID_VALUE)


def _field_path(*parts: object) -> str:
    return '.'.join(map(str, parts))


def _child_field(key_text: str, *path: object) -> str:
    return _field_path('children', key_text, *path)


def _refusal_value(error: ErrorDetails) -> object:
    """The offending value: a duplicate's repeated ids, None for a missing field, else the input."""
    if error['type'] == 'missing':
        return None
    return error.get('ctx', {}).get('repeated', error['input'])


def _child_key(key_text: object, *, parent_id: int, parent_project_id: str) -> ChildKey:
    if isinstance(key_text, str) and _LOCAL_KEY.fullmatch(key_text):
        child_id = int(key_text)
        if child_id == parent_id:
            raise CancelChildrenInvalid(
                CancelChildrenInvalidReason.CHILD_IS_PARENT, _child_field(key_text), key_text
            )
        return child_id
    external = _split_qualified_ref(key_text) if isinstance(key_text, str) else None
    if external is None or external[0] == normalize_project_id(parent_project_id):
        raise CancelChildrenInvalid(
            CancelChildrenInvalidReason.INVALID_CHILD_KEY, _child_field(str(key_text)), key_text
        )
    return ExternalChild(project_id=external[0], task_id=external[1])


def _disposition_refusal(
    error: ErrorDetails, key_text: str, raw: Mapping[Any, Any]
) -> CancelChildrenInvalid:
    path_below_tag = error['loc'][1:]
    if not path_below_tag:
        return CancelChildrenInvalid(
            _reason_for(error), _child_field(key_text, 'action'), raw.get('action')
        )
    return CancelChildrenInvalid(
        _reason_for(error), _child_field(key_text, *path_below_tag), _refusal_value(error)
    )


def _disposition(key_text: str, raw: object) -> ChildDispositionVariant:
    if not isinstance(raw, Mapping):
        raise CancelChildrenInvalid(
            CancelChildrenInvalidReason.INVALID_VALUE, _child_field(key_text), raw
        )
    action = raw.get('action')
    if isinstance(action, str) and action in ENTRY_ONLY_ACTIONS:
        raise CancelChildrenInvalid(
            CancelChildrenInvalidReason.ENTRY_ONLY_ACTION, _child_field(key_text, 'action'), action
        )
    try:
        return _DISPOSITION_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise _disposition_refusal(exc.errors()[0], key_text, raw) from exc


def _targets(disposition: ChildDispositionVariant) -> tuple[tuple[str, int], ...]:
    match disposition:
        case RepointDisposition():
            return tuple(('onto', target) for target in disposition.onto)
        case FoldDisposition():
            return (('into', disposition.into),)
        case _:
            return ()


def _check_self_reference(
    key_text: str, child: ChildKey, disposition: ChildDispositionVariant, parent_id: int
) -> None:
    if isinstance(child, ExternalChild) and not isinstance(disposition, RepointDisposition):
        raise CancelChildrenInvalid(
            CancelChildrenInvalidReason.EXTERNAL_NOT_REPOINT,
            _child_field(key_text, 'action'),
            disposition.action.value,
        )
    for field, target in _targets(disposition):
        if target == parent_id:
            reason = CancelChildrenInvalidReason.TARGET_IS_PARENT
        elif target == child:
            reason = CancelChildrenInvalidReason.TARGET_IS_CHILD
        else:
            continue
        raise CancelChildrenInvalid(reason, _child_field(key_text, field), target)


def parse_children(
    raw: object, *, parent_id: int, parent_project_id: str
) -> Mapping[ChildKey, ChildDispositionVariant]:
    """Validate a caller's `children` map: shape, keys and self-references, never the store."""
    if not isinstance(raw, Mapping):
        raise CancelChildrenInvalid(CancelChildrenInvalidReason.INVALID_VALUE, 'children', raw)
    parsed: dict[ChildKey, ChildDispositionVariant] = {}
    for key_text, raw_disposition in raw.items():
        child = _child_key(key_text, parent_id=parent_id, parent_project_id=parent_project_id)
        disposition = _disposition(key_text, raw_disposition)
        _check_self_reference(key_text, child, disposition, parent_id)
        parsed[child] = disposition
    return MappingProxyType(parsed)


def _key_text(child: ChildKey) -> str:
    return child.key if isinstance(child, ExternalChild) else str(child)


def children_to_json(
    children: Mapping[ChildKey, ChildDispositionVariant],
) -> dict[str, dict[str, Any]]:
    """The one serialization of a parsed children map back to its wire form."""
    return {
        _key_text(child): ChildDisposition(disposition).model_dump(mode='json')
        for child, disposition in children.items()
    }


def parse_superseded_by(raw: object, *, parent_id: int) -> tuple[int, ...]:
    """Validate a caller's `superseded_by`: one or more distinct task ids, none the parent."""
    if not isinstance(raw, list | tuple):
        raise CancelChildrenInvalid(CancelChildrenInvalidReason.INVALID_VALUE, 'superseded_by', raw)
    try:
        superseders = _TASK_IDS_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        error = exc.errors()[0]
        raise CancelChildrenInvalid(
            _reason_for(error), _field_path('superseded_by', *error['loc']), _refusal_value(error)
        ) from exc
    if parent_id in superseders:
        raise CancelChildrenInvalid(
            CancelChildrenInvalidReason.TARGET_IS_PARENT, 'superseded_by', parent_id
        )
    return superseders


def _child_order(child: ChildKey) -> tuple[bool, int, str]:
    if isinstance(child, ExternalChild):
        return True, 0, child.key
    return False, child, ''


def _reaches(start: int, goal: int, edges: Callable[[int], Sequence[int]]) -> bool:
    frontier = deque([start])
    seen = {start}
    while frontier:
        task = frontier.popleft()
        if task == goal:
            return True
        unseen = [dep for dep in edges(task) if dep not in seen]
        seen.update(unseen)
        frontier.extend(unseen)
    return False


def _check_target(
    field: str,
    target: int,
    target_status: Mapping[int, TaskStatus | None],
    cancelled: frozenset[int],
) -> None:
    status = target_status.get(target)
    if status is None:
        raise CancelChildrenInvalid(CancelChildrenInvalidReason.TARGET_NOT_FOUND, field, target)
    if status == TaskStatus.CANCELLED or target in cancelled:
        raise CancelChildrenInvalid(CancelChildrenInvalidReason.TARGET_CANCELLED, field, target)


def _cancelled_by(
    children: Mapping[ChildKey, ChildDispositionVariant],
    parent_id: int,
    co_cancelled: Collection[int],
) -> frozenset[int]:
    cancelled_children = (
        child
        for child, disposition in children.items()
        if isinstance(child, int) and disposition.action in CHILD_CANCELLING_ACTIONS
    )
    return frozenset({parent_id, *cancelled_children, *co_cancelled})


def _post_cancel_edges(
    children: Mapping[ChildKey, ChildDispositionVariant],
    depends_on: Mapping[int, Sequence[int]],
    *,
    parent_id: int,
    cancelled: frozenset[int],
) -> Callable[[int], Sequence[int]]:
    overrides: dict[int, Sequence[int]] = {task: () for task in cancelled}
    for child, disposition in children.items():
        if isinstance(child, int) and isinstance(disposition, RepointDisposition):
            kept = tuple(dep for dep in depends_on.get(child, ()) if dep != parent_id)
            overrides[child] = kept + disposition.onto
    return lambda task: overrides.get(task, depends_on.get(task, ()))


def _closes_cycle(
    child: ChildKey,
    disposition: ChildDispositionVariant,
    target: int,
    edges: Callable[[int], Sequence[int]],
) -> bool:
    return (
        isinstance(child, int)
        and isinstance(disposition, RepointDisposition)
        and _reaches(target, child, edges)
    )


def _check_child(
    child: ChildKey,
    disposition: ChildDispositionVariant,
    target_status: Mapping[int, TaskStatus | None],
    cancelled: frozenset[int],
    edges: Callable[[int], Sequence[int]],
) -> None:
    for field, target in _targets(disposition):
        child_field = _child_field(_key_text(child), field)
        _check_target(child_field, target, target_status, cancelled)
        if _closes_cycle(child, disposition, target, edges):
            raise CancelChildrenInvalid(
                CancelChildrenInvalidReason.REPOINT_CYCLE, child_field, target
            )


def _check_not_co_cancelled(
    children: Mapping[ChildKey, ChildDispositionVariant], co_cancelled: Collection[int]
) -> None:
    both = sorted(child for child in children if isinstance(child, int) and child in co_cancelled)
    if both:
        raise CancelChildrenInvalid(
            CancelChildrenInvalidReason.CO_CANCELLED_CHILD, _child_field(str(both[0])), both[0]
        )


def check_targets(
    children: Mapping[ChildKey, ChildDispositionVariant],
    superseded_by: Sequence[int],
    *,
    parent_id: int,
    co_cancelled: Collection[int],
    target_status: Mapping[int, TaskStatus | None],
    depends_on: Mapping[int, Sequence[int]],
) -> None:
    """Refuse a child the same act co-cancels, then a target that is missing, cancelled
    (already or by this cancel), or would close a dependency cycle in the graph this cancel
    leaves behind.

    co_cancelled names the tasks the same act cancels separately (PRD §4.1); target_status
    maps each named target to its status, None when no such task exists; depends_on is the
    pre-cancel task -> dependency-ids map, and the post-cancel overlay is derived from
    children, parent_id and co_cancelled. Local children are checked in ascending id order,
    then external ones by key, then superseded_by.
    """
    _check_not_co_cancelled(children, co_cancelled)
    cancelled = _cancelled_by(children, parent_id, co_cancelled)
    edges = _post_cancel_edges(children, depends_on, parent_id=parent_id, cancelled=cancelled)
    for child in sorted(children, key=_child_order):
        _check_child(child, children[child], target_status, cancelled, edges)
    for target in superseded_by:
        _check_target('superseded_by', target, target_status, cancelled)


class DispositionSource(StrEnum):
    CANCELLER = 'canceller'
    BACKSTOP = 'backstop'


class _EntryStamp(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    parent: QualifiedTaskRefText
    stamped_at: AwareDatetime
    stamped_by: str | None = None
    source: DispositionSource


class KeepEntry(_EntryStamp, KeepDisposition):
    pass


class RepointEntry(_EntryStamp, RepointDisposition):
    pass


class FoldEntry(_EntryStamp, FoldDisposition):
    pass


class MootEntry(_EntryStamp, MootDisposition):
    pass


class CoCancelledEntry(_EntryStamp):
    action: Literal[CancelAction.CO_CANCELLED] = CancelAction.CO_CANCELLED
    reason: ReasonText | None = None


_EntryUnion = KeepEntry | RepointEntry | FoldEntry | MootEntry | CoCancelledEntry
CancelDispositionEntryVariant = Annotated[_EntryUnion, Field(discriminator='action')]


class CancelDispositionEntry(RootModel[CancelDispositionEntryVariant]):
    model_config = ConfigDict(frozen=True)


_ENTRIES_ADAPTER: Final = TypeAdapter(list[CancelDispositionEntryVariant])


class _StampFields(TypedDict):
    parent: str
    stamped_at: datetime
    stamped_by: str | None
    source: DispositionSource


def stamp_entry(
    disposition: ChildDispositionVariant,
    *,
    parent: str,
    stamped_at: datetime,
    stamped_by: str | None,
    source: DispositionSource,
) -> CancelDispositionEntryVariant:
    """The one disposition-to-entry translation; parent is a qualified_task_ref."""
    stamp: _StampFields = {
        'parent': parent,
        'stamped_at': stamped_at,
        'stamped_by': stamped_by,
        'source': source,
    }
    match disposition:
        case KeepDisposition():
            return KeepEntry(reason=disposition.reason, **stamp)
        case RepointDisposition():
            return RepointEntry(reason=disposition.reason, onto=disposition.onto, **stamp)
        case FoldDisposition():
            return FoldEntry(reason=disposition.reason, into=disposition.into, **stamp)
        case MootDisposition():
            return MootEntry(reason=disposition.reason, **stamp)
        case _:
            assert_never(disposition)


def co_cancelled_entry(
    *, parent: str, stamped_at: datetime, stamped_by: str | None, source: DispositionSource
) -> CoCancelledEntry:
    return CoCancelledEntry(
        parent=parent, stamped_at=stamped_at, stamped_by=stamped_by, source=source
    )


def entry_for_parent(
    entries: Sequence[CancelDispositionEntryVariant], parent: str
) -> CancelDispositionEntryVariant | None:
    """The entry `parent` stamped, or None; a parent not spelled by qualified_task_ref raises."""
    wanted = _QUALIFIED_REF_ADAPTER.validate_python(parent)
    return next((entry for entry in entries if entry.parent == wanted), None)


def with_entry(
    entries: Sequence[CancelDispositionEntryVariant], entry: CancelDispositionEntryVariant
) -> tuple[CancelDispositionEntryVariant, ...]:
    """Entries with `entry` as its parent's only one: in the first one's place, else appended."""
    position = next(
        (index for index, existing in enumerate(entries) if existing.parent == entry.parent),
        len(entries),
    )
    others = [existing for existing in entries if existing.parent != entry.parent]
    return (*others[:position], entry, *others[position:])


def entries_to_json(entries: Sequence[CancelDispositionEntryVariant]) -> list[dict[str, Any]]:
    """The one serialization of entries into metadata.cancel_dispositions."""
    return [CancelDispositionEntry(entry).model_dump(mode='json') for entry in entries]


@dataclass(frozen=True)
class InvalidEntries:
    """A metadata.cancel_dispositions slice that does not parse; detail says why."""

    detail: str


def parse_entries(raw: object) -> tuple[CancelDispositionEntryVariant, ...] | InvalidEntries:
    """Any metadata.cancel_dispositions slice as its entries, one per parent: () when absent."""
    if raw is None:
        return ()
    if not isinstance(raw, list | tuple):
        return InvalidEntries(f'expected a list of entries, got {type(raw).__name__}')
    items = [item.root if isinstance(item, CancelDispositionEntry) else item for item in raw]
    try:
        entries = tuple(_ENTRIES_ADAPTER.validate_python(items))
    except ValidationError as exc:
        return InvalidEntries(str(exc))
    repeated = _repeated([entry.parent for entry in entries])
    if repeated:
        return InvalidEntries(f'more than one entry for parent {", ".join(repeated)}')
    return entries


class ChildRelation(StrEnum):
    SPAWNED_FROM = 'spawned_from'
    DEPENDENT = 'dependent'
    BOTH = 'both'
    EXTERNAL_DEPENDENT = 'external_dependent'


Count = Annotated[int, Field(strict=True, ge=0)]


class ChildRow(BaseModel):
    """One child's mechanically decidable facts (PRD §4.3), never the decision."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    id: TaskId
    project_id: CanonicalProjectId
    title: str
    status: TaskStatus
    relation: ChildRelation
    declared_files_total: Count
    declared_files_missing_from_main: tuple[str, ...] = Field(max_length=MISSING_FILES_CAP)
    declared_files_missing_from_main_total: Count
    depends_on_superseder: bool
    named_superseder: bool
    has_live_children: bool
    escalation_id: str | None = None
    existing_disposition: CancelDispositionEntryVariant | None = None

    @model_validator(mode='after')
    def _missing_files_fit_their_totals(self) -> Self:
        listed = len(self.declared_files_missing_from_main)
        missing = self.declared_files_missing_from_main_total
        declared = self.declared_files_total
        if not listed <= missing <= declared:
            raise ValueError(
                f'need listed missing files ({listed}) <= declared_files_missing_from_main_total '
                f'({missing}) <= declared_files_total ({declared})'
            )
        return self

    @property
    def key(self) -> str:
        """This child's key in a `children` map."""
        if self.relation == ChildRelation.EXTERNAL_DEPENDENT:
            return qualified_task_ref(self.project_id, self.id)
        return str(self.id)


def rows_to_json(rows: Sequence[ChildRow]) -> list[dict[str, Any]]:
    """The one serialization of child rows into a payload or an escalation record."""
    return [row.model_dump(mode='json') for row in rows]


class CancelChildrenView(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    task_id: TaskId
    project_id: CanonicalProjectId
    status: TaskStatus
    children: tuple[ChildRow, ...]
    superseded_by: DistinctTaskIds = ()
    external_dependents_checked: bool
    enforce: bool

    @model_validator(mode='after')
    def _each_child_listed_once(self) -> Self:
        repeated = _repeated([row.key for row in self.children])
        if repeated:
            raise ValueError(f'children lists {repeated} more than once')
        return self

    @model_validator(mode='after')
    def _only_external_dependents_are_in_another_project(self) -> Self:
        misplaced = [
            row.key
            for row in self.children
            if (row.project_id == self.project_id)
            == (row.relation == ChildRelation.EXTERNAL_DEPENDENT)
        ]
        if misplaced:
            raise ValueError(
                f'children {misplaced} must be in project {self.project_id!r} exactly when '
                f'their relation is not {ChildRelation.EXTERNAL_DEPENDENT.value!r}'
            )
        return self

    @model_validator(mode='after')
    def _the_task_does_not_supersede_itself(self) -> Self:
        if self.task_id in self.superseded_by:
            raise ValueError(f'superseded_by names the task itself, {self.task_id}')
        return self


def capped(field: str, items: Sequence[Any]) -> dict[str, Any]:
    """`field` cut to PAYLOAD_ID_CAP items, plus `<field>_total`: the house payload convention."""
    return {field: list(items[:PAYLOAD_ID_CAP]), f'{field}_total': len(items)}


GET_CANCEL_CHILDREN_TOOL: Final = 'get_cancel_children'
CANCEL_CHILDREN_SKILL: Final = '/cancel-with-children'


@dataclass(frozen=True)
class DecisionRow:
    """One ordered question of PRD §5.2, asked per child; the first yes decides."""

    order: int
    question: str
    evidence_check: str
    action_text: str
    dispositions: tuple[CancelAction, ...]
    report_examples: tuple[str, ...]


DECISION_TABLE: Final = (
    DecisionRow(
        order=0,
        question='Already delivered on main?',
        evidence_check=(
            "child's declared files/behaviour landed under another commit "
            '(`git log -S`/`delivered_checks` green)'
        ),
        action_text=(
            'mark child `done` `found_on_main` BEFORE the cancel '
            '(then it is terminal, unlisted)'
        ),
        dispositions=(),
        report_examples=('3378',),
    ),
    DecisionRow(
        order=1,
        question='Is the child one of the replacements the superseding act names?',
        evidence_check='`named_superseder` (in `superseded_by`)',
        action_text='**keep** (default)',
        dispositions=(CancelAction.KEEP,),
        report_examples=('8032–8058', '928/930'),
    ),
    DecisionRow(
        order=2,
        question='Does it stand on main independent of the parent?',
        evidence_check=(
            '`declared_files_missing_from_main == []` and its description names no '
            'parent-branch artifact; `spawned_from` is provenance only'
        ),
        action_text='**keep** (U)',
        dispositions=(CancelAction.KEEP,),
        report_examples=('6906', '7007', '8207', '733'),
    ),
    DecisionRow(
        order=3,
        question=(
            "Was its premise the parent's DELIVERY (sequencing, \"after N lands\", "
            'a gate clock)?'
        ),
        evidence_check=(
            "relation `dependent`/`both`/`external_dependent`; description references the "
            "parent's output"
        ),
        action_text=(
            '**repoint** onto the superseder(s) (P). This is the class where silence does '
            'harm: locally it releases early, cross-project it wedges'
        ),
        dispositions=(CancelAction.REPOINT,),
        report_examples=('952→1112', '5770→6580', '7738→8891', '6905/7996→8032'),
    ),
    DecisionRow(
        order=4,
        question=(
            "Do its artifacts exist only on the parent's branch or plan, and does a "
            'superseder cover its scope?'
        ),
        evidence_check=(
            "`declared_files_missing_from_main` non-empty; read the superseder's "
            'description/plan for each child item'
        ),
        action_text=(
            "covered → **fold** into it after copying uncovered items into the superseder's "
            '`details` while it is `pending` (F); not covered → **moot** (M)'
        ),
        dispositions=(CancelAction.FOLD, CancelAction.MOOT),
        report_examples=(
            'F: 937/960',
            'F: 7997/7998→8032',
            'F: 5160→5261',
            'M: 6367',
            'M: 771',
            'M: 152',
        ),
    ),
    DecisionRow(
        order=5,
        question='Partly on main?',
        evidence_check='some declared files present, some missing',
        action_text='**keep** + `update_task` trimming the child to what main lacks (R)',
        dispositions=(CancelAction.KEEP,),
        report_examples=('261', '792'),
    ),
    DecisionRow(
        order=6,
        question="Cannot decide (superseder's scope unknown to you)?",
        evidence_check='—',
        action_text=(
            'do NOT cancel the parent yet: ask (watcher → Leo; Stage 2 → recon_report)'
        ),
        dispositions=(),
        report_examples=(),
    ),
)

_TABLE_HEADER: Final = '| # | question | evidence check | action | report example |'
_TABLE_RULE: Final = '|---|---|---|---|---|'


def _table_line(row: DecisionRow) -> str:
    examples = ', '.join(row.report_examples) or '—'
    cells = (str(row.order), row.question, row.evidence_check, row.action_text, examples)
    return f'| {" | ".join(cells)} |'


def render_decision_table_markdown() -> str:
    """PRD §5.2's table, rendered from DECISION_TABLE."""
    return '\n'.join([_TABLE_HEADER, _TABLE_RULE, *map(_table_line, DECISION_TABLE)])


_ACTION_MEANINGS: Final[Mapping[CancelAction, str]] = {
    CancelAction.KEEP: (
        'the child stands on its own; the cancelled parent stops gating it'
    ),
    CancelAction.REPOINT: (
        "the child needed the parent's delivery; its dependency moves onto `onto`"
    ),
    CancelAction.FOLD: (
        "`into` covers the child's scope (copy any uncovered item into it first); "
        'the child is cancelled'
    ),
    CancelAction.MOOT: (
        "the child's premise existed only on the parent's branch or plan and nothing "
        'replaces it; the child is cancelled'
    ),
}


def _action_line(variant: type[_DispositionBase]) -> str:
    action: CancelAction = variant.model_fields['action'].default
    fields = [f'`{name}`' for name in variant.model_fields if name not in {'action', 'reason'}]
    return f'- `{action.value}` ({", ".join([*fields, "`reason`?"])}): {_ACTION_MEANINGS[action]}'


def _quoted_or(actions: frozenset[CancelAction]) -> str:
    return ' or '.join(f'`{action.value}`' for action in sorted(actions))


def render_cancel_children_rules() -> str:
    """The cancel-with-children rules for a prompt: one `###` section composed from this module.

    It uses no `## ` heading, because a prompt region is sliced at the next `## ` heading.
    """
    action_lines = '\n'.join(_action_line(variant) for variant in get_args(_DispositionUnion))
    child_cancelling = _quoted_or(CHILD_CANCELLING_ACTIONS)
    return (
        '### Cancelling a task that has children\n'
        f'Before you cancel any task, call `{GET_CANCEL_CHILDREN_TOOL}` with its id. It lists '
        'every live child: the tasks spawned from it, the tasks depending on it, and the tasks '
        'in other projects depending on it, each with the facts this table needs. Decide each '
        'child with the table, stopping at the first yes:\n\n'
        f'{render_decision_table_markdown()}\n\n'
        "Then cancel with ONE call: `set_task_status(<id>, 'cancelled', children={...}, "
        "superseded_by=[...])`. Key `children` by each listed row's `id`, or by "
        f"`<project_id>:<id>` for a `{ChildRelation.EXTERNAL_DEPENDENT.value}` row, which "
        f'admits only `{CancelAction.REPOINT.value}`. '
        '`superseded_by` names the task(s) replacing the cancelled one; a child it names '
        'defaults to `keep`. Each `children` value is one disposition:\n'
        f'{action_lines}\n\n'
        f'Work bottom-up: a {child_cancelling} child that has live children of its own is '
        'refused, so decide and cancel its children first. You may not '
        f'{child_cancelling} a child whose status is `{TaskStatus.IN_PROGRESS.value}`. A child '
        'you cannot decide means you do NOT cancel the parent: re-scope, or report it through '
        f'the recon_report tools. The `{CANCEL_CHILDREN_SKILL}` skill carries the same table.'
    )
