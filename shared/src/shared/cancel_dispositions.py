"""The one home of the cancel-with-children contract: shapes, validation and the decision table.

Contract: plans/cancel-with-children-prd.md §4 and §5.
"""

from __future__ import annotations

import json
import re
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal, Self, TypedDict, TypeVar

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

from shared.task_statuses import TaskStatus

__all__ = [
    'CALLER_ACTIONS',
    'CHILD_CANCELLING_ACTIONS',
    'ENTRY_ONLY_ACTIONS',
    'MISSING_FILES_CAP',
    'PAYLOAD_ID_CAP',
    'REASON_MAX_CHARS',
    'CancelAction',
    'CancelChildrenInvalid',
    'CancelChildrenInvalidReason',
    'CancelChildrenRefusalCode',
    'CancelChildrenView',
    'CancelDispositionEntry',
    'CancelDispositionEntryVariant',
    'ChildDisposition',
    'ChildDispositionVariant',
    'ChildKey',
    'ChildRelation',
    'ChildRow',
    'CoCancelledEntry',
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
    'rows_to_json',
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


_Sortable = TypeVar('_Sortable', int, str)


def _repeated(items: Sequence[_Sortable]) -> list[_Sortable]:
    return sorted(item for item, count in Counter(items).items() if count > 1)


def _distinct(ids: tuple[int, ...]) -> tuple[int, ...]:
    repeated = _repeated(ids)
    if repeated:
        raise PydanticCustomError(
            'duplicate_target', 'names {repeated} more than once', {'repeated': repeated}
        )
    return ids


TaskId = Annotated[int, Field(strict=True, gt=0)]
ReasonText = Annotated[str, StringConstraints(max_length=REASON_MAX_CHARS)]
DistinctTaskIds = Annotated[tuple[TaskId, ...], Field(min_length=1), AfterValidator(_distinct)]


class _DispositionBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra='forbid')

    reason: ReasonText | None = None


class KeepDisposition(_DispositionBase):
    action: Literal[CancelAction.KEEP] = CancelAction.KEEP


class RepointDisposition(_DispositionBase):
    action: Literal[CancelAction.REPOINT] = CancelAction.REPOINT
    onto: DistinctTaskIds


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


def _json_safe(value: object) -> object:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


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
            'value': _json_safe(self.value),
            'hint': _INVALID_HINT,
        }


_QUALIFIED_TASK_REF: Final = r'([^:\s]+):([1-9][0-9]*)'
QualifiedTaskRefText = Annotated[str, StringConstraints(pattern=f'^{_QUALIFIED_TASK_REF}$')]


def qualified_task_ref(project_id: str, task_id: int) -> str:
    """The one spelling of a task named across projects: `<project_id>:<id>`."""
    return f'{project_id}:{task_id}'


@dataclass(frozen=True)
class ExternalChild:
    """A dependent in another project, named by the `external_deps` it carries on the parent."""

    project_id: str
    task_id: int

    @property
    def key(self) -> str:
        return qualified_task_ref(self.project_id, self.task_id)


ChildKey = int | ExternalChild

_LOCAL_KEY: Final = re.compile(r'[1-9][0-9]*', re.ASCII)
_EXTERNAL_KEY: Final = re.compile(_QUALIFIED_TASK_REF, re.ASCII)
_DISPOSITION_ADAPTER: Final = TypeAdapter(ChildDispositionVariant)
_TASK_IDS_ADAPTER: Final = TypeAdapter(DistinctTaskIds)

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


def _child_field(key_text: str, *path: object) -> str:
    return '.'.join(['children', key_text, *map(str, path)])


def _child_key(key_text: object, *, parent_id: int, parent_project_id: str) -> ChildKey:
    if isinstance(key_text, str) and _LOCAL_KEY.fullmatch(key_text):
        child_id = int(key_text)
        if child_id == parent_id:
            raise CancelChildrenInvalid(
                CancelChildrenInvalidReason.CHILD_IS_PARENT, _child_field(key_text), key_text
            )
        return child_id
    external = _EXTERNAL_KEY.fullmatch(key_text) if isinstance(key_text, str) else None
    if external is None or external[1] == parent_project_id:
        raise CancelChildrenInvalid(
            CancelChildrenInvalidReason.INVALID_CHILD_KEY, _child_field(str(key_text)), key_text
        )
    return ExternalChild(project_id=external[1], task_id=int(external[2]))


def _disposition_refusal(
    error: ErrorDetails, key_text: str, raw: Mapping[Any, Any]
) -> CancelChildrenInvalid:
    path_below_tag = error['loc'][1:]
    if not path_below_tag:
        return CancelChildrenInvalid(
            _reason_for(error), _child_field(key_text, 'action'), raw.get('action')
        )
    value = None if error['type'] == 'missing' else error['input']
    return CancelChildrenInvalid(_reason_for(error), _child_field(key_text, *path_below_tag), value)


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
        raise CancelChildrenInvalid(_reason_for(exc.errors()[0]), 'superseded_by', raw) from exc
    if parent_id in superseders:
        raise CancelChildrenInvalid(
            CancelChildrenInvalidReason.TARGET_IS_PARENT, 'superseded_by', parent_id
        )
    return superseders


def _child_order(child: ChildKey) -> tuple[bool, int, str]:
    if isinstance(child, ExternalChild):
        return True, 0, child.key
    return False, child, ''


def _reaches(start: int, goal: int, depends_on: Mapping[int, Sequence[int]]) -> bool:
    frontier = deque([start])
    seen = {start}
    while frontier:
        task = frontier.popleft()
        if task == goal:
            return True
        unseen = [dep for dep in depends_on.get(task, ()) if dep not in seen]
        seen.update(unseen)
        frontier.extend(unseen)
    return False


def _check_target(
    field: str, target: int, target_status: Mapping[int, TaskStatus | None]
) -> None:
    status = target_status.get(target)
    if status is None:
        raise CancelChildrenInvalid(CancelChildrenInvalidReason.TARGET_NOT_FOUND, field, target)
    if status == TaskStatus.CANCELLED:
        raise CancelChildrenInvalid(CancelChildrenInvalidReason.TARGET_CANCELLED, field, target)


def _closes_cycle(
    child: ChildKey,
    disposition: ChildDispositionVariant,
    target: int,
    depends_on: Mapping[int, Sequence[int]],
) -> bool:
    return (
        isinstance(child, int)
        and isinstance(disposition, RepointDisposition)
        and _reaches(target, child, depends_on)
    )


def _check_child(
    child: ChildKey,
    disposition: ChildDispositionVariant,
    target_status: Mapping[int, TaskStatus | None],
    depends_on: Mapping[int, Sequence[int]],
) -> None:
    for field, target in _targets(disposition):
        child_field = _child_field(_key_text(child), field)
        _check_target(child_field, target, target_status)
        if _closes_cycle(child, disposition, target, depends_on):
            raise CancelChildrenInvalid(
                CancelChildrenInvalidReason.REPOINT_CYCLE, child_field, target
            )


def check_targets(
    children: Mapping[ChildKey, ChildDispositionVariant],
    superseded_by: Sequence[int],
    *,
    target_status: Mapping[int, TaskStatus | None],
    depends_on: Mapping[int, Sequence[int]],
) -> None:
    """Refuse a target that is missing, cancelled, or would close a local dependency cycle.

    target_status maps each named target to its status, None when no such task exists;
    depends_on is the project's task -> dependency-ids map. Local children are checked in
    ascending id order, then external ones by key, then superseded_by.
    """
    for child in sorted(children, key=_child_order):
        _check_child(child, children[child], target_status, depends_on)
    for target in superseded_by:
        _check_target('superseded_by', target, target_status)


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


def co_cancelled_entry(
    *, parent: str, stamped_at: datetime, stamped_by: str | None, source: DispositionSource
) -> CoCancelledEntry:
    return CoCancelledEntry(
        parent=parent, stamped_at=stamped_at, stamped_by=stamped_by, source=source
    )


def entry_for_parent(
    entries: Sequence[CancelDispositionEntryVariant], parent: str
) -> CancelDispositionEntryVariant | None:
    return next((entry for entry in entries if entry.parent == parent), None)


def with_entry(
    entries: Sequence[CancelDispositionEntryVariant], entry: CancelDispositionEntryVariant
) -> tuple[CancelDispositionEntryVariant, ...]:
    """Entries with `entry` replacing the one for its parent in place, else appended."""
    if entry_for_parent(entries, entry.parent) is None:
        return (*entries, entry)
    return tuple(entry if existing.parent == entry.parent else existing for existing in entries)


def entries_to_json(entries: Sequence[CancelDispositionEntryVariant]) -> list[dict[str, Any]]:
    """The one serialization of entries into metadata.cancel_dispositions."""
    return [CancelDispositionEntry(entry).model_dump(mode='json') for entry in entries]


@dataclass(frozen=True)
class InvalidEntries:
    """A metadata.cancel_dispositions slice that does not parse; detail says why."""

    detail: str


def parse_entries(raw: object) -> tuple[CancelDispositionEntryVariant, ...] | InvalidEntries:
    """Any metadata.cancel_dispositions slice as its entries: () when absent."""
    if raw is None:
        return ()
    if not isinstance(raw, list | tuple):
        return InvalidEntries(f'expected a list of entries, got {type(raw).__name__}')
    items = [item.root if isinstance(item, CancelDispositionEntry) else item for item in raw]
    try:
        return tuple(_ENTRIES_ADAPTER.validate_python(items))
    except ValidationError as exc:
        return InvalidEntries(str(exc))


class ChildRelation(StrEnum):
    SPAWNED_FROM = 'spawned_from'
    DEPENDENT = 'dependent'
    BOTH = 'both'
    EXTERNAL_DEPENDENT = 'external_dependent'


MISSING_FILES_CAP: Final = 20
PAYLOAD_ID_CAP: Final = 50

Count = Annotated[int, Field(strict=True, ge=0)]


class ChildRow(BaseModel):
    """One child's mechanically decidable facts (PRD §4.3), never the decision."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    id: TaskId
    project_id: str
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
    project_id: str
    status: TaskStatus
    children: tuple[ChildRow, ...]
    superseded_by: tuple[TaskId, ...] = ()
    external_dependents_checked: bool
    enforce: bool

    @model_validator(mode='after')
    def _each_child_listed_once(self) -> Self:
        repeated = _repeated([row.key for row in self.children])
        if repeated:
            raise ValueError(f'children lists {repeated} more than once')
        return self


def capped(field: str, items: Sequence[Any]) -> dict[str, Any]:
    """`field` cut to PAYLOAD_ID_CAP items, plus `<field>_total`: the house payload convention."""
    return {field: list(items[:PAYLOAD_ID_CAP]), f'{field}_total': len(items)}
