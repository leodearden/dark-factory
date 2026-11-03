"""The one home of the cancel-with-children contract: shapes, validation and the decision table.

Contract: plans/cancel-with-children-prd.md §4 and §5.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Any, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)
from pydantic_core import ErrorDetails, PydanticCustomError

__all__ = [
    'CALLER_ACTIONS',
    'CHILD_CANCELLING_ACTIONS',
    'ENTRY_ONLY_ACTIONS',
    'REASON_MAX_CHARS',
    'CancelAction',
    'CancelChildrenInvalid',
    'CancelChildrenInvalidReason',
    'CancelChildrenRefusalCode',
    'ChildDisposition',
    'ChildDispositionVariant',
    'ChildKey',
    'ExternalChild',
    'FoldDisposition',
    'KeepDisposition',
    'MootDisposition',
    'RepointDisposition',
    'children_to_json',
    'parse_children',
    'parse_superseded_by',
    'qualified_task_ref',
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


def _distinct(ids: tuple[int, ...]) -> tuple[int, ...]:
    repeated = sorted({task_id for task_id in ids if ids.count(task_id) > 1})
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
_EXTERNAL_KEY: Final = re.compile(r'([^:\s]+):([1-9][0-9]*)', re.ASCII)
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
