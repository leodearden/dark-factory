"""The write rules for a task's ``defects`` and ``defect_test_retirements`` records.

``plans/defect-tests-prd.md`` decision 7 and invariants I1 and I2 state the
rules; ``shared/src/shared/defects.py`` is the single home of their
vocabulary and transition tables, and this module only routes writes through
them: the creation floor, the per-ref event fold, and the wire parse. Pure:
no IO. Every refusal is a :class:`DefectWriteRefused` whose
:meth:`~DefectWriteRefused.to_error_dict` is its wire shape.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, NamedTuple, TypeVar, get_args

from pydantic import BaseModel, ValidationError
from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
    Declare,
    DeclareClosure,
    DeclareEvent,
    DeclareRetirement,
    DefectDeclaration,
    DuplicateDefectRef,
    IllegalDefectTransition,
    OutcomeEvent,
    Retirement,
    RetirementCode,
    Rule,
    RulingSubject,
    Withdrawal,
    WithdrawalCode,
    apply_event,
    apply_retirement_event,
    parse_declare_event,
    parse_outcome_event,
    require_unique_refs,
)

from fused_memory.backends.task_backend_errors import TASKMASTER_TOOL_ERROR, TaskmasterError

__all__ = [
    'DEFECT_RECORD_KEYS',
    'DefectCreationRefused',
    'DefectEventInvalid',
    'DefectRecords',
    'DefectRecordsChange',
    'DefectRefMismatch',
    'DefectRefNotLive',
    'DefectRefsNotUnique',
    'DefectTaskTerminal',
    'DefectTransitionRefused',
    'DefectWriteRefusalCode',
    'DefectWriteRefused',
    'LiveRefRequirement',
    'StoredDefectRecordsMalformed',
    'parse_declare_wire',
    'parse_outcome_wire',
    'refs_required_live',
    'require_creation_floor',
]

DEFECT_RECORD_KEYS: tuple[str, str] = (DEFECTS_METADATA_KEY, DEFECT_TEST_RETIREMENTS_METADATA_KEY)
"""The two metadata keys only the defect writers may move after creation."""


# --- Refusals ------------------------------------------------------------------


class DefectWriteRefusalCode(StrEnum):
    EVENT_INVALID = 'defect_event_invalid'
    REF_MISMATCH = 'defect_ref_mismatch'
    TRANSITION_REFUSED = 'defect_transition_refused'
    REFS_NOT_UNIQUE = 'defect_refs_not_unique'
    STORED_RECORDS_MALFORMED = 'stored_defect_records_malformed'
    TASK_TERMINAL = 'defect_write_to_terminal_task'
    REF_NOT_LIVE = 'defect_ref_not_live'
    CREATION_REFUSED = 'defect_creation_refused'


def _wire_value(value: StrEnum | None) -> str | None:
    return None if value is None else value.value


class DefectWriteRefused(TaskmasterError):
    """A defect-record write the rules refuse; nothing was written.

    ``fields`` is the refusal's structured data; :meth:`to_error_dict` renders
    it with the refusal code and the human-readable detail.
    """

    refusal: ClassVar[DefectWriteRefusalCode]

    def __init__(self, detail: str, **fields: object) -> None:
        super().__init__(TASKMASTER_TOOL_ERROR, detail)
        self.fields: Mapping[str, object] = MappingProxyType(fields)

    def to_error_dict(self, *, task_id: str | None = None) -> dict[str, Any]:
        return {
            'success': False,
            'error': self.refusal.value,
            'task_id': task_id,
            **self.fields,
            'detail': self.message,
        }


class DefectEventInvalid(DefectWriteRefused):
    refusal = DefectWriteRefusalCode.EVENT_INVALID

    def __init__(self, detail: str, *, expected_events: Sequence[str], **fields: object) -> None:
        super().__init__(detail, expected_events=list(expected_events), **fields)


class DefectRefMismatch(DefectEventInvalid):
    refusal = DefectWriteRefusalCode.REF_MISMATCH

    def __init__(self, *, ref: str, event_ref: str, expected_events: Sequence[str]) -> None:
        super().__init__(
            f'the call names ref {ref} but its event concerns {event_ref}',
            expected_events=expected_events,
            ref=ref,
            event_ref=event_ref,
        )


class DefectTransitionRefused(DefectWriteRefused):
    refusal = DefectWriteRefusalCode.TRANSITION_REFUSED

    def __init__(self, illegal: IllegalDefectTransition) -> None:
        super().__init__(
            str(illegal),
            ref=illegal.ref,
            from_status=_wire_value(illegal.from_status),
            event=illegal.event.value,
            trigger=_wire_value(illegal.trigger),
        )


class DefectRefsNotUnique(DefectWriteRefused):
    refusal = DefectWriteRefusalCode.REFS_NOT_UNIQUE

    def __init__(self, key: str, duplicate: DuplicateDefectRef) -> None:
        super().__init__(f'metadata.{key}: {duplicate}', key=key, refs=list(duplicate.refs))


class StoredDefectRecordsMalformed(DefectWriteRefused):
    """The stored records cannot be read; ``key`` is None when the whole blob is not an object."""

    refusal = DefectWriteRefusalCode.STORED_RECORDS_MALFORMED

    def __init__(self, key: str | None, index: int | None, detail: str) -> None:
        stored = 'metadata' if key is None else f'metadata.{key}'
        super().__init__(
            f'the stored {stored} cannot be read as defect records, so no event '
            f'can be applied to it: {detail}',
            key=key,
            index=index,
        )


class DefectTaskTerminal(DefectWriteRefused):
    refusal = DefectWriteRefusalCode.TASK_TERMINAL

    def __init__(self, task_id: str, status: str) -> None:
        super().__init__(
            f'task {task_id} is {status}, a terminal status; its defect records no longer change',
            status=status,
        )


class DefectRefNotLive(DefectWriteRefused):
    refusal = DefectWriteRefusalCode.REF_NOT_LIVE

    def __init__(self, ref: str, relation: str, *, unreadable_task_ids: Sequence[int] = ()) -> None:
        unreadable = f' (tasks {list(unreadable_task_ids)} could not be read)' if unreadable_task_ids else ''
        super().__init__(
            f'{relation} must name a live ref, one some task declares; no readable task '
            f'declares {ref}{unreadable}',
            ref=ref,
            relation=relation,
            unreadable_task_ids=list(unreadable_task_ids),
        )


class DefectCreationRefused(DefectWriteRefused):
    """``detail`` carries the ``shared.defects`` validator's own words, which name the allowed values."""

    refusal = DefectWriteRefusalCode.CREATION_REFUSED

    def __init__(self, key: str, index: int | None, detail: str) -> None:
        super().__init__(
            f'a creating write may carry metadata.{key} only as fresh records: {detail}',
            key=key,
            index=index,
        )


# --- The event fold ------------------------------------------------------------

_Record = TypeVar('_Record', DefectDeclaration, Retirement)
_AnyEvent = DeclareEvent | OutcomeEvent


def _is_retirement_event(event: _AnyEvent) -> bool:
    return isinstance(event, DeclareRetirement) or (
        isinstance(event, Rule) and event.ruling.subject is RulingSubject.RETIREMENT
    )


def _require_unique(key: str, records: Iterable[DefectDeclaration | Retirement]) -> None:
    try:
        require_unique_refs(records)
    except DuplicateDefectRef as exc:
        raise DefectRefsNotUnique(key, exc) from exc


def _stored_records(
    metadata: Mapping[str, object], key: str, model: type[_Record]
) -> tuple[_Record, ...]:
    if key not in metadata:
        return ()
    raw = metadata[key]
    if not isinstance(raw, list):
        raise StoredDefectRecordsMalformed(key, None, f'a {type(raw).__name__}, not a list')
    records: list[_Record] = []
    for index, entry in enumerate(raw):
        try:
            records.append(model.model_validate(entry))
        except ValidationError as exc:
            raise StoredDefectRecordsMalformed(key, index, str(exc)) from exc
    _require_unique(key, records)
    return tuple(records)


def _folded(
    records: tuple[_Record, ...],
    event: _AnyEvent,
    apply: Callable[[_Record | None, _AnyEvent], _Record],
) -> tuple[_Record, ...]:
    index = next((i for i, record in enumerate(records) if record.ref == event.ref), None)
    try:
        updated = apply(None if index is None else records[index], event)
    except IllegalDefectTransition as exc:
        raise DefectTransitionRefused(exc) from exc
    if index is None:
        return (*records, updated)
    return (*records[:index], updated, *records[index + 1 :])


def _json_list(records: Iterable[BaseModel]) -> list[dict[str, Any]]:
    return [record.model_dump(mode='json') for record in records]


@dataclass(frozen=True)
class DefectRecords:
    """One task's defect records: its ``defects`` entries and its ``defect_test_retirements``."""

    defects: tuple[DefectDeclaration, ...] = ()
    retirements: tuple[Retirement, ...] = ()

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> DefectRecords:
        """Read both keys from a stored metadata object; an absent key is no records.

        Raises:
            StoredDefectRecordsMalformed: a key holds a non-list or an entry
                its model refuses.
            DefectRefsNotUnique: a key's list repeats a ref.
        """
        return cls(
            defects=_stored_records(metadata, DEFECTS_METADATA_KEY, DefectDeclaration),
            retirements=_stored_records(metadata, DEFECT_TEST_RETIREMENTS_METADATA_KEY, Retirement),
        )

    @classmethod
    def from_stored_blob(cls, stored_raw: str | None) -> DefectRecords:
        """:meth:`from_metadata` over a task row's stored metadata text; an absent blob is no records.

        Raises:
            StoredDefectRecordsMalformed: with ``key`` None when the blob is
                not a JSON object, else as :meth:`from_metadata`.
            DefectRefsNotUnique: as :meth:`from_metadata`.
        """
        try:
            stored = json.loads(stored_raw) if stored_raw else {}
        except ValueError:
            stored = None
        if not isinstance(stored, dict):
            raise StoredDefectRecordsMalformed(
                None, None, f'it is not a JSON object (stored {(stored_raw or "")[:80]!r})'
            )
        return cls.from_metadata(stored)

    def apply(self, events: Iterable[_AnyEvent]) -> DefectRecordsChange:
        """The records after *events*, in order, each folded into the record its ref names.

        Retirement events go through ``apply_retirement_event``, every other
        event through ``apply_event``; any refusal refuses the whole batch.

        Raises:
            DefectTransitionRefused: an event the transition tables refuse.
            DefectRefsNotUnique: a result repeats a ref.
        """
        defects, retirements = self.defects, self.retirements
        addressed: set[str] = set()
        for event in events:
            if _is_retirement_event(event):
                retirements = _folded(retirements, event, apply_retirement_event)
                addressed.add(DEFECT_TEST_RETIREMENTS_METADATA_KEY)
            else:
                defects = _folded(defects, event, apply_event)
                addressed.add(DEFECTS_METADATA_KEY)
        _require_unique(DEFECTS_METADATA_KEY, defects)
        _require_unique(DEFECT_TEST_RETIREMENTS_METADATA_KEY, retirements)
        return DefectRecordsChange(DefectRecords(defects, retirements), frozenset(addressed))

    def as_metadata(self) -> dict[str, list[dict[str, Any]]]:
        """Both keys as the JSON lists a metadata blob stores."""
        return {
            DEFECTS_METADATA_KEY: _json_list(self.defects),
            DEFECT_TEST_RETIREMENTS_METADATA_KEY: _json_list(self.retirements),
        }


@dataclass(frozen=True)
class DefectRecordsChange:
    """The records after a batch of events, and the metadata keys the batch addressed."""

    records: DefectRecords
    addressed_keys: frozenset[str]

    def changed_metadata(self) -> dict[str, list[dict[str, Any]]]:
        """Only the addressed keys, as the JSON lists to merge into the stored blob."""
        full = self.records.as_metadata()
        return {key: full[key] for key in DEFECT_RECORD_KEYS if key in self.addressed_keys}


class LiveRefRequirement(NamedTuple):
    """A ref an event names that some task must already declare, and the relation naming it."""

    ref: str
    relation: WithdrawalCode | RetirementCode


def _live_ref_requirement(event: _AnyEvent) -> LiveRefRequirement | None:
    if isinstance(event, DeclareClosure) and isinstance(event.closure, Withdrawal):
        reason = event.closure.reason
        if reason.reason is WithdrawalCode.DUPLICATE_OF and reason.ref is not None:
            return LiveRefRequirement(reason.ref, reason.reason)
    if isinstance(event, DeclareRetirement):
        reason = event.retirement.reason
        if reason.reason is RetirementCode.SUPERSEDED_BY and reason.ref is not None:
            return LiveRefRequirement(reason.ref, reason.reason)
    return None


def refs_required_live(events: Iterable[_AnyEvent]) -> tuple[LiveRefRequirement, ...]:
    """Each ``duplicate_of`` and ``superseded_by`` ref *events* name (PRD decision 13)."""
    return tuple(
        requirement
        for requirement in map(_live_ref_requirement, events)
        if requirement is not None
    )


# --- The creation floor --------------------------------------------------------

def _creation_event(key: str, entry: object) -> Declare | DeclareRetirement:
    if key == DEFECTS_METADATA_KEY:
        return Declare.model_validate({'declaration': entry})
    return DeclareRetirement.model_validate({'retirement': entry})


def _require_fresh_list(key: str, raw: object) -> None:
    if not isinstance(raw, list):
        raise DefectCreationRefused(key, None, f'a {type(raw).__name__}, not a list')
    records: list[DefectDeclaration | Retirement] = []
    for index, entry in enumerate(raw):
        try:
            event = _creation_event(key, entry)
        except ValidationError as exc:
            raise DefectCreationRefused(key, index, str(exc)) from exc
        records.append(event.declaration if isinstance(event, Declare) else event.retirement)
    try:
        require_unique_refs(records)
    except DuplicateDefectRef as exc:
        raise DefectCreationRefused(key, None, str(exc)) from exc


def require_creation_floor(metadata: Mapping[str, object] | None) -> None:
    """Refuse a creating write whose defect keys carry anything but fresh records.

    A ``defects`` entry must be what a ``Declare`` event would declare (open,
    one sighting, no proof, closure or ruling); a ``defect_test_retirements``
    entry what a ``DeclareRetirement`` would (unruled). Each list holds a ref
    at most once.

    Raises:
        DefectCreationRefused: naming the key and the offending index (None
            for a non-list or a repeated ref); its detail names the allowed
            values.
    """
    if metadata is None:
        return
    for key in DEFECT_RECORD_KEYS:
        if key in metadata:
            _require_fresh_list(key, metadata[key])


# --- Wire parsing --------------------------------------------------------------


def _event_tags(union: object) -> tuple[str, ...]:
    members = get_args(get_args(union)[0])
    return tuple(member.model_fields['event'].default.value for member in members)


_DECLARE_EVENT_TAGS = _event_tags(DeclareEvent)
_OUTCOME_EVENT_TAGS = _event_tags(OutcomeEvent)


def _invalid(tags: tuple[str, ...], exc: ValidationError) -> DefectEventInvalid:
    return DefectEventInvalid(
        f'the event must be one of {", ".join(tags)} and validate as that event: {exc}',
        expected_events=tags,
    )


def parse_declare_wire(raw: object) -> DeclareEvent:
    """A ``declare_defect`` wire event, parsed; raises :class:`DefectEventInvalid`."""
    try:
        return parse_declare_event(raw)
    except ValidationError as exc:
        raise _invalid(_DECLARE_EVENT_TAGS, exc) from exc


def parse_outcome_wire(raw: object, *, ref: str) -> OutcomeEvent:
    """A ``record_defect_outcome`` wire event for *ref*, parsed.

    Raises:
        DefectEventInvalid: the event does not validate as an outcome event.
        DefectRefMismatch: it concerns a ref other than *ref*.
    """
    try:
        event = parse_outcome_event(raw)
    except ValidationError as exc:
        raise _invalid(_OUTCOME_EVENT_TAGS, exc) from exc
    if event.ref != ref:
        raise DefectRefMismatch(ref=ref, event_ref=event.ref, expected_events=_OUTCOME_EVENT_TAGS)
    return event
