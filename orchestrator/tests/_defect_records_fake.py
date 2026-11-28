"""A FakeScheduler-side store answering declare_defect and record_defect_outcome.

It folds each event into ``task_data[task_id]['metadata']`` the way
fused-memory/src/fused_memory/backends/defect_records.py::DefectRecords.apply
does: retirement events (a DeclareRetirement, or a Rule whose subject is
retirement) into ``defect_test_retirements`` through apply_retirement_event,
every other event into ``defects`` through apply_event. orchestrator does not
depend on fused-memory, so the fold is built on shared.defects, the single
home of the transition tables, and the fake applies the real rules. Replies
are the real tools' shapes, wrapped as the JSON-RPC body dispatch_tool returns.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from _orch_helpers import mcp_tool_envelope
from pydantic import ValidationError
from shared.defects import (
    DEFECT_TEST_RETIREMENTS_METADATA_KEY,
    DEFECTS_METADATA_KEY,
    DeclareEvent,
    DeclareRetirement,
    DefectDeclaration,
    IllegalDefectTransition,
    OutcomeEvent,
    Retirement,
    Rule,
    RulingSubject,
    apply_event,
    apply_retirement_event,
    parse_declare_event,
    parse_outcome_event,
)

DECLARE_DEFECT = 'declare_defect'
RECORD_DEFECT_OUTCOME = 'record_defect_outcome'

# fused-memory's DefectWriteRefusalCode values the fake can produce.
TRANSITION_REFUSED = 'defect_transition_refused'
EVENT_INVALID = 'defect_event_invalid'
REF_MISMATCH = 'defect_ref_mismatch'

_PARSERS: dict[str, Callable[[object], DeclareEvent | OutcomeEvent]] = {
    DECLARE_DEFECT: parse_declare_event,
    RECORD_DEFECT_OUTCOME: parse_outcome_event,
}


class _Refused(Exception):
    def __init__(self, code: str, detail: str, **fields: object) -> None:
        super().__init__(detail)
        self.code = code
        self.fields = fields


def _is_retirement_event(event: object) -> bool:
    return isinstance(event, DeclareRetirement) or (
        isinstance(event, Rule) and event.ruling.subject is RulingSubject.RETIREMENT
    )


_Record = TypeVar('_Record', DefectDeclaration, Retirement)


def _folded(
    stored: list[Any],
    model: type[_Record],
    event: DeclareEvent | OutcomeEvent,
    apply: Callable[[_Record | None, DeclareEvent | OutcomeEvent], _Record],
) -> list[dict[str, Any]]:
    records: list[_Record] = [model.model_validate(entry) for entry in stored]
    index = next((i for i, record in enumerate(records) if record.ref == event.ref), None)
    try:
        updated = apply(None if index is None else records[index], event)
    except IllegalDefectTransition as exc:
        raise _Refused(
            TRANSITION_REFUSED, str(exc), ref=exc.ref,
            from_status=None if exc.from_status is None else exc.from_status.value,
            event=exc.event.value,
            trigger=None if exc.trigger is None else exc.trigger.value,
        ) from exc
    if index is None:
        records.append(updated)
    else:
        records[index] = updated
    return [record.model_dump(mode='json') for record in records]


class FakeDefectRecords:
    """The two defect-record tools over *task_data*, the FakeScheduler's task store."""

    TOOLS = frozenset(_PARSERS)

    def __init__(self, task_data: dict[str, dict]) -> None:
        self.task_data = task_data
        self.calls: list[tuple[str, dict]] = []

    def handles(self, name: str) -> bool:
        return name in self.TOOLS

    async def dispatch(self, name: str, arguments: dict) -> dict:
        self.calls.append((name, arguments))
        return mcp_tool_envelope(self._reply(name, arguments))

    def _reply(self, name: str, arguments: dict) -> dict[str, Any]:
        task_id = str(arguments.get('task_id'))
        task = self.task_data.get(task_id)
        if task is None:
            return {'error': f'Task {task_id} not found', 'error_type': 'TaskNotFoundError'}
        try:
            metadata = self._applied(name, arguments, dict(task.get('metadata') or {}))
        except _Refused as refusal:
            return {
                'success': False, 'error': refusal.code, 'task_id': task_id,
                **refusal.fields, 'detail': str(refusal),
            }
        self.task_data[task_id] = {**task, 'metadata': metadata}
        return {
            'success': True,
            'id': task_id,
            'message': f'{name} applied to task {task_id}',
            DEFECTS_METADATA_KEY: metadata.get(DEFECTS_METADATA_KEY, []),
            DEFECT_TEST_RETIREMENTS_METADATA_KEY: metadata.get(DEFECT_TEST_RETIREMENTS_METADATA_KEY, []),
        }

    def _applied(self, name: str, arguments: dict, metadata: dict[str, Any]) -> dict[str, Any]:
        try:
            event = _PARSERS[name](arguments.get('event'))
        except ValidationError as exc:
            raise _Refused(EVENT_INVALID, str(exc)) from exc
        named_ref = arguments.get('ref', event.ref)
        if named_ref != event.ref:
            raise _Refused(
                REF_MISMATCH, f'the call names ref {named_ref} but its event concerns {event.ref}',
                ref=named_ref, event_ref=event.ref,
            )
        if _is_retirement_event(event):
            key = DEFECT_TEST_RETIREMENTS_METADATA_KEY
            folded = _folded(metadata.get(key, []), Retirement, event, apply_retirement_event)
        else:
            key = DEFECTS_METADATA_KEY
            folded = _folded(metadata.get(key, []), DefectDeclaration, event, apply_event)
        return {**metadata, key: folded}
