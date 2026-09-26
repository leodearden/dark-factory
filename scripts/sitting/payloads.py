"""The apply vocabulary: every store mutation a sitting may make, pre-built as structured data for the agent to execute.

Payloads are data, never formatted command strings. The brief renders them
with ``json.dumps(indent=2)``, and the apply step pastes that data into an MCP
call, so no agent ever parses prose back into arguments. Nothing here executes
anything.

``x_ruling``'s dict shape is P3-6's. The apply payload writes it now so that
task 4764's future server-side stamp can take the write over with no
migration. A live free-string value is superseded visibly through
``ApplyPayload.supersedes``, never silently.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

from orchestrator.session_registry import UNKNOWN_QUEUE, DecisionState
from sitting.inventory import OpenItem

from orchestrator import session_registry

PREPARED_MARKER = 'x_prepared:'
AGREED_MARKER = 'x_agreed:'
X_RULING_KEY = 'x_ruling'
X_RULING_TEXT_CAP = 600

RESOLVE_ACTIONS: tuple[str, ...] = ('resume', 'restart', 'park', 'abandon', 'close_only')
"""Mirrors ``escalation/src/escalation/server.py::RESOLVE_ACTIONS``; the test holds the two equal."""

SESSION_REGISTRY_TOOL = 'cli:session_registry'
FOLLOWUP_SOURCE = 'sitting-preparer'
CLOSING_STATES = frozenset({DecisionState.ANSWERED, DecisionState.DROPPED})


@dataclass(frozen=True)
class ApplyPayload:
    """One mutation: an MCP ``tool`` with its argument dict, or ``cli:session_registry`` with argv lists run in order."""

    tool: str
    args: dict[str, Any] | list[list[str]]
    supersedes: str = ''
    put_to_leo: bool = False


@dataclass(frozen=True)
class Ruling:
    esc_id: str
    action: str
    text: str
    resolved_by: str
    at: str


@dataclass(frozen=True)
class Finding:
    """Something the sitting learned that belongs with another task's owner."""

    escalation_id: str
    summary: str
    detail: str


@dataclass(frozen=True)
class RejectedMarker:
    """A marker line whose payload could not be read; counted by the caller, never raised."""

    marker: str
    line: str
    reason: str


@dataclass(frozen=True)
class PreparedMarker:
    """What the sitting recommended for an item: an option label, or an explicit no-lean with its reason."""

    recommendation: str
    no_lean_reason: str
    sitting_id: str
    prepared_at: str

    def __post_init__(self) -> None:
        _require_str_fields(self)
        if bool(self.recommendation) == bool(self.no_lean_reason):
            raise ValueError('a preparation carries exactly one of a recommendation or a no-lean reason')
        if not self.prepared_at:
            raise ValueError('prepared_at is required')


@dataclass(frozen=True)
class AgreedMarker:
    """Leo's answer and whether it matched the prepared recommendation (None when there was no lean)."""

    answer: str
    agreed: bool | None
    answer_rounds: int
    at: str

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer or not isinstance(self.at, str):
            raise ValueError('an agreed marker needs a non-empty answer and a timestamp')
        if self.agreed is not None and not isinstance(self.agreed, bool):
            raise ValueError(f'agreed must be a bool or None, got {self.agreed!r}')
        _require_positive_int('answer_rounds', self.answer_rounds)

    @classmethod
    def for_answer(cls, prepared: PreparedMarker, answer: str, *, answer_rounds: int, at: str) -> AgreedMarker:
        agreed = None if not prepared.recommendation else prepared.recommendation == answer
        return cls(answer=answer, agreed=agreed, answer_rounds=answer_rounds, at=at)


def resolve_issue_payload(
    escalation_id: str, resolution: str, action: str, *, resolved_by: str, resolution_turns: int,
) -> ApplyPayload:
    if action not in RESOLVE_ACTIONS:
        raise ValueError(f'unknown resolve action {action!r}; expected one of {list(RESOLVE_ACTIONS)}')
    _require_positive_int('resolution_turns', resolution_turns)
    return ApplyPayload('resolve_issue', {
        'escalation_id': escalation_id,
        'resolution': resolution,
        'action': action,
        'resolved_by': resolved_by,
        'resolution_turns': resolution_turns,
    })


def update_task_payload(task_id: str, project_root: str, ruling: Ruling, *, existing: object) -> ApplyPayload:
    """Stamp *ruling* as the task's ``x_ruling``; *existing* is the value it replaces, disclosed verbatim."""
    x_ruling = {**asdict(ruling), 'text': _cap_ruling_text(ruling.text, ruling.esc_id)}
    return ApplyPayload(
        'update_task',
        {'id': task_id, 'project_root': project_root, 'metadata': {X_RULING_KEY: x_ruling}, 'metadata_mode': 'merge'},
        supersedes=_quote(existing) if existing else '',
    )


def append_details_payload(task_id: str, project_root: str, finding: Finding) -> ApplyPayload:
    return ApplyPayload('update_task', {
        'id': task_id,
        'project_root': project_root,
        'details': f'\n\nSitting finding ({finding.escalation_id}): {finding.summary}\n{finding.detail}',
        'append': True,
    })


def add_dependency_payload(task_id: str, depends_on: str, project_root: str) -> ApplyPayload:
    if task_id == depends_on:
        raise ValueError(f'task {task_id} cannot depend on itself')
    return ApplyPayload('add_dependency', {'id': task_id, 'depends_on': depends_on, 'project_root': project_root})


def submit_task_payload(
    project_root: str,
    title: str,
    description: str,
    *,
    metadata: Mapping[str, Any],
    priority: str = 'low',
    put_to_leo: bool = False,
) -> ApplyPayload:
    return ApplyPayload('submit_task', {
        'project_root': project_root,
        'title': title,
        'description': description,
        'priority': priority,
        'metadata': dict(metadata),
    }, put_to_leo=put_to_leo)


def route_finding_to_owner(owner_task_id: str, owner_status: str, finding: Finding, project_root: str) -> ApplyPayload:
    """Append to the owner only while it is ``pending``; any other status files a follow-up that is put to Leo."""
    if owner_status == 'pending':
        return append_details_payload(owner_task_id, project_root, finding)
    return submit_task_payload(
        project_root,
        f'Sitting finding for task {owner_task_id}: {finding.summary}',
        f'Raised on {finding.escalation_id} while task {owner_task_id} was {owner_status or "of unknown status"}, '
        f'so it was not appended there.\n\n{finding.detail}',
        metadata={'source': FOLLOWUP_SOURCE, 'spawned_from': owner_task_id, 'escalation_id': finding.escalation_id},
        put_to_leo=True,
    )


def close_decision_argv(decision_id: str, state: str, evidence: str, *, create: OpenItem | None) -> ApplyPayload:
    """``close-decision`` for *decision_id*, preceded by a ``write-decision`` when *create* names an unfiled L2."""
    if state not in CLOSING_STATES:
        raise ValueError(f'close-decision closes to one of {sorted(CLOSING_STATES)}, not {state!r}')
    if not evidence.strip():
        raise ValueError('close-decision needs the deciding evidence verbatim')
    registry = ['python3', str(Path(session_registry.__file__).resolve())]
    close = [*registry, 'close-decision', '--id', decision_id, '--state', str(state), '--evidence', evidence]
    if create is None:
        return ApplyPayload(SESSION_REGISTRY_TOOL, [close])
    return ApplyPayload(SESSION_REGISTRY_TOOL, [[*registry, *_write_decision_args(decision_id, create)], close])


def render_prepared_marker(marker: PreparedMarker) -> str:
    return _render_marker(PREPARED_MARKER, marker)


def parse_prepared_marker(note: str) -> PreparedMarker | RejectedMarker | None:
    return _parse_last_marker(note, PREPARED_MARKER, PreparedMarker)


def render_agreed_marker(marker: AgreedMarker) -> str:
    return _render_marker(AGREED_MARKER, marker)


def parse_agreed_marker(note: str) -> AgreedMarker | RejectedMarker | None:
    return _parse_last_marker(note, AGREED_MARKER, AgreedMarker)


def append_markers(existing_note: str, *lines: str) -> str:
    """*existing_note* verbatim with each marker line appended; ``stamp_triage`` replaces the whole note."""
    for line in lines:
        if '\n' in line:
            raise ValueError(f'a marker is exactly one line: {line!r}')
    separator = '' if not existing_note or existing_note.endswith('\n') else '\n'
    return existing_note + separator + '\n'.join(lines)


_Marker = TypeVar('_Marker', PreparedMarker, AgreedMarker)


def _render_marker(token: str, marker: PreparedMarker | AgreedMarker) -> str:
    return f'{token} {json.dumps(asdict(marker), sort_keys=True)}'


def _parse_last_marker(note: str, token: str, build: Callable[..., _Marker]) -> _Marker | RejectedMarker | None:
    lines = [line for line in note.splitlines() if line.startswith(token)]
    if not lines:
        return None
    line = lines[-1]
    try:
        payload = json.loads(line[len(token):])
        expected = {f.name for f in fields(build)}  # type: ignore[arg-type]
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError(f'payload keys must be exactly {sorted(expected)}')
        return build(**payload)
    except (TypeError, ValueError) as exc:
        return RejectedMarker(marker=token, line=line, reason=str(exc))


def _write_decision_args(decision_id: str, item: OpenItem) -> list[str]:
    if item.queue_dir in ('', UNKNOWN_QUEUE):
        raise ValueError(f'write-decision needs the item\'s real queue, not {item.queue_dir!r}')
    args = ['write-decision', '--id', decision_id, '--project', item.project, '--text', item.text,
            '--escalations-dir', item.queue_dir]
    if item.task_id:
        args += ['--task-id', item.task_id]
    if item.escalation_id:
        args += ['--escalation-id', item.escalation_id]
    if item.severity:
        args += ['--severity', item.severity]
    return args


def _cap_ruling_text(text: str, esc_id: str) -> str:
    if len(text) <= X_RULING_TEXT_CAP:
        return text
    pointer = f' … [truncated; full ruling on {esc_id}]'
    return text[:X_RULING_TEXT_CAP - len(pointer)] + pointer


def _quote(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _require_positive_int(name: str, value: object) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive int, got {value!r}')


def _require_str_fields(marker: PreparedMarker) -> None:
    for field in fields(marker):
        if not isinstance(getattr(marker, field.name), str):
            raise ValueError(f'{field.name} must be a string')
