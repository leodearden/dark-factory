"""The C4 human-gate escalation as pure functions over task metadata.

plans/completion-driven-triggers-prd.md C4: a pure gate whose
``metadata.trigger_chain`` names it a run's human gate files ``review_due``
with a typed :class:`~escalation.models.ReviewDueState` instead of
``milestone_gate``. This module holds the chain parse, the reading of each
dependency as the completion gate, the payload, its summary and its detail
lines. ``orchestrator.deterministic_runner`` does the reads and composes the
escalation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from escalation.models import ReviewDueState
from pydantic import ValidationError
from shared.task_metadata import TRIGGER_CHAIN_KEY, TriggerChain, TriggerChainRole
from shared.task_read import TaskFound, TaskRead, TaskReadFailed

from orchestrator.predicate_recheck import PREDICATE_VERDICT_KEY

__all__ = [
    'REVIEW_DUE_CATEGORY',
    'CompletionGateReading',
    'CompletionGateUnread',
    'HumanGateFacts',
    'InvalidTriggerChain',
    'ReviewDue',
    'ReviewDueWithheld',
    'build_review_due',
    'completion_gate_reading',
    'first_completion_gate',
    'human_gate_chain',
]

REVIEW_DUE_CATEGORY = 'review_due'


@dataclass(frozen=True)
class InvalidTriggerChain:
    """A ``trigger_chain`` claiming a role that :class:`TriggerChain` refuses."""

    reason: str


@dataclass(frozen=True)
class ReviewDueWithheld:
    """A human gate that cannot file ``review_due``, so it files ``milestone_gate``."""

    reason: str

    def detail_line(self) -> str:
        return f'This human gate files milestone_gate rather than review_due because {self.reason}'


@dataclass(frozen=True)
class CompletionGateUnread:
    """A dependency that may be the completion gate, but could not be read as one."""

    task_id: str
    reason: str


CompletionGateReading: TypeAlias = dict | CompletionGateUnread | None


@dataclass(frozen=True)
class HumanGateFacts:
    """A human gate's validated chain and what its ``review_due`` needs from outside its task.

    ``completion_gate`` is the dependency whose chain role is
    ``completion_gate``, or ``None`` in the interim form, which has none.
    """

    chain: TriggerChain
    project_id: str
    project_root: str
    completion_gate: CompletionGateReading


@dataclass(frozen=True)
class ReviewDue:
    """The ``review_due`` part of a human gate's escalation."""

    payload: ReviewDueState
    summary: str
    detail_lines: tuple[str, ...]


def human_gate_chain(metadata: Mapping[str, object]) -> TriggerChain | InvalidTriggerChain | None:
    """The validated chain of a human gate; ``None`` for any other task."""
    return _chain_claiming(metadata, 'human_gate')


def completion_gate_reading(dep_id: str, read: TaskRead) -> CompletionGateReading:
    """One dependency read: the completion gate itself, possibly it but unread, or not it."""
    if isinstance(read, TaskReadFailed):
        return CompletionGateUnread(task_id=dep_id, reason=read.reason)
    if not isinstance(read, TaskFound):
        return None
    chain = _chain_claiming(read.task.get('metadata'), 'completion_gate')
    if isinstance(chain, InvalidTriggerChain):
        return CompletionGateUnread(task_id=dep_id, reason=f'trigger_chain is invalid: {chain.reason}')
    return read.task if chain is not None else None


def first_completion_gate(readings: Iterable[CompletionGateReading]) -> CompletionGateReading:
    """The first completion gate; else the first unread candidate; else ``None``."""
    first_unread: CompletionGateUnread | None = None
    for reading in readings:
        if isinstance(reading, dict):
            return reading
        if first_unread is None:
            first_unread = reading
    return first_unread


def build_review_due(
    facts: HumanGateFacts,
    *,
    human_gate_task_id: str,
    escalation_id: str,
    fired_at: str,
) -> ReviewDue:
    completion_gate_task_id, verdict, detail_lines = _completion_gate_detail(facts.completion_gate)
    chain = facts.chain
    payload: ReviewDueState = {
        'skill': chain.skill,
        'project_id': facts.project_id,
        'project_root': facts.project_root,
        'run_id': chain.run_id,
        'human_gate_task_id': human_gate_task_id,
        'completion_gate_task_id': completion_gate_task_id,
        'verdict': verdict,
        'fired_at': fired_at,
        'launched_from': escalation_id,
    }
    summary = f'Run /{chain.skill} on {facts.project_id} ({chain.run_id} landed)'
    return ReviewDue(payload=payload, summary=summary, detail_lines=detail_lines)


def _chain_claiming(
    metadata: object, role: TriggerChainRole,
) -> TriggerChain | InvalidTriggerChain | None:
    """*metadata*'s validated ``trigger_chain`` when it claims *role*; ``None`` when it claims none or another."""
    raw = metadata.get(TRIGGER_CHAIN_KEY) if isinstance(metadata, Mapping) else None
    if raw is None or (isinstance(raw, Mapping) and raw.get('role') != role):
        return None
    try:
        return TriggerChain.model_validate(raw)
    except ValidationError as exc:
        return InvalidTriggerChain(_one_line(exc))


def _completion_gate_detail(
    gate: CompletionGateReading,
) -> tuple[str | None, dict[str, object] | None, tuple[str, ...]]:
    """``(completion_gate_task_id, verdict, detail_lines)`` for one completion-gate reading."""
    if gate is None:
        return None, None, ()
    if isinstance(gate, CompletionGateUnread):
        return None, None, (
            f'Completion gate unknown: dependency {gate.task_id} could not be read '
            f'as one ({gate.reason}), so this review_due carries no verdict.',
        )
    gate_id = str(gate.get('id'))
    metadata = gate.get('metadata')
    raw = metadata.get(PREDICATE_VERDICT_KEY) if isinstance(metadata, Mapping) else None
    if isinstance(raw, dict):
        return gate_id, dict(raw), ()
    found = 'none' if raw is None else f'a {type(raw).__name__}'
    return gate_id, None, (
        f'Completion gate {gate_id} has {found} as its {PREDICATE_VERDICT_KEY}, '
        'not a mapping, so this review_due carries no verdict.',
    )


def _one_line(exc: ValidationError) -> str:
    return '; '.join(
        f"{'.'.join(map(str, error['loc']))}: {' '.join(error['msg'].split())}"
        if error['loc'] else ' '.join(error['msg'].split())
        for error in exc.errors()
    )
