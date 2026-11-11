"""The C4 human-gate escalation as pure functions over task metadata.

plans/completion-driven-triggers-prd.md C4: a pure gate whose
``metadata.trigger_chain`` names it a run's human gate files ``review_due``
with a typed :class:`~escalation.models.ReviewDueState` instead of
``milestone_gate``. This module holds the chain parse, the payload, its
summary and the completion-gate detail lines.
``orchestrator.deterministic_runner`` reads the completion gate and composes
the escalation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from escalation.models import ReviewDueState
from pydantic import ValidationError
from shared.task_metadata import TRIGGER_CHAIN_KEY, TriggerChain

from orchestrator.predicate_recheck import PREDICATE_VERDICT_KEY

__all__ = [
    'REVIEW_DUE_CATEGORY',
    'CompletionGateUnread',
    'HumanGateFacts',
    'InvalidTriggerChain',
    'ReviewDue',
    'human_gate_chain',
    'is_completion_gate',
    'review_due',
]

REVIEW_DUE_CATEGORY = 'review_due'

_HUMAN_GATE_ROLE = 'human_gate'
_COMPLETION_GATE_ROLE = 'completion_gate'


@dataclass(frozen=True)
class InvalidTriggerChain:
    """A human gate's ``trigger_chain`` that :class:`TriggerChain` refuses."""

    reason: str

    def detail_line(self) -> str:
        return (
            'trigger_chain is invalid, so this gate files milestone_gate rather '
            f'than review_due: {self.reason}'
        )


@dataclass(frozen=True)
class CompletionGateUnread:
    """A dependency whose read failed, so whether it is the completion gate is unknown."""

    task_id: str
    reason: str


@dataclass(frozen=True)
class HumanGateFacts:
    """What a human gate's escalation needs from outside its own task.

    ``completion_gate`` is the dependency whose chain role is
    ``completion_gate``, or ``None`` in the interim form, which has none.
    """

    project_id: str
    project_root: str
    completion_gate: dict | CompletionGateUnread | None


@dataclass(frozen=True)
class ReviewDue:
    """The ``review_due`` part of a human gate's escalation."""

    payload: ReviewDueState
    summary: str
    detail_lines: tuple[str, ...]


def human_gate_chain(metadata: Mapping[str, object]) -> TriggerChain | InvalidTriggerChain | None:
    """The validated chain of a human gate; ``None`` for any other task."""
    raw = metadata.get(TRIGGER_CHAIN_KEY)
    if raw is None or (isinstance(raw, Mapping) and raw.get('role') != _HUMAN_GATE_ROLE):
        return None
    try:
        return TriggerChain.model_validate(raw)
    except ValidationError as exc:
        return InvalidTriggerChain(_one_line(exc))


def is_completion_gate(task: Mapping[str, object]) -> bool:
    metadata = task.get('metadata')
    chain = metadata.get(TRIGGER_CHAIN_KEY) if isinstance(metadata, Mapping) else None
    return isinstance(chain, Mapping) and chain.get('role') == _COMPLETION_GATE_ROLE


def review_due(
    chain: TriggerChain,
    facts: HumanGateFacts,
    *,
    human_gate_task_id: str,
    escalation_id: str,
    fired_at: str,
) -> ReviewDue:
    completion_gate_task_id, verdict, detail_lines = _completion_gate_reading(facts.completion_gate)
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


def _completion_gate_reading(
    gate: dict | CompletionGateUnread | None,
) -> tuple[str | None, dict[str, object] | None, tuple[str, ...]]:
    """``(completion_gate_task_id, verdict, detail_lines)`` for one completion-gate read."""
    if gate is None:
        return None, None, ()
    if isinstance(gate, CompletionGateUnread):
        return None, None, (
            f'Completion gate unknown: dependency {gate.task_id} could not be '
            f'read ({gate.reason}), so this review_due carries no verdict.',
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
