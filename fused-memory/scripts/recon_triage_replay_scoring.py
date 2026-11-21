#!/usr/bin/env python3
"""Score recon's repairs of write-triage attaches against rubric v2.2's rated soak corpus.

Each replayed child carries the outcome triage filed it under and one repair
recon chose from a fixed vocabulary (:class:`Decision`). The correct repair is
read off the rubric's own cells by :func:`expected_action`, and every repair
is priced by deriving the repaired item row through the same rubric call that
priced the judge, so recon and the judge are compared on the same rows and
weights. Rationale and findings: ``fused-memory/calibration/recon_triage_replay_report.md``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
)
from fused_memory.utils.script_loader import load_script

_SCRIPTS = Path(__file__).resolve().parent


if TYPE_CHECKING:
    import score_write_triage_reader as _reader
    import write_triage_reader_rubric as _rubric
    import write_triage_stats as _stats
else:
    _rubric = load_script(_SCRIPTS / 'write_triage_reader_rubric.py', 'write_triage_reader_rubric')
    _reader = load_script(_SCRIPTS / 'score_write_triage_reader.py', 'score_write_triage_reader')
    _stats = load_script(_SCRIPTS / 'write_triage_stats.py', 'write_triage_stats')

Cell = _rubric.Cell


class Decision(StrEnum):
    """What recon does with one attached child."""

    KEEP = 'keep'
    DETACH = 'detach'
    RELABEL = 'relabel'
    CONTEST_UPHELD = 'contest_upheld'
    SUPERSEDED_PARENT = 'superseded_parent'
    NOT_A_CONTEST = 'not_a_contest'


RELABEL_TARGETS = frozenset({OUTCOME_AMENDED, OUTCOME_CONTESTED})
"""The outcomes a relabel can give a child."""


@dataclass(frozen=True)
class Action:
    """One repair: a decision, and the outcome a relabel gives (None for every other decision)."""

    decision: Decision
    relabel_to: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, Decision):
            raise ValueError(f'decision {self.decision!r} is not one of {sorted(Decision)}')
        if self.decision is Decision.RELABEL and self.relabel_to not in RELABEL_TARGETS:
            raise ValueError(
                f'decision {self.decision.value!r} needs relabel_to, one of'
                f' {sorted(RELABEL_TARGETS)}; got {self.relabel_to!r}'
            )
        if self.decision is not Decision.RELABEL and self.relabel_to is not None:
            raise ValueError(
                f'decision {self.decision.value!r} takes no relabel_to; got {self.relabel_to!r}'
            )

    @property
    def label(self) -> str:
        """The action as one word: the decision, or ``relabel-><outcome>``."""
        if self.relabel_to is None:
            return self.decision.value
        return f'{self.decision.value}->{self.relabel_to}'


KEEP = Action(Decision.KEEP)
DETACH = Action(Decision.DETACH)
CONTEST_UPHELD = Action(Decision.CONTEST_UPHELD)
SUPERSEDED_PARENT = Action(Decision.SUPERSEDED_PARENT)
NOT_A_CONTEST = Action(Decision.NOT_A_CONTEST)
RELABEL_AMENDED = Action(Decision.RELABEL, OUTCOME_AMENDED)
RELABEL_CONTESTED = Action(Decision.RELABEL, OUTCOME_CONTESTED)

_ALLOWED: Mapping[str, frozenset[Action]] = MappingProxyType({
    OUTCOME_CONTESTED: frozenset({CONTEST_UPHELD, NOT_A_CONTEST, DETACH, SUPERSEDED_PARENT}),
    OUTCOME_AMENDED: frozenset({KEEP, RELABEL_CONTESTED, DETACH, SUPERSEDED_PARENT}),
    OUTCOME_RESTATED: frozenset({KEEP, RELABEL_AMENDED, RELABEL_CONTESTED, DETACH, SUPERSEDED_PARENT}),
})


def allowed_actions(outcome: str) -> frozenset[Action]:
    """Every repair recon may choose for a child triage filed as *outcome*."""
    allowed = _ALLOWED.get(outcome)
    if allowed is None:
        raise ValueError(f'outcome {outcome!r} has no repair vocabulary; attach outcomes: {sorted(_ALLOWED)}')
    return allowed


def action_for(outcome: str, decision: str, relabel_to: str | None = None) -> Action:
    """The action *decision* and *relabel_to* name, refused unless *outcome* allows it."""
    try:
        action = Action(Decision(decision), relabel_to)
    except ValueError as exc:
        raise ValueError(
            f'outcome {outcome!r}: decision {decision!r} with relabel_to {relabel_to!r} is refused: {exc}'
        ) from exc
    allowed = allowed_actions(outcome)
    if action not in allowed:
        raise ValueError(
            f'outcome {outcome!r} does not allow decision {decision!r} with relabel_to'
            f' {relabel_to!r}; it allows {sorted(choice.label for choice in allowed)}'
        )
    return action


_CONFLICT_CELLS = frozenset({Cell.DIGEST_CONFLICT, Cell.HIDDEN_CONFLICT})


def expected_action(outcome: str, decided: _rubric.Decided) -> Action:
    """The repair that clears *decided*'s wrong cells, by one fixed precedence.

    Supersession, then a true dispute (an upheld contest, or a relabel to
    contested), then a wrong home (detach), then a false contest, then hidden
    detail (relabel to amended); otherwise the outcome stands.
    """
    allowed_actions(outcome)
    cells = decided.triage_cells
    if Cell.ATTACHED_SUPERSESSION in cells:
        return SUPERSEDED_PARENT
    if outcome == OUTCOME_CONTESTED:
        if decided.conflict:
            return CONTEST_UPHELD
        return DETACH if Cell.WRONG_HOME in cells else NOT_A_CONTEST
    if cells & _CONFLICT_CELLS:
        return RELABEL_CONTESTED
    if Cell.WRONG_HOME in cells:
        return DETACH
    if outcome == OUTCOME_RESTATED and Cell.HIDDEN_DETAIL in cells:
        return RELABEL_AMENDED
    return KEEP


def repaired_item(row: Mapping[str, Any], action: Action) -> dict[str, Any]:
    """A new item row: the write as it reads after *action*; *row* is left untouched.

    A detach or a supersession stores the write standalone against its old
    parent, a supersession also marking that parent; a relabel or a refuted
    contest re-files the write under its new outcome.
    """
    outcome = row['outcome']
    if action not in allowed_actions(outcome):
        raise ValueError(
            f'item {row.get("entry_id")!r}: outcome {outcome!r} does not allow {action.label!r}'
        )
    repaired = {
        key: list(value) if isinstance(value, list) else value for key, value in row.items()
    }
    if action.decision in (Decision.DETACH, Decision.SUPERSEDED_PARENT):
        parent = row['target_id']
        repaired.update(
            outcome=OUTCOME_STORED, target_id=None, earlier_ids=[parent], supersedes_target=False,
            superseded_marks=[parent] if action.decision is Decision.SUPERSEDED_PARENT else [],
        )
    elif action.decision is Decision.RELABEL:
        repaired['outcome'] = action.relabel_to
    elif action.decision is Decision.NOT_A_CONTEST:
        repaired['outcome'] = OUTCOME_AMENDED
    return repaired
