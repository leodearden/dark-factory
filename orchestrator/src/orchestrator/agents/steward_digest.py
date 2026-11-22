"""The record of L0s a steward instance concluded, and the one function that renders it into a fresh-session briefing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_never

from orchestrator.workflow_types import (
    StewardBudgetExhausted,
    StewardInterrupted,
    StewardOutcome,
    StewardReescalatedL1,
    StewardResolved,
    StewardTerminalDecision,
)

__all__ = [
    'DIGEST_MAX_ENTRIES',
    'DIGEST_RESOLUTION_MAX_CHARS',
    'HandledL0',
    'render_handled_digest',
]

DIGEST_MAX_ENTRIES = 10
DIGEST_RESOLUTION_MAX_CHARS = 500

_HEADING = '## Escalations This Steward Already Handled'
_LEAD = (
    'This is a fresh session: it does not remember earlier escalations. '
    'This list is the only record of the escalations this steward already '
    'handled for this task, most recent first.'
)


@dataclass(frozen=True)
class HandledL0:
    escalation_id: str
    category: str
    outcome: StewardOutcome


def _describe(outcome: StewardOutcome) -> str:
    match outcome:
        case StewardResolved(resolution_text=text):
            return f'resolved: {text}'
        case StewardReescalatedL1(esc_id=esc_id):
            return f're-escalated to level 1 as `{esc_id}`'
        case StewardInterrupted(reason=reason, wip_commits_present=wip):
            wip_flag = 'WIP commits present' if wip else 'no WIP commits'
            return f'interrupted ({reason}; {wip_flag})'
        case StewardBudgetExhausted():
            return 'steward lifetime budget exhausted'
        case StewardTerminalDecision(new_status=status):
            return f'task status became {status}'
        case _:
            assert_never(outcome)


def _bounded(text: str) -> str:
    flat = ' '.join(text.split())
    if len(flat) <= DIGEST_RESOLUTION_MAX_CHARS:
        return flat
    return f'{flat[:DIGEST_RESOLUTION_MAX_CHARS]}…'


def render_handled_digest(handled: Sequence[HandledL0]) -> str:
    if not handled:
        return ''
    newest_first = list(reversed(handled))
    shown = newest_first[:DIGEST_MAX_ENTRIES]
    lines = [
        f'- `{entry.escalation_id}` [{entry.category}] {_bounded(_describe(entry.outcome))}'
        for entry in shown
    ]
    hidden = len(newest_first) - len(shown)
    if hidden:
        lines.append(f'- … {hidden} older not shown')
    entries = '\n'.join(lines)
    return f'\n{_HEADING}\n\n{_LEAD}\n\n{entries}\n'
