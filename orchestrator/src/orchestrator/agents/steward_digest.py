"""What a fresh steward session is told about its steward's earlier work, and the one renderer for each part.

Two records: the L0s the steward instance concluded (``HandledL0``, rendered by
``render_handled_digest``) and the sessions that already ran on the L0 in hand
without concluding it (``EarlierAttempts``, rendered by
``render_earlier_attempts``).
"""

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
    'EarlierAttempts',
    'HandledL0',
    'render_earlier_attempts',
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
_ATTEMPTS_HEADING = '## Earlier Attempts at This Escalation'
_ATTEMPTS_LEAD = (
    'This is a fresh session: it does not remember them. Earlier sessions of '
    'this steward worked on this escalation and ended without concluding it:'
)
_ATTEMPTS_ADVICE = (
    'Their work is not in your context. Check the worktree (`git log`, '
    '`git status`) for what they left before you repeat an approach, and '
    'prefer a narrower one if a session already ran out of time.'
)


@dataclass(frozen=True)
class HandledL0:
    escalation_id: str
    category: str
    outcome: StewardOutcome


@dataclass(frozen=True)
class EarlierAttempts:
    timed_out: int = 0
    left_pending: int = 0
    no_output: int = 0


def _describe(outcome: StewardOutcome) -> str:
    match outcome:
        case StewardResolved(resolution_text=text):
            return f'resolved: {_bounded(text)}'
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
        f'- `{entry.escalation_id}` [{entry.category}] {_describe(entry.outcome)}'
        for entry in shown
    ]
    hidden = len(newest_first) - len(shown)
    if hidden:
        lines.append(f'- … {hidden} older not shown')
    entries = '\n'.join(lines)
    return f'\n{_HEADING}\n\n{_LEAD}\n\n{entries}\n'


def render_earlier_attempts(attempts: EarlierAttempts) -> str:
    counted = (
        (attempts.timed_out, 'killed at the session time limit'),
        (attempts.left_pending, 'ended with the escalation still pending'),
        (attempts.no_output, 'exited with no output'),
    )
    lines = [f'- {count} {what}' for count, what in counted if count]
    if not lines:
        return ''
    entries = '\n'.join(lines)
    return f'\n{_ATTEMPTS_HEADING}\n\n{_ATTEMPTS_LEAD}\n\n{entries}\n\n{_ATTEMPTS_ADVICE}\n'
