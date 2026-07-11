"""cockpit.panes.decision_queue — score-ordered decision queue (Fleet Cockpit C5b, PRD §9).

Pure logic only: scoring adapters mapping the C1 registry contract onto C3's
ScoringItem, row formatting, and order_queue's queue-build/sort. The
DecisionQueue(DataTable) widget itself lands in a later C5b step; this
module stays import-clean of any Textual dependency so it is fast and
deterministic to unit test (no pilot, no event loop).

Consumers import orchestrator.session_registry directly (mirrors
registry_reader.py/session_table.py -- PRD §6 G5: consumers import the
frozen C1 record shape, never re-derive it).
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from orchestrator.session_registry import DecisionRecord, SessionRecord

from cockpit.panes.session_table import format_age
from cockpit.priority import ScoringItem


def _parse_timestamp(raw: str | None, now: datetime) -> datetime:
    """Parse an ISO-8601 timestamp, degrading to *now* on any empty/unparseable input.

    Fail-soft (PRD §2): an adapter must never raise on a timestamp shape it
    didn't control. Falling back to *now* (age ~= 0) rather than the epoch
    keeps a missing/corrupt timestamp from manufacturing an age bonus the
    underlying record never earned.
    """
    if not raw:
        return now
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return now


def decision_to_scoring_item(decision: DecisionRecord, *, now: datetime) -> ScoringItem:
    """Map a C1 DecisionRecord onto a C3 ScoringItem.

    severity/category are left as '' (DecisionRecord carries neither) so
    score() falls back to its configured defaults for both. state is copied
    through verbatim -- order_queue is what filters to state=='open'.
    """
    return ScoringItem(
        severity='',
        category='',
        project=decision.project,
        filed_at=_parse_timestamp(decision.filed_at, now),
        manual_boost=decision.manual_boost,
        state=str(decision.state),
    )


def session_to_scoring_item(session: SessionRecord, *, now: datetime) -> ScoringItem:
    """Map a C1 SessionRecord onto a C3 ScoringItem.

    Only meaningful for an AWAITING_INPUT session (order_queue's filter);
    always scores as state='open' -- an awaiting-input session IS an open
    ask for attention, structurally equivalent to an open decision.
    severity/category are left as '' (SessionRecord carries neither) and
    manual_boost is always 0 (sessions have no persisted priority field --
    see order_queue's boosts overlay for the in-memory equivalent).
    filed_at prefers the pending question's asked_at, falling back to the
    session's own start_ts when there's no question (or an unstamped one).
    """
    raw = session.start_ts
    if session.question is not None and session.question.asked_at:
        raw = session.question.asked_at
    return ScoringItem(
        severity='',
        category='',
        project=session.project,
        filed_at=_parse_timestamp(raw, now),
        manual_boost=0,
        state='open',
    )


class _QueueRowLike(Protocol):
    """Structural contract format_queue_row reads.

    Mirrors ScoringItem's own "any object exposing these attributes works"
    convention -- format_queue_row reads attributes, not isinstance.
    QueueItem (a later step) satisfies this directly.
    """

    score: float
    filed_at: datetime
    project: str
    task_id: str | None
    question: str | None


_QUESTION_MAX_WIDTH = 60
_QUESTION_PLACEHOLDER = '(no question)'


def _one_line_question(question: str | None) -> str:
    """Collapse *question* to a single truncated line; empty/None -> a placeholder.

    Fail-soft (PRD §2): a view must degrade a bad question shape, not raise.
    """
    if not question:
        return _QUESTION_PLACEHOLDER
    collapsed = ' '.join(question.split())
    if not collapsed:
        return _QUESTION_PLACEHOLDER
    if len(collapsed) <= _QUESTION_MAX_WIDTH:
        return collapsed
    return collapsed[: _QUESTION_MAX_WIDTH - 1].rstrip() + '…'


def _format_project_task(project: str, task_id: str | None) -> str:
    """Render 'project#task_id' ('#task_id' segment omitted when absent). Mirrors
    session_table.format_title's title-shape convention, minus the role segment."""
    return f'{project}#{task_id}' if task_id else project


def format_queue_row(item: _QueueRowLike, now: datetime) -> tuple[str, str, str, str]:
    """Render *item* as the PRD row shape: score / age / project#task / question.

    Reuses session_table.format_age for the age column (fed item.filed_at's
    isoformat -- format_age's contract is a string timestamp) so the queue's
    age rendering stays byte-identical to the session table's.
    """
    return (
        f'{item.score:.1f}',
        format_age(item.filed_at.isoformat(), now),
        _format_project_task(item.project, item.task_id),
        _one_line_question(item.question),
    )
