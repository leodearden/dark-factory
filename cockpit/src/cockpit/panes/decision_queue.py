"""cockpit.panes.decision_queue — score-ordered decision queue (Fleet Cockpit C5b, PRD §9).

Pure logic (scoring adapters mapping the C1 registry contract onto C3's
ScoringItem, row formatting, and order_queue's queue-build/sort) plus the
DecisionQueue(DataTable) widget itself, mirroring session_table.py's own
mix of pure helpers + the SessionTable(DataTable) widget in one module. The
pure functions above DecisionQueue stay fast/deterministic to unit test
directly (no pilot, no event loop) -- only DecisionQueue itself requires a
running Textual app, and is covered by test_app.py's pilot tests instead.

Consumers import orchestrator.session_registry directly (mirrors
registry_reader.py/session_table.py -- PRD §6 G5: consumers import the
frozen C1 record shape, never re-derive it).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from orchestrator.session_registry import (
    DecisionRecord,
    DecisionState,
    Display,
    SessionRecord,
    Status,
)
from textual.widgets import DataTable
from textual.widgets.data_table import RowDoesNotExist

from cockpit.backends import DisplayTarget
from cockpit.panes.session_table import format_age
from cockpit.priority import Priorities, ScoringItem, score


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
    convention -- format_queue_row reads attributes, not isinstance. Declared
    as read-only properties (rather than plain mutable attributes) so a
    frozen dataclass -- QueueItem, below, is the real caller -- satisfies
    this structurally: a read-only Protocol attribute accepts either a
    read-only or a writable concrete attribute, but a writable Protocol
    attribute would reject a frozen field.
    """

    @property
    def score(self) -> float: ...
    @property
    def filed_at(self) -> datetime: ...
    @property
    def project(self) -> str: ...
    @property
    def task_id(self) -> str | None: ...
    @property
    def question(self) -> str | None: ...


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


@dataclass(frozen=True)
class QueueItem:
    """One score-ordered decision-queue row -- a decision or an awaiting-input session.

    key: a stable, kind-prefixed identity ('decision:<id>' / 'session:<slug>')
        used both as the DataTable row key and as the lookup key into the
        app's in-memory boosts/deferred/handling overlays.
    kind: 'decision' | 'session'.
    decision_id: the backing DecisionRecord's id, or None for a session item
        (sessions aren't cockpit-writable -- see set_manual_boost/
        update_decision_state's callers).
    target: this item's resolved focus target, or None when unresolvable
        (a gone/unlinked session) -- see resolve_target.
    handling: whether this item's key is in the app's in-memory "already
        acted on" set (stamped by order_queue's *handling* param).
    """

    key: str
    kind: str
    decision_id: str | None
    project: str
    task_id: str | None
    question: str
    filed_at: datetime
    score: float
    target: DisplayTarget | None
    handling: bool


def _to_display_target(display: Display | None) -> DisplayTarget | None:
    if display is None:
        return None
    return DisplayTarget(
        kind=display.kind,
        wm_title=display.wm_title,
        wm_window_id=display.wm_window_id,
        tmux_target=display.tmux_target,
    )


def resolve_target(
    record: DecisionRecord | SessionRecord,
    sessions_by_slug: Mapping[str, SessionRecord],
) -> DisplayTarget | None:
    """Resolve *record*'s focus target.

    A SessionRecord resolves from its own display. A DecisionRecord has no
    display of its own -- it resolves via the session named by its
    session_id, looked up in *sessions_by_slug*. Fail-soft (PRD §2): a
    session-less decision, an unresolvable session_id, or a linked session
    with no display all degrade to None rather than raising -- a gone or
    never-linked target simply means no focus/urgency call is possible for
    this item.
    """
    if isinstance(record, DecisionRecord):
        session = sessions_by_slug.get(record.session_id) if record.session_id else None
        display = session.display if session is not None else None
    else:
        display = record.display
    return _to_display_target(display)


def _apply_overrides(
    item: ScoringItem,
    key: str,
    boosts: Mapping[str, int],
    deferred: Mapping[str, datetime],
) -> ScoringItem:
    """Layer the in-memory boosts/deferred overlays onto *item* (PRD §2 design decisions).

    boosts ADDS to the item's own manual_boost (an ephemeral overlay for a
    session row; a no-op overlay for a decision row once its boost is
    persisted and re-scanned). deferred OVERRIDES filed_at to the stamped
    defer-moment, resetting the item's effective age -- never a rewrite of
    the underlying record's real filed_at/asked_at/start_ts.
    """
    filed_at = deferred.get(key, item.filed_at)
    manual_boost = item.manual_boost + boosts.get(key, 0)
    return replace(item, filed_at=filed_at, manual_boost=manual_boost)


def _decision_key(decision: DecisionRecord) -> str:
    return f'decision:{decision.id}'


def _session_key(session: SessionRecord) -> str:
    return f'session:{session.session_slug}'


def order_queue(
    decisions: Sequence[DecisionRecord],
    sessions: Sequence[SessionRecord],
    priorities: Priorities,
    now: datetime,
    *,
    boosts: Mapping[str, int] | None = None,
    deferred: Mapping[str, datetime] | None = None,
    handling: Sequence[str] | None = None,
) -> list[QueueItem]:
    """Build the score-ordered decision queue: every open decision + awaiting-input session.

    Pure and deterministic: identical inputs (including *now*) always
    produce an identical order (C3's score() is itself pure). Filters to
    state=='open' decisions and AWAITING_INPUT sessions only -- dropped/
    answered decisions and non-awaiting sessions never appear (a session
    row an operator has since dropped is excluded by the caller pre-
    filtering *sessions*, not by this function). Sorted by score
    descending, tiebroken by *key* for a total, stable order across calls.
    """
    boosts = boosts or {}
    deferred = deferred or {}
    handling_set = set(handling) if handling is not None else set()
    sessions_by_slug = {session.session_slug: session for session in sessions}

    items: list[QueueItem] = []

    for decision in decisions:
        if decision.state != DecisionState.OPEN:
            continue
        key = _decision_key(decision)
        scoring_item = _apply_overrides(
            decision_to_scoring_item(decision, now=now), key, boosts, deferred
        )
        items.append(
            QueueItem(
                key=key,
                kind='decision',
                decision_id=decision.id,
                project=decision.project,
                task_id=decision.task_id,
                question=decision.text,
                filed_at=scoring_item.filed_at,
                score=score(scoring_item, priorities, now),
                target=resolve_target(decision, sessions_by_slug),
                handling=key in handling_set,
            )
        )

    for session in sessions:
        if session.status != Status.AWAITING_INPUT:
            continue
        key = _session_key(session)
        scoring_item = _apply_overrides(
            session_to_scoring_item(session, now=now), key, boosts, deferred
        )
        question_text = session.question.text if session.question is not None else ''
        items.append(
            QueueItem(
                key=key,
                kind='session',
                decision_id=None,
                project=session.project,
                task_id=session.task_id,
                question=question_text,
                filed_at=scoring_item.filed_at,
                score=score(scoring_item, priorities, now),
                target=resolve_target(session, sessions_by_slug),
                handling=key in handling_set,
            )
        )

    return sorted(items, key=lambda item: (-item.score, item.key))


class DecisionQueue(DataTable):
    """The score-ordered decision/awaiting-input queue (PRD §9 C5b).

    Columns: score / age / project#task / question. Selection-preserving,
    mirroring SessionTable.replace_rows' idiom exactly: replace_rows
    re-locates the previously-highlighted item's key after a rebuild, so a
    live re-score (boost/defer/drop, or a poll pickup) never yanks the
    cursor away from the row an operator is looking at. Rows are keyed by
    QueueItem.key ('decision:<id>' / 'session:<slug>'), not by row position.
    """

    DEFAULT_CSS = """
    DecisionQueue {
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault('cursor_type', 'row')
        super().__init__(*args, **kwargs)

    def on_mount(self) -> None:
        self.add_columns('score', 'age', 'project#task', 'question')

    def highlighted_key(self) -> str | None:
        """Return the QueueItem.key of the currently-highlighted row, or None if empty."""
        if self.row_count == 0:
            return None
        row_key = self.coordinate_to_cell_key(self.cursor_coordinate).row_key
        return row_key.value

    def replace_rows(self, items: Sequence[QueueItem], now: datetime) -> None:
        """Rebuild rows from *items* (already ordered), preserving the cursor by key."""
        previous_key = self.highlighted_key()
        self.clear()
        for item in items:
            self.add_row(*format_queue_row(item, now), key=item.key)
        if not self.row_count:
            return
        if previous_key is not None:
            try:
                self.move_cursor(row=self.get_row_index(previous_key))
            except RowDoesNotExist:
                self.move_cursor(row=0)

    def select_key(self, key: str) -> bool:
        """Move the cursor to *key*'s row if present. Returns whether it was found."""
        try:
            self.move_cursor(row=self.get_row_index(key))
        except RowDoesNotExist:
            return False
        return True


def known_project_roots(
    records: Sequence[SessionRecord], extra: Sequence[str] = ()
) -> list[str]:
    """Candidate project roots for the spawn bar's project picker (PRD §9).

    Derived from the distinct non-empty ``cwd`` values across *records*
    (typically every scanned SessionRecord, not just awaiting-input ones --
    this is a picker source, not a queue filter) unioned with any
    explicitly-configured *extra* roots. Deduped and sorted for a stable,
    deterministic picker order. Fail-soft (PRD §2): a record with an empty
    cwd is simply excluded, never raises.
    """
    roots = {record.cwd for record in records if record.cwd}
    roots.update(extra)
    return sorted(roots)
