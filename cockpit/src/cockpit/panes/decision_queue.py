"""cockpit.panes.decision_queue — score-ordered decision queue (Fleet Cockpit C5b, PRD §9).

Pure logic (scoring adapters mapping the C1 registry contract onto C3's
ScoringItem, row formatting, and order_queue's queue-build/sort) plus the
DecisionQueue(DataTable) widget itself, mirroring session_table.py's own
mix of pure helpers + the SessionTable(DataTable) widget in one module. The
pure functions above DecisionQueue stay fast/deterministic to unit test
directly (no pilot, no event loop) -- only DecisionQueue itself requires a
running Textual app, and is covered by test_app.py's pilot tests instead.
Column sizing follows the same split: derive_question_width decides the
question column's bound as a pure function of the rows and a measured
width, and DecisionQueue only measures itself and applies the answer.
Widths throughout are CELLS, not characters -- Textual crops a column by
rich's cell measurement, so a bound counted in characters would let
double-width text overrun its column (and lose its ellipsis with it).

Consumers import orchestrator.session_registry directly (mirrors
registry_reader.py/session_table.py -- PRD §6 G5: consumers import the
frozen C1 record shape, never re-derive it).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
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
from rich.cells import cell_len, set_cell_size
from textual.events import Resize
from textual.widgets import DataTable
from textual.widgets.data_table import RowDoesNotExist

from cockpit.backends import DisplayTarget
from cockpit.panes.placeholders import ABSENT_PLACEHOLDER
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

    severity is threaded through from decision.severity (the parked
    escalation's severity -- info|blocking|critical|urgent, or '' when
    unknown) so score() weights a real ask instead of always falling back
    to its default (Fleet Cockpit F7). category is left as '' (DecisionRecord
    carries no category) so score() falls back to its configured default
    there. state is copied through verbatim -- order_queue is what filters
    to state=='open'.
    """
    return ScoringItem(
        severity=decision.severity,
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


# The four column labels, in render order, as the single source for both
# DecisionQueue.on_mount's columns and derive_question_width's per-column
# floors -- a label wider than every cell in its column (as 'project#task'
# usually is) is what the column actually renders at, so the budget must
# measure it.
_COLUMN_LABELS = ('score', 'age', 'project#task', 'question')
# Floor for a derived question bound; see derive_question_width.
_QUESTION_MIN_WIDTH = 20
# The bound used before the widget has a measurable width; see
# derive_question_width.
_UNMEASURED_QUESTION_WIDTH = 60
_QUESTION_PLACEHOLDER = '(no question)'


def _one_line_question(question: str | None, max_width: int) -> str:
    """Collapse *question* to a single line of at most *max_width* CELLS.

    The bound belongs to the caller: DecisionQueue derives it from the width
    its question column actually has (see derive_question_width), which is
    what lets a wide terminal show a long question and a narrow one reflow
    shorter. Only the pre-layout fallback is a fixed number.

    Cells, not characters: the bound is spent against rich's cell
    measurement -- the same measurement Textual crops the column by -- so a
    CJK or emoji question is cut where it will actually be cut on screen
    and the ellipsis always lands inside the column. Measured with len()
    instead, a 172-character Japanese question renders 343 cells wide and
    Textual crops it near its middle, ellipsis and all, leaving the operator
    no sign the text was truncated at all.

    Fail-soft (PRD §2): a view must degrade a bad question shape, not raise.
    Empty/None/whitespace-only yields the placeholder. *max_width* is
    clamped to the placeholder's own width here rather than trusted: this
    function is total for every int its signature accepts, instead of only
    for the ones derive_question_width happens to return, because
    format_queue_row exposes the bound as a public keyword and an unclamped
    non-positive one would silently become a negative slice.
    """
    if not question:
        return _QUESTION_PLACEHOLDER
    collapsed = ' '.join(question.split())
    if not collapsed:
        return _QUESTION_PLACEHOLDER
    bound = max(max_width, cell_len(_QUESTION_PLACEHOLDER))
    if cell_len(collapsed) <= bound:
        return collapsed
    # set_cell_size pads with a space when the cut splits a double-width
    # character, so rstrip before appending the 1-cell ellipsis.
    return set_cell_size(collapsed, bound - 1).rstrip() + '…'


def _format_project_task(project: str, task_id: str | None) -> str:
    """Render 'project#task_id' ('#task_id' segment omitted when absent). Mirrors
    session_table.format_title's title-shape convention, minus the role segment."""
    return f'{project}#{task_id}' if task_id else project


def _fixed_cells(item: _QueueRowLike, now: datetime) -> tuple[str, str, str]:
    """Render *item*'s three fixed-content cells: score / age / project#task.

    Split out of format_queue_row so the column-width derivation and the row
    render share ONE definition of these cells: derive_question_width
    measures the exact strings add_row will later receive, rather than a
    second formatter written for measuring that could drift from the one
    that renders.

    Reuses session_table.format_age for the age column (fed item.filed_at's
    isoformat -- format_age's contract is a string timestamp) so the queue's
    age rendering stays byte-identical to the session table's.
    """
    return (
        f'{item.score:.1f}',
        format_age(item.filed_at.isoformat(), now),
        _format_project_task(item.project, item.task_id),
    )


def derive_question_width(
    items: Sequence[_QueueRowLike],
    now: datetime,
    available_width: int,
    *,
    cell_padding: int = 1,
) -> int:
    """Cells the question column gets once the fixed columns have taken theirs.

    The question bound is DERIVED from the width the column actually has
    rather than hardcoded: each fixed column claims max(its label, its widest
    cell across *items*) plus *cell_padding* on either side, and the question
    column takes what is left of *available_width*. Feed the result to both
    format_queue_row's question_width= and DataTable.add_column's width= so
    the measured budget and the rendered text cannot disagree.

    Every width here is measured in CELLS (rich's cell_len), the unit
    Textual lays a column out in -- a double-width project name claims two
    cells per character on screen and must claim two here too.

    *available_width* <= 0 means "not laid out yet" -- NOT "no room". A
    widget's content region reads 0 before its first layout and a rebuild is
    genuinely reachable then, so that case returns
    _UNMEASURED_QUESTION_WIDTH, retaining the pre-derivation bound for a
    width nothing can measure yet.

    The _QUESTION_MIN_WIDTH floor exists because the fixed columns can claim
    more than a narrow terminal has: it keeps the column wider than both the
    'question' label and the '(no question)' placeholder, so neither is
    itself truncated. Below the floor the table falls back to DataTable's
    own horizontal scrolling -- the pre-existing behaviour for a terminal too
    narrow to hold the row. _one_line_question clamps the bound it is handed
    independently; the floor is a display choice, not that function's
    safety net.
    """
    if available_width <= 0:
        return _UNMEASURED_QUESTION_WIDTH
    rows = [_fixed_cells(item, now) for item in items]
    used = sum(
        2 * cell_padding + max([cell_len(label), *(cell_len(row[index]) for row in rows)])
        for index, label in enumerate(_COLUMN_LABELS[:-1])
    )
    return max(_QUESTION_MIN_WIDTH, available_width - used - 2 * cell_padding)


def format_queue_row(
    item: _QueueRowLike,
    now: datetime,
    *,
    question_width: int = _UNMEASURED_QUESTION_WIDTH,
) -> tuple[str, str, str, str]:
    """Render *item* as the PRD row shape: score / age / project#task / question.

    The first three cells come from _fixed_cells -- the same helper the
    width derivation measures -- so the derived column budget and the
    rendered row can never disagree about what the row contains.

    *question_width* bounds the question cell in CELLS and is the caller's to
    supply: DecisionQueue derives it from its own measured width (see
    derive_question_width). The default is the pre-layout fallback, for a
    caller with no width to measure yet -- not a display bound anyone should
    rely on.
    """
    return (*_fixed_cells(item, now), _one_line_question(item.question, question_width))


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
    escalation_id: the backing record's escalation_id, or None -- not
        rendered by format_queue_row, but consumed by format_copy_payload
        (the copy affordance, task 2517).
    session_slug: the backing SessionRecord's slug for a session item, None
        for a decision item. Carried as its own field rather than decoded
        back out of *key*: order_queue already holds the record the slug
        comes from, so encoding it into a string only to write a parser for
        it would be the meaningful-string shape heuristic 12 rules out --
        and the parser's fail-soft branch would be dead code, since every
        session key this codebase builds comes from _session_key.
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
    escalation_id: str | None
    session_slug: str | None



# Bounds the clipboard payload on BOTH copy legs, because there is only one
# payload: cockpit/src/cockpit/app.py::CockpitApp.action_copy formats it once,
# before it knows which leg will run. Only the OSC 52 leg needs the cap --
# some terminals silently DROP an over-long OSC 52 write rather than
# truncating it, so an unbounded question could land on the clipboard as
# nothing at all (task 2517 amendment). The local-helper leg
# (cockpit/src/cockpit/clipboard.py::copy_to_system_clipboard, task 5448) has
# no payload limit and still receives the truncated question: a truncation it
# does not need, paid so that whichever leg runs puts byte-identical text on
# the clipboard. Sized generously above any realistic question length and
# comfortably under the payload limits reported by common terminals, so the
# unnecessary truncation is unreachable for a real question anyway.
_COPY_QUESTION_MAX_CHARS = 4000


def _cap_for_clipboard(text: str, limit: int = _COPY_QUESTION_MAX_CHARS) -> str:
    """Truncate *text* to at most *limit* characters, marking the cut with an ellipsis.

    A no-op for text already at or under *limit* -- the overwhelmingly
    common case for a real question. Purely defensive: this bounds the
    clipboard payload size, it does not detect or target any particular
    terminal's actual OSC 52 limit.
    """
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + '…'


def format_copy_payload(item: QueueItem) -> str:
    """Render *item* as a labeled multi-line clipboard payload (the copy
    affordance, task 2517).

    Mirrors detail_pane.render_detail's 'label: value' line style, so the
    clipboard block matches what the detail pane already shows the
    operator. Pure, no clock/IO. Fail-soft (PRD §2): a None task_id/
    escalation_id degrades to a placeholder, never the literal string
    'None'. The trailing id line is derived from item.kind/item.key --
    'decision_id: <id>' for a decision, or 'session: <slug>' for a
    session (QueueItem.session_slug). The question
    line is defensively capped (_cap_for_clipboard) so a pathologically
    long question can't silently vanish on a terminal that drops rather
    than truncates an over-long OSC 52 payload. One payload serves both
    copy legs, so the cap applies even when the local helper --
    cockpit/src/cockpit/clipboard.py::copy_to_system_clipboard, which has
    no payload limit of its own -- is the leg that takes it (see
    _cap_for_clipboard's comment).
    """
    if item.kind == 'session':
        id_line = f'session: {item.session_slug}'
    else:
        id_line = f'decision_id: {item.decision_id}'
    question = _cap_for_clipboard(item.question or _QUESTION_PLACEHOLDER)
    return '\n'.join(
        [
            f'question: {question}',
            f'project: {item.project}',
            f'task_id: {item.task_id or ABSENT_PLACEHOLDER}',
            f'escalation_id: {item.escalation_id or ABSENT_PLACEHOLDER}',
            id_line,
        ]
    )


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
    handling: Iterable[str] | None = None,
    dropped: Iterable[str] | None = None,
) -> list[QueueItem]:
    """Build the score-ordered decision queue: every open decision + awaiting-input session.

    Pure and deterministic: identical inputs (including *now*) always
    produce an identical order (C3's score() is itself pure). Filters to
    state=='open' decisions and AWAITING_INPUT sessions only -- dropped/
    answered decisions and non-awaiting sessions never appear. *dropped*
    is a caller-supplied set of item keys ('decision:<id>'/'session:<slug>')
    excluded on top of that: a DECISION row is normally dropped by
    persisting state='dropped' and re-scanning (so it is already excluded
    by the state filter above before it ever reaches *dropped*), but a
    SESSION row has no cockpit-writable state field (PRD §2 design
    decisions) -- its drop lives ONLY as an in-memory key in *dropped*,
    which this function is what filters out. Sorted by score descending,
    tiebroken by *key* for a total, stable order across calls.
    """
    boosts = boosts or {}
    deferred = deferred or {}
    handling_set = set(handling) if handling is not None else set()
    dropped_set = set(dropped) if dropped is not None else set()
    sessions_by_slug = {session.session_slug: session for session in sessions}

    items: list[QueueItem] = []

    for decision in decisions:
        if decision.state != DecisionState.OPEN:
            continue
        key = _decision_key(decision)
        if key in dropped_set:
            continue
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
                escalation_id=decision.escalation_id,
                session_slug=None,
            )
        )

    for session in sessions:
        if session.status != Status.AWAITING_INPUT:
            continue
        key = _session_key(session)
        if key in dropped_set:
            continue
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
                escalation_id=session.escalation_id,
                session_slug=session.session_slug,
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

    The question column takes whatever the other three leave (see
    derive_question_width), so a wide terminal spends the line on the question
    rather than on a blank gutter. Every render re-declares all four columns
    through clear(columns=True) plus an EXPLICIT question width, because
    Textual's auto-width only ever grows: add_row raises a Column's
    content_width and clear() does not reset it, so an auto-sized question
    column could never give cells back when the terminal narrows.
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
        # The rows last handed to replace_rows, held so a width change can
        # re-render them without re-scanning the registry, and the question
        # width they were rendered at.
        self._rendered: tuple[tuple[QueueItem, ...], datetime] | None = None
        self._question_width = _UNMEASURED_QUESTION_WIDTH

    def on_mount(self) -> None:
        # Columns must exist before the first render, which happens before
        # there is any width to measure -- hence the unmeasured fallback.
        self._add_columns(_UNMEASURED_QUESTION_WIDTH)

    def _add_columns(self, question_width: int) -> None:
        """Declare the four columns: three auto-sized, the question one explicit."""
        self.add_columns(*_COLUMN_LABELS[:-1])
        self.add_column(_COLUMN_LABELS[-1], width=question_width)

    def _derive_question_width(self, items: Sequence[QueueItem], now: datetime) -> int:
        """The question bound for *items* at the width this widget has right now.

        scrollable_content_region excludes the border, the padding AND the
        vertical scrollbar, so it is the width the columns actually get;
        size.width would over-claim the scrollbar's cells and reintroduce
        horizontal overflow, but only once the queue grew long enough to
        scroll. It reads 0 before the first layout, which
        derive_question_width already handles as its "not laid out yet" case.
        """
        return derive_question_width(
            items, now, self.scrollable_content_region.width, cell_padding=self.cell_padding
        )

    def highlighted_key(self) -> str | None:
        """Return the QueueItem.key of the currently-highlighted row, or None if empty."""
        if self.row_count == 0:
            return None
        row_key = self.coordinate_to_cell_key(self.cursor_coordinate).row_key
        return row_key.value

    def _reflow(self) -> None:
        """Re-render the cached rows at the width this widget currently has.

        Re-declares the columns rather than mutating them: clear(columns=True)
        plus _add_columns is the public way to change the question column's
        explicit width, and it resets the three auto columns' stale
        content_width for free -- so one freak-width 'project#task' cannot
        keep stealing cells from the question column for the rest of the
        session.

        clear() also resets the cursor to row 0, so the previously-highlighted
        key is re-located afterwards exactly as replace_rows has always done:
        a rebuild must never yank the cursor off the row an operator is
        reading.
        """
        if self._rendered is None:
            return
        items, now = self._rendered
        width = self._derive_question_width(items, now)
        previous_key = self.highlighted_key()
        self.clear(columns=True)
        self._add_columns(width)
        self._question_width = width
        for item in items:
            self.add_row(*format_queue_row(item, now, question_width=width), key=item.key)
        if not self.row_count:
            return
        if previous_key is not None:
            try:
                self.move_cursor(row=self.get_row_index(previous_key))
            except RowDoesNotExist:
                self.move_cursor(row=0)

    def replace_rows(self, items: Sequence[QueueItem], now: datetime) -> None:
        """Rebuild rows from *items* (already ordered), preserving the cursor by key.

        The question column is sized to what the other three leave at this
        widget's current width. (*items*, *now*) are cached so a later width
        change can re-render the same rows without re-scanning the registry
        and without re-reading the clock.
        """
        self._rendered = (tuple(items), now)
        self._reflow()

    def on_resize(self, event: Resize) -> None:
        """Re-render at the new width, but only when the derived bound changed.

        This is also how the widget first learns its real width: at mount
        there is nothing to measure (scrollable_content_region reads 0 before
        layout), so replace_rows applies the unmeasured fallback and the
        first Resize is what replaces it with the measured bound.

        The equality guard matters because Resize fires for layout changes
        that leave the column budget alone, and a reflow goes through
        clear(columns=True), which resets the scroll position -- a resize
        that changes nothing must cost nothing on screen.

        The cached (items, now) are reused rather than re-scanned and
        re-clocked: re-running the app's registry rebuild would fire real
        backend.set_urgency calls on a window drag, and re-reading the clock
        would make the age column jump mid-drag.

        The reflow's cursor events are suppressed because a window drag is not
        an operator selection: _reflow re-enters the cursor through clear() +
        move_cursor, and CockpitApp.on_data_table_row_highlighted reads a
        RowHighlighted as the operator CLAIMING the detail pane for that
        table. Measured: unsuppressed, a resize silently takes the pane off
        the session row an operator parked on and hands it to the queue.
        Nothing needs refreshing afterwards -- a resize changes column widths
        only, never the underlying records, so leaving the pane exactly as it
        was is the correct outcome, not a gap.

        The suppression lives HERE rather than in _reflow or replace_rows on
        purpose. app.py wraps its own replace_rows calls and re-syncs the pane
        explicitly in the suppressed reposts' place, so that path's event
        policy stays the caller's; only the resize path, which this widget
        originates and no app code can wrap, suppresses for itself.

        DataTable defines its own private _on_resize; Textual dispatches both,
        so this handler augments the table's own resize handling rather than
        replacing it.
        """
        if self._rendered is None:
            return
        items, now = self._rendered
        if self._derive_question_width(items, now) == self._question_width:
            return
        with self.prevent(DataTable.RowHighlighted):
            self._reflow()

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
