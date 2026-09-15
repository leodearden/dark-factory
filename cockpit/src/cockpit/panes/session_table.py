"""cockpit.panes.session_table — pure session-table formatting + the DataTable widget.

Resolves PRD open-Q4 (failed-spawn/gone-window glyph set): the six Status
members fold into four glyphs -- awaiting-input is "blocked on you",
running/launching are "working", idle is "idle", and exited/
failed-to-start are "dead". An unrecognized/foreign status degrades to
'?' rather than raising (additive-safe, mirroring session_registry's own
no-coerce policy for spawn_mode/display).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from orchestrator.session_registry import TERMINAL_STATUSES, SessionRecord, Status
from textual.widgets import DataTable
from textual.widgets.data_table import RowDoesNotExist

_GLYPHS: dict[Status, str] = {
    Status.AWAITING_INPUT: '⏸',
    Status.RUNNING: '⚙',
    Status.LAUNCHING: '⚙',
    Status.IDLE: '✓',
    Status.EXITED: '☠',
    Status.FAILED_TO_START: '☠',
}

_FALLBACK_GLYPH = '?'


def state_glyph(status: Status | str) -> str:
    """Map a Status (or its wire string) to its display glyph.

    A foreign/unrecognized status value returns the fallback glyph rather
    than raising -- this must stay total over any status the registry
    hands it (fail-soft, PRD §2).
    """
    try:
        resolved = Status(status)
    except ValueError:
        return _FALLBACK_GLYPH
    return _GLYPHS.get(resolved, _FALLBACK_GLYPH)


# The focusability cue's vocabulary, deliberately disjoint from _GLYPHS
# above: status ("is it working?") and focusability ("can I get to it?")
# are orthogonal, so a marker that collided with a status glyph would make
# one column read as the other. Both are single-width, so the leading
# marker column can never shift the columns beside it.
_FOCUSABLE_MARKER = '▸'
_HEADLESS_MARKER = '·'


def is_focusable(record: SessionRecord) -> bool:
    """Can Enter raise a terminal for *record*?

    Mirrors decision_queue.resolve_target's SessionRecord branch, which is
    literally ``display -> DisplayTarget, None -> None`` and is the code
    app.py::_focus_slug actually runs -- a display-less (headless) agent
    session has no terminal anywhere, so focusing it is a no-op.

    Restated here rather than imported because decision_queue already
    imports this module (a reverse import would be a cycle); the agreement
    is pinned by test_session_table.py::TestFocusMarker::
    test_agrees_with_resolve_target instead, so the two statements cannot
    silently drift apart.
    """
    return record.display is not None


def focus_marker(record: SessionRecord) -> str:
    """Render *record*'s focusability as its row marker.

    Both states get a present, distinct glyph rather than marking one and
    leaving the other blank: ~85% of live rows are headless, so an absence
    would read as "column not populated yet" instead of "nothing to raise".
    Total over any record shape -- any display at all reads focusable,
    including an unrecognized kind (fail-soft, PRD §2: an unknown kind is
    still a real terminal, and mislabelling it unactionable is the worse
    error).
    """
    return _FOCUSABLE_MARKER if is_focusable(record) else _HEADLESS_MARKER


def format_title(record: SessionRecord) -> str:
    """Render 'role:project#task_id' (the '#task_id' segment omitted when absent).

    Never raises on an empty role/project -- a view must degrade, not
    crash, on a record shape it didn't control (fail-soft, PRD §2).
    """
    base = f'{record.role}:{record.project}'
    if record.task_id:
        return f'{base}#{record.task_id}'
    return base


_AGE_PLACEHOLDER = '?'


def format_age(start_ts: str, now: datetime) -> str:
    """Render the age of *start_ts* relative to *now* as its largest whole unit.

    An empty or unparseable start_ts degrades to '?' rather than raising
    (fail-soft, PRD §2). Mirrors cockpit.priority.score's naive-datetime
    handling: a naive start_ts or now is assumed to already be UTC, so
    mixing naive/aware timestamps never raises TypeError.
    """
    if not start_ts:
        return _AGE_PLACEHOLDER
    try:
        started = datetime.fromisoformat(start_ts)
    except ValueError:
        return _AGE_PLACEHOLDER
    started_aware = started if started.tzinfo is not None else started.replace(tzinfo=UTC)
    now_aware = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    age_seconds = max(0.0, (now_aware - started_aware).total_seconds())
    if age_seconds < 60:
        return f'{int(age_seconds)}s'
    if age_seconds < 3600:
        return f'{int(age_seconds // 60)}m'
    if age_seconds < 86400:
        return f'{int(age_seconds // 3600)}h'
    return f'{int(age_seconds // 86400)}d'


def count_outstanding_children(slug: str, all_records: list[SessionRecord]) -> int:
    """Count *slug*'s non-terminal children (records with parent_session_id == slug).

    A terminal child (exited/failed-to-start) is not "outstanding" -- it no
    longer needs attention -- so it's excluded per TERMINAL_STATUSES. A
    foreign/unrecognized child status is treated as non-terminal (kept,
    counted) rather than raising -- fail-soft via _is_terminal, mirroring
    _state_rank/state_glyph (PRD §2): one unclassifiable child must never
    abort the whole count.
    """
    return sum(
        1
        for record in all_records
        if record.parent_session_id == slug and not _is_terminal(record.status)
    )


# Documented state ranking for order_sessions: blocked-on-you first, then
# working, then idle, then dead. launching folds into "working" (rank 1)
# and both terminal statuses fold into "dead" (rank 3), mirroring _GLYPHS'
# four-bucket grouping. A foreign/unrecognized status sorts last of all.
_STATE_RANK: dict[Status, int] = {
    Status.AWAITING_INPUT: 0,
    Status.RUNNING: 1,
    Status.LAUNCHING: 1,
    Status.IDLE: 2,
    Status.EXITED: 3,
    Status.FAILED_TO_START: 3,
}

_UNKNOWN_STATE_RANK = 4


def _state_rank(status: Status | str) -> int:
    try:
        resolved = Status(status)
    except ValueError:
        return _UNKNOWN_STATE_RANK
    return _STATE_RANK.get(resolved, _UNKNOWN_STATE_RANK)


def _focus_rank(record: SessionRecord) -> int:
    return 0 if is_focusable(record) else 1


def _start_ts_sort_key(start_ts: str) -> tuple[int, str]:
    """Sort ascending by *start_ts* (oldest first); empty/unparseable sorts last.

    (0, start_ts) sorts before (1, '') for any parseable start_ts, so a
    record whose timestamp we can't make sense of never raises and never
    jumps the queue ahead of a record we can actually order.
    """
    if start_ts:
        try:
            datetime.fromisoformat(start_ts)
        except ValueError:
            pass
        else:
            return (0, start_ts)
    return (1, '')


def order_sessions(
    records: list[SessionRecord], *, focus_first: bool = False
) -> list[SessionRecord]:
    """Order *records* by state rank, then focusability if asked, then oldest first.

    State rank is always PRIMARY: blocked-on-you is the top signal, so a
    headless awaiting-input session outranks a focusable running one.

    focus_first sorts focusable-before-headless between that rank and the
    start_ts tiebreak. Opt-in because the two consumers want different
    things: the session table orders BEFORE capping (filter_live_sessions
    slices an already-ordered list), so it wants the rows an operator can
    actually act on to be the ones that survive the cap, while
    spawn_tree.py orders sibling groups, where oldest-first IS the signal
    (the spawn sequence) and must not be reshuffled by whether a child
    happens to own a terminal. Ordering only reorders; which rows are
    dropped is the cap's business, not this function's.

    A deterministic, dependency-free stand-in for the C5b priority score
    (this task's dependency surface is C1 only -- see design_decisions).
    Pure and total: no clock read, and an empty/unparseable start_ts sorts
    last within its rank rather than raising.
    """
    return sorted(
        records,
        key=lambda record: (
            _state_rank(record.status),
            _focus_rank(record) if focus_first else 0,
            _start_ts_sort_key(record.start_ts),
        ),
    )


def _is_terminal(status: Status | str) -> bool:
    """Is *status* one of TERMINAL_STATUSES (exited/failed-to-start)?

    A foreign/unrecognized status resolves to False (kept, not hidden) --
    mirrors _state_rank's fail-soft try/except (PRD §2): never hide a
    record we can't classify.
    """
    try:
        resolved = Status(status)
    except ValueError:
        return False
    return resolved in TERMINAL_STATUSES


# Hard ceiling on the default live band so ~10k scanned records never
# drown the view. Callers pass an already blocked-first-ordered list (see
# order_sessions), so slicing to the first `cap` retains the top-N most
# important sessions, not an arbitrary subset.
_DEFAULT_VISIBLE_CAP = 200


@dataclass(frozen=True)
class LiveSessions:
    """The records a view should render, plus how many there were before the cap.

    visible is a prefix of the ordered input and len(visible) <= total, so
    total - len(visible) is what the view is hiding. The two travel
    together from producer to renderer; see filter_live_sessions.
    """

    visible: list[SessionRecord]
    total: int


def filter_live_sessions(
    records: list[SessionRecord], *, cap: int = _DEFAULT_VISIBLE_CAP
) -> LiveSessions:
    """Drop terminal-status (exited/failed-to-start) records, preserving order.

    Then slices to the first `cap` of what remains -- pass an
    already-ordered list (see order_sessions) so the cap keeps the top-N
    of that order. The cap is now REPORTABLE rather than silent: the
    returned view carries the pre-cap live count alongside the slice, so a
    truncated table can say so (see format_visible_count) instead of
    looking identical to a complete one.

    Terminal records are excluded from the count as well as from the slice
    -- total is the size of the live band, not of the scanned set, so the
    notice never claims the cap hid history the view never meant to show.

    Pure and total: an empty input returns an empty view with total 0, and
    a foreign status is kept (see _is_terminal), never raising.
    """
    live = [record for record in records if not _is_terminal(record.status)]
    return LiveSessions(visible=live[:cap], total=len(live))


def format_visible_count(shown: int, total: int) -> str:
    """Render the cap notice: 'showing N of M', or '' when nothing is hidden.

    '' means "nothing to say", not "unknown" -- the caller assigns this
    result straight to border_subtitle, where an empty label renders
    nothing (measured on textual 8.2.8), so the notice appears only when
    the cap actually hid something and a complete table stays quiet.

    Total over any pair: shown > total is not a state filter_live_sessions
    can produce, but it degrades to '' rather than rendering a backwards
    count (fail-soft, PRD §2).
    """
    if total > shown:
        return f'showing {shown} of {total}'
    return ''


def _count_children_by_parent(all_records: list[SessionRecord]) -> dict[str, int]:
    """Precompute {parent_slug: outstanding-child-count} in a single pass over *all_records*.

    replace_rows previously called count_outstanding_children once per
    visible row, each call rescanning the full all_records list from
    scratch -- O(visible_rows * len(all_records)). With the history toggle
    able to set both to the full ~10k-record set, that's ~10k^2
    comparisons on every rebuild. Scanning all_records exactly once here
    and looking counts up by key is O(len(all_records)) regardless of how
    many rows are visible. Uses the same fail-soft _is_terminal check as
    count_outstanding_children (which is left in place, still directly
    tested, as the single-slug primitive).
    """
    counts: dict[str, int] = {}
    for record in all_records:
        parent = record.parent_session_id
        if parent and not _is_terminal(record.status):
            counts[parent] = counts.get(parent, 0) + 1
    return counts


class SessionTable(DataTable):
    """The session-registry table: one row per session, keyed by session_slug.

    Columns: focus marker / state glyph / title / age / project /
    outstanding children. The focus marker answers a different question
    from the state glyph -- a headless agent session has no terminal
    anywhere, so Enter can never raise anything for it, however busy it is.
    Selection-preserving: replace_rows re-locates the previously highlighted
    session_slug after a rebuild, so a poll tick never yanks the cursor away
    from the row an operator is looking at.
    """

    # The border is not decoration: textual paints border labels as part
    # of the border EDGE, so without one the "showing N of M" notice
    # replace_rows writes to border_subtitle is a silent no-op -- set and
    # readable, but invisible. See TestSessionTableCapNotice, which asserts
    # the border is still there for exactly this reason.
    DEFAULT_CSS = """
    SessionTable {
        width: 1fr;
        height: 1fr;
        border: round $panel;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault('cursor_type', 'row')
        super().__init__(*args, **kwargs)

    def on_mount(self) -> None:
        self.add_columns('', '', 'title', 'age', 'project', 'children')

    def highlighted_slug(self) -> str | None:
        """Return the session_slug of the currently-highlighted row, or None if empty."""
        if self.row_count == 0:
            return None
        row_key = self.coordinate_to_cell_key(self.cursor_coordinate).row_key
        return row_key.value

    def replace_rows(
        self,
        view: LiveSessions,
        now: datetime,
        *,
        all_records: list[SessionRecord] | None = None,
    ) -> None:
        """Rebuild VISIBLE rows from *view* (already ordered), preserving the cursor by slug.

        Takes the whole view rather than its slice: the rows and the count
        they were cut from cannot be separated at this boundary, so no
        caller can render a truncated table that looks complete. The notice
        is written unconditionally, so a rebuild that is no longer
        truncated clears a stale one rather than leaving it on screen.

        Outstanding-children counts are computed against *all_records* when
        given (the full scanned set), falling back to the visible rows
        otherwise -- so a filtered/capped visible subset never undercounts a
        visible parent's non-terminal children just because those children
        themselves are hidden from view. Counted via a single O(all_records)
        pass (_count_children_by_parent) rather than one full rescan of
        all_records per visible row.
        """
        records = view.visible
        counting_set = all_records if all_records is not None else records
        children_by_parent = _count_children_by_parent(counting_set)
        previous_slug = self.highlighted_slug()
        self.clear()
        for record in records:
            self.add_row(
                focus_marker(record),
                state_glyph(record.status),
                format_title(record),
                format_age(record.start_ts, now),
                record.project,
                str(children_by_parent.get(record.session_slug, 0)),
                key=record.session_slug,
            )
        self.border_subtitle = format_visible_count(len(records), view.total)
        if not self.row_count:
            return
        if previous_slug is not None:
            try:
                self.move_cursor(row=self.get_row_index(previous_slug))
            except RowDoesNotExist:
                self.move_cursor(row=0)

    def select_slug(self, slug: str) -> bool:
        """Move the cursor to *slug*'s row if present. Returns whether it was found."""
        try:
            self.move_cursor(row=self.get_row_index(slug))
        except RowDoesNotExist:
            return False
        return True
