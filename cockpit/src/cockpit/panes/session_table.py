"""cockpit.panes.session_table — pure session-table formatting + the DataTable widget.

Resolves PRD open-Q4 (failed-spawn/gone-window glyph set): the six Status
members fold into four glyphs -- awaiting-input is "blocked on you",
running/launching are "working", idle is "idle", and exited/
failed-to-start are "dead". An unrecognized/foreign status degrades to
'?' rather than raising (additive-safe, mirroring session_registry's own
no-coerce policy for spawn_mode/display).
"""

from __future__ import annotations

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
    longer needs attention -- so it's excluded per TERMINAL_STATUSES.
    """
    return sum(
        1
        for record in all_records
        if record.parent_session_id == slug and Status(record.status) not in TERMINAL_STATUSES
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


def order_sessions(records: list[SessionRecord]) -> list[SessionRecord]:
    """Order *records* blocked-first, then by state rank, then oldest start_ts first.

    A deterministic, dependency-free stand-in for the C5b priority score
    (this task's dependency surface is C1 only -- see design_decisions).
    Pure and total: no clock read, and an empty/unparseable start_ts sorts
    last within its rank rather than raising.
    """
    return sorted(
        records,
        key=lambda record: (_state_rank(record.status), _start_ts_sort_key(record.start_ts)),
    )


class SessionTable(DataTable):
    """The session-registry table: one row per session, keyed by session_slug.

    Columns: state glyph / title / age / project / outstanding children.
    Selection-preserving: replace_rows re-locates the previously highlighted
    session_slug after a rebuild, so a poll tick never yanks the cursor away
    from the row an operator is looking at.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault('cursor_type', 'row')
        super().__init__(*args, **kwargs)

    def on_mount(self) -> None:
        self.add_columns('', 'title', 'age', 'project', 'children')

    def highlighted_slug(self) -> str | None:
        """Return the session_slug of the currently-highlighted row, or None if empty."""
        if self.row_count == 0:
            return None
        row_key = self.coordinate_to_cell_key(self.cursor_coordinate).row_key
        return row_key.value

    def replace_rows(self, records: list[SessionRecord], now: datetime) -> None:
        """Rebuild rows from *records* (already ordered), preserving the cursor by slug.

        *records* is the FULL record set (ordered) -- outstanding-children
        counts are computed against it, not a filtered subset.
        """
        previous_slug = self.highlighted_slug()
        self.clear()
        for record in records:
            self.add_row(
                state_glyph(record.status),
                format_title(record),
                format_age(record.start_ts, now),
                record.project,
                str(count_outstanding_children(record.session_slug, records)),
                key=record.session_slug,
            )
        if not self.row_count:
            return
        if previous_slug is not None:
            try:
                self.move_cursor(row=self.get_row_index(previous_slug))
            except RowDoesNotExist:
                self.move_cursor(row=0)
