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
