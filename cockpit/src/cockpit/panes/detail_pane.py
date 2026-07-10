"""cockpit.panes.detail_pane — pure detail rendering + the DetailPane widget.

render_detail composes a single record's full detail (title, status, age,
ids, question, parent/children, result-file tail) as plain text. Fail-soft
throughout (PRD §2): a missing/unreadable result file degrades to a
placeholder rather than raising, and a record with no question/children
still renders.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from orchestrator.session_registry import SessionRecord

from cockpit.panes.session_table import format_age, format_title, state_glyph

logger = logging.getLogger(__name__)

_NO_RESULT_FILE_PLACEHOLDER = '(no result file)'


def result_file_tail(result_file: str | None, max_lines: int = 20) -> str:
    """Return the last *max_lines* lines of *result_file*, fail-soft.

    A None path, a missing file, or any OSError while reading (e.g. the
    path is a directory, or a permissions error) degrades to a defined
    placeholder rather than raising -- a view must not crash on a result
    file whose shape it doesn't control (fail-soft, PRD §2).
    """
    if not result_file:
        return _NO_RESULT_FILE_PLACEHOLDER
    try:
        text = Path(result_file).read_text()
    except OSError:
        logger.warning('result_file_tail: could not read %s', result_file, exc_info=True)
        return _NO_RESULT_FILE_PLACEHOLDER
    lines = text.splitlines()
    return '\n'.join(lines[-max_lines:])


def render_detail(record: SessionRecord, all_records: list[SessionRecord], now: datetime) -> str:
    """Render *record*'s full detail as plain text.

    Composes title/glyph/age (reusing session_table's pure formatters),
    project, task/escalation ids, the full pending-question text (if any),
    the parent slug (if any), every child slug (records whose
    parent_session_id == record.session_slug), and the result-file tail.
    Pure -- no clock read (now is injected), no writes.
    """
    children = [r.session_slug for r in all_records if r.parent_session_id == record.session_slug]
    lines = [
        f'{state_glyph(record.status)} {format_title(record)} ({format_age(record.start_ts, now)})',
        f'project: {record.project}',
        f'task_id: {record.task_id}',
        f'escalation_id: {record.escalation_id}',
        f'parent: {record.parent_session_id}',
        f'children: {", ".join(children) if children else "(none)"}',
    ]
    if record.question is not None:
        lines.append(f'question: {record.question.text}')
    lines.append('--- result ---')
    lines.append(result_file_tail(record.result_file))
    return '\n'.join(lines)
