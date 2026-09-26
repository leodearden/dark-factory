"""cockpit.panes.detail_pane — pure detail rendering + the DetailPane widget.

Two explicitly-typed render entry points, one per record kind the cockpit
shows: render_detail for a SessionRecord (title, status, age, ids,
question, parent/children, result-file tail) and render_decision_detail
for a DecisionRecord (ids, severity/state, age, options, full question).
Neither is a union over the other -- each reads only its own record's
fields, so a call site is type-checked against the kind it actually holds.

Fail-soft throughout (PRD §2), and on the same terms in both renderers: a
missing/unreadable result file degrades to a placeholder rather than
raising, a record with no question/children still renders, and an absent
id renders as the shared
cockpit/src/cockpit/panes/placeholders.py::ABSENT_PLACEHOLDER rather than
the literal string 'None'.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from orchestrator.session_registry import DecisionRecord, SessionRecord
from textual.widgets import Static

from cockpit.panes.placeholders import ABSENT_PLACEHOLDER
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
        f'task_id: {record.task_id or ABSENT_PLACEHOLDER}',
        f'escalation_id: {record.escalation_id or ABSENT_PLACEHOLDER}',
        f'parent: {record.parent_session_id or ABSENT_PLACEHOLDER}',
        f'children: {", ".join(children) if children else ABSENT_PLACEHOLDER}',
    ]
    if record.question is not None:
        lines.append(f'question: {record.question.text}')
    lines.append('--- result ---')
    lines.append(result_file_tail(record.result_file))
    return '\n'.join(lines)


def render_decision_detail(
    decision: DecisionRecord, sessions: Sequence[SessionRecord], now: datetime
) -> str:
    """Render *decision*'s full detail as plain text.

    The sibling of render_detail for the other record kind, not a widening
    of it: composes the decision's identity (id, project, task/escalation
    ids), its severity/state, and its filed timestamp with the age
    format_age gives the queue row, then the question text LAST -- verbatim,
    neither collapsed nor truncated (unlike decision_queue._one_line_question,
    which is why the pane exists). Question last so a long or multi-line
    question scrolls off the bottom rather than pushing the ids out of view.
    Pure -- no clock read (now is injected), no writes.

    The linked session resolves decision.session_id against *sessions* by
    session_slug. An UNRESOLVED session_id is the expected shape today, not
    a bug in this renderer: session_id carries a C8 watcher's lease token
    rather than a session_slug (task 4237). It is therefore rendered raw and
    marked unresolved, so an operator can tell a broken link from no link.
    """
    lines = [
        f'decision_id: {decision.id}',
        f'project: {decision.project}',
        f'task_id: {decision.task_id or ABSENT_PLACEHOLDER}',
        f'escalation_id: {decision.escalation_id or ABSENT_PLACEHOLDER}',
        f'severity: {decision.severity}',
        f'state: {decision.state}',
        f'filed: {decision.filed_at} ({format_age(decision.filed_at, now)})',
        f'session: {_linked_session(decision.session_id, sessions)}',
    ]
    if decision.options:
        lines.append(f'options: {", ".join(decision.options)}')
    lines.append(f'question: {decision.text}')
    return '\n'.join(lines)


def _linked_session(session_id: str | None, sessions: Sequence[SessionRecord]) -> str:
    """Render *session_id* as a linked-session value -- see render_decision_detail."""
    if not session_id:
        return ABSENT_PLACEHOLDER
    session = next((s for s in sessions if s.session_slug == session_id), None)
    if session is None:
        return f'{session_id} (unresolved)'
    return session.session_slug


_NO_SELECTION_PLACEHOLDER = '(no session selected)'


class DetailPane(Static):
    """Renders one record's full detail -- a session's or a decision's.

    Two mutators, one per record kind: show_record() renders a
    SessionRecord via render_detail() (or the no-selection placeholder,
    which is the session-flavoured empty state), and show_decision()
    renders a DecisionRecord via render_decision_detail(). Each writes both
    the widget's display and the plain-text `rendered_text` attribute, so
    callers/tests can read back what's currently shown without parsing the
    rendered widget.

    The pane deliberately holds no notion of WHICH TABLE put a record
    here: the decision queue's own session rows render through show_record
    exactly as the session table's do, so the rendered kind cannot tell the
    two apart. That arbitration lives in
    cockpit/src/cockpit/app.py::DetailOwner, keyed on the table whose cursor
    last moved.
    """

    DEFAULT_CSS = """
    DetailPane {
        width: 1fr;
        height: 1fr;
        overflow-y: auto;
        padding: 1 2;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rendered_text = _NO_SELECTION_PLACEHOLDER

    def show_record(
        self,
        record: SessionRecord | None,
        all_records: list[SessionRecord],
        now: datetime,
    ) -> None:
        """Render *record*'s detail, or a placeholder when *record* is None (no selection)."""
        self.rendered_text = (
            render_detail(record, all_records, now)
            if record is not None
            else _NO_SELECTION_PLACEHOLDER
        )
        self.update(self.rendered_text)

    def show_decision(
        self,
        decision: DecisionRecord,
        sessions: list[SessionRecord],
        now: datetime,
    ) -> None:
        """Render *decision*'s detail.

        Takes a non-optional DecisionRecord: "nothing selected" stays
        show_record(None, ...)'s single responsibility, so the pane has
        exactly one empty-state path.
        """
        self.rendered_text = render_decision_detail(decision, sessions, now)
        self.update(self.rendered_text)
