"""Tests for cockpit.panes.detail_pane — result-file tail + detail rendering.

result_file_tail is a fail-soft tail reader (PRD §2: a missing/unreadable
result file must degrade, never raise). render_detail is a pure,
injected-`now` text rendering of a SessionRecord's full detail — question,
task/escalation ids, parent, children, and the result-file tail. Mirrors
test_session_table.py's _make_record convention.
"""

from __future__ import annotations

from datetime import UTC, datetime

from orchestrator import session_registry as sr


def _make_record(**overrides):
    fields: dict = {
        'session_slug': 'unblock-df-2085-4242',
        'status': sr.Status.RUNNING,
        'title': 'unblock:df#2085 slug',
        'role': 'unblock',
        'project': 'df',
        'task_id': '2085',
        'start_ts': '2026-07-07T00:00:00+00:00',
    }
    fields.update(overrides)
    return sr.SessionRecord(**fields)


class TestResultFileTail:
    def test_returns_only_last_max_lines(self, tmp_path):
        from cockpit.panes.detail_pane import result_file_tail

        result_file = tmp_path / 'result.md'
        lines = [f'line-{i}' for i in range(30)]
        result_file.write_text('\n'.join(lines) + '\n')

        tail = result_file_tail(str(result_file), max_lines=5)

        assert tail.splitlines() == lines[-5:]

    def test_none_path_is_placeholder(self):
        from cockpit.panes.detail_pane import result_file_tail

        assert result_file_tail(None) == '(no result file)'

    def test_missing_path_is_placeholder(self, tmp_path):
        from cockpit.panes.detail_pane import result_file_tail

        missing = tmp_path / 'does-not-exist.md'

        assert result_file_tail(str(missing)) == '(no result file)'

    def test_unreadable_path_is_placeholder(self, tmp_path):
        """A path that exists but can't be read as text (e.g. a directory) degrades too."""
        from cockpit.panes.detail_pane import result_file_tail

        a_directory = tmp_path / 'a-directory'
        a_directory.mkdir()

        assert result_file_tail(str(a_directory)) == '(no result file)'


class TestRenderDetail:
    def test_renders_question_task_escalation_parent_and_children(self):
        from cockpit.panes.detail_pane import render_detail

        parent = _make_record(session_slug='parent-1')
        record = _make_record(
            session_slug='child-1',
            task_id='2085',
            escalation_id='esc-99',
            parent_session_id='parent-1',
            question=sr.Question(
                text='Which port should the server bind to?',
                asked_at='2026-07-07T00:00:00+00:00',
            ),
        )
        child_a = _make_record(session_slug='grandchild-a', parent_session_id='child-1')
        child_b = _make_record(session_slug='grandchild-b', parent_session_id='child-1')

        all_records = [parent, record, child_a, child_b]
        now = datetime(2026, 7, 7, 0, 5, 0, tzinfo=UTC)

        rendered = render_detail(record, all_records, now)

        assert 'Which port should the server bind to?' in rendered
        assert '2085' in rendered
        assert 'esc-99' in rendered
        assert 'parent-1' in rendered
        assert 'grandchild-a' in rendered
        assert 'grandchild-b' in rendered

    def test_absent_ids_render_the_shared_placeholder_and_never_the_word_none(self):
        """The same fail-soft rendering render_decision_detail gives an absent id:
        both renderers spell it with placeholders.ABSENT_PLACEHOLDER, so a session
        with no task_id never reads as though it carried one called 'None'."""
        from cockpit.panes.detail_pane import render_detail

        record = _make_record(task_id=None, escalation_id=None, parent_session_id=None)

        rendered = render_detail(record, [record], datetime(2026, 7, 7, tzinfo=UTC))

        assert 'task_id: (none)' in rendered
        assert 'escalation_id: (none)' in rendered
        assert 'parent: (none)' in rendered
        assert 'children: (none)' in rendered
        assert 'None' not in rendered

    def test_result_file_tail_is_included(self, tmp_path):
        from cockpit.panes.detail_pane import render_detail

        result_file = tmp_path / 'result.md'
        result_file.write_text('the result content')
        record = _make_record(result_file=str(result_file))

        rendered = render_detail(record, [record], datetime(2026, 7, 7, tzinfo=UTC))

        assert 'the result content' in rendered

    def test_no_question_and_no_children_does_not_raise(self):
        from cockpit.panes.detail_pane import render_detail

        record = _make_record(question=None)

        rendered = render_detail(record, [record], datetime(2026, 7, 7, tzinfo=UTC))

        assert isinstance(rendered, str)


def _make_decision(**overrides):
    fields: dict = {
        'id': 'dec-1',
        'project': 'df',
        'text': 'Should the reaper close this escalation?',
        'filed_at': '2026-07-07T00:00:00+00:00',
    }
    fields.update(overrides)
    return sr.DecisionRecord(**fields)


class TestRenderDecisionDetail:
    def test_renders_ids_severity_state_and_age(self):
        from cockpit.panes.detail_pane import render_decision_detail

        decision = _make_decision(
            id='dec-42',
            project='df',
            task_id='2085',
            escalation_id='esc-99',
            severity='blocking',
            state=sr.DecisionState.OPEN,
        )
        now = datetime(2026, 7, 7, 0, 5, 0, tzinfo=UTC)

        rendered = render_decision_detail(decision, [], now)

        assert 'dec-42' in rendered
        assert 'df' in rendered
        assert '2085' in rendered
        assert 'esc-99' in rendered
        assert 'blocking' in rendered
        assert 'open' in rendered
        assert '5m' in rendered

    def test_question_is_rendered_in_full_and_uncollapsed(self):
        """The whole point of the pane: the queue row truncates at 60 chars, this must not."""
        from cockpit.panes.detail_pane import render_decision_detail

        text = 'A' * 100 + '\n' + 'B' * 99
        assert len(text) == 200
        decision = _make_decision(text=text)

        rendered = render_decision_detail(decision, [], datetime(2026, 7, 7, tzinfo=UTC))

        assert text in rendered
        assert '\n' + 'B' * 99 in rendered

    def test_session_id_naming_a_scanned_session_renders_that_slug(self):
        from cockpit.panes.detail_pane import render_decision_detail

        session = _make_record(session_slug='unblock-df-2085-4242')
        decision = _make_decision(session_id='unblock-df-2085-4242')

        rendered = render_decision_detail(
            decision, [session], datetime(2026, 7, 7, tzinfo=UTC)
        )

        assert 'unblock-df-2085-4242' in rendered

    def test_unresolvable_session_id_is_shown_and_marked_unresolved(self):
        """The live shape today: session_id is a watcher lease token, not a slug
        (task 4237). An operator must be able to tell "no link" from "broken link"."""
        from cockpit.panes.detail_pane import render_decision_detail

        session = _make_record(session_slug='unblock-df-2085-4242')
        decision = _make_decision(session_id='watcher-lease-abc123')

        rendered = render_decision_detail(
            decision, [session], datetime(2026, 7, 7, tzinfo=UTC)
        )

        assert 'watcher-lease-abc123' in rendered
        assert 'unresolved' in rendered

    def test_no_session_id_renders_a_placeholder_not_the_word_none(self):
        from cockpit.panes.detail_pane import render_decision_detail

        decision = _make_decision(session_id=None)

        rendered = render_decision_detail(decision, [], datetime(2026, 7, 7, tzinfo=UTC))

        session_line = next(
            line for line in rendered.splitlines() if line.startswith('session:')
        )
        assert '(none)' in session_line
        assert 'None' not in session_line

    def test_empty_or_unparseable_filed_at_degrades_to_the_age_placeholder(self):
        """Asserted on the filed LINE, not on the whole render: a bare "'?' in
        rendered" is satisfied by any question ending in a question mark, so it
        would pass whatever format_age returned here."""
        from cockpit.panes.detail_pane import render_decision_detail

        now = datetime(2026, 7, 7, tzinfo=UTC)

        for filed_at in ('', 'not-a-timestamp'):
            rendered = render_decision_detail(_make_decision(filed_at=filed_at), [], now)

            filed_line = next(
                line for line in rendered.splitlines() if line.startswith('filed:')
            )
            assert filed_line.endswith('(?)')

    def test_absent_ids_render_a_placeholder_and_never_the_word_none(self):
        from cockpit.panes.detail_pane import render_decision_detail

        decision = _make_decision(task_id=None, escalation_id=None, session_id=None)

        rendered = render_decision_detail(decision, [], datetime(2026, 7, 7, tzinfo=UTC))

        assert 'task_id: (none)' in rendered
        assert 'escalation_id: (none)' in rendered
        assert 'None' not in rendered

    def test_unset_severity_and_unrecognized_state_still_render(self):
        """Total over any record shape the registry hands it -- severity defaults
        to '' (unknown) and state round-trips an unrecognized wire value."""
        from cockpit.panes.detail_pane import render_decision_detail

        decision = _make_decision(severity='', state='some-future-state')

        rendered = render_decision_detail(decision, [], datetime(2026, 7, 7, tzinfo=UTC))

        assert 'some-future-state' in rendered

    def test_options_are_rendered_only_when_present(self):
        from cockpit.panes.detail_pane import render_decision_detail

        now = datetime(2026, 7, 7, tzinfo=UTC)

        with_options = render_decision_detail(_make_decision(options=['a', 'b']), [], now)
        assert 'options:' in with_options
        assert 'a' in with_options
        assert 'b' in with_options

        for empty in (None, []):
            rendered = render_decision_detail(_make_decision(options=empty), [], now)

            assert 'options:' not in rendered


class TestDetailPaneShowDecision:
    """The widget's second mutator: show_decision renders via render_decision_detail."""

    def test_show_decision_renders_render_decision_detail_verbatim(self):
        from cockpit.panes.detail_pane import DetailPane, render_decision_detail

        decision = _make_decision(task_id='2085', escalation_id='esc-99')
        session = _make_record()
        now = datetime(2026, 7, 7, 0, 5, 0, tzinfo=UTC)
        pane = DetailPane()

        pane.show_decision(decision, [session], now)

        assert pane.rendered_text == render_decision_detail(decision, [session], now)

