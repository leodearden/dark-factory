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
