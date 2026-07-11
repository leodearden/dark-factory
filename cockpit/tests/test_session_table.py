"""Tests for cockpit.panes.session_table — pure glyph/title/age/order helpers.

Every helper here is pure with an injected `now` where relevant, so the
bulk of the session table's behavior is deterministically testable without
a running Textual app (PRD §9 C5a). Fail-soft is a hard constraint (PRD
§2): a foreign status, an empty/unparseable start_ts, or an empty
role/project must degrade gracefully, never raise.
"""

from __future__ import annotations

from datetime import UTC, datetime

from orchestrator import session_registry as sr


def _make_record(**overrides):
    """Build a SessionRecord with sane defaults; overrides tweak individual fields.

    Mirrors test_registry_reader.py's _make_record convention.
    """
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


class TestStateGlyph:
    def test_awaiting_input_is_blocked_glyph(self):
        from cockpit.panes.session_table import state_glyph

        assert state_glyph(sr.Status.AWAITING_INPUT) == '⏸'

    def test_running_and_launching_are_working_glyph(self):
        from cockpit.panes.session_table import state_glyph

        assert state_glyph(sr.Status.RUNNING) == '⚙'
        assert state_glyph(sr.Status.LAUNCHING) == '⚙'

    def test_idle_is_idle_glyph(self):
        from cockpit.panes.session_table import state_glyph

        assert state_glyph(sr.Status.IDLE) == '✓'

    def test_exited_and_failed_to_start_are_dead_glyph(self):
        from cockpit.panes.session_table import state_glyph

        assert state_glyph(sr.Status.EXITED) == '☠'
        assert state_glyph(sr.Status.FAILED_TO_START) == '☠'

    def test_accepts_wire_string_too(self):
        from cockpit.panes.session_table import state_glyph

        assert state_glyph('awaiting-input') == '⏸'

    def test_unknown_status_degrades_to_fallback_glyph(self):
        from cockpit.panes.session_table import state_glyph

        assert state_glyph('some-foreign-status') == '?'


class TestFormatTitle:
    def test_role_project_and_task_id(self):
        from cockpit.panes.session_table import format_title

        record = _make_record(role='unblock', project='df', task_id='2085')

        assert format_title(record) == 'unblock:df#2085'

    def test_no_task_id_omits_hash_segment(self):
        from cockpit.panes.session_table import format_title

        record = _make_record(role='unblock', project='df', task_id=None)

        assert format_title(record) == 'unblock:df'

    def test_empty_role_and_project_do_not_raise(self):
        from cockpit.panes.session_table import format_title

        record = _make_record(role='', project='', task_id=None)

        assert isinstance(format_title(record), str)


class TestFormatAge:
    def test_seconds(self):
        from cockpit.panes.session_table import format_age

        start_ts = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC).isoformat()
        now = datetime(2026, 7, 7, 12, 0, 45, tzinfo=UTC)

        assert format_age(start_ts, now) == '45s'

    def test_minutes(self):
        from cockpit.panes.session_table import format_age

        start_ts = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC).isoformat()
        now = datetime(2026, 7, 7, 12, 5, 0, tzinfo=UTC)

        assert format_age(start_ts, now) == '5m'

    def test_hours(self):
        from cockpit.panes.session_table import format_age

        start_ts = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC).isoformat()
        now = datetime(2026, 7, 7, 15, 0, 0, tzinfo=UTC)

        assert format_age(start_ts, now) == '3h'

    def test_days(self):
        from cockpit.panes.session_table import format_age

        start_ts = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC).isoformat()
        now = datetime(2026, 7, 9, 12, 0, 0, tzinfo=UTC)

        assert format_age(start_ts, now) == '2d'

    def test_empty_start_ts_is_placeholder(self):
        from cockpit.panes.session_table import format_age

        now = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)

        assert format_age('', now) == '?'

    def test_unparseable_start_ts_is_placeholder(self):
        from cockpit.panes.session_table import format_age

        now = datetime(2026, 7, 7, 12, 0, 0, tzinfo=UTC)

        assert format_age('not-a-timestamp', now) == '?'


class TestCountOutstandingChildren:
    def test_counts_only_non_terminal_children(self):
        from cockpit.panes.session_table import count_outstanding_children

        parent = _make_record(session_slug='parent-1')
        running_child = _make_record(
            session_slug='child-running', parent_session_id='parent-1', status=sr.Status.RUNNING
        )
        awaiting_child = _make_record(
            session_slug='child-awaiting',
            parent_session_id='parent-1',
            status=sr.Status.AWAITING_INPUT,
        )
        exited_child = _make_record(
            session_slug='child-exited', parent_session_id='parent-1', status=sr.Status.EXITED
        )
        failed_child = _make_record(
            session_slug='child-failed',
            parent_session_id='parent-1',
            status=sr.Status.FAILED_TO_START,
        )
        all_records = [parent, running_child, awaiting_child, exited_child, failed_child]

        assert count_outstanding_children('parent-1', all_records) == 2

    def test_no_children_is_zero(self):
        from cockpit.panes.session_table import count_outstanding_children

        parent = _make_record(session_slug='lonely-parent')

        assert count_outstanding_children('lonely-parent', [parent]) == 0

    def test_children_of_a_different_parent_not_counted(self):
        from cockpit.panes.session_table import count_outstanding_children

        parent = _make_record(session_slug='parent-1')
        other_parent = _make_record(session_slug='parent-2')
        unrelated_child = _make_record(
            session_slug='child-of-other',
            parent_session_id='parent-2',
            status=sr.Status.RUNNING,
        )

        all_records = [parent, other_parent, unrelated_child]

        assert count_outstanding_children('parent-1', all_records) == 0


class TestOrderSessions:
    def test_blocked_sorts_above_all_else_regardless_of_age(self):
        from cockpit.panes.session_table import order_sessions

        old_running = _make_record(
            session_slug='old-running',
            status=sr.Status.RUNNING,
            start_ts=datetime(2020, 1, 1, tzinfo=UTC).isoformat(),
        )
        young_awaiting = _make_record(
            session_slug='young-awaiting',
            status=sr.Status.AWAITING_INPUT,
            start_ts=datetime(2026, 7, 7, tzinfo=UTC).isoformat(),
        )

        ordered = order_sessions([old_running, young_awaiting])

        assert [r.session_slug for r in ordered] == ['young-awaiting', 'old-running']

    def test_state_rank_ordering(self):
        from cockpit.panes.session_table import order_sessions

        awaiting = _make_record(session_slug='s-awaiting', status=sr.Status.AWAITING_INPUT)
        running = _make_record(session_slug='s-running', status=sr.Status.RUNNING)
        launching = _make_record(session_slug='s-launching', status=sr.Status.LAUNCHING)
        idle = _make_record(session_slug='s-idle', status=sr.Status.IDLE)
        exited = _make_record(session_slug='s-exited', status=sr.Status.EXITED)
        failed = _make_record(session_slug='s-failed', status=sr.Status.FAILED_TO_START)

        ordered = order_sessions([failed, exited, idle, launching, running, awaiting])
        ranks = [r.session_slug for r in ordered]

        assert ranks.index('s-awaiting') < ranks.index('s-running')
        assert ranks.index('s-awaiting') < ranks.index('s-launching')
        assert ranks.index('s-running') < ranks.index('s-idle')
        assert ranks.index('s-launching') < ranks.index('s-idle')
        assert ranks.index('s-idle') < ranks.index('s-exited')
        assert ranks.index('s-idle') < ranks.index('s-failed')

    def test_within_rank_oldest_start_ts_sorts_first(self):
        from cockpit.panes.session_table import order_sessions

        newer = _make_record(
            session_slug='newer',
            status=sr.Status.RUNNING,
            start_ts=datetime(2026, 7, 7, tzinfo=UTC).isoformat(),
        )
        older = _make_record(
            session_slug='older',
            status=sr.Status.RUNNING,
            start_ts=datetime(2026, 7, 1, tzinfo=UTC).isoformat(),
        )

        ordered = order_sessions([newer, older])

        assert [r.session_slug for r in ordered] == ['older', 'newer']

    def test_empty_start_ts_does_not_raise_and_sorts_last_within_rank(self):
        from cockpit.panes.session_table import order_sessions

        has_ts = _make_record(
            session_slug='has-ts',
            status=sr.Status.RUNNING,
            start_ts=datetime(2026, 7, 7, tzinfo=UTC).isoformat(),
        )
        no_ts = _make_record(session_slug='no-ts', status=sr.Status.RUNNING, start_ts='')

        ordered = order_sessions([no_ts, has_ts])

        assert [r.session_slug for r in ordered] == ['has-ts', 'no-ts']
