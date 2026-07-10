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
