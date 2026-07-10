"""Tests for cockpit.panes.session_table — pure glyph/title/age/order helpers.

Every helper here is pure with an injected `now` where relevant, so the
bulk of the session table's behavior is deterministically testable without
a running Textual app (PRD §9 C5a). Fail-soft is a hard constraint (PRD
§2): a foreign status, an empty/unparseable start_ts, or an empty
role/project must degrade gracefully, never raise.
"""

from __future__ import annotations

from orchestrator import session_registry as sr


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
