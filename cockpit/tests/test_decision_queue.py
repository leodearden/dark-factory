"""Tests for cockpit.panes.decision_queue — pure scoring adapters, row
formatting, and order_queue's queue-build/sort (Fleet Cockpit C5b, PRD §9).

Pure, deterministic unit tests only -- no Textual import, no pilot. The
DecisionQueue(DataTable) widget itself is covered by test_app.py's pilot
tests (a later C5b step). Mirrors test_registry_reader.py/test_app.py's
_make_record convention: a fields dict + .update(overrides).
"""

from __future__ import annotations

from datetime import UTC, datetime

from orchestrator import session_registry as sr

_NOW = datetime(2026, 7, 11, tzinfo=UTC)


def _make_decision(**overrides):
    fields: dict = {
        'id': 'dec-1',
        'project': 'df',
        'text': 'Which port?',
        'filed_at': '2026-07-07T00:00:00+00:00',
    }
    fields.update(overrides)
    return sr.DecisionRecord(**fields)


def _make_session(**overrides):
    """Mirrors test_app.py/test_registry_reader.py's _make_record convention."""
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


class TestScoringAdapters:
    def test_decision_to_scoring_item_maps_fields(self):
        from cockpit.panes.decision_queue import decision_to_scoring_item
        from cockpit.priority import ScoringItem

        decision = _make_decision(
            project='df',
            manual_boost=3,
            state=sr.DecisionState.OPEN,
            filed_at='2026-07-01T00:00:00+00:00',
        )

        item = decision_to_scoring_item(decision, now=_NOW)

        assert isinstance(item, ScoringItem)
        assert item.project == 'df'
        assert item.manual_boost == 3
        assert item.state == 'open'
        assert item.severity == ''
        assert item.category == ''
        assert item.filed_at == datetime(2026, 7, 1, tzinfo=UTC)

    def test_decision_to_scoring_item_state_copied_through_for_non_open(self):
        """state is copied through verbatim -- order_queue is what filters to 'open'."""
        from cockpit.panes.decision_queue import decision_to_scoring_item

        decision = _make_decision(state=sr.DecisionState.DROPPED)

        item = decision_to_scoring_item(decision, now=_NOW)

        assert item.state == 'dropped'

    def test_decision_to_scoring_item_empty_filed_at_degrades_to_now(self):
        from cockpit.panes.decision_queue import decision_to_scoring_item

        decision = _make_decision(filed_at='')

        item = decision_to_scoring_item(decision, now=_NOW)

        assert item.filed_at == _NOW

    def test_decision_to_scoring_item_unparseable_filed_at_degrades_to_now(self):
        from cockpit.panes.decision_queue import decision_to_scoring_item

        decision = _make_decision(filed_at='not-a-timestamp')

        item = decision_to_scoring_item(decision, now=_NOW)

        assert item.filed_at == _NOW

    def test_session_to_scoring_item_awaiting_input_maps_to_open(self):
        from cockpit.panes.decision_queue import session_to_scoring_item

        session = _make_session(
            status=sr.Status.AWAITING_INPUT,
            project='df',
            question=sr.Question(text='Which port?', asked_at='2026-07-05T00:00:00+00:00'),
        )

        item = session_to_scoring_item(session, now=_NOW)

        assert item.project == 'df'
        assert item.state == 'open'
        assert item.manual_boost == 0
        assert item.severity == ''
        assert item.category == ''
        assert item.filed_at == datetime(2026, 7, 5, tzinfo=UTC)

    def test_session_to_scoring_item_falls_back_to_start_ts_without_question(self):
        from cockpit.panes.decision_queue import session_to_scoring_item

        session = _make_session(
            status=sr.Status.AWAITING_INPUT,
            question=None,
            start_ts='2026-07-03T00:00:00+00:00',
        )

        item = session_to_scoring_item(session, now=_NOW)

        assert item.filed_at == datetime(2026, 7, 3, tzinfo=UTC)

    def test_session_to_scoring_item_empty_start_ts_degrades_to_now(self):
        from cockpit.panes.decision_queue import session_to_scoring_item

        session = _make_session(status=sr.Status.AWAITING_INPUT, question=None, start_ts='')

        item = session_to_scoring_item(session, now=_NOW)

        assert item.filed_at == _NOW

    def test_session_to_scoring_item_unparseable_start_ts_degrades_to_now(self):
        from cockpit.panes.decision_queue import session_to_scoring_item

        session = _make_session(
            status=sr.Status.AWAITING_INPUT, question=None, start_ts='not-a-timestamp'
        )

        item = session_to_scoring_item(session, now=_NOW)

        assert item.filed_at == _NOW
