"""Tests for cockpit.panes.decision_queue — pure scoring adapters, row
formatting, and order_queue's queue-build/sort (Fleet Cockpit C5b, PRD §9).

Pure, deterministic unit tests only -- no Textual import, no pilot. The
DecisionQueue(DataTable) widget itself is covered by test_app.py's pilot
tests (a later C5b step). Mirrors test_registry_reader.py/test_app.py's
_make_record convention: a fields dict + .update(overrides).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

from orchestrator import session_registry as sr

_NOW = datetime(2026, 7, 11, tzinfo=UTC)


def _make_row_item(**overrides):
    """A duck-typed stand-in exposing exactly format_queue_row's attribute
    contract (score/filed_at/project/task_id/question) -- mirrors
    ScoringItem/DisplayTarget's own "structural, not isinstance" convention.
    QueueItem itself doesn't exist yet at this step (see TestOrderQueue
    below), so format_queue_row must not require one.
    """
    fields: dict = {
        'score': 1.0,
        'filed_at': datetime(2026, 7, 1, tzinfo=UTC),
        'project': 'df',
        'task_id': '2085',
        'question': 'Which port?',
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


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


class TestFormatQueueRow:
    def test_renders_four_column_row(self):
        from cockpit.panes.decision_queue import format_queue_row

        item = _make_row_item(score=2.567, project='df', task_id='2085', question='Which port?')
        now = datetime(2026, 7, 8, tzinfo=UTC)

        row = format_queue_row(item, now)

        assert len(row) == 4
        score_col, _age_col, project_task_col, question_col = row
        assert score_col == '2.6'
        assert project_task_col == 'df#2085'
        assert question_col == 'Which port?'

    def test_age_column_matches_session_table_format_age(self):
        from cockpit.panes.decision_queue import format_queue_row
        from cockpit.panes.session_table import format_age

        filed_at = datetime(2026, 7, 1, tzinfo=UTC)
        now = datetime(2026, 7, 8, tzinfo=UTC)
        item = _make_row_item(filed_at=filed_at)

        _score_col, age_col, _project_task_col, _question_col = format_queue_row(item, now)

        assert age_col == format_age(filed_at.isoformat(), now)

    def test_task_id_none_omits_hash_segment(self):
        from cockpit.panes.decision_queue import format_queue_row

        item = _make_row_item(project='df', task_id=None)
        now = datetime(2026, 7, 8, tzinfo=UTC)

        _, _, project_task_col, _ = format_queue_row(item, now)

        assert project_task_col == 'df'

    def test_question_newlines_collapsed_to_single_line(self):
        from cockpit.panes.decision_queue import format_queue_row

        item = _make_row_item(question='Which port?\nAlso which host?')
        now = datetime(2026, 7, 8, tzinfo=UTC)

        _, _, _, question_col = format_queue_row(item, now)

        assert '\n' not in question_col
        assert question_col == 'Which port? Also which host?'

    def test_long_question_truncated_with_ellipsis(self):
        from cockpit.panes.decision_queue import format_queue_row

        item = _make_row_item(question='x' * 200)
        now = datetime(2026, 7, 8, tzinfo=UTC)

        _, _, _, question_col = format_queue_row(item, now)

        assert len(question_col) < 200
        assert question_col.endswith('…')

    def test_empty_question_degrades_to_placeholder_not_raise(self):
        from cockpit.panes.decision_queue import format_queue_row

        item = _make_row_item(question='')
        now = datetime(2026, 7, 8, tzinfo=UTC)

        _, _, _, question_col = format_queue_row(item, now)

        assert question_col != ''

    def test_none_question_does_not_raise(self):
        from cockpit.panes.decision_queue import format_queue_row

        item = _make_row_item(question=None)
        now = datetime(2026, 7, 8, tzinfo=UTC)

        _, _, _, question_col = format_queue_row(item, now)

        assert question_col != ''


class TestOrderQueue:
    def test_includes_only_open_decisions_and_awaiting_sessions(self):
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import Priorities

        open_decision = _make_decision(id='dec-open', state=sr.DecisionState.OPEN)
        dropped_decision = _make_decision(id='dec-dropped', state=sr.DecisionState.DROPPED)
        answered_decision = _make_decision(id='dec-answered', state=sr.DecisionState.ANSWERED)
        awaiting_session = _make_session(session_slug='awaiting-1', status=sr.Status.AWAITING_INPUT)
        running_session = _make_session(session_slug='running-1', status=sr.Status.RUNNING)

        items = order_queue(
            [open_decision, dropped_decision, answered_decision],
            [awaiting_session, running_session],
            Priorities.default(),
            _NOW,
        )

        assert len(items) == 2
        assert {item.decision_id for item in items if item.kind == 'decision'} == {'dec-open'}
        session_items = [item for item in items if item.kind == 'session']
        assert len(session_items) == 1

    def test_returns_queue_item_instances(self):
        from cockpit.panes.decision_queue import QueueItem, order_queue
        from cockpit.priority import Priorities

        decision = _make_decision(state=sr.DecisionState.OPEN)

        items = order_queue([decision], [], Priorities.default(), _NOW)

        assert len(items) == 1
        assert isinstance(items[0], QueueItem)

    def test_higher_manual_boost_decision_precedes_lower(self):
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import Priorities

        low = _make_decision(id='dec-low', manual_boost=0)
        high = _make_decision(id='dec-high', manual_boost=5)

        items = order_queue([low, high], [], Priorities.default(), _NOW)

        assert [item.decision_id for item in items] == ['dec-high', 'dec-low']

    def test_pure_deterministic_across_calls(self):
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import Priorities

        decisions = [
            _make_decision(id='dec-a', manual_boost=1),
            _make_decision(id='dec-b', manual_boost=3),
        ]
        sessions = [_make_session(session_slug='s-1', status=sr.Status.AWAITING_INPUT)]

        first = order_queue(decisions, sessions, Priorities.default(), _NOW)
        second = order_queue(decisions, sessions, Priorities.default(), _NOW)

        assert [item.key for item in first] == [item.key for item in second]
        assert [item.score for item in first] == [item.score for item in second]

    def test_session_item_target_resolved_from_own_display(self):
        from cockpit.backends import DisplayTarget
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import Priorities

        display = sr.Display(kind='wm', wm_title='unblock:df#2085 slug')
        session = _make_session(
            session_slug='awaiting-1', status=sr.Status.AWAITING_INPUT, display=display
        )

        items = order_queue([], [session], Priorities.default(), _NOW)

        assert len(items) == 1
        assert items[0].target == DisplayTarget(kind='wm', wm_title='unblock:df#2085 slug')

    def test_decision_item_target_resolved_from_linked_session_display(self):
        from cockpit.backends import DisplayTarget
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import Priorities

        display = sr.Display(kind='tmux', tmux_target='fleet-df:3')
        linked_session = _make_session(
            session_slug='linked-1', status=sr.Status.RUNNING, display=display
        )
        decision = _make_decision(session_id='linked-1', state=sr.DecisionState.OPEN)

        items = order_queue([decision], [linked_session], Priorities.default(), _NOW)

        assert len(items) == 1
        assert items[0].target == DisplayTarget(kind='tmux', tmux_target='fleet-df:3')

    def test_decision_item_target_none_when_session_unresolvable(self):
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import Priorities

        decision = _make_decision(session_id='no-such-session', state=sr.DecisionState.OPEN)

        items = order_queue([decision], [], Priorities.default(), _NOW)

        assert len(items) == 1
        assert items[0].target is None

    def test_decision_item_target_none_when_no_session_id(self):
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import Priorities

        decision = _make_decision(session_id=None, state=sr.DecisionState.OPEN)

        items = order_queue([decision], [], Priorities.default(), _NOW)

        assert items[0].target is None

    def test_boosts_overlay_can_flip_order(self):
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import Priorities

        low = _make_decision(id='dec-low', manual_boost=0)
        high = _make_decision(id='dec-high', manual_boost=0)

        baseline = order_queue([low, high], [], Priorities.default(), _NOW)
        assert [item.decision_id for item in baseline] == ['dec-high', 'dec-low']

        boosted = order_queue(
            [low, high], [], Priorities.default(), _NOW, boosts={'decision:dec-low': 10}
        )

        assert boosted[0].decision_id == 'dec-low'

    def test_deferred_overlay_resets_effective_age(self):
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import Priorities

        older = _make_decision(id='dec-older', filed_at='2020-01-01T00:00:00+00:00')
        newer = _make_decision(id='dec-newer', filed_at=_NOW.isoformat())

        baseline = order_queue([older, newer], [], Priorities.default(), _NOW)
        assert baseline[0].decision_id == 'dec-older'

        deferred = order_queue(
            [older, newer],
            [],
            Priorities.default(),
            _NOW,
            deferred={'decision:dec-older': _NOW},
        )

        assert deferred[0].decision_id == 'dec-newer'

    def test_handling_flag_reflected_on_item(self):
        from cockpit.panes.decision_queue import order_queue
        from cockpit.priority import Priorities

        decision = _make_decision(id='dec-1', state=sr.DecisionState.OPEN)

        items = order_queue([decision], [], Priorities.default(), _NOW, handling={'decision:dec-1'})

        assert items[0].handling is True


class TestKnownProjectRoots:
    def test_distinct_nonempty_cwds_deduped_and_sorted(self):
        from cockpit.panes.decision_queue import known_project_roots

        sessions = [
            _make_session(session_slug='s-1', cwd='/home/leo/src/dark-factory'),
            _make_session(session_slug='s-2', cwd='/home/leo/src/other-project'),
            _make_session(session_slug='s-3', cwd='/home/leo/src/dark-factory'),
        ]

        roots = known_project_roots(sessions, [])

        assert roots == ['/home/leo/src/dark-factory', '/home/leo/src/other-project']

    def test_unions_with_extra_roots(self):
        from cockpit.panes.decision_queue import known_project_roots

        sessions = [_make_session(session_slug='s-1', cwd='/home/leo/src/dark-factory')]

        roots = known_project_roots(sessions, ['/home/leo/src/extra-project'])

        assert roots == ['/home/leo/src/dark-factory', '/home/leo/src/extra-project']

    def test_empty_records_returns_extras_alone(self):
        from cockpit.panes.decision_queue import known_project_roots

        roots = known_project_roots([], ['/home/leo/src/extra-project'])

        assert roots == ['/home/leo/src/extra-project']

    def test_empty_records_and_no_extras_returns_empty_list(self):
        from cockpit.panes.decision_queue import known_project_roots

        assert known_project_roots([], []) == []

    def test_empty_cwd_records_excluded_and_never_raise(self):
        from cockpit.panes.decision_queue import known_project_roots

        sessions = [
            _make_session(session_slug='s-1', cwd=''),
            _make_session(session_slug='s-2', cwd='/home/leo/src/dark-factory'),
        ]

        roots = known_project_roots(sessions, [])

        assert roots == ['/home/leo/src/dark-factory']

    def test_duplicate_extra_deduped_against_records(self):
        from cockpit.panes.decision_queue import known_project_roots

        sessions = [_make_session(session_slug='s-1', cwd='/home/leo/src/dark-factory')]

        roots = known_project_roots(sessions, ['/home/leo/src/dark-factory'])

        assert roots == ['/home/leo/src/dark-factory']
