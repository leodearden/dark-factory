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

    def test_foreign_status_child_is_treated_as_non_terminal_fail_soft(self):
        """A child with an unrecognized status must not abort the whole
        count -- fail-soft via _is_terminal, mirroring state_glyph/
        _state_rank (PRD §2): never let one unclassifiable record crash
        the view."""
        from cockpit.panes.session_table import count_outstanding_children

        parent = _make_record(session_slug='parent-1')
        foreign_child = _make_record(
            session_slug='child-foreign',
            parent_session_id='parent-1',
            status='some-foreign-status',
        )
        all_records = [parent, foreign_child]

        assert count_outstanding_children('parent-1', all_records) == 1


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

    def test_focusable_sorts_ahead_of_headless_within_a_status_band(self):
        """Inside one status band, a row Enter can act on comes first.

        This is load-bearing beyond mere ordering: app.py orders BEFORE
        capping (filter_live_sessions slices an already-ordered list), so
        under a 200+ live backlog the rows an operator can actually act on
        are the ones that survive the cap.
        """
        from cockpit.panes.session_table import order_sessions

        focusable = _make_record(
            session_slug='focusable',
            status=sr.Status.RUNNING,
            display=sr.Display(kind='wm', wm_title='t'),
        )
        headless = _make_record(
            session_slug='headless', status=sr.Status.RUNNING, display=None
        )

        assert [r.session_slug for r in order_sessions([headless, focusable])] == [
            'focusable',
            'headless',
        ]
        assert [r.session_slug for r in order_sessions([focusable, headless])] == [
            'focusable',
            'headless',
        ]

    def test_state_rank_still_dominates_focusability(self):
        """Blocked-on-you stays the top signal: a HEADLESS awaiting-input
        session outranks a FOCUSABLE running one. Focusability is a
        tiebreak inside a band, never a reordering across bands."""
        from cockpit.panes.session_table import order_sessions

        headless_blocked = _make_record(
            session_slug='headless-blocked',
            status=sr.Status.AWAITING_INPUT,
            display=None,
        )
        focusable_running = _make_record(
            session_slug='focusable-running',
            status=sr.Status.RUNNING,
            display=sr.Display(kind='wm', wm_title='t'),
        )

        ordered = order_sessions([focusable_running, headless_blocked])

        assert [r.session_slug for r in ordered] == [
            'headless-blocked',
            'focusable-running',
        ]

    def test_age_still_breaks_ties_below_focusability(self):
        """Two records alike in band and focusability still order oldest
        first -- focusability is inserted between the two existing keys,
        it does not replace the age tiebreak."""
        from cockpit.panes.session_table import order_sessions

        newer = _make_record(
            session_slug='newer',
            status=sr.Status.RUNNING,
            display=sr.Display(kind='wm', wm_title='t'),
            start_ts=datetime(2026, 7, 7, tzinfo=UTC).isoformat(),
        )
        older = _make_record(
            session_slug='older',
            status=sr.Status.RUNNING,
            display=sr.Display(kind='wm', wm_title='t'),
            start_ts=datetime(2026, 7, 1, tzinfo=UTC).isoformat(),
        )

        assert [r.session_slug for r in order_sessions([newer, older])] == ['older', 'newer']

    def test_headless_records_are_never_dropped(self):
        """Ordering reorders; it never hides. A headless session is real
        running work, and how many there are is itself fleet state."""
        from cockpit.panes.session_table import order_sessions

        records = [
            _make_record(session_slug='h-1', status=sr.Status.RUNNING, display=None),
            _make_record(
                session_slug='f-1',
                status=sr.Status.AWAITING_INPUT,
                display=sr.Display(kind='wm'),
            ),
            _make_record(session_slug='h-2', status=sr.Status.IDLE, display=None),
            _make_record(
                session_slug='f-2', status=sr.Status.RUNNING, display=sr.Display(kind='tmux')
            ),
        ]

        ordered = order_sessions(records)

        assert len(ordered) == len(records)
        assert {r.session_slug for r in ordered} == {r.session_slug for r in records}



class TestFilterLiveSessions:
    """The live band and, now, how much of it the cap hid.

    filter_live_sessions returns a LiveSessions(visible, total) view rather
    than a bare list: total is the live count BEFORE the cap, so a
    truncated table is no longer indistinguishable from a complete one.
    """

    def test_terminal_statuses_are_dropped(self):
        from cockpit.panes.session_table import filter_live_sessions

        exited = _make_record(session_slug='s-exited', status=sr.Status.EXITED)
        failed = _make_record(session_slug='s-failed', status=sr.Status.FAILED_TO_START)

        assert filter_live_sessions([exited, failed]).visible == []

    def test_non_terminal_statuses_are_all_kept(self):
        from cockpit.panes.session_table import filter_live_sessions

        awaiting = _make_record(session_slug='s-awaiting', status=sr.Status.AWAITING_INPUT)
        running = _make_record(session_slug='s-running', status=sr.Status.RUNNING)
        launching = _make_record(session_slug='s-launching', status=sr.Status.LAUNCHING)
        idle = _make_record(session_slug='s-idle', status=sr.Status.IDLE)
        records = [awaiting, running, launching, idle]

        kept = filter_live_sessions(records).visible

        assert [r.session_slug for r in kept] == [r.session_slug for r in records]

    def test_foreign_status_is_kept_fail_soft(self):
        from cockpit.panes.session_table import filter_live_sessions

        foreign = _make_record(session_slug='s-foreign', status='some-foreign-status')

        assert filter_live_sessions([foreign]).visible == [foreign]

    def test_relative_order_of_kept_records_is_preserved(self):
        from cockpit.panes.session_table import filter_live_sessions

        awaiting = _make_record(session_slug='s-awaiting', status=sr.Status.AWAITING_INPUT)
        exited = _make_record(session_slug='s-exited', status=sr.Status.EXITED)
        running = _make_record(session_slug='s-running', status=sr.Status.RUNNING)
        idle = _make_record(session_slug='s-idle', status=sr.Status.IDLE)
        ordered_input = [awaiting, exited, running, idle]

        kept = filter_live_sessions(ordered_input).visible

        assert [r.session_slug for r in kept] == ['s-awaiting', 's-running', 's-idle']

    def test_empty_input_returns_empty_view(self):
        from cockpit.panes.session_table import filter_live_sessions

        view = filter_live_sessions([])

        assert view.visible == []
        assert view.total == 0

    def test_explicit_cap_keeps_only_the_first_n_by_input_order(self):
        from cockpit.panes.session_table import filter_live_sessions

        records = [
            _make_record(session_slug=f's-{i}', status=sr.Status.RUNNING) for i in range(5)
        ]

        kept = filter_live_sessions(records, cap=3).visible

        assert [r.session_slug for r in kept] == ['s-0', 's-1', 's-2']

    def test_default_cap_limits_to_default_visible_cap(self):
        from cockpit.panes.session_table import _DEFAULT_VISIBLE_CAP, filter_live_sessions

        records = [
            _make_record(session_slug=f's-{i}', status=sr.Status.RUNNING)
            for i in range(_DEFAULT_VISIBLE_CAP + 10)
        ]

        kept = filter_live_sessions(records).visible

        assert len(kept) == _DEFAULT_VISIBLE_CAP
        assert [r.session_slug for r in kept] == [
            f's-{i}' for i in range(_DEFAULT_VISIBLE_CAP)
        ]

    def test_total_reports_the_live_count_the_cap_hid(self):
        """The number an operator could otherwise never see: reading exactly
        `cap` rows told you nothing about how many live sessions there
        really were."""
        from cockpit.panes.session_table import filter_live_sessions

        records = [
            _make_record(session_slug=f's-{i}', status=sr.Status.RUNNING) for i in range(5)
        ]

        view = filter_live_sessions(records, cap=3)

        assert len(view.visible) == 3
        assert view.total == 5

    def test_terminal_records_are_excluded_from_total_too(self):
        """total counts the LIVE band, not the scanned set -- otherwise the
        notice would claim the cap hid history rows the view never intended
        to show in the first place."""
        from cockpit.panes.session_table import filter_live_sessions

        records = [
            _make_record(session_slug='live-0', status=sr.Status.RUNNING),
            _make_record(session_slug='dead-0', status=sr.Status.EXITED),
            _make_record(session_slug='live-1', status=sr.Status.IDLE),
            _make_record(session_slug='dead-1', status=sr.Status.FAILED_TO_START),
            _make_record(session_slug='dead-2', status=sr.Status.EXITED),
        ]

        view = filter_live_sessions(records)

        assert view.total == 2
        assert len(view.visible) == 2

    def test_under_the_cap_total_equals_visible(self):
        from cockpit.panes.session_table import filter_live_sessions

        records = [
            _make_record(session_slug=f's-{i}', status=sr.Status.RUNNING) for i in range(4)
        ]

        view = filter_live_sessions(records, cap=10)

        assert view.total == len(view.visible) == 4

    def test_visible_never_exceeds_total(self):
        """The view's own invariant, checked across the cap boundary rather
        than at one convenient point."""
        from cockpit.panes.session_table import filter_live_sessions

        records = [
            _make_record(session_slug=f's-{i}', status=sr.Status.RUNNING) for i in range(6)
        ]

        for cap in range(0, 9):
            view = filter_live_sessions(records, cap=cap)

            assert len(view.visible) <= view.total


class TestFocusMarker:
    """The per-row focusability cue: can Enter raise a terminal for this row?

    `display` is the only thing that makes a row focusable -- decision_queue.
    resolve_target's SessionRecord branch resolves straight from it, and
    that is the code app.py::_focus_slug actually runs. is_focusable is
    restated in session_table rather than imported from decision_queue
    (which already imports this module -- a reverse import would be a
    cycle), so test_agrees_with_resolve_target below is what keeps the two
    statements honest.
    """

    def test_record_with_display_is_focusable(self):
        from cockpit.panes.session_table import focus_marker, is_focusable

        record = _make_record(display=sr.Display(kind='wm', wm_title='t'))

        assert is_focusable(record) is True
        assert focus_marker(record) == '▸'

    def test_record_without_display_is_not_focusable(self):
        from cockpit.panes.session_table import focus_marker, is_focusable

        record = _make_record(display=None)

        assert is_focusable(record) is False
        assert focus_marker(record) == '·'

    def test_markers_are_distinct_single_chars_outside_the_status_vocabulary(self):
        """The cue must read in BOTH directions and must never be confusable
        with a status glyph -- the two dimensions are orthogonal, so a
        marker colliding with '⚙'/'✓'/'⏸'/'☠'/'?' would make one column
        look like the other."""
        from cockpit.panes.session_table import focus_marker

        focusable = focus_marker(_make_record(display=sr.Display(kind='wm')))
        headless = focus_marker(_make_record(display=None))

        assert focusable != headless
        for marker in (focusable, headless):
            assert len(marker) == 1
            assert marker not in {'⚙', '✓', '⏸', '☠', '?'}

    def test_unrecognized_display_kind_still_reads_focusable_fail_soft(self):
        """A Display whose kind this view doesn't recognize is still a real
        terminal somewhere -- mirroring _is_terminal/state_glyph fail-soft
        (PRD §2), an unknown kind must never mislabel it unactionable."""
        from cockpit.panes.session_table import focus_marker, is_focusable

        record = _make_record(display=sr.Display(kind='weird', wm_title='t'))

        assert is_focusable(record) is True
        assert focus_marker(record) == '▸'

    def test_agrees_with_resolve_target(self):
        """SPOT guard: the cue must mean exactly what Enter does.

        resolve_target is imported HERE only -- production session_table
        must not import decision_queue (import cycle). If resolve_target's
        SessionRecord branch ever grows a stricter rule, this fails loudly
        rather than leaving the table quietly lying to the operator.
        """
        from cockpit.panes.decision_queue import resolve_target
        from cockpit.panes.session_table import is_focusable

        with_display = _make_record(
            session_slug='has-display', display=sr.Display(kind='wm', wm_title='t')
        )
        without_display = _make_record(session_slug='no-display', display=None)

        for record in (with_display, without_display):
            assert is_focusable(record) == (resolve_target(record, {}) is not None)
