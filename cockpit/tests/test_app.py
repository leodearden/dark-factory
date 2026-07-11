"""Tests for cockpit.app — CockpitApp: the C5a TUI skeleton (session table + detail pane + poll).

Textual headless pilot tests (App.run_test / Pilot.pause) drive the few
integration signals this task requires: the table renders scanned records on
mount, the poll timer picks up on-disk changes, row selection renders the
detail pane, and the pure-consumer write discipline holds end-to-end. The
bulk of the underlying logic (glyph/title/age/order/detail/config) is
covered by fast deterministic unit tests elsewhere in this package -- see
test_session_table.py / test_detail_pane.py / test_registry_reader.py /
test_ui_config.py.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from orchestrator import session_registry as sr
from textual.coordinate import Coordinate


def _make_record(**overrides):
    """Mirrors test_registry_reader.py's _make_record convention."""
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


def _snapshot_tree(base: Path) -> dict[str, tuple[int, bytes]]:
    """Map every file under *base* (relative path -> (mtime_ns, bytes)).

    Used to prove the cockpit's write discipline by diffing this snapshot
    across an app run: an absent path is not a KeyError, it's just missing
    from the mapping, so a deleted-or-never-written file shows up as a plain
    dict-equality mismatch rather than needing special-casing.
    """
    if not base.is_dir():
        return {}
    return {
        str(path.relative_to(base)): (path.stat().st_mtime_ns, path.read_bytes())
        for path in base.rglob('*')
        if path.is_file()
    }


class TestInitialRender:
    @pytest.mark.timeout(10)
    async def test_seeded_records_render_as_rows(self, tmp_path):
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        r1 = _make_record(session_slug='a-1', status=sr.Status.RUNNING)
        r2 = _make_record(session_slug='b-2', status=sr.Status.AWAITING_INPUT)
        for r in (r1, r2):
            sr.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            table = app.query_one(SessionTable)
            assert table.row_count == 2

            row = table.get_row('a-1')
            assert 'unblock:df#2085' in row


class TestPollRefresh:
    @pytest.mark.timeout(10)
    async def test_new_record_appears_and_orders_blocked_first(self, tmp_path):
        """A record written to disk after mount appears once refresh_registry runs,
        and a newly-blocked record is ordered first (blocked-first).

        Drives refresh_registry() directly rather than sleeping across real
        poll ticks: rebuild-on-change is the behavior under test, and coupling
        it to wall-clock timing (poll_interval vs. a pilot.pause duration) is
        a flake risk under a loaded runner. The timer wiring itself
        (set_interval actually calling refresh_registry) is covered
        separately by TestTimerIntegration below -- the one test in this
        module allowed to depend on real timing.
        """
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        running = _make_record(session_slug='running-1', status=sr.Status.RUNNING)
        sr.write_record(running, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert table.row_count == 1

            blocked = _make_record(session_slug='blocked-1', status=sr.Status.AWAITING_INPUT)
            sr.write_record(blocked, root=tmp_path)

            app.refresh_registry()
            await pilot.pause()

            assert table.row_count == 2
            first_row_key = table.coordinate_to_cell_key(Coordinate(0, 0)).row_key
            assert first_row_key.value == 'blocked-1'

    @pytest.mark.timeout(10)
    async def test_mutated_status_reflected_after_poll(self, tmp_path):
        """An existing record's status change on disk is picked up by refresh_registry.

        Same rationale as above: calls refresh_registry() directly instead of
        racing the real poll_interval timer via pilot.pause(<duration>).
        """
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable, state_glyph

        running = _make_record(session_slug='running-1', status=sr.Status.RUNNING)
        sr.write_record(running, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert state_glyph(sr.Status.RUNNING) in table.get_row('running-1')

            sr.update_status('running-1', root=tmp_path, status=sr.Status.IDLE)

            app.refresh_registry()
            await pilot.pause()

            row = table.get_row('running-1')
            assert state_glyph(sr.Status.IDLE) in row
            assert state_glyph(sr.Status.RUNNING) not in row


class TestTimerIntegration:
    """The one test allowed to depend on real poll timing (see TestPollRefresh).

    Everything in TestPollRefresh now drives refresh_registry() directly
    (deterministic, no sleeping across real ticks). This test alone proves
    on_mount's set_interval(poll_interval, refresh_registry) is actually
    wired end-to-end, so that guarantee isn't lost entirely to determinism.
    """

    @pytest.mark.timeout(10)
    async def test_timer_tick_triggers_refresh(self, tmp_path):
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert table.row_count == 0

            record = _make_record(session_slug='timer-1', status=sr.Status.RUNNING)
            sr.write_record(record, root=tmp_path)

            await pilot.pause(0.3)

            assert table.row_count == 1


class TestRowSelectionDetail:
    @pytest.mark.timeout(10)
    async def test_selecting_row_renders_detail(self, tmp_path):
        from cockpit.app import CockpitApp
        from cockpit.panes.detail_pane import DetailPane
        from cockpit.panes.session_table import SessionTable

        # other-1 is blocked, so it (not target-1) is the default row-0
        # cursor -- the assertions below only pass if selecting target-1
        # actually drives the detail pane, not just whatever's highlighted
        # by default after mount.
        other = _make_record(session_slug='other-1', status=sr.Status.AWAITING_INPUT)
        target = _make_record(
            session_slug='target-1',
            status=sr.Status.RUNNING,
            task_id='2085',
            escalation_id='esc-99',
            question=sr.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        for r in (other, target):
            sr.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            table.move_cursor(row=table.get_row_index('target-1'))
            await pilot.pause()

            detail = app.query_one(DetailPane)
            assert 'Which port?' in detail.rendered_text
            assert '2085' in detail.rendered_text
            assert 'esc-99' in detail.rendered_text


class TestWriteDiscipline:
    @pytest.mark.timeout(10)
    async def test_cockpit_writes_only_its_own_ui_config(self, tmp_path):
        """PRD §2/§5 hard invariant: C5a is a pure consumer of the session and
        decision registries. It may create/update its OWN cockpit-ui.json, but
        running the app end-to-end (mount, a poll tick, a row selection) must
        never create, modify, or delete a sessions/ or decisions/ file --
        those belong to spawn-claude.sh/T4-T7/C8/C5b, never the C5a view."""
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        record = _make_record(session_slug='write-disc-1', status=sr.Status.RUNNING)
        sr.write_record(record, root=tmp_path)

        decision = sr.DecisionRecord(
            id='dec-1',
            project='df',
            text='Which port?',
            filed_at='2026-07-07T00:00:00+00:00',
        )
        assert sr.write_decision(decision, root=tmp_path)

        before = _snapshot_tree(tmp_path)
        # sanity: prove the snapshot actually captured the seeded files, so a
        # later "unchanged" assertion isn't vacuously true over an empty dict.
        assert any(path.startswith('sessions/') for path in before)
        assert any(path.startswith('decisions/') for path in before)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            await pilot.pause(0.2)  # let at least one poll tick land

            table.move_cursor(row=table.get_row_index('write-disc-1'))
            await pilot.pause()

        after = _snapshot_tree(tmp_path)

        for path, value in before.items():
            assert after.get(path) == value, f'{path} was created/modified/removed by the cockpit'

        new_or_modified = set(after) - set(before)
        assert new_or_modified == {'cockpit-ui.json'}, (
            f'expected only cockpit-ui.json as a new path, got {new_or_modified}'
        )


class TestDecisionQueueRender:
    @pytest.mark.timeout(10)
    async def test_open_decisions_and_awaiting_sessions_render_score_ordered(self, tmp_path):
        """The DecisionQueue widget renders every open decision + awaiting-input
        session (never a plain running session), rows keyed stably by
        decision/session identity, with the higher-scored item ranked above
        the lower one (PRD §9 C5b: rows `[score][age][project#task][question]`).
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from textual.widgets.data_table import RowDoesNotExist

        running = _make_record(session_slug='running-1', status=sr.Status.RUNNING)
        awaiting = _make_record(
            session_slug='awaiting-1',
            status=sr.Status.AWAITING_INPUT,
            question=sr.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        for r in (running, awaiting):
            sr.write_record(r, root=tmp_path)

        low = sr.DecisionRecord(
            id='dec-low',
            project='df',
            text='Low priority question?',
            filed_at='2026-07-07T00:00:00+00:00',
            manual_boost=0,
        )
        high = sr.DecisionRecord(
            id='dec-high',
            project='df',
            text='High priority question?',
            filed_at='2026-07-07T00:00:00+00:00',
            manual_boost=5,
        )
        for d in (low, high):
            assert sr.write_decision(d, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 3

            # rows keyed stably by decision/session identity
            assert queue.get_row('decision:dec-low')
            assert queue.get_row('decision:dec-high')
            assert queue.get_row('session:awaiting-1')

            # the plain running session never appears in the queue
            with pytest.raises(RowDoesNotExist):
                queue.get_row('session:running-1')

            # the higher-scored item (manual_boost=5) ranks above the lower one
            high_index = queue.get_row_index('decision:dec-high')
            low_index = queue.get_row_index('decision:dec-low')
            assert high_index < low_index


class TestSignalDontMove:
    @pytest.mark.timeout(10)
    async def test_awaiting_input_transition_sets_urgency_never_focus(self, tmp_path):
        """PRD B3 (§6.2/§7), this task's leaf signal: the refresh/diff path may
        call only backend.set_urgency + backend.reorder -- NEVER focus/tile --
        when a session flips into (or out of) awaiting-input. A synthetic
        running->awaiting-input transition must reorder the queue to include
        the newly-blocked target and set its urgency hint, while a
        FakeBackend spy proves zero focus/tile/raise calls ever happen on
        this path; a subsequent awaiting-input->idle transition clears the
        urgency hint.
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display = sr.Display(kind='wm', wm_title='unblock:df#2085 slug')
        running = _make_record(session_slug='running-1', status=sr.Status.RUNNING, display=display)
        sr.write_record(running, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 0

            awaiting = _make_record(
                session_slug='running-1',
                status=sr.Status.AWAITING_INPUT,
                display=display,
                question=sr.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
            )
            sr.write_record(awaiting, root=tmp_path)

            app.refresh_registry()
            await pilot.pause()

            target = DisplayTarget(kind='wm', wm_title='unblock:df#2085 slug')
            assert queue.row_count == 1
            assert (target, True) in backend.set_urgency_calls
            assert backend.focus_calls == []
            assert backend.tile_calls == []

            idle = _make_record(session_slug='running-1', status=sr.Status.IDLE, display=display)
            sr.write_record(idle, root=tmp_path)

            app.refresh_registry()
            await pilot.pause()

            assert queue.row_count == 0
            assert (target, False) in backend.set_urgency_calls
            assert backend.focus_calls == []
            assert backend.tile_calls == []


class TestEnterFocus:
    @pytest.mark.timeout(10)
    async def test_enter_focuses_the_highlighted_row_not_row_zero(self, tmp_path):
        """Explicit-action focus (PRD §6.2/§9 C5b): pressing Enter on the
        DecisionQueue focuses the HIGHLIGHTED row's resolved target -- never
        whatever happens to sit at row 0 -- and records that item's key in
        the app's in-memory "handling" set. This is the ONLY place a focus()
        call may originate from (see TestSignalDontMove for the refresh
        path's complementary "never focus" half of the same invariant).
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display_a = sr.Display(kind='wm', wm_title='awaiting-a title')
        display_b = sr.Display(kind='wm', wm_title='awaiting-b title')
        awaiting_a = _make_record(
            session_slug='awaiting-a',
            status=sr.Status.AWAITING_INPUT,
            display=display_a,
            question=sr.Question(text='A?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        awaiting_b = _make_record(
            session_slug='awaiting-b',
            status=sr.Status.AWAITING_INPUT,
            display=display_b,
            question=sr.Question(text='B?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        for r in (awaiting_a, awaiting_b):
            sr.write_record(r, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 2

            # Identical filed_at/manual_boost/state ties the two sessions'
            # scores, so order_queue's (-score, key) tiebreak puts
            # 'session:awaiting-a' at row 0 -- NOT our target below. If the
            # handler wired Enter to "always focus row 0" instead of the
            # actual highlighted row, this test would still catch it.
            assert queue.highlighted_key() == 'session:awaiting-a'

            target_key = 'session:awaiting-b'
            queue.move_cursor(row=queue.get_row_index(target_key))
            await pilot.pause()

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == [DisplayTarget(kind='wm', wm_title='awaiting-b title')]
            assert target_key in app._handling

    @pytest.mark.timeout(10)
    async def test_enter_on_unresolvable_target_is_fail_soft(self, tmp_path):
        """A queue row with no resolvable focus target (an open decision
        with no linked session) must not focus anything and must not raise
        -- fail-soft, PRD §2. A gone/unlinked lead is still a no-op, never
        a crash.
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        orphan = sr.DecisionRecord(
            id='dec-orphan',
            project='df',
            text='Orphaned decision?',
            filed_at='2026-07-07T00:00:00+00:00',
        )
        assert sr.write_decision(orphan, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == []


class TestBoostReordersAndPersists:
    @pytest.mark.timeout(10)
    async def test_boost_and_digit_reorder_live_and_persist(self, tmp_path):
        """b/B/digit priority keys (PRD §9 C5b) re-score and re-render the
        DecisionQueue IMMEDIATELY -- no poll tick, no explicit
        refresh_registry() call -- and, for a decision-backed row, persist
        the new manual_boost via C1's set_manual_boost (the cockpit's own
        sanctioned decision write). 'b' is additive (each press nudges the
        boost by one); a digit sets manual_boost to that EXACT integer
        (absolute, not additive).

        A fixed now_fn + an explicit Priorities.default() keep the score
        comparison hermetic -- independent of wall-clock time and of
        whatever priorities.yaml (if any) happens to live on the host.
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.priority import Priorities

        fixed_now = datetime.fromisoformat('2026-07-07T00:00:00+00:00')

        older = sr.DecisionRecord(
            id='dec-a',
            project='df',
            text='A?',
            filed_at='2026-06-01T00:00:00+00:00',  # far enough back to saturate the age bonus
            manual_boost=0,
        )
        newer = sr.DecisionRecord(
            id='dec-b',
            project='df',
            text='B?',
            filed_at='2026-07-07T00:00:00+00:00',  # same instant as `now` -- age 0
            manual_boost=0,
        )
        for d in (older, newer):
            assert sr.write_decision(d, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(
            fleet_root=tmp_path,
            backend=backend,
            poll_interval=0.05,
            now_fn=lambda: fixed_now,
            priorities=Priorities.default(),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 2
            # sanity: A (much older -- a bigger age bonus) currently
            # outranks B (freshly filed) by age alone, both at boost=0 --
            # this is the "another factor" the boost below must overcome.
            assert queue.get_row_index('decision:dec-a') < queue.get_row_index('decision:dec-b')

            queue.move_cursor(row=queue.get_row_index('decision:dec-b'))
            await pilot.pause()

            for _ in range(3):
                await pilot.press('b')
                await pilot.pause()

            # (a) live reorder -- B now ranks above A, with no poll tick and
            # no explicit refresh_registry() call anywhere in this test.
            assert queue.get_row_index('decision:dec-b') < queue.get_row_index('decision:dec-a')

            # (b) persisted via C1 -- disk reflects the raised boost.
            persisted = {d.id: d for d in sr.list_decisions(root=tmp_path)}
            assert persisted['dec-b'].manual_boost == 3
            assert persisted['dec-a'].manual_boost == 0  # untouched

            # a digit key sets manual_boost to that EXACT integer --
            # absolute, not additive (3 + 7 would be 10, not 7).
            await pilot.press('7')
            await pilot.pause()

            persisted = {d.id: d for d in sr.list_decisions(root=tmp_path)}
            assert persisted['dec-b'].manual_boost == 7


class TestDropRemovesAndPersists:
    @pytest.mark.timeout(10)
    async def test_drop_decision_row_removes_live_and_persists(self, tmp_path):
        """'x' on a DECISION-backed row (PRD §9 C5b) removes it from the
        queue IMMEDIATELY -- no poll tick, no explicit refresh_registry()
        call -- and persists via C1's update_decision_state, the cockpit's
        other sanctioned decision write. A dropped decision never
        resurfaces (order_queue's own state=='open' filter excludes it).
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from textual.widgets.data_table import RowDoesNotExist

        first = sr.DecisionRecord(
            id='dec-1', project='df', text='First?', filed_at='2026-07-07T00:00:00+00:00'
        )
        second = sr.DecisionRecord(
            id='dec-2', project='df', text='Second?', filed_at='2026-07-07T00:00:00+00:00'
        )
        for d in (first, second):
            assert sr.write_decision(d, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 2

            queue.move_cursor(row=queue.get_row_index('decision:dec-1'))
            await pilot.pause()

            await pilot.press('x')
            await pilot.pause()

            # (a) removed from the live queue -- no poll tick, no explicit
            # refresh_registry() call anywhere in this test.
            assert queue.row_count == 1
            with pytest.raises(RowDoesNotExist):
                queue.get_row('decision:dec-1')
            assert queue.get_row('decision:dec-2')

            # (b) persisted via C1.
            persisted = {d.id: d for d in sr.list_decisions(root=tmp_path)}
            assert persisted['dec-1'].state == sr.DecisionState.DROPPED
            assert persisted['dec-2'].state == sr.DecisionState.OPEN  # untouched

    @pytest.mark.timeout(10)
    async def test_drop_session_row_removes_live_without_writing_the_session(self, tmp_path):
        """'x' on a SESSION-backed row hides it from the queue in-memory
        ONLY -- sessions aren't cockpit-writable (PRD §2/§6.6), so this
        must not touch the session's record on disk, and must not raise.
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        awaiting = _make_record(
            session_slug='awaiting-1',
            status=sr.Status.AWAITING_INPUT,
            question=sr.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        sr.write_record(awaiting, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1

            await pilot.press('x')
            await pilot.pause()

            assert queue.row_count == 0

            # the session's own record is untouched -- still awaiting-input,
            # proving no registry write was attempted for a session row.
            reread = sr.read_record('awaiting-1', root=tmp_path)
            assert reread.status == sr.Status.AWAITING_INPUT
