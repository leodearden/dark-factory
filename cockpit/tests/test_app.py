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

import asyncio
import shutil
import threading
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


class _BlockingScanner:
    """Fake SessionScanner for TestNonBlockingPoll: pins the exact moment a
    threaded poll scan is in-flight, deterministically and without needing
    10k real session dirs to make a scan slow.

    The FIRST scan() call (on_mount's synchronous initial scan, made before
    a CockpitApp ever launches a threaded poll worker) returns immediately
    with no records -- if it blocked too, mounting the app would deadlock
    the single-threaded event loop it runs on. Every call after that
    records the calling thread's ident, signals `started`, and blocks on
    `_release` until the test calls release() -- letting a test observe the
    app's event loop staying responsive while a slow scan is still
    in-flight on its own (non-UI) thread.
    """

    def __init__(self) -> None:
        self.scan_idents: list[int] = []
        self.started = threading.Event()
        self._release = threading.Event()
        self._armed = False

    def scan(self) -> list:
        self.scan_idents.append(threading.get_ident())
        if not self._armed:
            self._armed = True
            return []
        self.started.set()
        self._release.wait()
        return []

    def release(self) -> None:
        self._release.set()


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


class TestSessionTableDefaultFilter:
    @pytest.mark.timeout(10)
    async def test_terminal_sessions_are_hidden_by_default_on_mount(self, tmp_path):
        """FINDING F2 (C10 tour, esc-2303-1): the default session-table view
        hides terminal (exited/failed-to-start) sessions on launch, so
        "tens, not 10k" holds even when the registry carries stale history."""
        from textual.widgets.data_table import RowDoesNotExist

        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        live = _make_record(session_slug='live-1', status=sr.Status.RUNNING)
        dead = _make_record(session_slug='dead-1', status=sr.Status.EXITED)
        for r in (live, dead):
            sr.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            table = app.query_one(SessionTable)
            assert table.get_row('live-1')
            with pytest.raises(RowDoesNotExist):
                table.get_row('dead-1')
            assert table.row_count == 1


class TestToggleHistory:
    @pytest.mark.timeout(10)
    async def test_h_toggles_terminal_sessions_and_round_trips(self, tmp_path):
        """'h' (PRD-adjacent, FINDING F2 C10 tour esc-2303-1) reveals the full,
        unfiltered history on demand and hides it again on a second press --
        an in-memory-only toggle, re-rendered immediately (no poll tick)."""
        from textual.widgets.data_table import RowDoesNotExist

        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        live = _make_record(session_slug='live-1', status=sr.Status.RUNNING)
        dead = _make_record(session_slug='dead-1', status=sr.Status.EXITED)
        for r in (live, dead):
            sr.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            table = app.query_one(SessionTable)
            with pytest.raises(RowDoesNotExist):
                table.get_row('dead-1')
            assert table.row_count == 1

            await pilot.press('h')
            await pilot.pause()

            assert table.get_row('dead-1')
            assert table.row_count == 2

            await pilot.press('h')
            await pilot.pause()

            with pytest.raises(RowDoesNotExist):
                table.get_row('dead-1')
            assert table.row_count == 1

    @pytest.mark.timeout(10)
    async def test_toggle_persists_across_a_poll_tick(self, tmp_path):
        """_rebuild_session_table reads self._show_history fresh on every
        _apply_scan (both the synchronous refresh_registry path and the
        threaded poll path route through it) -- so toggling history on and
        then having a scan pick up a disk change must not silently revert
        to the filtered default view. A regression that reset the flag, or
        that hard-coded the filtered view inside _apply_scan, would only be
        caught by a test with an intervening scan -- pressing 'h' and
        asserting immediately (as above) is not enough."""
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        live = _make_record(session_slug='live-1', status=sr.Status.RUNNING)
        dead = _make_record(session_slug='dead-1', status=sr.Status.EXITED)
        for r in (live, dead):
            sr.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            await pilot.press('h')
            await pilot.pause()

            assert table.get_row('dead-1')
            assert table.row_count == 2

            live2 = _make_record(session_slug='live-2', status=sr.Status.RUNNING)
            sr.write_record(live2, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert table.get_row('dead-1')
            assert table.get_row('live-2')
            assert table.row_count == 3


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

        A large poll_interval keeps on_mount's own set_interval timer from
        firing an uncontrolled background poll during this test (mirrors the
        idiom in TestNonBlockingPoll/TestThreadedScanReachesUI/
        TestScanBackpressure below) -- _apply_scan has no protection against
        a stale threaded scan (one that read the registry before the
        blocked-1 write below) landing via call_from_thread AFTER this
        test's own direct refresh_registry() call and clobbering its
        fresher result back down to 1 row. Confirmed by direct repro: with a
        short real interval, an in-flight poll tick can race this exact
        write+refresh_registry() sequence under a loaded runner.
        """
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        running = _make_record(session_slug='running-1', status=sr.Status.RUNNING)
        sr.write_record(running, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
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
        racing the real poll_interval timer via pilot.pause(<duration>). Same
        large poll_interval as the test above too, for the same reason: it
        keeps a real background poll tick from ever landing (via
        call_from_thread) in between this test's own write and its direct
        refresh_registry() call.
        """
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable, state_glyph

        running = _make_record(session_slug='running-1', status=sr.Status.RUNNING)
        sr.write_record(running, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
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


class TestReplaceRowsChildrenCountAgainstFullSet:
    @pytest.mark.timeout(10)
    async def test_children_counted_against_all_records_not_visible_subset(self, tmp_path):
        """replace_rows' all_records param counts outstanding children against
        the FULL scanned set, not just the (possibly filtered) visible
        `records` -- so a visible parent's non-terminal child is never
        undercounted just because the child itself is hidden from view."""
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        parent = _make_record(session_slug='parent-1', parent_session_id=None)
        running_child = _make_record(
            session_slug='child-running',
            parent_session_id='parent-1',
            status=sr.Status.RUNNING,
        )

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            table.replace_rows(
                [parent],
                datetime.fromisoformat('2026-07-07T00:00:00+00:00'),
                all_records=[parent, running_child],
            )
            await pilot.pause()

            assert table.row_count == 1
            assert table.get_row('parent-1')[4] == '1'


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
        from textual.widgets.data_table import RowDoesNotExist

        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

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


class TestSharedTargetAttentionNotClearedPrematurely:
    @pytest.mark.timeout(10)
    async def test_urgency_stays_set_while_a_second_item_still_targets_it(self, tmp_path):
        """PRD B3 edge case: a DecisionRecord and the AWAITING_INPUT session
        it links to (via session_id) can resolve to the exact SAME
        DisplayTarget (resolve_target maps a decision through its session's
        display). _update_attention must diff by resolved TARGET, not by
        item key -- otherwise the decision leaving the queue (e.g. answered
        by a C8 watcher) while its session is STILL awaiting input would
        incorrectly clear the shared target's urgency hint, even though a
        live ask (the session) still points at it.
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import DisplayTarget, FakeBackend

        display = sr.Display(kind='wm', wm_title='shared title')
        awaiting = _make_record(
            session_slug='shared-1',
            status=sr.Status.AWAITING_INPUT,
            display=display,
            question=sr.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        sr.write_record(awaiting, root=tmp_path)
        decision = sr.DecisionRecord(
            id='dec-shared',
            project='df',
            text='Proceed?',
            filed_at='2026-07-07T00:00:00+00:00',
            session_id='shared-1',
        )
        assert sr.write_decision(decision, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            target = DisplayTarget(kind='wm', wm_title='shared title')
            assert (target, True) in backend.set_urgency_calls

            # The decision resolves elsewhere (e.g. a C8 watcher answers it)
            # while the session it's linked to is STILL awaiting input.
            sr.update_decision_state('dec-shared', sr.DecisionState.ANSWERED, root=tmp_path)

            calls_before = len(backend.set_urgency_calls)
            app.refresh_registry()
            await pilot.pause()

            # The session's own queue item still resolves to `target` -- its
            # urgency must NOT be cleared just because the decision left.
            new_calls = backend.set_urgency_calls[calls_before:]
            assert (target, False) not in new_calls


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


class TestHandlingPrunedOnQueueExit:
    @pytest.mark.timeout(10)
    async def test_handling_flag_does_not_stick_past_the_ask_it_was_set_for(self, tmp_path):
        """self._handling is an in-memory 'already acted on' marker keyed by
        QueueItem.key (PRD §9 C5b). For a SESSION-backed row the key is
        stable for the session's whole lifetime, so once Enter marks it
        handling, it must be cleared again once that ask resolves (the
        session leaves the queue) -- otherwise a brand new question from
        the SAME session would silently inherit a stale "already handled"
        flag left over from a completely different, earlier ask.
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend

        display = sr.Display(kind='wm', wm_title='handling title')
        awaiting = _make_record(
            session_slug='handling-1',
            status=sr.Status.AWAITING_INPUT,
            display=display,
            question=sr.Question(text='First ask?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        sr.write_record(awaiting, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            key = 'session:handling-1'
            await pilot.press('enter')
            await pilot.pause()
            assert key in app._handling

            # The ask resolves -- the session leaves the queue entirely.
            idle = _make_record(session_slug='handling-1', status=sr.Status.IDLE, display=display)
            sr.write_record(idle, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert key not in app._handling

            # A brand new question from the SAME session must not inherit
            # the stale handling flag left by the earlier, resolved ask.
            second_ask = _make_record(
                session_slug='handling-1',
                status=sr.Status.AWAITING_INPUT,
                display=display,
                question=sr.Question(text='Second ask?', asked_at='2026-07-08T00:00:00+00:00'),
            )
            sr.write_record(second_ask, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert app._queue_items_by_key[key].handling is False


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
        from textual.widgets.data_table import RowDoesNotExist

        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

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


class TestCopyAction:
    @pytest.mark.timeout(10)
    async def test_copy_highlighted_decision_puts_question_and_ids_on_clipboard(self, tmp_path):
        """'y' (the copy affordance, task 2517 / esc-2303-1 F4) copies the
        highlighted DecisionQueue row's question text + ids onto the system
        clipboard via Textual's in-app OSC 52 App.copy_to_clipboard --
        terminal-native, no xclip/wl-copy subprocess -- and is strictly
        READ-ONLY, never touching sessions/ or decisions/ (mirrors
        TestWriteDiscipline's before/after _snapshot_tree diff).
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        decision = sr.DecisionRecord(
            id='dec-1',
            project='df',
            text='Which port do we bind?',
            filed_at='2026-07-07T00:00:00+00:00',
            task_id='2517',
            escalation_id='esc-42',
        )
        assert sr.write_decision(decision, root=tmp_path)

        awaiting = _make_record(
            session_slug='awaiting-1',
            status=sr.Status.AWAITING_INPUT,
            question=sr.Question(text='Which host?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        sr.write_record(awaiting, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 2

            queue.move_cursor(row=queue.get_row_index('decision:dec-1'))
            await pilot.pause()

            before = _snapshot_tree(tmp_path)

            await pilot.press('y')
            await pilot.pause()

            # (a) the row's question + ids landed on the clipboard.
            assert app._clipboard
            assert 'Which port do we bind?' in app._clipboard
            assert 'esc-42' in app._clipboard
            assert '2517' in app._clipboard

            # (b) strictly read-only -- no sessions/ or decisions/ write.
            after = _snapshot_tree(tmp_path)
            assert after == before

    @pytest.mark.timeout(10)
    async def test_copy_highlighted_session_puts_slug_and_question_on_clipboard(self, tmp_path):
        """'y' on a SESSION-backed row (no decision behind it) copies the
        session slug + question text onto the clipboard. Covers the
        app-level action_copy -> highlighted SESSION row -> clipboard path
        end-to-end -- the decision-row case above and format_copy_payload's
        own pure-formatter unit tests don't exercise this branch through
        action_copy itself (reviewer_comprehensive test_coverage
        suggestion, test_app.py:911).
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        awaiting = _make_record(
            session_slug='awaiting-99',
            status=sr.Status.AWAITING_INPUT,
            question=sr.Question(text='Which region?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        sr.write_record(awaiting, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1

            queue.move_cursor(row=queue.get_row_index('session:awaiting-99'))
            await pilot.pause()

            before = _snapshot_tree(tmp_path)

            await pilot.press('y')
            await pilot.pause()

            # (a) the session row's question + slug landed on the clipboard.
            assert app._clipboard
            assert 'Which region?' in app._clipboard
            assert 'awaiting-99' in app._clipboard

            # (b) strictly read-only -- a session is never cockpit-written.
            after = _snapshot_tree(tmp_path)
            assert after == before

    @pytest.mark.timeout(10)
    async def test_copy_with_empty_queue_is_a_fail_soft_no_op(self, tmp_path):
        """'y' against an EMPTY queue -- highlighted_key() returns None,
        mirroring action_drop/action_defer's own fail-soft guard -- must
        not crash and must leave the clipboard untouched (reviewer_comprehensive
        test_coverage suggestion's optional no-highlight/no-op case).
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 0

            await pilot.press('y')
            await pilot.pause()

            assert app._clipboard == ''


class TestDeferResetsAge:
    @pytest.mark.timeout(10)
    async def test_defer_resets_age_live_without_writing_registry(self, tmp_path):
        """'d' (defer, PRD §9 C5b) resets the highlighted item's EFFECTIVE
        age to ~0 IMMEDIATELY -- no poll tick, no explicit
        refresh_registry() call -- sinking it below an item that was
        previously outranked purely by age. This is an in-memory age-reset
        ONLY: DecisionRecord.filed_at (provenance -- when a decision was
        actually filed) is NEVER rewritten, so the defer must leave every
        sessions/ and decisions/ file on disk byte-identical (mirrors
        TestWriteDiscipline's _snapshot_tree before/after convention).

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
            id='dec-old',
            project='df',
            text='Old?',
            filed_at='2026-06-01T00:00:00+00:00',  # far enough back to saturate the age bonus
            manual_boost=0,
        )
        newer = sr.DecisionRecord(
            id='dec-new',
            project='df',
            text='New?',
            filed_at='2026-07-07T00:00:00+00:00',  # same instant as `now` -- age 0
            manual_boost=0,
        )
        for d in (older, newer):
            assert sr.write_decision(d, root=tmp_path)

        before = _snapshot_tree(tmp_path)
        assert any(path.startswith('decisions/') for path in before)

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
            # sanity: the older item (a much bigger age bonus) currently
            # outranks the newer one, both at boost=0 -- this is the
            # ordering the defer below must flip.
            assert queue.get_row_index('decision:dec-old') < queue.get_row_index(
                'decision:dec-new'
            )

            queue.move_cursor(row=queue.get_row_index('decision:dec-old'))
            await pilot.pause()

            await pilot.press('d')
            await pilot.pause()

            # (a) live reorder -- the deferred item now sinks below the
            # newer one, with no poll tick and no explicit
            # refresh_registry() call anywhere in this test.
            assert queue.get_row_index('decision:dec-new') < queue.get_row_index(
                'decision:dec-old'
            )

        # (b) in-memory only -- not a single sessions/ or decisions/ file
        # was created/modified/removed (filed_at is never rewritten by a
        # defer). Taken after the app has unmounted, so this also covers
        # on_unmount's own _persist_ui_config write -- which never touches
        # sessions/ or decisions/ either way.
        after = _snapshot_tree(tmp_path)
        for path, value in before.items():
            assert after.get(path) == value, f'{path} was created/modified/removed by the defer'
        new_paths = set(after) - set(before)
        assert not any(path.startswith(('sessions/', 'decisions/')) for path in new_paths)


class TestSpawnBar:
    @pytest.mark.timeout(10)
    async def test_new_session_key_pushes_spawn_screen(self, tmp_path):
        """'n' (PRD §9 C5b spawn bar) pushes a SpawnScreen project/role/prompt picker."""
        from cockpit.app import CockpitApp
        from cockpit.panes.spawn_bar import SpawnScreen

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05, spawn_runner=lambda argv: None)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('n')
            await pilot.pause()

            assert isinstance(app.screen, SpawnScreen)

    @pytest.mark.timeout(10)
    async def test_spawn_session_invokes_runner_with_exact_argv(self, tmp_path):
        """spawn_session(project_root, role, prompt) -- the spawn bar's leaf
        signal (PRD §9 C5b) -- builds spawn-claude.sh's exact positional
        argv (build_spawn_argv) from the picked project (cwd), a
        role-derived title, and the prompt, then hands it to the injected
        spawn_runner. No real terminal/process is launched here (the
        runner is faked) -- driven directly per this task's plan, rather
        than through SpawnScreen's own Input/Select widgets.
        """
        from cockpit.app import CockpitApp
        from cockpit.panes.spawn_bar import build_spawn_argv

        spawned: list[list[str]] = []
        spawn_script = '/repo/skills/spawn/spawn-claude.sh'

        app = CockpitApp(
            fleet_root=tmp_path,
            poll_interval=0.05,
            spawn_runner=spawned.append,
            spawn_script=spawn_script,
        )
        async with app.run_test() as pilot:
            await pilot.pause()

            project_root = '/home/leo/src/dark-factory'
            role = 'unblock'
            prompt = 'Please look at this'
            app.spawn_session(project_root, role, prompt)
            await pilot.pause()

        assert len(spawned) == 1
        expected_title = f'{role}:{Path(project_root).name}'
        assert spawned[0] == build_spawn_argv(spawn_script, project_root, expected_title, prompt)


class TestSpawnTree:
    @pytest.mark.timeout(10)
    async def test_toggle_key_pushes_tree_screen_with_structure(self, tmp_path):
        """'t' (Fleet Cockpit C9a spawn-tree toggle) pushes a SpawnTreeScreen
        rendering the parent_session_id parent→child session-tree structure."""
        from cockpit.app import CockpitApp
        from cockpit.panes.spawn_tree import SpawnTree, SpawnTreeScreen

        parent = _make_record(session_slug='parent-1', parent_session_id=None)
        child = _make_record(session_slug='child-1', parent_session_id='parent-1')
        for r in (parent, child):
            sr.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()

            assert isinstance(app.screen, SpawnTreeScreen)
            tree = app.screen.query_one(SpawnTree)
            top_level_slugs = {node.data for node in tree.root.children}
            assert 'parent-1' in top_level_slugs

            parent_node = next(node for node in tree.root.children if node.data == 'parent-1')
            child_slugs = {node.data for node in parent_node.children}
            assert child_slugs == {'child-1'}

    @pytest.mark.timeout(10)
    async def test_outstanding_children_are_highlighted(self, tmp_path):
        """Outstanding (non-terminal) children are visually marked in the
        rendered tree (Fleet Cockpit C9a, PRD §9) -- pins the "outstanding
        children highlighted" half of the leaf signal at the widget level."""
        from cockpit.app import CockpitApp
        from cockpit.panes.spawn_tree import _OUTSTANDING_MARKER, SpawnTree, SpawnTreeScreen

        parent = _make_record(session_slug='parent-1', parent_session_id=None)
        running_child = _make_record(
            session_slug='child-running',
            parent_session_id='parent-1',
            status=sr.Status.RUNNING,
        )
        exited_child = _make_record(
            session_slug='child-exited',
            parent_session_id='parent-1',
            status=sr.Status.EXITED,
        )
        for r in (parent, running_child, exited_child):
            sr.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()

            assert isinstance(app.screen, SpawnTreeScreen)
            tree = app.screen.query_one(SpawnTree)
            parent_node = next(node for node in tree.root.children if node.data == 'parent-1')
            nodes_by_slug = {node.data: node for node in parent_node.children}

            running_label = str(nodes_by_slug['child-running'].label)
            exited_label = str(nodes_by_slug['child-exited'].label)

            assert _OUTSTANDING_MARKER in running_label
            assert _OUTSTANDING_MARKER not in exited_label


class TestSpawnTreeEnterFocus:
    @pytest.mark.timeout(10)
    async def test_enter_focuses_the_highlighted_child_node(self, tmp_path):
        """Enter on the spawn tree's CHILD node (deliberately not the
        parent/row-0) resolves that child's own Display and routes it to
        the focus backend -- the C9a leaf signal (PRD §9), mirroring
        TestEnterFocus's decision-queue contract at the tree-toggle level.
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.spawn_tree import SpawnTree, SpawnTreeScreen

        parent_display = sr.Display(kind='wm', wm_title='parent title')
        child_display = sr.Display(kind='wm', wm_title='child title')
        parent = _make_record(
            session_slug='parent-1', parent_session_id=None, display=parent_display
        )
        child = _make_record(
            session_slug='child-1', parent_session_id='parent-1', display=child_display
        )
        for r in (parent, child):
            sr.write_record(r, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()

            assert isinstance(app.screen, SpawnTreeScreen)
            tree = app.screen.query_one(SpawnTree)
            parent_node = next(node for node in tree.root.children if node.data == 'parent-1')
            child_node = next(node for node in parent_node.children if node.data == 'child-1')

            tree.move_cursor(child_node)
            await pilot.pause()

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == [DisplayTarget(kind='wm', wm_title='child title')]

    @pytest.mark.timeout(10)
    async def test_enter_on_child_with_no_display_is_fail_soft(self, tmp_path):
        """A child with no Display resolves to no focus target -- pressing
        Enter on it must not focus anything and must not raise (PRD §2)."""
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.spawn_tree import SpawnTree, SpawnTreeScreen

        parent = _make_record(session_slug='parent-1', parent_session_id=None)
        child = _make_record(
            session_slug='child-1', parent_session_id='parent-1', display=None
        )
        for r in (parent, child):
            sr.write_record(r, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()

            assert isinstance(app.screen, SpawnTreeScreen)
            tree = app.screen.query_one(SpawnTree)
            parent_node = next(node for node in tree.root.children if node.data == 'parent-1')
            child_node = next(node for node in parent_node.children if node.data == 'child-1')

            tree.move_cursor(child_node)
            await pilot.pause()

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == []

    @pytest.mark.timeout(10)
    async def test_focus_slug_missing_from_snapshot_is_fail_soft(self, tmp_path):
        """A slug that isn't in the snapshot passed to _focus_slug (e.g. the
        opened tree's own snapshot, once stale) resolves to no record at all
        -- the `record is None` branch -- and must not raise or focus
        anything (PRD §2), mirroring the no-Display fail-soft case above."""
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend

        some_record = _make_record(session_slug='parent-1', parent_session_id=None)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test():
            app._focus_slug('nonexistent-slug', [some_record])

            assert backend.focus_calls == []

    @pytest.mark.timeout(10)
    async def test_enter_resolves_against_the_opened_snapshot_not_live_records(self, tmp_path):
        """Enter must resolve against the snapshot action_toggle_tree closed
        over when the tree was opened, not whatever self._records has been
        reassigned to by an intervening refresh_registry poll tick -- see
        action_toggle_tree's and _focus_slug's docstrings. The reassignment
        is reproduced via a *real* refresh_registry() scan (not a hand-poked
        app._records assignment): the child's on-disk record is removed
        first, so the scan legitimately drops it and self._records
        genuinely no longer contains the child by the time Enter is
        pressed. This keeps the guard deterministic -- unlike overwriting
        app._records directly, which the still-running poll_interval=0.05
        background poller could silently re-populate from the (unchanged)
        on-disk child before Enter lands, since nothing about that
        alternate approach stops disk from still backing the child.
        Enter must still focus the original child's Display, exactly as if
        the poll tick had never landed."""
        from cockpit.app import CockpitApp
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.spawn_tree import SpawnTree, SpawnTreeScreen

        parent_display = sr.Display(kind='wm', wm_title='parent title')
        child_display = sr.Display(kind='wm', wm_title='child title')
        parent = _make_record(
            session_slug='parent-1', parent_session_id=None, display=parent_display
        )
        child = _make_record(
            session_slug='child-1', parent_session_id='parent-1', display=child_display
        )
        for r in (parent, child):
            sr.write_record(r, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()

            assert isinstance(app.screen, SpawnTreeScreen)
            tree = app.screen.query_one(SpawnTree)
            parent_node = next(node for node in tree.root.children if node.data == 'parent-1')
            child_node = next(node for node in parent_node.children if node.data == 'child-1')

            tree.move_cursor(child_node)
            await pilot.pause()

            # Simulate an intervening refresh_registry poll tick that
            # replaces self._records wholesale with a set no longer
            # containing the highlighted child -- deterministically: the
            # child's record.json is gone from disk, so this scan (like
            # any later background poll tick) can only ever find parent-1.
            shutil.rmtree(sr.record_path_for_slug('child-1', root=tmp_path).parent)
            app.refresh_registry()
            await pilot.pause()
            # sanity: the simulated poll tick actually dropped the child,
            # so the assertion below isn't vacuously true.
            assert [r.session_slug for r in app._records] == ['parent-1']

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == [DisplayTarget(kind='wm', wm_title='child title')]


class TestSessionTableEnterFocus:
    @pytest.mark.timeout(10)
    async def test_enter_focuses_the_highlighted_session_row(self, tmp_path):
        """Enter on the session table's HIGHLIGHTED row (deliberately not
        the default row-0) focuses THAT session's terminal -- the C10 tour
        F6 fix (PRD §1: "Pressing Enter on a row raises that terminal"),
        mirroring TestSpawnTreeEnterFocus's contract at the top-level
        session-table Enter path.
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.session_table import SessionTable

        session_a = _make_record(
            session_slug='session-a',
            status=sr.Status.RUNNING,
            start_ts='2026-07-07T00:00:00+00:00',
            display=sr.Display(kind='wm', wm_title='session-a title'),
        )
        session_b = _make_record(
            session_slug='session-b',
            status=sr.Status.RUNNING,
            start_ts='2026-07-07T01:00:00+00:00',
            display=sr.Display(kind='wm', wm_title='session-b title'),
        )
        for r in (session_a, session_b):
            sr.write_record(r, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            table.focus()
            await pilot.pause()

            # session-a (older start_ts) is the default row-0 cursor -- the
            # assertions below only pass if selecting session-b actually
            # drives the focus call, not just whatever's highlighted by
            # default after mount.
            assert table.highlighted_slug() != 'session-b'

            table.move_cursor(row=table.get_row_index('session-b'))
            await pilot.pause()

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == [DisplayTarget(kind='wm', wm_title='session-b title')]

    @pytest.mark.timeout(10)
    async def test_enter_on_session_row_with_no_display_is_fail_soft(self, tmp_path):
        """A session row with no Display resolves to no focus target --
        pressing Enter on it must not focus anything and must not raise
        (PRD §2), mirroring TestSpawnTreeEnterFocus's no-Display guard."""
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.session_table import SessionTable

        only_session = _make_record(session_slug='no-display-1', status=sr.Status.RUNNING)
        sr.write_record(only_session, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            table.focus()
            await pilot.pause()

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == []


class TestSpawnTreeToggle:
    @pytest.mark.timeout(10)
    async def test_toggle_key_closes_an_already_open_tree(self, tmp_path):
        """'t' toggles: pressing it again while the tree is open closes it
        (back to the default screen) rather than leaving it open or
        pushing a second modal on top (Fleet Cockpit C9a, PRD §9)."""
        from cockpit.app import CockpitApp
        from cockpit.panes.spawn_tree import SpawnTreeScreen

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()
            assert isinstance(app.screen, SpawnTreeScreen)

            await pilot.press('t')
            await pilot.pause()
            assert not isinstance(app.screen, SpawnTreeScreen)

    @pytest.mark.timeout(10)
    async def test_escape_closes_the_open_tree(self, tmp_path):
        """'escape' also dismisses the open spawn-tree modal, mirroring
        SpawnScreen.action_cancel's own escape-to-dismiss binding."""
        from cockpit.app import CockpitApp
        from cockpit.panes.spawn_tree import SpawnTreeScreen

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('t')
            await pilot.pause()
            assert isinstance(app.screen, SpawnTreeScreen)

            await pilot.press('escape')
            await pilot.pause()
            assert not isinstance(app.screen, SpawnTreeScreen)


class TestRefreshWriteDiscipline:
    @pytest.mark.timeout(10)
    async def test_refresh_path_writes_nothing_under_sessions_or_decisions(self, tmp_path):
        """Extends C5a's TestWriteDiscipline (PRD §2/§5 hard invariant) to the
        C5b queue/attention path: building/reordering the DecisionQueue and
        signaling an attention transition (set_urgency/reorder, PRD B3) on a
        refresh tick must never create, modify, or delete a sessions/ or
        decisions/ file. Only the explicit b/B/x/digit action handlers are
        sanctioned registry writers (TestBoostReordersAndPersists/
        TestDropRemovesAndPersists) -- refresh_registry (and everything it
        calls: _rebuild_queue, _update_attention) is a pure reader + a
        backend-signal emitter, never a writer.

        The session's RUNNING->AWAITING_INPUT flip below is this test's OWN
        setup write (sr.write_record), standing in for an external actor
        (e.g. a T4-T7 hook) -- mirrors TestSignalDontMove's convention. The
        write-discipline snapshot is taken AFTER that flip and BEFORE the
        direct refresh_registry() call, so the diff below isolates exactly
        what the automatic refresh/diff path itself writes: nothing.
        """
        from cockpit.app import CockpitApp
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display = sr.Display(kind='wm', wm_title='unblock:df#2085 slug')
        running = _make_record(
            session_slug='refresh-wd-1', status=sr.Status.RUNNING, display=display
        )
        sr.write_record(running, root=tmp_path)

        decision = sr.DecisionRecord(
            id='dec-refresh-wd-1',
            project='df',
            text='Which port?',
            filed_at='2026-07-07T00:00:00+00:00',
        )
        assert sr.write_decision(decision, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1  # the open decision only; session is still just RUNNING

            awaiting = _make_record(
                session_slug='refresh-wd-1',
                status=sr.Status.AWAITING_INPUT,
                display=display,
                question=sr.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
            )
            sr.write_record(awaiting, root=tmp_path)

            before = _snapshot_tree(tmp_path)
            # sanity: prove the snapshot actually captured the seeded files,
            # so the "unchanged" assertion below isn't vacuously true.
            assert any(path.startswith('sessions/') for path in before)
            assert any(path.startswith('decisions/') for path in before)

            app.refresh_registry()
            await pilot.pause()

            # sanity: the refresh actually did the queue-rebuild + attention
            # work under test, not a snapshot-unchanged short-circuit no-op.
            assert queue.row_count == 2
            assert backend.set_urgency_calls

        after = _snapshot_tree(tmp_path)

        # Scoped to sessions/decisions -- unlike TestWriteDiscipline/
        # TestDeferResetsAge's whole-lifecycle before/after (taken pre-mount,
        # before cockpit-ui.json exists at all), this test's "before" is
        # taken mid-lifecycle, after cockpit-ui.json already exists (from
        # on_mount's own initial refresh_registry() call). cockpit-ui.json
        # is the cockpit's own sanctioned, unconditional write target (PRD
        # §2/§5) and may legitimately be rewritten by a table rebuild's
        # RowHighlighted repost (see session_table.py) on every tick --
        # that's a pre-existing C5a behavior this test doesn't police. The
        # invariant under test here is narrower and exactly what step-25
        # specifies: zero sessions/ or decisions/ writes from the automatic
        # refresh/diff path.
        for path, value in before.items():
            if path.startswith(('sessions/', 'decisions/')):
                assert after.get(path) == value, (
                    f'{path} was created/modified/removed by the refresh path'
                )
        new_paths = set(after) - set(before)
        assert not any(path.startswith(('sessions/', 'decisions/')) for path in new_paths)


class TestNonBlockingPoll:
    """Fixes the C10 tour F1 / esc-2303-1 freeze: at 10k+ sessions,
    scan_sessions()+list_decisions() ran synchronously on the poll timer and
    froze all input for ~4.5s. The threaded poll worker
    (_scan_registry_worker, @work(thread=True)) moves that I/O off the
    event-loop thread -- proven here without needing 10k real session dirs,
    by injecting a scanner that blocks on command and records which thread
    actually called it.
    """

    @pytest.mark.timeout(10)
    async def test_poll_scan_runs_off_thread_and_event_loop_stays_responsive(self, tmp_path):
        from cockpit.app import CockpitApp
        from cockpit.panes.spawn_tree import SpawnTreeScreen

        scanner = _BlockingScanner()
        # A large poll_interval keeps on_mount's own set_interval timer from
        # firing (and launching a second, uncontrolled worker) during this
        # test -- the only scan driven here is the explicit
        # _scan_registry_worker() call below.
        app = CockpitApp(fleet_root=tmp_path, poll_interval=60, scanner=scanner)
        async with app.run_test() as pilot:
            await pilot.pause()  # on_mount's synchronous scan: call #1, non-blocking

            app._scan_registry_worker()
            while not scanner.started.is_set():
                await asyncio.sleep(0.01)

            # The event loop is still alive and responsive while the scan
            # sits blocked on its own thread -- the user-observable half of
            # this fix: a keypress still opens the spawn-tree modal.
            await pilot.press('t')
            await pilot.pause()
            assert isinstance(app.screen, SpawnTreeScreen)

            # The scan actually executed off the main/UI thread.
            assert scanner.scan_idents[-1] != threading.main_thread().ident

            scanner.release()
            await app.workers.wait_for_complete()


class TestThreadedScanReachesUI:
    """Extends TestRefreshWriteDiscipline to the threaded poll path
    (_scan_registry_worker, see TestNonBlockingPoll): the off-thread scan's
    result must actually reach the UI via the call_from_thread(_apply_scan)
    hand-off, and that hand-off must remain a pure reader -- writing
    nothing under sessions/ or decisions/, exactly like the synchronous
    refresh_registry path. Uses the real (default) SessionScanner, not an
    injected fake, to prove the actual production wiring end-to-end.
    """

    @pytest.mark.timeout(10)
    async def test_worker_result_reaches_table_and_writes_nothing(self, tmp_path):
        from cockpit.app import CockpitApp
        from cockpit.panes.session_table import SessionTable

        first = _make_record(session_slug='thread-1', status=sr.Status.RUNNING)
        sr.write_record(first, root=tmp_path)

        # A large poll_interval keeps on_mount's own set_interval timer from
        # firing a second, uncontrolled worker mid-test -- the only scan
        # driven here is the explicit _scan_registry_worker() call below.
        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert table.row_count == 1

            second = _make_record(session_slug='thread-2', status=sr.Status.AWAITING_INPUT)
            sr.write_record(second, root=tmp_path)

            before = _snapshot_tree(tmp_path)
            # sanity: prove the snapshot actually captured the seeded files,
            # so the "unchanged" assertion below isn't vacuously true.
            assert any(path.startswith('sessions/') for path in before)

            worker = app._scan_registry_worker()
            await worker.wait()
            await pilot.pause()

            # proves the off-thread -> call_from_thread(_apply_scan)
            # hand-off actually reached the live SessionTable.
            assert table.row_count == 2

        after = _snapshot_tree(tmp_path)
        for path, value in before.items():
            if path.startswith(('sessions/', 'decisions/')):
                assert after.get(path) == value, (
                    f'{path} was created/modified/removed by the threaded refresh'
                )
        new_paths = set(after) - set(before)
        assert not any(path.startswith(('sessions/', 'decisions/')) for path in new_paths)


class _BlockingCountingScanner:
    """Fake SessionScanner for TestScanBackpressure's drop-tick case: counts
    poll-tick scan() calls and blocks on command, so the assertion that an
    overlapping tick is dropped doesn't need 10k real session dirs or a
    wall-clock race against an actually-slow scan.

    The FIRST scan() call (on_mount's synchronous initial scan) returns
    immediately, uncounted -- mirrors _BlockingScanner's mount-call carve
    out so mounting the app can never deadlock the event loop. Every call
    after that increments `poll_calls`, signals `started`, and blocks on
    `_release` until the test calls release().
    """

    def __init__(self) -> None:
        self.poll_calls = 0
        self.started = threading.Event()
        self._release = threading.Event()
        self._armed = False

    def scan(self) -> list:
        if not self._armed:
            self._armed = True
            return []
        self.poll_calls += 1
        self.started.set()
        self._release.wait()
        return []

    def release(self) -> None:
        self._release.set()


class _RaisingThenOkScanner:
    """Fake SessionScanner for TestScanBackpressure's wedge-safety case: the
    mount's initial scan succeeds, the next call (the poll tick under test)
    raises, and every call after that succeeds again. Proves a raising scan
    still clears _scan_in_flight, so the very next poll tick launches a
    fresh scan instead of being dropped forever.
    """

    def __init__(self) -> None:
        self.calls = 0

    def scan(self) -> list:
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError('boom')
        return []


class TestScanBackpressure:
    """C10 tour F1 / esc-2303-1 follow-through: _scan_in_flight (step-12)
    drops an overlapping poll tick instead of piling up worker threads --
    in production poll_interval (1.5s) is routinely shorter than a full
    10k+-session scan (~4.5s), so an unguarded _poll_registry would launch
    one thread per tick for the scan's entire duration. It also proves a
    raising scan still clears the flag afterward, so a single bad tick
    never wedges every later poll.
    """

    @pytest.mark.timeout(10)
    async def test_overlapping_tick_is_dropped_while_scan_in_flight(self, tmp_path):
        from cockpit.app import CockpitApp

        scanner = _BlockingCountingScanner()
        # A large poll_interval keeps on_mount's own set_interval timer from
        # firing an uncontrolled worker mid-test -- the only ticks driven
        # here are the explicit _poll_registry() calls below.
        app = CockpitApp(fleet_root=tmp_path, poll_interval=60, scanner=scanner)
        async with app.run_test() as pilot:
            await pilot.pause()  # mount's synchronous scan: call #1, non-blocking

            app._poll_registry()
            while not scanner.started.is_set():
                await asyncio.sleep(0.01)
            assert scanner.poll_calls == 1

            try:
                app._poll_registry()  # overlapping tick: must be dropped, not queued
                await asyncio.sleep(0.1)
                assert scanner.poll_calls == 1, 'a second scan started while one was in flight'
            finally:
                # Always unblock every scan() call this test may have started
                # -- including a second one the pre-fix code wrongly lets
                # through -- so a failed assertion above can never leave a
                # worker thread parked on _release forever (which would hang
                # the whole test run at interpreter shutdown, not just this
                # test's own timeout).
                scanner.release()
                await app.workers.wait_for_complete()

    @pytest.mark.timeout(10)
    async def test_scan_in_flight_clears_after_a_raising_scan(self, tmp_path):
        from textual.worker import WorkerFailed

        from cockpit.app import CockpitApp

        scanner = _RaisingThenOkScanner()
        app = CockpitApp(fleet_root=tmp_path, poll_interval=60, scanner=scanner)
        async with app.run_test() as pilot:
            await pilot.pause()  # mount's synchronous scan: call #1, succeeds

            app._poll_registry()  # call #2: raises inside the threaded worker
            with pytest.raises(WorkerFailed):
                await app.workers.wait_for_complete()

            app._poll_registry()  # call #3: only runs if the flag was cleared
            await app.workers.wait_for_complete()

            assert scanner.calls == 3
