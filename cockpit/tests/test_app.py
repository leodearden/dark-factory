"""Tests for cockpit.app — CockpitApp: the C5a TUI skeleton (session table + detail pane + poll).

Textual headless pilot tests (App.run_test / Pilot.pause) drive the few
integration signals this task requires: the table renders scanned records on
mount, the poll timer picks up on-disk changes, row selection renders the
detail pane, the pure-consumer write discipline holds end-to-end, and the
selected-slug restore seam round-trips across a remount. The bulk of the
underlying logic (glyph/title/age/order/detail/config) is covered by fast
deterministic unit tests elsewhere in this package -- see
test_session_table.py / test_detail_pane.py / test_registry_reader.py /
test_ui_config.py. Pilot tests split out by area live in the sibling
test_app_*.py modules.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from _cockpit_session_records import make_record
from _cockpit_tree_snapshot import snapshot_tree
from orchestrator import decision_registry as dr
from orchestrator import session_records
from textual.coordinate import Coordinate

from cockpit.app import CockpitApp

if TYPE_CHECKING:
    from textual.pilot import Pilot

    from cockpit.registry_reader import SessionScannerProtocol


def _count_ui_config_writes(monkeypatch) -> list[str | None]:
    """Record every save_ui_config call app.py makes, delegating to the real one.

    Returns the live list of persisted selected_slug values, newest last.
    The property under test in TestUIConfigWriteDebounce is a COUNT ("one
    write per burst", "zero per idle rebuild"), and an mtime/bytes diff via
    snapshot_tree cannot tell one write from four when every write carries
    identical content inside the same mtime granule -- exactly the shape of
    the repeated-rebuild case. Patching the name in cockpit.app (not in
    cockpit.ui_config) is what intercepts _persist_ui_config: app.py binds
    the symbol at import time and resolves it from module globals at call
    time. Delegating to the real function keeps the on-disk half genuine,
    so a test can still assert load_ui_config(...).selected_slug.
    """
    import cockpit.app as app_module
    from cockpit.ui_config import save_ui_config as _real

    recorded: list[str | None] = []

    def _counting(cfg, root=None):
        recorded.append(cfg.selected_slug)
        _real(cfg, root)

    monkeypatch.setattr(app_module, 'save_ui_config', _counting)
    return recorded


def _debounce_running_records() -> list:
    """Three RUNNING records with distinct, ascending start_ts (no ties).

    order_sessions sorts within a state rank by oldest start_ts first, so
    these keep a fixed relative row order however they are delivered to the
    app -- seeded on disk by _seed_debounce_fleet, or returned straight from
    an injected _CountingRecordsScanner.
    """
    return [
        make_record(
            session_slug=f'debounce-run-{n}',
            status=session_records.Status.RUNNING,
            start_ts=f'2026-07-07T0{n}:00:00+00:00',
        )
        for n in (1, 2, 3)
    ]


def _seed_debounce_fleet(tmp_path: Path) -> list:
    """Seed one AWAITING_INPUT record plus three RUNNING ones; return the RUNNING records.

    order_sessions ranks AWAITING_INPUT above RUNNING, so the blocked record
    pins row 0 and the three RUNNING records occupy rows 1..3 in the
    returned order. Every row index is therefore deterministic, which is
    what lets TestUIConfigWriteDebounce park a cursor on a known NON-zero
    row.
    """
    session_records.write_record(
        make_record(session_slug='debounce-blocked', status=session_records.Status.AWAITING_INPUT),
        root=tmp_path,
    )
    running = _debounce_running_records()
    for record in running:
        session_records.write_record(record, root=tmp_path)
    return running


class _CountingRecordsScanner:
    """Fake SessionScanner for TestUIConfigWriteDebounce's starvation case:
    counts scan() calls and returns a fixed record list.

    Unlike test_app_scan_pipeline.py::_BlockingCountingScanner it never blocks and needs no
    mount-call carve-out, because that test simulates an in-flight scan by
    setting the documented main-thread-only _scan_in_flight flag directly
    rather than by actually holding one -- the property under test there is
    purely _poll_registry's statement ORDER relative to that flag, nothing
    about threads. `calls` is what proves the tick really was dropped for
    scanning, so the test cannot silently degrade into exercising the
    ordinary path.
    """

    def __init__(self, records: list) -> None:
        self.records = records
        self.calls = 0

    def scan(self) -> list:
        self.calls += 1
        return list(self.records)


async def _assert_rebuilds_cost_one_write(tmp_path, monkeypatch, *, drive, rounds, tag):
    """Park a NON-row-0 cursor, force *rounds* real table rebuilds under it,
    and assert the whole run costs exactly ONE cockpit-ui.json write.

    Shared body for TestUIConfigWriteDebounce's two rebuild tests, which
    differ only in how a rebuild is DRIVEN: *drive* is an async
    ``(app, pilot)`` callable that lands one rebuild plus the flush a real
    poll tick would have run. Everything else -- the seeding, the
    non-zero-index parking, the registry rewrite that defeats the snapshot
    short-circuit, the observability trick and the assertions -- is
    identical between them, so it lives here once: a change to
    registry_reader._SNAPSHOT_FIELDS or to which fields
    session_table.format_title renders then lands in one place rather than
    having to be mirrored in two ~90-line bodies.

    *tag* distinguishes the escalation_id/title values each caller writes,
    so a failure message names the driver it came from.

    The closing ``recorded == [parked.session_slug]`` is itself the
    non-vacuity guard, because it fails from both sides: a gate that
    suppressed every write would leave ``[]``, and a write per rebuild would
    leave one entry per round.
    """
    from cockpit.panes.detail_pane import DetailPane
    from cockpit.panes.session_table import SessionTable

    running = _seed_debounce_fleet(tmp_path)
    parked = running[0]

    app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.query_one(SessionTable)
        detail = app.query_one(DetailPane)

        # A non-zero row, so every rebuild's clear() resets the cursor away
        # from the parked selection before move_cursor restores it --
        # asserted BEFORE the cursor is moved there.
        assert table.get_row_index(parked.session_slug) != 0
        table.move_cursor(row=table.get_row_index(parked.session_slug))
        await pilot.pause()
        assert table.highlighted_slug() == parked.session_slug

        # Installed after the parking move, so neither the mount's own
        # initial RowHighlighted nor the parking one is counted -- the
        # rebuilds are what's under test.
        recorded = _count_ui_config_writes(monkeypatch)

        for n in range(1, rounds + 1):
            escalation_id = f'esc-{tag}-{n}'
            # `title` and `escalation_id` are both
            # registry_reader._SNAPSHOT_FIELDS members, so rewriting them
            # defeats _apply_scan's snapshot-unchanged short-circuit;
            # status/start_ts stay put so order_sessions keeps the row order
            # -- and the parked cursor's non-zero index -- fixed across the
            # rebuild. escalation_id is bumped alongside the title purely so
            # the rebuild is OBSERVABLE below: session_table.format_title
            # renders 'role:project#task_id' and ignores record.title, so a
            # title change alone would show up nowhere and the sanity check
            # would be vacuous.
            session_records.write_record(
                make_record(
                    session_slug=parked.session_slug,
                    status=parked.status,
                    start_ts=parked.start_ts,
                    title=f'{tag}-title-{n}',
                    escalation_id=escalation_id,
                ),
                root=tmp_path,
            )
            await drive(app, pilot)

            # sanity: the rebuild really landed in the UI, so this is not
            # `rounds` short-circuited no-op refreshes.
            assert escalation_id in detail.rendered_text
            assert table.highlighted_slug() == parked.session_slug

        # The first round's flush persists the parked cursor (it IS a real
        # change against the empty on-disk baseline); every later flush
        # finds the file already holding that selection.
        assert recorded == [parked.session_slug]


class TestInitialRender:
    @pytest.mark.timeout(10)
    async def test_seeded_records_render_as_rows(self, tmp_path):
        from cockpit.panes.session_table import SessionTable

        r1 = make_record(session_slug='a-1', status=session_records.Status.RUNNING)
        r2 = make_record(session_slug='b-2', status=session_records.Status.AWAITING_INPUT)
        for r in (r1, r2):
            session_records.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            table = app.query_one(SessionTable)
            assert table.row_count == 2

            row = table.get_row('a-1')
            # 'df' on disk renders as 'dark_factory' (task 3812): the record
            # went through the real SessionScanner, so registry_reader's
            # read-boundary fold already canonicalized .project. This
            # assertion is the end-to-end evidence that session_table.py's
            # rendering sites are covered WITHOUT any edit to session_table.py.
            assert 'unblock:dark_factory#2085' in row


class TestSessionTableDefaultFilter:
    @pytest.mark.timeout(10)
    async def test_terminal_sessions_are_hidden_by_default_on_mount(self, tmp_path):
        """FINDING F2 (C10 tour, esc-2303-1): the default session-table view
        hides terminal (exited/failed-to-start) sessions on launch, so
        "tens, not 10k" holds even when the registry carries stale history."""
        from textual.widgets.data_table import RowDoesNotExist

        from cockpit.panes.session_table import SessionTable

        live = make_record(session_slug='live-1', status=session_records.Status.RUNNING)
        dead = make_record(session_slug='dead-1', status=session_records.Status.EXITED)
        for r in (live, dead):
            session_records.write_record(r, root=tmp_path)

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

        from cockpit.panes.session_table import SessionTable

        live = make_record(session_slug='live-1', status=session_records.Status.RUNNING)
        dead = make_record(session_slug='dead-1', status=session_records.Status.EXITED)
        for r in (live, dead):
            session_records.write_record(r, root=tmp_path)

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
        from cockpit.panes.session_table import SessionTable

        live = make_record(session_slug='live-1', status=session_records.Status.RUNNING)
        dead = make_record(session_slug='dead-1', status=session_records.Status.EXITED)
        for r in (live, dead):
            session_records.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            await pilot.press('h')
            await pilot.pause()

            assert table.get_row('dead-1')
            assert table.row_count == 2

            live2 = make_record(session_slug='live-2', status=session_records.Status.RUNNING)
            session_records.write_record(live2, root=tmp_path)
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
        idiom in test_app_scan_pipeline.py::TestNonBlockingPoll,
        test_app_scan_pipeline.py::TestThreadedScanReachesUI and
        test_app_scan_pipeline.py::TestScanBackpressure) -- _apply_scan has no protection against
        a stale threaded scan (one that read the registry before the
        blocked-1 write below) landing via call_from_thread AFTER this
        test's own direct refresh_registry() call and clobbering its
        fresher result back down to 1 row. Confirmed by direct repro: with a
        short real interval, an in-flight poll tick can race this exact
        write+refresh_registry() sequence under a loaded runner.
        """
        from cockpit.panes.session_table import SessionTable

        running = make_record(session_slug='running-1', status=session_records.Status.RUNNING)
        session_records.write_record(running, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert table.row_count == 1

            blocked = make_record(session_slug='blocked-1', status=session_records.Status.AWAITING_INPUT)
            session_records.write_record(blocked, root=tmp_path)

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
        from cockpit.panes.session_table import SessionTable, state_glyph

        running = make_record(session_slug='running-1', status=session_records.Status.RUNNING)
        session_records.write_record(running, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert state_glyph(session_records.Status.RUNNING) in table.get_row('running-1')

            session_records.update_status('running-1', root=tmp_path, status=session_records.Status.IDLE)

            app.refresh_registry()
            await pilot.pause()

            row = table.get_row('running-1')
            assert state_glyph(session_records.Status.IDLE) in row
            assert state_glyph(session_records.Status.RUNNING) not in row


class _ScanStartCountingScanner:
    """Delegates to a real SessionScanner and counts scan() entries, so a test
    can count the poll scans that began after a point in time. The app
    serialises poll scans (test_app_scan_pipeline.py::TestScanBackpressure), so the plain int
    needs no lock.
    """

    def __init__(self, inner: SessionScannerProtocol) -> None:
        self._inner = inner
        self.starts = 0

    def scan(self) -> list[session_records.SessionRecord]:
        self.starts += 1
        return self._inner.scan()


async def _wait_until_a_poll_scan_has_applied(
    pilot: Pilot, scanner: _ScanStartCountingScanner, *, since: int, poll_interval: float
) -> None:
    """Return once the first poll scan to begin after *since* has been applied.

    Poll scans never overlap, and each one's apply lands before the next can
    begin, so by the time a second scan has begun after *since* the first has
    been read and applied, unless it raised: the poll worker swallows errors
    (exit_on_error=False), so a scan that raises still counts but is never
    applied. The overlap and ordering rest on
    cockpit/src/cockpit/app.py::CockpitApp._scan_registry_worker marshalling
    _apply_scan BEFORE its finally clears _scan_in_flight. There is no time
    bound here: *poll_interval* only paces the re-checks, and the calling
    test's pytest timeout guards liveness only (a poll timer that never fires).
    """
    while scanner.starts < since + 2:
        await pilot.pause(poll_interval)


class TestTimerIntegration:
    """The one test driven by the real set_interval poll timer (see TestPollRefresh).

    Everything in TestPollRefresh drives refresh_registry() directly. This
    test alone proves on_mount's set_interval wiring reaches disk end-to-end.
    It waits on poll scans, not a wall-clock budget;
    _wait_until_a_poll_scan_has_applied says why two post-write scan starts
    bound the wait.
    """

    @pytest.mark.timeout(10)
    async def test_timer_tick_triggers_refresh(self, tmp_path):
        from cockpit.panes.session_table import SessionTable
        from cockpit.registry_reader import SessionScanner

        scanner = _ScanStartCountingScanner(SessionScanner(tmp_path))
        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05, scanner=scanner)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert table.row_count == 0

            record = make_record(session_slug='timer-1', status=session_records.Status.RUNNING)
            session_records.write_record(record, root=tmp_path)
            starts_at_write = scanner.starts

            await _wait_until_a_poll_scan_has_applied(
                pilot, scanner, since=starts_at_write, poll_interval=app.poll_interval
            )

            assert table.row_count == 1, (
                f'row_count={table.row_count} after '
                f'{scanner.starts - starts_at_write} poll scans began since the write; '
                'a second post-write scan only begins once the first has been applied'
            )


class TestRowSelectionDetail:
    @pytest.mark.timeout(10)
    async def test_selecting_row_renders_detail(self, tmp_path):
        from cockpit.panes.detail_pane import DetailPane
        from cockpit.panes.session_table import SessionTable

        # other-1 is blocked, so it (not target-1) is the default row-0
        # cursor -- the assertions below only pass if selecting target-1
        # actually drives the detail pane, not just whatever's highlighted
        # by default after mount.
        other = make_record(session_slug='other-1', status=session_records.Status.AWAITING_INPUT)
        target = make_record(
            session_slug='target-1',
            status=session_records.Status.RUNNING,
            task_id='2085',
            escalation_id='esc-99',
            question=session_records.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        for r in (other, target):
            session_records.write_record(r, root=tmp_path)

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
        from cockpit.panes.session_table import LiveSessions, SessionTable

        parent = make_record(session_slug='parent-1', parent_session_id=None)
        running_child = make_record(
            session_slug='child-running',
            parent_session_id='parent-1',
            status=session_records.Status.RUNNING,
        )

        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            table.replace_rows(
                LiveSessions(visible=[parent], total=1),
                datetime.fromisoformat('2026-07-07T00:00:00+00:00'),
                all_records=[parent, running_child],
            )
            await pilot.pause()

            assert table.row_count == 1
            assert table.get_row('parent-1')[5] == '1'


class TestWriteDiscipline:
    @pytest.mark.timeout(10)
    async def test_cockpit_writes_only_its_own_ui_config(self, tmp_path):
        """PRD §2/§5 hard invariant: C5a is a pure consumer of the session and
        decision registries. It may create/update its OWN cockpit-ui.json, but
        running the app end-to-end (mount, a poll tick, a row selection) must
        never create, modify, or delete a sessions/ or decisions/ file --
        those belong to spawn-claude.sh/T4-T7/C8/C5b, never the C5a view."""
        from cockpit.panes.session_table import SessionTable
        from cockpit.registry_reader import SessionScanner

        record = make_record(session_slug='write-disc-1', status=session_records.Status.RUNNING)
        session_records.write_record(record, root=tmp_path)

        decision = dr.DecisionRecord(
            id='dec-1',
            project='df',
            text='Which port?',
            filed_at='2026-07-07T00:00:00+00:00',
        )
        assert dr.write_decision(decision, root=tmp_path)

        before = snapshot_tree(tmp_path)
        # sanity: prove the snapshot actually captured the seeded files, so a
        # later "unchanged" assertion isn't vacuously true over an empty dict.
        assert any(path.startswith('sessions/') for path in before)
        assert any(path.startswith('decisions/') for path in before)

        scanner = _ScanStartCountingScanner(SessionScanner(tmp_path))
        app = CockpitApp(fleet_root=tmp_path, poll_interval=0.05, scanner=scanner)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            starts_at_mount = scanner.starts

            await _wait_until_a_poll_scan_has_applied(
                pilot, scanner, since=starts_at_mount, poll_interval=app.poll_interval
            )

            table.move_cursor(row=table.get_row_index('write-disc-1'))
            await pilot.pause()

        after = snapshot_tree(tmp_path)

        for path, value in before.items():
            assert after.get(path) == value, f'{path} was created/modified/removed by the cockpit'

        new_or_modified = set(after) - set(before)
        assert new_or_modified == {'cockpit-ui.json'}, (
            f'expected only cockpit-ui.json as a new path, got {new_or_modified}'
        )


class TestSelectedSlugRestore:
    @pytest.mark.timeout(10)
    async def test_selection_round_trips_into_a_fresh_mount(self, tmp_path):
        """on_mount restores the operator's place via
        load_ui_config(...).selected_slug -> SessionTable.select_slug --
        ui_config's whole reason to exist. Both halves were unit-tested;
        the seam joining them was not. Also note here that the persisted
        poll_interval is deliberately inert (task 4054 scope)."""
        from cockpit.panes.session_table import SessionTable
        from cockpit.ui_config import load_ui_config

        # blocked-1 is AWAITING_INPUT, so order_sessions makes IT the
        # default row-0 cursor -- target-1 can only be highlighted on the
        # second mount if the restore branch actually ran.
        blocked = make_record(session_slug='blocked-1', status=session_records.Status.AWAITING_INPUT)
        target = make_record(session_slug='target-1', status=session_records.Status.RUNNING)
        for r in (blocked, target):
            session_records.write_record(r, root=tmp_path)

        first = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with first.run_test() as pilot:
            await pilot.pause()
            table = first.query_one(SessionTable)
            assert table.get_row_index('target-1') != 0
            assert table.highlighted_slug() == 'blocked-1'

            table.move_cursor(row=table.get_row_index('target-1'))
            await pilot.pause()
            assert table.highlighted_slug() == 'target-1'

        # the on-disk half of the seam -- localizes a future failure to
        # save-side vs restore-side
        assert load_ui_config(tmp_path).selected_slug == 'target-1'

        second = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with second.run_test() as pilot:
            await pilot.pause()
            table = second.query_one(SessionTable)
            assert table.get_row_index('target-1') != 0
            assert table.highlighted_slug() == 'target-1'

    @pytest.mark.timeout(10)
    async def test_a_rebuild_that_moves_the_cursor_persists_the_new_slug_on_the_next_tick(
        self, tmp_path
    ):
        """A hard kill must not restore the operator to a session that is gone.

        The rebuild suppresses its own RowHighlighted reposts (see
        _rebuild_session_table), so no event reports a cursor move it causes
        itself; the next poll tick's flush must still persist it. Read back
        from disk INSIDE the app's lifetime -- after on_unmount the value is
        written either way, which is exactly the gap this pins.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.session_table import SessionTable
        from cockpit.ui_config import load_ui_config

        for slug, start in (('session-a', '00:00:00'), ('session-b', '00:01:00')):
            session_records.write_record(
                make_record(session_slug=slug, start_ts=f'2026-07-07T{start}+00:00'),
                root=tmp_path,
            )

        # a large poll_interval keeps on_mount's own timer from racing the
        # direct refresh_registry() below -- TestPollRefresh's convention
        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            async def tick():
                app._poll_registry()
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()

            table.move_cursor(row=table.get_row_index('session-b'))
            await pilot.pause()
            await tick()
            assert load_ui_config(tmp_path).selected_slug == 'session-b'

            # session-b exits, so filter_live_sessions drops it from the default
            # view and the rebuild's cursor lands on a DIFFERENT slug -- a move
            # no operator made, and no RowHighlighted now reports
            session_records.write_record(
                make_record(
                    session_slug='session-b',
                    status=session_records.Status.EXITED,
                    start_ts='2026-07-07T00:01:00+00:00',
                ),
                root=tmp_path,
            )
            app.refresh_registry()
            await pilot.pause()

            assert table.highlighted_slug() == 'session-a'
            assert app._selected_slug == 'session-a'
            # recorded, not written: the rebuild itself does no I/O
            assert load_ui_config(tmp_path).selected_slug == 'session-b'
            await tick()
            assert load_ui_config(tmp_path).selected_slug == 'session-a'

    @pytest.mark.timeout(10)
    async def test_persisted_slug_that_no_longer_exists_degrades_to_row_zero(self, tmp_path):
        """A session selected before a restart may well have exited by the
        next launch; select_slug returns False rather than raising, so the
        cockpit must still mount and land on row 0. This test is a
        fail-soft regression guard, NOT a second proof of the restore --
        it passes with or without the restore branch, by design."""
        from cockpit.panes.session_table import SessionTable
        from cockpit.ui_config import CockpitUIConfig, save_ui_config

        save_ui_config(CockpitUIConfig(selected_slug='gone-1'), tmp_path)
        # Two live records, ranked by order_sessions (AWAITING_INPUT above
        # RUNNING), so "landed on row 0" is actually observable below -- with
        # only one record, row 0 is the only row and the assertion would pass
        # no matter where the cursor really landed.
        live = make_record(session_slug='live-1', status=session_records.Status.AWAITING_INPUT)
        other = make_record(session_slug='live-2', status=session_records.Status.RUNNING)
        for r in (live, other):
            session_records.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert table.get_row_index('live-2') != 0
            assert table.highlighted_slug() == 'live-1'


class TestUIConfigWriteDebounce:
    """cockpit-ui.json is written on a DEBOUNCED schedule -- a poll tick
    (CockpitApp._flush_ui_config) plus on_unmount's unconditional final
    write -- never once per cursor move and never once per table rebuild.

    Before this, on_data_table_row_highlighted called _persist_ui_config
    directly, so holding an arrow key down over a large session table did a
    full synchronous atomic write
    (cockpit/src/cockpit/ui_config.py::save_ui_config) per keypress on the
    event-loop thread, and CockpitApp._resync_session_detail wrote again
    whenever a rebuild moved the cursor.

    The gate these tests pin is CockpitApp._selected_slug vs
    CockpitApp._persisted_selected_slug -- "is the on-disk file stale", not
    "did the selection change since the last flush" -- so a selection that
    moves and comes back between ticks nets to no write, whichever seam
    recorded it.

    The round trip these tests must not break is proven next door:
    TestSelectedSlugRestore (a selection survives a remount) and
    TestWriteDiscipline (cockpit-ui.json is still created, and is still the
    ONLY file the cockpit writes). Both stay green because on_unmount's
    write remains unconditional -- see CockpitApp.on_unmount.
    """

    @pytest.mark.timeout(10)
    async def test_highlight_burst_writes_nothing_until_shutdown(self, tmp_path, monkeypatch):
        """A burst of cursor moves does ZERO I/O while the app runs; the one
        write that lands is on_unmount's, carrying the LAST slug."""
        from cockpit.panes.session_table import SessionTable
        from cockpit.ui_config import load_ui_config, ui_config_path

        running = _seed_debounce_fleet(tmp_path)
        slugs = [record.session_slug for record in running]

        # A large poll_interval keeps on_mount's own set_interval timer from
        # firing a flush tick mid-test (the idiom
        # test_app_scan_pipeline.py::TestNonBlockingPoll and
        # test_app_scan_pipeline.py::TestScanBackpressure use), so the ONLY thing that can resolve the
        # debounce here is the unmount write below.
        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            # Installed AFTER the mount, so the mount's own initial
            # RowHighlighted is never counted -- the burst is what's under test.
            recorded = _count_ui_config_writes(monkeypatch)

            for slug in slugs:
                table.move_cursor(row=table.get_row_index(slug))
                await pilot.pause()
                # sanity: the highlight really fired, so the empty-`recorded`
                # assertion below is not vacuously true over a still cursor.
                assert table.highlighted_slug() == slug

            assert recorded == []
            assert not ui_config_path(tmp_path).exists()

        # Exactly one write for the whole run: on_unmount's unconditional one.
        assert recorded == [slugs[-1]]
        # The on-disk half -- localizes a future failure to save-side vs
        # restore-side, same convention as TestSelectedSlugRestore.
        assert load_ui_config(tmp_path).selected_slug == slugs[-1]

    @pytest.mark.timeout(10)
    async def test_rebuild_that_leaves_the_selection_unchanged_writes_once(
        self, tmp_path, monkeypatch
    ):
        """However many rebuilds run under a parked NON-row-0 cursor,
        exactly ONE write lands -- the flush that first persists the parked
        cursor -- because every later flush finds the file already holding
        that selection.

        Each round FLUSHES rather than merely checking that a rebuild does
        no I/O of its own: SessionTable.replace_rows' clear() resets the
        cursor to row 0 before move_cursor puts it back, so a rebuild whose
        cursor events ever reached a selection seam would move the live
        selection away and back. Only a flush after the rebuild can tell a
        comparison against what was last PERSISTED apart from a latch keyed
        on "the slug changed", which that excursion would set.

        This is the DETERMINISTIC variant: refresh_registry() rebuilds
        in-thread, so a failure here is never a threading artefact.
        test_rebuild_bearing_poll_ticks_write_once is the same assertion
        through the real threaded poll path.
        """

        async def drive(app, pilot):
            app.refresh_registry()
            await pilot.pause()
            # The flush a real poll tick would have run (this variant drives
            # refresh_registry directly, so it must do it here).
            app._flush_ui_config()

        await _assert_rebuilds_cost_one_write(
            tmp_path, monkeypatch, drive=drive, rounds=3, tag='rebuild'
        )

    @pytest.mark.timeout(10)
    async def test_rebuild_bearing_poll_ticks_write_once(self, tmp_path, monkeypatch):
        """The end-to-end busy-fleet case, through the real threaded poll
        path: a live fleet churns, so most ticks DO rebuild the table (see
        test_rebuild_that_leaves_the_selection_unchanged_writes_once for
        the clear()/move_cursor excursion each one makes). Four rebuild-bearing ticks over
        a selection the operator never touches cost exactly ONE write --
        the one that persists the parked cursor -- not one per tick.

        Unlike the deterministic variant above, nothing here calls
        _flush_ui_config by hand: _poll_registry's own flush is what must
        fire, so this also pins that the debounce resolves on the real
        timer callback and not only when a test drives it.
        """

        async def drive(app, pilot):
            app._poll_registry()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

        await _assert_rebuilds_cost_one_write(
            tmp_path, monkeypatch, drive=drive, rounds=4, tag='tick'
        )

    @pytest.mark.timeout(10)
    async def test_burst_is_flushed_once_by_the_next_poll_tick(self, tmp_path, monkeypatch):
        """The debounce resolves on the poll tick: a whole burst of cursor
        moves collapses into ONE write carrying the LAST slug, and a further
        tick over an unchanged selection writes nothing more -- so this is a
        real debounce, not one-write-per-keypress traded for
        one-write-per-interval forever."""
        from cockpit.panes.session_table import SessionTable
        from cockpit.ui_config import load_ui_config

        running = _seed_debounce_fleet(tmp_path)
        slugs = [record.session_slug for record in running]

        # poll_interval=60 keeps on_mount's own set_interval timer from
        # firing: every tick in this test is an explicit _poll_registry()
        # call, which is what makes the write COUNT exact rather than a race
        # against the wall clock.
        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            recorded = _count_ui_config_writes(monkeypatch)

            for slug in slugs:
                table.move_cursor(row=table.get_row_index(slug))
                await pilot.pause()
                assert table.highlighted_slug() == slug
            assert recorded == []  # precondition: the burst is still debounced

            # _poll_registry launches the real threaded scan worker, so drain
            # it the way test_app_scan_pipeline.py::TestThreadedScanReachesUI and
            # test_app_scan_pipeline.py::TestScanBackpressure do.
            app._poll_registry()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert recorded == [slugs[-1]]
            # The on-disk half, asserted while the app is still RUNNING --
            # this is the timing change: the file now appears at the tick,
            # not at the keypress.
            assert load_ui_config(tmp_path).selected_slug == slugs[-1]

            # A second tick with the registry left STATIC -- the simplest
            # case: _apply_scan's snapshot-unchanged short-circuit means no
            # rebuild at all. The flush compares the live selection against
            # the persisted one, so a tick over an unchanged selection
            # writes nothing. test_rebuild_bearing_poll_ticks_write_once is
            # the counterpart that keeps the registry CHANGING.
            app._poll_registry()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert recorded == [slugs[-1]]

    @pytest.mark.timeout(10)
    async def test_a_failed_write_advances_the_baseline_and_is_never_retried(
        self, tmp_path, monkeypatch
    ):
        """A FAILED save still advances _persisted_selected_slug, so the same
        selection is never retried. That is a deliberate trade-off, pinned
        here rather than left implicit in _flush_ui_config's docstring.

        cockpit/src/cockpit/ui_config.py::save_ui_config logs and swallows
        any exception and returns None either way, so _persist_ui_config cannot
        tell a failed write from a successful one and advances the baseline
        regardless. The retry CADENCE did materially change when the write
        left the highlight handler: a persistently unwritable fleet_root
        used to be retried on every cursor move, and is now retried never.
        Recovery is by CHANGE -- the next selection makes the two values
        differ again -- plus on_unmount's one unconditional final attempt,
        which fails the same way.

        Deliberately not "fixed" by having save_ui_config report success and
        gating the baseline on it: cockpit-ui.json is fail-soft UI state
        whose total loss costs the operator one restored cursor position,
        and a retry loop over a read-only fleet_root would put the
        synchronous atomic write back on every single tick -- reintroducing, in
        the worst case, exactly the per-tick I/O this debounce removed.
        """
        from cockpit.panes.session_table import SessionTable
        from cockpit.ui_config import ui_config_path

        running = _seed_debounce_fleet(tmp_path)
        parked, moved_to = running[0], running[1]

        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            async def tick():
                app._poll_registry()
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()

            table.move_cursor(row=table.get_row_index(parked.session_slug))
            await pilot.pause()
            assert table.highlighted_slug() == parked.session_slug

            recorded = _count_ui_config_writes(monkeypatch)
            # Break the REAL writer from here on with a real filesystem
            # fault: a directory occupying the config path makes every save
            # fail, as a full or read-only fleet_root would. Everything below
            # observes what production observes when the write cannot land.
            ui_config_path(tmp_path).mkdir()

            await tick()
            # The flush ran and attempted the write ...
            assert recorded == [parked.session_slug]
            # ... the write really failed, fail-soft: no exception reached
            # the event loop, and no file landed ...
            assert not ui_config_path(tmp_path).is_file()
            # ... and the baseline advanced anyway ("persisted" means handed
            # to the writer, not on disk), so further ticks over the same
            # selection stay silent: no retry.
            await tick()
            await tick()
            assert recorded == [parked.session_slug]

            # Recovery is by change: moving the cursor makes the live
            # selection differ from the baseline again, so the flush
            # attempts once more (and fails again, still fail-soft).
            table.move_cursor(row=table.get_row_index(moved_to.session_slug))
            await pilot.pause()
            assert table.highlighted_slug() == moved_to.session_slug
            await tick()
            assert recorded == [parked.session_slug, moved_to.session_slug]
            assert not ui_config_path(tmp_path).is_file()

        # on_unmount's unconditional final attempt is the only other write
        # this selection ever gets, and it fails the same way -- so a
        # cockpit run over an unwritable fleet_root leaves no file at all
        # rather than crashing or spinning.
        assert recorded == [
            parked.session_slug,
            moved_to.session_slug,
            moved_to.session_slug,
        ]
        assert not ui_config_path(tmp_path).is_file()

    @pytest.mark.timeout(10)
    async def test_flush_is_not_starved_by_the_scan_backpressure_drop(self, tmp_path, monkeypatch):
        """Pins that the flush runs BEFORE _poll_registry's
        `if self._scan_in_flight: return` drop-tick guard.

        In production a full 10k+-session scan (~4.5s, esc-2303-1) routinely
        outlasts the 1.5s poll interval, so for that scan's whole duration
        every tick takes the early return. A flush placed after the guard
        would be starved for as long as scans keep overlapping -- exactly
        the busy-fleet case where the operator is most likely to be moving
        the cursor -- and the selection would sit unpersisted until unmount.
        """
        from cockpit.panes.session_table import SessionTable

        running = _debounce_running_records()
        parked = running[1]  # row 1: a NON-zero index, so highlighting it fires
        scanner = _CountingRecordsScanner(running)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=60, scanner=scanner)
        async with app.run_test() as pilot:
            await pilot.pause()  # mount's synchronous scan
            table = app.query_one(SessionTable)
            recorded = _count_ui_config_writes(monkeypatch)

            assert table.get_row_index(parked.session_slug) != 0
            table.move_cursor(row=table.get_row_index(parked.session_slug))
            await pilot.pause()
            assert table.highlighted_slug() == parked.session_slug
            assert recorded == []

            calls_before = scanner.calls
            try:
                # Stand in for a slow 10k-session scan still running. The
                # flag is documented main-thread-only, and this file already
                # drives _poll_registry directly, so setting it beats parking
                # a real worker thread that a failed assertion could leave
                # wedged.
                app._scan_in_flight = True
                app._poll_registry()
                await pilot.pause()

                # The tick really WAS dropped for scanning -- without this
                # the test could silently degrade into exercising the
                # ordinary, un-starved path.
                assert scanner.calls == calls_before
                # ... and the flush still ran anyway.
                assert recorded == [parked.session_slug]
            finally:
                app._scan_in_flight = False

    @pytest.mark.timeout(10)
    async def test_mount_over_a_restored_selection_writes_nothing(self, tmp_path, monkeypatch):
        """Mounting over a cockpit-ui.json whose selection restores verbatim
        must not rewrite the file with the value just read off it. The flush
        gate asks "is the on-disk file stale", not "did the selection
        change", so a successful restore leaves the live selection already
        equal to the persisted one and the first tick writes nothing."""
        from cockpit.ui_config import CockpitUIConfig, save_ui_config

        running = _seed_debounce_fleet(tmp_path)
        restored = running[0]
        save_ui_config(
            CockpitUIConfig(selected_slug=restored.session_slug, poll_interval=60), tmp_path
        )

        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            recorded = _count_ui_config_writes(monkeypatch)

            # sanity: the restore really happened. Without it this test
            # would pass over a cockpit that simply landed on row 0 having
            # never restored anything.
            assert app._selected_slug == restored.session_slug

            app._poll_registry()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert recorded == []

    @pytest.mark.timeout(10)
    async def test_a_stale_restored_selection_is_still_corrected_by_the_first_flush(
        self, tmp_path, monkeypatch
    ):
        """Over-suppression guard.

        Gating on "does the live selection differ from what is on disk" must
        still CORRECT a cockpit-ui.json naming a session that no longer
        exists: the restore fails soft onto row 0, which differs from the
        stale file, so the first flush rewrites it. Over-suppression is the
        characteristic failure mode of that gate, and this pins it from the
        opposite side to
        test_mount_over_a_restored_selection_writes_nothing -- do not
        "simplify" it away as a duplicate.
        """
        from cockpit.ui_config import CockpitUIConfig, load_ui_config, save_ui_config

        _seed_debounce_fleet(tmp_path)
        save_ui_config(CockpitUIConfig(selected_slug='ghost-gone', poll_interval=60), tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            recorded = _count_ui_config_writes(monkeypatch)

            # SessionTable.select_slug catches RowDoesNotExist and returns
            # False without moving the cursor, so the cursor stays on row 0
            # -- _seed_debounce_fleet's AWAITING_INPUT record, which
            # order_sessions ranks above every RUNNING one.
            assert app._selected_slug == 'debounce-blocked'

            app._poll_registry()
            await pilot.pause()
            await app.workers.wait_for_complete()
            await pilot.pause()

            assert recorded == ['debounce-blocked']
            assert load_ui_config(tmp_path).selected_slug == 'debounce-blocked'


class TestDecisionQueueRender:
    @pytest.mark.timeout(10)
    async def test_open_decisions_and_awaiting_sessions_render_score_ordered(self, tmp_path):
        """The DecisionQueue widget renders every open decision + awaiting-input
        session (never a plain running session), rows keyed stably by
        decision/session identity, with the higher-scored item ranked above
        the lower one (PRD §9 C5b: rows `[score][age][project#task][question]`).
        """
        from textual.widgets.data_table import RowDoesNotExist

        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        running = make_record(session_slug='running-1', status=session_records.Status.RUNNING)
        awaiting = make_record(
            session_slug='awaiting-1',
            status=session_records.Status.AWAITING_INPUT,
            question=session_records.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        for r in (running, awaiting):
            session_records.write_record(r, root=tmp_path)

        low = dr.DecisionRecord(
            id='dec-low',
            project='df',
            text='Low priority question?',
            filed_at='2026-07-07T00:00:00+00:00',
            manual_boost=0,
        )
        high = dr.DecisionRecord(
            id='dec-high',
            project='df',
            text='High priority question?',
            filed_at='2026-07-07T00:00:00+00:00',
            manual_boost=5,
        )
        for d in (low, high):
            assert dr.write_decision(d, root=tmp_path)

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


    @pytest.mark.timeout(10)
    async def test_a_wide_terminal_gives_the_question_the_leftover_width(self, tmp_path):
        """The question column is sized from the width the other three columns
        leave, not from a hardcoded cap -- so a wide terminal shows a long
        question instead of parking most of the line in a blank gutter.

        At terminal width 200 the fixed columns render at their LABEL widths
        ('score' 5, 'age' 3, 'project#task' 12 -- each wider than its cell
        here), which with one padding cell either side is 26. The question
        column takes the remaining 200 - 26 - 2 == 172.

        filed_at is three days back rather than a fixed calendar date so the
        age cell stays two characters forever: a fixed date would eventually
        render '100d', widen the age column past its label and silently move
        this arithmetic by one.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        long_question = 'Which port should the worker bind to, and why that one? ' * 6
        assert len(long_question) > 300

        decision = dr.DecisionRecord(
            id='dec-wide',
            project='df',
            text=long_question,
            filed_at=(datetime.now(UTC) - timedelta(days=3)).isoformat(),
        )
        assert dr.write_decision(decision, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=0.05)
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.pause()

            queue = app.query_one(DecisionQueue)
            question_cell = queue.get_row('decision:dec-wide')[3]

            # far past the old hardcoded 60-cell cap, and exactly the leftover
            assert len(question_cell) > 60
            assert len(question_cell) == 172

            # the formatting contract survived the wider bound
            assert question_cell.endswith('\u2026')
            assert '\n' not in question_cell

            # no blank gutter and no overflow: the columns fill the width exactly
            assert queue.scrollable_content_region.width == 200
            assert queue.virtual_size.width == queue.scrollable_content_region.width


    @pytest.mark.timeout(10)
    async def test_a_terminal_resize_reflows_the_question_column_both_ways(self, tmp_path):
        """Narrowing the terminal reflows the question SHORTER rather than
        overflowing, and widening it back reflows longer -- with the
        operator's selection and the rendered ages untouched.

        Textual's auto-width only ever grows, so giving cells BACK is the
        half of this that a formatter-only cap cannot do. Widths: 200 - 26
        fixed - 2 padding == 172, and 100 - 26 - 2 == 72.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        long_question = 'Which port should the worker bind to, and why that one? ' * 6
        filed_at = (datetime.now(UTC) - timedelta(days=3)).isoformat()
        for decision_id, boost in (('dec-high', 5), ('dec-low', 0)):
            assert dr.write_decision(
                dr.DecisionRecord(
                    id=decision_id,
                    project='df',
                    text=long_question,
                    filed_at=filed_at,
                    manual_boost=boost,
                ),
                root=tmp_path,
            )

        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=0.05)
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.pause()

            queue = app.query_one(DecisionQueue)

            # park the operator on the second row, so a yanked cursor would show
            assert queue.select_key('decision:dec-low')
            assert queue.highlighted_key() == 'decision:dec-low'
            _, wide_age, _, wide_question = queue.get_row('decision:dec-low')
            assert len(wide_question) == 172

            await pilot.resize_terminal(100, 40)
            await pilot.pause()

            _, narrow_age, _, narrow_question = queue.get_row('decision:dec-low')

            # the column gave width back instead of ratcheting
            assert len(narrow_question) == 72
            assert len(narrow_question) < len(wide_question)

            # nothing overflows and no gutter opens at the new width
            assert queue.scrollable_content_region.width == 100
            assert queue.virtual_size.width == queue.scrollable_content_region.width

            # a window drag is not a selection change, and not a clock read
            assert queue.highlighted_key() == 'decision:dec-low'
            assert narrow_age == wide_age

            await pilot.resize_terminal(200, 40)
            await pilot.pause()

            _, rewide_age, _, rewide_question = queue.get_row('decision:dec-low')

            assert len(rewide_question) == len(wide_question)
            assert queue.virtual_size.width == queue.scrollable_content_region.width == 200
            assert queue.highlighted_key() == 'decision:dec-low'
            assert rewide_age == wide_age

    @pytest.mark.timeout(10)
    async def test_a_queue_that_grows_past_its_height_pays_for_its_scrollbar(self, tmp_path):
        """A rebuild that grows the queue past its visible height brings in a
        vertical scrollbar, and the columns must give it back its two cells.

        The bound is necessarily derived BEFORE the new rows land, so the
        measurement cannot see a scrollbar those rows are about to cause --
        and Textual posts no Resize for a scrollbar appearing, so the
        overflow would stand until some later content change happened to
        re-derive. Growing the queue (rather than seeding it full) is what
        puts the measurement on the wrong side of the scrollbar: seeded full,
        the rows are already in place when the first Resize measures.
        Widths: 200 - 26 fixed - 2 padding == 172 with no scrollbar, 170 once
        two cells go to one.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        long_question = 'Which port should the worker bind to, and why that one? ' * 6
        filed_at = (datetime.now(UTC) - timedelta(days=3)).isoformat()

        def write(index):
            assert dr.write_decision(
                dr.DecisionRecord(
                    id=f'dec-{index:02d}',
                    project='df',
                    text=long_question,
                    filed_at=filed_at,
                ),
                root=tmp_path,
            )

        write(0)

        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=60)
        async with app.run_test(size=(200, 20)) as pilot:
            await pilot.pause()

            queue = app.query_one(DecisionQueue)

            # one row in 20 lines: no scrollbar, so the question gets all 172
            assert queue.scrollable_content_region.width == 200
            assert len(queue.get_row('decision:dec-00')[3]) == 172

            for index in range(1, 60):
                write(index)
            app.refresh_registry()
            await pilot.pause()

            # 60 rows in 20 lines: the scrollbar showed up, and was paid for
            assert queue.row_count == 60
            assert queue.scrollable_content_region.width == 198
            assert queue.virtual_size.width == queue.scrollable_content_region.width
            assert len(queue.get_row('decision:dec-00')[3]) == 170

    @pytest.mark.timeout(10)
    async def test_a_height_only_resize_does_not_reflow_and_keeps_the_scroll(self, tmp_path):
        """A resize that leaves the question bound alone must cost nothing on
        screen: a reflow re-declares the columns through clear(columns=True),
        which resets the scroll position, so an operator reading row 30 of a
        long queue would be thrown back to the top by a height-only drag.

        60 rows scroll at both heights, so the scrollbar -- and with it the
        derived bound -- is identical before and after; only the guard keeps
        the reflow from running anyway.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        long_question = 'Which port should the worker bind to, and why that one? ' * 6
        filed_at = (datetime.now(UTC) - timedelta(days=3)).isoformat()
        for index in range(60):
            assert dr.write_decision(
                dr.DecisionRecord(
                    id=f'dec-{index:02d}',
                    project='df',
                    text=long_question,
                    filed_at=filed_at,
                ),
                root=tmp_path,
            )

        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=0.05)
        async with app.run_test(size=(200, 20)) as pilot:
            await pilot.pause()

            queue = app.query_one(DecisionQueue)
            queue.scroll_to(y=10, animate=False)
            await pilot.pause()
            assert queue.scroll_y == 10

            await pilot.resize_terminal(200, 15)
            await pilot.pause()

            # the bound did not move, so no reflow ran and the scroll survived
            assert queue.scrollable_content_region.width == 198
            assert len(queue.get_row('decision:dec-00')[3]) == 170
            assert queue.scroll_y == 10


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
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display = session_records.Display(kind='wm', wm_title='unblock:df#2085 slug')
        running = make_record(session_slug='running-1', status=session_records.Status.RUNNING, display=display)
        session_records.write_record(running, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 0

            awaiting = make_record(
                session_slug='running-1',
                status=session_records.Status.AWAITING_INPUT,
                display=display,
                question=session_records.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
            )
            session_records.write_record(awaiting, root=tmp_path)

            app.refresh_registry()
            await pilot.pause()

            target = DisplayTarget(kind='wm', wm_title='unblock:df#2085 slug')
            assert queue.row_count == 1
            assert (target, True) in backend.set_urgency_calls
            assert backend.focus_calls == []
            assert backend.tile_calls == []

            idle = make_record(session_slug='running-1', status=session_records.Status.IDLE, display=display)
            session_records.write_record(idle, root=tmp_path)

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
        from cockpit.backends import DisplayTarget, FakeBackend

        display = session_records.Display(kind='wm', wm_title='shared title')
        awaiting = make_record(
            session_slug='shared-1',
            status=session_records.Status.AWAITING_INPUT,
            display=display,
            question=session_records.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        session_records.write_record(awaiting, root=tmp_path)
        decision = dr.DecisionRecord(
            id='dec-shared',
            project='df',
            text='Proceed?',
            filed_at='2026-07-07T00:00:00+00:00',
            session_id='shared-1',
        )
        assert dr.write_decision(decision, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            target = DisplayTarget(kind='wm', wm_title='shared title')
            assert (target, True) in backend.set_urgency_calls

            # The decision resolves elsewhere (e.g. a C8 watcher answers it)
            # while the session it's linked to is STILL awaiting input.
            dr.update_decision_state('dec-shared', dr.DecisionState.ANSWERED, root=tmp_path)

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
        whatever happens to sit at row 0. This is the ONLY place a focus()
        call may originate from (see TestSignalDontMove for the refresh
        path's complementary "never focus" half of the same invariant).
        """
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display_a = session_records.Display(kind='wm', wm_title='awaiting-a title')
        display_b = session_records.Display(kind='wm', wm_title='awaiting-b title')
        awaiting_a = make_record(
            session_slug='awaiting-a',
            status=session_records.Status.AWAITING_INPUT,
            display=display_a,
            question=session_records.Question(text='A?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        awaiting_b = make_record(
            session_slug='awaiting-b',
            status=session_records.Status.AWAITING_INPUT,
            display=display_b,
            question=session_records.Question(text='B?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        for r in (awaiting_a, awaiting_b):
            session_records.write_record(r, root=tmp_path)

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

    @pytest.mark.timeout(10)
    async def test_enter_on_unresolvable_target_is_fail_soft(self, tmp_path):
        """A queue row with no resolvable focus target (an open decision
        with no linked session) must not focus anything and must not raise
        -- fail-soft, PRD §2. A gone/unlinked lead is still a no-op, never
        a crash.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        orphan = dr.DecisionRecord(
            id='dec-orphan',
            project='df',
            text='Orphaned decision?',
            filed_at='2026-07-07T00:00:00+00:00',
        )
        assert dr.write_decision(orphan, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == []

    @staticmethod
    def _watcher_and_its_decision(root, *, record_slug):
        """A hand-launched watcher's record + the decision it filed (task 4237).

        session_id is the watcher's lease token -- never a registry key -- so
        only record_slug can link the row to the session.
        """
        watcher = make_record(
            session_slug='session-dark-factory-sess-watcher',
            display=session_records.Display(kind='wm', wm_title='escalation-watcher df'),
        )
        session_records.write_record(watcher, root=root)
        decision = dr.DecisionRecord(
            id='esc-4237-1',
            project='df',
            text='Adopt the plan?',
            filed_at='2026-07-07T00:00:00+00:00',
            session_id='watcher-df-1348600',
            record_slug=record_slug,
        )
        assert dr.write_decision(decision, root=root)
        return watcher, decision

    @pytest.mark.timeout(10)
    async def test_enter_on_a_watcher_filed_decision_focuses_the_watcher(self, tmp_path):
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        watcher, _ = self._watcher_and_its_decision(
            tmp_path, record_slug='session-dark-factory-sess-watcher'
        )

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.select_key('decision:esc-4237-1')
            await pilot.pause()

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == [
                DisplayTarget(kind='wm', wm_title='escalation-watcher df')
            ]

    @pytest.mark.timeout(10)
    async def test_a_record_slug_only_change_rebuilds_the_queues_focus_target(self, tmp_path):
        """Cross-queue enrichment can fill record_slug while every other
        queue-relevant field stays the first filer's, so the rebuild trigger
        itself must see record_slug -- not only resolve_target."""
        from dataclasses import replace

        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        _, unlinked = self._watcher_and_its_decision(tmp_path, record_slug='')

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            linked = replace(unlinked, record_slug='session-dark-factory-sess-watcher')
            assert dr.write_decision(linked, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.select_key('decision:esc-4237-1')
            await pilot.pause()

            await pilot.press('enter')
            await pilot.pause()

            assert backend.focus_calls == [
                DisplayTarget(kind='wm', wm_title='escalation-watcher df')
            ]


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
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.priority import Priorities

        fixed_now = datetime.fromisoformat('2026-07-07T00:00:00+00:00')

        older = dr.DecisionRecord(
            id='dec-a',
            project='df',
            text='A?',
            filed_at='2026-06-01T00:00:00+00:00',  # far enough back to saturate the age bonus
            manual_boost=0,
        )
        newer = dr.DecisionRecord(
            id='dec-b',
            project='df',
            text='B?',
            filed_at='2026-07-07T00:00:00+00:00',  # same instant as `now` -- age 0
            manual_boost=0,
        )
        for d in (older, newer):
            assert dr.write_decision(d, root=tmp_path)

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
            persisted = {d.id: d for d in dr.list_decisions(root=tmp_path)}
            assert persisted['dec-b'].manual_boost == 3
            assert persisted['dec-a'].manual_boost == 0  # untouched

            # a digit key sets manual_boost to that EXACT integer --
            # absolute, not additive (3 + 7 would be 10, not 7).
            await pilot.press('7')
            await pilot.pause()

            persisted = {d.id: d for d in dr.list_decisions(root=tmp_path)}
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

        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        first = dr.DecisionRecord(
            id='dec-1', project='df', text='First?', filed_at='2026-07-07T00:00:00+00:00'
        )
        second = dr.DecisionRecord(
            id='dec-2', project='df', text='Second?', filed_at='2026-07-07T00:00:00+00:00'
        )
        for d in (first, second):
            assert dr.write_decision(d, root=tmp_path)

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
            persisted = {d.id: d for d in dr.list_decisions(root=tmp_path)}
            assert persisted['dec-1'].state == dr.DecisionState.DROPPED
            assert persisted['dec-2'].state == dr.DecisionState.OPEN  # untouched

    @pytest.mark.timeout(10)
    async def test_drop_session_row_removes_live_without_writing_the_session(self, tmp_path):
        """'x' on a SESSION-backed row hides it from the queue in-memory
        ONLY -- sessions aren't cockpit-writable (PRD §2/§6.6), so this
        must not touch the session's record on disk, and must not raise.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        awaiting = make_record(
            session_slug='awaiting-1',
            status=session_records.Status.AWAITING_INPUT,
            question=session_records.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        session_records.write_record(awaiting, root=tmp_path)

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
            reread = session_records.read_record('awaiting-1', root=tmp_path)
            assert reread.status == session_records.Status.AWAITING_INPUT


class TestDecisionProjectIsCanonicalAtEveryReadSite:
    """The app reads decisions through registry_reader.scan_decisions at ALL
    THREE of its read sites (task 3812), not just the initial scan.

    _scan_registry is the obvious one, but action_boost and action_drop each
    RE-READ decisions from disk after their write, and the on-disk record
    stays raw (the cockpit is a pure consumer -- set_manual_boost/
    update_decision_state mutate one field and never renormalize .project).
    So a partial rewiring that folded only _scan_registry would hand back a
    canonical token on mount and then quietly refragment the queue on the
    very first boost or drop keypress -- and that is precisely the state in
    which an operator's project weight silently stops applying.
    """

    @pytest.mark.timeout(10)
    async def test_project_stays_canonical_across_the_boost_and_drop_re_reads(self, tmp_path):
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        decision = dr.DecisionRecord(
            id='dec-1',
            project='df',
            text='Which port?',
            filed_at='2026-07-07T00:00:00+00:00',
        )
        assert dr.write_decision(decision, root=tmp_path)
        # Baseline: the raw spelling is what is actually on disk, so each
        # re-read below really does read 'df' back and really does have to
        # fold it. (These are the legacy rows migrate_decision_project_tokens
        # has not been run over.)
        assert [d.project for d in dr.list_decisions(root=tmp_path)] == ['df']

        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            # (1) the initial scan (_scan_registry).
            assert [d.project for d in app._decisions] == ['dark_factory']

            queue = app.query_one(DecisionQueue)
            queue.move_cursor(row=queue.get_row_index('decision:dec-1'))
            await pilot.pause()

            # (2) the post-boost re-read (action_boost -> _apply_boost).
            await pilot.press('b')
            await pilot.pause()
            persisted = dr.list_decisions(root=tmp_path)
            assert persisted[0].manual_boost != 0  # the write really happened
            assert persisted[0].project == 'df'  # ...and left the disk raw
            assert [d.project for d in app._decisions] == ['dark_factory']

            # (3) the post-drop re-read (action_drop).
            await pilot.press('x')
            await pilot.pause()
            persisted = dr.list_decisions(root=tmp_path)
            assert persisted[0].state == dr.DecisionState.DROPPED  # the write happened
            assert persisted[0].project == 'df'  # ...and left the disk raw
            assert [d.project for d in app._decisions] == ['dark_factory']


class TestInjectedDecisionsReaderIsEveryDecisionRead:
    @pytest.mark.timeout(10)
    async def test_a_decision_the_reader_hides_stays_hidden_after_a_keypress_re_read(
        self, tmp_path
    ):
        """The decisions_reader kwarg is the app's only source of decisions, at the mount
        scan and at a keypress's post-write re-read alike. A read site that bypassed it
        would let a test fake see one registry while the app showed another."""
        from textual.widgets.data_table import RowDoesNotExist

        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.registry_reader import scan_decisions

        for decision_id in ('dec-shown', 'dec-hidden'):
            assert dr.write_decision(
                dr.DecisionRecord(
                    id=decision_id,
                    project='df',
                    text=f'{decision_id}?',
                    filed_at='2026-07-07T00:00:00+00:00',
                ),
                root=tmp_path,
            )

        def reader(root):
            return [d for d in scan_decisions(root) if d.id != 'dec-hidden']

        app = CockpitApp(
            fleet_root=tmp_path, backend=FakeBackend(), poll_interval=60, decisions_reader=reader
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1
            with pytest.raises(RowDoesNotExist):
                queue.get_row_index('decision:dec-hidden')

            queue.move_cursor(row=queue.get_row_index('decision:dec-shown'))
            await pilot.pause()
            await pilot.press('b')
            await pilot.pause()

            persisted = {d.id: d for d in dr.list_decisions(root=tmp_path)}
            assert persisted['dec-shown'].manual_boost == 1
            assert queue.row_count == 1
            with pytest.raises(RowDoesNotExist):
                queue.get_row_index('decision:dec-hidden')


class TestDroppedOverlayExpiresWithTheAsk:
    @pytest.mark.timeout(10)
    async def test_dropped_session_key_does_not_suppress_a_later_unrelated_ask(self, tmp_path):
        """self._dropped is an in-memory overlay keyed by QueueItem.key, and
        for a SESSION-backed row that key ('session:<slug>') is stable for
        the session's whole lifetime. Dropping it must therefore expire once
        the ask it was set against is gone -- otherwise a brand-new,
        completely unrelated question from the SAME session is silently
        suppressed for the cockpit's whole process lifetime, and the
        operator never learns an agent is blocked on them.

        The prune deliberately CANNOT use queue membership
        (`&= self._queue_items_by_key.keys()`) as its predicate: a dropped
        item is by construction absent from the rebuilt queue, so queue
        membership would clear every drop on the very rebuild action_drop
        itself triggers. Clause (b) below is the standing guard against that
        mistake -- a live drop must survive an unrelated rebuild.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display = session_records.Display(kind='wm', wm_title='drop title')
        first_ask = make_record(
            session_slug='drop-1',
            status=session_records.Status.AWAITING_INPUT,
            display=display,
            question=session_records.Question(text='First ask?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        session_records.write_record(first_ask, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1

            # (a) the operator drops the row -- it leaves the queue and the
            # overlay records the suppression.
            await pilot.press('x')
            await pilot.pause()
            assert queue.row_count == 0
            assert 'session:drop-1' in app._dropped

            # (b) GUARD: an unrelated session appears and forces a real
            # rebuild. The live drop must SURVIVE it -- this is exactly what
            # a copied `&= queue keys` prune would break.
            other = make_record(
                session_slug='other-1',
                status=session_records.Status.AWAITING_INPUT,
                display=session_records.Display(kind='wm', wm_title='other title'),
                question=session_records.Question(text='Unrelated ask?', asked_at='2026-07-07T01:00:00+00:00'),
            )
            session_records.write_record(other, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert queue.row_count == 1  # only other-1
            assert 'session:drop-1' in app._dropped

            # (c) the dropped session's ask resolves -- it stops awaiting
            # input entirely, so the overlay has nothing left to suppress.
            answered = make_record(
                session_slug='drop-1', status=session_records.Status.RUNNING, display=display
            )
            session_records.write_record(answered, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert 'session:drop-1' not in app._dropped

            # (d) the SAME session asks something brand new. It is a
            # different ask, so it must be visible to the operator again.
            second_ask = make_record(
                session_slug='drop-1',
                status=session_records.Status.AWAITING_INPUT,
                display=display,
                question=session_records.Question(text='Second ask?', asked_at='2026-07-08T00:00:00+00:00'),
            )
            session_records.write_record(second_ask, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert queue.get_row('session:drop-1')
            assert queue.row_count == 2

    @pytest.mark.timeout(10)
    async def test_dropped_session_key_expires_on_a_new_question_without_a_status_change(
        self, tmp_path
    ):
        """A bare status rule is NOT enough: the drop must also expire when
        the same session posts a brand-new question WITHOUT ever leaving
        AWAITING_INPUT.

        Reachability (the finding that motivates keying on ask identity
        rather than status): orchestrator.session_hooks.run_notification
        writes status=AWAITING_INPUT PLUS a fresh Question on every
        Notification hook, and does not require an intervening Stop hook
        (-> IDLE). So AWAITING_INPUT(Q1) -> AWAITING_INPUT(Q2) is a real
        transition an operator can hit, and a status-only expiry rule would
        still silently suppress Q2 -- the exact failure class this whole
        prune exists to kill.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display = session_records.Display(kind='wm', wm_title='renew title')
        first_ask = make_record(
            session_slug='renew-1',
            status=session_records.Status.AWAITING_INPUT,
            display=display,
            question=session_records.Question(text='First ask?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        session_records.write_record(first_ask, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1

            await pilot.press('x')
            await pilot.pause()
            assert queue.row_count == 0

            # A second Notification hook: STILL awaiting input (no Stop hook
            # in between), but asking something completely different.
            second_ask = make_record(
                session_slug='renew-1',
                status=session_records.Status.AWAITING_INPUT,
                display=display,
                question=session_records.Question(
                    text='Totally different ask?', asked_at='2026-07-08T00:00:00+00:00'
                ),
            )
            session_records.write_record(second_ask, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert 'session:renew-1' not in app._dropped
            assert queue.row_count == 1

    @pytest.mark.timeout(10)
    async def test_dropped_session_key_expires_on_a_reask_with_identical_text(self, tmp_path):
        """A re-ask carrying the SAME text as the dropped one must still expire it.

        Identical text is the COMMON case, not an edge case:
        orchestrator.session_hooks._extract_question sets
        `text=str(message)` verbatim from the Claude Code Notification
        hook's `message`, and `asked_at=datetime.now(UTC).isoformat()` -- a
        FRESH stamp on every hook. A repeated permission-prompt notification
        therefore carries its canned message string unchanged and differs
        from the previous ask ONLY in asked_at, which is exactly this shape.

        This is the case the sibling
        test_..._without_a_status_change cannot detect, because it also
        changes the question TEXT ('First ask?' -> 'Totally different
        ask?'): a text change moves build_snapshot's per-session tuple all
        by itself, so _apply_scan never short-circuits there. Here the whole
        difference lives in asked_at, so the prune only ever runs if the
        WAKE-UP TRIGGER (the snapshot) is at least as strong as the overlay
        identity (_ask_identity's `(question.text, question.asked_at)`).
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display = session_records.Display(kind='wm', wm_title='same title')
        first_ask = make_record(
            session_slug='same-1',
            status=session_records.Status.AWAITING_INPUT,
            display=display,
            question=session_records.Question(text='Continue?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        session_records.write_record(first_ask, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1

            await pilot.press('x')
            await pilot.pause()
            assert queue.row_count == 0
            assert 'session:same-1' in app._dropped

            # The next Notification hook: still AWAITING_INPUT, the very same
            # canned message text, only a fresh asked_at stamp. It is a
            # genuinely new ask and must be visible to the operator again.
            second_ask = make_record(
                session_slug='same-1',
                status=session_records.Status.AWAITING_INPUT,
                display=display,
                question=session_records.Question(text='Continue?', asked_at='2026-07-08T00:00:00+00:00'),
            )
            session_records.write_record(second_ask, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert 'session:same-1' not in app._dropped
            assert queue.row_count == 1

    @pytest.mark.timeout(10)
    async def test_dropped_session_key_expires_across_a_question_less_round_trip(self, tmp_path):
        """An AWAITING_INPUT session with NO question stamped is a real queue
        row (order_queue renders its question as ''), so it is droppable and
        its drop must expire too.

        Reachable shape: session_hooks._extract_question returns None for an
        absent/blank Notification `message` and deliberately leaves
        record.question untouched, so RUNNING -> AWAITING_INPUT(question=
        None) really happens.

        This pins the OBSERVED round trip -- the poll sees the intervening
        IDLE, so _ask_identity returns None on that tick and the drop
        expires there and then, before the second awaiting state is ever
        scanned. It is a regression pin rather than a defect reproduction:
        the sibling test below is the one that was red, since it is the
        identity itself (not the liveness gate) that has to do the work
        when the non-awaiting state is never observed.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display = session_records.Display(kind='wm', wm_title='quiet title')
        awaiting = make_record(
            session_slug='qless-1', status=session_records.Status.AWAITING_INPUT, display=display
        )
        assert awaiting.question is None
        session_records.write_record(awaiting, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1

            await pilot.press('x')
            await pilot.pause()
            assert queue.row_count == 0
            assert 'session:qless-1' in app._dropped

            idle = make_record(session_slug='qless-1', status=session_records.Status.IDLE, display=display)
            session_records.write_record(idle, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()
            assert 'session:qless-1' not in app._dropped

            # Asks again, still with nothing stamped -- must be visible.
            session_records.write_record(awaiting, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert 'session:qless-1' not in app._dropped
            assert queue.row_count == 1

    @pytest.mark.timeout(10)
    async def test_dropped_question_less_key_expires_when_the_record_itself_is_replaced(
        self, tmp_path
    ):
        """A question-less identity must not be a bare constant shared by every
        question-less ask in the fleet.

        A session slug outlives any one record: a reaped record can be
        recreated under the same slug (session_records's upsert path
        repopulates schema_version/session_slug/start_ts/status), and the
        cockpit's overlay keys are slug-derived. If a question-less awaiting
        session identifies as the same constant before and after, the
        operator's drop of the FIRST record silently suppresses the SECOND
        record's ask -- the exact failure class this machinery exists to
        kill, in the question-less corner. Keying it on the record's own
        (immutable, per-record) start_ts distinguishes them.

        The replacement record differs in `title` as well, which is what
        moves build_snapshot's tuple and so wakes the prune at all --
        start_ts is deliberately NOT a snapshot field (it would make a
        purely-time-passing tick diff). So this pins the IDENTITY's job,
        with the wake-up trigger supplied independently.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display = session_records.Display(kind='wm', wm_title='recycled title')
        first = make_record(
            session_slug='recycled-1',
            status=session_records.Status.AWAITING_INPUT,
            display=display,
            title='first run',
            start_ts='2026-07-07T00:00:00+00:00',
        )
        assert first.question is None
        session_records.write_record(first, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1

            await pilot.press('x')
            await pilot.pause()
            assert queue.row_count == 0
            assert 'session:recycled-1' in app._dropped

            second = make_record(
                session_slug='recycled-1',
                status=session_records.Status.AWAITING_INPUT,
                display=display,
                title='second run',
                start_ts='2026-07-09T00:00:00+00:00',
            )
            session_records.write_record(second, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert 'session:recycled-1' not in app._dropped
            assert queue.row_count == 1


class TestBoostAndDeferOverlaysExpireWithTheAsk:
    @pytest.mark.timeout(10)
    async def test_session_boost_expires_once_the_session_leaves_awaiting_input(self, tmp_path):
        """self._boosts leaks exactly like self._dropped: a SESSION boost is
        keyed by a slug-stable 'session:<slug>' key and nothing ever shrinks
        it, so a stale boost silently mis-ranks a brand-new, unrelated ask
        from the same session. It must expire under the same ask-liveness
        rule -- and self._overlay_asks, the bookkeeping side-table, must be
        garbage-collected along with it rather than becoming the new leak.
        """
        from cockpit.backends import FakeBackend

        display = session_records.Display(kind='wm', wm_title='boost title')
        awaiting = make_record(
            session_slug='boost-1',
            status=session_records.Status.AWAITING_INPUT,
            display=display,
            question=session_records.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        session_records.write_record(awaiting, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('b')
            await pilot.pause()
            assert 'session:boost-1' in app._boosts

            # The ask resolves -- the session leaves the queue entirely.
            idle = make_record(session_slug='boost-1', status=session_records.Status.IDLE, display=display)
            session_records.write_record(idle, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert 'session:boost-1' not in app._boosts
            assert app._overlay_asks == {}

    @pytest.mark.timeout(10)
    async def test_decision_defer_expires_once_the_decision_leaves_open(self, tmp_path):
        """self._deferred leaks the same way, and applies to BOTH kinds (a
        defer is uniform -- see action_defer). A decision key is live only
        while that decision is OPEN, so once it is answered elsewhere the
        stamp must expire rather than sit in memory forever suppressing the
        effective age of whatever later reuses that id.
        """
        from cockpit.backends import FakeBackend

        decision = dr.DecisionRecord(
            id='dec-defer',
            project='df',
            text='Proceed?',
            filed_at='2026-07-07T00:00:00+00:00',
        )
        assert dr.write_decision(decision, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            await pilot.press('d')
            await pilot.pause()
            assert 'decision:dec-defer' in app._deferred

            # Answered elsewhere (e.g. a C8 watcher) -- no longer OPEN.
            dr.update_decision_state('dec-defer', dr.DecisionState.ANSWERED, root=tmp_path)
            app.refresh_registry()
            await pilot.pause()

            assert 'decision:dec-defer' not in app._deferred
            assert app._overlay_asks == {}


class TestReorderTargetsAreDeduped:
    @pytest.mark.timeout(10)
    async def test_a_shared_target_is_passed_to_reorder_only_once(self, tmp_path):
        """A DecisionRecord and the AWAITING_INPUT session it links to
        resolve to the exact SAME DisplayTarget (resolve_target maps a
        decision through its session's display -- the case that method's own
        docstring calls out). _update_attention dedups the URGENCY half
        through a set, but builds its reorder list with a bare append per
        queue item, so backend.reorder() receives that one target twice.

        TmuxBackend.reorder assigns a running per-session index, so a
        duplicate consumes two indices, its second park fails against the
        already-vacated source (a warning), and the final compacted 0..N-1
        range is left with a gap. Each target must therefore be handed to
        reorder at most once. queue_items is score-ordered, so the dedup
        must keep the FIRST (highest-scoring) occurrence -- for tmux that
        surviving position IS the destination window index.

        Uses a tmux display because tmux is the only backend whose reorder
        is not a documented no-op.
        """
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display = session_records.Display(kind='tmux', tmux_target='s:1')
        awaiting = make_record(
            session_slug='shared-1',
            status=session_records.Status.AWAITING_INPUT,
            display=display,
            question=session_records.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        session_records.write_record(awaiting, root=tmp_path)
        decision = dr.DecisionRecord(
            id='dec-shared',
            project='df',
            text='Proceed?',
            filed_at='2026-07-07T00:00:00+00:00',
            session_id='shared-1',
        )
        assert dr.write_decision(decision, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()

            # Both items really are in the queue, so the duplicate target is
            # genuinely produced -- this isn't a vacuously-passing assertion.
            assert app.query_one(DecisionQueue).row_count == 2

            assert backend.reorder_calls[-1] == [DisplayTarget(kind='tmux', tmux_target='s:1')]

    @pytest.mark.timeout(10)
    async def test_dedup_keeps_the_highest_scoring_occurrence_of_a_repeated_target(
        self, tmp_path
    ):
        """The dedup must keep the FIRST occurrence, not just SOME occurrence.

        queue_items is score-descending, and for tmux the position a target
        holds in the reorder list IS its destination window index -- so
        keeping the first occurrence is what makes a shared target land at
        its highest-scoring item's position rather than its lowest-scoring
        one's. The sibling test above cannot see that: with a single
        distinct target, ordering is unobservable. Nor can the tmux
        backend's [s:1, s:1, s:0] case, whose surviving order is the same
        under keep-first and keep-last.

        Here the queue resolves to [A, B, A]: the two policies diverge --
        keep-first yields [A, B], keep-last [B, A] -- so a refactor to a
        keep-last dedup (or a set-then-sort) fails instead of passing
        silently while swapping which window lands at index 0.

        Scores are made deterministic with an injected now_fn, and the
        premise (the queue really is A, B, A before dedup) is asserted
        rather than assumed:
          - dec-a       severity='urgent' (weight 6.0)        -> ~2.87
          - session s-b asked_at 9d old, age term saturated   -> ~2.69
          - session s-a asked_at == now, no age term          -> ~2.60
        dec-a carries session_id='sess-a', so resolve_target maps it
        through sess-a's own display -- target A, the same object the
        sess-a row resolves to.
        """
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        now = datetime(2026, 7, 10, tzinfo=UTC)
        target_a = DisplayTarget(kind='tmux', tmux_target='s:0')
        target_b = DisplayTarget(kind='tmux', tmux_target='s:1')

        sess_a = make_record(
            session_slug='sess-a',
            status=session_records.Status.AWAITING_INPUT,
            display=session_records.Display(kind='tmux', tmux_target='s:0'),
            question=session_records.Question(text='Newest ask?', asked_at='2026-07-10T00:00:00+00:00'),
        )
        session_records.write_record(sess_a, root=tmp_path)
        sess_b = make_record(
            session_slug='sess-b',
            status=session_records.Status.AWAITING_INPUT,
            display=session_records.Display(kind='tmux', tmux_target='s:1'),
            question=session_records.Question(text='Older ask?', asked_at='2026-07-01T00:00:00+00:00'),
        )
        session_records.write_record(sess_b, root=tmp_path)
        decision = dr.DecisionRecord(
            id='dec-a',
            project='df',
            text='Proceed?',
            filed_at='2026-07-10T00:00:00+00:00',
            session_id='sess-a',
            severity='urgent',
        )
        assert dr.write_decision(decision, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05, now_fn=lambda: now)
        async with app.run_test() as pilot:
            await pilot.pause()

            assert app.query_one(DecisionQueue).row_count == 3
            # Premise: the pre-dedup target sequence really is [A, B, A].
            items = sorted(app._queue_items_by_key.values(), key=lambda i: (-i.score, i.key))
            assert [item.target for item in items] == [target_a, target_b, target_a]

            assert backend.reorder_calls[-1] == [target_a, target_b]


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
        TestWriteDiscipline's snapshot_tree before/after convention).

        A fixed now_fn + an explicit Priorities.default() keep the score
        comparison hermetic -- independent of wall-clock time and of
        whatever priorities.yaml (if any) happens to live on the host.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.priority import Priorities

        fixed_now = datetime.fromisoformat('2026-07-07T00:00:00+00:00')

        older = dr.DecisionRecord(
            id='dec-old',
            project='df',
            text='Old?',
            filed_at='2026-06-01T00:00:00+00:00',  # far enough back to saturate the age bonus
            manual_boost=0,
        )
        newer = dr.DecisionRecord(
            id='dec-new',
            project='df',
            text='New?',
            filed_at='2026-07-07T00:00:00+00:00',  # same instant as `now` -- age 0
            manual_boost=0,
        )
        for d in (older, newer):
            assert dr.write_decision(d, root=tmp_path)

        before = snapshot_tree(tmp_path)
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
        after = snapshot_tree(tmp_path)
        for path, value in before.items():
            assert after.get(path) == value, f'{path} was created/modified/removed by the defer'
        new_paths = set(after) - set(before)
        assert not any(path.startswith(('sessions/', 'decisions/')) for path in new_paths)


class TestSessionTableEnterFocus:
    @pytest.mark.timeout(10)
    async def test_enter_focuses_the_highlighted_session_row(self, tmp_path):
        """Enter on the session table's HIGHLIGHTED row (deliberately not
        the default row-0) focuses THAT session's terminal -- the C10 tour
        F6 fix (PRD §1: "Pressing Enter on a row raises that terminal"),
        mirroring test_app_spawn.py::TestSpawnTreeEnterFocus's contract at the top-level
        session-table Enter path.
        """
        from cockpit.backends import DisplayTarget, FakeBackend
        from cockpit.panes.session_table import SessionTable

        session_a = make_record(
            session_slug='session-a',
            status=session_records.Status.RUNNING,
            start_ts='2026-07-07T00:00:00+00:00',
            display=session_records.Display(kind='wm', wm_title='session-a title'),
        )
        session_b = make_record(
            session_slug='session-b',
            status=session_records.Status.RUNNING,
            start_ts='2026-07-07T01:00:00+00:00',
            display=session_records.Display(kind='wm', wm_title='session-b title'),
        )
        for r in (session_a, session_b):
            session_records.write_record(r, root=tmp_path)

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
        (PRD §2), mirroring test_app_spawn.py::TestSpawnTreeEnterFocus's no-Display guard."""
        from cockpit.backends import FakeBackend
        from cockpit.panes.session_table import SessionTable

        only_session = make_record(session_slug='no-display-1', status=session_records.Status.RUNNING)
        session_records.write_record(only_session, root=tmp_path)

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
        setup write (session_records.write_record), standing in for an external actor
        (e.g. a T4-T7 hook) -- mirrors TestSignalDontMove's convention. The
        write-discipline snapshot is taken AFTER that flip and BEFORE the
        direct refresh_registry() call, so the diff below isolates exactly
        what the automatic refresh/diff path itself writes: nothing.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue

        display = session_records.Display(kind='wm', wm_title='unblock:df#2085 slug')
        running = make_record(
            session_slug='refresh-wd-1', status=session_records.Status.RUNNING, display=display
        )
        session_records.write_record(running, root=tmp_path)

        decision = dr.DecisionRecord(
            id='dec-refresh-wd-1',
            project='df',
            text='Which port?',
            filed_at='2026-07-07T00:00:00+00:00',
        )
        assert dr.write_decision(decision, root=tmp_path)

        backend = FakeBackend()
        app = CockpitApp(fleet_root=tmp_path, backend=backend, poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            assert queue.row_count == 1  # the open decision only; session is still just RUNNING

            awaiting = make_record(
                session_slug='refresh-wd-1',
                status=session_records.Status.AWAITING_INPUT,
                display=display,
                question=session_records.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
            )
            session_records.write_record(awaiting, root=tmp_path)

            before = snapshot_tree(tmp_path)
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

        after = snapshot_tree(tmp_path)

        # Scoped to sessions/decisions -- unlike TestWriteDiscipline/
        # TestDeferResetsAge's whole-lifecycle before/after (taken pre-mount,
        # before cockpit-ui.json exists at all), this test's "before" is
        # taken mid-lifecycle and may or may not already include
        # cockpit-ui.json: the debounced flush writes it on a poll tick
        # (CockpitApp._flush_ui_config), so with poll_interval=0.05 whether
        # a tick has fired by then is not pinned here. It does not matter,
        # because cockpit-ui.json is the cockpit's own sanctioned write
        # target (PRD §2/§5) and this test does not police it either way; a
        # table rebuild never rewrites it, which TestUIConfigWriteDebounce::
        # test_rebuild_that_leaves_the_selection_unchanged_writes_once
        # polices. The invariant under test here is narrower and exactly what
        # step-25 specifies: zero sessions/ or decisions/ writes from the
        # automatic refresh/diff path.
        for path, value in before.items():
            if path.startswith(('sessions/', 'decisions/')):
                assert after.get(path) == value, (
                    f'{path} was created/modified/removed by the refresh path'
                )
        new_paths = set(after) - set(before)
        assert not any(path.startswith(('sessions/', 'decisions/')) for path in new_paths)


class TestDecisionQueueDetail:
    @pytest.mark.timeout(10)
    async def test_highlighting_a_decision_row_renders_its_full_question(self, tmp_path):
        """The task's whole signal: a decision's question is truncated to 60 chars
        in its queue row, so the detail pane is the only place it can be read in
        full -- and syncing it there must not disturb the session table's own
        detail sync or the selected-slug restore seam.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.panes.detail_pane import DetailPane
        from cockpit.panes.session_table import SessionTable

        # Two RUNNING sessions: neither joins the queue (order_queue takes
        # awaiting-input ones only), and the older start_ts takes session-table
        # row 0, so moving to session-target is a real move -- a move onto the
        # already-highlighted row posts no RowHighlighted at all.
        other_session = make_record(
            session_slug='session-other', start_ts='2026-07-07T00:00:00+00:00'
        )
        target_session = make_record(
            session_slug='session-target',
            start_ts='2026-07-07T00:01:00+00:00',
            task_id='7001',
            escalation_id='esc-7001-1',
            question=session_records.Question(text='Which port?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        for record in (other_session, target_session):
            session_records.write_record(record, root=tmp_path)

        # dec-first outscores the target (manual_boost, TestDecisionQueueRender's
        # own idiom) so the target is NOT the row highlighted at mount -- a move
        # onto the already-highlighted row posts no RowHighlighted, which would
        # make the assertion below vacuous.
        long_question = (
            'Should the reaper close this decision against the other escalation '
            'queue, or leave it open for the watcher to re-file it?'
        )
        assert len(long_question) > 60
        first = dr.DecisionRecord(
            id='dec-first', project='df', text='Short one?',
            filed_at='2026-07-07T00:00:00+00:00', manual_boost=5,
        )
        target = dr.DecisionRecord(
            id='dec-second', project='df', text=long_question,
            filed_at='2026-07-07T00:00:00+00:00',
            task_id='5449', escalation_id='esc-5449-2', severity='blocking',
        )
        for decision in (first, target):
            assert dr.write_decision(decision, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            detail = app.query_one(DetailPane)

            # (1) the move below is real: the target is not already highlighted
            assert queue.highlighted_key() != 'decision:dec-second'
            assert app._selected_slug == 'session-other'

            # the row itself cannot carry the question -- that is the defect
            assert long_question not in ' '.join(
                str(cell) for cell in queue.get_row('decision:dec-second')
            )

            # (2) highlighting the decision row renders its full detail
            assert queue.select_key('decision:dec-second')
            await pilot.pause()

            assert long_question in detail.rendered_text
            assert 'dec-second' in detail.rendered_text
            assert 'esc-5449-2' in detail.rendered_text

            # (4) the queue move left the session table's restore seam alone
            assert app._selected_slug == 'session-other'

            # (3) the session table's own sync still works, unclobbered
            table = app.query_one(SessionTable)
            table.move_cursor(row=table.get_row_index('session-target'))
            await pilot.pause()

            assert 'Which port?' in detail.rendered_text
            assert '7001' in detail.rendered_text
            assert 'esc-7001-1' in detail.rendered_text
            assert app._selected_slug == 'session-target'

    @pytest.mark.timeout(10)
    async def test_highlighting_a_session_queue_row_renders_that_session(self, tmp_path):
        """The queue mixes decisions and awaiting-input sessions and truncates both
        row kinds' questions, so both must populate the pane -- one list where some
        rows fill the pane and others silently don't is worse than either
        consistent alternative. The queue's session row still must not write
        _selected_slug: that restore seam is the session TABLE's alone.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.panes.detail_pane import DetailPane

        # Both awaiting, so both are queue rows; session-other's older start_ts
        # takes session-table row 0, so at mount the pane shows session-other and
        # the queue move below genuinely has to change it.
        awaiting = make_record(
            session_slug='session-awaiting',
            status=session_records.Status.AWAITING_INPUT,
            start_ts='2026-07-07T00:01:00+00:00',
            task_id='7002',
            question=session_records.Question(
                text='Which host should the worker bind to?',
                asked_at='2026-07-07T00:01:00+00:00',
            ),
        )
        other_session = make_record(
            session_slug='session-other',
            status=session_records.Status.AWAITING_INPUT,
            start_ts='2026-07-07T00:00:00+00:00',
            task_id='7003',
            question=session_records.Question(text='Unrelated?', asked_at='2026-07-07T00:00:00+00:00'),
        )
        for record in (awaiting, other_session):
            session_records.write_record(record, root=tmp_path)

        # outscores the session row, so the session row is not highlighted at mount
        decision = dr.DecisionRecord(
            id='dec-first', project='df', text='Short one?',
            filed_at='2026-07-07T00:00:00+00:00', manual_boost=5,
        )
        assert dr.write_decision(decision, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=0.05)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            detail = app.query_one(DetailPane)

            # the move is real: neither the queue cursor nor the pane is there yet
            assert queue.highlighted_key() != 'session:session-awaiting'
            assert 'Which host should the worker bind to?' not in detail.rendered_text
            assert app._selected_slug == 'session-other'

            assert queue.select_key('session:session-awaiting')
            await pilot.pause()

            assert 'Which host should the worker bind to?' in detail.rendered_text
            assert '7002' in detail.rendered_text
            assert app._selected_slug == 'session-other'

    @pytest.mark.timeout(10)
    async def test_a_registry_rebuild_does_not_yank_the_pane_off_a_decision(self, tmp_path):
        """Without this the fix is invisible on a live fleet: the first changed
        poll tick reverts the pane to session detail mid-read. A rebuild that
        shifts the session cursor's INDEX re-posts SessionTable RowHighlighted,
        and _rebuild_session_table additionally re-syncs the pane itself.

        A rebuild refreshes whichever pane kind currently owns the detail; only
        an operator cursor move transfers that ownership. The session table's
        own restore seam stays a function of the session table alone, rebuild
        or not.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.panes.detail_pane import DetailPane
        from cockpit.panes.session_table import SessionTable
        from cockpit.ui_config import load_ui_config

        for slug, start in (('session-a', '00:00:00'), ('session-b', '00:01:00')):
            session_records.write_record(
                make_record(session_slug=slug, start_ts=f'2026-07-07T{start}+00:00'),
                root=tmp_path,
            )

        long_question = (
            'Should the reaper close this decision against the other escalation '
            'queue, or leave it open for the watcher to re-file it?'
        )
        first = dr.DecisionRecord(
            id='dec-first', project='df', text='Short one?',
            filed_at='2026-07-07T00:00:00+00:00', manual_boost=5,
        )
        target = dr.DecisionRecord(
            id='dec-second', project='df', text=long_question,
            filed_at='2026-07-07T00:00:00+00:00',
        )
        for decision in (first, target):
            assert dr.write_decision(decision, root=tmp_path)

        # a large poll_interval keeps on_mount's own timer from racing the
        # direct refresh_registry() below -- TestPollRefresh's convention
        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            table = app.query_one(SessionTable)
            detail = app.query_one(DetailPane)

            assert queue.select_key('decision:dec-second')
            await pilot.pause()
            assert long_question in detail.rendered_text
            assert table.highlighted_slug() == 'session-a'

            # an awaiting-input session order_sessions ranks first, so the
            # highlighted session row's INDEX genuinely shifts on rebuild
            session_records.write_record(
                make_record(
                    session_slug='session-new',
                    status=session_records.Status.AWAITING_INPUT,
                    start_ts='2026-07-06T00:00:00+00:00',
                ),
                root=tmp_path,
            )
            app.refresh_registry()
            await pilot.pause()

            assert table.get_row_index('session-a') != 0

            # (a) the operator is still reading the decision they highlighted
            assert long_question in detail.rendered_text

            # (b) cockpit-ui.json's restore seam still tracks the session table
            assert table.highlighted_slug() == 'session-a'
            assert app._selected_slug == 'session-a'

        assert load_ui_config(tmp_path).selected_slug == 'session-a'

    @pytest.mark.timeout(10)
    async def test_at_mount_the_pane_shows_the_highlighted_session_not_the_queue(self, tmp_path):
        """A rebuild is programmatic, so it must never transfer the pane -- and the
        queue rebuild is a rebuild exactly as the session table's is.

        DecisionQueue.replace_rows does clear() + move_cursor, so it posts
        RowHighlighted on every rebuild whose highlighted row INDEX shifts --
        including the very first one, at mount. Left unprevented that hands the
        pane to the queue before the operator has touched anything, which is both
        a regression of startup behaviour and of the cockpit-ui.json restore (the
        restored slug's own repost fires only when its row index changes).
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.detail_pane import DetailPane
        from cockpit.panes.session_table import SessionTable

        # the older start_ts takes session-table row 0, so this is the session
        # whose detail the pane must be showing when the app settles
        session_records.write_record(
            make_record(
                session_slug='session-row0',
                start_ts='2026-07-07T00:00:00+00:00',
                question=session_records.Question(
                    text='AAA row-zero session question?', asked_at='2026-07-07T00:00:00+00:00'
                ),
            ),
            root=tmp_path,
        )
        session_records.write_record(
            make_record(session_slug='session-other', start_ts='2026-07-07T00:01:00+00:00'),
            root=tmp_path,
        )
        assert dr.write_decision(
            dr.DecisionRecord(
                id='dec-low', project='df', text='ZZZ decision question?',
                filed_at='2026-07-07T00:00:00+00:00',
            ),
            root=tmp_path,
        )

        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            detail = app.query_one(DetailPane)

            assert table.highlighted_slug() == 'session-row0'
            assert 'AAA row-zero session question?' in detail.rendered_text
            assert 'dec-low' not in detail.rendered_text
            assert 'ZZZ decision question?' not in detail.rendered_text

    @pytest.mark.timeout(10)
    async def test_a_new_decision_on_a_poll_tick_does_not_steal_the_session_pane(self, tmp_path):
        """The same hole on the path an operator actually meets it: a decision
        filed while they read a session row shifts the queue's highlighted INDEX,
        and the resulting repost yanks the pane out from under them.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.panes.detail_pane import DetailPane
        from cockpit.panes.session_table import SessionTable

        session_records.write_record(
            make_record(session_slug='session-a', start_ts='2026-07-07T00:00:00+00:00'),
            root=tmp_path,
        )
        session_records.write_record(
            make_record(
                session_slug='session-parked',
                start_ts='2026-07-07T00:01:00+00:00',
                question=session_records.Question(
                    text='BBB parked session question?', asked_at='2026-07-07T00:01:00+00:00'
                ),
            ),
            root=tmp_path,
        )
        assert dr.write_decision(
            dr.DecisionRecord(
                id='dec-existing', project='df', text='Short one?',
                filed_at='2026-07-07T00:00:00+00:00',
            ),
            root=tmp_path,
        )

        # a large poll_interval keeps on_mount's own timer from racing the
        # direct refresh_registry() below -- TestPollRefresh's convention
        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            table = app.query_one(SessionTable)
            detail = app.query_one(DetailPane)

            # park the operator on a session row that is NOT row 0, so the move
            # is real -- a move onto the already-highlighted row posts nothing
            assert table.highlighted_slug() == 'session-a'
            table.move_cursor(row=table.get_row_index('session-parked'))
            await pilot.pause()
            assert 'BBB parked session question?' in detail.rendered_text

            # a newly-filed, boosted decision lands ABOVE the existing one, so the
            # queue's highlighted row index genuinely shifts on the rebuild
            assert queue.get_row_index('decision:dec-existing') == 0
            assert dr.write_decision(
                dr.DecisionRecord(
                    id='dec-new', project='df', text='ZZZ brand new decision?',
                    filed_at='2026-07-07T00:02:00+00:00', manual_boost=9,
                ),
                root=tmp_path,
            )
            app.refresh_registry()
            await pilot.pause()

            assert queue.get_row_index('decision:dec-existing') == 1
            assert 'BBB parked session question?' in detail.rendered_text
            assert app._selected_slug == 'session-parked'

    @pytest.mark.timeout(10)
    async def test_a_terminal_resize_does_not_steal_the_pane_from_the_session_table(
        self, tmp_path
    ):
        """A window drag is not an operator selection, so it must not transfer
        the pane -- the same contract as a rebuild, on the one path the widget
        originates itself.

        A resize reflow re-enters the cursor through clear(columns=True) +
        move_cursor, so it posts RowHighlighted from inside DecisionQueue,
        where app.py's own prevent() block cannot reach it. Left unsuppressed
        that hands the pane to the queue mid-drag, off a session row the
        operator deliberately parked on.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.panes.detail_pane import DetailPane
        from cockpit.panes.session_table import SessionTable

        session_records.write_record(
            make_record(session_slug='session-a', start_ts='2026-07-07T00:00:00+00:00'),
            root=tmp_path,
        )
        session_records.write_record(
            make_record(
                session_slug='session-parked',
                start_ts='2026-07-07T00:01:00+00:00',
                question=session_records.Question(
                    text='BBB parked session question?', asked_at='2026-07-07T00:01:00+00:00'
                ),
            ),
            root=tmp_path,
        )
        long_question = 'ZZZ decision question? ' + ('padded out to overflow the column. ' * 8)
        filed_at = (datetime.now(UTC) - timedelta(days=3)).isoformat()
        for decision_id, boost in (('dec-high', 5), ('dec-low', 0)):
            assert dr.write_decision(
                dr.DecisionRecord(
                    id=decision_id,
                    project='df',
                    text=long_question,
                    filed_at=filed_at,
                    manual_boost=boost,
                ),
                root=tmp_path,
            )

        # a large poll_interval keeps on_mount's own timer out of the resize
        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=60)
        async with app.run_test(size=(200, 40)) as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            table = app.query_one(SessionTable)
            detail = app.query_one(DetailPane)

            # claim the pane for the QUEUE first, from a row that is not row 0
            assert queue.select_key('decision:dec-low')
            await pilot.pause()
            assert 'ZZZ decision question?' in detail.rendered_text

            # then hand it back to the SESSION table -- this is where the
            # operator is parked when the window gets dragged
            table.move_cursor(row=table.get_row_index('session-parked'))
            await pilot.pause()
            assert 'BBB parked session question?' in detail.rendered_text
            assert 'ZZZ decision question?' not in detail.rendered_text

            wide_question = queue.get_row('decision:dec-low')[3]

            await pilot.resize_terminal(100, 40)
            await pilot.pause()

            # the pane stayed where the operator put it
            assert 'BBB parked session question?' in detail.rendered_text
            assert 'ZZZ decision question?' not in detail.rendered_text

            # ... and the reflow did run, and restored the cursor by key
            assert len(queue.get_row('decision:dec-low')[3]) < len(wide_question)
            assert queue.highlighted_key() == 'decision:dec-low'


    @pytest.mark.timeout(10)
    async def test_a_queue_session_row_survives_and_refreshes_across_a_rebuild(self, tmp_path):
        """The pane belongs to whichever TABLE last moved, not to whichever record
        KIND is on screen.

        A queue SESSION row renders through the same show_record path the session
        table uses, so arbitrating on the rendered KIND hands the pane straight
        back to the session table's -- different -- highlighted session on the
        next rebuild. The parked session-table cursor below is load-bearing: a
        test that leaves both tables pointing at the same session passes for the
        wrong reason.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.panes.detail_pane import DetailPane
        from cockpit.panes.session_table import SessionTable

        awaiting = make_record(
            session_slug='session-awaiting',
            status=session_records.Status.AWAITING_INPUT,
            start_ts='2026-07-07T00:01:00+00:00',
            task_id='7002',
            question=session_records.Question(text='QQQ host bind?', asked_at='2026-07-07T00:01:00+00:00'),
        )
        session_records.write_record(awaiting, root=tmp_path)
        session_records.write_record(
            make_record(
                session_slug='session-other',
                start_ts='2026-07-07T00:00:00+00:00',
                task_id='7003',
                question=session_records.Question(text='Unrelated?', asked_at='2026-07-07T00:00:00+00:00'),
            ),
            root=tmp_path,
        )
        # outscores the session row, so the queue's session row is not row 0
        assert dr.write_decision(
            dr.DecisionRecord(
                id='dec-first', project='df', text='Short one?',
                filed_at='2026-07-07T00:00:00+00:00', manual_boost=5,
            ),
            root=tmp_path,
        )

        # a large poll_interval keeps on_mount's own timer from racing the
        # direct refresh_registry() below -- TestPollRefresh's convention
        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            table = app.query_one(SessionTable)
            detail = app.query_one(DetailPane)

            # park the session table on a DIFFERENT session than the queue row
            table.move_cursor(row=table.get_row_index('session-other'))
            await pilot.pause()
            assert app._selected_slug == 'session-other'

            assert queue.select_key('session:session-awaiting')
            await pilot.pause()
            assert 'QQQ host bind?' in detail.rendered_text

            # the queue's own row gets FRESH data on a rebuild -- neither stolen
            # by the session table's cursor nor left stale
            session_records.write_record(
                make_record(
                    session_slug='session-awaiting',
                    status=session_records.Status.AWAITING_INPUT,
                    start_ts='2026-07-07T00:01:00+00:00',
                    task_id='7002',
                    question=session_records.Question(
                        text='QQQ host bind, REVISED?', asked_at='2026-07-07T00:03:00+00:00'
                    ),
                ),
                root=tmp_path,
            )
            app.refresh_registry()
            await pilot.pause()

            assert 'QQQ host bind, REVISED?' in detail.rendered_text
            assert '7003' not in detail.rendered_text
            assert app._selected_slug == 'session-other'

    @pytest.mark.timeout(10)
    async def test_a_queue_decision_row_survives_and_refreshes_across_a_rebuild(self, tmp_path):
        """The decision-row counterpart of the test above, as a regression guard:
        a session-table reordering must not pull the pane off the queue's
        highlighted decision either. Already green -- it stays that way.

        Asserted on REVISED text, like its session-row sibling: "the question is
        still on screen" alone would also pass for a pane re-rendered from a
        stale self._decisions, since an unchanged question is exactly what a
        stale render shows.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.panes.detail_pane import DetailPane
        from cockpit.panes.session_table import SessionTable

        for slug, start in (('session-a', '00:00:00'), ('session-b', '00:01:00')):
            session_records.write_record(
                make_record(session_slug=slug, start_ts=f'2026-07-07T{start}+00:00'),
                root=tmp_path,
            )
        long_question = (
            'Should the reaper close this decision against the other escalation '
            'queue, or leave it open for the watcher to re-file it?'
        )
        for decision in (
            dr.DecisionRecord(
                id='dec-first', project='df', text='Short one?',
                filed_at='2026-07-07T00:00:00+00:00', manual_boost=5,
            ),
            dr.DecisionRecord(
                id='dec-second', project='df', text=long_question,
                filed_at='2026-07-07T00:00:00+00:00',
            ),
        ):
            assert dr.write_decision(decision, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            table = app.query_one(SessionTable)
            detail = app.query_one(DetailPane)

            assert queue.select_key('decision:dec-second')
            await pilot.pause()
            assert long_question in detail.rendered_text

            # an awaiting-input session order_sessions ranks first, so the
            # highlighted session row's INDEX genuinely shifts on rebuild
            session_records.write_record(
                make_record(
                    session_slug='session-new',
                    status=session_records.Status.AWAITING_INPUT,
                    start_ts='2026-07-06T00:00:00+00:00',
                ),
                root=tmp_path,
            )
            # the decision itself is rewritten too -- text is a snapshot field
            # (_DECISION_SNAPSHOT_FIELDS), so this alone would trigger the
            # rebuild even without the session above
            revised_question = long_question.replace('re-file it?', 're-file it, REVISED?')
            assert dr.write_decision(
                dr.DecisionRecord(
                    id='dec-second', project='df', text=revised_question,
                    filed_at='2026-07-07T00:00:00+00:00',
                ),
                root=tmp_path,
            )
            app.refresh_registry()
            await pilot.pause()

            assert table.get_row_index('session-a') != 0
            # still the queue's decision, and re-rendered from the fresh scan
            assert revised_question in detail.rendered_text
            assert long_question not in detail.rendered_text

    @pytest.mark.timeout(10)
    async def test_the_queue_emptying_hands_the_pane_back_to_the_session_table(self, tmp_path):
        """A queue-owned pane must not outlive the queue.

        When a watcher resolves the last open decision while the operator is
        reading it, the queue empties and its highlighted key becomes None. If
        ownership stayed with the queue, every later session-table rebuild would
        decline the pane too, leaving the operator staring at a decision that no
        longer exists with no rebuild able to clear it -- a wedged pane, not a
        transient miss. So an empty queue releases ownership.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.panes.detail_pane import DetailPane
        from cockpit.panes.session_table import SessionTable

        # session-parked is NOT session-table row 0, so parking on it is a real
        # cursor move and the fallback render below is a specific session, not
        # just "whatever was already there".
        session_records.write_record(
            make_record(
                session_slug='session-first',
                start_ts='2026-07-07T00:00:00+00:00',
                question=session_records.Question(text='Unrelated?', asked_at='2026-07-07T00:00:00+00:00'),
            ),
            root=tmp_path,
        )
        session_records.write_record(
            make_record(
                session_slug='session-parked',
                start_ts='2026-07-07T00:01:00+00:00',
                task_id='7004',
                question=session_records.Question(text='PPP parked question?', asked_at='2026-07-07T00:01:00+00:00'),
            ),
            root=tmp_path,
        )
        # Two decisions, because a move onto the already-highlighted row 0 posts
        # no RowHighlighted at all -- with one queue row the queue could never
        # take ownership and this test would pass vacuously.
        for decision in (
            dr.DecisionRecord(
                id='dec-top', project='df', text='Short one?',
                filed_at='2026-07-07T00:00:00+00:00', manual_boost=5,
            ),
            dr.DecisionRecord(
                id='dec-read', project='df', text='ZZZ the decision being read?',
                filed_at='2026-07-07T00:00:00+00:00',
            ),
        ):
            assert dr.write_decision(decision, root=tmp_path)

        # a large poll_interval keeps on_mount's own timer from racing the
        # direct refresh_registry() below -- TestPollRefresh's convention
        app = CockpitApp(fleet_root=tmp_path, backend=FakeBackend(), poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            table = app.query_one(SessionTable)
            detail = app.query_one(DetailPane)

            table.move_cursor(row=table.get_row_index('session-parked'))
            await pilot.pause()
            assert app._selected_slug == 'session-parked'

            assert queue.select_key('decision:dec-read')
            await pilot.pause()
            assert 'ZZZ the decision being read?' in detail.rendered_text

            # a watcher answers every open decision -- the queue empties under
            # the operator's cursor
            for decision_id in ('dec-top', 'dec-read'):
                assert dr.update_decision_state(
                    decision_id, dr.DecisionState.ANSWERED, root=tmp_path
                )
            app.refresh_registry()
            await pilot.pause()

            assert queue.row_count == 0
            assert 'ZZZ the decision being read?' not in detail.rendered_text
            assert 'PPP parked question?' in detail.rendered_text

            # and the pane is not merely cleared once: the session table can
            # reach it again, so a later rebuild still refreshes it
            session_records.write_record(
                make_record(
                    session_slug='session-parked',
                    start_ts='2026-07-07T00:01:00+00:00',
                    task_id='7004',
                    question=session_records.Question(
                        text='PPP parked question, REVISED?', asked_at='2026-07-07T00:03:00+00:00'
                    ),
                ),
                root=tmp_path,
            )
            app.refresh_registry()
            await pilot.pause()

            assert 'PPP parked question, REVISED?' in detail.rendered_text


class TestSessionTableFocusCue:
    """Signal (a): a row says whether Enter can raise anything for it.

    Rendered end-to-end rather than only at the helper level, because the
    defect being fixed is that replace_rows never consulted record.display
    at all -- a passing focus_marker unit test would say nothing about what
    an operator actually sees in the table.
    """

    @pytest.mark.timeout(10)
    async def test_focusable_and_headless_rows_are_distinguishable(self, tmp_path):
        from cockpit.panes.session_table import SessionTable, focus_marker, state_glyph

        focusable = make_record(
            session_slug='focusable-1',
            status=session_records.Status.RUNNING,
            display=session_records.Display(kind='wm', wm_title='focusable title'),
        )
        headless = make_record(
            session_slug='headless-1', status=session_records.Status.RUNNING, display=None
        )
        for r in (focusable, headless):
            session_records.write_record(r, root=tmp_path)

        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            focusable_row = table.get_row('focusable-1')
            headless_row = table.get_row('headless-1')

            assert focus_marker(focusable) in focusable_row
            assert focus_marker(focusable) not in headless_row
            assert focus_marker(headless) in headless_row
            assert focus_marker(headless) not in focusable_row

            # The status vocabulary is untouched: both sessions are RUNNING,
            # and the focusability cue is a second, orthogonal column -- not
            # a re-spelling of the state glyph.
            assert state_glyph(session_records.Status.RUNNING) in focusable_row
            assert state_glyph(session_records.Status.RUNNING) in headless_row


class _FixedScanner:
    """Fake SessionScanner returning a prebuilt in-memory record list.

    Satisfies SessionScannerProtocol structurally (see test_app_scan_pipeline.py::_BlockingScanner):
    a cap test needs a few hundred records to EXIST, not to be on disk, so
    this skips writing that many record.json files.
    """

    def __init__(self, records: list) -> None:
        self._records = records

    def scan(self) -> list:
        return list(self._records)


class TestSessionTableCapNotice:
    """Signal (b): a capped table says so, instead of looking complete.

    Driven through the widget rather than only the formatter because the
    notice's failure mode is silent: border labels are painted as part of
    the border edge (measured, textual 8.2.8), so a subtitle set on a
    border-less widget reads back correctly and renders NOTHING.
    """

    @pytest.mark.timeout(10)
    async def test_replace_rows_reports_and_clears_the_cap_notice(self, tmp_path):
        from cockpit.panes.session_table import LiveSessions, SessionTable

        records = [
            make_record(session_slug=f's-{i}', status=session_records.Status.RUNNING) for i in range(3)
        ]
        now = datetime.fromisoformat('2026-07-07T00:00:00+00:00')

        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            # The notice has somewhere to paint. Without a border this
            # whole class would keep passing while the operator saw
            # nothing -- which is the exact defect the task exists to fix.
            assert table.styles.border_bottom[0] != ''

            table.replace_rows(LiveSessions(visible=records, total=12), now)
            await pilot.pause()
            assert table.border_subtitle == 'showing 3 of 12'

            # Unconditional assignment: a rebuild that is no longer
            # truncated must CLEAR the notice, not leave a stale one up.
            table.replace_rows(LiveSessions(visible=records, total=3), now)
            await pilot.pause()
            assert table.border_subtitle == ''

            table.replace_rows(LiveSessions(visible=records, total=12), now)
            await pilot.pause()
            assert table.border_subtitle == 'showing 3 of 12'

    @pytest.mark.timeout(10)
    async def test_app_hands_the_table_the_true_live_total(self, tmp_path):
        """The end-to-end half: _rebuild_session_table must report what
        filter_live_sessions hid, or the whole mechanism is inert.

        Uses a fake scanner rather than writing _DEFAULT_VISIBLE_CAP+5
        record.json files -- the records only need to exist in memory for
        the cap to bite, and this keeps the test in the same sub-second
        band as its neighbours.
        """
        from cockpit.panes.session_table import _DEFAULT_VISIBLE_CAP, SessionTable

        over_cap = [
            make_record(session_slug=f'over-{i}', status=session_records.Status.RUNNING)
            for i in range(_DEFAULT_VISIBLE_CAP + 5)
        ]

        app = CockpitApp(
            fleet_root=tmp_path, scanner=_FixedScanner(over_cap), poll_interval=60
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            assert table.row_count == _DEFAULT_VISIBLE_CAP
            assert table.border_subtitle == (
                f'showing {_DEFAULT_VISIBLE_CAP} of {_DEFAULT_VISIBLE_CAP + 5}'
            )

    @pytest.mark.timeout(10)
    async def test_small_fleet_claims_no_truncation(self, tmp_path):
        """A fleet that fits stays quiet -- a complete table must not
        announce itself as capped."""
        from cockpit.panes.session_table import SessionTable

        records = [
            make_record(session_slug='small-0', status=session_records.Status.RUNNING),
            make_record(session_slug='small-1', status=session_records.Status.IDLE),
        ]

        app = CockpitApp(
            fleet_root=tmp_path, scanner=_FixedScanner(records), poll_interval=60
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            assert table.row_count == 2
            assert table.border_subtitle == ''

    @pytest.mark.timeout(15)
    async def test_history_toggle_clears_and_restores_the_notice(self, tmp_path):
        """The transition that matters: an actually-capped view toggled to
        history and back.

        History renders self._records UNFILTERED, so it hides nothing by
        construction and must clear the notice it inherits -- and toggling
        back must bring the notice returned. Driven from a genuinely
        over-cap fleet (plus terminal records, which history shows and the
        live band does not) because with a small fleet an empty subtitle
        after 'h' is true no matter what the history branch reports.
        """
        from cockpit.panes.session_table import _DEFAULT_VISIBLE_CAP, SessionTable

        live = [
            make_record(session_slug=f'live-{i}', status=session_records.Status.RUNNING)
            for i in range(_DEFAULT_VISIBLE_CAP + 5)
        ]
        exited = [
            make_record(session_slug=f'exited-{i}', status=session_records.Status.EXITED)
            for i in range(3)
        ]
        notice = f'showing {_DEFAULT_VISIBLE_CAP} of {_DEFAULT_VISIBLE_CAP + 5}'

        app = CockpitApp(
            fleet_root=tmp_path, scanner=_FixedScanner(live + exited), poll_interval=60
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)

            assert table.row_count == _DEFAULT_VISIBLE_CAP
            assert table.border_subtitle == notice

            await pilot.press('h')
            await pilot.pause()

            assert table.row_count == len(live) + len(exited)
            assert table.border_subtitle == ''

            await pilot.press('h')
            await pilot.pause()

            assert table.row_count == _DEFAULT_VISIBLE_CAP
            assert table.border_subtitle == notice
