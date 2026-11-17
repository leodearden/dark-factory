"""CockpitApp pilot tests for the threaded poll-scan pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from datetime import datetime

import pytest
from _cockpit_session_records import make_record
from _cockpit_tree_snapshot import snapshot_tree
from orchestrator import decision_registry as dr
from orchestrator import session_registry as sr

from cockpit.app import CockpitApp


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


class _HoldingDecisionsReader:
    """A decisions_reader that reads the real registry, then holds any OFF-main-thread
    caller until release().

    That caller is a poll worker, held after its whole registry read, because
    _scan_registry reads sessions before decisions. Main-thread reads (mount's
    refresh_registry, a keypress's re-read) pass through, because holding one would
    deadlock the event loop the pilot runs on. After release() every read passes
    through.
    """

    def __init__(self) -> None:
        self.held = threading.Event()
        self._release = threading.Event()
        self.held_result: list[dr.DecisionRecord] = []

    def __call__(self, root):
        from cockpit.registry_reader import scan_decisions

        decisions = scan_decisions(root)
        off_main_thread = threading.current_thread() is not threading.main_thread()
        if off_main_thread and not self._release.is_set():
            self.held_result = decisions
            self.held.set()
            self._release.wait()
        return decisions

    def release(self) -> None:
        self._release.set()


@contextlib.asynccontextmanager
async def _poll_held_after_its_read(
    app, reader: _HoldingDecisionsReader, *, reach_timeout_secs: float = 5.0
):
    """Run one real poll tick whose worker *reader* holds after its registry read.

    Leaving the block releases the worker and waits for it, so the held poll has
    landed through the real call_from_thread hand-off: call_from_thread blocks the
    worker until _apply_scan has run on the loop. The release sits in a finally, so
    a failing assertion inside the block never leaves the worker parked.
    """
    app._poll_registry()
    try:
        reached = await asyncio.to_thread(reader.held.wait, reach_timeout_secs)
        assert reached, (
            f'the poll worker never reached its decisions read within {reach_timeout_secs}s'
        )
        yield
    finally:
        reader.release()
        await app.workers.wait_for_complete()


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
        from cockpit.panes.spawn_tree import SpawnTreeScreen

        scanner = _BlockingScanner()
        # A large poll_interval keeps on_mount's own set_interval timer from
        # firing (and launching a second, uncontrolled worker) during this
        # test -- the only scan driven here is the explicit
        # _poll_registry() call below.
        app = CockpitApp(fleet_root=tmp_path, poll_interval=60, scanner=scanner)
        async with app.run_test() as pilot:
            await pilot.pause()  # on_mount's synchronous scan: call #1, non-blocking

            app._poll_registry()
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
    """Extends test_app.py::TestRefreshWriteDiscipline to the threaded poll path
    (_scan_registry_worker, see TestNonBlockingPoll): the off-thread scan's
    result must actually reach the UI via the call_from_thread(_apply_scan)
    hand-off, and that hand-off must remain a pure reader -- writing
    nothing under sessions/ or decisions/, exactly like the synchronous
    refresh_registry path. Uses the real (default) SessionScanner, not an
    injected fake, to prove the actual production wiring end-to-end.
    """

    @pytest.mark.timeout(10)
    async def test_worker_result_reaches_table_and_writes_nothing(self, tmp_path):
        from cockpit.panes.session_table import SessionTable

        first = make_record(session_slug='thread-1', status=sr.Status.RUNNING)
        sr.write_record(first, root=tmp_path)

        # A large poll_interval keeps on_mount's own set_interval timer from
        # firing a second, uncontrolled worker mid-test -- the only scan
        # driven here is the explicit _poll_registry() call below.
        app = CockpitApp(fleet_root=tmp_path, poll_interval=60)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert table.row_count == 1

            second = make_record(session_slug='thread-2', status=sr.Status.AWAITING_INPUT)
            sr.write_record(second, root=tmp_path)

            before = snapshot_tree(tmp_path)
            # sanity: prove the snapshot actually captured the seeded files,
            # so the "unchanged" assertion below isn't vacuously true.
            assert any(path.startswith('sessions/') for path in before)

            app._poll_registry()
            await app.workers.wait_for_complete()
            await pilot.pause()

            # proves the off-thread -> call_from_thread(_apply_scan)
            # hand-off actually reached the live SessionTable.
            assert table.row_count == 2

        after = snapshot_tree(tmp_path)
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


class TestALatePollCannotRegressAFresherScan:
    """esc-2517-1 / task 2606: a poll worker's call_from_thread hand-off that lands
    after a fresher scan must not regress the view, replayed through a real held
    worker. The last phase also proves the worker carries its own scan's freshness
    across the thread boundary.
    """

    @pytest.mark.timeout(10)
    async def test_a_poll_landing_after_a_fresher_scan_is_dropped_and_the_next_poll_applies(
        self, tmp_path
    ):
        from cockpit.panes.session_table import SessionTable

        sr.write_record(make_record(session_slug='r-1'), root=tmp_path)
        reader = _HoldingDecisionsReader()
        app = CockpitApp(fleet_root=tmp_path, poll_interval=60, decisions_reader=reader)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app.query_one(SessionTable)
            assert table.row_count == 1

            async with _poll_held_after_its_read(app, reader):
                sr.write_record(make_record(session_slug='r-2'), root=tmp_path)
                app.refresh_registry()
                await pilot.pause()
                assert table.row_count == 2

            await pilot.pause()
            assert table.row_count == 2, 'the late poll regressed the view'

            sr.write_record(make_record(session_slug='r-3'), root=tmp_path)
            app._poll_registry()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert table.row_count == 3, 'the stale guard froze the view'


class TestInFlightScanCannotUndoAKeypressWrite:
    """Task 5839: test_app.py::TestBoostReordersAndPersists's merge-gate red, replayed
    deterministically.

    _apply_boost and action_drop write a decision and then re-read
    decisions, but a poll scan issued BEFORE that write, whose registry read
    predates it, can land AFTER it. Each replay holds a REAL poll worker
    after its registry read (_HoldingDecisionsReader), interleaves the
    keypress, then lets the worker land through the real call_from_thread
    hand-off.
    """

    @pytest.mark.timeout(10)
    async def test_a_boost_survives_a_scan_issued_before_it(self, tmp_path):
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.priority import Priorities

        older = dr.DecisionRecord(
            id='dec-a',
            project='df',
            text='A?',
            filed_at='2026-06-01T00:00:00+00:00',
            manual_boost=0,
        )
        newer = dr.DecisionRecord(
            id='dec-b',
            project='df',
            text='B?',
            filed_at='2026-07-07T00:00:00+00:00',
            manual_boost=0,
        )
        for d in (older, newer):
            assert dr.write_decision(d, root=tmp_path)

        fixed_now = datetime.fromisoformat('2026-07-07T00:00:00+00:00')
        reader = _HoldingDecisionsReader()
        app = CockpitApp(
            fleet_root=tmp_path,
            backend=FakeBackend(),
            poll_interval=60,
            now_fn=lambda: fixed_now,
            priorities=Priorities.default(),
            decisions_reader=reader,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            queue.move_cursor(row=queue.get_row_index('decision:dec-b'))
            await pilot.pause()

            async with _poll_held_after_its_read(app, reader):
                assert {d.id: d.manual_boost for d in reader.held_result}['dec-b'] == 0
                await pilot.press('b')
                await pilot.pause()
                assert queue.get_row_index('decision:dec-b') < queue.get_row_index('decision:dec-a')

            await pilot.pause()
            assert queue.get_row_index('decision:dec-b') < queue.get_row_index('decision:dec-a'), (
                "the late scan reverted the keypress's persisted boost in the view"
            )

            for _ in range(2):
                await pilot.press('b')
                await pilot.pause()
            persisted = {d.id: d for d in dr.list_decisions(root=tmp_path)}
            assert persisted['dec-b'].manual_boost == 3, (
                'a press built on the reverted boost and was lost'
            )

    @pytest.mark.timeout(10)
    async def test_a_drop_survives_a_scan_issued_before_it(self, tmp_path):
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

        reader = _HoldingDecisionsReader()
        app = CockpitApp(
            fleet_root=tmp_path, backend=FakeBackend(), poll_interval=60, decisions_reader=reader
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            queue.move_cursor(row=queue.get_row_index('decision:dec-1'))
            await pilot.pause()

            async with _poll_held_after_its_read(app, reader):
                assert {d.id: d.state for d in reader.held_result} == {
                    'dec-1': dr.DecisionState.OPEN,
                    'dec-2': dr.DecisionState.OPEN,
                }
                await pilot.press('x')
                await pilot.pause()
                assert queue.row_count == 1

            await pilot.pause()
            assert queue.row_count == 1, 'the late scan resurrected the dropped decision'
            with pytest.raises(RowDoesNotExist):
                queue.get_row_index('decision:dec-1')

    @pytest.mark.timeout(10)
    async def test_the_scan_issued_before_a_boost_still_lands_its_sessions(self, tmp_path):
        """The keypress re-reads decisions only, so it may void only the late
        scan's decisions: the scan's sessions are still the freshest read.
        Voiding the whole scan froze the session table for as long as
        keypresses kept landing mid-scan -- on a large fleet, whose scan
        outlasts the gap between presses, a whole triage burst.
        """
        from cockpit.backends import FakeBackend
        from cockpit.panes.decision_queue import DecisionQueue
        from cockpit.panes.session_table import SessionTable
        from cockpit.priority import Priorities

        older = dr.DecisionRecord(
            id='dec-a',
            project='df',
            text='A?',
            filed_at='2026-06-01T00:00:00+00:00',
            manual_boost=0,
        )
        newer = dr.DecisionRecord(
            id='dec-b',
            project='df',
            text='B?',
            filed_at='2026-07-07T00:00:00+00:00',
            manual_boost=0,
        )
        for d in (older, newer):
            assert dr.write_decision(d, root=tmp_path)

        fixed_now = datetime.fromisoformat('2026-07-07T00:00:00+00:00')
        reader = _HoldingDecisionsReader()
        app = CockpitApp(
            fleet_root=tmp_path,
            backend=FakeBackend(),
            poll_interval=60,
            now_fn=lambda: fixed_now,
            priorities=Priorities.default(),
            decisions_reader=reader,
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            queue = app.query_one(DecisionQueue)
            table = app.query_one(SessionTable)
            queue.move_cursor(row=queue.get_row_index('decision:dec-b'))
            await pilot.pause()

            sr.write_record(make_record(session_slug='running-1'), root=tmp_path)
            async with _poll_held_after_its_read(app, reader):
                await pilot.press('b')
                await pilot.pause()
                assert table.row_count == 0

            await pilot.pause()
            assert table.row_count == 1, "the keypress's re-read voided the scan's sessions"
            assert queue.get_row_index('decision:dec-b') < queue.get_row_index('decision:dec-a'), (
                "the late scan's decisions reverted the keypress's persisted boost"
            )
