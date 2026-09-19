"""``get_pending_escalations``' queue scan runs OFF the event loop (task 4391).

WHY it must — this server shares the ORCHESTRATOR's loop, what the scan costs
and why that cost only grows, and which scans are deliberately left inline —
is stated ONCE, beside the production code it justifies: the rationale block
above ``read_pending()`` in
``escalation/src/escalation/server.py::create_server.get_pending_escalations``.
Restating the figures here would give hard numbers a second home in a file
nobody edits alongside that one.

What these tests pin, neither of them by TIMING the loop (racing a heartbeat
against the scan would make the verdict depend on wall-clock scheduling on a
loaded host):

- :class:`TestPendingScanThread` — thread identity, which is exact.
  ``asyncio.to_thread`` always runs its callable on an executor worker, so the
  recorded id either equals the loop thread's or it does not.  Both branches,
  since both reach a scan of the queue root.
- :class:`TestLoopStaysLiveDuringScan` — a happens-before ORDER, a fact rather
  than a duration: the loop keeps running, and can still file, while the tool
  is inside the scan.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from _pending_tool_fixtures import IN_PROGRESS, _file, _get_pending, _harness

from escalation.queue import EscalationQueue
from escalation.server import create_server


class _ThreadProbeQueue(EscalationQueue):
    """A REAL queue that records which thread each pending scan ran on.

    A subclass delegating through ``super()``, deliberately NOT a mock (the
    shape ``_CountingQueue`` in test_pins_recovery_annotation.py established):
    each probed call still does the real glob + JSON parse, so these tests
    cannot pass because the scan was stubbed out from under them.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.scan_threads: list[int] = []

    def get_by_task(self, task_id, status=None, level=None, agent_role=None):
        self.scan_threads.append(threading.get_ident())
        return super().get_by_task(task_id, status, level, agent_role)

    def get_pending(self):
        self.scan_threads.append(threading.get_ident())
        return super().get_pending()


@pytest.mark.asyncio
class TestPendingScanThread:
    """BOTH branches scan the queue root, so both must be hopped off-loop.

    Parametrized rather than split: the two arms differ only in which queue
    read they reach, and pinning them together is what stops a fix landing on
    one and reading as if the arms had different concurrency semantics.
    """

    @pytest.mark.parametrize('kwargs', [{}, {'task_id': '960'}])
    async def test_scan_does_not_run_on_the_loop_thread(self, tmp_path, kwargs):
        queue = _ThreadProbeQueue(tmp_path / 'esc')
        _file(queue, '960', level=1)
        server = create_server(queue, harness=_harness({'960': IN_PROGRESS}))

        recs = await _get_pending(server, **kwargs)

        # The scan still returns the right answer — the hop is not a stub.
        assert [r['task_id'] for r in recs] == ['960']
        assert recs[0]['pins_recovery'] == ['960']

        # This test body runs on the event-loop thread, so its id is the one
        # the scan must NOT have used.
        loop_thread = threading.get_ident()
        assert queue.scan_threads, 'the probe recorded no scan at all'
        assert loop_thread not in queue.scan_threads, (
            f'pending scan ran ON the event-loop thread {loop_thread}; '
            f'observed scan threads: {queue.scan_threads}'
        )


class _RendezvousQueue(EscalationQueue):
    """A REAL queue that parks on ENTRY to ``get_pending``, so the loop can act.

    The park BRACKETS the real scan rather than interposing inside it — record
    ``'scan-start'``, wait, record ``'scan-end'``, and only then delegate to
    ``super()``.  So what the loop does while parked happens-before the glob,
    not during it.  That is exactly enough to pin the ORDER this test exists
    for, and deliberately NOT a claim about interleaving WITHIN the directory
    read; a record moving between the glob and the per-file read is
    test_queue.py::TestScanSurvivesRecordArchivedMidScan's subject, via
    ``_scan_race_helpers.relocating_read_text``.

    The release wait is bounded at 2.0 s: under an INLINE implementation
    nothing on the loop can ever set the release event, so the wait must time
    out rather than hang — that turns the inline case
    into a clean order-assertion failure in ~2 s instead of a test that sits
    until this suite's 300 s per-test cap (see escalation/pyproject.toml).
    """

    RELEASE_TIMEOUT = 2.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[str] = []
        self.scan_entered = threading.Event()
        self.release = threading.Event()

    def get_pending(self):
        self.events.append('scan-start')
        self.scan_entered.set()
        self.release.wait(timeout=self.RELEASE_TIMEOUT)
        self.events.append('scan-end')
        return super().get_pending()


async def _await_event(event: threading.Event) -> None:
    """Wait for a worker thread's ``threading.Event`` without blocking the loop."""
    while not event.is_set():
        await asyncio.sleep(0.005)


@pytest.mark.asyncio
class TestLoopStaysLiveDuringScan:
    async def test_loop_runs_and_can_file_while_the_scan_is_in_flight(self, tmp_path):
        """The same property stated behaviourally, and what a submit gets to do.

        Asserting an ORDER makes this a happens-before fact rather than a
        duration, so it does not depend on how loaded the host is.  The record
        filed from the loop thread lands between the tool ENTERING the scan and
        the queue read beginning (see ``_RendezvousQueue`` for why that is the
        window, and where the inside-the-read one is covered) — a submit
        concurrent with the tool, which is the interleaving this task was asked
        to verify nothing depends on being absent.  The check that it was
        harmless is that every returned row is still whole.
        """
        queue = _RendezvousQueue(tmp_path / 'esc')
        _file(queue, '970', level=1)
        server = create_server(
            queue, harness=_harness({'970': IN_PROGRESS, '971': IN_PROGRESS}),
        )

        pending = asyncio.create_task(_get_pending(server))
        try:
            # Polled rather than awaited directly — a threading.Event is not
            # awaitable — and bounded, so a scan that never enters fails here
            # instead of spinning into the 300 s per-test cap.  Under an
            # INLINE implementation this line is not even reached until the
            # scan has finished, which is exactly what the order assertion
            # below catches.
            await asyncio.wait_for(_await_event(queue.scan_entered), 3.0)
            queue.events.append('loop-alive')
            # The concurrent mutation, from the loop thread, while it parks.
            _file(queue, '971', level=1)
        finally:
            queue.release.set()
        recs = await pending

        assert queue.events == ['scan-start', 'loop-alive', 'scan-end'], (
            f'the loop did not run while the scan was in flight: {queue.events}'
        )
        # Every row is well-formed: the submit that landed while the tool was
        # parked produced no partial or malformed record, and no false `[]` —
        # pins_recovery is present and correct on each, which is the
        # annotation's UNKNOWN-vs-empty contract.
        assert recs, 'the scan returned nothing'
        for rec in recs:
            assert rec['pins_recovery'] == [rec['task_id']], rec
