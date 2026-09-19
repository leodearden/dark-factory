"""``get_pending_escalations``' queue scan runs OFF the event loop (task 4391).

This is the one ``async def`` tool in ``escalation/server.py`` that scans the
queue root, and the loop it would block is not its own: the escalation MCP
server has no process of its own — ``_start_escalation_server``
(orchestrator/src/orchestrator/harness.py) runs it under
``asyncio.create_task`` on the ORCHESTRATOR's event loop.  So an inline scan
stalls the scheduler and the merge worker, not a dedicated server.

How long: 13.12 ms median, 30.92 ms p95, measured over 40 warm reps against
the live queue root — 9,461 dirents serving 41 pending records.  The cost is
DIRENT-dominated (a bare ``scandir`` of that root is already 6.45 ms) and the
dirent population tracks LIFETIME escalation count, not the pending set, via
the record-lock sidecars deliberately retained for archived records.  It only
grows.  Every dashboard poll and every watcher drain pays it.

The property is pinned by THREAD IDENTITY, not by timing the loop — a
heartbeat-raced-against-the-scan test would make the verdict depend on
wall-clock scheduling on a loaded host, whereas ``asyncio.to_thread`` always
runs its callable on an executor worker, so the recorded id either equals the
loop's or it does not.
"""

from __future__ import annotations

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
    async def test_unfiltered_scan_does_not_run_on_the_loop_thread(self, tmp_path):
        """The no-``task_id`` branch's ``get_pending()`` must be hopped off-loop."""
        queue = _ThreadProbeQueue(tmp_path / 'esc')
        _file(queue, '960', level=1)
        server = create_server(queue, harness=_harness({'960': IN_PROGRESS}))

        recs = await _get_pending(server)

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
