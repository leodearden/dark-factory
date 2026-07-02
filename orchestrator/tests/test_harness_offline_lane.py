"""Tests for Harness-side offline deep-test lane wiring (task 1953, β2).

Covers ``Harness._start_offline_lane`` / ``Harness._stop_offline_lane``: the
enable-gated construction of an ``OfflineLaneWorker``, its registration into
the ``_offline_lane_notifiee`` slot (task 1951, β1), and the background
``_offline_lane_task`` lifecycle.

See also ``test_offline_lane.py`` for the worker's own control-flow
contract, and ``test_harness_offline_lane_trigger.py`` (β1) for the
``_note_merge_all`` fan-out this wiring plugs into.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness
from orchestrator.offline_lane import OfflineLaneWorker

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """Minimal harness with mocked heavy deps, modeled on test_harness_offline_lane_trigger.py."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.event_store = MagicMock()
    return h


# ---------------------------------------------------------------------------
# _start_offline_lane / _stop_offline_lane (step-17/18)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOfflineLaneWiring:
    """Enable-gated launch/stop of the singleton OfflineLaneWorker.

    Step 17 (RED): _start_offline_lane / _stop_offline_lane do not exist yet
    on Harness — every test below must fail with AttributeError before impl.
    """

    async def test_start_offline_lane_builds_worker_and_registers_notifiee(
        self, harness: Harness
    ):
        """Both knobs True: build a worker, register the notifiee, launch a task."""
        harness.config.git.offline_lane_enabled = True
        harness.config.git.persistent_offline_deep_worktree = True

        await harness._start_offline_lane()

        try:
            assert isinstance(harness._offline_lane_worker, OfflineLaneWorker)
            assert harness._offline_lane_notifiee == harness._offline_lane_worker.on_post_merge
            assert harness._offline_lane_task is not None
            assert not harness._offline_lane_task.done()
        finally:
            await harness._stop_offline_lane()

    async def test_start_offline_lane_wires_task_client_and_escalation_queue(
        self, harness: Harness
    ):
        """_start_offline_lane wires a concrete task_client + the harness's
        escalation_queue into the worker (task 2016, closing Bug #1).

        Without this wiring, a confirmed red run's _file_new_fix_task /
        _file_info_escalation / _file_blocker_escalation all degrade to
        log-only no-ops (offline_lane.py task_client/escalation_queue None
        checks) — a confirmed red run files NOTHING.

        Step 5 (RED): _start_offline_lane currently builds the worker with
        neither task_client= nor escalation_queue=, so both default to None
        — must fail before impl (step-6).
        """
        from orchestrator.harness import _OfflineLaneTaskClient

        harness.config.git.offline_lane_enabled = True
        harness.config.git.persistent_offline_deep_worktree = True
        harness._escalation_queue = MagicMock()

        await harness._start_offline_lane()

        try:
            worker = harness._offline_lane_worker
            assert isinstance(worker, OfflineLaneWorker)
            assert worker.task_client is not None
            assert isinstance(worker.task_client, _OfflineLaneTaskClient)
            assert worker.escalation_queue is harness._escalation_queue
        finally:
            await harness._stop_offline_lane()

    async def test_start_offline_lane_noop_when_offline_lane_disabled(
        self, harness: Harness
    ):
        """offline_lane_enabled=False (even with the worktree knob on) is a no-op."""
        harness.config.git.offline_lane_enabled = False
        harness.config.git.persistent_offline_deep_worktree = True

        await harness._start_offline_lane()

        assert harness._offline_lane_notifiee is None
        assert harness._offline_lane_worker is None
        assert harness._offline_lane_task is None

    async def test_start_offline_lane_noop_when_worktree_knob_disabled(
        self, harness: Harness
    ):
        """persistent_offline_deep_worktree=False (even with the enable knob on) is a no-op.

        The worker cannot run without its dedicated _offline-deep worktree (δ).
        """
        harness.config.git.offline_lane_enabled = True
        harness.config.git.persistent_offline_deep_worktree = False

        await harness._start_offline_lane()

        assert harness._offline_lane_notifiee is None
        assert harness._offline_lane_worker is None
        assert harness._offline_lane_task is None

    async def test_start_offline_lane_skips_when_lock_refused(
        self, harness: Harness
    ):
        """A refused lock acquire (e.g. a second instance) registers nothing."""
        harness.config.git.offline_lane_enabled = True
        harness.config.git.persistent_offline_deep_worktree = True

        with patch.object(OfflineLaneWorker, 'acquire_lock', return_value=False):
            await harness._start_offline_lane()

        assert harness._offline_lane_notifiee is None
        assert harness._offline_lane_worker is None
        assert harness._offline_lane_task is None

    async def test_stop_offline_lane_cancels_task_and_releases_lock(
        self, harness: Harness
    ):
        """_stop_offline_lane cancels the background task, releases the lock, clears slots."""
        harness.config.git.offline_lane_enabled = True
        harness.config.git.persistent_offline_deep_worktree = True
        await harness._start_offline_lane()
        task = harness._offline_lane_task
        worker = harness._offline_lane_worker
        assert task is not None, 'sanity: the task must exist after start'
        assert worker is not None, 'sanity: the worker must exist after start'
        assert worker._lock_file is not None, 'sanity: the lock must be held after start'

        await harness._stop_offline_lane()

        assert task.cancelled() or task.done()
        assert worker._lock_file is None, 'the lock must be released on stop'
        assert harness._offline_lane_task is None
        assert harness._offline_lane_worker is None
        assert harness._offline_lane_notifiee is None, (
            'a stopped lane must not leave a dangling notifiee registered'
        )

    async def test_stop_offline_lane_noop_when_never_started(
        self, harness: Harness
    ):
        """_stop_offline_lane is a clean no-op when the lane was never started."""
        await harness._stop_offline_lane()  # must not raise

        assert harness._offline_lane_task is None
        assert harness._offline_lane_worker is None
        assert harness._offline_lane_notifiee is None


# ---------------------------------------------------------------------------
# _OfflineLaneTaskClient — concrete MCP adapter wrapping the scheduler
# (task 2016, step-3/4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_offline_lane_task_client_dispatches_via_scheduler():
    """_OfflineLaneTaskClient's three methods delegate to the scheduler.

    Covers the concrete implementation of the ``offline_lane.OfflineLaneTaskClient``
    Protocol that the harness wires into ``OfflineLaneWorker`` — submit via
    ``dispatch_tool('submit_task', ...)``, suspect-range append via
    ``update_task(..., metadata_mode='additive')`` (atomic list-union, no
    read-modify-write), and status via ``get_statuses([id])``.

    Step 3 (RED): ``_OfflineLaneTaskClient`` does not exist on
    ``orchestrator.harness`` yet — the import must fail before impl (step-4).
    """
    from orchestrator.harness import _OfflineLaneTaskClient

    scheduler = MagicMock()
    scheduler.dispatch_tool = AsyncMock(return_value={'task_id': 'fix-7'})
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_statuses = AsyncMock(return_value=({'fix-7': 'in-progress'}, None))

    client = _OfflineLaneTaskClient(scheduler)

    # (a) submit_fix_task dispatches 'submit_task' with the arguments dict and
    # extracts the new task's id from the MCP response envelope.
    arguments = {'title': 't'}
    task_id = await client.submit_fix_task(arguments)
    assert task_id == 'fix-7'
    scheduler.dispatch_tool.assert_awaited_once()
    await_args = scheduler.dispatch_tool.await_args
    assert await_args.args[0] == 'submit_task'
    assert await_args.args[1] == arguments

    # (b) append_suspect_range additive-updates metadata.suspect_ranges —
    # a recursive list-union append, not a read-modify-write.
    await client.append_suspect_range('fix-7', 'A..B')
    scheduler.update_task.assert_awaited_once_with(
        'fix-7', {'suspect_ranges': ['A..B']}, metadata_mode='additive',
    )

    # (c) get_status resolves via get_statuses([task_id]).
    status = await client.get_status('fix-7')
    assert status == 'in-progress'
    scheduler.get_statuses.assert_awaited_once_with(['fix-7'])
