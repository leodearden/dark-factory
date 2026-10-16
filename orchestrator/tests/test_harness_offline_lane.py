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

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _lane_episode_fakes import Clock
from shared.task_submit import SubmittedTask

from orchestrator.config import LaneCommand
from orchestrator.harness import Harness
from orchestrator.offline_lane import OfflineLaneWorker
from orchestrator.offline_lane_episode_state import EpisodeStore

_COMMAND = 'lane-q'
_FAILING_ONE_TEST = "printf 'FAILED tests/test_q.py::test_a - AssertionError: boom\\n'; exit 1"
_WAIT_SECS = 60.0


async def _until(done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + _WAIT_SECS
    while not done():
        assert time.monotonic() < deadline, 'the harness-built lane worker did not get there in time'
        await asyncio.sleep(0.02)

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

    @pytest.mark.timeout(120)
    async def test_the_lane_files_a_reproduced_red_through_the_harness_scheduler(
        self, harness: Harness, tmp_path: Path
    ):
        """The harness-built worker files through the harness's OWN scheduler (task 2016, Bug #1).

        A real fixture command fails one test. Its first pass holds an episode and files
        nothing; with main idle, the engine's tick names it due 30 minutes on, the worker
        re-runs it, and the second reproduced pass files ONE lane fix task through
        ``scheduler.submit_task``, carrying the harness's project root. The worker's clock
        is its module default, replaced here because ``_start_offline_lane`` takes none.
        """
        worktree = tmp_path / 'wt'
        worktree.mkdir()
        harness.git_ops = MagicMock()
        harness.git_ops.get_main_sha = AsyncMock(return_value='HEAD1')
        harness.git_ops.reset_persistent_offline_deep_worktree = AsyncMock(return_value=worktree)
        harness.git_ops.is_ancestor_checked = AsyncMock(return_value=True)
        submit_task = AsyncMock(return_value=SubmittedTask('7001', 'deferred'))
        harness.scheduler.submit_task = submit_task
        harness.scheduler.dispatch_tool = AsyncMock(return_value={})
        git = harness.config.git
        git.offline_lane_enabled = True
        git.persistent_offline_deep_worktree = True
        git.offline_lane_legacy_numeric_enabled = False
        git.offline_lane_poll_interval_secs = 0.05
        git.offline_lane_commands = [LaneCommand(name=_COMMAND, command=_FAILING_ONE_TEST, cwd='.')]
        clock = Clock()

        with patch('orchestrator.offline_lane._utc_now', new=clock):
            await harness._start_offline_lane()
        try:
            await _until(lambda: bool(EpisodeStore(tmp_path, clock=clock).load(_COMMAND).episodes))
            submit_task.assert_not_awaited()
            clock.advance(minutes=30)
            await _until(lambda: submit_task.await_count > 0)
        finally:
            await harness._stop_offline_lane()

        submit_task.assert_awaited_once()
        assert submit_task.await_args is not None
        arguments = submit_task.await_args.args[0]
        assert arguments['metadata']['spawn_context'] == 'offline_lane_red'
        assert arguments['project_root'] == str(tmp_path)

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
