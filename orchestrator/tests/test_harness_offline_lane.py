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
from unittest.mock import MagicMock, patch

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
