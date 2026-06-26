"""Tests for θ=1893: Harness no-landings circuit-breaker wiring.

Covers:
  step-09 RED  — Harness FIRE path: _run_no_landings_breaker_pass() halts
                 scheduler + files a deduped INFO escalation on trip
  step-11 RED  — Harness RESUME path: recovery pass un-pauses scheduler and
                 resolves the open INFO escalation
  step-13 RED  — Lifecycle: _start/_stop_no_landings_breaker kill-switch +
                 dedup + cancel
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.merge_queue import NoLandingsCircuitBreaker
from orchestrator.run_store import RunStore

# Role tag used by the breaker escalation (must match implementation)
_BREAKER_ROLE = 'orchestrator-no-landings-breaker'
# Sentinel task_id used for breaker INFO escalation dedup
_BREAKER_SENTINEL = '__no_landings_breaker__'


# ---------------------------------------------------------------------------
# Test factories
# ---------------------------------------------------------------------------


def _make_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Harness with a real Scheduler + EscalationQueue and a spy RunStore.

    Mirrors test_harness_resume_scheduler._make_harness.  Returns
    (harness, mock_run_store).
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-breaker-0001'
    harness.event_store = EventStore(tmp_path / 'events.db', 'run-breaker-0001')
    harness._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    return harness, mock_run_store


def _stub_merge_worker(harness: Harness, landings_total: int) -> MagicMock:
    """Attach a MagicMock merge worker that reports a fixed landings_total."""
    worker = MagicMock()
    worker.snapshot.return_value = {'metrics': {'landings_total': landings_total}}
    harness._merge_worker = worker
    return worker


def _small_breaker(window: int = 3, floor: int = 1000) -> NoLandingsCircuitBreaker:
    """Return a breaker with a small window for test speed."""
    return NoLandingsCircuitBreaker(window_samples=window, disk_free_floor_bytes=floor)


def _make_falling_disk_iter(
    start: int = 200_000, drop: int = 10_000,
):
    """Generate strictly-falling free-bytes values."""
    current = start
    while True:
        yield current
        current -= drop


# ---------------------------------------------------------------------------
# step-09: FIRE path
# ---------------------------------------------------------------------------


class TestHarnessNoLandingsFire:
    """_run_no_landings_breaker_pass() halts scheduler + files INFO escalation.

    RED until step-10 GREEN adds _run_no_landings_breaker_pass and
    _file_no_landings_info_escalation to Harness.
    """

    @pytest.mark.asyncio
    async def test_fire_halts_scheduler(self, tmp_path: Path) -> None:
        """After window passes with flat landings + falling disk, scheduler is paused."""
        harness, _rs = _make_harness(tmp_path)
        # Override the breaker with a small-window instance
        harness._no_landings_breaker = _small_breaker(window=3, floor=1000)
        _stub_merge_worker(harness, landings_total=5)  # landings stay flat

        disk_iter = _make_falling_disk_iter(start=200_000, drop=10_000)

        with patch('shutil.disk_usage') as mock_du:
            mock_du.side_effect = lambda _p: MagicMock(free=next(disk_iter))
            for _ in range(3):
                await harness._run_no_landings_breaker_pass()

        assert harness.scheduler.is_paused, (
            'scheduler should be paused after trip window'
        )

    @pytest.mark.asyncio
    async def test_fire_files_info_escalation(self, tmp_path: Path) -> None:
        """After trip, exactly one INFO escalation with correct attributes is filed."""
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=1000)
        _stub_merge_worker(harness, landings_total=5)

        disk_iter = _make_falling_disk_iter(start=200_000, drop=10_000)

        with patch('shutil.disk_usage') as mock_du:
            mock_du.side_effect = lambda _p: MagicMock(free=next(disk_iter))
            for _ in range(3):
                await harness._run_no_landings_breaker_pass()

        # Exactly one pending INFO escalation with the right role
        assert harness._escalation_queue is not None
        pending = [
            e
            for e in harness._escalation_queue.get_by_task(
                _BREAKER_SENTINEL, status='pending'
            )
            if e.agent_role == _BREAKER_ROLE
        ]
        assert len(pending) == 1, f'expected 1 pending breaker INFO, got {len(pending)}'
        esc = pending[0]
        assert esc.severity == 'info'
        assert esc.level == 0
        assert esc.category == 'risk_identified'
        assert esc.task_id == _BREAKER_SENTINEL

    @pytest.mark.asyncio
    async def test_fire_escalation_mentions_window_and_disk(self, tmp_path: Path) -> None:
        """Escalation summary and detail describe window, landings, and disk trend."""
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=1000)
        _stub_merge_worker(harness, landings_total=5)

        disk_iter = _make_falling_disk_iter(start=200_000, drop=10_000)

        with patch('shutil.disk_usage') as mock_du:
            mock_du.side_effect = lambda _p: MagicMock(free=next(disk_iter))
            for _ in range(3):
                await harness._run_no_landings_breaker_pass()

        assert harness._escalation_queue is not None
        pending = [
            e
            for e in harness._escalation_queue.get_by_task(
                _BREAKER_SENTINEL, status='pending'
            )
            if e.agent_role == _BREAKER_ROLE
        ]
        assert pending, 'expected escalation to be filed'
        esc = pending[0]
        # Summary must not be empty and should reference the window context
        assert len(esc.summary) > 0
        assert len(esc.detail) > 0

    @pytest.mark.asyncio
    async def test_fire_dedup_one_escalation_across_multiple_trips(
        self, tmp_path: Path
    ) -> None:
        """Multiple passes after trip do not stack duplicate escalations.

        Use a high floor (500_000) so the falling disk values (200k..150k)
        never trigger disk-recovery resume — only a clean landing would resume,
        which never happens here (landings_total stays flat at 5).
        """
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=500_000)
        _stub_merge_worker(harness, landings_total=5)

        disk_iter = _make_falling_disk_iter(start=200_000, drop=10_000)

        with patch('shutil.disk_usage') as mock_du:
            mock_du.side_effect = lambda _p: MagicMock(free=next(disk_iter))
            # Run more than window passes — should still file exactly one escalation
            for _ in range(6):
                await harness._run_no_landings_breaker_pass()

        assert harness._escalation_queue is not None
        all_pending = [
            e
            for e in harness._escalation_queue.get_by_task(
                _BREAKER_SENTINEL, status='pending'
            )
            if e.agent_role == _BREAKER_ROLE
        ]
        assert len(all_pending) == 1, (
            f'expected exactly 1 breaker INFO escalation (dedup), got {len(all_pending)}'
        )

    @pytest.mark.asyncio
    async def test_bare_harness_no_worker_is_graceful_noop(self, tmp_path: Path) -> None:
        """Harness without _merge_worker makes the pass a graceful no-op."""
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=1000)
        # No _merge_worker attached (harness._merge_worker is None by default)
        assert harness._merge_worker is None
        # Must not raise
        await harness._run_no_landings_breaker_pass()
        # Scheduler stays unpaused
        assert not harness.scheduler.is_paused

    @pytest.mark.asyncio
    async def test_disk_stat_oserror_is_fail_open(self, tmp_path: Path) -> None:
        """OSError from shutil.disk_usage must never halt dispatch (fail-open)."""
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=1000)
        _stub_merge_worker(harness, landings_total=5)

        with patch('shutil.disk_usage', side_effect=OSError('disk stat failed')):
            # Run multiple passes — OSError must be swallowed, not raise
            for _ in range(3):
                await harness._run_no_landings_breaker_pass()

        # Scheduler must NOT be paused (fail-open: stat failure never halts)
        assert not harness.scheduler.is_paused


# ---------------------------------------------------------------------------
# step-11: RESUME path
# ---------------------------------------------------------------------------


class TestHarnessNoLandingsResume:
    """_run_no_landings_breaker_pass() unpauses + resolves INFO on recovery.

    RED until step-12 GREEN adds resume wiring.
    """

    async def _drive_to_trip(
        self, harness: Harness, window: int = 3
    ) -> None:
        """Drive the breaker to tripped + scheduler halted state."""
        _stub_merge_worker(harness, landings_total=5)
        disk_iter = _make_falling_disk_iter(start=200_000, drop=10_000)
        with patch('shutil.disk_usage') as mock_du:
            mock_du.side_effect = lambda _p: MagicMock(free=next(disk_iter))
            for _ in range(window):
                await harness._run_no_landings_breaker_pass()

    @pytest.mark.asyncio
    async def test_clean_landing_resumes_scheduler(self, tmp_path: Path) -> None:
        """A pass where landings_total increased unpauses the scheduler."""
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=1000)
        await self._drive_to_trip(harness)
        assert harness.scheduler.is_paused, 'precondition: scheduler must be paused'

        # Recovery: landings_total increased (clean landing happened)
        _stub_merge_worker(harness, landings_total=6)
        with patch('shutil.disk_usage') as mock_du:
            mock_du.return_value = MagicMock(free=150_000)
            await harness._run_no_landings_breaker_pass()

        assert not harness.scheduler.is_paused, (
            'scheduler should be unpaused after clean landing'
        )

    @pytest.mark.asyncio
    async def test_resume_resolves_info_escalation(self, tmp_path: Path) -> None:
        """After resume, the open breaker INFO escalation is resolved (no pending)."""
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=1000)
        await self._drive_to_trip(harness)

        # Confirm escalation was filed
        assert harness._escalation_queue is not None
        before = [
            e
            for e in harness._escalation_queue.get_by_task(
                _BREAKER_SENTINEL, status='pending'
            )
            if e.agent_role == _BREAKER_ROLE
        ]
        assert before, 'precondition: INFO escalation must be filed before resume'

        # Recovery pass
        _stub_merge_worker(harness, landings_total=6)
        with patch('shutil.disk_usage') as mock_du:
            mock_du.return_value = MagicMock(free=150_000)
            await harness._run_no_landings_breaker_pass()

        # After resume, no open breaker INFO escalation
        assert harness._escalation_queue is not None
        after = [
            e
            for e in harness._escalation_queue.get_by_task(
                _BREAKER_SENTINEL, status='pending'
            )
            if e.agent_role == _BREAKER_ROLE
        ]
        assert not after, (
            f'expected 0 pending breaker INFO escalations after resume, got {len(after)}'
        )

    @pytest.mark.asyncio
    async def test_disk_recovery_resumes(self, tmp_path: Path) -> None:
        """Disk recovered above absolute floor also unpauses the scheduler."""
        floor = 150_000  # floor well below free_at_trip=180_000
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=floor)
        await self._drive_to_trip(harness)
        assert harness.scheduler.is_paused

        # After filling window=3 with drop=10000 starting at 200000,
        # _free_at_trip = 180_000.  With absolute-floor semantics, any free_bytes
        # >= floor (150_000) resumes — including values below free_at_trip.
        _stub_merge_worker(harness, landings_total=5)  # same landings — disk path
        with patch('shutil.disk_usage') as mock_du:
            mock_du.return_value = MagicMock(free=160_000)  # 160k >= 150k floor
            await harness._run_no_landings_breaker_pass()

        assert not harness.scheduler.is_paused, (
            'scheduler should be unpaused after disk recovery above absolute floor'
        )


# ---------------------------------------------------------------------------
# step-13: lifecycle (start/stop/kill-switch/dedup)
# ---------------------------------------------------------------------------


class TestHarnessNoLandingsLifecycle:
    """_start/_stop_no_landings_breaker lifecycle mirrors main-tip-sweep.

    RED until step-14 GREEN adds the lifecycle methods.
    """

    @pytest.mark.asyncio
    async def test_start_creates_task_when_enabled(self, tmp_path: Path) -> None:
        """_start_no_landings_breaker() creates a live asyncio.Task when flag=True."""
        harness, _rs = _make_harness(tmp_path)
        harness.config.no_landings_breaker_enabled = True
        harness._start_no_landings_breaker()
        assert harness._no_landings_breaker_task is not None
        assert not harness._no_landings_breaker_task.done()
        # Cleanup
        await harness._stop_no_landings_breaker()

    @pytest.mark.asyncio
    async def test_start_noop_when_disabled(self, tmp_path: Path) -> None:
        """_start_no_landings_breaker() is a no-op when flag=False."""
        harness, _rs = _make_harness(tmp_path)
        harness.config.no_landings_breaker_enabled = False
        harness._start_no_landings_breaker()
        assert harness._no_landings_breaker_task is None

    @pytest.mark.asyncio
    async def test_start_dedup_no_duplicate_task(self, tmp_path: Path) -> None:
        """Calling _start_no_landings_breaker() twice does not spawn a duplicate."""
        harness, _rs = _make_harness(tmp_path)
        harness.config.no_landings_breaker_enabled = True
        harness._start_no_landings_breaker()
        task1 = harness._no_landings_breaker_task
        harness._start_no_landings_breaker()  # second call
        task2 = harness._no_landings_breaker_task
        assert task1 is task2, 'second start should not spawn a new task'
        await harness._stop_no_landings_breaker()

    @pytest.mark.asyncio
    async def test_stop_cancels_and_clears_task(self, tmp_path: Path) -> None:
        """_stop_no_landings_breaker() cancels the task and sets the field to None."""
        harness, _rs = _make_harness(tmp_path)
        harness.config.no_landings_breaker_enabled = True
        harness._start_no_landings_breaker()
        assert harness._no_landings_breaker_task is not None
        await harness._stop_no_landings_breaker()
        assert harness._no_landings_breaker_task is None

    @pytest.mark.asyncio
    async def test_stop_when_no_task_is_noop(self, tmp_path: Path) -> None:
        """_stop_no_landings_breaker() is a no-op when the task was never started."""
        harness, _rs = _make_harness(tmp_path)
        # Should not raise
        await harness._stop_no_landings_breaker()
        assert harness._no_landings_breaker_task is None
