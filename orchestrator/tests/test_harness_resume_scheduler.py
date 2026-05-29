"""Tests for the scheduler-resume escape hatch and auto-resume-on-resolve.

Covers:
  - Harness.force_resume_scheduler (backs the resume_scheduler MCP tool):
    structured return, idempotency, persisted-pause clear.
  - _on_escalation_resolved auto-resuming the scheduler when the
    ``__scheduler__`` pause L1 is resolved (the documented-but-previously-
    unwired resume path).
  - The 2026-05-29 reify regression: the resolve callback can fire OFF the
    orchestrator loop (sync MCP resolve_issue on a FastMCP threadpool worker),
    where a bare asyncio.create_task raised "no running event loop".
    _schedule_coro_threadsafe must route the resume back onto the loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from escalation.models import Escalation

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.run_store import RunStore

SENTINEL = Harness._SCHEDULER_PAUSE_SENTINEL


def _make_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Harness with a real Scheduler + EventStore and a spy RunStore.

    Mirrors test_harness_park_stop._make_harness_with_mocks but kept local so
    this module stands alone.  Returns (harness, mock_run_store).
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-resume-0001'
    harness.event_store = EventStore(tmp_path / 'events.db', 'run-resume-0001')
    return harness, mock_run_store


def _sentinel_esc(status: str = 'resolved', resolved_by: str = 'interactive') -> Escalation:
    """A resolved scheduler-pause L1 against the synthetic __scheduler__ task."""
    return Escalation(
        id=f'esc-{SENTINEL}-36',
        task_id=SENTINEL,
        agent_role='orchestrator-scheduler',
        severity='blocking',
        category='scheduler_paused',
        summary='Scheduler paused — dispatch halted',
        level=1,
        status=status,
        resolved_by=resolved_by,
    )


async def _wait_until(predicate, timeout: float = 1.0) -> None:
    """Poll *predicate* (cheap, no-arg) until true or timeout, yielding the loop."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            return
        await asyncio.sleep(0.02)


class TestForceResumeScheduler:
    """Harness.force_resume_scheduler — the structured wrapper the MCP tool calls."""

    @pytest.mark.asyncio
    async def test_resume_when_paused_reports_and_clears(self, tmp_path: Path) -> None:
        harness, mock_run_store = _make_harness(tmp_path)
        await harness.pause_scheduler('park-stop: test burst')

        result = await harness.force_resume_scheduler('collision burst cleared')

        assert harness.scheduler.is_paused is False
        assert result == {
            'resumed': True,
            'was_paused': True,
            'prior_reason': 'park-stop: test burst',
            'reason': 'collision burst cleared',
        }
        # Persisted pause cleared so a restart won't resurrect it.
        mock_run_store.clear_scheduler_pause.assert_called_once()

    @pytest.mark.asyncio
    async def test_resume_when_not_paused_is_idempotent(self, tmp_path: Path) -> None:
        """No pause → resumed=False, but still clears any stale persisted row."""
        harness, mock_run_store = _make_harness(tmp_path)
        assert harness.scheduler.is_paused is False

        result = await harness.force_resume_scheduler('nothing to do')

        assert result['resumed'] is False
        assert result['was_paused'] is False
        assert result['prior_reason'] is None
        # Disk-heal: clear is called unconditionally so a divergent runs.db row
        # (memory says running, disk says paused) cannot re-pause on restart.
        mock_run_store.clear_scheduler_pause.assert_called_once()


class TestAutoResumeOnEscalationResolve:
    """Resolving the __scheduler__ pause L1 must auto-resume the scheduler."""

    @pytest.mark.asyncio
    async def test_resolve_sentinel_resumes_in_loop(self, tmp_path: Path) -> None:
        harness, _ = _make_harness(tmp_path)
        await harness.pause_scheduler('park-stop: collision burst')

        # Called on the loop thread (like the watcher supervisor) → create_task.
        harness._on_escalation_resolved(_sentinel_esc())
        await asyncio.gather(*list(harness._background_tasks))

        assert harness.scheduler.is_paused is False, (
            'Resolving the scheduler-pause L1 must auto-resume dispatch'
        )

    @pytest.mark.asyncio
    async def test_resolve_sentinel_when_not_paused_is_noop(self, tmp_path: Path) -> None:
        """Gate on is_paused: a resolution arriving post-resume schedules nothing."""
        harness, _ = _make_harness(tmp_path)
        assert harness.scheduler.is_paused is False

        harness._on_escalation_resolved(_sentinel_esc())

        # Nothing was scheduled (no resume coro), and we stay unpaused.
        assert harness._background_tasks == set()
        assert harness.scheduler.is_paused is False

    @pytest.mark.asyncio
    async def test_dismissed_sentinel_does_not_resume(self, tmp_path: Path) -> None:
        """Dismissing (not resolving) the pause L1 means abandon — stay paused."""
        harness, _ = _make_harness(tmp_path)
        await harness.pause_scheduler('park-stop: abandoned')

        harness._on_escalation_resolved(_sentinel_esc(status='dismissed'))
        await asyncio.gather(*list(harness._background_tasks))

        assert harness.scheduler.is_paused is True, (
            'A dismissed pause L1 must NOT resume the scheduler'
        )

    @pytest.mark.asyncio
    async def test_off_loop_resolve_still_resumes(self, tmp_path: Path) -> None:
        """REGRESSION (2026-05-29): callback fires off the loop → resume still fires.

        Reproduces the sync-MCP-tool path: resolve_issue runs on a FastMCP
        threadpool worker, so _on_escalation_resolved executes with no running
        loop in its thread.  The old bare asyncio.create_task raised
        "no running event loop"; _schedule_coro_threadsafe must route the
        resume coroutine onto the captured loop instead.
        """
        harness, _ = _make_harness(tmp_path)
        # run() normally captures this; set it directly since we don't start run().
        harness._loop = asyncio.get_running_loop()
        await harness.pause_scheduler('park-stop: off-loop burst')

        # Run the callback on a worker thread — no event loop in that thread.
        await asyncio.to_thread(harness._on_escalation_resolved, _sentinel_esc())

        await _wait_until(lambda: not harness.scheduler.is_paused)
        assert harness.scheduler.is_paused is False, (
            'Off-loop resolve must still resume via run_coroutine_threadsafe'
        )
