"""The harness rewrites scheduler_state.json at every pause change.

A paused scheduler never ticks, so the throttled snapshot writer never runs
while it is paused. These tests pin the four forced writes that keep the file
truthful: pause, resume, restore and shutdown
(plans/instruments-honesty-prd.md decision 5, boundary rows 3 and 5).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from shared.project_layout import runs_db, scheduler_state_json
from shared.scheduler_state import (
    CheckedSchedulerState,
    SchedulerStateSnapshot,
    read_scheduler_state_checked,
)

from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.pause_policy import PauseClass
from orchestrator.run_store import RunStore

_COST_REASON = 'cost_ceiling_orch_exceeded'
_PRIOR_PAUSE_AT = '2026-11-20T03:11:09+00:00'


def _config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=tmp_path)


def _harness(config: OrchestratorConfig) -> Harness:
    harness = Harness(config)
    harness._run_store = RunStore(runs_db(config.project_root))
    harness._run_id = 'flush-run-id'
    harness.event_store = EventStore(config.project_root / 'events.db', 'flush-run-id')
    return harness


def _seed_pause(
    root: Path,
    project_id: str,
    *,
    reason: str = _COST_REASON,
    pause_class: PauseClass = PauseClass.COST_CEILING,
    ewa_value: float | None = None,
) -> None:
    RunStore(runs_db(root)).save_scheduler_pause(
        project_id=project_id,
        reason=reason,
        pause_at_iso=_PRIOR_PAUSE_AT,
        set_by_run_id='prior-run-id',
        ewa_value=ewa_value,
        pause_class=pause_class,
    )


def _file(root: Path) -> dict[str, Any]:
    return json.loads(scheduler_state_json(root).read_text())


def _read(root: Path) -> CheckedSchedulerState:
    return read_scheduler_state_checked(root, now=datetime.now(UTC))


class TestPauseAndResumeFlush:
    @pytest.mark.asyncio
    async def test_a_pause_reaches_the_file_without_a_tick(self, tmp_path: Path) -> None:
        harness = _harness(_config(tmp_path))
        before = datetime.now(UTC)

        await harness.pause_scheduler(_COST_REASON, PauseClass.COST_CEILING, file_escalation=False)

        written = _file(tmp_path)
        assert written['is_paused'] is True
        assert written['pause_class'] == 'cost_ceiling'
        assert datetime.fromisoformat(written['snapshot_at']) >= before
        view = _read(tmp_path)
        assert view.is_paused is True
        assert view.pause_disagreement is False
        assert view.pause_row is not None
        assert view.pause_row.pause_class == 'cost_ceiling'

    @pytest.mark.asyncio
    async def test_a_resume_reaches_the_file_without_a_tick(self, tmp_path: Path) -> None:
        harness = _harness(_config(tmp_path))
        await harness.pause_scheduler(_COST_REASON, PauseClass.COST_CEILING, file_escalation=False)

        await harness.resume_scheduler()

        assert _file(tmp_path)['is_paused'] is False
        view = _read(tmp_path)
        assert view.pause_row is None
        assert view.pause_disagreement is False

    @pytest.mark.asyncio
    async def test_pause_and_resume_record_their_events_before_they_flush(self, tmp_path: Path) -> None:
        harness = _harness(_config(tmp_path))
        order: list[str] = []
        event_store = harness.event_store
        assert event_store is not None
        emit = event_store.emit
        flush = harness.scheduler.flush_state_snapshot

        def recording_emit(event_type: Any, *args: Any, **kwargs: Any) -> Any:
            order.append(str(event_type.value))
            return emit(event_type, *args, **kwargs)

        async def recording_flush() -> None:
            order.append('flush')
            await flush()

        event_store.emit = recording_emit  # type: ignore[method-assign]
        harness.scheduler.flush_state_snapshot = recording_flush  # type: ignore[method-assign]

        await harness.pause_scheduler(_COST_REASON, PauseClass.COST_CEILING, file_escalation=False)
        await harness.resume_scheduler()

        assert order == ['scheduler_paused', 'flush', 'scheduler_resumed', 'flush']


class TestRestoreFlush:
    @pytest.mark.asyncio
    async def test_a_restored_pause_writes_the_file(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        _seed_pause(tmp_path, config.fused_memory.project_id)
        harness = _harness(config)
        assert not scheduler_state_json(tmp_path).exists()

        await harness._load_persisted_scheduler_pause()

        written = _file(tmp_path)
        assert written['is_paused'] is True
        assert written['pause_reason'] == _COST_REASON
        assert _read(tmp_path).pause_disagreement is False

    @pytest.mark.asyncio
    async def test_a_declined_restore_replaces_the_dead_process_file(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        _seed_pause(
            tmp_path,
            config.fused_memory.project_id,
            reason='ewa_trip_30.0000',
            pause_class=PauseClass.EWA_RATE,
            ewa_value=config.digest_ewa_threshold / 2,
        )
        stale_at = datetime(2026, 11, 20, 3, 10, 54, tzinfo=UTC)
        path = scheduler_state_json(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            SchedulerStateSnapshot(is_paused=True, pause_reason='stale', snapshot_at=stale_at).to_json(),
        ))
        harness = _harness(config)

        await harness._load_persisted_scheduler_pause()

        written = _file(tmp_path)
        assert written['is_paused'] is False
        assert datetime.fromisoformat(written['snapshot_at']) > stale_at
        view = _read(tmp_path)
        assert view.is_paused is False
        assert view.pause_row is None
        assert view.pause_disagreement is False


class TestRunFlushOrder:
    @pytest.mark.asyncio
    async def test_run_flushes_after_the_restore_and_at_shutdown(
        self, tmp_path: Path, mock_orch_config: MagicMock,
    ) -> None:
        mock_orch_config.fused_memory.project_id = 'test'
        _seed_pause(tmp_path, 'test')

        with patch('orchestrator.harness.McpLifecycle') as mock_mcp_cls, \
             patch('orchestrator.harness.Scheduler'), \
             patch('orchestrator.harness.BriefingAssembler'):
            h = Harness(mock_orch_config)

        mock_mcp = mock_mcp_cls.return_value
        mock_mcp.start = AsyncMock()
        mock_mcp.stop = AsyncMock()

        h.git_ops = MagicMock()
        h.git_ops.has_dirty_working_tree = AsyncMock(return_value=None)
        h.git_ops.worktree_base = tmp_path / '.worktrees'

        h._start_escalation_server = AsyncMock()  # type: ignore[method-assign]
        h._start_merge_worker = AsyncMock()  # type: ignore[method-assign]
        h._dismiss_stale_escalations = AsyncMock()  # type: ignore[method-assign]
        h._start_orphan_l0_reaper = MagicMock()  # type: ignore[method-assign]
        h._start_terminal_status_watcher = MagicMock()  # type: ignore[method-assign]
        h._tag_task_modules = AsyncMock()  # type: ignore[method-assign]
        h._recover_crashed_tasks = AsyncMock()  # type: ignore[method-assign]
        h._reconcile_lane_checkouts = AsyncMock()  # type: ignore[method-assign]
        h._reconcile_stranded_in_progress = AsyncMock()  # type: ignore[method-assign]
        h._tag_prd_metadata = AsyncMock()  # type: ignore[method-assign]

        h.scheduler = MagicMock()
        h.scheduler.get_tasks = AsyncMock(return_value=[])
        h.scheduler.get_statuses = AsyncMock(return_value=({}, None))
        h.scheduler.set_task_status = AsyncMock()

        order: list[str] = []
        h.scheduler.pause = MagicMock(side_effect=lambda *_a, **_k: order.append('pause'))
        h.scheduler.flush_state_snapshot = AsyncMock(side_effect=lambda: order.append('flush'))
        h.scheduler.finish_startup = MagicMock(side_effect=lambda: order.append('finish_startup'))

        await h.run(dry_run=True)

        h.scheduler.pause.assert_called_once_with(_COST_REASON, PauseClass.COST_CEILING)
        restored = order.index('pause')
        assert 'flush' in order[restored:order.index('finish_startup')], (
            f'the restore must flush before startup finishes; got {order}'
        )
        assert order[-1] == 'flush', f'shutdown must flush last; got {order}'
