"""Tests for the WP-D backlog escalation policy + orchestrator detector."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from fused_memory.reconciliation.backlog_policy import BacklogPolicy
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.services.orchestrator_detector import (
    is_orchestrator_live_for,
)

if TYPE_CHECKING:
    pass


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'backlog_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


class _StubQueue:
    """Minimal EventQueue-like stub exposing ``stats()``."""

    def __init__(self, queue_depth: int = 0, retry_in_flight: int = 0) -> None:
        self._depth = queue_depth
        self._retry = retry_in_flight

    def stats(self) -> dict:
        return {
            'queue_depth': self._depth,
            'retry_in_flight': self._retry,
            'queue_capacity': 10_000,
            'last_commit_ts': None,
            'events_committed': 0,
            'overflow_drops': 0,
            'dead_letters': 0,
            'drainer_running': True,
        }


async def _seed_buffered(event_buffer: EventBuffer, project_id: str, n: int) -> None:
    """Insert ``n`` buffered events for ``project_id`` directly via the schema."""
    import uuid
    from datetime import UTC, datetime

    from fused_memory.models.reconciliation import (
        EventSource,
        EventType,
        ReconciliationEvent,
    )

    for _ in range(n):
        event = ReconciliationEvent(
            id=str(uuid.uuid4()),
            type=EventType.task_created,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload={'seed': True},
        )
        await event_buffer.push(event)


# ── BacklogPolicy.check ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ok_verdict_when_under_threshold(event_buffer):
    """Under the hard limit → ok verdict, no escalation, no rejection."""
    await _seed_buffered(event_buffer, 'proj', n=3)

    def detector(_root: str) -> bool:
        return False

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        detector,
        hard_limit=10,
    )
    verdict = await policy.check('proj', project_root='/does/not/matter')
    assert verdict.outcome == 'ok'
    assert verdict.to_error_dict() == {}


@pytest.mark.asyncio
async def test_rejection_verdict_when_over_threshold_and_no_orchestrator(
    event_buffer, tmp_path,
):
    """Over threshold + no orchestrator → structured rejection, no file written."""
    await _seed_buffered(event_buffer, 'proj', n=6)
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()

    def detector(_root: str) -> bool:
        return False

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        detector,
        hard_limit=5,
    )
    verdict = await policy.check('proj', project_root=str(project_root))
    assert verdict.outcome == 'rejection'
    assert verdict.backlog == 6
    assert verdict.threshold == 5
    assert verdict.project_id == 'proj'
    err = verdict.to_error_dict()
    assert err['error_type'] == 'ReconciliationBacklogExceeded'
    assert 'backlog 6 > limit 5' in err['error']
    # No escalation file.
    esc_dir = project_root / 'data' / 'escalations'
    assert not esc_dir.exists() or not any(esc_dir.iterdir())


@pytest.mark.asyncio
async def test_escalation_when_over_threshold_and_orchestrator_live(
    event_buffer, tmp_path,
):
    """Over threshold + orchestrator live → escalation JSON on disk."""
    await _seed_buffered(event_buffer, 'proj', n=12)
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()

    def detector(root: str) -> bool:
        return root == str(project_root)

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        detector,
        hard_limit=10,
    )
    verdict = await policy.check('proj', project_root=str(project_root))
    assert verdict.outcome == 'escalated'
    assert verdict.escalation_path is not None
    path = Path(verdict.escalation_path)
    assert path.exists()
    body = json.loads(path.read_text())
    assert body['id'].startswith('esc-reconciliation-backlog-')
    assert body['severity'] == 'blocking'
    assert body['level'] == 1
    assert body['suggested_action'] == 'drain_reconciliation'
    assert body['backlog'] == 12
    assert body['threshold'] == 10
    assert body['project_id'] == 'proj'
    assert body['workflow_state'] == 'infra'
    assert body['category'] == 'infra_issue'


@pytest.mark.asyncio
async def test_rate_limit_prevents_spam(event_buffer, tmp_path):
    """Two triggers inside the rate window → one file only."""
    await _seed_buffered(event_buffer, 'proj', n=12)
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()
    clock = {'now': 1_000_000.0}

    def now() -> float:
        return clock['now']

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: True,
        hard_limit=10,
        rate_limit_seconds=900.0,
        time_provider=now,
    )
    v1 = await policy.check('proj', project_root=str(project_root))
    clock['now'] += 60.0  # within window
    v2 = await policy.check('proj', project_root=str(project_root))

    assert v1.outcome == 'escalated'
    assert v2.outcome == 'escalated'
    # Only the first wrote a file; second returned escalated verdict with no path.
    assert v1.escalation_path is not None
    assert v2.escalation_path is None
    esc_files = list((project_root / 'data' / 'escalations').iterdir())
    assert len(esc_files) == 1


@pytest.mark.asyncio
async def test_rate_limit_allows_after_window(event_buffer, tmp_path):
    """Advance clock past the rate window → second trigger writes another file."""
    await _seed_buffered(event_buffer, 'proj', n=12)
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()
    clock = {'now': 1_000_000.0}

    def now() -> float:
        return clock['now']

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: True,
        hard_limit=10,
        rate_limit_seconds=900.0,
        time_provider=now,
    )
    v1 = await policy.check('proj', project_root=str(project_root))
    clock['now'] += 901.0  # just past window
    v2 = await policy.check('proj', project_root=str(project_root))

    assert v1.escalation_path is not None
    assert v2.escalation_path is not None
    assert v1.escalation_path != v2.escalation_path
    esc_files = sorted((project_root / 'data' / 'escalations').iterdir())
    assert len(esc_files) == 2


@pytest.mark.asyncio
async def test_on_judge_halt_writes_escalation(event_buffer, tmp_path):
    """Judge halt routes through escalation path when orchestrator is live."""
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: True,
        hard_limit=500,
    )
    policy.register_project_root('proj', str(project_root))
    verdict = await policy.on_judge_halt('proj', reason='too many serious findings')

    assert verdict.outcome == 'escalated'
    assert verdict.error_type == 'ReconciliationJudgeHalted'
    files = list((project_root / 'data' / 'escalations').iterdir())
    assert len(files) == 1
    body = json.loads(files[0].read_text())
    assert body['error_type'] == 'ReconciliationJudgeHalted'
    assert 'too many serious findings' in body['detail']


@pytest.mark.asyncio
async def test_on_watchdog_wedge_writes_escalation_with_wedge_error_type(
    event_buffer, tmp_path,
):
    """Wedge payload → escalation with error_type=SqliteDrainerWedged."""
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: True,
        hard_limit=500,
    )
    policy.register_project_root('proj', str(project_root))
    verdicts = await policy.on_watchdog_wedge({
        'stale_for_seconds': 180.0,
        'queue_depth': 42,
        'retry_in_flight': 3,
    })

    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.outcome == 'escalated'
    assert v.error_type == 'SqliteDrainerWedged'
    files = list((project_root / 'data' / 'escalations').iterdir())
    assert len(files) == 1
    body = json.loads(files[0].read_text())
    assert body['error_type'] == 'SqliteDrainerWedged'
    assert 'stale_for_seconds' in body['detail']


# ── orchestrator_detector ─────────────────────────────────────────────────


def test_orchestrator_detector_stale_lock_pid_dead(tmp_path):
    """PID in lock points to a process that doesn't exist → not live."""
    project_root = tmp_path / 'proj_root'
    lock_dir = project_root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True)
    # Very high PID unlikely to be live.
    lock_dir.joinpath('orchestrator.lock').write_text(
        'PID 2147483646 started 2026-04-13T00:00:00Z\n', encoding='utf-8',
    )
    assert is_orchestrator_live_for(project_root) is False


def test_orchestrator_detector_live_pid(tmp_path):
    """PID for the current process → live (os.kill(pid,0) succeeds)."""
    project_root = tmp_path / 'proj_root'
    lock_dir = project_root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True)
    my_pid = os.getpid()
    lock_dir.joinpath('orchestrator.lock').write_text(
        f'PID {my_pid} started 2026-04-18T00:00:00Z\n', encoding='utf-8',
    )
    assert is_orchestrator_live_for(project_root) is True


def test_orchestrator_detector_no_lock_file(tmp_path):
    """No orchestrator.lock → not live."""
    project_root = tmp_path / 'proj_root'
    project_root.mkdir()
    assert is_orchestrator_live_for(project_root) is False


def test_orchestrator_detector_unparseable_lock(tmp_path):
    """Garbage in lock → not live (defensive)."""
    project_root = tmp_path / 'proj_root'
    lock_dir = project_root / 'data' / 'orchestrator'
    lock_dir.mkdir(parents=True)
    lock_dir.joinpath('orchestrator.lock').write_text('not a pid\n', encoding='utf-8')
    assert is_orchestrator_live_for(project_root) is False


# ── TaskInterceptor integration ───────────────────────────────────────────


@pytest.fixture
def _taskmaster_mock():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={'id': '1', 'status': 'pending', 'title': 'T'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '2', 'title': 'New'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.add_subtask = AsyncMock(return_value={'id': '1.1'})
    tm.remove_task = AsyncMock(return_value={'success': True})
    tm.add_dependency = AsyncMock(return_value={'success': True})
    tm.remove_dependency = AsyncMock(return_value={'success': True})
    tm.ensure_connected = AsyncMock()
    return tm


@pytest.mark.asyncio
async def test_task_interceptor_add_task_rejects_when_over_limit(
    event_buffer, tmp_path, _taskmaster_mock,
):
    """When policy rejects, interceptor returns error dict without mutating state."""
    import contextlib

    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.middleware.ticket_store import TicketStore

    await _seed_buffered(event_buffer, 'proj_root', n=20)

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: False,  # no orchestrator
        hard_limit=5,
    )
    store = TicketStore(tmp_path / 'reject_tickets.db')
    await store.initialize()
    try:
        interceptor = TaskInterceptor(
            _taskmaster_mock,
            targeted_reconciler=None,
            event_buffer=event_buffer,
            backlog_policy=policy,
            ticket_store=store,
        )
        project_root = str(tmp_path / 'proj_root')
        result = await interceptor.add_task(
            project_root=project_root, title='Should be rejected',
        )
        assert result.get('error_type') == 'ReconciliationBacklogExceeded'
        assert result['backlog'] == 20
        assert result['threshold'] == 5
        # taskmaster.add_task must NOT be called — the whole point of the rejection.
        _taskmaster_mock.add_task.assert_not_called()
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt


@pytest.mark.asyncio
async def test_task_interceptor_add_task_ok_when_under_limit(
    event_buffer, tmp_path, _taskmaster_mock,
):
    """Under-threshold → normal add_task flow, taskmaster called."""
    import contextlib

    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.middleware.ticket_store import TicketStore

    policy = BacklogPolicy(
        event_buffer,
        _StubQueue(),
        lambda _: False,
        hard_limit=500,
    )
    config = MagicMock()
    config.curator.enabled = False
    store = TicketStore(tmp_path / 'ok_tickets.db')
    await store.initialize()
    try:
        interceptor = TaskInterceptor(
            _taskmaster_mock,
            targeted_reconciler=None,
            event_buffer=event_buffer,
            backlog_policy=policy,
            config=config,
            ticket_store=store,
        )
        project_root = str(tmp_path / 'proj_root')
        result = await interceptor.add_task(
            project_root=project_root, title='Under the limit',
        )
        assert result == {'id': '2', 'title': 'New'}
        _taskmaster_mock.add_task.assert_called_once()
    finally:
        await store.close()
        for _wt in list(interceptor._worker_tasks.values()):
            if not _wt.done():
                _wt.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await _wt
