"""Harness-level tests for scheduler-pause digest ergonomics (task 5311).

Covers the EWA-trip storm annotation on the pause reason, the paused re-pend
marker on a resolution, and the digest's '## Paused, waiting' section.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from _orch_helpers import claimant_row, wire_scheduler_liveness_mock
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.cost_store import CostStore
from shared.project_layout import escalation_queue_dir
from test_harness_park_stop import _make_harness_with_mocks, _query_events

import orchestrator.harness as harness_mod
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.pause_policy import PauseClass
from orchestrator.run_store import RunStore


@pytest_asyncio.fixture
async def cost_store(tmp_path: Path):
    store = CostStore(tmp_path / 'cost.db')
    await store.open()
    yield store
    await store.close()


def _submit(
    queue: EscalationQueue, task_id: str, *, category: str, level: int, **fields: object,
) -> str:
    escalation = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='test-filer',
        severity='blocking',
        category=category,
        summary=f'{category} for task {task_id}',
        level=level,
    )
    return queue.submit(dataclasses.replace(escalation, **fields))


def _digest_files(tmp_path: Path) -> list[Path]:
    return sorted((tmp_path / 'data' / 'digests').glob('digest-*.md'))


def _tripping_harness(tmp_path: Path, cost_store: CostStore, lifecycle_events: int):
    harness, mock_run_store, event_store = _make_harness_with_mocks(tmp_path)
    harness.config = OrchestratorConfig(
        project_root=tmp_path,
        digest_every_n_escalations=1,
        digest_ewa_threshold=0.01,
        digest_ewa_alpha=1.0,
    )
    harness.cost_store = cost_store
    harness._escalation_event_count = lifecycle_events
    harness._last_digest_event_count = 0
    harness._escalation_submit_count = lifecycle_events
    harness._last_digest_submit_count = 0
    return harness, mock_run_store, event_store


def _queue_for(harness: Harness, tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(escalation_queue_dir(harness.config.project_root))


class TestEwaTripStormAnnotation:
    @pytest.mark.asyncio
    async def test_trip_reason_names_the_dominant_category(
        self, tmp_path: Path, cost_store: CostStore,
    ) -> None:
        harness, mock_run_store, event_store = _tripping_harness(tmp_path, cost_store, 4)
        queue = _queue_for(harness, tmp_path)
        for task_id in ('101', '102', '103'):
            _submit(queue, task_id, category='retry_cap_exhausted', level=1)
        _submit(queue, '104', category='design_concern', level=0)

        await harness._maybe_write_digest()

        assert harness.scheduler.is_paused is True
        reason = harness.scheduler.pause_reason
        assert reason is not None
        assert re.match(r'^ewa_trip_\d+\.\d{4} · storm: 3/4 retry_cap_exhausted$', reason)
        assert reason.startswith('ewa_trip_')

        assert mock_run_store.save_scheduler_pause.call_args.kwargs['reason'] == reason

        [digest_file] = _digest_files(tmp_path)
        markdown = digest_file.read_text()
        assert '**TRIPPED**' in markdown
        assert '3/4 retry_cap_exhausted' in markdown

        [paused_row] = _query_events(event_store, 'scheduler_paused')
        assert json.loads(paused_row['data'])['reason'] == reason

        harness._ewa_value = 5.0
        await harness.resume_scheduler()
        assert harness._ewa_value == 0.0

    @pytest.mark.asyncio
    async def test_empty_window_reason_carries_no_storm_suffix(
        self, tmp_path: Path, cost_store: CostStore,
    ) -> None:
        harness, _, _ = _tripping_harness(tmp_path, cost_store, 1)

        await harness._maybe_write_digest()

        assert harness.scheduler.is_paused is True
        reason = harness.scheduler.pause_reason
        assert reason is not None
        assert re.match(r'^ewa_trip_\d+\.\d{4}$', reason)


PAUSE_REASON = 'ewa_trip_29.1818 · storm: 22/29 retry_cap_exhausted'


class _ResolutionRig(NamedTuple):
    harness: Harness
    scheduler: MagicMock
    queue: EscalationQueue
    event_store: EventStore


@pytest.fixture
def rig(tmp_path: Path, mock_orch_config) -> _ResolutionRig:
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        harness = Harness(mock_orch_config)
    scheduler = MagicMock()
    scheduler.get_status = AsyncMock(return_value='blocked')
    scheduler.set_task_status = AsyncMock()
    scheduler.update_task = AsyncMock(return_value=True)
    scheduler.get_task = AsyncMock(return_value=claimant_row('blocked', task_id='5260'))
    wire_scheduler_liveness_mock(scheduler)
    scheduler.is_paused = True
    scheduler.pause_reason = PAUSE_REASON
    harness.scheduler = scheduler
    queue = EscalationQueue(tmp_path / 'esc')
    harness._escalation_queue = queue
    event_store = EventStore(tmp_path / 'events.db', 'run-x')
    harness.event_store = event_store
    return _ResolutionRig(harness, scheduler, queue, event_store)


async def _resolve_stranded(
    harness: Harness, queue: EscalationQueue, **fields: object,
) -> str:
    esc_id = _submit(
        queue, '5260',
        category='stranded_blocked',
        level=1,
        agent_role='harness-stranded-blocked-reaper',
        suggested_action='resume',
        **fields,
    )
    resolved = queue.resolve(esc_id, 'resume: lane is green', resolved_by='steward')
    assert resolved is not None
    harness._on_escalation_resolved(resolved)
    await asyncio.gather(*list(harness._background_tasks))
    return esc_id


def _repend_rows(event_store: EventStore) -> list[dict]:
    return _query_events(event_store, 'repend_while_paused')


def _resolution(queue: EscalationQueue, esc_id: str) -> str:
    stored = queue.get(esc_id)
    assert stored is not None
    assert stored.resolution is not None
    return stored.resolution


def _assert_flagged(rig: _ResolutionRig, esc_id: str) -> None:
    resolution = _resolution(rig.queue, esc_id)
    assert resolution.startswith('resume: lane is green')
    assert harness_mod.REPEND_WHILE_PAUSED_MARKER in resolution
    assert PAUSE_REASON in resolution
    [row] = _repend_rows(rig.event_store)
    assert row['task_id'] == '5260'
    assert json.loads(row['data']) == {'escalation_id': esc_id, 'pause_reason': PAUSE_REASON}


@pytest.mark.asyncio
class TestRependWhilePaused:
    async def test_resume_while_paused_marks_resolution_and_emits_event(
        self, rig: _ResolutionRig, caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING):
            esc_id = await _resolve_stranded(rig.harness, rig.queue)

        rig.scheduler.set_task_status.assert_awaited_once_with('5260', 'pending')
        _assert_flagged(rig, esc_id)
        assert any(
            r.levelno == logging.WARNING and '5260' in r.getMessage() and esc_id in r.getMessage()
            for r in caplog.records
        )

    async def test_not_paused_flips_without_marker(self, rig: _ResolutionRig) -> None:
        rig.scheduler.is_paused = False

        esc_id = await _resolve_stranded(rig.harness, rig.queue)

        rig.scheduler.set_task_status.assert_awaited_once_with('5260', 'pending')
        assert _resolution(rig.queue, esc_id) == 'resume: lane is green'
        assert _repend_rows(rig.event_store) == []

    async def test_no_flip_means_no_marker(self, rig: _ResolutionRig) -> None:
        rig.scheduler.get_status = AsyncMock(return_value='done')

        esc_id = await _resolve_stranded(rig.harness, rig.queue)

        rig.scheduler.set_task_status.assert_not_awaited()
        assert _resolution(rig.queue, esc_id) == 'resume: lane is green'
        assert _repend_rows(rig.event_store) == []

    async def test_restart_while_paused_is_flagged(self, rig: _ResolutionRig) -> None:
        esc_id = await _resolve_stranded(rig.harness, rig.queue, resolution_action='restart')

        rig.scheduler.set_task_status.assert_awaited_once_with('5260', 'pending')
        _assert_flagged(rig, esc_id)

    async def test_infra_hold_resume_while_paused_is_flagged(self, rig: _ResolutionRig) -> None:
        rig.scheduler.get_task = AsyncMock(
            return_value=claimant_row('infra-hold', task_id='5260'),
        )

        esc_id = await _resolve_stranded(rig.harness, rig.queue)

        rig.scheduler.set_task_status.assert_awaited_once_with('5260', 'pending')
        _assert_flagged(rig, esc_id)

    async def test_missing_queue_still_emits_event_and_never_raises(
        self, rig: _ResolutionRig,
    ) -> None:
        rig.harness._escalation_queue = None

        esc_id = await _resolve_stranded(rig.harness, rig.queue)

        rig.scheduler.set_task_status.assert_awaited_once_with('5260', 'pending')
        assert _resolution(rig.queue, esc_id) == 'resume: lane is green'
        [row] = _repend_rows(rig.event_store)
        assert json.loads(row['data'])['escalation_id'] == esc_id


def _hours_ago(hours: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


def _statuses_read(
    statuses: dict[str, str], error: Exception | None = None,
) -> AsyncMock:
    return AsyncMock(return_value=(statuses, error))


class _DigestRig(NamedTuple):
    harness: Harness
    queue: EscalationQueue
    tmp_path: Path


@pytest.fixture
def digest_rig(
    tmp_path: Path, cost_store: CostStore, monkeypatch: pytest.MonkeyPatch,
) -> _DigestRig:
    harness, _, _ = _make_harness_with_mocks(tmp_path)
    harness.config = OrchestratorConfig(
        project_root=tmp_path,
        digest_every_n_escalations=1,
        digest_ewa_threshold=1e9,
    )
    harness.cost_store = cost_store
    queue = _queue_for(harness, tmp_path)
    harness._escalation_queue = queue
    scheduler = harness.scheduler
    monkeypatch.setattr(
        scheduler, 'get_task', AsyncMock(return_value=claimant_row('blocked', task_id='5260')),
    )
    monkeypatch.setattr(scheduler, 'get_status', AsyncMock(return_value='blocked'))
    monkeypatch.setattr(scheduler, 'set_task_status', AsyncMock())
    monkeypatch.setattr(scheduler, 'update_task', AsyncMock(return_value=True))
    monkeypatch.setattr(scheduler, 'get_statuses', _statuses_read({'5260': 'pending'}))
    return _DigestRig(harness, queue, tmp_path)


async def _write_digest(rig: _DigestRig) -> str:
    before = _digest_files(rig.tmp_path)
    rig.harness._escalation_event_count = rig.harness._last_digest_event_count + 1
    await rig.harness._maybe_write_digest()
    after = _digest_files(rig.tmp_path)
    assert len(after) == len(before) + 1
    return after[-1].read_text()


async def _pause_with_overdue_l1_and_repend(rig: _DigestRig) -> tuple[str, str]:
    pause_esc_id = _submit(
        rig.queue, Harness._SCHEDULER_PAUSE_SENTINEL,
        category='scheduler_paused', level=1, timestamp=_hours_ago(3),
    )
    await rig.harness.pause_scheduler(PAUSE_REASON, PauseClass.EWA_RATE)
    assert len(rig.queue.get_pending()) == 1
    repend_esc_id = await _resolve_stranded(rig.harness, rig.queue)
    return pause_esc_id, repend_esc_id


def _paused_waiting(markdown: str) -> str:
    start = markdown.index('## Paused, waiting\n')
    end = markdown.find('\n## ', start + 1)
    return markdown[start:end]


@pytest.mark.asyncio
class TestPausedWaitingDigest:
    async def test_digest_lists_the_waiting_task_and_the_overdue_pause_l1(
        self, digest_rig: _DigestRig,
    ) -> None:
        pause_esc_id, repend_esc_id = await _pause_with_overdue_l1_and_repend(digest_rig)

        section = _paused_waiting(await _write_digest(digest_rig))

        assert 'Paused with 1 re-pended tasks waiting' in section
        [task_line] = [line for line in section.splitlines() if 'task 5260' in line]
        assert repend_esc_id in task_line
        [overdue_line] = [line for line in section.splitlines() if pause_esc_id in line]
        assert re.search(r'pending 3\.\dh', overdue_line)

    async def test_overdue_pause_l1_resurfaces_in_every_digest(
        self, digest_rig: _DigestRig,
    ) -> None:
        pause_esc_id, _ = await _pause_with_overdue_l1_and_repend(digest_rig)

        await _write_digest(digest_rig)
        second = _paused_waiting(await _write_digest(digest_rig))

        assert pause_esc_id in second

    async def test_resume_clears_the_waiting_line_but_not_the_overdue_l1(
        self, digest_rig: _DigestRig,
    ) -> None:
        pause_esc_id, _ = await _pause_with_overdue_l1_and_repend(digest_rig)
        await _write_digest(digest_rig)

        await digest_rig.harness.resume_scheduler()
        section = _paused_waiting(await _write_digest(digest_rig))

        assert '- Not paused' in section.splitlines()
        assert 'task 5260' not in section
        assert pause_esc_id in section

    @pytest.mark.parametrize('status', ['cancelled', 'blocked', 'done'])
    async def test_a_repended_task_that_left_pending_is_not_waiting(
        self, digest_rig: _DigestRig, status: str,
    ) -> None:
        await _pause_with_overdue_l1_and_repend(digest_rig)
        digest_rig.harness.scheduler.get_statuses = _statuses_read({'5260': status})

        section = _paused_waiting(await _write_digest(digest_rig))

        assert 'Paused with 0 re-pended tasks waiting' in section
        assert 'task 5260' not in section
        digest_rig.harness.scheduler.get_statuses.assert_awaited_once_with(['5260'])

    async def test_a_repended_task_that_no_longer_exists_is_not_waiting(
        self, digest_rig: _DigestRig,
    ) -> None:
        await _pause_with_overdue_l1_and_repend(digest_rig)
        digest_rig.harness.scheduler.get_statuses = _statuses_read({})

        section = _paused_waiting(await _write_digest(digest_rig))

        assert 'Paused with 0 re-pended tasks waiting' in section

    async def test_an_unreadable_status_keeps_the_waiting_task(
        self, digest_rig: _DigestRig, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await _pause_with_overdue_l1_and_repend(digest_rig)
        digest_rig.harness.scheduler.get_statuses = _statuses_read(
            {}, RuntimeError('fused-memory down'),
        )

        with caplog.at_level(logging.WARNING):
            section = _paused_waiting(await _write_digest(digest_rig))

        assert 'Paused with 1 re-pended tasks waiting' in section
        assert 'task 5260' in section
        assert any('fused-memory down' in r.getMessage() for r in caplog.records)

    async def test_young_pause_l1_is_not_listed(self, digest_rig: _DigestRig) -> None:
        young_esc_id = _submit(
            digest_rig.queue, Harness._SCHEDULER_PAUSE_SENTINEL,
            category='scheduler_paused', level=1, timestamp=_hours_ago(1),
        )

        section = _paused_waiting(await _write_digest(digest_rig))

        assert young_esc_id not in section


def _runs_db(tmp_path: Path) -> Path:
    db_dir = tmp_path / 'data' / 'orchestrator'
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / 'runs.db'


async def _pause_and_repend_in_first_process(
    rig: _DigestRig, reason: str, pause_class: PauseClass,
) -> None:
    rig.harness._run_store = RunStore(_runs_db(rig.tmp_path))
    await rig.harness.pause_scheduler(reason, pause_class, file_escalation=False)
    await _resolve_stranded(rig.harness, rig.queue)


async def _restart_second_process(rig: _DigestRig, cost_store: CostStore) -> _DigestRig:
    restarted, _, event_store = _make_harness_with_mocks(rig.tmp_path)
    restarted._run_store = RunStore(_runs_db(rig.tmp_path))
    restarted.config = OrchestratorConfig(
        project_root=rig.tmp_path,
        digest_every_n_escalations=1,
        digest_ewa_threshold=1e9,
    )
    restarted.cost_store = cost_store
    restarted.scheduler.get_statuses = _statuses_read({'5260': 'pending'})
    assert [row['task_id'] for row in _repend_rows(event_store)] == ['5260']
    await restarted._load_persisted_scheduler_pause()
    return _DigestRig(restarted, rig.queue, rig.tmp_path)


@pytest.mark.asyncio
class TestPauseEndedByRestart:
    async def test_declined_restore_does_not_carry_repends_into_the_next_pause(
        self, digest_rig: _DigestRig, cost_store: CostStore,
    ) -> None:
        await _pause_and_repend_in_first_process(digest_rig, PAUSE_REASON, PauseClass.EWA_RATE)

        restarted = await _restart_second_process(digest_rig, cost_store)
        assert restarted.harness.scheduler.is_paused is False
        await restarted.harness.pause_scheduler(
            'park_stop_trip: later', PauseClass.PARK_STOP, file_escalation=False,
        )
        section = _paused_waiting(await _write_digest(restarted))

        assert 'Paused with 0 re-pended tasks waiting' in section
        assert 'task 5260' not in section

    async def test_reasserted_restore_keeps_the_waiting_task(
        self, digest_rig: _DigestRig, cost_store: CostStore,
    ) -> None:
        await _pause_and_repend_in_first_process(
            digest_rig, 'park_stop_trip: test', PauseClass.PARK_STOP,
        )

        restarted = await _restart_second_process(digest_rig, cost_store)
        assert restarted.harness.scheduler.is_paused is True
        section = _paused_waiting(await _write_digest(restarted))

        assert 'Paused with 1 re-pended tasks waiting' in section
        assert sum('task 5260' in line for line in section.splitlines()) == 1
