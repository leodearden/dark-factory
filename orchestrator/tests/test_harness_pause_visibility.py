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
from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from _orch_helpers import claimant_row, wire_scheduler_liveness_mock
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.cost_store import CostStore
from test_harness_park_stop import _make_harness_with_mocks, _query_events

import orchestrator.harness as harness_mod
from orchestrator.config import OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness


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
    return EscalationQueue(Path(tmp_path) / harness.config.escalation.queue_dir)


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
