"""Harness-level tests for scheduler-pause digest ergonomics (task 5311).

Covers the EWA-trip storm annotation on the pause reason, the paused re-pend
marker on a resolution, and the digest's '## Paused, waiting' section.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest
import pytest_asyncio
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from shared.cost_store import CostStore
from test_harness_park_stop import _make_harness_with_mocks, _query_events

from orchestrator.config import OrchestratorConfig
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
