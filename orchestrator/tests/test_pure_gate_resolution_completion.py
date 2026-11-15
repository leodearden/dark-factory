"""A resumed pure gate closes at resolution time, through the production resume route (task 3638).

Every test seeds a gate and dispatches it once, leaving it blocked behind one
pending born-at-L2 record with both gate stamps. It then resolves a record via
``Harness._on_escalation_resolved`` and does NOT dispatch again: whatever the
store says afterwards is what the resolution alone delivered.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from test_deterministic_task import _build_harness, _dispatch, _gate_task, _StoreScheduler

from orchestrator.deterministic_runner import (
    CURATOR_ADJUDICATION_MISSING_CATEGORY,
    DETERMINISTIC_AGENT_ROLE,
    MILESTONE_GATE_CATEGORY,
)
from orchestrator.harness import Harness
from orchestrator.scheduler import ProvenanceValidationRejection

pytestmark = pytest.mark.asyncio


class _RejectingDoneStoreScheduler(_StoreScheduler):
    """Refuses every ``done`` write the way the real provenance validator can."""

    async def set_task_status(
        self, task_id: str, status: str, *, done_provenance: dict | None = None,
    ) -> None:
        if status == 'done':
            raise ProvenanceValidationRejection(
                task_id=str(task_id),
                error_code='done_provenance_invalid',
                raw='done_provenance rejected by the test store',
            )
        await super().set_task_status(task_id, status, done_provenance=done_provenance)


@pytest.fixture
def det_harness(tmp_path: Path, mock_orch_config) -> Harness:
    return _build_harness(mock_orch_config, tmp_path / 'queue', _StoreScheduler())


def _store(h: Harness) -> _StoreScheduler:
    store = h.scheduler
    assert isinstance(store, _StoreScheduler)
    return store


def _queue(h: Harness) -> EscalationQueue:
    queue = h._escalation_queue
    assert queue is not None
    return queue


async def _blocked_gate(h: Harness, task_id: str, **metadata) -> Escalation:
    """Seed a pure gate, dispatch it once, and return its pending gate record."""
    _store(h).seed(_gate_task(task_id=task_id, **metadata))
    await _dispatch(h, task_id)
    pending = _queue(h).get_by_task(task_id, status='pending')
    assert len(pending) == 1, pending
    assert await _store(h).get_status(task_id) == 'blocked'
    assert _store(h)._tasks[task_id]['metadata'].get('gate_escalated_at')
    return pending[0]


async def _resolve(h: Harness, esc_id: str, action: str = 'resume') -> Escalation:
    esc = _queue(h).resolve(esc_id, 'proceed', resolution_action=action)
    assert esc is not None
    h._on_escalation_resolved(esc)
    await asyncio.gather(*list(h._background_tasks))
    return esc


def _pending(h: Harness, task_id: str) -> list[Escalation]:
    return _queue(h).get_by_task(task_id, status='pending')


async def test_resume_closes_a_dependency_free_gate_without_a_dispatch(
    det_harness: Harness,
) -> None:
    record = await _blocked_gate(det_harness, 'gate-2864')

    await _resolve(det_harness, record.id)

    store = _store(det_harness)
    assert await store.get_status('gate-2864') == 'done'
    provenance = store._tasks['gate-2864']['metadata']['done_provenance']
    assert provenance['kind'] == 'deterministic-gate'
    assert provenance['note'] == 'pure gate resolved'
    assert provenance['escalation_id'] == record.id
    history = store._status_history['gate-2864']
    assert 'pending' not in history[history.index('blocked'):], history
    assert _pending(det_harness, 'gate-2864') == []


async def test_resume_after_an_unsatisfied_hold_re_pends(det_harness: Harness) -> None:
    store = _store(det_harness)
    store.seed({'id': 'hold-b', 'title': 'hold', 'metadata': {}, 'status': 'pending'})
    record = await _blocked_gate(det_harness, 'gate-b')
    store._tasks['gate-b']['dependencies'] = [{'id': 'hold-b'}]

    await _resolve(det_harness, record.id)

    assert await store.get_status('gate-b') == 'pending'
    assert _pending(det_harness, 'gate-b') == []


async def test_resume_after_a_hold_that_already_landed_re_files_inline(
    det_harness: Harness,
) -> None:
    store = _store(det_harness)
    store.seed({'id': 'hold-c', 'title': 'hold', 'metadata': {}, 'status': 'done'})
    record = await _blocked_gate(det_harness, 'gate-c')
    store._tasks['gate-c']['dependencies'] = [{'id': 'hold-c'}]

    await _resolve(det_harness, record.id)

    assert await store.get_status('gate-c') == 'blocked'
    refiled = _pending(det_harness, 'gate-c')
    assert [esc.category for esc in refiled] == [MILESTONE_GATE_CATEGORY]
    assert refiled[0].id != record.id


async def test_resume_with_an_unfired_milestone_re_pends(det_harness: Harness) -> None:
    store = _store(det_harness)
    record = await _blocked_gate(det_harness, 'gate-d')
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    store._tasks['gate-d']['metadata']['milestone'] = {'mode': 'dated', 'at': tomorrow}

    await _resolve(det_harness, record.id)

    assert await store.get_status('gate-d') == 'pending'
    assert _pending(det_harness, 'gate-d') == []


async def test_resume_with_an_external_dependency_re_pends(det_harness: Harness) -> None:
    record = await _blocked_gate(det_harness, 'gate-e', external_deps=['other_project:1'])

    await _resolve(det_harness, record.id)

    assert await _store(det_harness).get_status('gate-e') == 'pending'


async def test_resume_of_an_unadjudicated_curator_gate_asks_again(
    det_harness: Harness,
) -> None:
    record = await _blocked_gate(det_harness, 'gate-f', human_curator_gate=True)

    await _resolve(det_harness, record.id)

    assert await _store(det_harness).get_status('gate-f') == 'blocked'
    assert [esc.category for esc in _pending(det_harness, 'gate-f')] == [
        CURATOR_ADJUDICATION_MISSING_CATEGORY,
    ]


async def test_resuming_another_record_leaves_the_held_gate_blocked(
    det_harness: Harness,
) -> None:
    gate_record = await _blocked_gate(det_harness, 'gate-g')
    queue = _queue(det_harness)
    other = Escalation(
        id=queue.make_id('gate-g'),
        task_id='gate-g',
        agent_role='workflow',
        severity='blocking',
        category='infra_issue',
        summary='an unrelated record on the gate task',
        level=1,
    )
    queue.submit(other)

    await _resolve(det_harness, other.id)

    assert await _store(det_harness).get_status('gate-g') == 'blocked'
    assert [esc.id for esc in _pending(det_harness, 'gate-g')] == [gate_record.id]
    assert len(queue.get_by_task('gate-g')) == 2


async def test_a_failed_inline_close_falls_back_to_the_re_pend(
    tmp_path: Path, mock_orch_config,
) -> None:
    h = _build_harness(mock_orch_config, tmp_path / 'queue', _RejectingDoneStoreScheduler())
    record = await _blocked_gate(h, 'gate-h')

    await _resolve(h, record.id)

    assert await _store(h).get_status('gate-h') == 'pending'


async def test_a_predicate_gate_keeps_the_re_pend(det_harness: Harness) -> None:
    store = _store(det_harness)
    task = _gate_task(
        task_id='gate-i',
        before_done={
            'kind': 'predicate',
            'script': 'scripts/does-not-exist-3638.sh',
            'timeout_secs': 5,
        },
        gate_escalated_at='2026-08-02T19:07:41.617308+00:00',
    )
    task['status'] = 'blocked'
    store.seed(task)
    queue = _queue(det_harness)
    record = Escalation(
        id=queue.make_id('gate-i'),
        task_id='gate-i',
        agent_role=DETERMINISTIC_AGENT_ROLE,
        severity='blocking',
        category=MILESTONE_GATE_CATEGORY,
        summary='a predicate gate record',
        level=2,
    )
    queue.submit(record)

    await _resolve(det_harness, record.id)

    assert await store.get_status('gate-i') == 'pending'
