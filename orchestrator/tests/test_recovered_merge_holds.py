"""A task whose merge request is recovered at startup holds its files again (task 10093).

These tests drive the real step 1c0a, ``Harness._recover_pending_merges``, over a
real Scheduler whose lock table is real and whose task-store reads are stubbed.

This file imports no ``orchestrator.merge_lane`` or ``orchestrator.git_ops``
module: it reads harness privates, which the merge-lane ratchet
(``scripts/merge_lane_metrics.py``) would count against any lane-importing file.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import AsyncMock

import pytest
from _landed_rows import record_landing
from _orch_helpers import HermeticMcpSession
from _park_test_helpers import make_task
from _startup_harness import (
    MERGE_WORKER_SHUTDOWN_OUTCOME,
    drain_background_tasks,
    journal_merge_request,
    merge_lane_harness_factory,
    recover_requests,
    recovered_request,
    start_real_merge_worker,
)
from escalation.queue import EscalationQueue

from orchestrator.config import OrchestratorConfig, PsiAdmissionConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.module_charter import derive_modules
from orchestrator.pin_reservation import PinOrder
from orchestrator.recovered_merge_holds import LockRestoreReason
from orchestrator.scheduler import Scheduler

_TID = '40'
# Not .md: this scene's config gives root-level markdown unlimited holders.
_FILES = ['a.py', 'b.py']


class _Scene(NamedTuple):
    harness: Harness
    scheduler: Scheduler
    store: EventStore


def _scheduler(config: OrchestratorConfig, store: EventStore, *, status: str) -> Scheduler:
    """A real Scheduler with a real lock table; every task reads as *status*."""
    config.psi_admission = PsiAdmissionConfig(enabled=False)
    scheduler = Scheduler(config, event_store=store, mcp_session=HermeticMcpSession())
    scheduler.get_statuses = AsyncMock(
        side_effect=lambda ids=None: ({tid: status for tid in ids or []}, None),
    )
    return scheduler


def _scene(
    tmp_path: Path, *, status: str = 'in-progress', real_steps: frozenset[str] = frozenset(),
) -> _Scene:
    """Step 1c0a's harness over a real Scheduler built from the harness's own config."""
    h = merge_lane_harness_factory(tmp_path, real_steps=real_steps)()
    store = EventStore(tmp_path / 'runs.db', 'run-test')
    scheduler = _scheduler(h.config, store, status=status)
    h.scheduler = scheduler
    h.event_store = store
    return _Scene(h, scheduler, store)


async def _scene_with_real_merge_worker(
    tmp_path: Path, *, real_steps: frozenset[str] = frozenset(),
) -> _Scene:
    """Step 1c0a's harness over a real Scheduler, holding a real merge worker."""
    scene = _scene(tmp_path, real_steps=real_steps)
    await start_real_merge_worker(scene.harness, scene.scheduler)
    return scene


def _keys(h: Harness, files: list[str], task_id: str = _TID) -> tuple[str, ...]:
    return tuple(sorted(derive_modules(files, h.config.lock_depth, task_id=task_id)))


def _event_data(store: EventStore, name: str, task_id: str) -> list[dict]:
    return [e['data'] for e in store.fetch_events_by_type(name) if e['task_id'] == task_id]


def _held(scheduler: Scheduler) -> Mapping[str, tuple[str, ...]]:
    return scheduler.get_state_snapshot().held_by_task


async def _dispatch_all(scheduler: Scheduler) -> list[str]:
    dispatched: list[str] = []
    for _ in range(3):
        assignment = await scheduler.acquire_next()
        if assignment is None:
            break
        dispatched.append(assignment.task_id)
    return dispatched


@pytest.mark.asyncio
class TestRestoreAtStartup:
    async def test_in_progress_task_holds_its_journalled_files_after_recovery(
        self, tmp_path: Path,
    ) -> None:
        h, scheduler, store = _scene(tmp_path)
        await journal_merge_request(h, _TID, task_files=_FILES)

        await h._recover_pending_merges()

        keys = _keys(h, _FILES)
        assert _held(scheduler)[_TID] == keys
        [acquired] = _event_data(store, 'lock_acquired', _TID)
        assert tuple(sorted(acquired['modules'])) == keys
        assert acquired['reason'] == LockRestoreReason.RECOVERED_MERGE

        scheduler.get_tasks = AsyncMock(return_value=[
            make_task('50', 'high', ['a.py']),
            make_task('51', 'high', ['c.py']),
        ])
        scheduler.finish_startup()

        assert await _dispatch_all(scheduler) == ['51']
        assert '50' not in _held(scheduler)

    @pytest.mark.parametrize('status', ['deferred', 'blocked', 'cancelled', 'done'])
    async def test_a_task_not_in_progress_at_startup_gains_no_hold(
        self, tmp_path: Path, status: str,
    ) -> None:
        h, scheduler, store = _scene(tmp_path, status=status)
        await journal_merge_request(h, _TID, task_files=_FILES)

        await h._recover_pending_merges()

        assert _TID not in _held(scheduler)
        assert _event_data(store, 'lock_acquired', _TID) == []


def _request() -> SimpleNamespace:
    return recovered_request(_TID, task_files=list(_FILES))


def _stub_routing_writes(scheduler: Scheduler) -> AsyncMock:
    """Stub the reads and the block write settling needs; the block write's mock."""
    scheduler.get_status = AsyncMock(return_value='in-progress')
    block_write = AsyncMock()
    scheduler.clear_claim_then_set_status = block_write
    return block_write


_VERIFY_RED = SimpleNamespace(
    status='blocked', reason='post-merge verify red',
    block_category=SimpleNamespace(value='merge_error'),
    failure_category=None, failure_cause_hint=None,
)


@pytest.mark.asyncio
class TestRestoreToleratesWhatItCannotTake:
    async def test_a_key_already_taken_is_left_unheld_and_named(
        self, tmp_path: Path,
    ) -> None:
        h, scheduler, store = _scene(tmp_path)
        files = {'40': ['shared.py', 'a.py'], '41': ['shared.py', 'b.py']}
        for tid, task_files in files.items():
            await journal_merge_request(h, tid, task_files=task_files)

        report = await h._recover_pending_merges()

        request_ids = {r.task_id: r.request_id for r in report['requests']}
        assert sorted(request_ids) == ['40', '41']
        [shared] = _keys(h, ['shared.py'])
        held = _held(scheduler)
        [first] = [tid for tid in files if shared in held.get(tid, ())]
        [other] = [tid for tid in files if tid != first]
        own = {tid: _keys(h, task_files[1:], tid) for tid, task_files in files.items()}
        assert held[first] == tuple(sorted((shared, *own[first])))
        assert held[other] == own[other]
        [incomplete] = store.fetch_events_by_type('recovered_merge_hold_incomplete')
        assert incomplete['task_id'] == other
        assert incomplete['data'] == {
            'request_id': request_ids[other],
            'restored': list(own[other]),
            'unheld': [{'module': shared, 'owner': first, 'kind': 'held'}],
        }

    async def test_a_request_finalized_before_the_restore_gains_no_hold(
        self, tmp_path: Path,
    ) -> None:
        h, scheduler, store = _scene(tmp_path)
        _stub_routing_writes(scheduler)
        request = _request()

        def finalize_then_answer(ids=None):
            request.result.set_result(_VERIFY_RED)
            return {_TID: 'in-progress'}, None

        scheduler.get_statuses = AsyncMock(side_effect=finalize_then_answer)

        await recover_requests(h, request)

        assert _TID not in _held(scheduler)
        assert _event_data(store, 'lock_acquired', _TID) == []
        await drain_background_tasks(h)

    @pytest.mark.parametrize('failure', ['error_result', 'raises'])
    async def test_an_unreadable_task_store_restores_nothing_and_does_not_raise(
        self, tmp_path: Path, failure: str,
    ) -> None:
        h, scheduler, _store = _scene(tmp_path)
        await journal_merge_request(h, _TID, task_files=_FILES)
        scheduler.get_statuses = (
            AsyncMock(return_value=({}, RuntimeError('store down')))
            if failure == 'error_result'
            else AsyncMock(side_effect=RuntimeError('store down'))
        )

        report = await h._recover_pending_merges()

        assert [r.task_id for r in report['requests']] == [_TID]
        assert _held(scheduler) == {}


def _recording_hold(scheduler: Scheduler, seen: list) -> AsyncMock:
    """An AsyncMock that records the task's hold at the moment it is awaited."""
    return AsyncMock(side_effect=lambda *_a, **_k: seen.append(_held(scheduler).get(_TID)))


@pytest.mark.asyncio
class TestRestoredHoldEndsWithItsRequest:
    async def test_a_landed_request_releases_its_files_after_the_task_is_marked_done(
        self, tmp_path: Path,
    ) -> None:
        h, scheduler, store = await _scene_with_real_merge_worker(tmp_path)
        assert h._merge_worker is not None
        outbox = h._merge_worker._landed_outbox
        assert outbox is not None
        main_sha = await h.git_ops.get_main_sha()
        record_landing(outbox, _TID, advanced_sha=main_sha)
        scheduler.get_status = AsyncMock(return_value='in-progress')
        scheduler.get_task = AsyncMock(return_value={'metadata': {}})
        held_at_done: list = []
        scheduler.mark_done = _recording_hold(scheduler, held_at_done)
        request = _request()
        await recover_requests(h, request)
        keys = _keys(h, _FILES)

        request.result.set_result(SimpleNamespace(status='done', merge_sha=main_sha, reason=''))
        await drain_background_tasks(h)

        scheduler.mark_done.assert_awaited_once_with(_TID, kind='merged', sha=main_sha)
        assert held_at_done == [keys]
        assert _TID not in _held(scheduler)
        [released] = _event_data(store, 'lock_released', _TID)
        assert tuple(sorted(released['modules'])) == keys

    async def test_a_failed_request_releases_its_files_after_blocking_the_task(
        self, tmp_path: Path,
    ) -> None:
        h, scheduler, _store = _scene(tmp_path)
        h._escalation_queue = EscalationQueue(tmp_path / 'esc')
        scheduler.get_status = AsyncMock(return_value='in-progress')
        held_at_block: list = []
        scheduler.clear_claim_then_set_status = _recording_hold(scheduler, held_at_block)
        request = _request()
        await recover_requests(h, request)

        request.result.set_exception(RuntimeError('verify host lost'))
        await drain_background_tasks(h)

        scheduler.clear_claim_then_set_status.assert_awaited_once_with(_TID, 'blocked')
        assert held_at_block == [_keys(h, _FILES)]
        assert _TID not in _held(scheduler)
        [esc] = h._escalation_queue.get_by_task(_TID, status='pending', level=1)
        assert esc.category == 'merge_error'

    async def test_an_operator_cancel_releases_the_files(self, tmp_path: Path) -> None:
        h, scheduler, _store = _scene(tmp_path)
        block_write = _stub_routing_writes(scheduler)
        request = _request()
        await recover_requests(h, request)

        request.result.cancel()
        await drain_background_tasks(h)

        assert _TID not in _held(scheduler)
        block_write.assert_not_awaited()

    async def test_a_worker_shutdown_outcome_keeps_the_hold(self, tmp_path: Path) -> None:
        h, scheduler, store = _scene(tmp_path)
        _stub_routing_writes(scheduler)
        request = _request()
        await recover_requests(h, request)

        request.result.set_result(MERGE_WORKER_SHUTDOWN_OUTCOME)
        await drain_background_tasks(h)

        assert _held(scheduler)[_TID] == _keys(h, _FILES)
        assert _event_data(store, 'lock_released', _TID) == []

    async def test_a_task_dispatched_since_the_restore_keeps_its_workflows_locks(
        self, tmp_path: Path,
    ) -> None:
        h, scheduler, _store = _scene(tmp_path)
        block_write = _stub_routing_writes(scheduler)
        request = _request()
        await recover_requests(h, request)
        scheduler.get_tasks = AsyncMock(return_value=[make_task(_TID, 'high', _FILES)])
        scheduler.finish_startup()

        assignment = await scheduler.acquire_next()

        assert assignment is not None
        assert assignment.task_id == _TID
        request.result.set_exception(RuntimeError('verify host lost'))
        await drain_background_tasks(h)
        block_write.assert_not_awaited()
        assert _TID in _held(scheduler)

    async def test_a_dispatch_supersedes_the_restored_hold_for_good(
        self, tmp_path: Path,
    ) -> None:
        h, scheduler, _store = _scene(tmp_path)
        block_write = _stub_routing_writes(scheduler)
        request = _request()
        await recover_requests(h, request)
        scheduler.get_tasks = AsyncMock(return_value=[make_task(_TID, 'high', _FILES)])
        scheduler.finish_startup()
        assignment = await scheduler.acquire_next()
        assert assignment is not None
        assert assignment.task_id == _TID
        scheduler.note_workflow_cancelled(_TID)
        scheduler.release(_TID)
        scheduler.lock_table.install_parks(_TID, ['pin/mod'], PinOrder(1))

        request.result.set_exception(RuntimeError('verify host lost'))
        await drain_background_tasks(h)

        block_write.assert_not_awaited()
        assert _TID in scheduler.get_state_snapshot().pin_reservations
