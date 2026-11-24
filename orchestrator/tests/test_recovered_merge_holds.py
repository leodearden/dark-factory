"""A task whose merge request is recovered at startup holds its files again (task 10093).

These tests drive the real step 1c0a, ``Harness._recover_pending_merges``, over a
real Scheduler whose lock table is real and whose task-store reads are stubbed.

This file imports no ``orchestrator.merge_lane`` or ``orchestrator.git_ops``
module: it reads harness privates, which the merge-lane ratchet
(``scripts/merge_lane_metrics.py``) would count against any lane-importing file.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import AsyncMock, patch

import pytest
from _orch_helpers import HermeticMcpSession
from _park_test_helpers import make_task
from _startup_harness import journal_merge_request, merge_lane_harness_factory

from orchestrator.config import PsiAdmissionConfig
from orchestrator.event_store import EventStore
from orchestrator.harness import Harness
from orchestrator.module_charter import derive_modules
from orchestrator.recovered_merge_holds import LockRestoreReason
from orchestrator.scheduler import Scheduler

_TID = '40'
# Not .md: this scene's config gives root-level markdown unlimited holders.
_FILES = ['a.py', 'b.py']


class _Scene(NamedTuple):
    harness: Harness
    scheduler: Scheduler
    store: EventStore


def _scene(tmp_path: Path, *, status: str = 'in-progress') -> _Scene:
    """Step 1c0a's harness over a real Scheduler; every task reads as *status*."""
    h = merge_lane_harness_factory(tmp_path)()
    h.config.psi_admission = PsiAdmissionConfig(enabled=False)
    store = EventStore(tmp_path / 'runs.db', 'run-test')
    scheduler = Scheduler(h.config, event_store=store, mcp_session=HermeticMcpSession())
    scheduler.get_statuses = AsyncMock(
        side_effect=lambda ids=None: ({tid: status for tid in ids or []}, None),
    )
    h.scheduler = scheduler
    h.event_store = store
    return _Scene(h, scheduler, store)


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


def _fake_request(task_id: str = _TID, files: list[str] = _FILES) -> SimpleNamespace:
    return SimpleNamespace(
        task_id=task_id, request_id=f'mr-recovered-{task_id}',
        branch=SimpleNamespace(bare_id=task_id), task_files=list(files),
        result=asyncio.get_running_loop().create_future(),
    )


async def _recover(h: Harness, *requests: SimpleNamespace) -> dict:
    report = {
        'recovered': len(requests), 'dropped': 0, 'coalesced': 0,
        'requests': list(requests), 'journal_corrupt': False,
    }
    with patch('orchestrator.harness.recover_pending_merges', AsyncMock(return_value=report)):
        return await h._recover_pending_merges()


async def _drain(h: Harness) -> None:
    await asyncio.sleep(0)
    await asyncio.gather(*h._background_tasks)


def _stub_routing_writes(scheduler: Scheduler) -> None:
    scheduler.get_status = AsyncMock(return_value='in-progress')
    scheduler.clear_claim_then_set_status = AsyncMock()


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
        request = _fake_request()

        def finalize_then_answer(ids=None):
            request.result.set_result(_VERIFY_RED)
            return {_TID: 'in-progress'}, None

        scheduler.get_statuses = AsyncMock(side_effect=finalize_then_answer)

        await _recover(h, request)

        assert _TID not in _held(scheduler)
        assert _event_data(store, 'lock_acquired', _TID) == []
        await _drain(h)

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
