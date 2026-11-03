"""``merge_relane`` moves an already-enqueued merge request between lanes (task 5246).

A request's lane is decided once, at enqueue, so an operator expediting a fix
that is already queued needs a lever on the queue entry itself. These tests
drive the MCP tool against a real (not running) merge worker whose queue holds
the requests, and observe the result through the tool's answer and
``get_merge_queue``. The worker-side re-bucketing and pick order are pinned in
``orchestrator/tests/test_merge_lane_relane.py``.
"""
from __future__ import annotations

import asyncio
import types
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _merge_tool_calls import call_get_merge_queue

from escalation.queue import EscalationQueue
from escalation.server import create_server

worker_mod = pytest.importorskip('orchestrator.merge_lane.worker')
config_mod = pytest.importorskip('orchestrator.config')

pytestmark = pytest.mark.asyncio


def _request(tmp_path: Path, task_id: str) -> Any:
    return worker_mod.MergeRequest(
        task_id=task_id,
        branch=worker_mod.QueuedBranch.parse(task_id, 'task/'),
        worktree=tmp_path,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config_mod.OrchestratorConfig(project_root=tmp_path / 'repo'),
        result=asyncio.get_running_loop().create_future(),
    )


def _train(tmp_path: Path, train_id: str, member_task_ids: list[str]) -> Any:
    tip = _request(tmp_path, member_task_ids[-1])
    return worker_mod.GroupMergeRequest(
        task_id=tip.task_id,
        branch=tip.branch,
        worktree=tip.worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=tip.config,
        result=asyncio.get_running_loop().create_future(),
        train_id=train_id,
        member_task_ids=member_task_ids,
        tip_branch=tip.branch,
        tip_task_id=tip.task_id,
        status_check=AsyncMock(),
        mark_member_done=AsyncMock(),
    )


def _scheduler(written: dict[str, bool] | None = None) -> types.SimpleNamespace:
    """A scheduler whose ``update_task`` answers True, or *written*[task_id]."""
    answers = written or {}
    return types.SimpleNamespace(update_task=AsyncMock(
        side_effect=lambda tid, metadata, **_: answers.get(tid, True),
    ))


def _server(tmp_path: Path, queued: list[Any], *, scheduler: Any = None):
    mq: asyncio.Queue[Any] = asyncio.Queue()
    for req in queued:
        mq.put_nowait(req)
    git_ops: Any = types.SimpleNamespace()
    worker = worker_mod.SpeculativeMergeWorker(git_ops, mq)
    harness = types.SimpleNamespace(_merge_worker=worker)
    if scheduler is not None:
        harness.scheduler = scheduler
    return create_server(
        EscalationQueue(tmp_path / 'esc'),
        merge_queue=mq,
        orch_config=config_mod.OrchestratorConfig(project_root=tmp_path / 'repo'),
        harness=harness,
    )


async def _relane(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('merge_relane')
    return await tool.fn(**kwargs)


async def test_relaned_request_is_listed_first_in_the_high_lane(tmp_path: Path) -> None:
    older, urgent = _request(tmp_path, 'older'), _request(tmp_path, 'urgent')
    scheduler = _scheduler()
    server = _server(tmp_path, [older, urgent], scheduler=scheduler)

    answer = await _relane(server, lane='high', request_id=urgent.request_id)

    assert answer == {
        'relaned': True, 'lane': 'high', 'request_id': urgent.request_id,
        'from_lane': 'normal', 'via_train': None, 'task_ids': ('urgent',),
        'metadata_writes': ({'task_id': 'urgent', 'written': True},),
        'refusal': None, 'state': None,
    }
    scheduler.update_task.assert_awaited_once_with(
        'urgent', {'merge_lane': 'high'}, metadata_mode='merge',
    )
    entries = (await call_get_merge_queue(server))['entries']
    assert [(e['task_id'], e['lane']) for e in entries] == [('urgent', 'high'), ('older', 'normal')]


async def test_a_coalesced_member_is_reached_by_task_id_through_its_train(
    tmp_path: Path,
) -> None:
    train = _train(tmp_path, 'coalesce-m-b-1234', ['m-a', 'm-b'])
    scheduler = _scheduler()
    server = _server(tmp_path, [train], scheduler=scheduler)

    answer = await _relane(server, lane='high', task_id='m-a')

    assert (answer['relaned'], answer['request_id'], answer['via_train']) == (
        True, train.request_id, 'coalesce-m-b-1234',
    )
    assert answer['metadata_writes'] == (
        {'task_id': 'm-a', 'written': True}, {'task_id': 'm-b', 'written': True},
    )
    assert sorted(c.args for c in scheduler.update_task.await_args_list) == [
        ('m-a', {'merge_lane': 'high'}), ('m-b', {'merge_lane': 'high'}),
    ]


async def test_a_failed_metadata_write_is_reported_per_task(tmp_path: Path) -> None:
    train = _train(tmp_path, 'coalesce-f-b-1234', ['f-a', 'f-b'])
    server = _server(tmp_path, [train], scheduler=_scheduler({'f-b': False}))

    answer = await _relane(server, lane='high', task_id='f-a')

    assert answer['relaned'] is True
    assert answer['metadata_writes'] == (
        {'task_id': 'f-a', 'written': True}, {'task_id': 'f-b', 'written': False},
    )


async def test_without_a_scheduler_every_metadata_write_is_reported_failed(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path, [_request(tmp_path, 'solo')])

    answer = await _relane(server, lane='high', task_id='solo')

    assert answer['metadata_writes'] == ({'task_id': 'solo', 'written': False},)


async def test_demoting_back_to_normal_is_allowed(tmp_path: Path) -> None:
    req = _request(tmp_path, 'demote')
    server = _server(tmp_path, [req])
    await _relane(server, lane='high', task_id='demote')

    answer = await _relane(server, lane='normal', task_id='demote')

    assert (answer['relaned'], answer['from_lane'], answer['lane']) == (True, 'high', 'normal')


async def test_an_unknown_request_is_refused_and_nothing_moves(tmp_path: Path) -> None:
    scheduler = _scheduler()
    server = _server(tmp_path, [_request(tmp_path, 'kept')], scheduler=scheduler)

    answer = await _relane(server, lane='high', request_id='mr-nothere')

    assert (answer['relaned'], answer['refusal']) == (False, 'not_found')
    assert 'task_id' in answer['hint']
    assert answer['metadata_writes'] == ()
    scheduler.update_task.assert_not_awaited()
    entries = (await call_get_merge_queue(server))['entries']
    assert [e['lane'] for e in entries] == ['normal']


@pytest.mark.parametrize(
    ('kwargs', 'code'),
    [
        ({'lane': 'higgh', 'task_id': 'x'}, 'invalid_lane'),
        ({'lane': 'high'}, 'invalid_target'),
        ({'lane': 'high', 'task_id': 'x', 'request_id': 'mr-x'}, 'invalid_target'),
    ],
)
async def test_bad_input_is_rejected_loudly(
    tmp_path: Path, kwargs: dict[str, Any], code: str,
) -> None:
    server = _server(tmp_path, [])

    answer = await _relane(server, **kwargs)

    assert answer['code'] == code, answer


async def test_without_a_merge_worker_it_says_so(tmp_path: Path) -> None:
    server = create_server(EscalationQueue(tmp_path / 'esc'))

    assert (await _relane(server, lane='high', task_id='x'))['code'] == 'no_merge_worker'
    assert (await _relane(server, lane='higgh', task_id='x'))['code'] == 'invalid_lane'
