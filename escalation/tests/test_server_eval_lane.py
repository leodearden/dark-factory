"""Eval-lane containment at the escalation server.

An eval-lane filing (an eval fixture task id, or a filing from an eval worktree
— ``shared/src/shared/eval_lane.py``) is never a production signal. The server
files it already-resolved, so it never goes pending and never reaches the
orphan reaper, the auto-watcher or a human L2.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from _filing_tools import call_blocker, call_info

from escalation.queue import EscalationQueue
from escalation.server import create_server

ADV_FIXTURE_ID = 'df_task_2430_adv_plan'
SHADOW_FIXTURE_ID = 'shadow_5383_01JCELL'
EVAL_WORKTREE = '/home/leo/src/dark-factory-eval-worktrees/df_task_2339/run-ac3ab562'
CONTAINMENT_RESOLVER = 'escalation-eval-lane-containment'

_FILING: dict[str, str] = {
    'agent_role': 'implementer',
    'category': 'design_concern',
    'summary': 'refusing seeded wrong step',
}


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


def _server(
    queue: EscalationQueue,
    *,
    status: str | None = 'in-progress',
    claimant: str | None = 'run-live',
) -> tuple[object, AsyncMock, AsyncMock]:
    status_lookup = AsyncMock(return_value=status)
    claimant_lookup = AsyncMock(return_value=claimant)
    server = create_server(
        queue,
        startup_sweep=False,
        task_status_lookup=status_lookup,
        task_claimant_lookup=claimant_lookup,
    )
    return server, status_lookup, claimant_lookup


def _assert_contained(queue: EscalationQueue, result: dict, reason: str) -> None:
    assert result['status'] == 'resolved', result
    assert result['resolved_by'] == CONTAINMENT_RESOLVER
    assert queue.get_pending() == []
    record = queue.get(result['id'])
    assert record is not None
    assert record.status == 'resolved'
    assert record.resolved_by == CONTAINMENT_RESOLVER
    assert record.resolution_class == 'benign'
    assert reason in (record.resolution or '')


class TestFilingIsContained:
    @pytest.mark.asyncio
    async def test_blocker_with_fixture_id_is_filed_resolved(self, queue):
        server, _, _ = _server(queue)

        result = await call_blocker(server, task_id=ADV_FIXTURE_ID, **_FILING)

        _assert_contained(queue, result, f'fixture-task-id:{ADV_FIXTURE_ID}')
        assert result['level'] == 0
        assert result['action'] == 'terminate_cleanly'
        assert queue.get(result['id']).level == 0

    @pytest.mark.asyncio
    async def test_info_with_fixture_id_is_filed_resolved(self, queue):
        server, _, _ = _server(queue)

        result = await call_info(server, task_id=ADV_FIXTURE_ID, **_FILING)

        _assert_contained(queue, result, f'fixture-task-id:{ADV_FIXTURE_ID}')
        assert 'action' not in result

    @pytest.mark.asyncio
    async def test_live_shadow_fixture_id_is_contained(self, queue):
        server, _, _ = _server(queue)

        result = await call_blocker(server, task_id=SHADOW_FIXTURE_ID, **_FILING)

        _assert_contained(queue, result, f'fixture-task-id:{SHADOW_FIXTURE_ID}')

    @pytest.mark.asyncio
    async def test_numeric_id_filed_from_eval_worktree_is_contained(self, queue):
        server, _, _ = _server(queue)

        result = await call_blocker(
            server, task_id='2339', worktree=EVAL_WORKTREE, **_FILING
        )

        _assert_contained(queue, result, f'eval-worktree:{EVAL_WORKTREE}')


class TestContainmentCannotBeArguedAround:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('severity', ['critical', 'urgent'])
    async def test_born_at_l2_severity_is_contained_at_level_0(self, queue, severity):
        server, _, _ = _server(queue)

        result = await call_blocker(
            server, task_id=ADV_FIXTURE_ID, severity=severity, **_FILING
        )

        _assert_contained(queue, result, f'fixture-task-id:{ADV_FIXTURE_ID}')
        assert result['level'] == 0
        assert [r.level for r in queue.get_by_task(ADV_FIXTURE_ID)] == [0]

    @pytest.mark.asyncio
    async def test_terminal_state_is_the_bug_does_not_bypass(self, queue):
        server, _, _ = _server(queue)

        result = await call_blocker(
            server, task_id=ADV_FIXTURE_ID, terminal_state_is_the_bug=True, **_FILING
        )

        _assert_contained(queue, result, f'fixture-task-id:{ADV_FIXTURE_ID}')

    @pytest.mark.asyncio
    async def test_level_1_filing_leaves_no_pending_l1(self, queue):
        server, _, _ = _server(queue)

        result = await call_blocker(server, task_id=ADV_FIXTURE_ID, level=1, **_FILING)

        _assert_contained(queue, result, f'fixture-task-id:{ADV_FIXTURE_ID}')
        assert queue.get_by_task(ADV_FIXTURE_ID, status='pending', level=1) == []


class TestContainmentRunsFirst:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'task_id, worktree',
        [(ADV_FIXTURE_ID, None), (SHADOW_FIXTURE_ID, None), ('2339', EVAL_WORKTREE)],
    )
    async def test_no_task_lookup_is_awaited(self, queue, task_id, worktree):
        server, status_lookup, claimant_lookup = _server(queue)

        await call_blocker(server, task_id=task_id, worktree=worktree, **_FILING)

        status_lookup.assert_not_awaited()
        claimant_lookup.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_contained_with_no_lookups_wired(self, queue):
        server = create_server(queue, startup_sweep=False)

        result = await call_blocker(server, task_id=ADV_FIXTURE_ID, **_FILING)

        _assert_contained(queue, result, f'fixture-task-id:{ADV_FIXTURE_ID}')
        assert queue.get(result['id']).filing_claimant_run_id is None


class TestProductionFilingsStillQueue:
    @pytest.mark.asyncio
    async def test_numeric_task_in_its_own_worktree_queues(self, queue):
        server, status_lookup, _ = _server(queue)

        result = await call_blocker(
            server,
            task_id='3096',
            worktree='/home/leo/src/dark-factory/.worktrees/3096',
            **_FILING,
        )

        assert result['status'] == 'queued', result
        assert [e.id for e in queue.get_pending()] == [result['id']]
        status_lookup.assert_awaited_once_with('3096')

    @pytest.mark.asyncio
    @pytest.mark.parametrize('task_id', ['task-path-guard', '__recovery_veto_streak__5381'])
    async def test_sentinel_ids_queue(self, queue, task_id):
        server, _, _ = _server(queue)

        result = await call_blocker(server, task_id=task_id, **_FILING)

        assert result['status'] == 'queued', result
        assert [e.id for e in queue.get_pending()] == [result['id']]
