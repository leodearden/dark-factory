"""A restart resolved with ``recut=True`` issues a re-cut ruling before the re-pend (task 5516).

The harness's restart teardown writes ``metadata.recut_ruling`` BEFORE it
writes ``'pending'``, so any dispatch that observes ``'pending'`` also
observes the ruling, and the task-3154 dispatch gate
(``orchestrator/src/orchestrator/recut_ruling.py``) re-cuts the task fresh off
main. If the ruling cannot be persisted the teardown fails closed: the task is
held ``'blocked'``, none of the restart-only clears run, and one
``RESTART_RECUT_UNRECORDED`` infra_issue L1 is filed. A ruling the dying
workflow's metadata write drops in the kill window is re-asserted unchanged.

Driven end to end through the real ``resolve_issue`` tool, a real
``EscalationQueue`` and the harness's resolve callback; observed through a
stateful task-store double and the queue.
"""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from _orch_helpers import wire_scheduler_liveness_mock
from _task_store_doubles import TaskStoreDouble
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server
from fastmcp.tools.function_tool import FunctionTool

from orchestrator.harness import Harness
from orchestrator.merge_lane.gates import PLAN_FILES_NOT_TOUCHED_REASON_PREFIX
from orchestrator.recut_ruling import RECUT_RULING_KEY, ruling_to_enforce

pytestmark = pytest.mark.asyncio


@pytest.fixture
def store() -> TaskStoreDouble:
    return TaskStoreDouble()


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


@pytest.fixture
def harness(mock_orch_config, store: TaskStoreDouble, queue: EscalationQueue) -> Harness:
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)
    h.scheduler = MagicMock()
    wire_scheduler_liveness_mock(h.scheduler)
    store.wire_into(h.scheduler)
    h.is_workflow_active = lambda task_id: False  # type: ignore[method-assign]
    h._escalation_queue = queue
    queue.set_resolve_callback(h._on_escalation_resolved)
    return h


@pytest.fixture
def server(queue: EscalationQueue, harness: Harness):
    return create_server(queue, startup_sweep=False)


def _file_l1(queue: EscalationQueue, task_id: str, *, level: int = 1, **kw: Any) -> Escalation:
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role='orchestrator',
        severity='blocking',
        category='design_concern',
        summary=f'{PLAN_FILES_NOT_TOUCHED_REASON_PREFIX}: a.py, b.py.',
        level=level,
        **kw,
    )
    queue.submit(esc)
    return esc


async def _resolve(server, harness: Harness, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('resolve_issue')
    assert isinstance(tool, FunctionTool)
    result = tool.fn(resolution='re-plan on current main', **kwargs)
    while harness._background_tasks:
        await asyncio.gather(*list(harness._background_tasks))
    return result


class TestRestartRecutIssuesARuling:
    async def test_ruling_is_written_before_the_re_pend(self, server, harness, store, queue):
        store.add('42')
        l1 = _file_l1(queue, '42')

        result = await _resolve(
            server, harness, escalation_id=l1.id, action='restart', recut=True,
        )

        assert 'error' not in result, result
        ruling = ruling_to_enforce(store.tasks['42'])
        assert ruling is not None and ruling.ruling_id == l1.id
        [ruling_write] = store.writes_of('42', RECUT_RULING_KEY)
        assert ruling_write < store.index_of(lambda c: c == ('set_task_status', '42', 'pending'))
        assert store.status('42') == 'pending'

    async def test_plain_restart_issues_no_ruling(self, server, harness, store, queue):
        store.add('42')
        l1 = _file_l1(queue, '42')

        await _resolve(server, harness, escalation_id=l1.id, action='restart')

        assert RECUT_RULING_KEY not in store.metadata('42')
        assert store.writes_of('42', RECUT_RULING_KEY) == []
        assert store.status('42') == 'pending'

    async def test_cascade_member_inherits_the_l2_ruling(self, server, harness, store, queue):
        store.add('42')
        store.add('43')
        member = _file_l1(queue, '43')
        l2 = _file_l1(queue, '42', level=2, members=[member.id])

        await _resolve(server, harness, escalation_id=l2.id, action='restart', recut=True)

        archived = queue.get(l2.id)
        assert archived is not None and archived.recut is True
        for task_id in ('42', '43'):
            ruling = ruling_to_enforce(store.tasks[task_id])
            assert ruling is not None and ruling.ruling_id == l2.id, task_id
            assert store.status(task_id) == 'pending'

    async def test_unrecorded_ruling_holds_the_task_blocked_and_files_an_l1(
        self, server, harness, store, queue,
    ):
        store.add('42')
        store.refused_metadata_keys.add(RECUT_RULING_KEY)
        l1 = _file_l1(queue, '42')

        await _resolve(server, harness, escalation_id=l1.id, action='restart', recut=True)

        assert store.status('42') == 'blocked'
        assert ('set_task_status', '42', 'pending') not in store.calls
        filed = [
            e for e in queue.get_by_task('42', status='pending', level=1)
            if e.category == 'infra_issue'
        ]
        assert len(filed) == 1, filed
        assert filed[0].summary.startswith('RESTART_RECUT_UNRECORDED')
        assert l1.id in filed[0].detail

    async def test_terminal_task_is_left_alone(self, server, harness, store, queue):
        store.add('42', status='done')
        l1 = _file_l1(queue, '42')

        await _resolve(server, harness, escalation_id=l1.id, action='restart', recut=True)

        assert store.status('42') == 'done'
        assert RECUT_RULING_KEY not in store.metadata('42')
        assert store.calls == []

    async def test_ruling_survives_the_merge_retry_clear(self, server, harness, store, queue):
        store.add(
            '42', metadata={'merge_retry_pending': {'attempt': 1}, 'files': ['a.py', 'b.py']},
        )
        l1 = _file_l1(queue, '42')

        await _resolve(server, harness, escalation_id=l1.id, action='restart', recut=True)

        metadata = store.metadata('42')
        assert 'merge_retry_pending' not in metadata
        assert RECUT_RULING_KEY in metadata
        assert metadata['files'] == ['a.py', 'b.py']
        assert store.status('42') == 'pending'


def _kill_lands_a_stale_replace_write(
    harness: Harness, store: TaskStoreDouble, task_id: str, *,
    then_refuse: frozenset[str] = frozenset(),
) -> None:
    """A live workflow whose dying metadata write replaces the record it read before the ruling."""
    stale = copy.deepcopy(store.metadata(task_id))
    active = {task_id}
    harness.is_workflow_active = lambda tid: tid in active  # type: ignore[method-assign]

    def _cancel(tid: str) -> None:
        store.tasks[tid]['metadata'] = copy.deepcopy(stale)
        store.refused_metadata_keys |= then_refuse
        active.discard(tid)

    harness.cancel_workflow = _cancel  # type: ignore[method-assign]


class TestRulingDroppedDuringTheKillWindow:
    async def test_the_same_ruling_is_re_asserted(self, server, harness, store, queue):
        store.add('42', status='in-progress', metadata={'files': ['a.py']})
        _kill_lands_a_stale_replace_write(harness, store, '42')
        l1 = _file_l1(queue, '42')

        await _resolve(server, harness, escalation_id=l1.id, action='restart', recut=True)

        first, again = store.writes_of('42', RECUT_RULING_KEY)
        assert store.calls[again][2] == store.calls[first][2]
        assert store.metadata('42')[RECUT_RULING_KEY] == store.calls[first][2][RECUT_RULING_KEY]
        ruling = ruling_to_enforce(store.tasks['42'])
        assert ruling is not None and ruling.ruling_id == l1.id
        assert store.status('42') == 'pending'

    async def test_a_failed_re_assert_files_an_l1(self, server, harness, store, queue):
        store.add('42', status='in-progress')
        _kill_lands_a_stale_replace_write(
            harness, store, '42', then_refuse=frozenset({RECUT_RULING_KEY}),
        )
        l1 = _file_l1(queue, '42')

        await _resolve(server, harness, escalation_id=l1.id, action='restart', recut=True)

        assert RECUT_RULING_KEY not in store.metadata('42')
        filed = [
            e for e in queue.get_by_task('42', status='pending', level=1)
            if e.category == 'infra_issue'
        ]
        assert len(filed) == 1, filed
        assert filed[0].summary.startswith('RESTART_RECUT_UNRECORDED')
        assert l1.id in filed[0].detail


class TestUnrecordedRulingUndoesNothing:
    async def test_restart_only_clears_do_not_run(self, server, harness, store, queue):
        held = {
            'task_kind': 'deterministic',
            'gate_escalated_at': '2026-10-30T09:00:00+00:00',
            'before_done_ran_at': '2026-10-30T09:05:00+00:00',
            'merge_retry_pending': {'attempt': 1},
        }
        store.add('42', metadata=held)
        store.refused_metadata_keys.add(RECUT_RULING_KEY)
        l1 = _file_l1(queue, '42')

        await _resolve(server, harness, escalation_id=l1.id, action='restart', recut=True)

        assert store.status('42') == 'blocked'
        assert store.metadata('42') == held
