"""A restart resolved with ``recut=True`` issues a re-cut ruling before the re-pend (task 5516).

The harness's restart teardown writes ``metadata.recut_ruling`` BEFORE it
writes ``'pending'``, so any dispatch that observes ``'pending'`` also
observes the ruling, and the task-3154 dispatch gate
(``orchestrator/src/orchestrator/recut_ruling.py``) re-cuts the task fresh off
main. If the ruling cannot be persisted the teardown fails closed: the task is
held ``'blocked'`` and one ``RESTART_RECUT_UNRECORDED`` infra_issue L1 is filed.

Driven end to end through the real ``resolve_issue`` tool, a real
``EscalationQueue`` and the harness's resolve callback; observed through a
stateful task-store double and the queue.
"""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import wire_scheduler_liveness_mock
from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server
from fastmcp.tools.function_tool import FunctionTool

from orchestrator.harness import Harness
from orchestrator.merge_lane.gates import PLAN_FILES_NOT_TOUCHED_REASON_PREFIX
from orchestrator.recut_ruling import RECUT_RULING_KEY, ruling_to_enforce

pytestmark = pytest.mark.asyncio


class _TaskStore:
    """Task records in one dict, with the scheduler's metadata-mode semantics."""

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[Any, ...]] = []
        self.reject_ruling_writes = False

    def add(self, task_id: str, status: str = 'blocked', **metadata: Any) -> None:
        self.tasks[task_id] = {'id': task_id, 'status': status, 'metadata': metadata}

    def metadata(self, task_id: str) -> dict[str, Any]:
        return self.tasks[task_id]['metadata']

    def status(self, task_id: str) -> str:
        return self.tasks[task_id]['status']

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        return copy.deepcopy(self.tasks.get(task_id))

    async def get_status(self, task_id: str) -> str | None:
        task = self.tasks.get(task_id)
        return None if task is None else task['status']

    async def update_task(
        self, task_id: str, metadata: dict[str, Any], *,
        append: bool = False, metadata_mode: str | None = None,
    ) -> bool:
        self.calls.append(('update_task', task_id, copy.deepcopy(metadata), metadata_mode))
        if self.reject_ruling_writes and RECUT_RULING_KEY in metadata:
            return False
        task = self.tasks[task_id]
        if metadata_mode == 'replace':
            task['metadata'] = copy.deepcopy(metadata)
        else:
            task['metadata'] = {**task['metadata'], **copy.deepcopy(metadata)}
        return True

    async def set_task_status(self, task_id: str, status: str, **_: Any) -> None:
        self.calls.append(('set_task_status', task_id, status))
        self.tasks[task_id]['status'] = status

    def index_of(self, predicate) -> int:
        return next(i for i, call in enumerate(self.calls) if predicate(call))

    def ruling_writes(self, task_id: str) -> list[int]:
        return [
            i for i, call in enumerate(self.calls)
            if call[:2] == ('update_task', task_id) and RECUT_RULING_KEY in call[2]
        ]


@pytest.fixture
def store() -> _TaskStore:
    return _TaskStore()


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


@pytest.fixture
def harness(mock_orch_config, store: _TaskStore, queue: EscalationQueue) -> Harness:
    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)
    h.scheduler = MagicMock()
    wire_scheduler_liveness_mock(h.scheduler)
    h.scheduler.get_task = AsyncMock(side_effect=store.get_task)
    h.scheduler.get_status = AsyncMock(side_effect=store.get_status)
    h.scheduler.update_task = AsyncMock(side_effect=store.update_task)
    h.scheduler.set_task_status = AsyncMock(side_effect=store.set_task_status)
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
        [ruling_write] = store.ruling_writes('42')
        assert ruling_write < store.index_of(lambda c: c == ('set_task_status', '42', 'pending'))
        assert store.status('42') == 'pending'

    async def test_plain_restart_issues_no_ruling(self, server, harness, store, queue):
        store.add('42')
        l1 = _file_l1(queue, '42')

        await _resolve(server, harness, escalation_id=l1.id, action='restart')

        assert RECUT_RULING_KEY not in store.metadata('42')
        assert store.ruling_writes('42') == []
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
        store.reject_ruling_writes = True
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
            '42', merge_retry_pending={'attempt': 1}, files=['a.py', 'b.py'],
        )
        l1 = _file_l1(queue, '42')

        await _resolve(server, harness, escalation_id=l1.id, action='restart', recut=True)

        metadata = store.metadata('42')
        assert 'merge_retry_pending' not in metadata
        assert RECUT_RULING_KEY in metadata
        assert metadata['files'] == ['a.py', 'b.py']
        assert store.status('42') == 'pending'
