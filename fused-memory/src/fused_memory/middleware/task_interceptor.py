"""Intercepts task state transitions for targeted reconciliation."""

import logging
import uuid as uuid_mod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.reconciliation.event_buffer import EventBuffer

if TYPE_CHECKING:
    from fused_memory.reconciliation.targeted import TargetedReconciler

logger = logging.getLogger(__name__)


class TaskInterceptor:
    """Wraps Taskmaster operations, intercepts state transitions for targeted reconciliation."""

    STATUS_TRIGGERS = {'done', 'blocked', 'cancelled', 'deferred'}
    BULK_TRIGGERS = {'parse_prd', 'expand_task'}

    def __init__(
        self,
        taskmaster: TaskmasterBackend,
        targeted_reconciler: 'TargetedReconciler | None',
        event_buffer: EventBuffer,
    ):
        self.taskmaster = taskmaster
        self.reconciler = targeted_reconciler
        self.buffer = event_buffer

    def _make_event(
        self, event_type: EventType, project_id: str, payload: dict
    ) -> ReconciliationEvent:
        return ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=event_type,
            source=EventSource.agent,
            project_id=project_id,
            timestamp=datetime.now(UTC),
            payload=payload,
        )

    # ── Status transitions (with targeted reconciliation) ──────────────

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict:
        """Proxy to Taskmaster, then run targeted reconciliation if triggered."""
        # 1. Get before-state
        before = await self.taskmaster.get_task(task_id, project_root, tag)

        # 2. Execute status change
        result = await self.taskmaster.set_task_status(task_id, status, project_root, tag)

        # 3. Emit event
        old_status = _extract_status(before)
        event = self._make_event(
            EventType.task_status_changed,
            project_root,
            {'task_id': task_id, 'old_status': old_status, 'new_status': status},
        )
        await self.buffer.push(event)

        # 4. Targeted reconciliation for trigger statuses
        if status in self.STATUS_TRIGGERS and self.reconciler:
            try:
                recon_result = await self.reconciler.reconcile_task(
                    task_id=task_id,
                    transition=status,
                    project_id=project_root,
                    task_before=before,
                )
                result['reconciliation'] = recon_result
                logger.info(
                    'reconciliation.targeted_completed',
                    extra={
                        'task_id': task_id,
                        'transition': status,
                        'project_id': project_root,
                    },
                )
            except Exception as e:
                logger.error(f'Targeted reconciliation failed for task {task_id}: {e}')
                result['reconciliation_error'] = str(e)

        return result

    # ── Bulk operations (with targeted reconciliation) ─────────────────

    async def expand_task(
        self,
        task_id: str,
        project_root: str,
        num: str | None = None,
        prompt: str | None = None,
        force: bool = False,
        tag: str | None = None,
    ) -> dict:
        result = await self.taskmaster.expand_task(
            task_id, project_root, num=num, prompt=prompt, force=force, tag=tag
        )
        event = self._make_event(
            EventType.tasks_bulk_created,
            project_root,
            {'parent_task_id': task_id, 'operation': 'expand_task'},
        )
        await self.buffer.push(event)

        if self.reconciler:
            try:
                recon_result = await self.reconciler.reconcile_bulk_tasks(
                    parent_task_id=task_id, project_id=project_root
                )
                result['reconciliation'] = recon_result
            except Exception as e:
                logger.error(f'Bulk reconciliation failed for expand_task {task_id}: {e}')
                result['reconciliation_error'] = str(e)

        return result

    async def parse_prd(
        self,
        input_path: str,
        project_root: str,
        num_tasks: str | None = None,
        tag: str | None = None,
    ) -> dict:
        result = await self.taskmaster.parse_prd(
            input_path, project_root, num_tasks=num_tasks, tag=tag
        )
        event = self._make_event(
            EventType.tasks_bulk_created,
            project_root,
            {'input_path': input_path, 'operation': 'parse_prd'},
        )
        await self.buffer.push(event)

        if self.reconciler:
            try:
                recon_result = await self.reconciler.reconcile_bulk_tasks(
                    parent_task_id=None, project_id=project_root
                )
                result['reconciliation'] = recon_result
            except Exception as e:
                logger.error(f'Bulk reconciliation failed for parse_prd: {e}')
                result['reconciliation_error'] = str(e)

        return result

    # ── Write pass-throughs (emit event, no targeted reconciliation) ───

    async def add_task(self, project_root: str, **kwargs: Any) -> dict:
        result = await self.taskmaster.add_task(project_root=project_root, **kwargs)
        event = self._make_event(
            EventType.task_created,
            project_root,
            {'operation': 'add_task'},
        )
        await self.buffer.push(event)
        return result

    async def update_task(
        self, task_id: str, project_root: str, **kwargs: Any
    ) -> dict:
        result = await self.taskmaster.update_task(
            task_id=task_id, project_root=project_root, **kwargs
        )
        event = self._make_event(
            EventType.task_modified,
            project_root,
            {'task_id': task_id, 'operation': 'update_task'},
        )
        await self.buffer.push(event)
        return result

    async def add_subtask(
        self, parent_id: str, project_root: str, **kwargs: Any
    ) -> dict:
        result = await self.taskmaster.add_subtask(
            parent_id=parent_id, project_root=project_root, **kwargs
        )
        event = self._make_event(
            EventType.task_created,
            project_root,
            {'parent_id': parent_id, 'operation': 'add_subtask'},
        )
        await self.buffer.push(event)
        return result

    async def remove_task(
        self, task_id: str, project_root: str, tag: str | None = None
    ) -> dict:
        result = await self.taskmaster.remove_task(task_id, project_root, tag)
        event = self._make_event(
            EventType.task_deleted,
            project_root,
            {'task_id': task_id, 'operation': 'remove_task'},
        )
        await self.buffer.push(event)
        return result

    async def add_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict:
        result = await self.taskmaster.add_dependency(
            task_id, depends_on, project_root, tag
        )
        event = self._make_event(
            EventType.task_modified,
            project_root,
            {'task_id': task_id, 'depends_on': depends_on, 'operation': 'add_dependency'},
        )
        await self.buffer.push(event)
        return result

    async def remove_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict:
        result = await self.taskmaster.remove_dependency(
            task_id, depends_on, project_root, tag
        )
        event = self._make_event(
            EventType.task_modified,
            project_root,
            {'task_id': task_id, 'depends_on': depends_on, 'operation': 'remove_dependency'},
        )
        await self.buffer.push(event)
        return result

    # ── Pure reads (direct pass-through) ───────────────────────────────

    async def get_tasks(
        self, project_root: str, tag: str | None = None
    ) -> dict:
        return await self.taskmaster.get_tasks(project_root, tag)

    async def get_task(
        self, task_id: str, project_root: str, tag: str | None = None
    ) -> dict:
        return await self.taskmaster.get_task(task_id, project_root, tag)


def _extract_status(task_data: dict) -> str:
    """Extract status from Taskmaster get_task response."""
    if 'status' in task_data:
        return task_data['status']
    data = task_data.get('data', {})
    if isinstance(data, dict):
        return data.get('status', 'unknown')
    return 'unknown'
