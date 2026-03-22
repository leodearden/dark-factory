"""Intercepts task state transitions for targeted reconciliation."""

import asyncio
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
from fused_memory.models.scope import resolve_project_id
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
        taskmaster: TaskmasterBackend | None,
        targeted_reconciler: 'TargetedReconciler | None',
        event_buffer: EventBuffer,
    ):
        self.taskmaster = taskmaster
        self.reconciler = targeted_reconciler
        self.buffer = event_buffer

    async def _ensure_taskmaster(self) -> TaskmasterBackend:
        """Return a connected TaskmasterBackend, or raise with a structured error."""
        if self.taskmaster is None:
            raise RuntimeError('Taskmaster is not configured.')
        await self.taskmaster.ensure_connected()
        return self.taskmaster

    def _make_event(
        self, event_type: EventType, project_root: str, payload: dict
    ) -> ReconciliationEvent:
        payload = {**payload, '_project_root': project_root}
        return ReconciliationEvent(
            id=str(uuid_mod.uuid4()),
            type=event_type,
            source=EventSource.agent,
            project_id=resolve_project_id(project_root),
            timestamp=datetime.now(UTC),
            payload=payload,
        )

    @staticmethod
    def _on_reconciliation_done(task: asyncio.Task) -> None:
        """Callback for fire-and-forget reconciliation tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f'Background reconciliation failed: {exc}')

    # ── Status transitions (with targeted reconciliation) ──────────────

    async def set_task_status(
        self,
        task_id: str,
        status: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict:
        """Proxy to Taskmaster, then fire-and-forget targeted reconciliation if triggered."""
        tm = await self._ensure_taskmaster()
        # 1. Get before-state
        before = await tm.get_task(task_id, project_root, tag)

        # 2. Same-status guard: no-op if nothing changed
        old_status = _extract_status(before)
        if status == old_status:
            return {'success': True, 'no_op': True, 'task_id': task_id}

        # 3. Terminal state guard (defense in depth)
        _TERMINAL = frozenset({'done', 'cancelled'})
        if old_status in _TERMINAL and status != old_status:
            logger.warning(
                'Task %s: rejecting %s->%s (terminal state)', task_id, old_status, status
            )
            return {
                'success': False,
                'error': f'Cannot transition from terminal status {old_status!r} to {status!r}',
                'task_id': task_id,
            }

        # 4. Execute status change
        result = await tm.set_task_status(task_id, status, project_root, tag)

        # 5. Emit event
        event = self._make_event(
            EventType.task_status_changed,
            project_root,
            {'task_id': task_id, 'old_status': old_status, 'new_status': status},
        )
        await self.buffer.push(event)

        # 6. Targeted reconciliation for trigger statuses (fire-and-forget)
        if status in self.STATUS_TRIGGERS and self.reconciler:
            task = asyncio.create_task(
                self.reconciler.reconcile_task(
                    task_id=task_id,
                    transition=status,
                    project_id=resolve_project_id(project_root),
                    project_root=project_root,
                    task_before=before,
                ),
                name=f'targeted-recon-{task_id}-{status}',
            )
            task.add_done_callback(self._on_reconciliation_done)
            result['reconciliation'] = {'status': 'async', 'task_id': task_id}

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
        tm = await self._ensure_taskmaster()
        result = await tm.expand_task(
            task_id, project_root, num=num, prompt=prompt, force=force, tag=tag
        )
        event = self._make_event(
            EventType.tasks_bulk_created,
            project_root,
            {'parent_task_id': task_id, 'operation': 'expand_task'},
        )
        await self.buffer.push(event)

        if self.reconciler:
            task = asyncio.create_task(
                self.reconciler.reconcile_bulk_tasks(
                    parent_task_id=task_id,
                    project_id=resolve_project_id(project_root),
                    project_root=project_root,
                ),
                name=f'bulk-recon-expand-{task_id}',
            )
            task.add_done_callback(self._on_reconciliation_done)
            result['reconciliation'] = {'status': 'async', 'task_id': task_id}

        return result

    async def parse_prd(
        self,
        input_path: str,
        project_root: str,
        num_tasks: str | None = None,
        tag: str | None = None,
    ) -> dict:
        tm = await self._ensure_taskmaster()
        result = await tm.parse_prd(
            input_path, project_root, num_tasks=num_tasks, tag=tag
        )
        event = self._make_event(
            EventType.tasks_bulk_created,
            project_root,
            {'input_path': input_path, 'operation': 'parse_prd'},
        )
        await self.buffer.push(event)

        if self.reconciler:
            task = asyncio.create_task(
                self.reconciler.reconcile_bulk_tasks(
                    parent_task_id=None,
                    project_id=resolve_project_id(project_root),
                    project_root=project_root,
                ),
                name='bulk-recon-parse-prd',
            )
            task.add_done_callback(self._on_reconciliation_done)
            result['reconciliation'] = {'status': 'async', 'operation': 'parse_prd'}

        return result

    # ── Write pass-throughs (emit event, no targeted reconciliation) ───

    async def add_task(self, project_root: str, **kwargs: Any) -> dict:
        tm = await self._ensure_taskmaster()
        result = await tm.add_task(project_root=project_root, **kwargs)
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
        tm = await self._ensure_taskmaster()
        result = await tm.update_task(
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
        tm = await self._ensure_taskmaster()
        result = await tm.add_subtask(
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
        tm = await self._ensure_taskmaster()
        result = await tm.remove_task(task_id, project_root, tag)
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
        tm = await self._ensure_taskmaster()
        result = await tm.add_dependency(
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
        tm = await self._ensure_taskmaster()
        result = await tm.remove_dependency(
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
        tm = await self._ensure_taskmaster()
        return await tm.get_tasks(project_root, tag)

    async def get_task(
        self, task_id: str, project_root: str, tag: str | None = None
    ) -> dict:
        tm = await self._ensure_taskmaster()
        return await tm.get_task(task_id, project_root, tag)


def _extract_status(task_data: dict) -> str:
    """Extract status from Taskmaster get_task response."""
    if 'status' in task_data:
        return task_data['status']
    data = task_data.get('data', {})
    if isinstance(data, dict):
        return data.get('status', 'unknown')
    return 'unknown'
