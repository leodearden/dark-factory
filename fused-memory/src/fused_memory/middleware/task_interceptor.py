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
    from fused_memory.middleware.task_file_committer import TaskFileCommitter
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
        task_committer: 'TaskFileCommitter | None' = None,
    ):
        self.taskmaster = taskmaster
        self.reconciler = targeted_reconciler
        self.buffer = event_buffer
        self.task_committer = task_committer
        self._background_tasks: set[asyncio.Task] = set()

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

    def _schedule_commit(self, project_root: str, operation: str) -> None:
        """Fire-and-forget auto-commit of tasks.json."""
        if self.task_committer is None:
            return
        task = asyncio.create_task(
            self.task_committer.commit(project_root, operation),
            name=f'auto-commit-{operation}',
        )
        self._background_tasks.add(task)
        task.add_done_callback(lambda t: self._background_tasks.discard(t))
        task.add_done_callback(self._on_commit_done)

    @staticmethod
    def _on_commit_done(task: asyncio.Task) -> None:
        """Callback for fire-and-forget commit tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f'Background auto-commit failed: {exc}')

    async def _await_commit(self, project_root: str, operation: str) -> None:
        """Await commit directly (used by bulk ops that must capture full batch)."""
        if self.task_committer is None:
            return
        await self.task_committer.commit(project_root, operation)

    async def drain(self) -> None:
        """Await all pending background tasks (commits + reconciliation).

        Call at shutdown or when you need to guarantee all fire-and-forget
        work has completed.
        """
        if not self._background_tasks:
            return
        tasks = list(self._background_tasks)
        logger.info('Draining %d background tasks', len(tasks))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error('Background task failed during drain: %s', result)

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

        # 3. Execute status change
        result = await tm.set_task_status(task_id, status, project_root, tag)

        # 5. Emit event
        event = self._make_event(
            EventType.task_status_changed,
            project_root,
            {'task_id': task_id, 'old_status': old_status, 'new_status': status},
        )
        await self.buffer.push(event)
        self._schedule_commit(project_root, f'set_task_status({task_id}={status})')

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
            self._background_tasks.add(task)
            task.add_done_callback(lambda t: self._background_tasks.discard(t))
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
        await self._await_commit(project_root, f'expand_task({task_id})')

        if self.reconciler:
            task = asyncio.create_task(
                self.reconciler.reconcile_bulk_tasks(
                    parent_task_id=task_id,
                    project_id=resolve_project_id(project_root),
                    project_root=project_root,
                ),
                name=f'bulk-recon-expand-{task_id}',
            )
            self._background_tasks.add(task)
            task.add_done_callback(lambda t: self._background_tasks.discard(t))
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
        await self._await_commit(project_root, 'parse_prd')

        if self.reconciler:
            task = asyncio.create_task(
                self.reconciler.reconcile_bulk_tasks(
                    parent_task_id=None,
                    project_id=resolve_project_id(project_root),
                    project_root=project_root,
                ),
                name='bulk-recon-parse-prd',
            )
            self._background_tasks.add(task)
            task.add_done_callback(lambda t: self._background_tasks.discard(t))
            task.add_done_callback(self._on_reconciliation_done)
            result['reconciliation'] = {'status': 'async', 'operation': 'parse_prd'}

        return result

    # ── Write pass-throughs (emit event, no targeted reconciliation) ───

    async def add_task(self, project_root: str, **kwargs: Any) -> dict:
        # Extract metadata before forwarding — taskmaster's add_task doesn't accept it
        metadata = kwargs.pop('metadata', None)

        tm = await self._ensure_taskmaster()
        result = await tm.add_task(project_root=project_root, **kwargs)

        # Persist metadata via follow-up update_task if provided
        task_id = None
        if isinstance(result, dict):
            task_id = str(result.get('id', ''))
        if metadata and task_id:
            try:
                await tm.update_task(
                    task_id=task_id, metadata=metadata, project_root=project_root
                )
            except Exception as e:
                logger.warning(f'add_task: metadata update for task {task_id} failed: {e}')

        event = self._make_event(
            EventType.task_created,
            project_root,
            {'operation': 'add_task', 'task_id': task_id},
        )
        await self.buffer.push(event)
        self._schedule_commit(project_root, 'add_task')
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
        self._schedule_commit(project_root, f'update_task({task_id})')
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
        self._schedule_commit(project_root, f'add_subtask({parent_id})')
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
        self._schedule_commit(project_root, f'remove_task({task_id})')
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
        self._schedule_commit(project_root, f'add_dependency({task_id}<-{depends_on})')
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
        self._schedule_commit(project_root, f'remove_dependency({task_id}<-{depends_on})')
        return result

    # ── Pure reads (direct pass-through) ───────────────────────────────

    async def get_tasks(
        self,
        project_root: str,
        tag: str | None = None,
        status: list[str] | None = None,
        compact: bool = False,
    ) -> dict:
        """Get tasks, optionally filtered by status and/or compacted.

        Args:
            project_root: Absolute path to project root.
            tag: Tag context (optional).
            status: If provided, only return tasks whose status is in this list.
                    Subtasks are filtered recursively.
            compact: If True, strip verbose fields (description, details) from
                     each task dict, recursively. Reduces payload size for
                     reconciliation agents. Defaults to False (backward compat).
        """
        tm = await self._ensure_taskmaster()
        result = await tm.get_tasks(project_root, tag)
        if status and isinstance(result, dict):
            tasks = result.get('tasks', [])
            if isinstance(tasks, list):
                result = {**result, 'tasks': _filter_tasks_by_status(tasks, status)}
        if compact and isinstance(result, dict):
            tasks = result.get('tasks', [])
            if isinstance(tasks, list):
                result = {**result, 'tasks': _compact_tasks(tasks)}
        return result

    async def get_task(
        self, task_id: str, project_root: str, tag: str | None = None
    ) -> dict:
        tm = await self._ensure_taskmaster()
        return await tm.get_task(task_id, project_root, tag)

    async def get_task_summary(
        self, project_root: str, tag: str | None = None
    ) -> dict:
        """Return a lightweight summary of the task tree.

        Returns:
            {
                "counts": {"pending": N, "done": N, ...},
                "tasks": [{"id": ..., "status": ..., "title": ...}, ...]
            }

        The counts include all tasks (top-level + subtasks, recursively).
        The tasks list is flat (not hierarchical) and contains only
        id/status/title for each task/subtask.
        """
        result = await self.get_tasks(project_root, tag=tag)
        tasks = result.get('tasks', []) if isinstance(result, dict) else []
        all_tasks = _collect_all_tasks(tasks)

        counts: dict[str, int] = {}
        compact_list = []
        for t in all_tasks:
            status = t.get('status', 'unknown')
            counts[status] = counts.get(status, 0) + 1
            compact_list.append({
                'id': t.get('id'),
                'status': status,
                'title': t.get('title', ''),
            })

        return {'counts': counts, 'tasks': compact_list}


def _collect_all_tasks(tasks: list) -> list:
    """Recursively flatten a task tree into a flat list of task dicts.

    Includes top-level tasks and all subtasks at every nesting level.
    Non-dict elements are ignored.
    """
    flat: list = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        flat.append(t)
        subtasks = t.get('subtasks', [])
        if isinstance(subtasks, list) and subtasks:
            flat.extend(_collect_all_tasks(subtasks))
    return flat


def _compact_task(task: object) -> object:
    """Return a compact version of a task dict with verbose fields stripped.

    Keeps: id, status, title, dependencies, priority.
    Strips: description, details, and any other unlisted verbose fields.
    Subtasks are recursively compacted.

    Non-dict inputs are returned unchanged.
    """
    if not isinstance(task, dict):
        return task

    _VERBOSE_FIELDS = {'description', 'details'}
    result = {k: v for k, v in task.items() if k not in _VERBOSE_FIELDS}

    # Recursively compact subtasks
    if 'subtasks' in result and isinstance(result['subtasks'], list):
        result['subtasks'] = [_compact_task(st) for st in result['subtasks']]

    return result


def _compact_tasks(tasks: list) -> list:
    """Apply _compact_task to each element in a list."""
    return [_compact_task(t) for t in tasks]


def _filter_tasks_by_status(tasks: list, statuses: list[str]) -> list:
    """Filter a task list (with subtasks) to only include tasks matching statuses.

    Recursively filters subtasks as well.  A parent task is kept if it matches
    *or* if any of its subtasks match (to preserve the tree structure).
    """
    status_set = set(statuses)
    filtered: list = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        task_status = t.get('status', 'unknown')
        subtasks = t.get('subtasks', [])

        # Recursively filter subtasks
        if isinstance(subtasks, list) and subtasks:
            filtered_subtasks = _filter_tasks_by_status(subtasks, statuses)
        else:
            filtered_subtasks = []

        if task_status in status_set:
            # Include this task; attach filtered subtasks
            filtered.append({**t, 'subtasks': filtered_subtasks})
        elif filtered_subtasks:
            # Parent doesn't match but has matching subtasks — keep parent for tree context
            filtered.append({**t, 'subtasks': filtered_subtasks})
    return filtered


def _extract_status(task_data: dict) -> str:
    """Extract status from Taskmaster get_task response."""
    if 'status' in task_data:
        return task_data['status']
    data = task_data.get('data', {})
    if isinstance(data, dict):
        return data.get('status', 'unknown')
    return 'unknown'
