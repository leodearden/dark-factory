"""Targeted reconciliation — lightweight, triggered by task state transitions."""

import json
import logging
import uuid as uuid_mod
from datetime import datetime, timezone

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.reconciliation import (
    MemoryHints,
    ReconciliationRun,
)
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.verify import CodebaseVerifier
from fused_memory.services.memory_service import MemoryService

logger = logging.getLogger(__name__)


class TargetedReconciler:
    """Lightweight reconciliation triggered by task state transitions."""

    def __init__(
        self,
        memory_service: MemoryService,
        taskmaster: TaskmasterBackend,
        journal: ReconciliationJournal,
        config: FusedMemoryConfig,
    ):
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.config = config.reconciliation
        self.verifier = CodebaseVerifier(config.reconciliation)

    async def reconcile_task(
        self,
        task_id: str,
        transition: str,
        project_id: str,
        task_before: dict,
    ) -> dict:
        """Run targeted reconciliation for a single task state transition."""
        run_id = str(uuid_mod.uuid4())
        start = datetime.now(timezone.utc)

        run = ReconciliationRun(
            id=run_id,
            project_id=project_id,
            run_type='targeted',
            trigger_reason=f'task_{transition}:{task_id}',
            started_at=start,
            events_processed=1,
            status='running',
        )
        await self.journal.start_run(run)

        try:
            handler = {
                'done': self._on_task_done,
                'blocked': self._on_task_blocked,
                'cancelled': self._on_task_cancelled,
                'deferred': self._on_task_deferred,
            }.get(transition)

            if handler is None:
                result = {'task_id': task_id, 'actions': [], 'note': f'No handler for {transition}'}
            else:
                result = await handler(task_id, project_id, task_before, run_id)

            elapsed = (datetime.now(timezone.utc) - start).total_seconds()
            await self.journal.complete_run(run_id, 'completed')
            logger.info(
                'reconciliation.targeted_completed',
                extra={
                    'task_id': task_id,
                    'transition': transition,
                    'duration_seconds': round(elapsed, 2),
                    'actions': len(result.get('actions', [])),
                },
            )
            return result

        except Exception as e:
            await self.journal.complete_run(run_id, 'failed')
            logger.error(f'Targeted reconciliation failed: {e}')
            return {'error': str(e), 'task_id': task_id}

    async def _on_task_done(
        self, task_id: str, project_id: str, task_before: dict, run_id: str
    ) -> dict:
        """Task completed. Verify knowledge capture, note dependent unblocks."""
        task = _extract_task(task_before)
        result: dict = {'task_id': task_id, 'actions': []}

        title = task.get('title', '')
        description = task.get('description', '')

        # 1. Search for existing knowledge about this task
        related = await self.memory.search(
            query=f'{title} {description}',
            project_id=project_id,
            limit=5,
        )

        # 2. If sparse knowledge, verify against codebase and write findings
        if len(related) < 2:
            try:
                verification = await self.verifier.verify(
                    claim=f"Task '{title}' has been completed",
                    context=f'Task details: {task.get("details", description)}',
                    scope_hints=_extract_scope_hints(task),
                )
                if verification.verdict in ('confirmed', 'contradicted'):
                    await self.memory.add_memory(
                        content=f"Completed task '{title}': {verification.summary}",
                        category='observations_and_summaries',
                        project_id=project_id,
                        metadata={
                            'source': 'targeted_reconciliation',
                            'task_id': task_id,
                            'verification_verdict': verification.verdict,
                        },
                    )
                    result['actions'].append({
                        'type': 'knowledge_captured',
                        'verification': verification.verdict,
                    })
            except Exception as e:
                logger.warning(f'Verification failed for task {task_id}: {e}')

        # 3. Check dependent tasks — are they unblocked?
        try:
            all_tasks_data = await self.taskmaster.get_tasks(project_root=project_id)
            all_tasks = all_tasks_data.get('tasks', [])
            if isinstance(all_tasks, list):
                for t in all_tasks:
                    if not isinstance(t, dict):
                        continue
                    deps = t.get('dependencies', [])
                    if task_id in [str(d) for d in deps]:
                        all_deps_done = all(
                            any(
                                str(dt.get('id')) == str(dep_id) and dt.get('status') == 'done'
                                for dt in all_tasks
                                if isinstance(dt, dict)
                            )
                            for dep_id in deps
                        )
                        if all_deps_done and t.get('status') == 'pending':
                            result['actions'].append({
                                'type': 'dependent_unblocked',
                                'task_id': t.get('id'),
                                'title': t.get('title'),
                            })
        except Exception as e:
            logger.warning(f'Dependency check failed: {e}')

        return result

    async def _on_task_blocked(
        self, task_id: str, project_id: str, task_before: dict, run_id: str
    ) -> dict:
        """Task blocked. Search for relevant knowledge, attach as hints."""
        task = _extract_task(task_before)
        result: dict = {'task_id': task_id, 'actions': []}

        title = task.get('title', '')
        description = task.get('description', '')

        related = await self.memory.search(
            query=f'blockers for: {title} {description}',
            project_id=project_id,
            limit=5,
        )

        if related:
            entities = []
            for r in related:
                entities.extend(r.entities)
            hints = MemoryHints(
                entities=list(set(entities)),
                queries=[f'resolution for: {title}'],
            )
            try:
                await self.taskmaster.update_task(
                    task_id=task_id,
                    metadata=json.dumps({'memory_hints': hints.model_dump()}),
                    project_root=project_id,
                )
                result['actions'].append({
                    'type': 'hints_attached',
                    'hints': hints.model_dump(),
                })
            except Exception as e:
                logger.warning(f'Failed to attach hints to task {task_id}: {e}')

        return result

    async def _on_task_cancelled(
        self, task_id: str, project_id: str, task_before: dict, run_id: str
    ) -> dict:
        """Task cancelled. Flag subtasks and dependents for review."""
        task = _extract_task(task_before)
        result: dict = {'task_id': task_id, 'actions': []}

        # Check for subtasks
        subtasks = task.get('subtasks', [])
        if subtasks:
            active_subtasks = [
                s for s in subtasks
                if isinstance(s, dict) and s.get('status') not in ('done', 'cancelled')
            ]
            if active_subtasks:
                result['actions'].append({
                    'type': 'subtasks_need_review',
                    'count': len(active_subtasks),
                    'subtask_ids': [s.get('id') for s in active_subtasks],
                })

        # Check for tasks that depend on this one
        try:
            all_tasks_data = await self.taskmaster.get_tasks(project_root=project_id)
            all_tasks = all_tasks_data.get('tasks', [])
            if isinstance(all_tasks, list):
                for t in all_tasks:
                    if not isinstance(t, dict):
                        continue
                    deps = t.get('dependencies', [])
                    if task_id in [str(d) for d in deps] and t.get('status') not in ('done', 'cancelled'):
                        result['actions'].append({
                            'type': 'dependent_affected',
                            'task_id': t.get('id'),
                            'title': t.get('title'),
                        })
        except Exception as e:
            logger.warning(f'Dependent check failed for cancelled task {task_id}: {e}')

        return result

    async def _on_task_deferred(
        self, task_id: str, project_id: str, task_before: dict, run_id: str
    ) -> dict:
        """Task deferred. Similar to blocked — attach relevant knowledge hints."""
        return await self._on_task_blocked(task_id, project_id, task_before, run_id)

    async def reconcile_bulk_tasks(
        self,
        parent_task_id: str | None,
        project_id: str,
    ) -> dict:
        """Reconcile after expand_task or parse_prd — cross-reference against knowledge."""
        result: dict = {'parent_task_id': parent_task_id, 'actions': []}

        try:
            all_tasks_data = await self.taskmaster.get_tasks(project_root=project_id)
            all_tasks = all_tasks_data.get('tasks', [])
            if not isinstance(all_tasks, list):
                return result

            # Get tasks to check
            if parent_task_id:
                # Subtasks of the expanded task
                parent = next(
                    (t for t in all_tasks if isinstance(t, dict) and str(t.get('id')) == parent_task_id),
                    None,
                )
                tasks_to_check = (parent or {}).get('subtasks', [])
            else:
                # All pending tasks (parse_prd creates top-level tasks)
                tasks_to_check = [
                    t for t in all_tasks
                    if isinstance(t, dict) and t.get('status') == 'pending'
                ]

            # Cross-reference each against knowledge
            for task in tasks_to_check[:20]:  # Limit to avoid excessive API calls
                if not isinstance(task, dict):
                    continue
                title = task.get('title', '')
                tid = task.get('id', '')

                related = await self.memory.search(
                    query=title, project_id=project_id, limit=3
                )

                if related:
                    entities = []
                    for r in related:
                        entities.extend(r.entities)
                    if entities:
                        hints = MemoryHints(
                            entities=list(set(entities))[:10],
                            queries=[title],
                        )
                        try:
                            await self.taskmaster.update_task(
                                task_id=str(tid),
                                metadata=json.dumps({'memory_hints': hints.model_dump()}),
                                project_root=project_id,
                            )
                            result['actions'].append({
                                'type': 'hints_attached',
                                'task_id': tid,
                            })
                        except Exception:
                            pass

        except Exception as e:
            logger.warning(f'Bulk reconciliation failed: {e}')
            result['error'] = str(e)

        return result


def _extract_task(task_data: dict) -> dict:
    """Normalize Taskmaster response to get the task dict."""
    if 'data' in task_data and isinstance(task_data['data'], dict):
        return task_data['data']
    return task_data


def _extract_scope_hints(task: dict) -> list[str]:
    """Extract file/directory hints from task data."""
    hints = []
    details = task.get('details', '') or ''
    # Look for file paths in the details
    for word in details.split():
        if '/' in word and ('.' in word.split('/')[-1] or word.endswith('/')):
            cleaned = word.strip('`"\'(),;:')
            if cleaned:
                hints.append(cleaned)
    return hints[:5]
