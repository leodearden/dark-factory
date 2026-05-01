"""Targeted reconciliation — lightweight, triggered by task state transitions."""

import contextlib
import json
import logging
import uuid as uuid_mod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.models.reconciliation import (
    MemoryHints,
    ReconciliationRun,
    RunStatus,
    RunType,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.journal import ReconciliationJournal
from fused_memory.reconciliation.verify import CodebaseVerifier
from fused_memory.services.memory_service import MemoryService
from fused_memory.utils.validation import InputValidationError, require_project_root

if TYPE_CHECKING:
    from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry

logger = logging.getLogger(__name__)


class TargetedReconciler:
    """Lightweight reconciliation triggered by task state transitions."""

    def __init__(
        self,
        memory_service: MemoryService,
        taskmaster: TaskmasterBackend,
        journal: ReconciliationJournal,
        config: FusedMemoryConfig,
        event_buffer: EventBuffer | None = None,
    ):
        self.memory = memory_service
        self.taskmaster = taskmaster
        self.journal = journal
        self.config = config.reconciliation
        self.verifier = CodebaseVerifier(config.reconciliation)
        self.buffer = event_buffer
        self.planned_episode_registry: PlannedEpisodeRegistry | None = None

    async def _fenced_add_memory(
        self,
        content: str,
        category: str,
        project_id: str,
        metadata: dict,
        causation_id: str,
    ) -> bool:
        """Write memory if no full cycle is active; defer otherwise.

        Returns True if written immediately, False if deferred.
        """
        if self.buffer is not None and await self.buffer.is_full_recon_active(project_id):
            metadata = {**metadata, '_deferred': True, '_causation_id': causation_id}
            await self.buffer.defer_write(
                project_id=project_id,
                content=content,
                category=category,
                metadata=metadata,
                agent_id='targeted-reconciliation',
            )
            logger.info(f'Deferred write during active full cycle for task in {project_id}')
            return False

        await self.memory.add_memory(
            content=content,
            category=category,
            project_id=project_id,
            metadata=metadata,
            causation_id=causation_id,
            _source='targeted_recon',
        )
        return True

    async def reconcile_task(
        self,
        task_id: str,
        transition: str,
        project_id: str,
        task_before: dict,
        project_root: str,
    ) -> dict:
        """Run targeted reconciliation for a single task state transition."""
        run_id = str(uuid_mod.uuid4())
        start = datetime.now(UTC)

        run = ReconciliationRun(
            id=run_id,
            project_id=project_id,
            run_type=RunType.targeted,
            trigger_reason=f'task_{transition}:{task_id}',
            started_at=start,
            events_processed=1,
            status=RunStatus.running,
        )
        await self.journal.start_run(run)

        try:
            require_project_root(project_root)

            handler = {
                'done': self._on_task_done,
                'blocked': self._on_task_blocked,
                'cancelled': self._on_task_cancelled,
                'deferred': self._on_task_deferred,
            }.get(transition)

            if handler is None:
                result = {'task_id': task_id, 'actions': [], 'note': f'No handler for {transition}'}
            else:
                result = await handler(task_id, project_id, project_root, task_before, run_id)

            elapsed = (datetime.now(UTC) - start).total_seconds()
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

        except InputValidationError as e:
            logger.warning(f'Targeted reconciliation rejected invalid input: {e}')
            with contextlib.suppress(Exception):
                await self.journal.complete_run(run_id, 'failed')
            raise

        except Exception as e:
            await self.journal.complete_run(run_id, 'failed')
            logger.error(f'Targeted reconciliation failed: {e}')
            return {'error': str(e), 'task_id': task_id}

    async def _on_task_done(
        self, task_id: str, project_id: str, project_root: str, task_before: dict, run_id: str
    ) -> dict:
        """Task completed. Verify knowledge capture, note dependent unblocks."""
        task = _extract_task(task_before)
        result: dict = {'task_id': task_id, 'actions': []}

        title = task.get('title', '')
        description = task.get('description', '')

        # 0. Fast-path: write completion fact immediately (no search/verify needed)
        try:
            content = f"Task '{title}' completed."
            if description:
                content += f" {description}"
            details = task.get('details', '')
            if details:
                content += f"\nDetails: {details[:500]}"

            written = await self._fenced_add_memory(
                content=content,
                category='observations_and_summaries',
                project_id=project_id,
                metadata={
                    'source': 'targeted_reconciliation',
                    'task_id': task_id,
                    'transition': 'done',
                },
                causation_id=run_id,
            )
            action_type = 'knowledge_captured_fast' if written else 'knowledge_deferred_fast'
            result['actions'].append({'type': action_type})
            await self.journal.add_run_action(
                run_id, 'write', 'memory', 'add_memory',
                {'task_id': task_id, 'type': 'completion_fast', 'deferred': not written},
                causation_id=run_id,
            )
        except Exception as e:
            logger.warning(f'Fast-path write failed for task {task_id}: {e}')

        # 1. Search for existing knowledge about this task
        related = await self.memory.search(
            query=f'{title} {description}',
            project_id=project_id,
            limit=5,
            causation_id=run_id,
        )
        await self.journal.add_run_action(
            run_id, 'read', 'search', 'search',
            {'query': f'{title} {description}'[:200], 'results': len(related)},
            causation_id=run_id,
        )

        # 1.5. Promote planned episodes related to this completed task.
        #      Now that the task is done, aspirational edges become factual —
        #      remove them from the planned registry so they appear in normal search.
        if self.planned_episode_registry is not None:
            try:
                planned_related = await self.memory.search(
                    query=f'{title} {description}',
                    project_id=project_id,
                    limit=10,
                    causation_id=run_id,
                    include_planned=True,
                )
                ep_uuids = {
                    ep
                    for r in planned_related
                    if r.metadata.get('planned')
                    for ep in r.provenance
                }
                for ep_uuid in ep_uuids:
                    await self.planned_episode_registry.promote(ep_uuid)
                if ep_uuids:
                    result['actions'].append({
                        'type': 'planned_episodes_promoted',
                        'count': len(ep_uuids),
                    })
                    await self.journal.add_run_action(
                        run_id, 'write', 'registry', 'promote',
                        {'task_id': task_id, 'promoted_count': len(ep_uuids)},
                        causation_id=run_id,
                    )
            except Exception as e:
                logger.warning(f'Planned episode promotion failed for task {task_id}: {e}')

        # 2. If sparse knowledge, verify against codebase and write findings
        if len(related) < 2:
            try:
                verification = await self.verifier.verify(
                    claim=f"Task '{title}' has been completed",
                    context=f'Task details: {task.get("details", description)}',
                    scope_hints=_extract_scope_hints(task),
                )
                if verification.verdict in ('confirmed', 'contradicted'):
                    written = await self._fenced_add_memory(
                        content=f"Completed task '{title}': {verification.summary}",
                        category='observations_and_summaries',
                        project_id=project_id,
                        metadata={
                            'source': 'targeted_reconciliation',
                            'task_id': task_id,
                            'verification_verdict': verification.verdict,
                        },
                        causation_id=run_id,
                    )
                    action_type = 'knowledge_captured' if written else 'knowledge_deferred'
                    result['actions'].append({
                        'type': action_type,
                        'verification': verification.verdict,
                    })
                    await self.journal.add_run_action(
                        run_id, 'write', 'memory', 'add_memory',
                        {'task_id': task_id, 'type': 'verification', 'verdict': verification.verdict,
                         'deferred': not written},
                        causation_id=run_id,
                    )
            except Exception as e:
                logger.warning(f'Verification failed for task {task_id}: {e}')

        # 3. Check dependent tasks — are they unblocked?
        try:
            all_tasks_data = await self.taskmaster.get_tasks(project_root=project_root)
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
        self, task_id: str, project_id: str, project_root: str, task_before: dict, run_id: str
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
            causation_id=run_id,
        )
        await self.journal.add_run_action(
            run_id, 'read', 'search', 'search',
            {'query': f'blockers for: {title}'[:200], 'results': len(related)},
            causation_id=run_id,
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
                    project_root=project_root,
                )
                result['actions'].append({
                    'type': 'hints_attached',
                    'hints': hints.model_dump(),
                })
                await self.journal.add_run_action(
                    run_id, 'write', 'taskmaster', 'update_task',
                    {'task_id': task_id, 'type': 'hints_attached'},
                    causation_id=run_id,
                )
            except Exception as e:
                logger.warning(f'Failed to attach hints to task {task_id}: {e}')

        return result

    async def _on_task_cancelled(
        self, task_id: str, project_id: str, project_root: str, task_before: dict, run_id: str
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
            all_tasks_data = await self.taskmaster.get_tasks(project_root=project_root)
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
        self, task_id: str, project_id: str, project_root: str, task_before: dict, run_id: str
    ) -> dict:
        """Task deferred. Similar to blocked — attach relevant knowledge hints."""
        return await self._on_task_blocked(task_id, project_id, project_root, task_before, run_id)

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
