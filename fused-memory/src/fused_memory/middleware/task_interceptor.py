"""Intercepts task state transitions for targeted reconciliation."""

import asyncio
import hashlib
import logging
import time
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
    from fused_memory.config.schema import FusedMemoryConfig
    from fused_memory.middleware.task_dedup import TaskDeduplicator
    from fused_memory.middleware.task_file_committer import TaskFileCommitter
    from fused_memory.reconciliation.targeted import TargetedReconciler

logger = logging.getLogger(__name__)

# Dedup cache: suppress identical add_task calls within this window (seconds).
_DEDUP_CACHE_TTL = 300


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
        config: 'FusedMemoryConfig | None' = None,
    ):
        self.taskmaster = taskmaster
        self.reconciler = targeted_reconciler
        self.buffer = event_buffer
        self.task_committer = task_committer
        self._background_tasks: set[asyncio.Task] = set()

        # Layer 1: in-memory title-hash cache  {hash -> (task_result, timestamp)}
        self._dedup_cache: dict[str, tuple[dict, float]] = {}
        # Layer 2: Qdrant vector similarity (created lazily)
        self._config = config
        self._deduplicator: TaskDeduplicator | None = None

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

        # Pre-snapshot: capture the task tree before Taskmaster generates subtasks
        try:
            pre_snapshot = await tm.get_tasks(project_root)
        except Exception as pre_exc:
            logger.warning(
                'bulk_dedup: pre-snapshot failed for expand_task(%s): %s', task_id, pre_exc,
            )
            pre_snapshot = None

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

        # Post-hoc dedup: remove newly-created tasks that duplicate pre-existing ones
        try:
            if pre_snapshot is not None:
                dedup = await self._dedupe_bulk_created(
                    project_root, pre_snapshot, parent_task_id=task_id,
                )
            else:
                dedup = {
                    'removed': [], 'kept': [], 'errors': [],
                    'skipped_reason': 'pre_snapshot_failed',
                }
            result['dedup'] = dedup
        except Exception as dedup_exc:
            logger.warning(
                'bulk_dedup: dedup block failed for expand_task(%s): %s', task_id, dedup_exc,
            )
            result['dedup'] = {'skipped_reason': f'exception: {dedup_exc}'}

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

        # Pre-snapshot: capture the task tree before Taskmaster generates tasks
        try:
            pre_snapshot = await tm.get_tasks(project_root)
        except Exception as pre_exc:
            logger.warning('bulk_dedup: pre-snapshot failed for parse_prd: %s', pre_exc)
            pre_snapshot = None

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

        # Post-hoc dedup: remove newly-created tasks that duplicate pre-existing ones
        try:
            if pre_snapshot is not None:
                dedup = await self._dedupe_bulk_created(project_root, pre_snapshot)
            else:
                dedup = {
                    'removed': [], 'kept': [], 'errors': [],
                    'skipped_reason': 'pre_snapshot_failed',
                }
            result['dedup'] = dedup
        except Exception as dedup_exc:
            logger.warning('bulk_dedup: dedup block failed for parse_prd: %s', dedup_exc)
            result['dedup'] = {'skipped_reason': f'exception: {dedup_exc}'}

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

    # ── Dedup helpers ─────────��─────────────────────────────────────────

    @staticmethod
    def _title_hash(title: str) -> str:
        """Deterministic hash of a normalized title for the dedup cache."""
        normalized = ' '.join(title.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _check_dedup_cache(self, title: str) -> dict | None:
        """Layer 1: check the in-memory title-hash cache.

        Returns the cached task result if a matching title was created
        within the TTL window, otherwise None.  Also evicts stale entries.
        """
        now = time.monotonic()
        # Lazy eviction
        stale = [k for k, (_, ts) in self._dedup_cache.items() if now - ts > _DEDUP_CACHE_TTL]
        for k in stale:
            del self._dedup_cache[k]

        h = self._title_hash(title)
        entry = self._dedup_cache.get(h)
        if entry is not None:
            cached_result, ts = entry
            if now - ts <= _DEDUP_CACHE_TTL:
                return cached_result
            del self._dedup_cache[h]
        return None

    def _store_dedup_cache(self, title: str, result: dict) -> None:
        self._dedup_cache[self._title_hash(title)] = (result, time.monotonic())

    async def _get_deduplicator(self) -> 'TaskDeduplicator | None':
        """Lazily create the Qdrant-backed deduplicator."""
        if self._deduplicator is not None:
            return self._deduplicator
        if self._config is None:
            return None
        try:
            from fused_memory.middleware.task_dedup import TaskDeduplicator

            self._deduplicator = TaskDeduplicator(self._config)
            return self._deduplicator
        except Exception:
            logger.warning('Failed to create TaskDeduplicator', exc_info=True)
            return None

    @staticmethod
    def _flatten_tasks(tasks_data: dict) -> list[tuple[str, str, str]]:
        """Flatten a get_tasks response into (id, normalized_title, status) tuples.

        Walks all top-level tasks and their subtasks recursively so that every
        task in the tree is represented exactly once.
        """
        result: list[tuple[str, str, str]] = []

        def _walk(task_list: list) -> None:
            for task in task_list:
                tid = str(task.get('id', ''))
                title = ' '.join(task.get('title', '').lower().split())
                status = task.get('status', 'unknown')
                result.append((tid, title, status))
                subtasks = task.get('subtasks', [])
                if subtasks:
                    _walk(subtasks)

        _walk(tasks_data.get('tasks', []))
        return result

    @staticmethod
    def _build_pre_title_index(
        pre_flattened: list[tuple[str, str, str]],
    ) -> dict[str, tuple[str, str]]:
        """Build hash→(task_id, title) index from a flattened pre-snapshot.

        Used to detect exact-title duplicates among newly-created tasks.
        First occurrence wins when the same normalized title appears multiple times.
        """
        index: dict[str, tuple[str, str]] = {}
        for tid, title, _ in pre_flattened:
            if title:
                h = hashlib.sha256(title.encode()).hexdigest()[:16]
                index.setdefault(h, (tid, title))
        return index

    async def _dedupe_bulk_created(
        self,
        project_root: str,
        pre_snapshot: dict,
        parent_task_id: str | None = None,
    ) -> dict:
        """Post-hoc dedup after a bulk task-creation operation.

        Reads the current task tree, diffs it against pre_snapshot, removes any
        new tasks whose normalized title matches a pre-existing one (Layer 1: hash)
        or is semantically similar to one (Layer 2: Qdrant vector), then records
        survivors in both dedup stores for future protection.

        Returns {'removed': [...], 'kept': [...], 'errors': []}
        """
        removed: list[dict] = []
        kept: list[dict] = []
        errors: list[dict] = []

        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)

        # Build pre-snapshot index (step-4: extracted into _build_pre_title_index)
        pre_flattened = self._flatten_tasks(pre_snapshot)
        pre_ids = {tid for tid, _, _ in pre_flattened}
        pre_title_index = self._build_pre_title_index(pre_flattened)

        # Get post-snapshot and find newly-created tasks (not in pre-snapshot)
        post_snapshot = await tm.get_tasks(project_root)
        post_flattened = self._flatten_tasks(post_snapshot)
        new_tasks = [
            (tid, title, status)
            for tid, title, status in post_flattened
            if tid not in pre_ids
        ]

        if not new_tasks:
            return {'removed': removed, 'kept': kept, 'errors': errors}

        # Lazy-load the vector deduplicator once for all new tasks (step-6)
        deduplicator = await self._get_deduplicator()

        for tid, title, _status in new_tasks:
            is_dup = False

            # Layer 1: exact normalized-title hash match against pre-existing titles
            if title:
                h = self._title_hash(title)
                if h in pre_title_index:
                    is_dup = True
                    matched_id, matched_title = pre_title_index[h]
                    try:
                        await tm.remove_task(tid, project_root)
                        removed.append({
                            'task_id': tid,
                            'title': title,
                            'reason': 'exact_title_match',
                            'matched_task_id': matched_id,
                        })
                        logger.warning(
                            'bulk_dedup: removed duplicate task %s '
                            '(title matches pre-existing task %s: %r)',
                            tid, matched_id, title,
                        )
                    except Exception as exc:
                        logger.warning(
                            'bulk_dedup: remove_task failed for %s: %s', tid, exc,
                        )
                        errors.append({'task_id': tid, 'title': title, 'error': str(exc)})

            # Layer 2: vector similarity check (step-6) — only if Layer 1 missed
            if not is_dup and title and deduplicator is not None:
                try:
                    match = await deduplicator.find_duplicate(title, project_id)
                    if match is not None:
                        is_dup = True
                        try:
                            await tm.remove_task(tid, project_root)
                            removed.append({
                                'task_id': tid,
                                'title': title,
                                'reason': 'vector_similarity',
                                'matched_task_id': match['task_id'],
                                'similarity_score': match['score'],
                            })
                            logger.warning(
                                'bulk_dedup: removed duplicate task %s '
                                '(vector similarity %.3f vs task %s: %r)',
                                tid, match['score'], match['task_id'], title,
                            )
                        except Exception as exc:
                            logger.warning(
                                'bulk_dedup: remove_task failed for %s: %s', tid, exc,
                            )
                            errors.append({'task_id': tid, 'title': title, 'error': str(exc)})
                except Exception as exc:
                    logger.warning('bulk_dedup: find_duplicate failed for %s: %s', tid, exc)

            # Record survivors in dedup stores (step-8)
            if not is_dup:
                kept.append({'task_id': tid, 'title': title})
                # Layer 1: populate cache so future add_task/bulk calls dedup against it
                self._store_dedup_cache(title, {'id': tid, 'title': title})
                # Layer 2: fire-and-forget embedding write (same pattern as add_task)
                if deduplicator is not None:
                    bg = asyncio.create_task(
                        deduplicator.record_task(tid, title, project_id),
                        name=f'bulk-dedup-record-{tid}',
                    )
                    self._background_tasks.add(bg)
                    bg.add_done_callback(lambda t: self._background_tasks.discard(t))

        return {'removed': removed, 'kept': kept, 'errors': errors}

    # ── Write pass-throughs (emit event, no targeted reconciliation) ───

    async def add_task(self, project_root: str, **kwargs: Any) -> dict:
        # Extract metadata before forwarding — taskmaster's add_task doesn't accept it
        metadata = kwargs.pop('metadata', None)
        title = kwargs.get('title') or kwargs.get('prompt') or ''

        # ── Dedup Layer 1: in-memory title-hash cache ────────────────
        if title:
            cached = self._check_dedup_cache(title)
            if cached is not None:
                task_id = cached.get('id', '?')
                logger.warning(
                    'task_dedup: exact title match in cache — returning existing task %s '
                    'instead of creating duplicate: %s',
                    task_id, title[:80],
                )
                return cached

        # ── Dedup Layer 2: Qdrant vector similarity ──────────────────
        if title:
            deduplicator = await self._get_deduplicator()
            if deduplicator is not None:
                project_id = resolve_project_id(project_root)
                match = await deduplicator.find_duplicate(title, project_id)
                if match is not None:
                    logger.warning(
                        'task_dedup: similar task found (score=%.3f) — returning '
                        'existing task %s instead of creating duplicate: %s',
                        match['score'], match['task_id'], title[:80],
                    )
                    return {
                        'id': match['task_id'],
                        'title': match['task_title'],
                        'deduplicated': True,
                        'similarity_score': match['score'],
                    }

        # ── Create task ──────────────────────────────────────────────
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

        # ── Record in dedup stores ────────────���──────────────────────
        if title and isinstance(result, dict):
            self._store_dedup_cache(title, result)
            deduplicator = await self._get_deduplicator()
            if deduplicator is not None and task_id:
                project_id = resolve_project_id(project_root)
                # Fire-and-forget — don't block on embedding write
                bg = asyncio.create_task(
                    deduplicator.record_task(task_id, title, project_id),
                    name=f'dedup-record-{task_id}',
                )
                self._background_tasks.add(bg)
                bg.add_done_callback(lambda t: self._background_tasks.discard(t))

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
