"""Intercepts task state transitions for targeted reconciliation."""

import asyncio
import json
import logging
import uuid as uuid_mod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from fused_memory.backends.taskmaster_client import TaskmasterBackend
from fused_memory.middleware.task_curator import (
    CandidateTask,
    CuratorDecision,
    TaskCurator,
    _to_pool_entry,
    flatten_task_tree,
)
from fused_memory.models.reconciliation import (
    EventSource,
    EventType,
    ReconciliationEvent,
)
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.event_buffer import EventBuffer

if TYPE_CHECKING:
    from fused_memory.config.schema import FusedMemoryConfig
    from fused_memory.middleware.curator_escalator import CuratorEscalator
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
        config: 'FusedMemoryConfig | None' = None,
        escalator: 'CuratorEscalator | None' = None,
    ):
        self.taskmaster = taskmaster
        self.reconciler = targeted_reconciler
        self.buffer = event_buffer
        self.task_committer = task_committer
        self._background_tasks: set[asyncio.Task] = set()

        # Task curator: LLM-judged drop/combine/create gate. Lazy-initialized in
        # _get_curator() because it pulls in a Qdrant client + embedder.
        self._config = config
        self._curator: TaskCurator | None = None
        self._escalator = escalator
        # R3: per-project async lock serialises add_task / add_subtask
        # calls. Concurrent triages of the same suggestion no longer race
        # to create duplicate tasks — the second call waits, sees the
        # first's ``note_created`` entry, and short-circuits to drop.
        self._project_locks: dict[str, asyncio.Lock] = {}
        # One-shot flag: prevents redundant auto-backfill checks on subsequent calls.
        self._backfill_triggered: bool = False
        # Set by close(); prevents _get_curator() from re-creating a curator.
        self._closed: bool = False

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

    async def close(self) -> None:
        """Release resources held by the interceptor.

        Drains pending background tasks first (defensive — callers *should*
        drain before close, but this guarantees correctness even if they
        forget), then closes the :class:`TaskCurator`'s Qdrant connection.
        """
        await self.drain()
        if self._curator is not None:
            await self._curator.close()
            self._curator = None
        self._closed = True

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

    # ── Curator helpers ────────────────────────────────────────────────

    async def _get_curator(self) -> TaskCurator | None:
        """Lazily construct the task curator gate.

        On first construction, triggers a background backfill check via
        ``_maybe_backfill_corpus()`` so pre-existing tasks are indexed into the
        curator corpus without blocking the caller.

        Returns ``None`` after :meth:`close` to prevent re-creating resources.
        """
        if self._closed:
            return None
        if self._curator is not None:
            return self._curator
        if self._config is None or not self._config.curator.enabled:
            return None
        try:
            from pathlib import Path as _Path

            cwd = None
            project_root: str | None = None
            if self.taskmaster is not None:
                pr = getattr(self.taskmaster.config, 'project_root', None)
                if pr:
                    cwd = _Path(pr)
                    project_root = str(pr)
            self._curator = TaskCurator(
                config=self._config,
                taskmaster=self.taskmaster,
                cwd=cwd,
                escalator=self._escalator,
            )
            # Trigger the one-shot backfill check as a background task so the
            # caller is not delayed by the Qdrant count() round-trip.
            if project_root is not None:
                bg = asyncio.create_task(
                    self._maybe_backfill_corpus(self._curator, project_root),
                    name='curator-backfill-check',
                )
                self._background_tasks.add(bg)
                bg.add_done_callback(lambda t: self._background_tasks.discard(t))
            return self._curator
        except Exception:
            logger.warning('Failed to create TaskCurator', exc_info=True)
            return None

    async def _maybe_backfill_corpus(self, curator: TaskCurator, project_root: str) -> None:
        """Trigger a one-shot background backfill if the collection is empty.

        Called after the curator is constructed (or lazily on first use).
        Checks collection point count via Qdrant. If count is 0 (or the
        collection doesn't exist), fetches the full task tree, flattens it,
        and awaits ``curator.backfill_corpus()`` inline (this method itself
        runs as a background task spawned by ``_get_curator``).

        The ``_backfill_triggered`` flag prevents re-triggering on subsequent
        calls. All failures degrade silently — nothing here must ever block
        task creation.
        """
        if self._backfill_triggered:
            return
        self._backfill_triggered = True

        try:
            project_id = resolve_project_id(project_root)

            # Check whether the collection already has tasks via the public API.
            count = await curator.corpus_count(project_id)

            if count > 0:
                logger.debug(
                    'task_curator: corpus for project %s has %d points — skipping auto-backfill',
                    project_id, count,
                )
                return

            # Fetch the task tree so we can backfill.
            if self.taskmaster is None:
                return
            try:
                tasks_result = await self.taskmaster.get_tasks(project_root)
                flat_tasks = flatten_task_tree(tasks_result)
            except Exception:
                logger.warning(
                    'task_curator: auto-backfill: get_tasks failed', exc_info=True,
                )
                return

            if not flat_tasks:
                return

            try:
                result = await curator.backfill_corpus(flat_tasks, project_id)
                logger.info(
                    'task_curator: auto-backfill complete — '
                    'upserted=%d skipped=%d errors=%d',
                    result.upserted, result.skipped, result.errors,
                )
            except Exception:
                logger.warning(
                    'task_curator: auto-backfill failed', exc_info=True,
                )

        except Exception:
            logger.warning('task_curator: _maybe_backfill_corpus raised', exc_info=True)

    @staticmethod
    def _build_candidate(kwargs: dict[str, Any]) -> CandidateTask | None:
        """Extract a CandidateTask from add_task / add_subtask kwargs.

        Returns None if there's no title (e.g. pure prompt-only add_task) —
        the curator cannot judge a candidate it cannot read.
        """
        title = str(kwargs.get('title') or '').strip()
        if not title:
            return None

        meta = kwargs.get('metadata') or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if not isinstance(meta, dict):
            meta = {}

        files = meta.get('files_to_modify') or meta.get('modules') or []
        if isinstance(files, str):
            files = [files]
        files = [str(f) for f in files if f]

        return CandidateTask(
            title=title,
            description=str(kwargs.get('description') or ''),
            details=str(kwargs.get('details') or ''),
            files_to_modify=files,
            priority=str(kwargs.get('priority') or 'medium'),
            spawned_from=meta.get('spawned_from'),
            spawn_context=str(meta.get('spawn_context') or 'manual'),
        )

    async def _execute_combine(
        self,
        project_root: str,
        decision: CuratorDecision,
    ) -> dict | None:
        """Apply a curator combine decision to the target task.

        Returns the update_task result on success, None on failure (caller
        falls back to create). Combine is implemented via Taskmaster's
        ``update_task`` with a ``prompt`` that instructs a verbatim replacement
        plus metadata carrying the combine marker.
        """
        if decision.rewritten_task is None or decision.target_id is None:
            return None
        rt = decision.rewritten_task
        tm = await self._ensure_taskmaster()

        files_block = '\n'.join(f'  - {f}' for f in rt.files_to_modify)
        combine_prompt = (
            'Replace this task with EXACTLY the following fields, verbatim. '
            'Do not paraphrase, do not merge with existing content, do not add '
            'commentary. The task curator already produced this coherent '
            "rewrite that subsumes a duplicate task's work.\n\n"
            f'TITLE: {rt.title}\n\n'
            f'DESCRIPTION: {rt.description}\n\n'
            f'PRIORITY: {rt.priority}\n\n'
            f'FILES_TO_MODIFY:\n{files_block}\n\n'
            f'DETAILS:\n{rt.details}\n'
        )
        combine_metadata = json.dumps({
            'curator_action': 'combine',
            'curator_justification': decision.justification[:500],
            'combined_at': datetime.now(UTC).isoformat(),
        })

        try:
            return await tm.update_task(
                task_id=decision.target_id,
                project_root=project_root,
                prompt=combine_prompt,
                metadata=combine_metadata,
                append=False,
            )
        except Exception as exc:
            logger.warning(
                'task_curator: combine update failed for target=%s: %s',
                decision.target_id, exc,
            )
            return None

    async def _dedupe_bulk_created(
        self,
        project_root: str,
        pre_snapshot: dict,
        parent_task_id: str | None = None,
    ) -> dict:
        """Post-hoc curator pass after a bulk task-creation operation.

        Reads the current task tree, diffs against pre_snapshot, and invokes
        the curator on each newly-created task. Drop → remove the new task.
        Combine → rewrite the target, then remove the new task. Create → keep
        and record.

        Returns {'removed': [...], 'kept': [...], 'errors': []}
        """
        removed: list[dict] = []
        kept: list[dict] = []
        errors: list[dict] = []

        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)

        pre_ids = {
            str(t.get('id', '')) for t in flatten_task_tree(pre_snapshot)
            if t.get('id')
        }
        post_snapshot = await tm.get_tasks(project_root)
        new_task_dicts = [
            t for t in flatten_task_tree(post_snapshot)
            if str(t.get('id', '')) and str(t.get('id', '')) not in pre_ids
        ]
        if not new_task_dicts:
            return {'removed': removed, 'kept': kept, 'errors': errors}

        curator = await self._get_curator()
        if curator is None:
            for t in new_task_dicts:
                kept.append({
                    'task_id': str(t.get('id', '')),
                    'title': str(t.get('title', '')),
                })
            return {'removed': removed, 'kept': kept, 'errors': errors}

        for t in new_task_dicts:
            tid = str(t.get('id', ''))
            title = str(t.get('title', ''))
            pool_entry = _to_pool_entry(t, source='module', lock_depth=2)
            files = pool_entry.files_to_modify if pool_entry is not None else []
            candidate = CandidateTask(
                title=title,
                description=str(t.get('description', '') or ''),
                details=str(t.get('details', '') or ''),
                files_to_modify=files,
                priority=str(t.get('priority', 'medium')),
                spawned_from=parent_task_id,
                spawn_context='expand' if parent_task_id else 'parse_prd',
            )
            if not candidate.title:
                kept.append({'task_id': tid, 'title': title})
                continue

            try:
                decision = await curator.curate(candidate, project_id, project_root)
            except Exception as exc:
                logger.warning('bulk_curator: curate() failed for %s: %s', tid, exc)
                kept.append({'task_id': tid, 'title': title})
                continue

            if decision.action == 'drop' and decision.target_id:
                try:
                    await tm.remove_task(tid, project_root)
                    removed.append({
                        'task_id': tid,
                        'title': title,
                        'reason': 'curator_drop',
                        'matched_task_id': decision.target_id,
                        'justification': decision.justification[:200],
                    })
                    logger.warning(
                        'bulk_curator: dropped duplicate task %s (matched %s): %s',
                        tid, decision.target_id, decision.justification[:80],
                    )
                except Exception as exc:
                    errors.append({'task_id': tid, 'title': title, 'error': str(exc)})
                continue

            if decision.action == 'combine' and decision.target_id:
                combine_result = await self._execute_combine(project_root, decision)
                if combine_result is None:
                    kept.append({'task_id': tid, 'title': title})
                    continue
                try:
                    await tm.remove_task(tid, project_root)
                except Exception as exc:
                    errors.append({'task_id': tid, 'title': title, 'error': str(exc)})
                    continue
                removed.append({
                    'task_id': tid,
                    'title': title,
                    'reason': 'curator_combine',
                    'matched_task_id': decision.target_id,
                    'justification': decision.justification[:200],
                })
                if decision.rewritten_task is not None:
                    rt_candidate = CandidateTask(
                        title=decision.rewritten_task.title,
                        description=decision.rewritten_task.description,
                        details=decision.rewritten_task.details,
                        files_to_modify=decision.rewritten_task.files_to_modify,
                        priority=decision.rewritten_task.priority,
                    )
                    bg = asyncio.create_task(
                        curator.reembed_task(
                            decision.target_id, rt_candidate, project_id,
                        ),
                        name=f'curator-reembed-{decision.target_id}',
                    )
                    self._background_tasks.add(bg)
                    bg.add_done_callback(lambda t: self._background_tasks.discard(t))
                continue

            # action == 'create' or degenerate fall-through
            kept.append({'task_id': tid, 'title': title})
            bg = asyncio.create_task(
                curator.record_task(tid, candidate, project_id),
                name=f'curator-record-{tid}',
            )
            self._background_tasks.add(bg)
            bg.add_done_callback(lambda t: self._background_tasks.discard(t))

        return {'removed': removed, 'kept': kept, 'errors': errors}

    # ── Write pass-throughs (emit event, no targeted reconciliation) ───

    def _project_lock(self, project_id: str) -> asyncio.Lock:
        """Return (lazily) the per-project serialisation lock for adds."""
        lock = self._project_locks.get(project_id)
        if lock is None:
            lock = asyncio.Lock()
            self._project_locks[project_id] = lock
        return lock

    @staticmethod
    def _extract_metadata_dict(metadata) -> dict | None:
        """Best-effort parse of ``metadata`` into a dict, or None."""
        if metadata is None:
            return None
        if isinstance(metadata, dict):
            return metadata
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
            except (json.JSONDecodeError, ValueError):
                return None
            return parsed if isinstance(parsed, dict) else None
        return None

    async def _check_escalation_idempotency(
        self, *, project_root: str, metadata,
    ) -> dict | None:
        """Return an existing task's identity if ``(escalation_id,
        suggestion_hash)`` in ``metadata`` matches a non-cancelled task.

        R4: authoritative and cheap — no LLM, no embedding lookup.
        Works even when the curator is disabled or broken, which is
        exactly the regime that triggered the duplicate Type::Error
        tasks (esc-1912-179 → esc-1912-190).
        """
        meta = self._extract_metadata_dict(metadata)
        if not meta:
            return None
        esc_id = meta.get('escalation_id')
        sug_hash = meta.get('suggestion_hash')
        if not isinstance(esc_id, str) or not isinstance(sug_hash, str):
            return None
        if not esc_id or not sug_hash:
            return None

        if self.taskmaster is None:
            return None
        try:
            tasks_result = await self.taskmaster.get_tasks(project_root)
        except Exception:
            logger.debug(
                'r4: get_tasks failed during idempotency check', exc_info=True,
            )
            return None
        for task in flatten_task_tree(tasks_result):
            tmeta = task.get('metadata')
            if not isinstance(tmeta, dict):
                continue
            if tmeta.get('escalation_id') != esc_id:
                continue
            if tmeta.get('suggestion_hash') != sug_hash:
                continue
            if str(task.get('status', '')) == 'cancelled':
                continue
            tid = str(task.get('id', ''))
            if not tid:
                continue
            logger.warning(
                'r4: idempotency hit — returning existing task %s for '
                'escalation_id=%s suggestion_hash=%s',
                tid, esc_id, sug_hash,
            )
            return {
                'id': tid,
                'title': str(task.get('title', '')),
                'deduplicated': True,
                'action': 'idempotency_hit',
                'reason': 'escalation+suggestion matched',
            }
        return None

    async def add_task(self, project_root: str, **kwargs: Any) -> dict:
        # Extract metadata before forwarding — taskmaster's add_task doesn't accept it
        metadata = kwargs.pop('metadata', None)
        if metadata is not None:
            kwargs_for_candidate = dict(kwargs)
            kwargs_for_candidate['metadata'] = metadata
        else:
            kwargs_for_candidate = kwargs
        candidate = self._build_candidate(kwargs_for_candidate)
        project_id = resolve_project_id(project_root)

        async with self._project_lock(project_id):
            return await self._add_task_locked(
                project_root=project_root,
                project_id=project_id,
                candidate=candidate,
                metadata=metadata,
                kwargs=kwargs,
            )

    async def _add_task_locked(
        self,
        *,
        project_root: str,
        project_id: str,
        candidate,
        metadata,
        kwargs: dict,
    ) -> dict:
        # ── R4: escalation-level idempotency ─────────────────────────
        # When the caller stamps (escalation_id, suggestion_hash) into
        # metadata, walk existing tasks and skip the curator entirely if
        # a match is found. This covers the steward-timeout requeue case
        # (esc-1912-179 → esc-1912-190 on Type::Error) without an LLM
        # call or embedding lookup.
        idempotency_hit = await self._check_escalation_idempotency(
            project_root=project_root, metadata=metadata,
        )
        if idempotency_hit is not None:
            return idempotency_hit

        # ── Curator gate: drop / combine / create ────────────────────
        curator = await self._get_curator()
        if curator is not None and candidate is not None:
            decision = await curator.curate(candidate, project_id, project_root)

            if decision.action == 'drop' and decision.target_id:
                logger.warning(
                    'task_curator: drop — returning existing task %s instead of '
                    'creating duplicate: %s',
                    decision.target_id, candidate.title[:80],
                )
                return {
                    'id': decision.target_id,
                    'title': candidate.title,
                    'deduplicated': True,
                    'action': 'drop',
                    'justification': decision.justification,
                }

            if decision.action == 'combine' and decision.target_id:
                combine_result = await self._execute_combine(project_root, decision)
                if combine_result is not None:
                    logger.warning(
                        'task_curator: combine — folded candidate into task %s: %s',
                        decision.target_id, decision.justification[:120],
                    )
                    # Re-embed target so the corpus reflects the rewrite.
                    if decision.rewritten_task is not None:
                        rt = decision.rewritten_task
                        rt_candidate = CandidateTask(
                            title=rt.title,
                            description=rt.description,
                            details=rt.details,
                            files_to_modify=rt.files_to_modify,
                            priority=rt.priority,
                        )
                        bg = asyncio.create_task(
                            curator.reembed_task(
                                decision.target_id, rt_candidate, project_id,
                            ),
                            name=f'curator-reembed-{decision.target_id}',
                        )
                        self._background_tasks.add(bg)
                        bg.add_done_callback(
                            lambda t: self._background_tasks.discard(t),
                        )
                    return {
                        'id': decision.target_id,
                        'title': (
                            decision.rewritten_task.title
                            if decision.rewritten_task else candidate.title
                        ),
                        'deduplicated': True,
                        'action': 'combine',
                        'justification': decision.justification,
                    }
                # combine failed → fall through to create

            # action == 'create' or degenerate — proceed to Taskmaster

        # ── Create task ──────────────────────────────────────────────
        tm = await self._ensure_taskmaster()

        # Normalise metadata to a JSON string for taskmaster. Serialising here
        # keeps the MCP boundary (which demands a string) simple and preserves
        # the plain-dict shape for the fallback update_task path.
        metadata_json: str | None = None
        if metadata:
            metadata_json = (
                metadata if isinstance(metadata, str) else json.dumps(metadata)
            )

        # Atomic path: pass metadata in the initial add_task so the task is
        # never visible without its metadata (prevents the race that dropped
        # files_to_modify under concurrent load — see task #1922).
        try:
            result = await tm.add_task(
                project_root=project_root,
                metadata=metadata_json,
                **kwargs,
            )
            atomic_metadata_written = metadata_json is not None
        except TypeError:
            # Backwards-compat: pre-R5 taskmaster backends reject the new
            # ``metadata`` kwarg. Fall through to the legacy two-step path.
            result = await tm.add_task(project_root=project_root, **kwargs)
            atomic_metadata_written = False

        # Legacy fallback: follow-up update_task when the atomic write was
        # unavailable. Racy by construction; kept only for rollout safety
        # while both sides upgrade.
        task_id = None
        if isinstance(result, dict):
            task_id = str(result.get('id', ''))
        if metadata_json and task_id and not atomic_metadata_written:
            try:
                await tm.update_task(
                    task_id=task_id,
                    metadata=metadata_json,
                    project_root=project_root,
                )
            except Exception as e:
                logger.warning(
                    f'add_task: metadata update for task {task_id} failed: {e}',
                )

        # Record survivor in the curator. Two layers:
        # 1. ``note_created`` — synchronous, in-memory exact-match cache.
        #    MUST run inside the project lock so the *next* waiter on the
        #    same lock sees the new entry on its pre-LLM check (R3).
        # 2. ``record_task`` — awaited (not fire-and-forget) so the Qdrant
        #    corpus has the point before the lock releases, letting a
        #    near-miss follower's embedding lookup see it too.
        if curator is not None and candidate is not None and task_id:
            curator.note_created(project_id, candidate, task_id)
            try:
                await curator.record_task(task_id, candidate, project_id)
            except Exception:
                logger.warning(
                    'add_task: curator.record_task awaited path failed for %s',
                    task_id, exc_info=True,
                )

        event = self._make_event(
            EventType.task_created,
            project_root,
            {'operation': 'add_task', 'task_id': task_id},
        )
        await self.buffer.push(event)
        self._schedule_commit(project_root, 'add_task')
        return result

    async def update_task(
        self, task_id: str, project_root: str, **kwargs: Any,
    ) -> dict:
        tm = await self._ensure_taskmaster()
        result = await tm.update_task(
            task_id=task_id, project_root=project_root, **kwargs,
        )
        event = self._make_event(
            EventType.task_modified,
            project_root,
            {'task_id': task_id, 'operation': 'update_task'},
        )
        await self.buffer.push(event)
        self._schedule_commit(project_root, f'update_task({task_id})')

        # Re-embed if any corpus-relevant field changed. Taskmaster's update_task
        # accepts a free-form ``prompt`` for AI-driven edits, so we can't know
        # exactly what changed without re-fetching. Re-embed unconditionally when
        # the caller passed any of these hints.
        should_reembed = any(
            k in kwargs for k in ('prompt', 'title', 'description', 'details')
        )
        if should_reembed:
            curator = await self._get_curator()
            if curator is not None:
                try:
                    refreshed = await tm.get_task(task_id, project_root)
                    task_data = (
                        refreshed.get('data') if isinstance(refreshed, dict) else None
                    )
                    if isinstance(task_data, dict):
                        refreshed = task_data
                    if isinstance(refreshed, dict):
                        candidate = CandidateTask(
                            title=str(refreshed.get('title', '') or ''),
                            description=str(refreshed.get('description', '') or ''),
                            details=str(refreshed.get('details', '') or ''),
                            files_to_modify=[],
                            priority=str(refreshed.get('priority', 'medium')),
                        )
                        if candidate.title:
                            project_id = resolve_project_id(project_root)
                            bg = asyncio.create_task(
                                curator.reembed_task(
                                    task_id, candidate, project_id,
                                ),
                                name=f'curator-reembed-{task_id}',
                            )
                            self._background_tasks.add(bg)
                            bg.add_done_callback(
                                lambda t: self._background_tasks.discard(t),
                            )
                except Exception:
                    logger.debug(
                        'task_curator: reembed_task on update failed for %s',
                        task_id, exc_info=True,
                    )
        return result

    async def add_subtask(
        self, parent_id: str, project_root: str, **kwargs: Any,
    ) -> dict:
        candidate = self._build_candidate(kwargs)
        project_id = resolve_project_id(project_root)
        async with self._project_lock(project_id):
            return await self._add_subtask_locked(
                parent_id=parent_id,
                project_root=project_root,
                project_id=project_id,
                candidate=candidate,
                kwargs=kwargs,
            )

    async def _add_subtask_locked(
        self,
        *,
        parent_id: str,
        project_root: str,
        project_id: str,
        candidate,
        kwargs: dict,
    ) -> dict:
        # Curator gate for subtasks — previously bypassed entirely.
        curator = await self._get_curator()
        if curator is not None and candidate is not None:
            candidate.spawned_from = str(parent_id)
            candidate.spawn_context = candidate.spawn_context or 'manual'
            decision = await curator.curate(candidate, project_id, project_root)
            if decision.action == 'drop' and decision.target_id:
                logger.warning(
                    'task_curator: drop (subtask) — returning existing task %s '
                    'instead of creating duplicate: %s',
                    decision.target_id, candidate.title[:80],
                )
                return {
                    'id': decision.target_id,
                    'title': candidate.title,
                    'deduplicated': True,
                    'action': 'drop',
                    'justification': decision.justification,
                }
            if decision.action == 'combine' and decision.target_id:
                combine_result = await self._execute_combine(project_root, decision)
                if combine_result is not None:
                    logger.warning(
                        'task_curator: combine (subtask) — folded into task %s',
                        decision.target_id,
                    )
                    return {
                        'id': decision.target_id,
                        'title': (
                            decision.rewritten_task.title
                            if decision.rewritten_task else candidate.title
                        ),
                        'deduplicated': True,
                        'action': 'combine',
                        'justification': decision.justification,
                    }

        tm = await self._ensure_taskmaster()
        result = await tm.add_subtask(
            parent_id=parent_id, project_root=project_root, **kwargs,
        )
        event = self._make_event(
            EventType.task_created,
            project_root,
            {'parent_id': parent_id, 'operation': 'add_subtask'},
        )
        await self.buffer.push(event)
        self._schedule_commit(project_root, f'add_subtask({parent_id})')

        # Record the new subtask in the curator corpus (synchronous cache
        # update + awaited Qdrant upsert — see add_task for the rationale).
        if curator is not None and candidate is not None and isinstance(result, dict):
            new_id = str(result.get('id', ''))
            if new_id:
                curator.note_created(project_id, candidate, new_id)
                try:
                    await curator.record_task(new_id, candidate, project_id)
                except Exception:
                    logger.warning(
                        'add_subtask: curator.record_task awaited path failed for %s',
                        new_id, exc_info=True,
                    )
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
