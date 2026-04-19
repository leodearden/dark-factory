"""Intercepts task state transitions for targeted reconciliation."""

import asyncio
import json
import logging
import os
import uuid as uuid_mod
from datetime import UTC, datetime
from pathlib import Path
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
    from shared.usage_gate import UsageGate

    from fused_memory.config.schema import FusedMemoryConfig
    from fused_memory.middleware.curator_escalator import CuratorEscalator
    from fused_memory.middleware.task_file_committer import TaskFileCommitter
    from fused_memory.reconciliation.backlog_policy import BacklogPolicy
    from fused_memory.reconciliation.event_queue import EventQueue
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
        event_queue: 'EventQueue | None' = None,
        backlog_policy: 'BacklogPolicy | None' = None,
        usage_gate: 'UsageGate | None' = None,
    ):
        self.taskmaster = taskmaster
        self.reconciler = targeted_reconciler
        self.buffer = event_buffer
        # WP-B: when an event queue is wired in, journalling is fire-and-forget
        # via the background drainer. If None, fall back to inline buffer.push
        # — preserves the legacy call pattern for tests that haven't yet been
        # updated to construct a queue.
        self.event_queue = event_queue
        self.task_committer = task_committer
        # WP-D: bounded-backlog enforcement. Each mutating public method calls
        # ``_backlog_policy.check(project_id, project_root)`` before acquiring
        # the project lock; a rejection verdict short-circuits to a structured
        # error dict with no taskmaster mutation.
        self._backlog_policy = backlog_policy
        self._background_tasks: set[asyncio.Task] = set()

        # Task curator: LLM-judged drop/combine/create gate. Lazy-initialized in
        # _get_curator() because it pulls in a Qdrant client + embedder.
        self._config = config
        self._curator: TaskCurator | None = None
        self._escalator = escalator
        # Forwarded to ``TaskCurator`` for cap-aware LLM invocation across the
        # shared account pool. ``None`` falls back to the legacy single-shot
        # path with no cap retry — preserved for tests.
        self._usage_gate = usage_gate
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

    async def _backlog_gate(self, project_root: str) -> dict | None:
        """WP-D guard. Returns a structured error dict if the policy rejects;
        otherwise ``None`` and the caller proceeds.
        """
        if self._backlog_policy is None:
            return None
        project_id = resolve_project_id(project_root)
        verdict = await self._backlog_policy.check(project_id, project_root)
        if verdict.is_rejection:
            return verdict.to_error_dict()
        return None

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

    async def _journal(self, event: ReconciliationEvent) -> None:
        """Hand off an event for persistence.

        WP-B: when a fire-and-forget ``EventQueue`` is configured, enqueue
        synchronously and return immediately — the MCP hot path never
        awaits SQLite. Without a queue (legacy / test setups), fall back
        to an inline ``buffer.push``.
        """
        if self.event_queue is not None:
            self.event_queue.enqueue(event)
            return
        # Legacy inline push (no queue wired) — tests and degraded setups.
        buffer = self.buffer
        await buffer.push(event)

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
        done_provenance: dict | None = None,
    ) -> dict:
        """Proxy to Taskmaster, then fire-and-forget targeted reconciliation if triggered."""
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)
        # WP-E: hold the per-project lock across the read + no-op check +
        # write so two concurrent status transitions can't observe the
        # same before-state and both mutate tasks.json.
        resolved_provenance: dict | None = None
        async with self._project_lock(project_id):
            # 1. Get before-state
            before = await tm.get_task(task_id, project_root, tag)

            # 2. Same-status guard: no-op if nothing changed
            old_status = _extract_status(before)
            if status == old_status:
                return {'success': True, 'no_op': True, 'task_id': task_id}

            # 2b. Phantom-done gate: if transitioning to done and the task
            # advertises concrete files in metadata.files, refuse when any of
            # those files is absent at project_root. Catches set_task_status
            # calls that bypass the orchestrator's merge gate (reify tasks
            # 1746/1747/1749, 2026-04-19).
            if status == 'done':
                declared = _extract_metadata_files(before)
                if declared:
                    missing = _missing_files(project_root, declared)
                    if missing:
                        return _done_gate_error(task_id, declared, missing)

            # 2c. Done-provenance gate: require done_provenance={commit?, note?}
            # so Stage-2 reconciliation has verified evidence to reference
            # instead of fabricating 'shipped via X' edges from metadata.modules.
            if status == 'done':
                validation_err, resolved_provenance = await _validate_done_provenance(
                    task_id, done_provenance, project_root,
                    require=self._require_done_provenance(),
                )
                if validation_err is not None:
                    return validation_err
                if resolved_provenance is not None:
                    try:
                        await tm.update_task(
                            task_id=task_id,
                            metadata=json.dumps({'done_provenance': resolved_provenance}),
                            project_root=project_root,
                            tag=tag,
                        )
                    except Exception as e:
                        logger.warning(
                            'Failed to persist done_provenance for task %s: %s', task_id, e,
                        )

            # 3. Execute status change
            result = await tm.set_task_status(task_id, status, project_root, tag)

        # 5. Emit event
        payload: dict[str, Any] = {
            'task_id': task_id, 'old_status': old_status, 'new_status': status,
        }
        if resolved_provenance is not None:
            payload['done_provenance'] = resolved_provenance
        event = self._make_event(
            EventType.task_status_changed,
            project_root,
            payload,
        )
        await self._journal(event)
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
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)

        # WP-E: serialise the pre-snapshot + bulk mutation + commit so a
        # concurrent add_task can't slip a task into the snapshot gap and
        # get misclassified as newly-bulk-created.
        async with self._project_lock(project_id):
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
            await self._await_commit(project_root, f'expand_task({task_id})')
        event = self._make_event(
            EventType.tasks_bulk_created,
            project_root,
            {'parent_task_id': task_id, 'operation': 'expand_task'},
        )
        await self._journal(event)

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
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)

        # WP-E: serialise pre-snapshot + bulk mutation + commit; see
        # expand_task for the rationale.
        async with self._project_lock(project_id):
            # Pre-snapshot: capture the task tree before Taskmaster generates tasks
            try:
                pre_snapshot = await tm.get_tasks(project_root)
            except Exception as pre_exc:
                logger.warning('bulk_dedup: pre-snapshot failed for parse_prd: %s', pre_exc)
                pre_snapshot = None

            result = await tm.parse_prd(
                input_path, project_root, num_tasks=num_tasks, tag=tag
            )
            await self._await_commit(project_root, 'parse_prd')
        event = self._make_event(
            EventType.tasks_bulk_created,
            project_root,
            {'input_path': input_path, 'operation': 'parse_prd'},
        )
        await self._journal(event)

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
                usage_gate=self._usage_gate,
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

        Before writing, verifies the target's fingerprint and status. A
        mismatched fingerprint (curator targeted the wrong task) or terminal
        status (done/cancelled) aborts the write and returns None so the
        caller degrades to ``create`` instead of silently clobbering an
        unrelated task.
        """
        if decision.rewritten_task is None or decision.target_id is None:
            return None
        rt = decision.rewritten_task
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)

        # ── Guard: fetch live target and verify fingerprint + status ──
        try:
            raw_target = await tm.get_task(decision.target_id, project_root)
        except Exception as exc:
            logger.warning(
                'combine-guard: get_task failed for target=%s: %s — aborting combine',
                decision.target_id, exc,
            )
            return None

        target = _extract_task_dict(raw_target)
        if target is None:
            logger.warning(
                'combine-guard: target %s returned no task dict — aborting combine',
                decision.target_id,
            )
            return None

        target_status = str(target.get('status', '') or '')
        if target_status in {'done', 'cancelled'}:
            logger.warning(
                'combine-guard: target %s has terminal status %r — aborting '
                'combine to avoid silently losing candidate work',
                decision.target_id, target_status,
            )
            return None

        target_title = str(target.get('title', '') or '')
        expected_fp = _normalize_title(target_title)
        got_fp = _normalize_title(decision.target_fingerprint or '')
        if not got_fp or expected_fp != got_fp:
            logger.warning(
                'combine-guard: fingerprint mismatch for target=%s: '
                'expected=%r got=%r — aborting combine',
                decision.target_id,
                target_title[:80],
                (decision.target_fingerprint or '')[:80],
            )
            return None

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

        # ── Audit: persist old-vs-new BEFORE the mutation so we can
        # recover if the write crashes mid-flight.
        _append_combine_audit(
            project_id=project_id,
            target_id=decision.target_id,
            old_title=target_title,
            old_description=str(target.get('description', '') or ''),
            old_status=target_status,
            new_title=rt.title,
            new_description=rt.description,
            justification=decision.justification,
        )

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
                    # WP-E: serialise the tasks.json mutation.
                    async with self._project_lock(project_id):
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
                # WP-E: combine writes to the target *and* removes the new
                # task — hold the lock across both so no concurrent writer
                # can observe the intermediate "new still present" state.
                try:
                    async with self._project_lock(project_id):
                        combine_result = await self._execute_combine(project_root, decision)
                        if combine_result is None:
                            kept.append({'task_id': tid, 'title': title})
                            continue
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
        if err := await self._backlog_gate(project_root):
            return err
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
        await self._journal(event)
        self._schedule_commit(project_root, 'add_task')
        return result

    async def update_task(
        self, task_id: str, project_root: str, **kwargs: Any,
    ) -> dict:
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)
        # WP-E: serialise the write; re-embed below reads only and stays
        # outside the lock.
        async with self._project_lock(project_id):
            result = await tm.update_task(
                task_id=task_id, project_root=project_root, **kwargs,
            )
        event = self._make_event(
            EventType.task_modified,
            project_root,
            {'task_id': task_id, 'operation': 'update_task'},
        )
        await self._journal(event)
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
        if err := await self._backlog_gate(project_root):
            return err
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
        await self._journal(event)
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
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)
        # WP-E: serialise against concurrent mutations on the same project.
        async with self._project_lock(project_id):
            result = await tm.remove_task(task_id, project_root, tag)
        event = self._make_event(
            EventType.task_deleted,
            project_root,
            {'task_id': task_id, 'operation': 'remove_task'},
        )
        await self._journal(event)
        self._schedule_commit(project_root, f'remove_task({task_id})')
        return result

    async def add_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict:
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)
        # WP-E: serialise against concurrent mutations on the same project.
        async with self._project_lock(project_id):
            result = await tm.add_dependency(
                task_id, depends_on, project_root, tag
            )
        event = self._make_event(
            EventType.task_modified,
            project_root,
            {'task_id': task_id, 'depends_on': depends_on, 'operation': 'add_dependency'},
        )
        await self._journal(event)
        self._schedule_commit(project_root, f'add_dependency({task_id}<-{depends_on})')
        return result

    async def remove_dependency(
        self,
        task_id: str,
        depends_on: str,
        project_root: str,
        tag: str | None = None,
    ) -> dict:
        if err := await self._backlog_gate(project_root):
            return err
        tm = await self._ensure_taskmaster()
        project_id = resolve_project_id(project_root)
        # WP-E: serialise against concurrent mutations on the same project.
        async with self._project_lock(project_id):
            result = await tm.remove_dependency(
                task_id, depends_on, project_root, tag
            )
        event = self._make_event(
            EventType.task_modified,
            project_root,
            {'task_id': task_id, 'depends_on': depends_on, 'operation': 'remove_dependency'},
        )
        await self._journal(event)
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

    def _require_done_provenance(self) -> bool:
        """True when the provenance gate is enforcing (reject missing/invalid).

        False during phased rollout — in that mode a missing/malformed
        provenance payload logs a warning and the transition proceeds.
        """
        cfg = self._config
        if cfg is None:
            return False
        recon = getattr(cfg, 'reconciliation', None)
        return bool(getattr(recon, 'require_done_provenance', False))


def _extract_status(task_data: dict) -> str:
    """Extract status from Taskmaster get_task response."""
    if 'status' in task_data:
        return task_data['status']
    data = task_data.get('data', {})
    if isinstance(data, dict):
        return data.get('status', 'unknown')
    return 'unknown'


def _extract_metadata_files(task_data: Any) -> list[str]:
    """Return ``metadata.files`` as a list[str] from a Taskmaster get_task response.

    Handles the two common envelope shapes (bare task dict, or ``{'data': {...}}``
    wrapper). Silently returns ``[]`` when the field is absent, empty, or
    malformed — the phantom-done gate only fires when files is a non-empty
    list of strings, so defensive behaviour here means "do not gate".
    """
    inner = _extract_task_dict(task_data) or {}
    metadata = inner.get('metadata')
    if not isinstance(metadata, dict):
        return []
    files = metadata.get('files')
    if not isinstance(files, list):
        return []
    return [f for f in files if isinstance(f, str) and f]


def _missing_files(project_root: str, declared: list[str]) -> list[str]:
    """Return the subset of ``declared`` that does not exist under ``project_root``."""
    root = Path(project_root)
    return [f for f in declared if not (root / f).exists()]


def _done_gate_error(task_id: str, declared: list[str], missing: list[str]) -> dict:
    """Structured error returned when the phantom-done gate trips."""
    return {
        'success': False,
        'error': 'done_gate_missing_files',
        'task_id': task_id,
        'missing_files': missing,
        'files_checked': declared,
        'hint': (
            'Cannot mark task done: files listed in metadata.files do not exist '
            "at project_root. Either the implementation wasn't committed, or "
            'metadata.files is stale. Land the implementation (or fix '
            'metadata.files via update_task) before retrying.'
        ),
    }


async def _validate_done_provenance(
    task_id: str,
    raw: object,
    project_root: str,
    *,
    require: bool,
) -> tuple[dict | None, dict | None]:
    """Validate + resolve done_provenance for set_task_status(done).

    Returns ``(error_payload, resolved_provenance)``. Error payload is a
    structured dict suitable for returning to the MCP caller; when it is
    non-None the transition must be aborted. Resolved provenance is a dict
    with normalised ``commit`` (full 40-char SHA) and/or ``note`` keys, plus
    the original ``commit_input`` when a short hash or ref was resolved.

    When ``require`` is False, missing/empty provenance logs a warning but
    returns ``(None, None)`` so the transition proceeds. Malformed provenance
    (wrong type, unresolvable commit) still errors regardless of ``require``
    — we never want to record corrupt provenance on the task.
    """
    if raw is None or raw == {}:
        if require:
            return _done_provenance_missing_error(task_id), None
        logger.warning(
            'set_task_status(%s, done) called without done_provenance; '
            'Stage-2 reconciliation will treat this task as provenance-unknown. '
            'Pass done_provenance={"commit": "..."} or {"note": "..."} '
            'to record verified evidence.',
            task_id,
        )
        return None, None

    if not isinstance(raw, dict):
        return (
            _done_provenance_error(
                task_id,
                'done_provenance must be an object with keys "commit" and/or "note"',
            ),
            None,
        )

    commit_input = raw.get('commit')
    note = raw.get('note')

    if commit_input is not None and not isinstance(commit_input, str):
        return _done_provenance_error(task_id, 'commit must be a string'), None
    if note is not None and not isinstance(note, str):
        return _done_provenance_error(task_id, 'note must be a string'), None

    commit_input = (commit_input or '').strip() or None
    note = (note or '').strip() or None

    if commit_input is None and note is None:
        return (
            _done_provenance_error(
                task_id,
                'done_provenance requires at least one non-empty commit or note',
            ),
            None,
        )

    resolved: dict = {}
    if commit_input is not None:
        sha_or_err = await _resolve_commit_sha(project_root, commit_input)
        if isinstance(sha_or_err, dict):
            return _done_provenance_error(
                task_id,
                f'commit {commit_input!r} not found in {project_root}: '
                f'{sha_or_err.get("reason", "rev-parse failed")}',
            ), None
        resolved['commit'] = sha_or_err
        if sha_or_err != commit_input:
            resolved['commit_input'] = commit_input
    if note is not None:
        resolved['note'] = note

    return None, resolved


async def _resolve_commit_sha(project_root: str, commit: str) -> str | dict:
    """Resolve a tree-ish to a 40-char SHA via ``git rev-parse --verify``.

    Returns the SHA on success or a ``{'reason': ...}`` dict on failure.
    Uses a commit-peeling suffix (``^{commit}``) so a tag that points at a
    commit resolves to the commit SHA directly.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'git', '-C', project_root, 'rev-parse', '--verify',
            f'{commit}^{{commit}}',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except TimeoutError:
            proc.kill()
            return {'reason': 'git rev-parse timed out'}
    except FileNotFoundError:
        return {'reason': 'git binary not found'}
    except Exception as e:
        return {'reason': f'{type(e).__name__}: {e}'}

    if proc.returncode != 0:
        msg = (stderr.decode('utf-8', errors='replace') or '').strip() or 'no such ref'
        return {'reason': msg}
    sha = stdout.decode('utf-8', errors='replace').strip()
    if not sha or len(sha) < 7:
        return {'reason': 'empty rev-parse output'}
    return sha


def _done_provenance_missing_error(task_id: str) -> dict:
    return {
        'success': False,
        'error': 'done_provenance_required',
        'task_id': task_id,
        'hint': (
            'Cannot mark task done: done_provenance is required when '
            'reconciliation.require_done_provenance is enabled. Pass '
            'done_provenance={"commit": "<sha-or-ref>"} with the merge '
            'commit that landed the implementation, or '
            'done_provenance={"note": "<explanation>"} when no commit applies '
            '(fast-forward merge, covered by sibling task, interactive session).'
        ),
    }


def _done_provenance_error(task_id: str, reason: str) -> dict:
    return {
        'success': False,
        'error': 'done_provenance_invalid',
        'task_id': task_id,
        'reason': reason,
    }


def _extract_task_dict(raw: Any) -> dict | None:
    """Normalise a Taskmaster get_task response to the inner task dict.

    Taskmaster's MCP responses are sometimes wrapped in ``{'data': {...}}``;
    callers that care about the task's fields (title, description, status)
    need the unwrapped shape.
    """
    if not isinstance(raw, dict):
        return None
    data = raw.get('data')
    if isinstance(data, dict) and ('title' in data or 'status' in data):
        return data
    return raw


def _normalize_title(title: str) -> str:
    """Lowercase + collapse whitespace for forgiving title comparison.

    Used by the combine-guard fingerprint check — the LLM echoes the target's
    title verbatim, but accepting case/whitespace drift costs us nothing
    while avoiding false-negative aborts on trivial formatting noise.
    """
    return ' '.join(title.strip().lower().split())


def _combine_audit_path() -> Path:
    """Resolve the combine-audit log path.

    Honours ``DARK_FACTORY_DATA_DIR``; defaults to ``data/`` relative to the
    process CWD (which for the shared systemd server is the repo root).
    """
    return Path(os.getenv('DARK_FACTORY_DATA_DIR', 'data')) / 'combine_audit.jsonl'


def _append_combine_audit(
    *,
    project_id: str,
    target_id: str,
    old_title: str,
    old_description: str,
    old_status: str,
    new_title: str,
    new_description: str,
    justification: str,
) -> None:
    """Append a one-line JSON record documenting an about-to-happen combine.

    Append-only, best-effort. Failures log WARN but never propagate —
    a flaky audit write should not block task-merge progress.
    """
    record = {
        'ts': datetime.now(UTC).isoformat(),
        'project_id': project_id,
        'target_id': target_id,
        'curator_decision_id': str(uuid_mod.uuid4()),
        'old': {
            'title': old_title,
            'description_truncated': old_description[:500],
            'status': old_status,
        },
        'new': {
            'title': new_title,
            'description_truncated': new_description[:500],
        },
        'justification_truncated': justification[:500],
    }
    path = _combine_audit_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(record) + '\n')
    except Exception as exc:
        logger.warning(
            'combine-audit: failed to append audit record for target=%s: %s',
            target_id, exc,
        )
