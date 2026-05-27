"""Targeted reconciliation — lightweight, triggered by task state transitions."""

import contextlib
import json
import logging
import uuid as uuid_mod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.middleware.task_interceptor import (
    _extract_metadata_files,
    _missing_files,
    interceptor_write_succeeded,
)
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
from fused_memory.services.orchestrator_detector import is_orchestrator_live_for
from fused_memory.utils.validation import InputValidationError, require_project_root

# Defensive import — escalation is an optional workspace package (mirrors
# scope_violation_escalator.py:43-48).  When absent, L1 escalation degrades
# to a logged no-op; the sweep still cancels deterministic orphans and
# blocks ambiguous ones, so the dashboard still surfaces orphan state.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]
    _HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    Escalation = None  # type: ignore[assignment,misc]
    EscalationQueue = None  # type: ignore[assignment,misc]
    _HAS_ESCALATION = False

if TYPE_CHECKING:
    from fused_memory.backends.task_backend_protocol import TaskBackendProtocol
    from fused_memory.middleware.task_interceptor import TaskInterceptor
    from fused_memory.services.planned_episode_registry import PlannedEpisodeRegistry

logger = logging.getLogger(__name__)

# Reopen-reason prefix used by the cancellation sweep when it auto-cancels or
# auto-blocks a descendant of a cancelled parent.  Recursion guard: when the
# sweep's own writes fire `_on_task_cancelled` again on the descendant, the
# inner handler short-circuits via `reason.startswith(...)` so the sweep
# doesn't fan out a second time on each descendant.
_PARENT_CANCELLED_REOPEN_PREFIX = 'parent_cancelled:'

# Default location of the per-project escalation queue.  Matches the
# convention used by scope_violation_escalator.py and ticket_janitor.py.
_ESCALATION_QUEUE_DIRNAME = 'data/escalations'


class TargetedReconciler:
    """Lightweight reconciliation triggered by task state transitions."""

    def __init__(
        self,
        memory_service: MemoryService,
        taskmaster: 'TaskBackendProtocol',
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
        # Wire post-construction (mirroring the planned_episode_registry pattern) so that
        # _on_task_blocked routes metadata writes through the interceptor's write_lock.
        # Set by server/main.py after TaskInterceptor is constructed — see task 1136.
        self.task_interceptor: TaskInterceptor | None = None

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
        reopen_reason: str | None = None,
    ) -> dict:
        """Run targeted reconciliation for a single task state transition.

        ``reopen_reason`` is plumbed from ``TaskInterceptor._apply_status_transition``
        and consumed by the cancellation-sweep recursion guard: when the sweep
        auto-cancels a descendant with ``reopen_reason="parent_cancelled:<A>"``,
        the resulting transition fires this handler again on the descendant,
        and the guard short-circuits to prevent redundant re-sweep.
        """
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
                # Only _on_task_cancelled consumes reopen_reason today (sweep
                # recursion guard).  Threading it via **kwargs avoids touching
                # the other handlers' signatures.
                handler_kwargs: dict[str, Any] = {}
                if transition == 'cancelled':
                    handler_kwargs['reopen_reason'] = reopen_reason
                result = await handler(
                    task_id, project_id, project_root, task_before, run_id,
                    **handler_kwargs,
                )

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
                metadata_payload = json.dumps({'memory_hints': hints.model_dump()})
                # Route through TaskInterceptor when wired so the per-project
                # _write_lock is acquired around this metadata-only update — see task 1136.
                # Fall back to direct taskmaster call when interceptor is not set
                # (keeps existing unit-test fixtures working; the wiring-contract test
                # in TestServerWiringContract ensures production always wires correctly).
                resp: Any
                if self.task_interceptor is not None:
                    resp = await self.task_interceptor.update_task(
                        task_id=task_id,
                        metadata=metadata_payload,
                        project_root=project_root,
                    )
                else:
                    resp = await self.taskmaster.update_task(
                        task_id=task_id,
                        metadata=metadata_payload,
                        project_root=project_root,
                    )

                # TaskInterceptor gates (_backlog_gate, _reject_status_in_update_task,
                # _reject_done_provenance_in_update_metadata) return early with a dict
                # shaped {'success': False, 'error': '<code>', ...} — they do NOT raise.
                # The audit trail must not lie: only emit 'hints_attached' when the
                # write actually succeeded; emit 'hints_skipped' with diagnostics on
                # rejection so operators have a positive, queryable signal (task 1136).
                # Non-dict responses (e.g. None) are always failures — contract is
                # centralised in interceptor_write_succeeded (task 1184).
                write_succeeded = interceptor_write_succeeded(resp)

                if write_succeeded:
                    result['actions'].append({
                        'type': 'hints_attached',
                        'hints': hints.model_dump(),
                    })
                    await self.journal.add_run_action(
                        run_id, 'write', 'taskmaster', 'update_task',
                        {'task_id': task_id, 'type': 'hints_attached'},
                        causation_id=run_id,
                    )
                else:
                    # Prefer 'error_type' (stable machine-friendly code, e.g. BacklogVerdict)
                    # over 'error' (rendered human message) for audit-query aggregation — task 1215.
                    error_code = (resp.get('error_type') or resp.get('error')) if isinstance(resp, dict) else 'unknown'
                    reason = resp.get('reason') if isinstance(resp, dict) else None
                    result['actions'].append({
                        'type': 'hints_skipped',
                        'error': error_code,
                        'reason': reason,
                    })
                    # Symmetric durable audit row for the rejection path — mirrors the
                    # 'write' row emitted on success (above). Verb 'skip' lets operators
                    # filter `WHERE action_type='skip'` to compute rejection cardinality
                    # without inspecting detail JSON (task 1184).
                    await self.journal.add_run_action(
                        run_id, 'skip', 'taskmaster', 'update_task',
                        {
                            'task_id': task_id,
                            'type': 'hints_skipped',
                            'error': error_code,
                            'reason': reason,
                        },
                        causation_id=run_id,
                    )
                    logger.info(
                        f'Hints skipped for task {task_id}: '
                        f'update_task rejected with error={error_code!r} reason={reason!r}'
                    )
            except Exception as e:
                logger.warning(f'Failed to attach hints to task {task_id}: {e}')

        return result

    async def _on_task_cancelled(
        self,
        task_id: str,
        project_id: str,
        project_root: str,
        task_before: dict,
        run_id: str,
        *,
        reopen_reason: str | None = None,
    ) -> dict:
        """Task cancelled. Flag subtasks; sweep top-level dep-tree for orphans.

        The sweep classifies each non-terminal descendant (tasks where
        ``metadata.spawned_from == task_id`` OR ``task_id`` is in ``dependencies``)
        into one of three routes:

        1. **Cancel** — deterministic orphan: top-level review-followup of A
           (``spawned_from == A`` AND ``escalation_id`` set) whose declared
           ``metadata.files`` (if any) no longer exist at ``project_root``.
        2. **L1 escalate** — ambiguous descendant + orchestrator live for the
           project. A blocking ``scope_violation`` escalation is filed.
        3. **Block** — ambiguous descendant + orchestrator dead.
           ``set_task_status('blocked', reopen_reason='parent_cancelled:<A>;
           needs_recheck_against_main')`` + metadata.parent_cancelled = A.

        Subtasks (``parent_id == task_id``) remain flag-only per the
        pre-existing behaviour — out of scope for this sweep.

        Recursion guard: when the sweep's own cancel/block writes fire
        ``_on_task_cancelled`` again on the descendant, the inner call
        short-circuits before touching the dep-tree.  Detection is via
        ``reopen_reason.startswith('parent_cancelled:')`` — the sweep is
        the only writer that uses that prefix.
        """
        task = _extract_task(task_before)
        result: dict = {'task_id': task_id, 'actions': []}

        # Recursion guard: skip when this cancellation was itself triggered
        # by the sweep.  Without the guard, each cancel write would re-enter
        # the handler and walk the dep-tree again — quadratic for a chain.
        if (
            reopen_reason
            and reopen_reason.startswith(_PARENT_CANCELLED_REOPEN_PREFIX)
        ):
            return result

        # Flag subtasks for review (preserved behaviour — out of sweep scope).
        subtasks = task.get('subtasks', []) or []
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

        # Sweep top-level descendants — orphan review-followups + dependents.
        sweep_actions = await self._sweep_cancelled_descendants(
            parent_id=task_id,
            project_root=project_root,
            run_id=run_id,
        )
        result['actions'].extend(sweep_actions)

        return result

    async def _sweep_cancelled_descendants(
        self,
        *,
        parent_id: str,
        project_root: str,
        run_id: str,
    ) -> list[dict]:
        """Classify and route each non-terminal descendant of ``parent_id``.

        Returns the list of actions taken so the caller can surface them in
        the reconciliation result.  Failures are logged and swallowed so a
        single descendant's I/O glitch doesn't abandon the rest of the sweep.

        Requires ``self.task_interceptor`` to be wired (production wiring is
        verified by ``TestServerWiringContract``); without it the sweep
        no-ops because direct ``self.taskmaster`` writes would bypass the
        per-project write_lock and the terminal-exit / done-provenance gates.
        """
        actions: list[dict] = []
        if self.task_interceptor is None:
            logger.warning(
                'sweep: task_interceptor not wired; skipping descendant sweep '
                'for cancelled parent %s',
                parent_id,
            )
            return actions

        try:
            tasks_data = await self.taskmaster.get_tasks(project_root=project_root)
        except Exception as e:
            logger.warning(
                'sweep: get_tasks failed for cancelled parent %s: %s',
                parent_id, e,
            )
            return actions

        all_tasks = tasks_data.get('tasks', []) if isinstance(tasks_data, dict) else []
        if not isinstance(all_tasks, list):
            return actions

        orchestrator_live = is_orchestrator_live_for(project_root)
        parent_id_str = str(parent_id)

        # Lazy memo for the co-cancellation live re-check (task 1490).
        # Populated at most once per sweep, only when an ambiguous descendant
        # is encountered.  The initial get_tasks snapshot above (L548) may
        # predate sibling-cancel DB commits because targeted reconciliation is
        # fire-and-forget (task_interceptor.py:810-826) while user cancels are
        # awaited sequentially — so sibling DB writes land before this
        # background task re-reads.  A fresh re-read at decision time catches
        # co-cancelled siblings that the stale snapshot missed.
        #
        # Tradeoff: if _live_status_map raises and returns {}, the memo is set
        # to an empty dict, so all *subsequent* ambiguous descendants in this
        # sweep also skip the re-read and fall through to escalate/block.  This
        # is intentional: one transient failure short-circuits further attempts
        # to avoid a per-descendant retry storm.  The fail-open behaviour
        # (escalating/blocking rather than silently suppressing) preserves the
        # genuine-orphan detection contract on infrastructure glitches.
        live_status: dict[str, str] | None = None

        for t in all_tasks:
            if not isinstance(t, dict):
                continue
            tid = str(t.get('id', '') or '')
            if not tid or tid == parent_id_str:
                continue
            if str(t.get('status', '') or '') in ('done', 'cancelled'):
                continue

            metadata = t.get('metadata') if isinstance(t.get('metadata'), dict) else {}
            spawned_from = metadata.get('spawned_from') if isinstance(metadata, dict) else None
            spawned_from_str = (
                spawned_from if isinstance(spawned_from, str) and spawned_from else None
            )
            escalation_id_raw = (
                metadata.get('escalation_id') if isinstance(metadata, dict) else None
            )
            escalation_id = (
                escalation_id_raw
                if isinstance(escalation_id_raw, str) and escalation_id_raw
                else None
            )
            deps = [str(d) for d in (t.get('dependencies') or [])]

            is_spawn_from_parent = spawned_from_str == parent_id_str
            is_dependent = parent_id_str in deps
            if not is_spawn_from_parent and not is_dependent:
                continue  # leave: unrelated top-level task

            declared_files = _extract_metadata_files(t)
            missing = (
                _missing_files(project_root, declared_files) if declared_files else []
            )
            files_some_exist = bool(declared_files) and len(missing) < len(declared_files)

            # Deterministic orphan: a review-followup (spawned_from + escalation_id)
            # whose declared files don't exist on main.  Both signals must agree —
            # spawned_from alone is "ambiguous", and file-evidence on main means
            # the task may still be actionable.
            deterministic_orphan = (
                is_spawn_from_parent
                and escalation_id is not None
                and not files_some_exist
            )

            if deterministic_orphan:
                # No co-cancellation guard needed here: _sweep_cancel_orphan
                # emits a status transition to 'cancelled'.  If the dependent
                # was co-cancelled by the user between the initial snapshot and
                # now, the TaskInterceptor same-status guard
                # (task_interceptor.py:620) returns {'no_op': True}, making
                # this branch safely idempotent under the same race.
                action = await self._sweep_cancel_orphan(
                    task_id=tid, parent_id=parent_id_str, project_root=project_root,
                )
            else:
                # Ambiguous route (escalate when orch live; block when orch dead).
                # Before acting, do a lazy live re-read to detect co-cancelled
                # siblings whose cancel DB write landed after the initial snapshot.
                if live_status is None:
                    live_status = await self._live_status_map(project_root)
                if live_status.get(tid) == 'cancelled':
                    action = {
                        'type': 'descendant_skipped_co_cancelled',
                        'task_id': tid,
                        'parent_id': parent_id_str,
                    }
                elif orchestrator_live:
                    action = self._sweep_escalate_l1(
                        task_id=tid,
                        parent_id=parent_id_str,
                        escalation_id=escalation_id,
                        is_dependent=is_dependent,
                        project_root=project_root,
                    )
                else:
                    action = await self._sweep_block_orphan(
                        task_id=tid,
                        parent_id=parent_id_str,
                        escalation_id=escalation_id,
                        is_dependent=is_dependent,
                        project_root=project_root,
                    )

            if action is not None:
                actions.append(action)

        return actions

    async def _live_status_map(self, project_root: str) -> dict[str, str]:
        """Return a fresh {str(task_id): str(status)} map for the project.

        Called at most once per sweep (lazy-memoized by the caller) to detect
        co-cancelled siblings whose status DB write landed after the initial
        snapshot was taken.  Fails open: any exception logs a warning and
        returns {}, so the ambiguous branch falls through to the existing
        escalate/block path rather than silently dropping a genuine orphan.
        """
        try:
            tasks_data = await self.taskmaster.get_tasks(project_root=project_root)
        except Exception as e:
            logger.warning(
                'sweep: live status re-check failed for parent sweep at %s: %s',
                project_root, e,
            )
            return {}
        all_tasks = tasks_data.get('tasks', []) if isinstance(tasks_data, dict) else []
        if not isinstance(all_tasks, list):
            return {}
        return {
            str(t.get('id', '') or ''): str(t.get('status', '') or '')
            for t in all_tasks
            if isinstance(t, dict)
        }

    async def _sweep_cancel_orphan(
        self, *, task_id: str, parent_id: str, project_root: str,
    ) -> dict | None:
        """Auto-cancel a deterministic-orphan review-followup."""
        reason = f'{_PARENT_CANCELLED_REOPEN_PREFIX}{parent_id}'
        try:
            assert self.task_interceptor is not None  # narrowed by caller
            await self.task_interceptor.set_task_status(
                task_id=task_id,
                status='cancelled',
                project_root=project_root,
                reopen_reason=reason,
            )
        except Exception as e:
            logger.warning(
                'sweep: cancel failed for orphan %s (parent %s): %s',
                task_id, parent_id, e,
            )
            return None
        return {
            'type': 'descendant_cancelled',
            'task_id': task_id,
            'parent_id': parent_id,
            'reopen_reason': reason,
        }

    async def _sweep_block_orphan(
        self,
        *,
        task_id: str,
        parent_id: str,
        escalation_id: str | None,
        is_dependent: bool,
        project_root: str,
    ) -> dict | None:
        """Auto-block an ambiguous descendant + record parent_cancelled metadata.

        Two-write pattern: set_task_status('blocked') first, then update_task
        with ``append=True`` to merge parent_cancelled audit fields without
        clobbering existing metadata (memory_hints / files / spawned_from).
        Per feedback_set_task_status_replaces_metadata.md.
        """
        reason = (
            f'{_PARENT_CANCELLED_REOPEN_PREFIX}{parent_id}; needs_recheck_against_main'
        )
        try:
            assert self.task_interceptor is not None  # narrowed by caller
            await self.task_interceptor.set_task_status(
                task_id=task_id,
                status='blocked',
                project_root=project_root,
                reopen_reason=reason,
            )
        except Exception as e:
            logger.warning(
                'sweep: block (status) failed for descendant %s (parent %s): %s',
                task_id, parent_id, e,
            )
            return None

        meta: dict[str, Any] = {
            'parent_cancelled': parent_id,
            'needs_recheck_against_main': True,
            'parent_cancelled_relation': 'dependent' if is_dependent else 'spawned_from',
        }
        if escalation_id:
            meta['review_escalation_id'] = escalation_id

        try:
            await self.task_interceptor.update_task(
                task_id=task_id,
                project_root=project_root,
                metadata=json.dumps(meta),
                append=True,
            )
        except Exception as e:
            logger.warning(
                'sweep: block (metadata) failed for descendant %s (parent %s): %s',
                task_id, parent_id, e,
            )
        return {
            'type': 'descendant_blocked',
            'task_id': task_id,
            'parent_id': parent_id,
            'reopen_reason': reason,
        }

    def _sweep_escalate_l1(
        self,
        *,
        task_id: str,
        parent_id: str,
        escalation_id: str | None,
        is_dependent: bool,
        project_root: str,
    ) -> dict | None:
        """File an L1 escalation for an ambiguous descendant when orch is live."""
        # is-None narrows the optional types for pyright (mirrors
        # stage1_stall_detector.py:241).  HAS_ESCALATION is informational.
        if not _HAS_ESCALATION or Escalation is None or EscalationQueue is None:
            logger.debug(
                'sweep: escalation pkg unavailable; cannot file L1 for orphan %s',
                task_id,
            )
            return None
        relation = 'dependent' if is_dependent else 'review-followup'
        summary = (
            f'Task {task_id} orphaned: parent {parent_id} cancelled '
            f'({relation})'
        )
        detail_lines = [
            f'Parent task {parent_id} was cancelled.',
            f'This task is a {relation} of the cancelled parent; its scope likely '
            'references artifacts that only existed on the parent branch.',
            'Decide whether to expand_scope (re-scope against main) or abort_task '
            '(cancel as orphan).',
        ]
        if escalation_id:
            detail_lines.append(f'Originating review escalation: {escalation_id}')
        detail = '\n'.join(detail_lines)

        try:
            queue = EscalationQueue(Path(project_root) / _ESCALATION_QUEUE_DIRNAME)
            esc = Escalation(
                id=queue.make_id(task_id),
                task_id=task_id,
                agent_role='reconciler',
                severity='blocking',
                category='scope_violation',
                summary=summary,
                detail=detail,
                suggested_action='expand_scope|abort_task',
                level=1,
            )
            esc_id = queue.submit(esc)
        except Exception as e:
            logger.warning(
                'sweep: L1 escalate failed for orphan %s (parent %s): %s',
                task_id, parent_id, e,
            )
            return None
        return {
            'type': 'descendant_escalated',
            'task_id': task_id,
            'parent_id': parent_id,
            'escalation_id': esc_id,
        }

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
