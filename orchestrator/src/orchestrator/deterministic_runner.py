"""DeterministicRunner — orchestrator-side runner for deterministic gate tasks (β).

A *deterministic* task (``metadata.task_kind == 'deterministic'``) is routed
here by ``Harness._run_slot`` instead of ``TaskWorkflow``.  The runner holds
only ``scheduler`` + ``escalation_queue`` (no git_ops) — structurally proving
that no worktree, branch, agent, or steward is created for a gate (I4/B2).

Phase β delivers the **pure-gate** pattern
(``before_done=None``, ``always_escalates=True``):

1. **Idempotency / quiescence** (checked first):
   If ``metadata.gate_escalated_at`` is already set:
   - If a pending escalation still exists for the task → return BLOCKED (B3:
     no second escalation on quiescence).
   - Else (escalation resolved) → drive the task to ``done`` and return DONE
     (I2/B4/B11: resume path).
   NOTE (γ): the resume branch must also run ``before_done`` before setting done
   once task γ introduces before_done execution.  The NotImplementedError guard
   in the resume branch below ensures a misconfigured task cannot silently bypass
   before_done work.

2. **Pure gate** (``before_done=None``, ``always_escalates=True``):
   - File one born-at-L2 escalation (I3: in-process submit, sentinel role
     ``orchestrator-deterministic`` keeps level=2 past the server downgrade gate).
     Dedup: if a pending escalation already exists for the task (e.g. from a prior
     crash-safe re-dispatch), filing is skipped to avoid duplicate L2 escalations.
   - Stamp ``metadata.gate_escalated_at`` (crash-safe: file-before-stamp means a
     stamp failure re-files rather than silently skipping the gate).
   - Set task status to ``blocked``.
   - Return BLOCKED (B2).

3. **before_done present** → raise ``NotImplementedError`` (task γ delivers this).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from orchestrator.workflow import WorkflowOutcome

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)


class DeterministicRunner:
    """Per-slot runner for deterministic gate tasks.

    Constructed by ``Harness._run_deterministic_slot`` with only the minimal
    dependencies needed (no git_ops — provably no worktree creation).

    Args:
        scheduler: The orchestrator Scheduler instance.
        escalation_queue: The in-process EscalationQueue for filing L2 gates.

    Note (γ): task γ will extend this runner with event_store and before_done
    execution.  Adding event_store back then avoids a misleading dead dependency
    in β.
    """

    def __init__(self, scheduler, escalation_queue: EscalationQueue):
        self.scheduler = scheduler
        self.escalation_queue = escalation_queue

    async def run(self, assignment) -> WorkflowOutcome:
        """Execute the deterministic gate logic for *assignment*.

        Returns:
            WorkflowOutcome.DONE  — gate resolved, task driven to done (resume path).
            WorkflowOutcome.BLOCKED — gate filed or open escalation (initial/quiescence).

        Raises:
            NotImplementedError — if ``metadata.before_done`` is not None (task γ),
                either on the initial dispatch or on the resume path.
            ValueError — if ``always_escalates`` is False with ``before_done=None``
                (unsupported misconfiguration in β).
        """
        from escalation.models import Escalation

        task_id = str(assignment.task_id)
        task = assignment.task
        metadata = task.get('metadata') or {}
        title = task.get('title', '')
        description = task.get('description', '')
        before_done = metadata.get('before_done')
        always_escalates = metadata.get('always_escalates', False)
        gate_escalated_at = metadata.get('gate_escalated_at')

        # ── 1. Idempotency / quiescence ──────────────────────────────────────
        # If the gate escalation was already filed in a prior dispatch, check
        # whether it is still open or has been resolved.
        if gate_escalated_at:
            pending = self.escalation_queue.get_by_task(task_id, status='pending')
            if pending:
                # Escalation still open — quiescence (B3): return BLOCKED without
                # re-escalating.  The existing L2 is still awaiting human action.
                logger.debug(
                    'DeterministicRunner: task %s quiescent — %d pending escalation(s)',
                    task_id, len(pending),
                )
                return WorkflowOutcome.BLOCKED
            else:
                # Escalation resolved — drive to done (I2/B4/B11).
                # γ-guard: before_done execution on the resume path is task γ's scope.
                # Without this guard a task with before_done set would silently skip its
                # before_done work and be driven straight to done — bypassing the
                # NotImplementedError guard below (which is unreachable on the resume path).
                if before_done is not None:
                    raise NotImplementedError(
                        f'DeterministicRunner: before_done={before_done!r} is not '
                        'implemented on the resume path in β (task γ delivers this). '
                        f'Task id={task_id}.'
                    )
                logger.info(
                    'DeterministicRunner: task %s gate resolved — setting done',
                    task_id,
                )
                await self.scheduler.set_task_status(task_id, 'done')
                return WorkflowOutcome.DONE

        # ── 2. before_done guard ─────────────────────────────────────────────
        # β is pure-gate only; before_done execution is task γ's scope.
        if before_done is not None:
            raise NotImplementedError(
                f'DeterministicRunner: before_done={before_done!r} is not '
                'implemented in β (task γ delivers this). '
                f'Task id={task_id}.'
            )

        # ── 3. Pure gate ─────────────────────────────────────────────────────
        # Assertion: always_escalates must be True for a pure gate task.
        # (A non-escalating deterministic task with no before_done would be a
        # no-op — guard against misconfiguration loudly.)
        if not always_escalates:
            raise ValueError(
                f'DeterministicRunner: task {task_id} has before_done=None and '
                'always_escalates=False — this combination is not supported in β.'
            )

        # Build the escalation detail: description + dep IDs.
        deps = task.get('dependencies', [])
        dep_ids = [
            str(d.get('id', d) if isinstance(d, dict) else d) for d in deps
        ]
        detail_parts = [description]
        if dep_ids:
            detail_parts.append(f'\nLanded dependencies: {", ".join(dep_ids)}')
        detail = '\n'.join(detail_parts)

        gate_options = metadata.get('gate_options') or []

        # File the born-at-L2 escalation FIRST (crash-safe ordering: a stamp
        # failure on the following update_task re-files the gate on next dispatch
        # rather than silently skipping it).
        # Dedup guard: if a prior crash-safe re-dispatch already submitted the
        # escalation but failed before stamping gate_escalated_at, skip the
        # submit to avoid duplicate L2 escalations (human sees multiple gates).
        existing_pending = self.escalation_queue.get_by_task(task_id, status='pending')
        if existing_pending:
            logger.info(
                'DeterministicRunner: task %s already has %d pending escalation(s) — '
                'skipping re-file (gate_escalated_at stamp must have failed on prior dispatch)',
                task_id, len(existing_pending),
            )
        else:
            esc = Escalation(
                id=self.escalation_queue.make_id(task_id),
                task_id=task_id,
                agent_role='orchestrator-deterministic',
                severity='critical',
                category='milestone_gate',
                summary=title[:200],
                detail=detail,
                options=list(gate_options),
                level=2,
            )
            self.escalation_queue.submit(esc)
            logger.info(
                'DeterministicRunner: filed L2 milestone gate escalation %s for task %s',
                esc.id, task_id,
            )

        # Stamp gate_escalated_at AFTER successful escalation submit.
        now_iso = datetime.now(UTC).isoformat()
        await self.scheduler.update_task(
            task_id,
            {'gate_escalated_at': now_iso},
            metadata_mode='merge',
        )

        # Set status to blocked — gate awaits human decision.
        await self.scheduler.set_task_status(task_id, 'blocked')
        logger.info(
            'DeterministicRunner: task %s blocked at deterministic gate', task_id,
        )

        return WorkflowOutcome.BLOCKED
