"""DeterministicRunner — orchestrator-side runner for deterministic gate tasks (β/γ).

A *deterministic* task (``metadata.task_kind == 'deterministic'``) is routed
here by ``Harness._run_slot`` instead of ``TaskWorkflow``.  The runner holds
only ``scheduler`` + ``escalation_queue`` (no git_ops) — structurally proving
that no worktree, branch, agent, or steward is created for a gate (I4/B2).

Phase β delivers the **pure-gate** pattern
(``before_done=None``, ``always_escalates=True``).

Phase γ adds the **before_done blocking cross-unit deploy** path
(``before_done`` is a dict, ``always_escalates=False``):

1. **Idempotency / quiescence** (checked first):
   If ``metadata.gate_escalated_at`` is already set:
   - If a pending escalation still exists for the task → return BLOCKED (B3:
     no second escalation on quiescence).
   - Else (escalation resolved) → drive the task to ``done`` and return DONE
     (I2/B4/B11: resume path).

   If ``metadata.before_done_ran_at`` is already set (γ):
   - If a pending infra_issue escalation exists → return BLOCKED (B7: reaper
     no-rerun, I1 once-only).
   - Else (escalation resolved) → drive to ``done`` and return DONE (resume
     after human resolution, no re-run).

2. **before_done execution** (γ: ``before_done`` is not None):
   - Stamp ``metadata.before_done_ran_at`` FIRST (crash-safe I1: never re-run
     the deploy if we crash mid-flight).
   - Capture baseline unit state (``unit_inspector``).
   - Run the deploy script to completion (``script_runner``, blocking).
   - If ``rc != 0``: file born-at-L2 ``infra_issue`` escalation, set blocked
     (B7a).
   - Re-inspect and verify a fresh ``MainPID`` (>0, non-sentinel) and a
     strictly-later ``ActiveEnterTimestampMonotonic`` (B7b).
   - If ``always_escalates=False``: set task to ``done`` with
     ``done_provenance.kind='deterministic-deploy'`` carrying the fresh PID
     and timestamp (B6); return DONE.
   - If ``always_escalates=True``: fall through to gate (act-then-ask).

3. **Pure gate** (``before_done=None``, ``always_escalates=True``):
   - File one born-at-L2 escalation (I3: in-process submit, sentinel role
     ``orchestrator-deterministic`` keeps level=2 past the server downgrade gate).
     Dedup: if a pending escalation already exists for the task (e.g. from a prior
     crash-safe re-dispatch), filing is skipped to avoid duplicate L2 escalations.
   - Stamp ``metadata.gate_escalated_at`` (crash-safe: file-before-stamp means a
     stamp failure re-files rather than silently skipping the gate).
   - Set task status to ``blocked``.
   - Return BLOCKED (B2).
"""

from __future__ import annotations

import asyncio
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
        unit_inspector: Optional callable ``(unit: str) -> dict`` returning
            ``{MainPID, ActiveState, ActiveEnterTimestamp,
            ActiveEnterTimestampMonotonic}`` for the given systemd unit.
            Defaults to ``_default_inspect_unit`` (systemctl --user show).
            Injected in tests to avoid touching real systemd.
        script_runner: Optional callable ``(before_done: dict) -> (rc, tail)``
            that runs the deploy script to completion.  Defaults to
            ``_default_run_script`` (awaited create_subprocess_exec).
            Injected in tests to avoid spawning real processes.
    """

    def __init__(
        self,
        scheduler,
        escalation_queue: EscalationQueue,
        unit_inspector=None,
        script_runner=None,
    ):
        self.scheduler = scheduler
        self.escalation_queue = escalation_queue
        self._unit_inspector = unit_inspector
        self._script_runner = script_runner

    # ------------------------------------------------------------------
    # Default injectable seam implementations
    # ------------------------------------------------------------------

    async def _default_inspect_unit(self, unit: str) -> dict:
        """Query systemctl for unit state fields needed for fresh-PID verify.

        Returns a dict with at minimum: MainPID (int), ActiveState (str),
        ActiveEnterTimestamp (str), ActiveEnterTimestampMonotonic (int).
        Integers default to 0 on parse failure (sentinel-safe).
        """
        proc = await asyncio.create_subprocess_exec(
            'systemctl', '--user', 'show', unit,
            '-p', 'MainPID,ActiveState,ActiveEnterTimestamp,ActiveEnterTimestampMonotonic',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await proc.communicate()
        result: dict = {}
        for line in (stdout or b'').decode(errors='replace').splitlines():
            if '=' in line:
                key, _, val = line.partition('=')
                result[key.strip()] = val.strip()
        # Coerce numeric fields — a missing / unparseable value is treated as 0 (sentinel)
        for field in ('MainPID', 'ActiveEnterTimestampMonotonic'):
            try:
                result[field] = int(result.get(field, 0))
            except (TypeError, ValueError):
                result[field] = 0
        return result

    async def _default_run_script(self, before_done: dict) -> tuple[int, str]:
        """Run the deploy script to completion under a timeout.

        Adapts ``service_restart.py``'s spawn pattern but AWAITS completion
        (blocking cross-unit deploy — no self-kill risk on this path).

        Returns:
            (rc, output_tail) — rc is the process return code; output_tail is
            the last 2000 chars of combined stdout/stderr.
        """
        script = before_done['script']
        args = before_done.get('args') or []
        env = before_done.get('env') or None  # empty dict → None (inherit env)
        cwd = before_done.get('cwd') or None
        timeout_secs = before_done.get('timeout_secs', 60)

        proc = await asyncio.create_subprocess_exec(
            script, *args,
            env=env,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_secs)
            tail = (stdout or b'').decode(errors='replace')[-2000:]
            return proc.returncode or 0, tail
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 1, f'<script timed out after {timeout_secs}s>'

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

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

        # ── 2. before_done execution (γ) ────────────────────────────────────
        # Cross-unit blocking deploy: stamp → baseline → run script → verify → done.
        # Self-target detection + detached systemd-run is deferred to ε.
        if before_done is not None:
            target_unit: str = before_done.get('target_unit', '')

            # Stamp before_done_ran_at FIRST (crash-safe I1: stamp-before-run means a
            # crash mid-deploy leaves the stamp set → re-dispatch does NOT re-run).
            now_iso = datetime.now(UTC).isoformat()
            await self.scheduler.update_task(
                task_id,
                {'before_done_ran_at': now_iso},
                metadata_mode='merge',
            )

            # Capture baseline unit state before the deploy fires
            inspect_fn = self._unit_inspector or self._default_inspect_unit
            baseline = await inspect_fn(target_unit)

            # Run the deploy script to completion (blocking, cross-unit)
            run_fn = self._script_runner or self._default_run_script
            rc, out = await run_fn(before_done)

            if rc != 0:
                # B7a: script failed — file infra_issue escalation (step-4)
                raise NotImplementedError(
                    f'DeterministicRunner: rc={rc} failure handling for before_done '
                    'is task γ step-4. '
                    f'Task id={task_id}.'
                )

            # Re-inspect to verify a fresh MainPID + strictly-later monotonic timestamp
            new_state = await inspect_fn(target_unit)
            pid: int = new_state.get('MainPID', 0)
            new_monotonic: int = new_state.get('ActiveEnterTimestampMonotonic', 0)
            baseline_monotonic: int = baseline.get('ActiveEnterTimestampMonotonic', 0)
            fresh: bool = (
                isinstance(pid, int)
                and pid > 0
                and new_monotonic > baseline_monotonic
            )

            if not fresh:
                # B7b: stale/missing PID — file infra_issue escalation (step-6)
                raise NotImplementedError(
                    'DeterministicRunner: stale/missing PID verify for before_done '
                    'is task γ step-6. '
                    f'Task id={task_id}. pid={pid!r} new_monotonic={new_monotonic} '
                    f'baseline_monotonic={baseline_monotonic}.'
                )

            if not always_escalates:
                # Pure cross-unit deploy (B6): verified → set done with provenance
                logger.info(
                    'DeterministicRunner: task %s before_done deploy verified — '
                    'pid=%d unit=%s — setting done',
                    task_id, pid, target_unit,
                )
                await self.scheduler.set_task_status(
                    task_id,
                    'done',
                    done_provenance={
                        'kind': 'deterministic-deploy',
                        'pid': pid,
                        'active_enter_timestamp': new_state.get('ActiveEnterTimestamp', ''),
                        'unit': target_unit,
                    },
                )
                return WorkflowOutcome.DONE

            # always_escalates=True with before_done: action already ran — fall through
            # to the gate below (act-then-ask; γ scope = always_escalates=False only,
            # but the structure is left consistent so act-then-ask does not crash).
            logger.info(
                'DeterministicRunner: task %s before_done ran with always_escalates=True '
                '— falling through to gate',
                task_id,
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
