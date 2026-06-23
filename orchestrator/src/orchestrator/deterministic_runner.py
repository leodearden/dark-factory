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
   - Else (no pending escalation) → require POSITIVE proof of a terminal
     outcome before completing (phantom-done guard for the crash window
     between stamping ``before_done_ran_at`` and recording a terminal result):
     * ``before_done_verified_at`` set → verify passed (crash before the done
       write) → drive to ``done``.
     * an escalation was filed and resolved → human acted (act-then-ask) →
       drive to ``done``, no re-run.
     * neither → crash mid-deploy before any terminal decision → re-escalate
       (infra_issue, blocked); never phantom-done, never re-run (I1).

2. **before_done execution** (γ: ``before_done`` is not None):
   - Stamp ``metadata.before_done_ran_at`` FIRST (crash-safe I1: never re-run
     the deploy if we crash mid-flight).
   - Capture baseline unit state (``unit_inspector``).
   - Run the deploy script to completion (``script_runner``, blocking).
   - If ``rc != 0``: file born-at-L2 ``infra_issue`` escalation, set blocked
     (B7a).
   - Re-inspect and verify a fresh ``MainPID`` (>0, non-sentinel) and a
     strictly-later ``ActiveEnterTimestampMonotonic`` (B7b).
   - If ``always_escalates=False``: stamp ``before_done_verified_at`` (the
     positive proof the resume path requires), then set task to ``done`` with
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
import os
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
        own_unit_resolver=None,
        restart_scheduler=None,
    ):
        self.scheduler = scheduler
        self.escalation_queue = escalation_queue
        self._unit_inspector = unit_inspector
        self._script_runner = script_runner
        self._own_unit_resolver = own_unit_resolver
        self._restart_scheduler = restart_scheduler

    # ------------------------------------------------------------------
    # Default injectable seam implementations
    # ------------------------------------------------------------------

    def _default_resolve_own_unit(self) -> str:
        """Return the orchestrator's own systemd unit name from ORCH_UNIT env var.

        Returns an empty string if ORCH_UNIT is not set (fail-open to cross-unit
        path so existing CI tests with ORCH_UNIT unset stay on the cross-unit path).
        Operators set ORCH_UNIT in the [Service] Environment of the orchestrator unit.
        """
        return os.environ.get('ORCH_UNIT', '')

    async def _default_schedule_detached_restart(
        self,
        before_done: dict,
        *,
        transient_unit: str,
        on_active_secs: int,
        task_id: str,
        summary: str = '',
    ) -> tuple[int, str]:
        """Schedule a detached systemd-run transient unit for a self-restart.

        Uses two systemd-run registrations:
        1. A companion failure-handler transient unit that runs δ's escalation-
           submit CLI if the restart fires and fails.
        2. The main restart transient unit (--on-active) wired with
           ``--property=OnFailure=<failure-unit>`` so systemd routes fire-time
           failures to the handler.

        systemd-run returns immediately after registering the transient unit;
        the orchestrator is NEVER blocked or killed — the payload fires later
        under the user systemd manager.

        Returns:
            (rc, tail) — rc=0 on successful registration; rc≠0 if either
            registration fails (tail carries the error output).
        """
        import sys

        queue_dir = str(self.escalation_queue.queue_dir)
        target_unit = before_done.get('target_unit', 'unknown')
        script = before_done['script']
        args = before_done.get('args') or []

        # Derive companion failure-handler unit name from the transient unit name
        if transient_unit.endswith('.service'):
            failure_unit = transient_unit[:-len('.service')] + '-failure.service'
        else:
            failure_unit = transient_unit + '-failure.service'

        esc_summary = summary or (
            f'Self-restart fire-time failure: {target_unit}'
        )

        # ── 1. Register the OnFailure handler unit ───────────────────────────
        # Runs δ's escalation-submit CLI if the restart fails at fire time.
        # sys.executable → python -m escalation submit (robust against PATH in
        # the detached systemd user environment).
        failure_argv = [
            'systemd-run', '--user',
            f'--unit={failure_unit}',
            '--remain-after-exit',
            '--collect',
            sys.executable, '-m', 'escalation', 'submit',
            '--queue-dir', queue_dir,
            '--task', task_id,
            '--severity', 'critical',
            '--category', 'infra_issue',
            '--summary', esc_summary[:200],
        ]
        proc1 = await asyncio.create_subprocess_exec(
            *failure_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout1, _ = await proc1.communicate()
        rc1 = proc1.returncode or 0
        if rc1 != 0:
            tail1 = (stdout1 or b'').decode(errors='replace')[-2000:]
            logger.warning(
                'DeterministicRunner: failed to register OnFailure handler unit %s '
                '(rc=%d) for task %s',
                failure_unit, rc1, task_id,
            )
            return rc1, tail1

        # ── 2. Register the main restart transient unit ──────────────────────
        # --on-active=<N>: fires N seconds after this run() returns (manifest §53).
        # --property=OnFailure=: wires δ's failure-handler unit.
        main_argv = [
            'systemd-run', '--user',
            f'--on-active={on_active_secs}',
            f'--unit={transient_unit}',
            '--collect',
            f'--property=OnFailure={failure_unit}',
            script, *args,
        ]
        proc2 = await asyncio.create_subprocess_exec(
            *main_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout2, _ = await proc2.communicate()
        rc2 = proc2.returncode or 0
        tail2 = (stdout2 or b'').decode(errors='replace')[-2000:]
        if rc2 != 0:
            logger.warning(
                'DeterministicRunner: failed to register restart transient unit %s '
                '(rc=%d) for task %s',
                transient_unit, rc2, task_id,
            )
        return rc2, tail2

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
        # Merge over os.environ so the child sees a full environment (PATH, HOME,
        # XDG_RUNTIME_DIR …).  An empty / absent env dict means full inherit.
        env = {**os.environ, **before_done['env']} if before_done.get('env') else None
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
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return 1, f'<script timed out after {timeout_secs}s>'

    async def _file_infra_issue_and_block(
        self,
        task_id: str,
        summary: str,
        detail: str,
    ) -> WorkflowOutcome:
        """File a born-at-L2 infra_issue escalation and set the task to blocked.

        Reuses β's escalation construction pattern (sentinel role keeps level=2
        past the server downgrade gate).  Includes a dedup guard — if a pending
        escalation already exists (e.g. prior crash-safe re-dispatch), filing is
        skipped to avoid duplicate L2 escalations.

        Returns:
            WorkflowOutcome.BLOCKED
        """
        from escalation.models import Escalation

        existing_pending = self.escalation_queue.get_by_task(task_id, status='pending')
        if existing_pending:
            logger.info(
                'DeterministicRunner: task %s already has %d pending escalation(s) — '
                'skipping re-file (infra_issue dedup guard)',
                task_id, len(existing_pending),
            )
        else:
            esc = Escalation(
                id=self.escalation_queue.make_id(task_id),
                task_id=task_id,
                agent_role='orchestrator-deterministic',
                severity='critical',
                category='infra_issue',
                summary=summary[:200],
                detail=detail,
                level=2,
            )
            self.escalation_queue.submit(esc)
            logger.info(
                'DeterministicRunner: filed L2 infra_issue escalation %s for task %s',
                esc.id, task_id,
            )

        await self.scheduler.set_task_status(task_id, 'blocked')
        logger.info('DeterministicRunner: task %s blocked — infra_issue', task_id)
        return WorkflowOutcome.BLOCKED

    # ------------------------------------------------------------------
    # Main runner
    # ------------------------------------------------------------------

    async def run(self, assignment) -> WorkflowOutcome:
        """Execute the deterministic gate logic for *assignment*.

        Returns:
            WorkflowOutcome.DONE  — gate resolved, task driven to done.
            WorkflowOutcome.BLOCKED — gate filed, open escalation, or deploy failure.

        Raises:
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
                # γ: if before_done is set, the action must have already ran (I1) for us
                # to safely drive to done here.  Check before_done_ran_at as proof.
                if before_done is not None:
                    before_done_ran_at_check = metadata.get('before_done_ran_at')
                    if not before_done_ran_at_check:
                        # Gate resolved but before_done never ran — unexpected state.
                        # Conservatively raise so the operator can investigate; this path
                        # should not occur in normal operation once γ ships.
                        raise NotImplementedError(
                            f'DeterministicRunner: gate resolved but before_done_ran_at '
                            f'is not set — cannot safely drive to done without proof the '
                            f'action ran.  Task id={task_id}.'
                        )
                    # before_done already ran (I1) — proceed to done (act-then-ask resume)
                    logger.info(
                        'DeterministicRunner: task %s act-then-ask resume — '
                        'before_done_ran_at=%s, gate resolved — setting done',
                        task_id, before_done_ran_at_check,
                    )
                logger.info(
                    'DeterministicRunner: task %s gate resolved — setting done',
                    task_id,
                )
                if before_done is not None:
                    # Act-then-ask resume: include deploy provenance so the audit trail
                    # matches the B6 / resume paths and passes require_done_provenance.
                    _ata_unit = before_done.get('target_unit', '')
                    await self.scheduler.set_task_status(
                        task_id,
                        'done',
                        done_provenance={
                            'kind': 'deterministic-deploy',
                            'unit': _ata_unit,
                            'note': 'resumed after gate resolution',
                        },
                    )
                else:
                    await self.scheduler.set_task_status(task_id, 'done')
                return WorkflowOutcome.DONE

        # ── 2. before_done execution (γ) ────────────────────────────────────
        # Cross-unit blocking deploy: stamp → baseline → run script → verify → done.
        # Self-target detection + detached systemd-run is deferred to ε.
        if before_done is not None:
            target_unit: str = before_done.get('target_unit', '')
            before_done_ran_at = metadata.get('before_done_ran_at')

            # I1 once-only idempotency guard (parallel to β's gate_escalated_at branch):
            # if the deploy already ran, check whether its escalation is still open.
            if before_done_ran_at:
                pending = self.escalation_queue.get_by_task(task_id, status='pending')
                if pending:
                    # Pending infra_issue → quiescent BLOCKED (B7 reaper / I1 no-rerun)
                    logger.debug(
                        'DeterministicRunner: task %s before_done already ran, '
                        'pending escalation — quiescent BLOCKED (B7/I1)',
                        task_id,
                    )
                    return WorkflowOutcome.BLOCKED

                # No pending escalation.  Before driving to done we MUST have
                # POSITIVE proof the deploy reached a terminal decision —
                # otherwise a crash in the window between stamping
                # ``before_done_ran_at`` (above) and recording a terminal outcome
                # (verify-success stamp, or the failure escalation) lands us here
                # with neither verification nor a human in the loop, producing a
                # phantom-done.  The bare "stamp set + empty queue → done" rule
                # conflated 'crashed mid-deploy' with 'human resolved escalation'.
                # Three sub-states are distinguished:
                #   (a) before_done_verified_at set → the fresh-PID check passed
                #       (crash between the verify stamp and the done write) →
                #       safe to drive to done.
                #   (b) an escalation was filed for this task and later resolved →
                #       a human acted on the failure (act-then-ask resume) →
                #       drive to done, no re-run (I1).
                #   (c) neither → crash mid-deploy before any terminal decision →
                #       re-escalate.  NEVER phantom-done; NEVER re-run (I1).
                before_done_verified_at = metadata.get('before_done_verified_at')
                # status=None scans the archive too → detects a resolved/dismissed
                # escalation, i.e. proof a human was in the loop on a prior failure.
                ever_escalated = bool(self.escalation_queue.get_by_task(task_id))

                if before_done_verified_at:
                    # (a) Deploy verified OK; crash before the done write.
                    logger.info(
                        'DeterministicRunner: task %s before_done verified (%s) — '
                        'resume after crash-before-done-write, setting done',
                        task_id, before_done_verified_at,
                    )
                    await self.scheduler.set_task_status(
                        task_id,
                        'done',
                        done_provenance={
                            'kind': 'deterministic-deploy',
                            'pid': metadata.get('before_done_verified_pid'),
                            'note': 'resumed after verified deploy (crash before done write)',
                            'unit': target_unit,
                        },
                    )
                    return WorkflowOutcome.DONE

                if ever_escalated:
                    # (b) A failure escalation was filed and resolved by a human.
                    logger.info(
                        'DeterministicRunner: task %s before_done ran + escalation '
                        'resolved — resume-after-resolution, setting done (no re-run)',
                        task_id,
                    )
                    await self.scheduler.set_task_status(
                        task_id,
                        'done',
                        done_provenance={
                            'kind': 'deterministic-deploy',
                            'note': 'resumed after human resolution',
                            'unit': target_unit,
                        },
                    )
                    return WorkflowOutcome.DONE

                # (c) Crash window: stamped but never verified and never escalated.
                # Re-escalate instead of phantom-completing; the deploy is NOT
                # re-run (I1 once-only) — a human must verify the unit state.
                logger.warning(
                    'DeterministicRunner: task %s before_done_ran_at set but neither '
                    'verified nor escalated — crash-window detected; re-escalating '
                    'instead of phantom-done',
                    task_id,
                )
                crash_detail = '\n'.join([
                    description,
                    f'Target unit: {target_unit}',
                    'before_done_ran_at is stamped but the deploy recorded neither a '
                    'verification (before_done_verified_at unset) nor a failure '
                    'escalation — the orchestrator crashed mid-deploy between '
                    'stamping and completing.  The deploy is NOT re-run (I1 '
                    'once-only); a human must inspect the unit and resolve.',
                ])
                return await self._file_infra_issue_and_block(
                    task_id,
                    summary=f'Deploy state unknown after crash: {target_unit}',
                    detail=crash_detail,
                )

            # Stamp before_done_ran_at FIRST (crash-safe I1: stamp-before-run means a
            # crash mid-deploy leaves the stamp set → re-dispatch does NOT re-run).
            # This stamp is SHARED for both self-target and cross-unit paths so I1
            # holds for both (ε design decision 5).
            now_iso = datetime.now(UTC).isoformat()
            await self.scheduler.update_task(
                task_id,
                {'before_done_ran_at': now_iso},
                metadata_mode='merge',
            )

            # ── ε: self-target detection ─────────────────────────────────────
            # If target_unit IS the orchestrator's own unit, running the blocking
            # cross-unit deploy would kill this runner mid-execution (self-kill risk).
            # Instead, schedule a detached transient unit via systemd-run so the
            # restart fires AFTER run() returns, and set the task done immediately
            # with kind='deterministic-deploy-scheduled' (done = scheduled, not
            # verified).  See PRD §3, §4 decisions 8/9.
            own_unit = (self._own_unit_resolver or self._default_resolve_own_unit)()
            self_target = bool(own_unit) and (target_unit == own_unit)
            if self_target:
                transient_unit = f'orch-redeploy-restart-{task_id}.service'
                on_active_secs = int(before_done.get('on_active_delay_secs', 60))
                restart_fn = self._restart_scheduler or self._default_schedule_detached_restart
                rc, tail = await restart_fn(
                    before_done,
                    transient_unit=transient_unit,
                    on_active_secs=on_active_secs,
                    task_id=task_id,
                    summary=f'Self-restart scheduling failed: {target_unit}',
                )
                if rc != 0:
                    detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Transient unit: {transient_unit}',
                        f'systemd-run exit code: rc={rc}',
                        f'Output:\n{tail}',
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Self-restart scheduling failed: {target_unit}',
                        detail=detail,
                    )
                if not always_escalates:
                    logger.info(
                        'DeterministicRunner: task %s self-restart scheduled — '
                        'transient_unit=%s on_active_secs=%d — setting done',
                        task_id, transient_unit, on_active_secs,
                    )
                    await self.scheduler.set_task_status(
                        task_id,
                        'done',
                        done_provenance={
                            'kind': 'deterministic-deploy-scheduled',
                            'unit': target_unit,
                            'transient_unit': transient_unit,
                            'fire_delay_secs': on_active_secs,
                        },
                    )
                    return WorkflowOutcome.DONE
                # always_escalates=True (act-then-ask, non-exemplar): fall through
                # to the gate below (mirrors γ line 495 behaviour).
                logger.info(
                    'DeterministicRunner: task %s self-restart scheduled with '
                    'always_escalates=True — falling through to gate',
                    task_id,
                )

            # ── end ε self-target branch ─────────────────────────────────────
            # Capture baseline unit state before the deploy fires
            inspect_fn = self._unit_inspector or self._default_inspect_unit
            baseline = await inspect_fn(target_unit)

            # Run the deploy script to completion (blocking, cross-unit)
            run_fn = self._script_runner or self._default_run_script
            rc, out = await run_fn(before_done)

            if rc != 0:
                # B7a: script failed — file infra_issue escalation, set blocked (B7a)
                deploy_detail = '\n'.join([
                    description,
                    f'Target unit: {target_unit}',
                    f'Script exit code: rc={rc}',
                    f'Output:\n{out}',
                ])
                return await self._file_infra_issue_and_block(
                    task_id,
                    summary=f'Deploy failed: {target_unit}',
                    detail=deploy_detail,
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
                # B7b: verify failed — file infra_issue escalation, set blocked
                verify_detail = '\n'.join([
                    description,
                    f'Target unit: {target_unit}',
                    (
                        f'Verify failed: new MainPID={pid!r} '
                        f'new_monotonic={new_monotonic} '
                        f'baseline_monotonic={baseline_monotonic}'
                    ),
                    'Expected a fresh non-sentinel MainPID (>0) and a strictly-later '
                    'ActiveEnterTimestampMonotonic after the deploy.',
                ])
                return await self._file_infra_issue_and_block(
                    task_id,
                    summary=f'Deploy verify failed: {target_unit}',
                    detail=verify_detail,
                )

            if not always_escalates:
                # Pure cross-unit deploy (B6): verified → set done with provenance.
                # Stamp before_done_verified_at FIRST: it is the positive proof a
                # later resume (after a crash in the window before the done write)
                # requires to drive to done rather than re-escalate as a crash-window.
                verified_iso = datetime.now(UTC).isoformat()
                await self.scheduler.update_task(
                    task_id,
                    {
                        'before_done_verified_at': verified_iso,
                        'before_done_verified_pid': pid,
                    },
                    metadata_mode='merge',
                )
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
