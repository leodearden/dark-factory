"""DeterministicRunner — orchestrator-side runner for deterministic gate tasks (β/γ/ε).

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
   **Stop-instruction guard** (task 2509, reconciliation finding 0aac21b4):
   on the FIRST dispatch of a non-predicate ``before_done`` task — after the
   predicate dispatch below and after the ``before_done_ran_at`` idempotency
   block above, but BEFORE the ``before_done_ran_at`` stamp is written —
   ``orchestrator.stop_instruction.detect_stop_instruction`` scans the task
   description for an explicit stop instruction (e.g. "do not apply").  If
   found: file a born-at-L2 ``stop_instruction`` escalation and block WITHOUT
   running the deploy and WITHOUT stamping ``before_done_ran_at``.  Unlike
   the other escalations in this section, ``gate_escalated_at`` is
   deliberately NOT stamped — this is a re-checkable HALT (mirrors task
   2273's SIGTERM-kill on its human-rehearsal mandate as a self-halt rather
   than an external kill), not a resolve-to-done gate: it guarantees the
   NEXT dispatch re-evaluates this guard from scratch instead of taking
   section 1's ``if gate_escalated_at:`` resume-to-done branch (which would
   either raise ``NotImplementedError`` — no ``before_done_ran_at`` proof —
   or wrongly drive to done without the action ever running).  That
   re-evaluation only happens once the task is back in ``pending``, though:
   a ``blocked`` task holding this open pending escalation is NOT
   re-dispatched automatically, so removing the stop instruction from the
   description is necessary but not sufficient — full recovery needs BOTH
   (1) editing the description (else the guard re-fires and re-escalates on
   the very next dispatch) AND (2) resolving the ``stop_instruction``
   escalation itself (e.g. ``resolve_issue`` action ``resume``/``restart``,
   which is what actually flips the task's status back to ``pending`` — see
   ``escalation.server.resolve_issue`` Table B), or cancelling the task.
   Excludes predicates (read-only, apply nothing — see the predicate
   sub-path below) and excludes every resume/idempotency branch above (those
   already have ``before_done_ran_at`` or ``gate_escalated_at`` set, so they
   never reach this first-dispatch check).

   The ``before_done_ran_at`` stamp is written FIRST (crash-safe I1) and is
   SHARED between the self-target (ε) and cross-unit sub-paths below.

   Phase ε **self-restart** sub-path (``before_done.target_unit`` == own unit):
   - Detects self-target by comparing ``before_done.target_unit`` to the
     orchestrator's own systemd unit name, resolved from the ``ORCH_UNIT``
     environment variable via ``_default_resolve_own_unit()``.  An empty or
     unresolved ``ORCH_UNIT`` fails-open to the cross-unit path so existing
     CI runs (where ``ORCH_UNIT`` is unset) are unaffected.
   - **Operator requirement**: set ``ORCH_UNIT=<unit-name>`` in the
     ``[Service] Environment`` of the orchestrator's own systemd unit.
   - Instead of running the blocking cross-unit deploy (which would kill this
     runner mid-execution), schedules a detached ``systemd-run`` transient
     unit that fires *after* ``run()`` returns.
   - The transient unit's payload is a ``/bin/sh -c`` wrapper that runs the
     restart and, *only if it exits non-zero*, fires δ's ``escalation submit``
     CLI (file-backed, no MCP server needed in the detached unit) to file a
     born-at-L2 ``infra_issue`` escalation.  Because the whole unit is deferred
     via ``--on-active``, nothing runs at scheduling time — the escalation is
     reached only through the shell failure branch at fire time, never on the
     success path.
   - If scheduling succeeds (``rc == 0``): stamp ``before_done_scheduled_at``
     (a dict carrying the transient unit name and fire delay) as a crash-safe
     marker.  If the runner crashes between this stamp and the done write or
     gate filing, the resume path (b-self) below handles recovery.
   - If scheduling succeeds (``rc == 0``) and ``always_escalates=False``:
     set task ``done`` with
     ``done_provenance.kind='deterministic-deploy-scheduled'`` carrying the
     transient unit name and fire delay; return DONE (done = *scheduled*, not
     *verified*).  A crash between the stamp and done write is recovered by
     (b-self) driving to done with scheduled provenance.
   - If scheduling succeeds (``rc == 0``) and ``always_escalates=True``
     (act-then-ask): fall through directly to section 3 (the gate) WITHOUT
     running the blocking cross-unit deploy.  The detached restart is already
     scheduled; the gate is filed as a ``milestone_gate`` escalation; the task
     is set to ``blocked``.  Neither ``unit_inspector`` nor ``script_runner``
     are invoked on this path (self-kill and double-deploy prevention).  A
     crash between the stamp and the gate filing is recovered by (b-self)
     re-filing the gate — the human-approval gate is NEVER bypassed.
   - If scheduling fails (``rc != 0``): file born-at-L2 infra_issue, block
     (parallel to γ's rc≠0/verify-fail handling); ``before_done_ran_at``
     already stamped (not re-run, I1).

   Phase γ **cross-unit** sub-path (``target_unit`` ≠ own unit or ``ORCH_UNIT`` unset):
   - Capture baseline unit state (``unit_inspector``).
   - Run the deploy script to completion (``script_runner``, blocking).
   - If ``rc != 0``: file born-at-L2 ``infra_issue`` escalation, set blocked
     (B7a).
   - Re-inspect and verify a fresh ``MainPID`` (>0, non-sentinel) and a
     strictly-later ``ActiveEnterTimestampMonotonic`` (B7b).
   - If ``always_escalates=False``: hand off to ``_writeback_deploy_success``
     (task 2066), which stamps ``before_done_verified_at`` (the positive proof
     the resume path requires) and then sets the task to ``done`` with
     ``done_provenance.kind='deterministic-deploy'`` carrying the fresh PID
     and timestamp (B6).  Both writes are patiently retried — bounded and
     paced — because the deploy script may itself have severed the
     orchestrator's own fused-memory/MCP connection (e.g. it restarted the
     service backing that connection); a plain retry recovers once the
     connection auto-reconnects.  If the writeback budget is exhausted
     first, files a durable local ``infra_issue`` escalation and returns
     BLOCKED instead of silently stranding the task with an empty queue.
   - If ``always_escalates=True``: fall through to gate (act-then-ask).

   Phase γ **predicate** sub-path (``before_done['kind'] == 'predicate'``):
   - A read-only exit-code VERDICT check — NOT a deploy.  Dispatched at the
     very top of section 2, above the deploy-only ``target_unit``/I1/baseline
     machinery, so a predicate NEVER touches systemd (no ``unit_inspector``,
     no fresh-PID verify) and NEVER stamps ``before_done_ran_at`` (re-running
     a read-only check on crash-resume is harmless — there is no I1 side
     effect to double-apply).  This dispatch also happens BEFORE
     ``always_escalates`` is ever consulted, so a predicate task's
     ``always_escalates`` value is simply ignored — it is meaningful only on
     the deploy/gate paths' act-then-ask semantics.  A ``kind='predicate'``
     task with ``always_escalates=True`` set still goes straight to
     ``done``/``milestone_check_failed`` from the check's exit code alone,
     with no gate ever filed.  Rejecting (or normalizing) that combination,
     if ever desired, belongs in fused-memory's ``submit_task``/
     ``deterministic_task_guard`` validation, not here (out of this runner's
     scope).
   - Runs the check script under the same outer wall-clock guard pattern as
     the deploy path (``script_runner``/``_default_run_script`` +
     ``asyncio.wait_for(timeout_secs + run_timeout_grace_secs)``).
   - ``rc == 0`` (invariant holds): set task ``done`` with
     ``done_provenance.kind='deterministic-milestone'`` (never
     ``'deterministic-deploy'`` — no deploy happened).
   - ``rc != 0`` (invariant violated — a VERDICT, not an infra fault): file a
     born-at-L2 ``milestone_check_failed`` escalation, stamp
     ``gate_escalated_at`` (reusing the gate resume machinery so a human
     resolving the escalation on the next dispatch routes through section 1's
     quiescence/resolve-to-done fork), and block.
   - Timeout or an unexpected ``run_fn`` error (no verdict was produced — an
     infra fault, not a check failure): file born-at-L2 ``infra_issue`` (the
     existing timeout→escalate path) and block — NO ``gate_escalated_at``
     stamp, so the check is simply re-attempted on the next dispatch.
   - Section-1 resume: when ``gate_escalated_at`` is set and the
     ``milestone_check_failed`` escalation is resolved, RE-RUNS the predicate
     check (delegating back to ``_run_predicate`` — read-only, so repeating it
     is harmless) rather than trusting the resolution blindly: resolving an
     escalation is not proof the invariant now holds (a human may resolve it
     prematurely or in error).  ``rc == 0`` drives to ``done`` with
     ``deterministic-milestone`` provenance; ``rc != 0`` re-files
     ``milestone_check_failed`` (the dedup guard sees no pending escalation,
     since it was just resolved) and stays blocked; a timeout/error routes to
     ``infra_issue``, identical to the first-dispatch path.  This
     short-circuits BEFORE the ``before_done_ran_at`` proof-check (a
     predicate never stamps it).

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
import contextlib
import logging
import os
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from shared.task_metadata import DoneProvenance

from orchestrator import systemd_inspect
from orchestrator.deploy_state import DeployPhase, DeployState, VerifyBaseline, enforce_transition
from orchestrator.proc_supervision import (
    EscalationSpec,
    FreshPidVerify,
    RestartDisposition,
    RestartPlan,
)
from orchestrator.stop_instruction import detect_stop_instruction
from orchestrator.systemd_inspect import inspect_systemd_unit
from orchestrator.workflow import WorkflowOutcome

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)

# Task 2066: cross-unit before_done writeback resilience.
#
# The cross-unit blocking deploy may restart the very service backing the
# orchestrator's own fused-memory/MCP connection (e.g. task 2059), severing
# it for the duration of a `--drain` restart.  `McpSession._raw_call`
# (mcp_lifecycle.py) opens a fresh httpx.AsyncClient and re-inits the session
# on any transient failure, so the NEXT call auto-reconnects — a successful
# write IS the reconnection proof.  These constants size the post-deploy
# verify+writeback retry budget to patiently outlast such a restart instead
# of the much shorter Scheduler._TRANSIENT_RETRIES budget (sized for a plain
# API hiccup, not a service restart).
#
# 6 attempts / base 2.0s exponential backoff -> 5 sleeps summing to
# 2.0 * (2**5 - 1) = 62s of patient retrying before giving up — bounded (no
# infinite loop) while comfortably exceeding a typical `--drain` restart.
_WRITEBACK_MAX_ATTEMPTS: int = 6
_WRITEBACK_BACKOFF_BASE: float = 2.0

# Task 2090: bound the reap after a whole-process-group SIGKILL on a
# before_done subprocess timeout.  A process stuck in an uninterruptible
# state (or one that otherwise ignores SIGKILL) must never be allowed to hang
# `_terminate_process_tree` — and therefore `_default_run_script` — forever.
# 5s comfortably covers a normal kernel-mediated reap while staying bounded.
_REAP_GRACE_SECS: float = 5.0

# Task 2090: outer wall-clock backstop around the cross-unit `run_fn` call in
# run().  This is a pure safety margin ON TOP OF before_done['timeout_secs'] —
# it must never fire before the inner script-runner timeout in the normal
# case, so 30s is deliberately generous (covers process-group teardown +
# reap_grace_secs with room to spare) rather than tight.
_RUN_TIMEOUT_GRACE_SECS: float = 30.0

# Task 2091 / 2119: bound `_default_inspect_unit`'s `systemctl --user show`
# call — a parallel latent-hang gap to task 2090, which only wraps the
# before_done run_fn subprocess and not this inspect call (esc-2090-11).
# This runs on both the baseline inspect and the post-deploy verify inspect;
# an unbounded `communicate()` here strands the runner exactly like task 2087
# (before_done_ran_at stamped, before_done_verified_at never stamped, no
# escalation filed). The hardening itself (and its 10s default timeout) now
# lives in ``systemd_inspect.py`` — see ``_default_inspect_unit`` below,
# which is a thin delegate to ``systemd_inspect.inspect_systemd_unit``.
_INSPECT_TIMEOUT_SECS: float = systemd_inspect._INSPECT_TIMEOUT_SECS

# Task 2120: sentinel agent_role the runner stamps on its own escalations and
# scopes all its get_by_task queries to — an unrelated escalation with the
# same task_id (e.g. a starvation-watchdog filing) must never alias as this
# runner's own dedup/quiescence/resolution-proof signal.
DETERMINISTIC_AGENT_ROLE: str = 'orchestrator-deterministic'

# Task 2238 (W10-δ): a guaranteed-non-self placeholder `own_unit` fed to
# proc_supervision.RestartPlan.execute() on the cross-unit blocking deploy
# path when the runner's own ORCH_UNIT-derived own_unit is falsy (the
# fail-open case, ORCH_UNIT unset).  RestartPlan.execute()'s RP-1 fail-closed
# guard refuses a blocking restart whenever own_unit is falsy — but the
# runner only reaches the cross-unit branch AFTER its own `self_target` check
# above has already ruled out a same-unit restart, so forcing a truthy,
# provably-non-target own_unit here routes execute() into RP-2 (cross-unit
# blocking + verify) instead of RP-1 (refuse), preserving the existing
# fail-open-to-cross-unit behaviour (see test_env_unset_takes_cross_unit_path).
# Not a valid systemd unit name (unit names never contain angle brackets), so
# it can never collide with a real target_unit.
_CROSS_UNIT_OWN_UNIT_SENTINEL: str = '<no-self-target-known>'


def _build_done_provenance(kind: str, **fields: object) -> dict:
    """Build a ``done_provenance`` dict via the shared ``DoneProvenance`` model.

    THE single seam every runner ``done_provenance`` construction routes
    through (task 2167 — W3-δ SEAM B), sharing ONE valid-kinds enum with the
    fused-memory validator (I2) instead of six independent inline dict
    literals.  An unknown/typo *kind* raises ``pydantic.ValidationError``
    here, at build time, on the orchestrator side — structurally preventing
    the 1902/1976/1982 permanently-blocked self-restart failure mode (a
    ``kind`` fused-memory silently rejects).

    ``exclude_none=True`` keeps the emitted wire dict compatible with the
    hand-written literals this replaces for every key that carries a
    non-``None`` value (they never carried explicit ``None`` values either).
    One path diverges: the crash-resume call at ``before_done_verified_pid``
    used to always emit a ``pid`` key (``None`` when that metadata field was
    absent), whereas ``exclude_none`` now omits the key entirely in that
    case.  All known consumers (``task_interceptor.py`` and this module's
    tests) read it via ``.get('pid')``, so a missing key and an explicit
    ``None`` are observationally identical there; extra fields such as
    ``transient_unit`` / ``fire_delay_secs`` survive via ``DoneProvenance``'s
    ``extra='allow'``.

    ``kind``/``**fields`` are intentionally loosely typed (``str``/``object``):
    the six call sites forward heterogeneous field subsets, so pyright's
    dataclass-transform-synthesized ``DoneProvenance.__init__`` (each
    parameter narrowly typed, e.g. ``kind: Literal[...]``, ``pid: int | None``)
    cannot be satisfied by a generic pass-through signature. The real
    validation happens at runtime, in the model itself.
    """
    return DoneProvenance(kind=kind, **fields).model_dump(exclude_none=True)  # type: ignore[arg-type]


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
        writeback_max_attempts: Bound on the post-deploy verify+writeback
            retry loop (task 2066).  Defaults to ``_WRITEBACK_MAX_ATTEMPTS``.
        writeback_backoff_base: Exponential backoff base (seconds) for the
            writeback retry loop.  Defaults to ``_WRITEBACK_BACKOFF_BASE``.
        sleeper: Optional async callable ``(seconds: float) -> None`` used to
            pace writeback retries.  Defaults to ``asyncio.sleep``.  Injected
            in tests so retries don't actually sleep.
        reap_grace_secs: Bound (seconds) on reaping a subprocess after a
            whole-process-group SIGKILL on timeout (task 2090).  Defaults to
            ``_REAP_GRACE_SECS``.  Injected in tests so an unkillable-process
            test doesn't actually wait.
        run_timeout_grace_secs: Extra margin (seconds), added to
            ``before_done['timeout_secs']``, for the outer wall-clock guard
            around the cross-unit ``run_fn`` call in ``run()`` (task 2090).
            Defaults to ``_RUN_TIMEOUT_GRACE_SECS``.  Injected in tests so a
            stuck-run_fn test doesn't actually wait.
        inspect_timeout_secs: Bound (seconds) on the default unit-inspector's
            ``systemctl --user show`` ``communicate()`` call (task 2091).
            Defaults to ``_INSPECT_TIMEOUT_SECS``.  Injected in tests so a
            wedged-inspect test doesn't actually wait.
    """

    def __init__(
        self,
        scheduler,
        escalation_queue: EscalationQueue,
        unit_inspector=None,
        script_runner=None,
        own_unit_resolver=None,
        restart_scheduler=None,
        writeback_max_attempts=_WRITEBACK_MAX_ATTEMPTS,
        writeback_backoff_base=_WRITEBACK_BACKOFF_BASE,
        sleeper=None,
        reap_grace_secs=_REAP_GRACE_SECS,
        run_timeout_grace_secs=_RUN_TIMEOUT_GRACE_SECS,
        inspect_timeout_secs=_INSPECT_TIMEOUT_SECS,
    ):
        self.scheduler = scheduler
        self.escalation_queue = escalation_queue
        self._unit_inspector = unit_inspector
        self._script_runner = script_runner
        self._own_unit_resolver = own_unit_resolver
        self._restart_scheduler = restart_scheduler
        self._writeback_max_attempts = writeback_max_attempts
        self._writeback_backoff_base = writeback_backoff_base
        self._sleeper = sleeper or asyncio.sleep
        self._reap_grace_secs = reap_grace_secs
        self._run_timeout_grace_secs = run_timeout_grace_secs
        self._inspect_timeout_secs = inspect_timeout_secs

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

        Thin ``proc_supervision.RestartPlan`` caller (task 2238/δ) — mirrors
        ``service_restart.schedule_detached_systemd_restart``'s conversion
        (task 2237/γ). Builds a same-unit, DETACHED ``RestartPlan``
        (``target_unit == own_unit == transient_unit``, ``verify=None``,
        ``transient_unit`` set) carrying an ``EscalationSpec`` for the RP-4
        on-failure wrapper, and delegates the actual systemd-run registration
        to ``plan.execute()``. ``RestartPlan.execute()``'s detached path
        (``_execute_detached_systemd_run``) builds a SINGLE ``--on-active``
        transient unit whose payload is a ``/bin/sh -c`` wrapper that runs the
        restart script and, *only if it exits non-zero*, fires δ's
        escalation-submit CLI before re-raising the original exit code (so
        journald records the unit as failed):

            <script> <args>
            __rc=$?
            if [ "$__rc" -ne 0 ]; then <escalation submit …>; fi
            exit "$__rc"

        Why not a separate ``OnFailure=`` handler unit?  ``systemd-run`` has no
        register-without-start mode — registering a companion handler transient
        *service* would activate it immediately at scheduling time, filing a
        spurious born-at-L2 on EVERY successful self-deploy rather than only on a
        fire-time failure (the bug this method previously had).  ``--on-active``
        defers the whole unit, so NOTHING runs at registration; the escalation
        is reached only through the shell's failure branch when the restart
        actually fails at fire time.  This preserves the intended semantics —
        "run δ's escalation-submit CLI iff the restart fires and fails" — with a
        single deferred unit and no eager execution.

        systemd-run returns immediately after registering the transient unit;
        the orchestrator is NEVER blocked or killed — the payload fires later
        under the user systemd manager.

        Returns:
            (rc, tail) — rc=0 on successful registration; rc≠0 if registration
            fails (tail carries the error detail).
        """
        target_unit = before_done.get('target_unit', 'unknown')
        script = before_done['script']
        args = before_done.get('args') or []

        # The transient unit runs under the systemd --user manager, which does
        # NOT inherit the orchestrator's own working directory — it defaults to
        # $HOME.  A relative deploy `script` would therefore fail to be found
        # (exit 127) once the unit fires.  Resolve an explicit cwd from
        # before_done['cwd'] when the caller supplied one; otherwise fall back
        # to this process's own os.getcwd(), which is project_root because the
        # orchestrator's own systemd unit pins WorkingDirectory=project_root.
        # RestartPlan.__post_init__ absolutizes a relative `script` against
        # this `cwd` (RP-3), byte-identical to this method's prior inline
        # absolutization.
        cwd = before_done.get('cwd') or os.getcwd()

        esc_summary = summary or (
            f'Self-restart fire-time failure: {target_unit}'
        )

        # δ's escalation-submit CLI, fired ONLY when the restart fails at fire
        # time — built here as data and handed to RestartPlan/EscalationSpec,
        # which owns the RP-4 on-failure ``/bin/sh -c`` wrapper + the
        # ``python -m escalation submit`` argv construction (byte-identical to
        # this method's prior inline argv — see EscalationSpec.to_submit_argv).
        #
        # Deployment assumption: the `escalation` package must be importable from
        # sys.executable's interpreter (i.e. installed into site-packages, not
        # just reachable via a PYTHONPATH side-channel from the orchestrator
        # service unit).  If it is not, the OnFailure branch itself fails and no
        # L2 is filed — the task is already marked done=scheduled at this point,
        # so the failure would be silently lost.  Operators should verify with:
        #   <sys.executable> -c "import escalation"
        # before deploying.  A marker-file fallback is intentionally not
        # implemented here to keep the failure path auditable via journald.
        escalation_spec = EscalationSpec(
            queue_dir=str(self.escalation_queue.queue_dir),
            task_id=task_id,
            summary=esc_summary,
            detail=(
                f'Transient unit {transient_unit} fired and failed (task {task_id}). '
                f'Check journald for restart output: '
                f'journalctl --user -u {transient_unit}'
            ),
            severity='critical',
            category='infra_issue',
            agent_role=DETERMINISTIC_AGENT_ROLE,
        )

        plan = RestartPlan(
            script=Path(script),
            args=list(args),
            cwd=Path(cwd),
            target_unit=transient_unit,
            own_unit=transient_unit,
            on_failure_escalation=escalation_spec,
            verify=None,
            transient_unit=transient_unit,
            on_active_secs=on_active_secs,
        )
        outcome = await plan.execute()
        if outcome.disposition == RestartDisposition.REGISTRATION_FAILED:
            logger.warning(
                'DeterministicRunner: failed to register restart transient unit %s '
                'for task %s: %s',
                transient_unit, task_id, outcome.detail,
            )
            return 1, outcome.detail
        return 0, ''

    async def _default_inspect_unit(self, unit: str) -> dict:
        """Query systemctl for unit state fields needed for fresh-PID verify.

        Task 2119: thin delegate to the hoisted, hardened
        ``systemd_inspect.inspect_systemd_unit`` (task 2091's timeout/kill/
        reap hardening now lives there exactly once). Returns a dict with at
        minimum: MainPID (int), ActiveState (str), ActiveEnterTimestamp
        (str), ActiveEnterTimestampMonotonic (int). Integers default to 0 on
        parse failure (sentinel-safe).
        """
        return await inspect_systemd_unit(
            unit,
            timeout_secs=self._inspect_timeout_secs,
            reap_grace_secs=self._reap_grace_secs,
        )

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
            start_new_session=True,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_secs)
            tail = (stdout or b'').decode(errors='replace')[-2000:]
            return proc.returncode or 0, tail
        except TimeoutError:
            # Task 2090: the deploy script may spawn grandchildren (e.g.
            # restart-fused-memory.sh --drain forking systemctl/curl/journalctl/
            # sleep/the restarted daemon) that inherit the write end of the
            # merged stdout pipe above.  Killing only this direct child leaves
            # them alive as orphans — still holding that pipe open and still
            # running indefinitely.  start_new_session=True (above) makes this
            # process its own session/group leader so the WHOLE tree can be
            # torn down together instead of leaking orphaned processes.
            await self._terminate_process_tree(proc)
            return 1, f'<script timed out after {timeout_secs}s>'

    async def _terminate_process_tree(self, proc: asyncio.subprocess.Process) -> None:
        """Kill *proc*'s entire process group and bound the reap (task 2090).

        ``proc`` must have been spawned with ``start_new_session=True`` so its
        process group is its own — NOT the orchestrator's — otherwise
        ``os.killpg`` would SIGKILL the orchestrator itself.

        Tries a whole-group SIGKILL first, which reaches any grandchildren
        (e.g. a backgrounded ``systemctl``/``curl``/``sleep``) that would
        otherwise survive the direct child's death as leaked orphans —
        continuing to run (and keep the inherited stdout pipe's write end
        open) indefinitely.  Note this is NOT needed to unblock ``proc.wait()``
        below: asyncio's ``Process.wait()`` resolves on the child's own exit
        notification (via the event loop's child watcher), not on pipe EOF —
        only ``communicate()`` waits on that.  The process-group kill's value
        is purely in not leaking the grandchildren themselves.  If the process
        has already exited (or the group lookup otherwise fails), falls back
        to a direct ``proc.kill()``.  Both kill attempts tolerate an
        already-dead process — this helper must NEVER raise, so a timeout
        branch can always return cleanly even if the process is already gone.

        The reap itself is bounded by ``self._reap_grace_secs`` so a process
        stuck in an uninterruptible state cannot hang this helper (and
        therefore ``_default_run_script``) forever.

        Note (residual race): ``os.killpg(os.getpgid(proc.pid), ...)`` targets
        ``proc.pid`` by number.  If asyncio's child watcher has already reaped
        the zombie in the background before this call runs, the OS could in
        principle recycle that PID before ``os.getpgid``/``os.killpg`` execute,
        making the SIGKILL target an unrelated process group.  This is a
        low-probability race shared with any PID-based kill (not specific to
        this helper); the ``ProcessLookupError``/``OSError`` suppression above
        is the existing mitigation, not a full fix for the underlying race.
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError) as exc:
            logger.debug(
                'DeterministicRunner: killpg(%s) failed (%s: %s) — falling back '
                'to direct kill()',
                proc.pid, type(exc).__name__, exc,
            )
            with contextlib.suppress(ProcessLookupError, OSError):
                proc.kill()

        try:
            await asyncio.wait_for(proc.wait(), timeout=self._reap_grace_secs)
        except TimeoutError:
            logger.warning(
                'DeterministicRunner: process %s did not exit within '
                'reap_grace_secs=%s after termination signal — abandoning reap '
                '(process may be unkillable)',
                proc.pid, self._reap_grace_secs,
            )

    async def _file_infra_issue_and_block(
        self,
        task_id: str,
        summary: str,
        detail: str,
        *,
        metadata: dict | None = None,
    ) -> WorkflowOutcome:
        """File a born-at-L2 infra_issue escalation and set the task to blocked.

        Reuses β's escalation construction pattern (sentinel role keeps level=2
        past the server downgrade gate).  Includes a dedup guard — if a pending
        escalation already exists (e.g. prior crash-safe re-dispatch), filing is
        skipped to avoid duplicate L2 escalations.

        The trailing ``set_task_status(..., 'blocked')`` write is best-effort
        (task 2066 amendment): callers — including
        ``_writeback_deploy_success`` on writeback-budget exhaustion — may
        reach this method with the scheduler's own connection persistently
        severed, i.e. the same failure mode that triggered the escalation in
        the first place.  The escalation itself is already durable at this
        point (``EscalationQueue.submit`` writes to local disk, independent of
        fused-memory), so a failure to *also* persist the 'blocked' status
        must not propagate — doing so would defeat this method's "always
        returns BLOCKED, never a raw exception" contract in exactly the
        scenario it exists to cover.

        ``metadata``, when passed by a DEPLOY-path caller (``before_done``
        set — every ``run()``-internal call site qualifies; ``_run_predicate``
        never passes it), best-effort advances ``deploy_state.phase`` to
        ``ESCALATED`` too (ζ DS-1) — same tolerate-a-severed-connection
        posture as the trailing blocked-status write below, since the
        escalation above is already durable regardless.

        Returns:
            WorkflowOutcome.BLOCKED
        """
        from escalation.models import Escalation

        existing_pending = self.escalation_queue.get_by_task(
            task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
        )
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
                agent_role=DETERMINISTIC_AGENT_ROLE,
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

        if metadata is not None and metadata.get('before_done') is not None:
            try:
                await self._advance_deploy_phase(task_id, metadata, DeployPhase.ESCALATED)
            except Exception as exc:
                logger.warning(
                    'DeterministicRunner: task %s deploy_state phase-escalated '
                    'advance failed (%s: %s) — the infra_issue escalation above '
                    'is already durable regardless',
                    task_id, type(exc).__name__, exc,
                )

        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
            logger.info('DeterministicRunner: task %s blocked — infra_issue', task_id)
        except Exception as exc:
            # Do NOT let a still-severed connection turn this into a
            # propagated exception — it would bubble past run() into the
            # harness's generic handler and produce a SILENT blocked report,
            # defeating the point of filing a durable escalation above.
            logger.warning(
                'DeterministicRunner: task %s blocked-status writeback failed '
                '(connection may still be severed): %s: %s — the infra_issue '
                'escalation is already durable on local disk, so returning '
                'BLOCKED regardless',
                task_id, type(exc).__name__, exc,
            )
        return WorkflowOutcome.BLOCKED

    async def _file_stop_instruction_and_block(
        self,
        task_id: str,
        summary: str,
        detail: str,
    ) -> WorkflowOutcome:
        """File a born-at-L2 stop_instruction escalation and set the task to blocked.

        Task 2509 (reconciliation finding 0aac21b4): mirrors
        ``_file_infra_issue_and_block``'s dedup guard, escalation construction,
        and best-effort blocked-status writeback, but uses ``category=
        'stop_instruction'`` so the halt is distinguishable from an infra
        fault — and, critically, does NOT stamp ``metadata.gate_escalated_at``.

        Unlike ``_file_infra_issue_and_block``'s dedup guard, this one is
        scoped to ``category == 'stop_instruction'`` (review amendment): a
        pre-existing, unrelated pending escalation on the same task (e.g. an
        ``infra_issue`` filed by a prior crash) must not silently suppress
        filing THIS one — the stop-instruction halt is a distinct, higher-
        authority signal that a human resolving the other escalation could
        otherwise miss entirely.
        This is a re-checkable HALT, not a resolve-to-done gate: leaving
        ``gate_escalated_at`` unset means the next dispatch re-evaluates the
        stop-instruction guard from scratch — section 1's
        ``if gate_escalated_at:`` resume-to-done branch is never taken —
        rather than silently driving to done with no proof the deploy ran.

        That re-evaluation only happens once the task is back in
        ``pending``, though: a task in ``blocked`` holding this open pending
        escalation is NOT re-dispatched automatically (reviewer amendment —
        see task 2509 review). So the full recovery is two steps, not one:
        (1) edit the task description to remove the stop instruction — if
        left in place, the guard simply re-fires and re-escalates on the
        very next dispatch, since it re-reads the description fresh every
        time — and (2) resolve this escalation (e.g. ``resolve_issue``
        action ``resume`` or ``restart``, either of which flips the task's
        status back to ``pending`` per ``escalation.server.resolve_issue``'s
        Table B — this runner has no live agent to resume, so both actions
        have the same practical effect here: a fresh dispatch). Editing the
        description alone, with the escalation left pending, does NOT
        re-enable dispatch. A human may instead cancel the task outright.

        Returns:
            WorkflowOutcome.BLOCKED
        """
        from escalation.models import Escalation

        # category-scoped (review amendment): get_by_task has no server-side
        # category filter, so filter client-side — an unrelated pending
        # escalation (e.g. infra_issue) on the same task must not suppress
        # filing this one. See the docstring note above.
        existing_pending = [
            e for e in self.escalation_queue.get_by_task(
                task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
            )
            if e.category == 'stop_instruction'
        ]
        if existing_pending:
            logger.info(
                'DeterministicRunner: task %s already has %d pending '
                'stop_instruction escalation(s) — skipping re-file '
                '(stop_instruction dedup guard)',
                task_id, len(existing_pending),
            )
        else:
            esc = Escalation(
                id=self.escalation_queue.make_id(task_id),
                task_id=task_id,
                agent_role=DETERMINISTIC_AGENT_ROLE,
                severity='critical',
                category='stop_instruction',
                summary=summary[:200],
                detail=detail,
                level=2,
            )
            self.escalation_queue.submit(esc)
            logger.info(
                'DeterministicRunner: filed L2 stop_instruction escalation %s for task %s',
                esc.id, task_id,
            )

        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
            logger.info('DeterministicRunner: task %s blocked — stop_instruction', task_id)
        except Exception as exc:
            # Same rationale as _file_infra_issue_and_block: a still-severed
            # connection must not turn this into a propagated exception — the
            # escalation is already durable on local disk, so return BLOCKED
            # regardless.
            logger.warning(
                'DeterministicRunner: task %s blocked-status writeback failed '
                '(connection may still be severed): %s: %s — the stop_instruction '
                'escalation is already durable on local disk, so returning '
                'BLOCKED regardless',
                task_id, type(exc).__name__, exc,
            )
        return WorkflowOutcome.BLOCKED

    def _deploy_transition_escalation_sink(
        self, task_id: str, old: DeployPhase, new: DeployPhase,
    ) -> None:
        """DS-2 sink: file a born-at-L2 escalation for an illegal deploy-phase edge.

        Wired into every ``_advance_deploy_phase`` call via
        ``enforce_transition``'s ``escalation_sink`` — files BEFORE
        ``enforce_transition`` raises ``IllegalDeployTransition``, so an
        illegal edge is never silently swallowed (D2).  Mirrors the other
        escalation-filing helpers' construction (sentinel role keeps level=2
        past the server downgrade gate) but is unconditional — no dedup
        guard — since an illegal transition is itself a bug signal that must
        never be suppressed by an unrelated pending escalation.
        """
        from escalation.models import Escalation

        esc = Escalation(
            id=self.escalation_queue.make_id(task_id),
            task_id=task_id,
            agent_role=DETERMINISTIC_AGENT_ROLE,
            severity='critical',
            category='illegal_deploy_transition',
            summary=f'Illegal deploy-phase transition {old} -> {new}'[:200],
            detail=(
                f'DeterministicRunner attempted an illegal deploy-phase '
                f'transition {old!r} -> {new!r} for task {task_id}. This '
                'indicates a runner bug or corrupted deploy_state metadata — '
                'investigate before resuming.'
            ),
            level=2,
        )
        self.escalation_queue.submit(esc)
        logger.error(
            'DeterministicRunner: filed L2 illegal_deploy_transition escalation '
            '%s for task %s (%s -> %s)',
            esc.id, task_id, old, new,
        )

    def _compute_deploy_phase_advance(
        self,
        task_id: str,
        metadata: dict,
        new_phase: DeployPhase,
        *,
        verify_baseline: VerifyBaseline | dict | None = None,
    ) -> DeployState:
        """DS-2: enforce the transition and build the advanced ``DeployState``.

        Pure computation, no I/O — reads the CURRENT phase from *metadata*
        (the in-memory dict threaded through ``run()``), enforces the
        transition (files a loud escalation then raises on an illegal edge
        via ``_deploy_transition_escalation_sink``), and returns the new
        ``DeployState`` with the OLD ``ran_at``/``verified_at``/
        ``escalated_at``/``verify_baseline`` evidence carried forward.

        Shared by ``_advance_deploy_phase`` (the single-write path) and
        ``_writeback_deploy_success`` (which folds the resulting
        ``to_metadata()`` into its own retry-loop write rather than issuing a
        second ``update_task`` call — the retry loop needs the RAW boolean
        ``update_task`` return to detect a transient failure, which
        ``_advance_deploy_phase`` does not expose).

        ``verify_baseline``, when omitted, carries forward whatever baseline
        was already recorded — a phase advance must never drop previously
        persisted DS-3 evidence.
        """
        old_state = DeployState.from_metadata(metadata)
        old_phase = old_state.phase if old_state is not None else None
        enforce_transition(
            old_phase, new_phase, task_id=task_id,
            escalation_sink=self._deploy_transition_escalation_sink,
        )

        if verify_baseline is not None and not isinstance(verify_baseline, VerifyBaseline):
            verify_baseline = VerifyBaseline(**verify_baseline)
        carried_baseline = (
            verify_baseline if verify_baseline is not None
            else (old_state.verify_baseline if old_state is not None else None)
        )
        return DeployState(
            phase=new_phase,
            verify_baseline=carried_baseline,
            ran_at=old_state.ran_at if old_state is not None else None,
            verified_at=old_state.verified_at if old_state is not None else None,
            escalated_at=old_state.escalated_at if old_state is not None else None,
        )

    async def _advance_deploy_phase(
        self,
        task_id: str,
        metadata: dict,
        new_phase: DeployPhase,
        *,
        evidence: dict | None = None,
        verify_baseline: VerifyBaseline | dict | None = None,
    ) -> DeployState:
        """DS-1/DS-2: atomically advance ``metadata.deploy_state.phase`` + evidence.

        Persists ``{**new DeployState.to_metadata(), **evidence}`` in ONE
        ``update_task(metadata_mode='merge')`` call, and refreshes
        ``metadata['deploy_state']`` in place so a LATER call within the SAME
        ``run()`` invocation observes the just-written phase as ``old``. See
        ``_compute_deploy_phase_advance`` for the transition-enforcement +
        state-construction logic this wraps.
        """
        new_state = self._compute_deploy_phase_advance(
            task_id, metadata, new_phase, verify_baseline=verify_baseline,
        )
        payload = {**new_state.to_metadata(), **(evidence or {})}
        await self.scheduler.update_task(task_id, payload, metadata_mode='merge')
        metadata['deploy_state'] = new_state.to_metadata()['deploy_state']
        return new_state

    async def _enrich_deploy_state_baseline(
        self, task_id: str, metadata: dict, baseline: dict,
    ) -> None:
        """DS-3: persist ``verify_baseline`` WITHOUT a phase transition.

        Bypasses ``enforce_transition`` entirely — this is a same-phase
        enrichment (advancing RAN -> RAN would hit the pinned-illegal
        self-loop edge). Best-effort: this is a freshness UPGRADE, never a
        new failure mode — a crash (or a still-severed connection) before
        this write lands leaves ``phase==RAN`` with no baseline, which the
        strand detector still correctly classifies as a RAN-strand and the
        freshness verdict's no-baseline fallback still handles (see ζ design
        decision on verify_baseline persistence timing).
        """
        old_state = DeployState.from_metadata(metadata)
        if old_state is None:
            # Defensive: the shared before_done_ran_at/phase=ran write always
            # precedes baseline capture — nothing to enrich onto if absent.
            return
        verify_baseline = VerifyBaseline(
            active_enter_timestamp_monotonic=baseline.get('ActiveEnterTimestampMonotonic', 0),
            main_pid=baseline.get('MainPID', 0),
        )
        new_state = DeployState(
            phase=old_state.phase,
            verify_baseline=verify_baseline,
            ran_at=old_state.ran_at,
            verified_at=old_state.verified_at,
            escalated_at=old_state.escalated_at,
        )
        try:
            await self.scheduler.update_task(
                task_id, new_state.to_metadata(), metadata_mode='merge',
            )
            metadata['deploy_state'] = new_state.to_metadata()['deploy_state']
        except Exception as exc:
            logger.warning(
                'DeterministicRunner: task %s verify_baseline enrichment failed '
                '(%s: %s) — freshness will fall back to liveness-only; not a new '
                'failure mode (phase==ran is still correctly detected as a strand)',
                task_id, type(exc).__name__, exc,
            )

    async def _writeback_deploy_success(
        self,
        task_id: str,
        metadata: dict,
        new_state: dict,
        target_unit: str,
        description: str,
    ) -> WorkflowOutcome:
        """Patiently persist a successful cross-unit deploy's verify+done writeback.

        Task 2066: the cross-unit deploy script may restart the very service
        backing the orchestrator's own fused-memory/MCP connection (e.g. task
        2059), severing it for the duration of a ``--drain`` restart.  Because
        ``McpSession._raw_call`` auto-reconnects on the next call after any
        transient failure, a plain patient retry — not an explicit reconnect —
        is sufficient: a successful write IS the reconnection proof.

        Retries the ``before_done_verified_at``/``before_done_verified_pid``
        stamp (treating a ``False`` return from ``update_task`` as transient)
        until it lands, then retries the ``done`` write (catching a transient/
        ``RuntimeError`` from ``set_task_status``'s own exhausted retry) —
        both within the SAME bounded, paced ``_writeback_max_attempts``/
        ``_writeback_backoff_base`` budget, without re-stamping and without
        re-running the deploy script (I1 once-only — this helper only
        persists an already-completed deploy's outcome).

        ζ DS-1: the verified-stamp write also atomically carries
        ``deploy_state.phase=='verified'`` — folded into the SAME write
        computed ONCE up front (the pre-write phase does not change across
        retries within this call), so the shared retry budget stays exactly
        two writes.  The done-write below intentionally does NOT also advance
        phase to ``done`` — ``verified -> done`` is a real DS-2-legal edge,
        but leaving the PERSISTED phase at ``verified`` avoids a third write
        (verified already proves success; done tasks are never swept).

        On budget exhaustion (fused-memory never recovers in-window), files a
        durable local ``infra_issue`` escalation (disk-backed, connection-
        independent) and returns BLOCKED — so the task is never silently
        stranded with an empty escalation queue (the EVIDENCE failure on task
        2059).

        Returns:
            WorkflowOutcome.DONE — verified stamp + done write both persisted.
            WorkflowOutcome.BLOCKED — writeback budget exhausted; infra_issue filed.
        """
        verified_iso = datetime.now(UTC).isoformat()
        pid = new_state.get('MainPID', 0)
        active_enter_timestamp = new_state.get('ActiveEnterTimestamp', '')

        verified_deploy_state = self._compute_deploy_phase_advance(
            task_id, metadata, DeployPhase.VERIFIED,
        )
        deploy_state_payload = verified_deploy_state.to_metadata()

        stamped = False
        for attempt in range(self._writeback_max_attempts):
            if not stamped:
                stamped = bool(await self.scheduler.update_task(
                    task_id,
                    {
                        'before_done_verified_at': verified_iso,
                        'before_done_verified_pid': pid,
                        **deploy_state_payload,
                    },
                    metadata_mode='merge',
                ))
                if stamped:
                    metadata['deploy_state'] = deploy_state_payload['deploy_state']
                if not stamped:
                    logger.warning(
                        'DeterministicRunner: task %s verified-stamp writeback failed '
                        '(attempt %d/%d) — connection may be severed by the deploy; '
                        'retrying',
                        task_id, attempt + 1, self._writeback_max_attempts,
                    )
                    if attempt + 1 < self._writeback_max_attempts:
                        await self._sleeper(self._writeback_backoff_base * (2 ** attempt))
                    continue

            # Shared-budget design (task 2066 amendment): the verified-stamp
            # write above and the done-write below intentionally draw from
            # the SAME `_writeback_max_attempts` budget and `attempt` counter
            # rather than each getting its own sub-budget.  If the stamp only
            # lands on the final iteration, the done-write gets exactly one
            # remaining try before the loop falls through to the
            # budget-exhausted escalation path below — there is no dedicated
            # done-write retry allowance.  This keeps the overall ceiling
            # simple (the ~62s default bounds BOTH writes together) at the
            # cost of a possibly-single done-write attempt in the worst case;
            # see
            # test_stamp_lands_on_final_attempt_then_done_write_fails_still_escalates
            # for the regression guard covering this exact edge.
            logger.info(
                'DeterministicRunner: task %s before_done deploy verified — '
                'pid=%s unit=%s — setting done',
                task_id, pid, target_unit,
            )
            try:
                await self.scheduler.set_task_status(
                    task_id,
                    'done',
                    done_provenance=_build_done_provenance(
                        'deterministic-deploy',
                        pid=pid,
                        active_enter_timestamp=active_enter_timestamp,
                        unit=target_unit,
                    ),
                )
                return WorkflowOutcome.DONE
            except Exception as exc:
                # set_task_status already exhausted its own transient retry
                # budget (Scheduler._TRANSIENT_RETRIES) before raising — the
                # connection may still be severed.  Retry the done-write
                # within the SAME writeback budget rather than letting this
                # propagate out of run() (stamped stays True — no re-stamp).
                logger.warning(
                    'DeterministicRunner: task %s done-write failed (attempt %d/%d): '
                    '%s: %s — connection may still be severed; retrying',
                    task_id, attempt + 1, self._writeback_max_attempts,
                    type(exc).__name__, exc,
                )
                if attempt + 1 < self._writeback_max_attempts:
                    await self._sleeper(self._writeback_backoff_base * (2 ** attempt))
                continue

        # Budget exhausted — fused-memory never recovered in-window.  File a
        # durable, connection-independent local escalation (EscalationQueue.submit
        # writes to disk) so the task is never silently stranded — the exact
        # EVIDENCE failure on task 2059 (empty queue + blocked forever).
        # before_done_ran_at is already stamped (I1 — the deploy is NOT re-run
        # here).  This files an L2 escalation and blocks the task; the
        # quiescence guard (section 1, before_done_ran_at+pending sub-case)
        # prevents re-dispatch while that escalation stays open, so resume is
        # NOT automatic while blocked.  If the verified stamp landed before
        # the budget ran out, a human resolving this escalation — and the
        # task then being re-dispatched — is a second safety net: crash-resume
        # sub-case (a) will drive it to done at that point, without re-running
        # the deploy.
        detail = '\n'.join([
            description,
            f'Target unit: {target_unit}',
            'The cross-unit deploy ran and verified successfully, but the '
            "orchestrator's own fused-memory/MCP connection was severed by "
            'the deploy (e.g. the deploy restarted the service backing that '
            'connection) and the verify+writeback could not be persisted '
            f'within the reconnect budget ({self._writeback_max_attempts} '
            'attempts). before_done_ran_at is already stamped (I1 — the '
            'deploy is NOT re-run). Resolve this escalation to unblock the '
            'task: if the verified stamp landed before the connection was '
            'lost, resolving it and letting the task re-dispatch will then '
            'resume it to done automatically (no re-run); otherwise inspect '
            f'{target_unit} and the task metadata to determine whether the '
            'deploy needs to be retried manually.',
        ])
        return await self._file_infra_issue_and_block(
            task_id,
            summary=f'Deploy verify/writeback stranded (connection severed): {target_unit}',
            detail=detail,
            metadata=metadata,
        )

    async def _file_milestone_gate_and_block(
        self, task_id: str, task: dict, metadata: dict
    ) -> WorkflowOutcome:
        """File a born-at-L2 ``milestone_gate`` escalation and block the task.

        Encapsulates section 3's gate-filing logic for reuse by:
        - The pure-gate path (section 3) in ``run()``.
        - The (b-self) crash-resume path when ``always_escalates=True``
          (crash between ``before_done_scheduled_at`` stamp and gate filing).

        Includes a dedup guard — if a pending escalation already exists
        (e.g. prior crash-safe re-dispatch), filing is skipped to avoid
        duplicate L2 escalations.  Stamps ``gate_escalated_at`` so the
        next resume routes through section-1 quiescence.

        Returns:
            ``WorkflowOutcome.BLOCKED``
        """
        from escalation.models import Escalation

        title = task.get('title', '')
        description = task.get('description', '')
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
        existing_pending = self.escalation_queue.get_by_task(
            task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
        )
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
                agent_role=DETERMINISTIC_AGENT_ROLE,
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

        # Stamp gate_escalated_at AFTER successful escalation submit.  ζ: on the
        # DEPLOY path (before_done set — this helper is also reached from the
        # self-restart and cross-unit act-then-ask fallthroughs, and the
        # self-restart crash-resume re-file), this atomically advances
        # deploy_state.phase to ESCALATED too (DS-1). A pure gate
        # (before_done=None) is not a deploy and gets no deploy_state.
        now_iso = datetime.now(UTC).isoformat()
        if metadata.get('before_done') is not None:
            await self._advance_deploy_phase(
                task_id, metadata, DeployPhase.ESCALATED,
                evidence={'gate_escalated_at': now_iso},
            )
        else:
            await self.scheduler.update_task(
                task_id,
                {'gate_escalated_at': now_iso},
                metadata_mode='merge',
            )

        # Set status to blocked — gate awaits human decision.  Best-effort, same
        # as _file_infra_issue_and_block (reviewer amendment): the escalation +
        # gate_escalated_at stamp above are already durable, so a severed
        # connection on this trailing write must not propagate and mask them
        # behind a raw exception.
        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
            logger.info(
                'DeterministicRunner: task %s blocked at deterministic gate', task_id,
            )
        except Exception as exc:
            logger.warning(
                'DeterministicRunner: task %s blocked-status writeback failed '
                '(connection may still be severed): %s: %s — the milestone_gate '
                'escalation is already durable on local disk, so returning '
                'BLOCKED regardless',
                task_id, type(exc).__name__, exc,
            )

        return WorkflowOutcome.BLOCKED

    async def _file_milestone_check_failed_and_block(
        self,
        task_id: str,
        summary: str,
        detail: str,
    ) -> WorkflowOutcome:
        """File a born-at-L2 ``milestone_check_failed`` escalation and block (γ-predicate).

        A non-zero predicate exit code is a milestone VERDICT ("the invariant
        does not hold") — semantically distinct from an infra fault, so this
        mirrors ``_file_infra_issue_and_block``'s dedup/file/block shape but
        with a dedicated category AND additionally stamps ``gate_escalated_at``
        (mirroring ``_file_milestone_gate_and_block``) so the next dispatch
        routes through section-1's existing quiescence/resolve-to-done fork —
        a human resolving the escalation drives the task to done with no
        extra wiring.

        Includes a dedup guard — if a pending escalation already exists (e.g.
        prior crash-safe re-dispatch), filing is skipped to avoid duplicate L2
        escalations.

        Returns:
            WorkflowOutcome.BLOCKED
        """
        from escalation.models import Escalation

        existing_pending = self.escalation_queue.get_by_task(
            task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
        )
        if existing_pending:
            logger.info(
                'DeterministicRunner: task %s already has %d pending escalation(s) — '
                'skipping re-file (milestone_check_failed dedup guard)',
                task_id, len(existing_pending),
            )
        else:
            esc = Escalation(
                id=self.escalation_queue.make_id(task_id),
                task_id=task_id,
                agent_role=DETERMINISTIC_AGENT_ROLE,
                severity='critical',
                category='milestone_check_failed',
                summary=summary[:200],
                detail=detail,
                level=2,
            )
            self.escalation_queue.submit(esc)
            logger.info(
                'DeterministicRunner: filed L2 milestone_check_failed escalation %s for task %s',
                esc.id, task_id,
            )

        # Stamp gate_escalated_at AFTER successful escalation submit (file-before-stamp
        # ordering, same as _file_milestone_gate_and_block) so a stamp failure re-files
        # on the next dispatch rather than silently skipping it.  A predicate NEVER
        # stamps before_done_ran_at — there is no I1 side effect to guard here.
        now_iso = datetime.now(UTC).isoformat()
        await self.scheduler.update_task(
            task_id,
            {'gate_escalated_at': now_iso},
            metadata_mode='merge',
        )

        # Best-effort, same as _file_infra_issue_and_block (reviewer amendment):
        # the escalation + gate_escalated_at stamp above are already durable, so
        # a severed connection on this trailing write must not propagate and
        # mask them behind a raw exception.
        try:
            await self.scheduler.set_task_status(task_id, 'blocked')
            logger.info(
                'DeterministicRunner: task %s blocked — milestone_check_failed', task_id,
            )
        except Exception as exc:
            logger.warning(
                'DeterministicRunner: task %s blocked-status writeback failed '
                '(connection may still be severed): %s: %s — the '
                'milestone_check_failed escalation is already durable on local '
                'disk, so returning BLOCKED regardless',
                task_id, type(exc).__name__, exc,
            )

        return WorkflowOutcome.BLOCKED

    async def _run_predicate(
        self, task_id: str, before_done: dict, description: str,
    ) -> WorkflowOutcome:
        """Run a read-only predicate check and map its exit code to a verdict (γ-predicate).

        Unlike the ``before_done`` deploy path, a predicate is a READ-ONLY
        exit-code verdict check — NOT a systemd deploy.  There is no unit to
        inspect, no baseline/fresh-PID verify, and no ``before_done_ran_at``
        I1 stamp (re-running a read-only check on crash-resume is harmless;
        see the module docstring's γ-predicate subsection).

        Called from two sites: the first dispatch (section 2), and section 1's
        resume branch once a ``milestone_check_failed`` escalation is
        resolved — resume re-invokes this method to re-verify the invariant
        rather than trusting the resolution blindly (a human may resolve the
        escalation without the underlying check now passing).

        ``metadata.always_escalates`` is never consulted here — see the
        module docstring's γ-predicate subsection.

        Maps the outcome:
        - ``rc == 0`` -> ``WorkflowOutcome.DONE``, with
          ``done_provenance.kind='deterministic-milestone'``.
        - ``rc != 0`` -> a milestone VERDICT failure: born-at-L2
          ``milestone_check_failed`` escalation + ``gate_escalated_at`` stamp
          + blocked (routes through section-1's resume/quiescence machinery
          on the next dispatch).
        - Timeout / unexpected error -> an INFRA fault (no verdict was
          produced): born-at-L2 ``infra_issue`` escalation + blocked
          (re-attempted on the next dispatch, no ``gate_escalated_at`` stamp).

        Returns:
            WorkflowOutcome.DONE or WorkflowOutcome.BLOCKED.
        """
        run_fn = self._script_runner or self._default_run_script
        outer_timeout = before_done.get('timeout_secs', 60) + self._run_timeout_grace_secs

        async def _invoke_run_fn():
            # See run()'s identical inner wrapper: translate a seam-internal
            # TimeoutError into a distinct exception type here so `except
            # TimeoutError` below can only ever mean "the outer wall-clock
            # guard itself fired" — never a misattributed application error.
            try:
                return await run_fn(before_done)
            except TimeoutError as exc:
                raise RuntimeError(
                    f'run_fn raised TimeoutError internally (not the '
                    f'outer guard): {exc!r}'
                ) from exc

        try:
            rc, out = await asyncio.wait_for(_invoke_run_fn(), timeout=outer_timeout)
        except TimeoutError:
            # A hung/unresponsive seam produced NO exit code, so there is no
            # verdict to report — this is an INFRA fault (parity with the
            # deploy path's outer-guard handling), not milestone_check_failed.
            # infra_issue does NOT stamp gate_escalated_at, so the check is
            # re-attempted on the next dispatch rather than latched into the
            # resolve-to-done path.
            timeout_detail = '\n'.join([
                description,
                f'Predicate check run_fn exceeded the outer guard timeout ({outer_timeout}s = '
                f"before_done['timeout_secs'] + run_timeout_grace_secs).",
                'The subprocess may be detached/unkillable — check out-of-band '
                '(e.g. ps) before taking further action.',
                'This is a read-only predicate — safe to re-run on the next '
                'dispatch (no before_done_ran_at stamp to worry about).',
            ])
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Predicate check timed out (subprocess hung)',
                detail=timeout_detail,
            )
        except Exception as exc:
            # Likewise an infra fault, not a verdict — an unexpected error
            # means the check never ran to completion.
            error_detail = '\n'.join([
                description,
                f'Predicate check run_fn raised an unexpected error: {exc!r}',
            ])
            return await self._file_infra_issue_and_block(
                task_id,
                summary='Predicate check run_fn failed (unexpected error)',
                detail=error_detail,
            )

        if rc != 0:
            # A non-zero exit is a milestone VERDICT ("invariant does not
            # hold"), not an infra fault — file milestone_check_failed (NOT
            # infra_issue) and stamp gate_escalated_at so a human resolving
            # the escalation drives the task to done on the next dispatch.
            fail_detail = '\n'.join([
                description,
                f'Predicate check exit code: rc={rc}',
                f'Output:\n{out}',
            ])
            return await self._file_milestone_check_failed_and_block(
                task_id,
                summary=f'Milestone predicate check failed (rc={rc})',
                detail=fail_detail,
            )

        logger.info(
            'DeterministicRunner: task %s predicate check passed (rc=0) — setting done',
            task_id,
        )
        await self.scheduler.set_task_status(
            task_id,
            'done',
            done_provenance=_build_done_provenance('deterministic-milestone', note=out),
        )
        return WorkflowOutcome.DONE

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
        task_id = str(assignment.task_id)
        task = assignment.task
        metadata = task.get('metadata') or {}
        description = task.get('description', '')
        before_done = metadata.get('before_done')
        always_escalates = metadata.get('always_escalates', False)
        gate_escalated_at = metadata.get('gate_escalated_at')

        # ── 1. Idempotency / quiescence ──────────────────────────────────────
        # If the gate escalation was already filed in a prior dispatch, check
        # whether it is still open or has been resolved.
        if gate_escalated_at:
            pending = self.escalation_queue.get_by_task(
                task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
            )
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
                # γ-predicate: a read-only predicate NEVER stamps before_done_ran_at
                # (no I1 side effect to prove), so it must short-circuit here, BEFORE
                # the before_done_ran_at proof-check below — otherwise a resolved
                # milestone_check_failed escalation would wrongly raise
                # NotImplementedError.
                #
                # Unlike a deploy (where the human's out-of-band fix IS the
                # resolution, and re-running risks a double-apply), a predicate
                # is a cheap, idempotent, READ-ONLY exit-code check — resolving
                # the escalation alone is NOT proof the invariant now holds (a
                # human may resolve it prematurely or in error).  So resume
                # RE-RUNS the check via _run_predicate rather than trusting the
                # resolution blindly (reviewer amendment): rc==0 drives to done
                # (deterministic-milestone, never 'deterministic-deploy' — no
                # deploy happened); rc!=0 re-files milestone_check_failed (the
                # dedup guard sees no pending escalation, since it was just
                # resolved) and stays blocked; a timeout/error routes to
                # infra_issue — identical to the first-dispatch path.
                if before_done is not None and before_done.get('kind') == 'predicate':
                    logger.info(
                        'DeterministicRunner: task %s predicate resume — gate '
                        'resolved — re-running predicate check (read-only, '
                        'safe to repeat) before trusting the resolution',
                        task_id,
                    )
                    return await self._run_predicate(task_id, before_done, description)

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
                        done_provenance=_build_done_provenance(
                            'deterministic-deploy',
                            unit=_ata_unit,
                            note='resumed after gate resolution',
                        ),
                    )
                else:
                    # Pure-gate resume: no before_done action ran, so there is no
                    # deploy/unit/pid evidence — stamp a dedicated gate-kind
                    # provenance so the done write passes require_done_provenance
                    # without lying about a deploy having happened (task 2331).
                    await self.scheduler.set_task_status(
                        task_id,
                        'done',
                        done_provenance=_build_done_provenance(
                            'deterministic-gate',
                            note='pure gate resolved',
                        ),
                    )
                return WorkflowOutcome.DONE

        # ── 2. before_done execution (γ) ────────────────────────────────────
        # Cross-unit blocking deploy: stamp → baseline → run script → verify → done.
        # Self-target detection + detached systemd-run is deferred to ε.
        if before_done is not None:
            # γ-predicate: a read-only exit-code verdict check — NOT a deploy.
            # Dispatched FIRST, above target_unit/I1/baseline, so a predicate
            # never touches systemd and never stamps before_done_ran_at.  This
            # is also BEFORE `always_escalates` is read below — a predicate
            # task's always_escalates is intentionally never consulted (see
            # module docstring's γ-predicate subsection); a
            # kind='predicate' + always_escalates=True task is not rejected
            # here and simply behaves as a plain predicate.
            if before_done.get('kind') == 'predicate':
                return await self._run_predicate(task_id, before_done, description)

            target_unit: str = before_done.get('target_unit', '')
            before_done_ran_at = metadata.get('before_done_ran_at')

            # I1 once-only idempotency guard (parallel to β's gate_escalated_at branch):
            # if the deploy already ran, check whether its escalation is still open.
            if before_done_ran_at:
                pending = self.escalation_queue.get_by_task(
                    task_id, status='pending', agent_role=DETERMINISTIC_AGENT_ROLE,
                )
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
                # agent_role scopes this to the runner's OWN escalations — an
                # unrelated escalation sharing this task_id (e.g. a starvation-
                # watchdog filing) must never alias as proof a human resolved
                # THIS runner's failure (task 2120).
                ever_escalated = bool(self.escalation_queue.get_by_task(
                    task_id, agent_role=DETERMINISTIC_AGENT_ROLE,
                ))

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
                        done_provenance=_build_done_provenance(
                            'deterministic-deploy',
                            pid=metadata.get('before_done_verified_pid'),
                            note='resumed after verified deploy (crash before done write)',
                            unit=target_unit,
                        ),
                    )
                    return WorkflowOutcome.DONE

                before_done_scheduled_at_stamp = metadata.get('before_done_scheduled_at')
                if before_done_scheduled_at_stamp:
                    # (b-self) Self-restart was successfully scheduled
                    # (before_done_scheduled_at set) but the done write or gate
                    # filing did not complete (crash in the window after stamp).
                    _sched = before_done_scheduled_at_stamp if isinstance(before_done_scheduled_at_stamp, dict) else {}
                    logger.info(
                        'DeterministicRunner: task %s self-restart scheduled (%s) + '
                        'crash before done/gate write — resume path '
                        '(transient_unit=%s, always_escalates=%s)',
                        task_id, _sched.get('at', ''), _sched.get('transient_unit', ''),
                        always_escalates,
                    )
                    if not always_escalates:
                        # always_escalates=False: the transient unit is registered and WILL
                        # fire — driving to done here is safe and avoids a spurious
                        # crash-window L2.
                        await self.scheduler.set_task_status(
                            task_id,
                            'done',
                            done_provenance=_build_done_provenance(
                                'deterministic-deploy-scheduled',
                                unit=target_unit,
                                transient_unit=_sched.get('transient_unit', ''),
                                fire_delay_secs=_sched.get('fire_delay_secs', 0),
                                note='resumed after self-restart scheduled (crash before done write)',
                            ),
                        )
                        return WorkflowOutcome.DONE
                    # always_escalates=True (act-then-ask): the milestone gate must NOT
                    # be bypassed.  Re-file the gate and block; gate_escalated_at is
                    # stamped so the next resume routes through section-1 quiescence
                    # rather than entering this (b-self) branch again.
                    logger.info(
                        'DeterministicRunner: task %s scheduled-resume with '
                        'always_escalates=True — re-filing milestone gate (gate not bypassed)',
                        task_id,
                    )
                    return await self._file_milestone_gate_and_block(task_id, task, metadata)

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
                        done_provenance=_build_done_provenance(
                            'deterministic-deploy',
                            note='resumed after human resolution',
                            unit=target_unit,
                        ),
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
                    metadata=metadata,
                )

            # ── Stop-instruction guard (task 2509) ───────────────────────────
            # First dispatch of a non-predicate before_done task (reached here
            # only when before_done_ran_at was falsy above — a resume/idempotency
            # dispatch never reaches this point).  Scan the task description for
            # an explicit stop instruction BEFORE running the deploy and BEFORE
            # stamping before_done_ran_at — a stop instruction is the
            # highest-authority human directive and must never be self-
            # authorized around (task 2407's autonomous /unblock session did
            # exactly that; task 2273's SIGTERM-killed sibling session is the
            # "what right looks like" precedent this guard makes a self-halt
            # rather than relying on an external kill).
            stop_phrase = detect_stop_instruction(description)
            if stop_phrase:
                stop_detail = '\n'.join([
                    description,
                    f'Target unit: {target_unit}',
                    f'Explicit stop instruction detected: {stop_phrase!r}.',
                    'The deploy was NOT run and before_done_ran_at was NOT '
                    'stamped.  Recovery needs BOTH steps: (1) edit the task '
                    'description to remove the stop instruction — otherwise '
                    'the guard re-fires and re-escalates on the very next '
                    "dispatch — and (2) resolve this escalation (e.g. "
                    "resolve_issue action='resume' or 'restart') to move the "
                    'task back to pending; a blocked task is not '
                    're-dispatched automatically.  Alternatively, cancel the '
                    'task outright.',
                ])
                return await self._file_stop_instruction_and_block(
                    task_id,
                    summary=f'Explicit stop instruction — deploy withheld: {target_unit}',
                    detail=stop_detail,
                )

            # Stamp before_done_ran_at FIRST (crash-safe I1: stamp-before-run means a
            # crash mid-deploy leaves the stamp set → re-dispatch does NOT re-run).
            # This stamp is SHARED for both self-target and cross-unit paths so I1
            # holds for both (ε design decision 5). ζ: this is also the initial
            # deploy_state.phase write (None -> RAN), atomically merged in the
            # SAME update_task call (DS-1).
            now_iso = datetime.now(UTC).isoformat()
            await self._advance_deploy_phase(
                task_id, metadata, DeployPhase.RAN,
                evidence={'before_done_ran_at': now_iso},
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
                # Clamp to a sane minimum (5 s) so a task that sets
                # on_active_delay_secs=0 (or a non-positive value) cannot
                # produce --on-active=0, which would make the transient unit fire
                # effectively immediately — re-introducing the self-kill window
                # this detached-deferral design exists to prevent.
                on_active_secs = max(int(before_done.get('on_active_delay_secs', 60)), 5)

                async def _completion_failure(exc: Exception) -> WorkflowOutcome:
                    # Defense-in-depth (task 2004): ANY exception in the restart-fn ->
                    # stamp -> scheduled-done-write window (including a terminal write
                    # rejected by the provenance validator) must surface LOUDLY instead
                    # of bubbling past Harness._run_slot's generic `except Exception`
                    # handler, which returns a silent BLOCKED TaskReport with no
                    # escalation and no way to self-heal — the exact signature observed
                    # on tasks 1976/1982.  before_done_ran_at is already stamped (I1
                    # once-only), so it is always safe to resolve via a born-at-L2
                    # infra_issue rather than re-running the restart.
                    #
                    # Deliberately NOT used to guard the rc!=0 branch's own
                    # _file_infra_issue_and_block call below (task 2004 amendment): if
                    # that filing itself raises, letting it propagate avoids a
                    # misleading re-file under a "completion failed" summary — the
                    # dedup guard in _file_infra_issue_and_block makes a second filing
                    # harmless, but the mismatched summary/detail is not.
                    completion_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Transient unit: {transient_unit}',
                        f'Self-restart completion failed: {exc!r}',
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Self-restart completion failed: {target_unit}',
                        detail=completion_detail,
                        metadata=metadata,
                    )

                try:
                    restart_fn = self._restart_scheduler or self._default_schedule_detached_restart
                    rc, tail = await restart_fn(
                        before_done,
                        transient_unit=transient_unit,
                        on_active_secs=on_active_secs,
                        task_id=task_id,
                        summary=f'Self-restart scheduling failed: {target_unit}',
                    )
                except Exception as exc:
                    return await _completion_failure(exc)

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
                        metadata=metadata,
                    )

                try:
                    # Stamp before_done_scheduled_at — positive proof the transient unit
                    # was successfully registered.  If the orchestrator crashes between
                    # this stamp and the done write, the resume path (sub-case b-self
                    # above) drives to done with scheduled provenance instead of
                    # re-escalating as a generic crash-window.  ζ: this is also the
                    # ran->scheduled phase advance, atomically merged in the SAME
                    # update_task call (DS-1).
                    await self._advance_deploy_phase(
                        task_id, metadata, DeployPhase.SCHEDULED,
                        evidence={'before_done_scheduled_at': {
                            'at': datetime.now(UTC).isoformat(),
                            'transient_unit': transient_unit,
                            'fire_delay_secs': on_active_secs,
                        }},
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
                            done_provenance=_build_done_provenance(
                                'deterministic-deploy-scheduled',
                                unit=target_unit,
                                transient_unit=transient_unit,
                                fire_delay_secs=on_active_secs,
                            ),
                        )
                        return WorkflowOutcome.DONE
                except Exception as exc:
                    return await _completion_failure(exc)
                # always_escalates=True (act-then-ask, non-exemplar): fall through
                # to the gate (section 3) WITHOUT running the blocking cross-unit
                # deploy.  The `if not self_target:` guard below ensures the
                # cross-unit baseline→run→verify block is SKIPPED entirely so the
                # orchestrator is not self-killed and the restart is not double-deployed.
                logger.info(
                    'DeterministicRunner: task %s self-restart scheduled with '
                    'always_escalates=True — falling through to gate (no cross-unit deploy)',
                    task_id,
                )

            # ── end ε self-target branch ─────────────────────────────────────
            # Cross-unit blocking deploy: guarded by `if not self_target:` so a
            # self-target task NEVER runs the blocking deploy against its own unit.
            # Self-target always_escalates=False already returned DONE above.
            # Self-target always_escalates=True falls through directly to the gate
            # (section 3) WITHOUT running the blocking deploy — this prevents:
            #   (a) self-kill: the blocking deploy would kill this runner mid-run;
            #   (b) double-deploy: the detached transient restart was already scheduled.
            if not self_target:
                # Capture baseline unit state before the deploy fires
                inspect_fn = self._unit_inspector or self._default_inspect_unit
                baseline = await inspect_fn(target_unit)

                # Task 2091 (baseline-leg hardening): a wedged/failed baseline
                # inspect returns the same MainPID=0/ActiveState='' sentinel
                # dict used on the verify leg (see _default_inspect_unit's
                # TimeoutError branch). On the VERIFY leg that sentinel is
                # already caught by the `pid > 0` half of the freshness check
                # below. On the BASELINE leg it is NOT: baseline_monotonic
                # would silently become 0, and `new_monotonic >
                # baseline_monotonic` is then trivially true for any active
                # unit — a wedged baseline would be swallowed and a deploy
                # falsely reported verified even though freshness was never
                # actually established. ActiveState is the signal: a real
                # `systemctl show` always populates it (even 'inactive' for a
                # nonexistent unit); only the timeout sentinel leaves it ''.
                # Fail closed exactly like the other pre-deploy failure paths
                # below — before_done_ran_at is already stamped (I1
                # once-only), so the deploy is NOT attempted on an untrusted
                # baseline.
                if not baseline.get('ActiveState'):
                    baseline_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Baseline inspect failed/wedged before deploy: {baseline!r}',
                        'Cannot establish a trustworthy pre-deploy baseline — '
                        'the deploy was NOT attempted (before_done_ran_at is '
                        'already stamped; I1 once-only — a human must inspect '
                        'the unit and resolve).',
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Baseline inspect failed before deploy: {target_unit}',
                        detail=baseline_detail,
                        metadata=metadata,
                    )

                # ζ DS-3: persist the trustworthy baseline into deploy_state as a
                # same-phase enrichment (phase stays RAN — see
                # _enrich_deploy_state_baseline). Best-effort; never blocks the
                # deploy itself.
                await self._enrich_deploy_state_baseline(task_id, metadata, baseline)

                # Run the deploy script + fresh-PID verify by delegating to
                # proc_supervision.RestartPlan.execute() (task 2238/δ, RP-2/
                # RP-5) instead of an inline run-then-reinspect block.  The
                # runner still owns: the pre-flight baseline above, the outer
                # wall-clock guard below (task 2090 Layer B — now bounding the
                # WHOLE execute() call, run + verify, rather than only the
                # run), and the disposition -> outcome mapping.
                # proc_supervision owns the RP-2 blocking-run/RP-5 fresh-PID-
                # verify mechanics.
                #
                # The subprocess RUN itself is routed through the runner's own
                # task-2090-hardened script_runner/_default_run_script seam
                # via a create_subprocess_exec-compatible SHIM below — NOT
                # through execute()'s own bare (untimed, non-process-group-
                # aware) spawn — so Layer A (process-group kill of leaked
                # grandchildren) and the (before_done)->(rc, tail) seam
                # contract are both preserved.
                run_fn = self._script_runner or self._default_run_script
                outer_timeout = before_done.get('timeout_secs', 60) + self._run_timeout_grace_secs

                async def _invoke_run_fn():
                    # A custom/injected seam could itself raise a TimeoutError
                    # internally (e.g. its own inner asyncio.wait_for) BEFORE
                    # outer_timeout elapses.  asyncio.wait_for cannot tell that
                    # apart from its OWN outer-guard timeout — both surface as
                    # the same `TimeoutError` type at the call site below.
                    # Translate a seam-internal TimeoutError into a distinct
                    # exception type HERE, before it can reach wait_for's
                    # propagation path, so `except TimeoutError` below can only
                    # ever mean "the outer wall-clock guard itself fired" —
                    # never a misattributed application error.
                    try:
                        return await run_fn(before_done)
                    except TimeoutError as exc:
                        raise RuntimeError(
                            f'run_fn raised TimeoutError internally (not the '
                            f'outer guard): {exc!r}'
                        ) from exc

                class _RunFnProcShim:
                    """Adapts a (rc, tail) pair from run_fn into a
                    create_subprocess_exec-compatible object for
                    RestartPlan._execute_cross_unit_blocking's injectable
                    `runner` seam (exposes .returncode + async communicate())."""

                    def __init__(self, rc: int, tail: str) -> None:
                        self.returncode = rc
                        self._tail = tail

                    async def communicate(self) -> tuple[bytes, None]:
                        return self._tail.encode(errors='replace'), None

                async def _shim_runner(*_args, **_kwargs):
                    rc, tail = await _invoke_run_fn()
                    return _RunFnProcShim(rc, tail)

                # RestartOutcome carries only disposition/escalated/detail —
                # not the verified MainPID/timestamp _writeback_deploy_success
                # needs for done_provenance.  This wrapper stashes execute()'s
                # single post-deploy re-inspect (via the SAME inspect_fn used
                # for the baseline above) so the fresh unit state survives
                # past plan.execute() without a second, wasteful/racy inspect.
                captured: dict = {}

                async def _capturing_inspector(unit: str, **_kwargs) -> dict:
                    state = await inspect_fn(unit)
                    captured['new_state'] = state
                    return state

                verify = FreshPidVerify(
                    baseline_active_enter_monotonic=baseline.get(
                        'ActiveEnterTimestampMonotonic', 0,
                    ),
                    baseline_main_pid=baseline.get('MainPID', 0),
                    inspect_timeout_secs=self._inspect_timeout_secs,
                )
                plan = RestartPlan(
                    script=Path(before_done['script']),
                    args=list(before_done.get('args') or []),
                    cwd=Path(before_done.get('cwd') or os.getcwd()).resolve(),
                    target_unit=target_unit,
                    # own_unit must be truthy and provably non-self here (this
                    # branch is only reached when the runner's OWN self_target
                    # check above already ruled out self-target), so
                    # RestartPlan.execute() takes the RP-2 cross-unit-blocking
                    # path — never RP-1's fail-closed refuse — even when
                    # ORCH_UNIT is unset (own_unit == ''), preserving the
                    # existing fail-open-to-cross-unit behaviour.
                    own_unit=own_unit or _CROSS_UNIT_OWN_UNIT_SENTINEL,
                    on_failure_escalation=None,
                    verify=verify,
                    transient_unit=None,
                )

                try:
                    outcome = await asyncio.wait_for(
                        plan.execute(runner=_shim_runner, inspector=_capturing_inspector),
                        timeout=outer_timeout,
                    )
                except TimeoutError:
                    # This guard now bounds the delegated plan.execute() call,
                    # which runs BOTH the deploy subprocess and the post-deploy
                    # verify re-inspect (_capturing_inspector -> inspect_fn) —
                    # asyncio.wait_for cannot tell which of the two was still
                    # pending when the timeout fired, so the message must not
                    # pin the blame solely on "the subprocess".
                    timeout_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Deploy run+verify exceeded the outer guard timeout ({outer_timeout}s = '
                        f"before_done['timeout_secs'] + run_timeout_grace_secs).",
                        'The hang may be in the deploy subprocess itself or in the post-deploy '
                        'unit inspect (verify) — the outer guard wraps both and cannot '
                        'distinguish which was pending. Check the unit out-of-band (e.g. '
                        'systemctl --user status, ps) before taking further action.',
                        'before_done_ran_at is already stamped (I1) — the deploy is NOT re-run.',
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy run+verify exceeded outer guard: {target_unit}',
                        detail=timeout_detail,
                        metadata=metadata,
                    )
                except Exception as exc:
                    error_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Deploy run_fn raised an unexpected error: {exc!r}',
                        'before_done_ran_at is already stamped (I1) — the deploy is NOT re-run.',
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy run_fn failed (unexpected error): {target_unit}',
                        detail=error_detail,
                        metadata=metadata,
                    )

                if outcome.disposition == RestartDisposition.RESTART_FAILED:
                    # B7a: script failed — file infra_issue escalation, set blocked (B7a)
                    deploy_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        outcome.detail,
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy failed: {target_unit}',
                        detail=deploy_detail,
                        metadata=metadata,
                    )

                if outcome.disposition == RestartDisposition.VERIFY_FAILED:
                    # B7b: verify failed — file infra_issue escalation, set blocked
                    verify_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        outcome.detail,
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy verify failed: {target_unit}',
                        detail=verify_detail,
                        metadata=metadata,
                    )

                if outcome.disposition != RestartDisposition.DEPLOYED_AND_VERIFIED:
                    # Defensive: REFUSED is structurally unreachable from this
                    # call site (own_unit is always forced truthy+non-self
                    # above, so RP-1 never refuses here) — never silently
                    # swallow an unexpected disposition rather than surfacing
                    # it as an infra_issue.
                    refused_detail = '\n'.join([
                        description,
                        f'Target unit: {target_unit}',
                        f'Unexpected RestartPlan disposition: {outcome.disposition!r}',
                        outcome.detail,
                    ])
                    return await self._file_infra_issue_and_block(
                        task_id,
                        summary=f'Deploy refused unexpectedly: {target_unit}',
                        detail=refused_detail,
                        metadata=metadata,
                    )

                new_state = captured.get('new_state', {})

                if not always_escalates:
                    # Pure cross-unit deploy (B6): verified → set done with provenance.
                    # Task 2066: the deploy may have severed the orchestrator's own
                    # fused-memory/MCP connection (e.g. it restarted the service
                    # backing that connection) — delegate to the connection-resilient
                    # writeback helper rather than a single unguarded write pair.
                    return await self._writeback_deploy_success(
                        task_id, metadata, new_state, target_unit, description,
                    )

                # always_escalates=True with before_done (cross-unit act-then-ask):
                # action already ran — fall through to the gate below.
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

        return await self._file_milestone_gate_and_block(task_id, task, metadata)
