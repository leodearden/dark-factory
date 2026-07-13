"""Single owned restart-seam (M1) — ``RestartPlan`` + async ``execute()`` (task 2237, W10-gamma).

Hosts the one restart-execution contract every restart caller in the
orchestrator package builds a plan against: ``RestartPlan``,
``EscalationSpec``, ``FreshPidVerify``, ``RestartOutcome``/
``RestartDisposition``, and a single ``async def execute()`` honoring 5
invariants (RP-1..5, PRD Sec 5.1):

- RP-1 (fail-closed self-kill guard): a BLOCKING restart (``verify`` set)
  with an unknown ``own_unit`` refuses rather than risking a same-unit
  ``systemctl restart`` SIGKILL of the caller (the 2064 bug — today's
  ``self_target = bool(own_unit) and (target_unit == own_unit)`` fails OPEN
  when ``own_unit`` is falsy).
- RP-2 (cross-unit blocking + RP-5 verify): a provably different target unit
  runs the restart script to completion and re-inspects for a fresh PID.
- RP-3 (no implicit cwd — the 2105 fix): every spawn carries an explicit,
  absolute cwd; a detached systemd-run transient unit additionally gets
  ``--working-directory=<cwd>`` since ``systemd --user`` defaults to $HOME.
- RP-4 (on-failure escalation wrapper): a detached ``--on-active`` systemd-run
  payload is wrapped in ``/bin/sh -c`` so a FIRE-TIME failure (after this
  call has already returned) still files a born-at-L2 escalation.
- RP-5 (persisted verify baseline — the 2074 caveat): the fresh-PID baseline
  is a caller-persisted field on ``FreshPidVerify``, never a local re-inspect.

IMPORTS :func:`orchestrator.systemd_inspect.inspect_systemd_unit` — never a
second copy of the ``systemctl --user show`` subprocess call (program seam
table: "never a second copy"; systemd_inspect.py's own module doc
anticipates this import).

Deliberately carries NO orchestrator-instance state (no
DeterministicRunner/Harness/EscalationQueue handle) — mirrors
systemd_inspect.py's statelessness. Escalations are filed from an
``EscalationSpec`` that itself carries ``queue_dir``+``task_id``: in-process
(RP-1/RP-5) via ``EscalationQueue(Path(spec.queue_dir)).submit(spec.to_escalation(queue))``,
and in the RP-4 shell branch via ``spec.to_submit_argv(sys.executable)``
(``python -m escalation submit ...``).

RP-2/RP-5's cross-unit BLOCKING verify path (``_execute_cross_unit_blocking``,
``FreshPidVerify``, the ``DEPLOYED_AND_VERIFIED``/``VERIFY_FAILED``/
``RESTART_FAILED`` dispositions) has no production caller in this task —
every caller converted here (``service_restart.py``'s two mechanisms) passes
``verify=None``. This is not speculative: it is the full RP-1..5 contract
this module is scoped to provide (PRD Sec 5.1), and its first real caller is
already filed and tracked — task 2238 (W10-delta, "DeterministicRunner
detached + blocking-verify restart paths delegate to RestartPlan.execute()"),
which converts ``deterministic_runner.py``'s existing blocking cross-unit
run + inline fresh-PID verify (:1267, :1310-1347) to build a ``FreshPidVerify``
and delegate here. Kept correct by the RP-2/RP-4/RP-5 test cells in
test_proc_supervision.py pending that conversion landing.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import sys
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator.systemd_inspect import inspect_systemd_unit

if TYPE_CHECKING:
    from escalation.models import Escalation
    from escalation.queue import EscalationQueue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EscalationSpec:
    """Everything needed to file a born-at-L2 escalation, in-process or via CLI.

    Carries no live queue/connection — ``queue_dir`` + ``task_id`` are plain
    data so a ``RestartPlan`` (and the ``/bin/sh -c`` wrapper it may build)
    stays stateless and reusable by every future restart caller.
    """

    queue_dir: str
    task_id: str
    summary: str
    detail: str = ''
    severity: str = 'critical'
    category: str = 'infra_issue'
    agent_role: str = 'orchestrator-deterministic'

    def to_escalation(self, queue: EscalationQueue) -> Escalation:
        """Build the in-process ``Escalation`` record (RP-1/RP-5 filing path).

        ``level=2`` plus the sentinel ``agent_role`` default
        ('orchestrator-deterministic') mirrors
        ``DeterministicRunner._file_infra_issue_and_block``'s born-at-L2
        pattern (deterministic_runner.py:587-641) — the id is minted from the
        queue's durable per-task_id counter (``queue.make_id``), never
        derived locally.
        """
        from escalation.models import Escalation

        return Escalation(
            id=queue.make_id(self.task_id),
            task_id=self.task_id,
            agent_role=self.agent_role,
            severity=self.severity,
            category=self.category,
            summary=self.summary[:200],
            detail=self.detail,
            level=2,
        )

    def to_submit_argv(self, python_exe: str) -> list[str]:
        """Build the ``python -m escalation submit ...`` argv (RP-4 shell branch).

        Mirrors ``DeterministicRunner._default_schedule_detached_restart``'s
        ``escalation_cmd`` list (deterministic_runner.py:414-427) exactly, so
        the two restart mechanisms produce byte-identical submit invocations.
        """
        return [
            python_exe, '-m', 'escalation', 'submit',
            '--queue-dir', self.queue_dir,
            '--task', self.task_id,
            '--severity', self.severity,
            '--category', self.category,
            '--summary', self.summary[:200],
            '--agent-role', self.agent_role,
            '--detail', self.detail,
        ]


def _file_inprocess_escalation(
    spec: EscalationSpec, *, summary: str | None = None, detail: str | None = None,
) -> bool:
    """File *spec*'s L2 escalation in-process (RP-1/RP-5 filing path).

    Reused by both the RP-1 fail-closed refuse and the RP-5 verify-fail/
    restart-fail branches — one filing routine for every in-process escalation
    ``RestartPlan.execute()`` can produce.

    *summary*/*detail*, when given, OVERRIDE ``spec.summary``/``spec.detail``
    for this filing only (``spec`` itself is frozen and unchanged). RP-5's
    verify-fail/restart-fail branches use this to carry the target_unit and
    the pid/monotonic values discovered at ``execute()``-time, which the
    caller could not have known when constructing the ``RestartPlan``/
    ``EscalationSpec`` up front (mirrors deterministic_runner.py:1608-1620,
    1633-1652's per-failure detail text).

    Includes a dedup guard mirroring
    ``DeterministicRunner._file_infra_issue_and_block`` (deterministic_runner.py:
    611-620): if a pending escalation already exists for ``spec.task_id`` +
    ``spec.agent_role``, filing is skipped so a crash-safe re-dispatch of the
    same plan does not double-file L2 escalations.

    Returns ``True`` iff a NEW escalation was actually submitted this call,
    ``False`` when the dedup guard skipped filing because a pending
    escalation for ``spec.task_id``+``spec.agent_role`` already existed.
    Callers propagate this into ``RestartOutcome.escalated`` so ``escalated``
    means "this call filed a new L2 escalation", never "an escalation exists
    for this task" (which could also be true from an earlier call).
    """
    from escalation.queue import EscalationQueue

    queue = EscalationQueue(Path(spec.queue_dir))
    existing_pending = queue.get_by_task(
        spec.task_id, status='pending', agent_role=spec.agent_role,
    )
    if existing_pending:
        logger.info(
            'proc_supervision: task %s already has %d pending escalation(s) — '
            'skipping re-file (dedup guard)',
            spec.task_id, len(existing_pending),
        )
        return False
    if summary is not None or detail is not None:
        spec = replace(
            spec,
            summary=summary if summary is not None else spec.summary,
            detail=detail if detail is not None else spec.detail,
        )
    esc = spec.to_escalation(queue)
    queue.submit(esc)
    logger.info(
        'proc_supervision: filed L2 %s escalation %s for task %s',
        spec.category, esc.id, spec.task_id,
    )
    return True


@dataclass(frozen=True)
class FreshPidVerify:
    """RP-5 fresh-PID verify parameters.

    ``baseline_active_enter_monotonic``/``baseline_main_pid`` are a
    CALLER-PERSISTED field, not a local re-inspect captured inside
    ``execute()`` (the 2074 caveat) — the caller must inspect the target
    unit and persist the baseline BEFORE invoking the restart.

    Both baseline fields participate in ``_execute_cross_unit_blocking``'s
    freshness check: a re-inspected unit counts as "fresh" only when its new
    MainPID is live (``> 0``) AND differs from ``baseline_main_pid`` AND its
    ``ActiveEnterTimestampMonotonic`` is strictly later than
    ``baseline_active_enter_monotonic``. Comparing the persisted PID is a
    genuinely stronger identity signal than the monotonic timestamp alone
    (deterministic_runner.py:1624-1630's reference check uses monotonic only,
    since it never captures a baseline PID to compare) — it requires the
    process identity to have actually changed, not merely the unit's
    activation clock.
    """

    baseline_active_enter_monotonic: int
    baseline_main_pid: int
    inspect_timeout_secs: float


class RestartDisposition(StrEnum):
    """Outcome classification for a ``RestartPlan.execute()`` call."""

    REFUSED = 'refused'
    SCHEDULED = 'scheduled'
    DEPLOYED_AND_VERIFIED = 'deployed_and_verified'
    VERIFY_FAILED = 'verify_failed'
    REGISTRATION_FAILED = 'registration_failed'
    RESTART_FAILED = 'restart_failed'


@dataclass(frozen=True)
class RestartOutcome:
    """Result of a ``RestartPlan.execute()`` call.

    ``escalated`` means "this call itself filed a NEW L2 escalation" — not
    "an escalation exists for this task". When ``on_failure_escalation`` is
    configured but ``_file_inprocess_escalation``'s dedup guard skips filing
    because a pending escalation for the same task_id+agent_role already
    exists (e.g. filed by an earlier, crash-recovered call), ``escalated`` is
    ``False`` even though an escalation record for this task does exist.
    """

    disposition: RestartDisposition
    escalated: bool = False
    detail: str = ''


@dataclass(frozen=True)
class RestartPlan:
    """A fully-specified restart, ready to ``execute()``.

    ``__post_init__`` enforces RP-3's "no implicit cwd" structurally: a
    non-absolute ``cwd`` raises ``ValueError`` (the 2105 fix — a systemd
    ``--user`` transient unit's cwd otherwise defaults to $HOME), and a
    relative ``script`` is absolutized against ``cwd`` so every downstream
    argv carries an absolute path. Script existence is deliberately NOT
    stat-checked here (would race / force hermetic tests to create real
    files) — a missing script surfaces at runtime as exit-127, escalated by
    the RP-4 on-failure wrapper.
    """

    script: Path
    args: list[str]
    cwd: Path
    target_unit: str
    own_unit: str | None
    on_failure_escalation: EscalationSpec | None
    verify: FreshPidVerify | None
    transient_unit: str | None = None
    on_active_secs: int = 60

    def __post_init__(self) -> None:
        if not self.cwd.is_absolute():
            raise ValueError(
                f'RestartPlan.cwd must be an absolute path (no implicit cwd — the '
                f'2105 fix); got {self.cwd!r}'
            )
        if not self.script.is_absolute():
            object.__setattr__(self, 'script', self.cwd / self.script)

    async def execute(self, *, runner=None, inspector=None) -> RestartOutcome:
        """Run this restart plan, honoring RP-1..5.

        ``runner`` defaults to ``asyncio.create_subprocess_exec``; ``inspector``
        defaults to :func:`orchestrator.systemd_inspect.inspect_systemd_unit`.
        Both are injectable seams for tests.

        Decision tree (own=own_unit or ''; wants_blocking=verify is not None;
        self_target=bool(own) and target_unit==own):

        1. wants_blocking and not own -> RP-1 fail-closed REFUSE (implemented
           below, step-8).
        2. wants_blocking and not self_target -> RP-2 cross-unit BLOCKING +
           RP-5 verify (implemented below, step-10/12).
           (wants_blocking and self_target — a KNOWN own_unit provably equal
           to target_unit — deliberately does NOT reach this branch: RP-1's
           safety proof forbids ever blocking a same-unit restart.
           * If transient_unit is ALSO unset, this REFUSES too (2a below),
             same shape as 1 — the only other path, the leaf plain-spawn in
             4, is an IMMEDIATE synchronous spawn, not deferred, so it is
             exactly the 2064-class self-kill hazard RP-1 exists to prevent;
             being "detached" in the fire-and-forget sense does not make an
             immediate same-unit restart safe.
           * Only when transient_unit IS set does this fall through to 3,
             with ``verify`` dropped and a warning logged (a fresh-PID check
             would be meaningless for a fire-and-forget restart that hasn't
             happened yet) — see the guards immediately before the
             ``transient_unit`` check.)
        2a. wants_blocking and self_target and not transient_unit -> REFUSE
            (implemented below, alongside 1's branch).
        3. transient_unit set -> DETACHED systemd-run (RP-3/RP-4; implemented
           below, step-4). Reached either with wants_blocking False, or via
           2's fallthrough (wants_blocking True, self_target True, verify
           dropped).
        4. else -> DETACHED leaf plain-spawn, fused-memory/dashboard parity
           (implemented below, step-16). Only reachable with wants_blocking
           False — see 2a.
        """
        runner = runner or asyncio.create_subprocess_exec
        inspector = inspector or inspect_systemd_unit

        own = self.own_unit or ''
        wants_blocking = self.verify is not None
        self_target = bool(own) and self.target_unit == own

        if wants_blocking and not own:
            # RP-1 fail-closed refuse (the 2064 self-kill guard): a blocking
            # restart's safety proof requires a KNOWN own_unit to show
            # target_unit != own_unit. own_unit unknown -> cannot prove ->
            # refuse BEFORE touching runner/inspector — no blocking subprocess
            # is ever spawned in this branch.
            detail = (
                f'Refusing blocking restart of target_unit={self.target_unit!r}: '
                f'own_unit is unknown ({self.own_unit!r}), so this process cannot '
                f'prove target_unit != own_unit before running a synchronous '
                f'restart that could SIGKILL itself (the 2064 self-kill bug). '
                f'No restart subprocess was spawned.'
            )
            escalated = False
            if self.on_failure_escalation is not None:
                escalated = _file_inprocess_escalation(self.on_failure_escalation)
            else:
                logger.warning(
                    'proc_supervision: RP-1 refused a blocking restart of %s '
                    '(own_unit unknown) but no on_failure_escalation was '
                    'configured — no L2 escalation filed for this refusal',
                    self.target_unit,
                )
            return RestartOutcome(
                disposition=RestartDisposition.REFUSED,
                escalated=escalated,
                detail=detail,
            )

        if wants_blocking and not self_target:
            return await self._execute_cross_unit_blocking(runner, inspector)

        if wants_blocking:
            # Reaching here with wants_blocking still True implies self_target
            # is True (the two branches above already handled "not own" and
            # "not self_target"): a KNOWN own_unit provably equal to
            # target_unit. RP-1's safety proof forbids ever blocking a
            # same-unit restart (that is precisely the 2064 self-kill risk
            # this guard exists to prevent), so this never routes to
            # _execute_cross_unit_blocking.
            if not self.transient_unit:
                # RP-1 fail-closed REFUSE, cell 2a: with no transient_unit,
                # the only remaining path is the LEAF plain-spawn branch
                # below (``_execute_detached_leaf_plain_spawn``), which is an
                # IMMEDIATE, synchronous ``create_subprocess_exec`` — not
                # deferred. For a same-unit restart script (e.g. one that
                # runs ``systemctl --user restart <own_unit>``), that
                # immediate spawn can itself SIGKILL this very process before
                # this call returns — the exact 2064-class hazard RP-1 exists
                # to prevent. Being "detached" in the fire-and-forget sense
                # does not make an immediate same-unit restart safe, so this
                # refuses rather than risk it, exactly mirroring the RP-1
                # branch above. A caller that legitimately wants a
                # same-unit verified restart must set transient_unit to
                # schedule it DEFERRED instead (the branch just below).
                detail = (
                    f'Refusing blocking restart of target_unit={self.target_unit!r}: '
                    f'target_unit == own_unit ({own!r}) and no transient_unit was '
                    f'set, so the only remaining path is an IMMEDIATE leaf spawn — '
                    f'not deferred — which could SIGKILL this process before this '
                    f'call returns (the 2064 self-kill bug). Set transient_unit to '
                    f'schedule a same-unit restart DETACHED instead. No restart '
                    f'subprocess was spawned.'
                )
                escalated = False
                if self.on_failure_escalation is not None:
                    escalated = _file_inprocess_escalation(self.on_failure_escalation)
                else:
                    logger.warning(
                        'proc_supervision: refused a same-unit blocking restart '
                        'of %s (no transient_unit) but no on_failure_escalation '
                        'was configured — no L2 escalation filed for this '
                        'refusal',
                        self.target_unit,
                    )
                return RestartOutcome(
                    disposition=RestartDisposition.REFUSED,
                    escalated=escalated,
                    detail=detail,
                )
            # transient_unit IS set: the DEFERRED systemd-run path below never
            # blocks the event loop (it registers now, fires LATER), so it is
            # safe even for a same-unit restart — but the caller-supplied
            # ``verify`` is still INTENTIONALLY dropped: a fresh-PID check
            # would be meaningless for a fire-and-forget restart that hasn't
            # happened yet. Log it so the degrade is observable rather than a
            # future caller mistaking a SCHEDULED outcome for a verified one.
            logger.warning(
                'proc_supervision: verify was set on a same-unit plan '
                '(target_unit=%r == own_unit=%r) — blocking verify is never '
                'used for a same-unit restart (would risk the 2064 self-kill '
                'bug); downgrading to the detached systemd-run path with '
                'verify dropped. No fresh-PID check will run for this restart.',
                self.target_unit, own,
            )

        if self.transient_unit:
            return await self._execute_detached_systemd_run(runner)

        return await self._execute_detached_leaf_plain_spawn(runner)

    async def _execute_cross_unit_blocking(self, runner, inspector) -> RestartOutcome:
        """RP-2/RP-5: run a BLOCKING restart against a provably different unit.

        Unlike the detached systemd-run path, this runs the restart script to
        completion in-process (no ``/bin/sh`` wrapper, no transient unit) and
        then re-inspects the target unit for a fresh MainPID and a
        strictly-later monotonic timestamp than the caller-persisted
        ``self.verify.baseline_active_enter_monotonic`` (RP-5 — the 2074
        caveat: the baseline is a FIELD on ``FreshPidVerify``, never a local
        re-inspect captured inside this call). This method is only reachable
        when ``self_target`` is False, i.e. ``own_unit`` is known AND
        provably differs from ``target_unit`` (RP-1 already refused the
        unknown-own case in ``execute()`` above) — so a synchronous restart
        here can never SIGKILL the caller.

        Models deterministic_runner.py:1546-1652's blocking-deploy-then-verify
        shape: run to completion, re-inspect, then check freshness. This seam
        deliberately checks a stronger condition than
        deterministic_runner.py:1624-1630's reference
        ``fresh = isinstance(pid, int) and pid > 0 and new_monotonic > baseline_monotonic``:
        because ``FreshPidVerify`` carries a persisted ``baseline_main_pid``
        (which deterministic_runner.py's local baseline dict never captures),
        this method also requires the re-inspected MainPID to differ from
        that baseline — ``fresh = isinstance(pid, int) and pid > 0 and pid !=
        baseline_main_pid and new_monotonic > baseline_monotonic`` — so a
        "fresh" verdict requires the process identity to have actually
        changed, not just the unit's activation clock ticking forward. A
        script failure (rc != 0) or a non-fresh re-inspect never falsely
        reports DEPLOYED_AND_VERIFIED — both escalate via the shared
        ``_file_inprocess_escalation`` filing routine (skipped, with a logged
        warning, when no ``on_failure_escalation`` was configured for this
        plan).
        """
        assert self.verify is not None, 'router only calls this when wants_blocking'

        proc = await runner(
            str(self.script), *self.args,
            cwd=str(self.cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        rc = proc.returncode or 0
        if rc != 0:
            # Script failed outright — no point inspecting the target unit.
            tail = (stdout or b'').decode(errors='replace')[-2000:]
            detail = (
                f'Cross-unit restart of {self.target_unit!r} failed: script '
                f'exit code rc={rc}. Output:\n{tail}'
            )
            escalated = self._file_verify_stage_escalation(
                summary=f'Restart failed: {self.target_unit}',
                detail=detail,
                log_context=f'restart of {self.target_unit} failed (rc={rc})',
            )
            return RestartOutcome(
                disposition=RestartDisposition.RESTART_FAILED,
                escalated=escalated,
                detail=detail,
            )

        new_state = await inspector(
            self.target_unit, timeout_secs=self.verify.inspect_timeout_secs,
        )
        pid = new_state.get('MainPID', 0)
        new_monotonic = new_state.get('ActiveEnterTimestampMonotonic', 0)
        baseline_monotonic = self.verify.baseline_active_enter_monotonic
        baseline_pid = self.verify.baseline_main_pid
        fresh = (
            isinstance(pid, int)
            and pid > 0
            and pid != baseline_pid
            and new_monotonic > baseline_monotonic
        )
        if not fresh:
            detail = (
                f'Cross-unit restart verify failed for {self.target_unit!r}: '
                f'new MainPID={pid!r} baseline_pid={baseline_pid} '
                f'new_monotonic={new_monotonic} '
                f'baseline_monotonic={baseline_monotonic}. Expected a fresh '
                f'non-sentinel MainPID (>0, and different from baseline_pid) '
                f'and a strictly-later ActiveEnterTimestampMonotonic after '
                f'the restart.'
            )
            escalated = self._file_verify_stage_escalation(
                summary=f'Restart verify failed: {self.target_unit}',
                detail=detail,
                log_context=f'restart verify failed for {self.target_unit}',
            )
            return RestartOutcome(
                disposition=RestartDisposition.VERIFY_FAILED,
                escalated=escalated,
                detail=detail,
            )

        return RestartOutcome(disposition=RestartDisposition.DEPLOYED_AND_VERIFIED)

    def _file_verify_stage_escalation(
        self, *, summary: str, detail: str, log_context: str,
    ) -> bool:
        """File ``self.on_failure_escalation`` (if configured) and report whether a NEW escalation was filed.

        Shared by the RESTART_FAILED and VERIFY_FAILED branches above — both
        need the identical "file if configured, else log a warning and don't
        claim escalated" fallback that RP-1's refuse branch also uses.

        Returns ``False`` both when unconfigured (nothing to file) and when
        ``_file_inprocess_escalation``'s dedup guard skipped filing an
        already-pending duplicate — either way, no NEW escalation was filed
        by this call, so ``RestartOutcome.escalated`` (fed directly from this
        return value) means "this call filed a new L2", not "an escalation
        exists for this task".
        """
        if self.on_failure_escalation is None:
            logger.warning(
                'proc_supervision: %s but no on_failure_escalation was '
                'configured — no L2 escalation filed',
                log_context,
            )
            return False
        return _file_inprocess_escalation(
            self.on_failure_escalation, summary=summary, detail=detail,
        )

    async def _execute_detached_systemd_run(self, runner) -> RestartOutcome:
        """RP-3/RP-4: register a deferred ``systemd-run --user`` transient unit.

        Registers immediately and returns; the payload (script + args) fires
        LATER under the user systemd manager, so this never blocks or risks
        killing the caller — used both for a same-unit self-restart and for
        any other detached fire-and-forget-with-verification need.

        ``--working-directory=<cwd>`` (RP-3) guards against the ``systemd
        --user`` manager's $HOME-default cwd (the 2105 exit-127 bug: a
        relative deploy script is not found once a ``--user`` transient unit
        fires under $HOME) — the transient unit's payload always resolves
        relative to an explicit, absolute cwd instead. This mirrors
        ``DeterministicRunner._default_schedule_detached_restart``'s own 2105
        fix (deterministic_runner.py:388-394 cwd resolution, :434-438 script
        absolutization, :455 ``--working-directory=<cwd>``). Two independent
        layers enforce this here, belt-and-braces: ``RestartPlan.__post_init__``
        absolutizes ``self.script`` against ``self.cwd`` at construction time
        (so ``str(self.script)`` in the payload below is always absolute), and
        ``--working-directory=<self.cwd>`` is added to EVERY detached
        systemd-run argv UNCONDITIONALLY (not gated on whether the script
        happened to already be relative) — so the invariant holds even for a
        future caller that somehow bypasses ``__post_init__``. The
        ``/bin/sh -c`` wrapper's on-failure branch (RP-4) files a born-at-L2
        escalation via
        ``EscalationSpec.to_submit_argv`` ONLY when the deferred payload
        itself exits non-zero at fire time — never at registration time, so a
        successful self-deploy never spuriously escalates. When
        ``on_failure_escalation`` is None the payload is left unbranched (no
        wrapper) — still a valid ``/bin/sh -c`` invocation, just with no
        on-failure reporting.
        """
        on_active_secs = max(int(self.on_active_secs), 5)
        payload = ' '.join(shlex.quote(p) for p in [str(self.script), *self.args])
        if self.on_failure_escalation is not None:
            on_failure_argv = self.on_failure_escalation.to_submit_argv(sys.executable)
            on_failure = ' '.join(shlex.quote(p) for p in on_failure_argv)
            # Byte-for-byte reuse of deterministic_runner.py:439-445's wrapper
            # shape (`_default_schedule_detached_restart`) — same three-part
            # `payload; __rc=$?; if [ "$__rc" -ne 0 ]; then on_failure; fi;
            # exit "$__rc"` shell. This IS intentional interim duplication,
            # not an oversight: task 2237 (this module) is scoped OUT of
            # deterministic_runner.py — that file belongs to task 2237's
            # sibling, delta — so until delta lands, this wrapper and
            # EscalationSpec.to_submit_argv's submit argv are TWO independent
            # copies of the same shape that must be kept byte-identical by
            # hand/convention, pinned only by
            # TestDetachedWrapperExactnessAndRegistrationFailure (step-13)
            # here and its deterministic_runner.py counterpart. Tracked as a
            # follow-up (filed as a non-blocking escalation alongside this
            # comment) for delta to make DeterministicRunner's two mechanisms
            # delegate to this one helper (EscalationSpec.to_submit_argv /
            # RestartPlan) instead, retiring the hand-kept-identical coupling
            # with no behaviour delta.
            wrapped = (
                f'{payload}; __rc=$?; '
                f'if [ "$__rc" -ne 0 ]; then {on_failure}; fi; '
                f'exit "$__rc"'
            )
        else:
            wrapped = payload

        argv = [
            'systemd-run', '--user',
            f'--on-active={on_active_secs}',
            f'--unit={self.transient_unit}',
            '--collect',
            f'--working-directory={self.cwd}',
            '/bin/sh', '-c', wrapped,
        ]
        proc = await runner(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        rc = proc.returncode or 0
        tail = (stdout or b'').decode(errors='replace')[-2000:]
        if rc != 0:
            logger.warning(
                'proc_supervision: failed to register restart transient unit %s (rc=%d)',
                self.transient_unit, rc,
            )
            return RestartOutcome(
                disposition=RestartDisposition.REGISTRATION_FAILED,
                escalated=False,
                detail=f'systemd-run registration of {self.transient_unit} failed (rc={rc}): {tail}',
            )
        return RestartOutcome(disposition=RestartDisposition.SCHEDULED)

    async def _execute_detached_leaf_plain_spawn(self, runner) -> RestartOutcome:
        """Leaf DETACHED path (no ``transient_unit``, no ``verify``): a bare
        immediate ``create_subprocess_exec`` spawn — fused-memory/dashboard
        parity with ``service_restart._default_restart_executor``.

        That parity covers argv shape and fire-and-forget semantics, NOT a
        byte-identical spawn-kwargs set: this path passes ``cwd=str(self.cwd)``
        (the caller's ``project_root``) explicitly, whereas the pre-2237
        ``_default_restart_executor`` spawned with only
        ``start_new_session=True`` and no explicit ``cwd``, inheriting the
        orchestrator process's own cwd. This is a deliberate, narrow behavior
        change, not an oversight: RP-3's "no implicit cwd" invariant is
        applied uniformly to every ``RestartPlan`` spawn in this seam, not
        just the systemd-run path, so a future leaf caller can never regress
        into an implicit-cwd bug the way the systemd-run path once did (the
        2105 incident). It is confirmed safe for the two current leaf
        callers: ``scripts/restart-fused-memory.sh`` and
        ``scripts/restart-dashboard.sh`` are themselves cwd-insensitive (they
        act only via ``systemctl --user``/``curl``/``journalctl`` against
        absolute unit names and URLs, with no relative file access), and
        ``self.script`` is already absolutized by ``__post_init__``, so the
        script's own lookup never depended on cwd either way. A future leaf
        script that DOES rely on relative file access would need an absolute
        path — the same requirement RP-3 already imposes on the systemd-run
        path.

        No ``/bin/sh -c`` on-failure wrapper here (unlike RP-4's systemd-run
        payload): a plain immediate spawn has no DEFERRED fire-time gap to
        guard — a spawn failure (e.g. a missing/non-executable script) raises
        SYNCHRONOUSLY out of ``runner(...)``, straight into the caller's own
        try/except (``StaleServiceRestartCoordinator.maybe_restart`` branches
        on FileNotFoundError/PermissionError == permanent vs. any other
        Exception == transient-retry). RP-4's wrapper exists only because a
        deferred ``--on-active`` systemd-run payload's failure happens LATER,
        out-of-band from this call having already returned — there is no such
        gap here, so no wrapper is needed or wanted. Accordingly
        FileNotFoundError/PermissionError are deliberately NOT caught below —
        they propagate uncaught, preserving the coordinator's
        permanent-vs-transient retry contract.

        Fire-and-forget: the spawned process's exit is intentionally never
        awaited (mirrors ``_default_restart_executor``'s own comment) — the
        restart script runs detached so its own health-poll never blocks this
        event loop.
        """
        await runner(
            str(self.script), *self.args,
            cwd=str(self.cwd),
            start_new_session=True,
        )
        return RestartOutcome(disposition=RestartDisposition.SCHEDULED)
