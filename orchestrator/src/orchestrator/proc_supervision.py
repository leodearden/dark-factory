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
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import sys
from dataclasses import dataclass
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

        Not yet implemented — driven by step-8's RED test (R3 self-kill cell).
        """
        raise NotImplementedError

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


@dataclass(frozen=True)
class FreshPidVerify:
    """RP-5 fresh-PID verify parameters.

    ``baseline_active_enter_monotonic``/``baseline_main_pid`` are a
    CALLER-PERSISTED field, not a local re-inspect captured inside
    ``execute()`` (the 2074 caveat) — the caller must inspect the target
    unit and persist the baseline BEFORE invoking the restart.
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
    """Result of a ``RestartPlan.execute()`` call."""

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

        1. wants_blocking and not own -> RP-1 fail-closed REFUSE (step-7/8; not
           yet implemented here).
        2. wants_blocking and not self_target -> RP-2 cross-unit BLOCKING +
           RP-5 verify (step-9/10/11/12; not yet implemented here).
        3. transient_unit set -> DETACHED systemd-run (RP-3/RP-4; implemented
           below, step-4).
        4. else -> DETACHED leaf plain-spawn (step-15/16; not yet implemented
           here).
        """
        runner = runner or asyncio.create_subprocess_exec
        inspector = inspector or inspect_systemd_unit

        own = self.own_unit or ''
        wants_blocking = self.verify is not None
        self_target = bool(own) and self.target_unit == own

        if wants_blocking and not own:
            # RP-1 fail-closed refuse — implemented in step-8.
            raise NotImplementedError

        if wants_blocking and not self_target:
            # RP-2 cross-unit blocking + RP-5 verify — implemented in step-10/12.
            raise NotImplementedError

        if self.transient_unit:
            return await self._execute_detached_systemd_run(runner)

        # Leaf plain-spawn path — implemented in step-16.
        raise NotImplementedError

    async def _execute_detached_systemd_run(self, runner) -> RestartOutcome:
        """RP-3/RP-4: register a deferred ``systemd-run --user`` transient unit.

        Registers immediately and returns; the payload (script + args) fires
        LATER under the user systemd manager, so this never blocks or risks
        killing the caller — used both for a same-unit self-restart and for
        any other detached fire-and-forget-with-verification need.

        ``--working-directory=<cwd>`` (RP-3) guards against the ``systemd
        --user`` manager's $HOME-default cwd (the 2105 exit-127 bug) — the
        transient unit's payload always resolves relative to an explicit,
        absolute cwd instead. The ``/bin/sh -c`` wrapper's on-failure branch
        (RP-4) files a born-at-L2 escalation via
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
