"""Tests for orchestrator.proc_supervision — RestartPlan + execute() (task 2237, W10-gamma).

proc_supervision.py is the single owned restart-seam (M1): RestartPlan,
EscalationSpec, FreshPidVerify, RestartOutcome/RestartDisposition, and one
async execute() honoring 5 invariants (RP-1..5, PRD Sec 5.1).

Prerequisite-1: shared test scaffolding (no behaviour assertions here) —
FakeRunner (records every (argv, kwargs) call, returns a configurable-
returncode fake proc), fake_inspector (configurable inspect-result async
callable), tmp_queue_dir fixture, and read_escalations() helper. Mirrors the
fake-runner pattern in test_service_restart.py:1042+ (patch of
asyncio.create_subprocess_exec / injected `runner=`) and the injected-
inspector pattern in test_deterministic_runner.py (AsyncMock unit_inspector).
"""

from __future__ import annotations

import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue


class FakeRunner:
    """Async callable matching ``asyncio.create_subprocess_exec``'s signature.

    Records every ``(argv, kwargs)`` call in ``self.calls`` so tests can
    assert on exact positional argv and keyword args (cwd=, stdout=, ...).
    Returns a ``MagicMock`` proc whose ``communicate()`` is an
    ``AsyncMock(return_value=(stdout, None))`` and whose ``returncode`` is
    configurable (default 0) — mirroring the ``fake_proc`` idiom already used
    throughout test_service_restart.py and test_deterministic_runner.py.
    """

    def __init__(self, returncode: int = 0, stdout: bytes = b'') -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.calls: list[tuple[tuple, dict]] = []
        # Every fake proc this runner has handed back, in call order — lets a
        # test assert on a spawned proc's post-return state (e.g. that
        # .communicate() was never awaited on a fire-and-forget leaf spawn)
        # without execute() needing to hand the proc back to the caller.
        self.procs: list[MagicMock] = []

    async def __call__(self, *args: object, **kwargs: object):
        self.calls.append((args, kwargs))
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(self.stdout, None))
        proc.returncode = self.returncode
        self.procs.append(proc)
        return proc


def make_fake_inspector(state: dict):
    """Build an async inspector callable: ``(unit, *, timeout_secs, reap_grace_secs=5.0) -> dict``.

    Returns a shallow copy of *state* on every call regardless of args, and
    records each call's kwargs on the returned callable's ``.calls`` list
    (each entry: ``{'unit', 'timeout_secs', 'reap_grace_secs'}``) so tests can
    assert the exact unit/timeout the RP-5 verify leg was invoked with.
    """
    calls: list[dict] = []

    async def _inspector(unit: str, *, timeout_secs: float, reap_grace_secs: float = 5.0) -> dict:
        calls.append({
            'unit': unit,
            'timeout_secs': timeout_secs,
            'reap_grace_secs': reap_grace_secs,
        })
        return dict(state)

    _inspector.calls = calls  # type: ignore[attr-defined]
    return _inspector


@pytest.fixture
def tmp_queue_dir(tmp_path: Path) -> Path:
    """A ``tmp_path`` subdirectory usable as an ``EscalationQueue`` queue_dir.

    Not pre-created — ``EscalationQueue.__init__`` mkdir(parents=True,
    exist_ok=True)'s it, matching production (the queue dir may not yet exist
    when the first escalation is filed).
    """
    return tmp_path / 'escalations'


def read_escalations(
    queue_dir: str | Path,
    task_id: str,
    *,
    status: str | None = None,
    level: int | None = None,
    agent_role: str | None = None,
) -> list:
    """Construct ``EscalationQueue(queue_dir)`` and return ``get_by_task(task_id, ...)``.

    Product read path (PRD Sec 7): tests assert filed escalations through
    this helper rather than inspecting on-disk JSON directly.
    """
    return EscalationQueue(Path(queue_dir)).get_by_task(
        task_id, status=status, level=level, agent_role=agent_role,
    )


# ---------------------------------------------------------------------------
# step-1: RED — types + construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """All five contract types construct; RestartPlan.__post_init__ validates cwd
    and absolutizes a relative script (the 2105 "no implicit cwd" fix)."""

    def test_escalation_spec_constructs_with_defaults(self) -> None:
        from orchestrator.proc_supervision import EscalationSpec

        spec = EscalationSpec(queue_dir='/tmp/q', task_id='t1', summary='boom')

        assert spec.queue_dir == '/tmp/q'
        assert spec.task_id == 't1'
        assert spec.summary == 'boom'
        assert spec.detail == ''
        assert spec.severity == 'critical'
        assert spec.category == 'infra_issue'
        assert spec.agent_role == 'orchestrator-deterministic'

    def test_fresh_pid_verify_constructs(self) -> None:
        from orchestrator.proc_supervision import FreshPidVerify

        verify = FreshPidVerify(
            baseline_active_enter_monotonic=1000,
            baseline_main_pid=42,
            inspect_timeout_secs=10.0,
        )

        assert verify.baseline_active_enter_monotonic == 1000
        assert verify.baseline_main_pid == 42
        assert verify.inspect_timeout_secs == 10.0

    def test_restart_disposition_has_six_members(self) -> None:
        from orchestrator.proc_supervision import RestartDisposition

        names = {member.name for member in RestartDisposition}

        assert names == {
            'REFUSED',
            'SCHEDULED',
            'DEPLOYED_AND_VERIFIED',
            'VERIFY_FAILED',
            'REGISTRATION_FAILED',
            'RESTART_FAILED',
        }

    def test_restart_outcome_defaults_escalated_false(self) -> None:
        from orchestrator.proc_supervision import RestartDisposition, RestartOutcome

        outcome = RestartOutcome(disposition=RestartDisposition.SCHEDULED)

        assert outcome.escalated is False
        assert outcome.detail == ''

    def test_restart_plan_absolutizes_relative_script_against_cwd(self) -> None:
        from orchestrator.proc_supervision import RestartPlan

        plan = RestartPlan(
            script=Path('scripts/x.sh'),
            args=[],
            cwd=Path('/proj'),
            target_unit='unit.service',
            own_unit=None,
            on_failure_escalation=None,
            verify=None,
        )

        assert plan.script == Path('/proj/scripts/x.sh')

    def test_restart_plan_leaves_absolute_script_unchanged(self) -> None:
        """An already-absolute script is left unchanged (no double-join)."""
        from orchestrator.proc_supervision import RestartPlan

        plan = RestartPlan(
            script=Path('/abs/scripts/x.sh'),
            args=[],
            cwd=Path('/proj'),
            target_unit='unit.service',
            own_unit=None,
            on_failure_escalation=None,
            verify=None,
        )

        assert plan.script == Path('/abs/scripts/x.sh')

    def test_restart_plan_rejects_non_absolute_cwd(self) -> None:
        """Structural "no implicit cwd" (the 2105 fix): a relative cwd raises ValueError."""
        from orchestrator.proc_supervision import RestartPlan

        with pytest.raises(ValueError):
            RestartPlan(
                script=Path('/abs/scripts/x.sh'),
                args=[],
                cwd=Path('relative/dir'),
                target_unit='unit.service',
                own_unit=None,
                on_failure_escalation=None,
                verify=None,
            )


# ---------------------------------------------------------------------------
# step-3: RED — R1 own-unit self-restart cell (RP-2/3/4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSelfRestartSystemdRunArgv:
    """own_unit == target_unit + transient_unit set -> DETACHED systemd-run
    with the RP-3 --working-directory flag and the RP-4 /bin/sh -c on-failure
    escalation wrapper. Never blocks (SCHEDULED), never escalates in-process
    on the success path."""

    async def test_self_restart_builds_systemd_run_argv_with_wrapper(
        self, tmp_queue_dir: Path,
    ) -> None:
        from orchestrator.proc_supervision import (
            EscalationSpec,
            RestartDisposition,
            RestartPlan,
        )

        runner = FakeRunner(returncode=0)
        spec = EscalationSpec(
            queue_dir=str(tmp_queue_dir),
            task_id='task-99',
            summary='Self-restart fire-time failure',
        )
        plan = RestartPlan(
            script=Path('/proj/scripts/restart-orchestrator.sh'),
            args=['--foo'],
            cwd=Path('/proj'),
            target_unit='orch.service',
            own_unit='orch.service',
            on_failure_escalation=spec,
            verify=None,
            transient_unit='orch-redeploy-restart-99.service',
            on_active_secs=10,
        )

        outcome = await plan.execute(runner=runner)

        assert len(runner.calls) == 1
        argv, _kwargs = runner.calls[0]
        assert argv[:7] == (
            'systemd-run', '--user',
            '--on-active=10',
            '--unit=orch-redeploy-restart-99.service',
            '--collect',
            '--working-directory=/proj',
            '/bin/sh',
        )
        assert argv[7] == '-c'
        assert len(argv) == 9
        wrapped = argv[8]

        expected_payload = ' '.join(
            shlex.quote(p) for p in ['/proj/scripts/restart-orchestrator.sh', '--foo']
        )
        assert wrapped.startswith(expected_payload)
        assert '__rc=$?;' in wrapped
        assert 'if [ "$__rc" -ne 0 ]; then' in wrapped
        assert wrapped.rstrip().endswith('fi; exit "$__rc"')
        # on-failure branch carries the escalation-submit argv (RP-4)
        assert '-m escalation submit' in wrapped
        assert '--task task-99' in wrapped

        assert outcome.disposition == RestartDisposition.SCHEDULED
        assert outcome.escalated is False

        # Success path: no in-process escalation filed (the on-failure branch
        # above only fires later, at systemd-run's deferred fire time).
        assert read_escalations(tmp_queue_dir, 'task-99') == []


# ---------------------------------------------------------------------------
# step-5: RED — R2 2105 cwd cell (RP-3, the exit-127 fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDetachedSystemdRunAlwaysHasWorkingDirectory:
    """A systemd --user transient unit's cwd defaults to $HOME, so a RELATIVE
    deploy script would 127 once the deferred unit fires (the 2105 incident).
    Every detached systemd-run argv must carry --working-directory=<cwd> AND
    an ABSOLUTE script token inside the /bin/sh -c payload — never the bare
    relative path."""

    async def test_relative_script_absolutized_and_working_directory_present(
        self, tmp_queue_dir: Path,
    ) -> None:
        from orchestrator.proc_supervision import RestartPlan

        runner = FakeRunner(returncode=0)
        plan = RestartPlan(
            script=Path('scripts/restart-orchestrator.sh'),  # RELATIVE
            args=[],
            cwd=Path('/proj'),
            target_unit='orch.service',
            own_unit='orch.service',
            on_failure_escalation=None,
            verify=None,
            transient_unit='orch-redeploy-restart-1.service',
            on_active_secs=10,
        )

        await plan.execute(runner=runner)

        argv, _kwargs = runner.calls[0]
        assert '--working-directory=/proj' in argv, (
            'a detached systemd-run argv must never omit --working-directory'
        )
        wrapped = argv[-1]
        assert '/proj/scripts/restart-orchestrator.sh' in wrapped, (
            'the script token inside the /bin/sh -c payload must be absolute'
        )
        assert wrapped.split()[0] == '/proj/scripts/restart-orchestrator.sh', (
            'the bare relative path must never appear as the payload script token'
        )


# ---------------------------------------------------------------------------
# step-7: RED — R3 2064 self-kill cell (RP-1 fail-closed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFailClosedSelfKillGuard:
    """RP-1: a BLOCKING restart (verify set) with an UNKNOWN own_unit refuses
    rather than risking a same-unit self-kill (the 2064 bug — today's
    ``self_target = bool(own_unit) and (target_unit == own_unit)`` fails OPEN
    when own_unit is falsy, routing an unprovable-self case onto the blocking
    cross-unit path). No blocking subprocess may ever be spawned in this cell
    — safety must not depend on the runner/inspector ever being called."""

    async def _assert_refuses_without_running_anything(
        self, tmp_queue_dir: Path, own_unit: str | None, task_id: str,
    ) -> None:
        from orchestrator.proc_supervision import (
            EscalationSpec,
            FreshPidVerify,
            RestartDisposition,
            RestartPlan,
        )

        runner = FakeRunner(returncode=0)
        inspector = make_fake_inspector(
            {'MainPID': 99, 'ActiveState': 'active', 'ActiveEnterTimestampMonotonic': 2000}
        )
        plan = RestartPlan(
            script=Path('/proj/scripts/deploy.sh'),
            args=[],
            cwd=Path('/proj'),
            target_unit='some-target.service',
            own_unit=own_unit,
            on_failure_escalation=EscalationSpec(
                queue_dir=str(tmp_queue_dir),
                task_id=task_id,
                summary='Cannot verify own unit before blocking restart',
            ),
            verify=FreshPidVerify(
                baseline_active_enter_monotonic=1000,
                baseline_main_pid=42,
                inspect_timeout_secs=10.0,
            ),
        )

        outcome = await plan.execute(runner=runner, inspector=inspector)

        assert outcome.disposition == RestartDisposition.REFUSED
        assert outcome.escalated is True
        assert runner.calls == [], (
            'no blocking self-restart subprocess may ever be spawned when '
            'own_unit is unknown — this is the 2064 self-kill guard'
        )
        assert inspector.calls == [], 'no inspect call before the refusal'

        escalations = read_escalations(
            tmp_queue_dir, task_id, agent_role='orchestrator-deterministic',
        )
        assert len(escalations) == 1
        esc = escalations[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.category == 'infra_issue'

    async def test_refuses_and_escalates_when_own_unit_empty_string(
        self, tmp_queue_dir: Path,
    ) -> None:
        await self._assert_refuses_without_running_anything(tmp_queue_dir, '', 'task-77')

    async def test_refuses_and_escalates_when_own_unit_none(
        self, tmp_queue_dir: Path,
    ) -> None:
        await self._assert_refuses_without_running_anything(tmp_queue_dir, None, 'task-78')


# ---------------------------------------------------------------------------
# amend (task 2237): self_target + wants_blocking degrades observably
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSameUnitBlockingVerifyFallsThroughObservably:
    """wants_blocking (verify set) with a KNOWN own_unit provably EQUAL to
    target_unit (self_target=True) AND a transient_unit set is the one plan
    shape neither RP-1's refuse branch (only fires when own_unit is unknown)
    nor RP-2's cross-unit-blocking branch (only fires when self_target is
    False) claims. It falls through to the DEFERRED systemd-run path instead
    — never the blocking one, since blocking a same-unit restart is exactly
    the 2064 self-kill risk RP-1 exists to prevent — with ``verify``
    intentionally dropped (a fresh-PID check would be meaningless for a
    fire-and-forget detached restart that hasn't happened yet). This pins the
    fallthrough behaviourally: the inspector/blocking runner must never be
    touched, and the detached shape (systemd-run, here) must still be taken.

    This fallthrough is conditioned on transient_unit being set — see
    ``TestSameUnitBlockingVerifyRefusesWithoutTransientUnit`` below for the
    sibling cell where no transient_unit means REFUSE instead, because the
    only remaining path would otherwise be an IMMEDIATE (not deferred) leaf
    spawn, which reintroduces the same self-kill hazard."""

    async def test_self_target_with_verify_falls_through_to_systemd_run_scheduled(
        self, tmp_queue_dir: Path,
    ) -> None:
        from orchestrator.proc_supervision import (
            FreshPidVerify,
            RestartDisposition,
            RestartPlan,
        )

        runner = FakeRunner(returncode=0)
        inspector = make_fake_inspector(
            {'MainPID': 99, 'ActiveState': 'active', 'ActiveEnterTimestampMonotonic': 2000}
        )
        plan = RestartPlan(
            script=Path('/proj/scripts/restart-orchestrator.sh'),
            args=[],
            cwd=Path('/proj'),
            target_unit='orch.service',
            own_unit='orch.service',
            on_failure_escalation=None,
            verify=FreshPidVerify(
                baseline_active_enter_monotonic=1000,
                baseline_main_pid=42,
                inspect_timeout_secs=10.0,
            ),
            transient_unit='orch-redeploy-restart-5.service',
            on_active_secs=10,
        )

        outcome = await plan.execute(runner=runner, inspector=inspector)

        # The blocking cross-unit path was never taken: no verify occurred.
        assert inspector.calls == []
        # The detached systemd-run path WAS taken (the fallthrough), not a
        # direct blocking run of the script.
        assert len(runner.calls) == 1
        argv, _kwargs = runner.calls[0]
        assert argv[0] == 'systemd-run'

        assert outcome.disposition == RestartDisposition.SCHEDULED
        assert outcome.escalated is False


@pytest.mark.asyncio
class TestSameUnitBlockingVerifyRefusesWithoutTransientUnit:
    """wants_blocking + self_target (a KNOWN own_unit provably equal to
    target_unit) with NO transient_unit must REFUSE rather than fall through
    to the leaf plain-spawn path (``_execute_detached_leaf_plain_spawn``):
    that path is an IMMEDIATE, synchronous ``create_subprocess_exec`` — not
    deferred — so for a same-unit restart script it is exactly the
    2064-class self-kill hazard RP-1 exists to prevent. Being "detached" in
    the fire-and-forget sense does not make an immediate same-unit restart
    safe. Mirrors RP-1's fail-closed refuse shape exactly: no runner/
    inspector call, escalation only when configured."""

    async def test_refuses_and_escalates_without_running_anything(
        self, tmp_queue_dir: Path,
    ) -> None:
        from orchestrator.proc_supervision import (
            EscalationSpec,
            FreshPidVerify,
            RestartDisposition,
            RestartPlan,
        )

        runner = FakeRunner(returncode=0)
        inspector = make_fake_inspector(
            {'MainPID': 99, 'ActiveState': 'active', 'ActiveEnterTimestampMonotonic': 2000}
        )
        plan = RestartPlan(
            script=Path('/proj/scripts/restart-orchestrator.sh'),
            args=[],
            cwd=Path('/proj'),
            target_unit='orch.service',
            own_unit='orch.service',
            on_failure_escalation=EscalationSpec(
                queue_dir=str(tmp_queue_dir),
                task_id='task-210',
                summary='Same-unit blocking restart refused (no transient_unit)',
            ),
            verify=FreshPidVerify(
                baseline_active_enter_monotonic=1000,
                baseline_main_pid=42,
                inspect_timeout_secs=10.0,
            ),
            transient_unit=None,
        )

        outcome = await plan.execute(runner=runner, inspector=inspector)

        assert outcome.disposition == RestartDisposition.REFUSED
        assert outcome.escalated is True
        assert runner.calls == [], (
            'no subprocess may ever be spawned for a same-unit blocking plan '
            'with no transient_unit — the only other path is an IMMEDIATE '
            'leaf spawn, the same self-kill hazard RP-1 exists to prevent'
        )
        assert inspector.calls == [], 'no inspect call before the refusal'

        escalations = read_escalations(
            tmp_queue_dir, 'task-210', agent_role='orchestrator-deterministic',
        )
        assert len(escalations) == 1
        esc = escalations[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.category == 'infra_issue'

    async def test_refuses_without_escalation_when_none_configured(
        self, tmp_queue_dir: Path,
    ) -> None:
        from orchestrator.proc_supervision import (
            FreshPidVerify,
            RestartDisposition,
            RestartPlan,
        )

        runner = FakeRunner(returncode=0)
        plan = RestartPlan(
            script=Path('/proj/scripts/restart-orchestrator.sh'),
            args=[],
            cwd=Path('/proj'),
            target_unit='orch.service',
            own_unit='orch.service',
            on_failure_escalation=None,
            verify=FreshPidVerify(
                baseline_active_enter_monotonic=1000,
                baseline_main_pid=42,
                inspect_timeout_secs=10.0,
            ),
            transient_unit=None,
        )

        outcome = await plan.execute(runner=runner, inspector=make_fake_inspector({}))

        assert outcome.disposition == RestartDisposition.REFUSED
        assert outcome.escalated is False
        assert runner.calls == []


# ---------------------------------------------------------------------------
# step-9: RED — R4 cross-unit blocking verify-pass cell (RP-2/5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCrossUnitBlockingVerifyPass:
    """own_unit != target_unit (both truthy, provably different) + verify set
    -> BLOCKING restart: run the script directly to completion (no
    systemd-run), then re-inspect the target unit for a fresh MainPID and a
    strictly-later monotonic timestamp than the persisted baseline (RP-5). A
    provably different target can never self-kill the caller, so this path
    is allowed to block the event loop."""

    async def test_cross_unit_blocking_restart_runs_script_and_verifies_fresh_pid(
        self, tmp_queue_dir: Path,
    ) -> None:
        from orchestrator.proc_supervision import (
            EscalationSpec,
            FreshPidVerify,
            RestartDisposition,
            RestartPlan,
        )

        runner = FakeRunner(returncode=0)
        inspector = make_fake_inspector({
            'MainPID': 99,
            'ActiveState': 'active',
            'ActiveEnterTimestampMonotonic': 2000,
        })
        spec = EscalationSpec(
            queue_dir=str(tmp_queue_dir),
            task_id='task-55',
            summary='Cross-unit deploy verify failed',
        )
        plan = RestartPlan(
            script=Path('/proj/scripts/deploy.sh'),
            args=['--flag'],
            cwd=Path('/proj'),
            target_unit='fused-memory.service',
            own_unit='orch.service',
            on_failure_escalation=spec,
            verify=FreshPidVerify(
                baseline_active_enter_monotonic=1000,
                baseline_main_pid=42,
                inspect_timeout_secs=10.0,
            ),
        )

        outcome = await plan.execute(runner=runner, inspector=inspector)

        assert len(runner.calls) == 1
        args, kwargs = runner.calls[0]
        assert args == ('/proj/scripts/deploy.sh', '--flag'), (
            'the blocking path runs the script directly — no systemd-run wrapping'
        )
        assert 'systemd-run' not in args
        assert kwargs.get('cwd') == '/proj'

        assert inspector.calls == [
            {'unit': 'fused-memory.service', 'timeout_secs': 10.0, 'reap_grace_secs': 5.0},
        ]

        assert outcome.disposition == RestartDisposition.DEPLOYED_AND_VERIFIED
        assert outcome.escalated is False

        # Success path: no in-process escalation filed even though one was
        # configured via on_failure_escalation.
        assert read_escalations(tmp_queue_dir, 'task-55') == []


# ---------------------------------------------------------------------------
# step-11: RED — R5 verify-fail-escalate cell (RP-5), 3 sub-cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCrossUnitBlockingVerifyFail:
    """RP-5: a cross-unit blocking restart that does NOT produce a provably
    fresh target process must never report DEPLOYED_AND_VERIFIED — it
    escalates instead. Four sub-cases: a stale-sentinel MainPID, a monotonic
    timestamp that is not strictly later than the persisted baseline, a
    restart script that itself exits non-zero (in which case the inspector is
    never even consulted — no point verifying freshness after a failed
    restart), and a re-inspected MainPID identical to the persisted
    ``baseline_main_pid`` despite a strictly-later monotonic timestamp (the
    process identity never actually changed, so a ticked-forward activation
    clock alone must not count as fresh)."""

    async def test_stale_pid_sentinel_is_verify_failed(self, tmp_queue_dir: Path) -> None:
        from orchestrator.proc_supervision import RestartDisposition

        await self._assert_not_deployed_and_escalates(
            tmp_queue_dir,
            task_id='task-201',
            runner=FakeRunner(returncode=0),
            inspector=make_fake_inspector({
                'MainPID': 0,  # stale sentinel — never a real fresh PID
                'ActiveState': 'active',
                'ActiveEnterTimestampMonotonic': 2000,
            }),
            inspector_called=True,
            expected_disposition=RestartDisposition.VERIFY_FAILED,
        )

    async def test_stale_monotonic_timestamp_is_verify_failed(self, tmp_queue_dir: Path) -> None:
        from orchestrator.proc_supervision import RestartDisposition

        await self._assert_not_deployed_and_escalates(
            tmp_queue_dir,
            task_id='task-202',
            runner=FakeRunner(returncode=0),
            inspector=make_fake_inspector({
                'MainPID': 99,
                'ActiveState': 'active',
                'ActiveEnterTimestampMonotonic': 500,  # <= baseline 1000: not fresh
            }),
            inspector_called=True,
            expected_disposition=RestartDisposition.VERIFY_FAILED,
        )

    async def test_restart_script_failure_is_restart_failed_and_skips_inspect(
        self, tmp_queue_dir: Path,
    ) -> None:
        from orchestrator.proc_supervision import RestartDisposition

        await self._assert_not_deployed_and_escalates(
            tmp_queue_dir,
            task_id='task-203',
            runner=FakeRunner(returncode=1),
            inspector=make_fake_inspector({
                'MainPID': 99,
                'ActiveState': 'active',
                'ActiveEnterTimestampMonotonic': 2000,
            }),
            inspector_called=False,
            expected_disposition=RestartDisposition.RESTART_FAILED,
        )

    async def test_same_pid_as_baseline_is_verify_failed_despite_fresh_monotonic(
        self, tmp_queue_dir: Path,
    ) -> None:
        """``baseline_main_pid`` is a real signal in the freshness check, not
        a dead field: a re-inspected MainPID identical to the persisted
        baseline (42, per ``_assert_not_deployed_and_escalates`` below) means
        the process identity never actually changed, even though the
        monotonic timestamp ticked forward past the baseline — this must not
        be reported fresh."""
        from orchestrator.proc_supervision import RestartDisposition

        await self._assert_not_deployed_and_escalates(
            tmp_queue_dir,
            task_id='task-204',
            runner=FakeRunner(returncode=0),
            inspector=make_fake_inspector({
                'MainPID': 42,  # == baseline_main_pid: no identity change
                'ActiveState': 'active',
                'ActiveEnterTimestampMonotonic': 2000,  # > baseline 1000
            }),
            inspector_called=True,
            expected_disposition=RestartDisposition.VERIFY_FAILED,
        )

    async def _assert_not_deployed_and_escalates(
        self,
        tmp_queue_dir: Path,
        *,
        task_id: str,
        runner,
        inspector,
        inspector_called: bool,
        expected_disposition,
    ) -> None:
        from orchestrator.proc_supervision import (
            EscalationSpec,
            FreshPidVerify,
            RestartDisposition,
            RestartPlan,
        )

        spec = EscalationSpec(
            queue_dir=str(tmp_queue_dir),
            task_id=task_id,
            summary='Cross-unit deploy verify failed',
        )
        plan = RestartPlan(
            script=Path('/proj/scripts/deploy.sh'),
            args=[],
            cwd=Path('/proj'),
            target_unit='fused-memory.service',
            own_unit='orch.service',
            on_failure_escalation=spec,
            verify=FreshPidVerify(
                baseline_active_enter_monotonic=1000,
                baseline_main_pid=42,
                inspect_timeout_secs=10.0,
            ),
        )

        outcome = await plan.execute(runner=runner, inspector=inspector)

        assert outcome.disposition == expected_disposition
        assert outcome.disposition != RestartDisposition.DEPLOYED_AND_VERIFIED, (
            'a non-fresh or failed cross-unit restart must never be reported done'
        )
        assert outcome.escalated is True
        assert bool(inspector.calls) == inspector_called

        escalations = read_escalations(
            tmp_queue_dir, task_id, agent_role='orchestrator-deterministic',
        )
        assert len(escalations) == 1
        esc = escalations[0]
        assert esc.level == 2
        assert esc.severity == 'critical'
        assert esc.category == 'infra_issue'


# ---------------------------------------------------------------------------
# amend (task 2237): escalated reflects actual filing, not dedup-existence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEscalatedReflectsActualFiling:
    """``RestartOutcome.escalated`` means "this call filed a NEW L2
    escalation" — not "an escalation exists for this task". When
    ``on_failure_escalation`` is configured but a pending escalation for the
    same task_id+agent_role already exists (e.g. filed by an earlier,
    crash-recovered call), ``_file_inprocess_escalation``'s dedup guard skips
    the re-file — and ``escalated`` must be ``False`` in that case, even
    though an escalation record for this task does exist and
    ``on_failure_escalation`` was configured. Exercised through both call
    sites that thread a bool into ``RestartOutcome.escalated``: RP-1's refuse
    branch and the shared ``_file_verify_stage_escalation`` helper (used by
    both RESTART_FAILED and VERIFY_FAILED)."""

    async def test_rp1_refuse_reports_escalated_false_when_deduped(
        self, tmp_queue_dir: Path,
    ) -> None:
        from orchestrator.proc_supervision import (
            EscalationSpec,
            FreshPidVerify,
            RestartDisposition,
            RestartPlan,
        )

        spec = EscalationSpec(
            queue_dir=str(tmp_queue_dir),
            task_id='task-dedup-1',
            summary='Cannot verify own unit before blocking restart',
        )
        # Pre-file a pending escalation for the same task_id+agent_role, as
        # if an earlier RestartPlan.execute() call already filed it.
        queue = EscalationQueue(tmp_queue_dir)
        queue.submit(spec.to_escalation(queue))

        plan = RestartPlan(
            script=Path('/proj/scripts/deploy.sh'),
            args=[],
            cwd=Path('/proj'),
            target_unit='some-target.service',
            own_unit=None,
            on_failure_escalation=spec,
            verify=FreshPidVerify(
                baseline_active_enter_monotonic=1000,
                baseline_main_pid=42,
                inspect_timeout_secs=10.0,
            ),
        )

        outcome = await plan.execute(
            runner=FakeRunner(), inspector=make_fake_inspector({}),
        )

        assert outcome.disposition == RestartDisposition.REFUSED
        assert outcome.escalated is False, (
            'the dedup guard skipped filing a duplicate — this call filed no '
            'NEW escalation, so escalated must be False even though '
            'on_failure_escalation was configured and an escalation record '
            'for this task does exist'
        )
        escalations = read_escalations(
            tmp_queue_dir, 'task-dedup-1', agent_role='orchestrator-deterministic',
        )
        assert len(escalations) == 1, 'still exactly one — the dedup guard prevented a duplicate'

    async def test_verify_stage_reports_escalated_false_when_deduped(
        self, tmp_queue_dir: Path,
    ) -> None:
        from orchestrator.proc_supervision import (
            EscalationSpec,
            FreshPidVerify,
            RestartDisposition,
            RestartPlan,
        )

        spec = EscalationSpec(
            queue_dir=str(tmp_queue_dir),
            task_id='task-dedup-2',
            summary='Cross-unit deploy verify failed',
        )
        queue = EscalationQueue(tmp_queue_dir)
        queue.submit(spec.to_escalation(queue))

        plan = RestartPlan(
            script=Path('/proj/scripts/deploy.sh'),
            args=[],
            cwd=Path('/proj'),
            target_unit='fused-memory.service',
            own_unit='orch.service',
            on_failure_escalation=spec,
            verify=FreshPidVerify(
                baseline_active_enter_monotonic=1000,
                baseline_main_pid=42,
                inspect_timeout_secs=10.0,
            ),
        )

        # rc != 0 -> RESTART_FAILED, routed through the shared
        # _file_verify_stage_escalation helper (also used by VERIFY_FAILED).
        outcome = await plan.execute(
            runner=FakeRunner(returncode=1), inspector=make_fake_inspector({}),
        )

        assert outcome.disposition == RestartDisposition.RESTART_FAILED
        assert outcome.escalated is False, (
            'deduped by the same guard — no NEW escalation was filed this call'
        )
        escalations = read_escalations(
            tmp_queue_dir, 'task-dedup-2', agent_role='orchestrator-deterministic',
        )
        assert len(escalations) == 1


# ---------------------------------------------------------------------------
# step-13: RED — RP-4 wrapper exactness + registration failure
# ---------------------------------------------------------------------------


class TestEscalationSpecSubmitArgvExact:
    """EscalationSpec.to_submit_argv builds the exact ``python -m escalation
    submit ...`` argv (RP-4 shell branch) — byte-for-byte, so the detached
    on-failure wrapper and any future direct caller produce an identical
    invocation."""

    def test_to_submit_argv_exact_shape(self) -> None:
        from orchestrator.proc_supervision import EscalationSpec

        spec = EscalationSpec(
            queue_dir='/tmp/q',
            task_id='task-1',
            summary='boom',
            detail='full context',
        )

        argv = spec.to_submit_argv('/usr/bin/python3')

        assert argv == [
            '/usr/bin/python3', '-m', 'escalation', 'submit',
            '--queue-dir', '/tmp/q',
            '--task', 'task-1',
            '--severity', 'critical',
            '--category', 'infra_issue',
            '--summary', 'boom',
            '--agent-role', 'orchestrator-deterministic',
            '--detail', 'full context',
        ]


@pytest.mark.asyncio
class TestDetachedWrapperExactnessAndRegistrationFailure:
    """(b) The ``/bin/sh -c`` payload for a detached self-restart is
    byte-for-byte
    ``f'{quoted_payload}; __rc=$?; if [ "$__rc" -ne 0 ]; then {quoted_on_failure}; fi; exit "$__rc"'``.
    (c) A non-zero systemd-run registration rc reports REGISTRATION_FAILED —
    no crash, no false SCHEDULED."""

    async def test_wrapper_payload_is_byte_for_byte(self, tmp_queue_dir: Path) -> None:
        import sys

        from orchestrator.proc_supervision import EscalationSpec, RestartPlan

        runner = FakeRunner(returncode=0)
        spec = EscalationSpec(
            queue_dir=str(tmp_queue_dir),
            task_id='task-99',
            summary='Self-restart fire-time failure',
        )
        plan = RestartPlan(
            script=Path('/proj/scripts/restart-orchestrator.sh'),
            args=['--foo'],
            cwd=Path('/proj'),
            target_unit='orch.service',
            own_unit='orch.service',
            on_failure_escalation=spec,
            verify=None,
            transient_unit='orch-redeploy-restart-99.service',
            on_active_secs=10,
        )

        await plan.execute(runner=runner)

        argv, _kwargs = runner.calls[0]
        wrapped = argv[-1]

        quoted_payload = ' '.join(
            shlex.quote(p) for p in ['/proj/scripts/restart-orchestrator.sh', '--foo']
        )
        quoted_on_failure = ' '.join(
            shlex.quote(p) for p in spec.to_submit_argv(sys.executable)
        )
        expected = (
            f'{quoted_payload}; __rc=$?; '
            f'if [ "$__rc" -ne 0 ]; then {quoted_on_failure}; fi; '
            f'exit "$__rc"'
        )
        assert wrapped == expected

    async def test_registration_failure_reports_registration_failed(
        self, tmp_queue_dir: Path,
    ) -> None:
        from orchestrator.proc_supervision import RestartDisposition, RestartPlan

        runner = FakeRunner(returncode=1, stdout=b'systemd-run: unit already exists\n')
        plan = RestartPlan(
            script=Path('/proj/scripts/restart-orchestrator.sh'),
            args=[],
            cwd=Path('/proj'),
            target_unit='orch.service',
            own_unit='orch.service',
            on_failure_escalation=None,
            verify=None,
            transient_unit='orch-redeploy-restart-1.service',
        )

        outcome = await plan.execute(runner=runner)

        assert outcome.disposition == RestartDisposition.REGISTRATION_FAILED
        assert outcome.escalated is False


# ---------------------------------------------------------------------------
# step-15: RED — detached LEAF plain-spawn path (fused-memory/dashboard parity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDetachedLeafPlainSpawn:
    """transient_unit=None + verify=None -> DETACHED leaf plain-spawn: a bare
    ``create_subprocess_exec(str(script), *args, cwd=str(cwd),
    start_new_session=True)`` with NO ``/bin/sh`` wrapper and NO systemd-run —
    fused-memory/dashboard parity (service_restart.py's
    ``_default_restart_executor``). Fire-and-forget: the process exit is
    never awaited. Spawn errors (FileNotFoundError/PermissionError) propagate
    uncaught — the coordinator's permanent-failure branch
    (``maybe_restart``) depends on seeing them."""

    async def test_plain_spawn_argv_and_kwargs_no_wrapper(self) -> None:
        from orchestrator.proc_supervision import RestartDisposition, RestartPlan

        runner = FakeRunner(returncode=0)
        plan = RestartPlan(
            script=Path('/proj/scripts/restart-fused-memory.sh'),
            args=['--drain'],
            cwd=Path('/proj'),
            target_unit='fused-memory.service',
            own_unit='',
            on_failure_escalation=None,
            verify=None,
            transient_unit=None,
        )

        outcome = await plan.execute(runner=runner)

        assert len(runner.calls) == 1
        args, kwargs = runner.calls[0]
        assert args == ('/proj/scripts/restart-fused-memory.sh', '--drain')
        assert 'systemd-run' not in args
        assert '/bin/sh' not in args
        assert kwargs.get('start_new_session') is True
        assert kwargs.get('cwd') == '/proj'

        assert runner.procs[0].communicate.called is False, (
            'the leaf plain-spawn path is fire-and-forget — communicate() '
            'must never be awaited'
        )

        assert outcome.disposition == RestartDisposition.SCHEDULED
        assert outcome.escalated is False

    async def test_spawn_error_propagates_uncaught(self) -> None:
        """FileNotFoundError from the runner is NOT swallowed — it propagates
        so the coordinator's permanent-failure branch (``maybe_restart``) can
        see it and clear pending instead of crash-looping on a permanently
        missing/non-executable script."""
        from orchestrator.proc_supervision import RestartPlan

        async def raising_runner(*args: object, **kwargs: object):
            raise FileNotFoundError('restart-fused-memory.sh not found')

        plan = RestartPlan(
            script=Path('/proj/scripts/restart-fused-memory.sh'),
            args=[],
            cwd=Path('/proj'),
            target_unit='fused-memory.service',
            own_unit='',
            on_failure_escalation=None,
            verify=None,
            transient_unit=None,
        )

        with pytest.raises(FileNotFoundError):
            await plan.execute(runner=raising_runner)
