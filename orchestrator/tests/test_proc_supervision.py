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

    async def __call__(self, *args: object, **kwargs: object):
        self.calls.append((args, kwargs))
        proc = MagicMock()
        proc.communicate = AsyncMock(return_value=(self.stdout, None))
        proc.returncode = self.returncode
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


def read_escalations(queue_dir: str | Path, task_id: str, **kwargs: object) -> list:
    """Construct ``EscalationQueue(queue_dir)`` and return ``get_by_task(task_id, **kwargs)``.

    Product read path (PRD Sec 7): tests assert filed escalations through
    this helper rather than inspecting on-disk JSON directly.
    """
    return EscalationQueue(Path(queue_dir)).get_by_task(task_id, **kwargs)


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
    escalates instead. Three sub-cases: a stale-sentinel MainPID, a monotonic
    timestamp that is not strictly later than the persisted baseline, and a
    restart script that itself exits non-zero (in which case the inspector is
    never even consulted — no point verifying freshness after a failed
    restart)."""

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
