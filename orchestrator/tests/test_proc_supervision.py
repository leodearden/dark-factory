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
