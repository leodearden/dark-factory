"""Test doubles shared by the orchestrator.proc_supervision test files.

Imported by bare module name (``from _proc_supervision_doubles import ...``),
like ``_orch_helpers`` -- ``orchestrator/tests/`` has no ``__init__.py``.

One copy of the subprocess runner double and of the canonical detached RP-4
restart plan, so every proc_supervision test file pins the same plan shape.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from orchestrator.proc_supervision import EscalationSpec, RestartPlan


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


def detached_rp4_plan(queue_dir: Path, *, with_spec: bool) -> RestartPlan:
    """The canonical detached (RP-4) self-restart plan.

    *with_spec* attaches the on-failure escalation filed into *queue_dir* as
    ``task-99``; without it the plan has no submit child at all.
    """
    spec = EscalationSpec(
        queue_dir=str(queue_dir),
        task_id='task-99',
        summary='Self-restart fire-time failure',
    )
    return RestartPlan(
        script=Path('/proj/scripts/restart-orchestrator.sh'),
        args=['--foo'],
        cwd=Path('/proj'),
        target_unit='orch.service',
        own_unit='orch.service',
        on_failure_escalation=spec if with_spec else None,
        verify=None,
        transient_unit='orch-redeploy-restart-99.service',
        on_active_secs=10,
    )
