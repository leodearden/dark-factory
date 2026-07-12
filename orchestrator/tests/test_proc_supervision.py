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
