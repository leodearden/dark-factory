"""Shared fixtures for driving the ``get_pending_escalations`` MCP tool.

Lives here rather than in either suite because
``test_pins_recovery_annotation.py`` (the computed ``pins_recovery``
annotation, task 3543) and ``test_pending_scan_off_loop.py`` (the queue scan's
``asyncio.to_thread`` hop, task 4391) drive the SAME tool through the same
four-piece harness — filing a pending record, a scheduler status stub, a
duck-typed harness, and the ``await tool.fn(...)`` invocation.

Not a ``conftest.py`` fixture: ``_file`` and ``_harness`` are parameterised
factories rather than per-test setup, so a plain importable module is the
honest shape.  The flat import resolves in every collection configuration
because ``escalation/tests/conftest.py`` explicitly inserts this directory at
the front of ``sys.path`` (enforced by
``tests/scripts/test_pytest_workspace_collection.py``) — prepend import mode
alone does NOT put it there: conftest records the measurement that in a
repo-root multi-package run another subproject's tests dir wins that slot.

``_Scheduler`` in particular must not be copied: it encodes the real
``(statuses, error)`` TUPLE shape, which a drifted second copy would get wrong
silently.
"""

from __future__ import annotations

import types
from typing import Any

from escalation.models import Escalation
from escalation.queue import EscalationQueue

IN_PROGRESS = 'in-progress'


def _file(queue: EscalationQueue, task_id: str, **kw: Any) -> Escalation:
    """Submit one pending escalation and return it."""
    esc = Escalation(
        id=queue.make_id(task_id),
        task_id=task_id,
        agent_role=kw.pop('agent_role', 'implementer'),
        severity=kw.pop('severity', 'blocking'),
        category=kw.pop('category', 'scope_violation'),
        summary=kw.pop('summary', 'pins-recovery fixture'),
        level=kw.pop('level', 1),
        **kw,
    )
    queue.submit(esc)
    return esc


class _Scheduler:
    """Minimal stand-in for orchestrator.scheduler's status accessors.

    ``get_statuses`` returns a ``(statuses, error)`` TUPLE — the real shape at
    orchestrator/src/orchestrator/scheduler.py:2523, and the reason a caller
    that assumes a bare dict silently treats an error as "no tasks".
    """

    def __init__(self, statuses: dict[str, str], error: Exception | None = None):
        self._statuses = statuses
        self._error = error
        self.calls: list[Any] = []

    async def get_statuses(self, ids: list[str] | None = None):
        self.calls.append(ids)
        if self._error is not None:
            return {}, self._error
        if ids is None:
            return dict(self._statuses), None
        return {i: self._statuses[i] for i in ids if i in self._statuses}, None


def _harness(statuses: dict[str, str], *, live: set[str] | None = None, **kw: Any):
    live_ids = live or set()
    scheduler = _Scheduler(statuses, error=kw.pop('error', None))
    return types.SimpleNamespace(
        scheduler=scheduler,
        is_workflow_active=lambda tid: tid in live_ids,
        **kw,
    )


async def _get_pending(server, **kwargs: Any) -> list[dict[str, Any]]:
    """get_pending_escalations is an ASYNC def as of task 3543 — it awaits a
    batched scheduler status read to compute pins_recovery."""
    tool = await server.get_tool('get_pending_escalations')
    return await tool.fn(**kwargs)
