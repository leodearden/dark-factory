"""The escalation and agent-result builders, and the ``invocation_end`` reader, the steward test files share.

A module rather than ``conftest.py`` for the reason ``_briefing_helpers.py``
gives, and the names keep their leading underscore so the importing files read
unchanged at every call site.
"""

from __future__ import annotations

from typing import Any

from escalation.models import Escalation
from shared.cli_invoke import AgentResult

from orchestrator.event_store import EventType


def _make_result(
    cost=1.0, turns=5, session_id='sess-abc', success=True,
    duration_ms=5000, stderr='', timed_out=False, subtype='',
    ended_awaiting_background=False,
) -> AgentResult:
    return AgentResult(
        success=success,
        output='done',
        stderr=stderr,
        cost_usd=cost,
        duration_ms=duration_ms,
        turns=turns,
        session_id=session_id,
        timed_out=timed_out,
        subtype=subtype,
        ended_awaiting_background=ended_awaiting_background,
    )


def _make_escalation(**overrides: Any) -> Escalation:
    defaults: dict[str, Any] = dict(
        id='esc-42-1',
        task_id='42',
        agent_role='orchestrator',
        severity='blocking',
        category='limit_exhausted',
        summary='execute limit exhausted',
    )
    defaults.update(overrides)
    return Escalation(**defaults)


def _sole_invocation_end_data(emit_mock) -> dict:
    """Return the ``data`` payload of the single ``invocation_end`` emit.

    task η: ``_invoke_with_session`` now emits a ``routing_decision`` event
    before the ``invocation_end`` event, so a bare ``emit.assert_called_once()``
    no longer holds. Filter to the ``invocation_end`` call (asserting it fired
    exactly once) rather than relaxing to the last-call heuristic.
    """
    calls = [
        c for c in emit_mock.call_args_list
        if c.args and c.args[0] == EventType.invocation_end
    ]
    assert len(calls) == 1, (
        f'expected exactly one invocation_end emit; got {emit_mock.call_args_list!r}'
    )
    return calls[0].kwargs['data']
