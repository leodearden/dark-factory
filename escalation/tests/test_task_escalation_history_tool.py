"""Tests for the get_task_escalation_history escalation MCP tool.

PRD ``plans/escalation-store-ambiguity-prd.md`` task β / boundary test B6.
``get_task_escalation_history`` is a thin, envelope-shaping wrapper over
task 3023's ``get_task_escalations`` (escalation/src/escalation/server.py:951)
— it does not re-scan the queue or re-serialize records itself; it calls the
sibling tool and reshapes the returned list into a self-describing envelope
(``{task_id, count, level_filter, escalations}``).

The archive-inclusive PRIMITIVE this tool relies on (the two-tier
queue-root + ``archive/<date>/`` scan) is already covered by
``test_server.py:426-571`` (``TestGetTaskEscalations``) and
``test_queue.py:366``. This file does not re-litigate that coverage — it
covers only what THIS tool adds on top of the sibling: the envelope shape,
the delegation, the ``level_filter`` echo, and the no-``status`` safety pin
(a caller must not be able to narrow this tool back to the pending-only
fast path — that is the exact false-absence trap this PRD exists to
remove).

Uses the sync-tool FastMCP unit-test pattern established by
``test_task_runtime_state_tool.py``: ``tool = await server.get_tool(name);
tool.fn(**kwargs)`` — no ``await`` on ``.fn()`` itself, since the tool body
is a sync ``def``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from escalation.models import Escalation
from escalation.queue import EscalationQueue
from escalation.server import create_server


async def _history(server: Any, **kwargs: Any) -> dict[str, Any]:
    """Invoke the get_task_escalation_history MCP tool directly (sync tool)."""
    tool = await server.get_tool('get_task_escalation_history')
    return tool.fn(**kwargs)


async def _get_pending(server: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Invoke get_pending_escalations directly (sync tool), for the B6 contrast."""
    tool = await server.get_tool('get_pending_escalations')
    return tool.fn(**kwargs)


@pytest.mark.asyncio
class TestGetTaskEscalationHistoryTool:
    """get_task_escalation_history(task_id, level=None) — envelope, archive-inclusive."""

    def _seed(
        self,
        queue: EscalationQueue,
        esc_id: str,
        *,
        task_id: str = '3164',
        level: int = 0,
        agent_role: str = 'implementer',
    ) -> Escalation:
        """Submit a pending escalation with an explicit id."""
        esc = Escalation(
            id=esc_id,
            task_id=task_id,
            agent_role=agent_role,
            severity='blocking',
            category='task_failure',
            summary=f'{esc_id} test escalation',
            level=level,
        )
        queue.submit(esc)
        return esc

    def _mixed_queue(self, tmp_path: Path) -> EscalationQueue:
        """One resolved+archived record and one still-pending record for task '3164'.

        ``esc-3164-1`` mirrors a human-resolved born-at-L2 deterministic
        gate: submitted at level 2 by the deterministic runner, then
        resolved (which moves the file out of the queue root into
        ``archive/<date>/``). ``esc-3164-2`` stays in the queue root.

        Reproduces (does not import — the original is a private method on a
        test class in another module) the recipe at ``test_server.py:459-471``
        (``TestGetTaskEscalations._mixed_queue``, task 3023), so this suite's
        pending/history contrast is proven on the same on-disk layout that
        ``test_queue.py:376``/``:386`` spec-lock.
        """
        queue = EscalationQueue(tmp_path / 'esc')
        self._seed(queue, 'esc-3164-1', level=2, agent_role='deterministic')
        queue.resolve('esc-3164-1', 'Human reviewed the gate')
        self._seed(queue, 'esc-3164-2', level=0, agent_role='implementer')
        return queue

    async def test_pending_probe_misses_archived_record_but_history_envelope_finds_it(
        self, tmp_path: Path,
    ):
        """The B6 contrast, on ONE server instance: get_pending_escalations
        misses the archived record; get_task_escalation_history's envelope
        contains it — and calling history does not perturb the pending-only
        path. Also pins the envelope shape itself (dict, not the sibling's
        bare list) and the task_id/count echo.
        """
        queue = self._mixed_queue(tmp_path)
        server = create_server(queue)

        pending = await _get_pending(server, task_id='3164')
        assert {e['id'] for e in pending} == {'esc-3164-2'}, (
            f'get_pending_escalations must stay root-only, got {pending}'
        )

        result = await _history(server, task_id='3164')

        assert isinstance(result, dict), (
            f'Expected a dict envelope (unlike the sibling get_task_escalations, '
            f'which returns a bare list), got {type(result)}'
        )
        assert {e['id'] for e in result['escalations']} == {'esc-3164-1', 'esc-3164-2'}, (
            f"Expected both the archived and the pending record, got {result['escalations']}"
        )
        archived = next(e for e in result['escalations'] if e['id'] == 'esc-3164-1')
        assert archived['status'] == 'resolved'
        assert result['task_id'] == '3164'
        assert result['count'] == len(result['escalations']) == 2

    async def test_envelope_has_no_store_identity_fields(self, tmp_path: Path):
        """γ1/γ3 own store identity (StoreIdentity, project_root/queue_dir/
        project_id) — delivered at the tool DESCRIPTION layer, not this
        envelope. β must not reach for it (PRD §5.1, design decision on
        identity)."""
        queue = self._mixed_queue(tmp_path)
        server = create_server(queue)

        result = await _history(server, task_id='3164')

        assert 'queue_dir' not in result
        assert 'project_id' not in result
