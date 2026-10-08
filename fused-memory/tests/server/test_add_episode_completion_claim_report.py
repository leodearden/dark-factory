"""`add_episode` reports an unverified completion claim only once the service
has accepted the episode (task 4715).

The report is the operator-facing half of the completion-claim gate: a WARNING
per claim and one escalation per ref, both saying the write was INGESTED and
tagged. `MemoryService.add_episode` re-raises a failed enqueue, so a report
filed before the service call would describe an episode that never landed.
`test_completion_claim_gate_ingestion.py` pins the rest of the gate on this
tool; `test_add_memory_completion_claim_gate.py` pins the same ordering on
`add_memory`.

The emitter is NOT monkeypatched: the writer's project is registered at
`tmp_path`, so the real `emit_unverified_claim_escalation` files into
`tmp_path/data/escalations` and the assertions read that queue through its API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'
# `applied` is outside task_filter.PRESENT_TENSE_COMPLETION_RE, so only this
# gate can act on the claim.
_CLAIM_CONTENT = "task 5422's de-flake fix has been applied"


def _server(mock_service, root: Path):
    task_interceptor = MagicMock()
    task_interceptor.get_statuses = AsyncMock(return_value={'5422': 'in-progress'})
    task_interceptor.get_ticket_row = AsyncMock(return_value=None)
    return create_mcp_server(
        mock_service,
        task_interceptor=task_interceptor,
        known_projects={_PROJECT_ID: str(root)},
    )


async def _call(server) -> dict:
    return await server._tool_manager.call_tool(
        'add_episode',
        {
            'content': _CLAIM_CONTENT,
            'agent_id': 'claude-interactive',
            'project_id': _PROJECT_ID,
        },
    )


def _pending_escalations(root: Path) -> list:
    queue_dir = root / 'data' / 'escalations'
    if not queue_dir.exists():
        return []
    return EscalationQueue(queue_dir).get_pending()


def _gate_warnings(caplog) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.getMessage().startswith('completion_claim_gate.unverified')
    ]


class TestTheReportFollowsTheWrite:

    @pytest.mark.asyncio
    async def test_an_accepted_episode_is_reported(self, tmp_path, caplog):
        mock_service = AsyncMock()
        ep_result = MagicMock()
        ep_result.model_dump.return_value = {'id': 'ep'}
        mock_service.add_episode.return_value = ep_result

        with caplog.at_level(logging.DEBUG):
            result = await _call(_server(mock_service, tmp_path))

        pending = _pending_escalations(tmp_path)
        assert len(pending) == 1, f'{pending!r}'
        assert result['unverified_claim'].get('escalation_id') == pending[0].id
        assert len(_gate_warnings(caplog)) == 1, f'{_gate_warnings(caplog)!r}'

    @pytest.mark.asyncio
    async def test_a_failed_enqueue_reports_nothing(self, tmp_path, caplog):
        mock_service = AsyncMock()
        mock_service.add_episode.side_effect = RuntimeError('durable queue enqueue failed')

        with caplog.at_level(logging.DEBUG):
            result = await _call(_server(mock_service, tmp_path))

        assert result.get('error_type') == 'RuntimeError', f'{result!r}'
        assert mock_service.add_episode.call_args.kwargs.get('unverified_claim') is True
        assert _pending_escalations(tmp_path) == [], (
            'an escalation saying the episode was ingested was filed for an '
            'episode the service never accepted'
        )
        assert _gate_warnings(caplog) == [], f'{_gate_warnings(caplog)!r}'
