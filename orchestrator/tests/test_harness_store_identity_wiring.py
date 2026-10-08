"""The orchestrator harness tells its escalation server which store it serves.

Task 3165 (γ1 of ``plans/escalation-store-ambiguity-prd.md``):
``Harness._start_escalation_server`` passes a ``kind='project'``
``escalation.store_identity::StoreIdentity`` into ``create_server``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.store_identity import StoreIdentity

from orchestrator.harness import Harness


def _build_harness(mock_orch_config) -> Harness:
    mock_orch_config.max_concurrent_tasks = 2

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


async def _start_escalation_server_capturing_create_server(h: Harness) -> MagicMock:
    """Drive ``_start_escalation_server`` without ever serving.

    The serve coroutine handed to ``asyncio.create_task`` is captured and
    closed rather than scheduled: an un-awaited coroutine GC'd later raises an
    unraisable exception that xdist pins on an unrelated test (the idiom of
    ``test_harness_escalation_filing_identity.py::TestEscalationServerWiring``).
    """
    serve_task = MagicMock()
    serve_task.done.return_value = False
    created_coros: list = []

    def _capture_task(coro, **kwargs):
        created_coros.append(coro)
        return serve_task

    with patch('orchestrator.harness.create_server') as mock_create, \
         patch('asyncio.create_task', side_effect=_capture_task), \
         patch('asyncio.sleep', new=AsyncMock()):
        await h._start_escalation_server()

    for coro in created_coros:
        coro.close()

    assert mock_create.called, '_start_escalation_server did not reach create_server'
    return mock_create


@pytest.mark.asyncio
async def test_create_server_receives_a_project_store_identity(
    mock_orch_config, tmp_path: Path,
) -> None:
    mock_orch_config.fused_memory.project_id = 'test-project'
    mock_orch_config.escalation.queue_dir = str(tmp_path / 'esc')
    mock_orch_config.escalation.host = '127.0.0.1'
    mock_orch_config.escalation.port = 0
    h = _build_harness(mock_orch_config)
    h.review_checkpoint = None

    mock_create = await _start_escalation_server_capturing_create_server(h)

    kwargs = mock_create.call_args.kwargs
    assert 'store_identity' in kwargs, (
        f'store_identity not passed to create_server; got {sorted(kwargs)}'
    )
    identity = kwargs['store_identity']
    assert isinstance(identity, StoreIdentity)
    assert identity.kind == 'project'
    assert identity.project_id == 'test-project'
    assert identity.project_root == mock_orch_config.project_root.resolve()
    assert h._escalation_queue is not None
    assert identity.queue_dir == h._escalation_queue.queue_dir.resolve()
