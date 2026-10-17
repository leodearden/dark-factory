"""The orchestrator harness tells its escalation server which store it serves.

Task 3165 (γ1 of ``plans/escalation-store-ambiguity-prd.md``):
``Harness._start_escalation_server`` passes a ``kind='project'``
``escalation.store_identity::StoreIdentity`` into ``create_server``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from _escalation_server_capture import create_server_kwargs
from escalation.store_identity import StoreIdentity

from orchestrator.harness import Harness


def _build_harness(mock_orch_config) -> Harness:
    mock_orch_config.max_concurrent_tasks = 2

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


@pytest.mark.asyncio
async def test_create_server_receives_a_project_store_identity(
    mock_orch_config, tmp_path: Path,
) -> None:
    configured_queue_dir = tmp_path / 'esc'
    mock_orch_config.fused_memory.project_id = 'test-project'
    mock_orch_config.escalation.queue_dir = str(configured_queue_dir)
    mock_orch_config.escalation.host = '127.0.0.1'
    mock_orch_config.escalation.port = 0
    h = _build_harness(mock_orch_config)
    h.review_checkpoint = None

    kwargs = await create_server_kwargs(h)

    assert 'store_identity' in kwargs, (
        f'store_identity not passed to create_server; got {sorted(kwargs)}'
    )
    identity = kwargs['store_identity']
    assert isinstance(identity, StoreIdentity)
    assert identity.kind == 'project'
    assert identity.project_id == 'test-project'
    assert identity.project_root == mock_orch_config.project_root.resolve()
    assert identity.queue_dir == configured_queue_dir.resolve()
