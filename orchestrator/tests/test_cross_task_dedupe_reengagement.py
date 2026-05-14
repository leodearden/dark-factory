"""Orchestrator-integration test for the cross-task escalation dedupe re-engagement contract.

Task 1342. Complements escalation/tests/test_server_dedupe.py::TestCrossTaskChildResumeContract
(which pins the contract at the escalation-API layer) by verifying one layer up: that the real
Harness._on_escalation_resolved callback signals only the parent task's asyncio.Event, and that
after parent resolution the EscalationQueue gates (get_by_task(B, pending, L0) == [] and
has_open_l1(B) is False) are clear, making B eligible for re-acquisition by a natural scheduler
sweep.

Contract source: server.py:88-90 pointer → DESIGN.md
"Escalation cross-task dedupe: re-run-on-next-invocation contract".
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from escalation.queue import EscalationQueue
from escalation.server import create_server

from orchestrator.harness import Harness


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config) -> Harness:
    """Real Harness wired to a real EscalationQueue; heavy subsystems mocked.

    Both notify and resolve callbacks are wired so _on_escalation and
    _on_escalation_resolved fire for real on queue operations.
    """
    mock_orch_config.orphan_l0_reaper_enabled = False
    mock_orch_config.terminal_status_watcher_enabled = False
    mock_orch_config.stranded_reconcile_enabled = False

    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    h._escalation_queue.set_notify_callback(h._on_escalation)
    h._escalation_queue.set_resolve_callback(h._on_escalation_resolved)
    return h


# ---------------------------------------------------------------------------
# Helpers (mirror escalation/tests/test_server_dedupe.py:30-39)
# ---------------------------------------------------------------------------


async def _blocker(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_blocker')
    # escalate_blocker is a sync tool — tool.fn() returns dict directly
    return tool.fn(**kwargs)


async def _info(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_info')
    # escalate_info is a sync tool — tool.fn() returns dict directly
    return tool.fn(**kwargs)
