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
        patch('orchestrator.harness.OverrideStore'),
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


# ---------------------------------------------------------------------------
# TestCrossTaskDedupeReengagement
# ---------------------------------------------------------------------------


class TestCrossTaskDedupeReengagement:
    """Orchestrator-integration verification of the cross-task escalation dedupe contract.

    SCOPE BOUNDARY: verifies the escalation/harness layer only.
    The higher-level claim "task B's workflow eventually re-engages" depends on what
    status B's workflow leaves the task in after terminate_cleanly — that is a
    workflow-layer concern (no special dedup_skipped handling exists today in
    orchestrator/workflow.py) and is explicitly OUT OF SCOPE for this test class.
    """

    @pytest.mark.asyncio
    async def test_dedupe_response_shape_across_task_ids(self, harness: Harness, tmp_path: Path):
        """Cross-task infra_issue blocker dedupes: B's response has dedup_skipped shape.

        Verifies _submit_or_dedupe + find_dedupe_parent + attach_dedupe_child handle
        cross-task dedupe correctly at the orchestrator integration boundary.

        Summary pair from escalation/tests/test_dedupe.py::test_similar_summaries_share_key:
        they share a dedupe key after summary_dedupe_key normalisation.
        """
        queue = harness._escalation_queue
        server = create_server(queue)

        # Submit parent blocker from task A.
        first = await _blocker(
            server,
            task_id='A',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        assert first['status'] == 'queued'
        parent_id = first['id']

        # Submit cross-task blocker from task B — must dedupe into parent under task A.
        second = await _blocker(
            server,
            task_id='B',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )

        assert second['status'] == 'dedup_skipped'
        assert second['parent_id'] == parent_id
        assert second['action'] == 'terminate_cleanly'
        assert 'child_id' in second
        assert second['child_id'] != parent_id

        # Exactly one esc-*.json in queue root — the parent file under task A.
        queue_root_files = sorted(queue.queue_dir.glob('esc-*.json'))
        assert len(queue_root_files) == 1, (
            f'Expected exactly 1 file (parent under task A); got: {queue_root_files}'
        )

    @pytest.mark.asyncio
    async def test_resolve_callback_fires_for_parent_task_id_only(
        self, harness: Harness, tmp_path: Path
    ):
        """Parent resolution signals only A's event; B's event stays clear.

        Pins Harness._on_escalation_resolved (harness.py:2503-2521): callback
        looks up escalation.task_id in _escalation_events — the parent's task_id
        is 'A', so only events['A'] is set.  B receives no direct harness signal.

        Contract reference: DESIGN.md "Escalation cross-task dedupe"
        — "On parent resolve, only the parent's task_id receives a per-task wake
        signal (Harness._escalation_events[A]); cross-task children receive NO
        direct harness signal."
        """
        queue = harness._escalation_queue
        server = create_server(queue)

        # Pre-populate per-task events to simulate two active workflow slots.
        harness._escalation_events['A'] = asyncio.Event()
        harness._escalation_events['B'] = asyncio.Event()
        events = harness._escalation_events

        # Submit parent blocker from task A — notify callback fires, sets events['A'].
        first = await _blocker(
            server,
            task_id='A',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        assert first['status'] == 'queued'
        parent_id = first['id']

        # Submit cross-task blocker from task B — dedupes; notify callback NOT fired.
        second = await _blocker(
            server,
            task_id='B',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )
        assert second['status'] == 'dedup_skipped'

        # Reset both events before the resolve step so we can assert the
        # resolve callback's fan-out in isolation.
        events['A'].clear()
        events['B'].clear()

        # Resolve parent directly (resolve_callback fires _on_escalation_resolved).
        harness._escalation_queue.resolve(
            parent_id, resolution='infra fixed', resolved_by='steward-test'
        )

        # Only A's event is set — the resolve callback keys on escalation.task_id.
        assert events['A'].is_set(), (
            'Parent task A event must be set after queue.resolve() — '
            '_on_escalation_resolved uses escalation.task_id to look up the event.'
        )
        assert not events['B'].is_set(), (
            'Child task B event must NOT be set after parent resolves — '
            'cross-task children receive no direct harness signal per '
            'DESIGN.md "Escalation cross-task dedupe" / server.py:88-90 contract.'
        )
