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
        assert queue is not None
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
        assert queue is not None
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
        queue.resolve(
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

    @pytest.mark.asyncio
    async def test_child_task_has_no_remaining_escalation_gates_after_resolve(
        self, harness: Harness, tmp_path: Path
    ):
        """After parent resolve, both A and B have clear scheduler-gate predicates.

        Asserts the exact gates that Scheduler.acquire_next checks before
        re-dispatching a task:
          - get_by_task(task_id, status='pending', level=0) == []
          - has_open_l1(task_id) is False

        Both A (parent resolved/archived) and B (no file ever written under B)
        must satisfy these gates, making them eligible for re-acquisition.
        """
        queue = harness._escalation_queue
        assert queue is not None
        server = create_server(queue)

        # Submit parent from A, dedupe child from B, then resolve the parent.
        first = await _blocker(
            server,
            task_id='A',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        parent_id = first['id']

        second = await _blocker(
            server,
            task_id='B',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )
        assert second['status'] == 'dedup_skipped'

        queue.resolve(
            parent_id, resolution='infra fixed', resolved_by='steward-test'
        )

        # B: no pending L0 — dedupe meant no file was ever written under task_id='B'.
        assert queue.get_by_task('B', status='pending', level=0) == [], (
            'B must have no pending L0 escalations — cross-task dedupe wrote no '
            'file under task_id=B, so the scheduler gate is clear.'
        )
        assert not queue.has_open_l1('B'), (
            'B must have no open L1 escalations after parent resolve.'
        )

        # A: parent archived on resolve — no pending L0 or L1 remains.
        assert queue.get_by_task('A', status='pending', level=0) == [], (
            'A must have no pending L0 escalations after parent is resolved/archived.'
        )
        assert not queue.has_open_l1('A'), (
            'A must have no open L1 escalations after parent resolve.'
        )

    @pytest.mark.asyncio
    async def test_natural_sweep_can_re_engage_child_after_resolve(
        self, harness: Harness, tmp_path: Path
    ):
        """B becomes re-acquirable by a natural scheduler sweep after parent resolves.

        Uses a thin local _SweepModel that mirrors Scheduler.acquire_next's terminal
        gates — skip non-pending tasks, skip tasks with pending L0, skip tasks with
        open L1 — rather than spinning up the real Scheduler (which requires a
        fused-memory backend and full dispatch graph).

        SCOPE BOUNDARY: this verifies the escalation/harness layer only.
        The higher-level claim "task B's workflow eventually re-engages" depends on
        what status B's workflow leaves the task in after terminate_cleanly — that
        is a workflow-layer concern and is out of scope here.

        Design note: the sweep model pre-sets both tasks to 'pending' after the
        escalation and resolve sequence, modelling the assumption that "task B
        returns to pending after the agent terminates cleanly" that the contract
        relies on.  If B's status is not returned to pending by the workflow layer,
        the scheduler cannot pick B up regardless of escalation gate state — but
        that is the workflow-layer concern noted above.
        """
        queue = harness._escalation_queue
        assert queue is not None
        server = create_server(queue)

        # --- Local sweep model mirroring Scheduler.acquire_next's terminal gates ---

        class _SweepModel:
            """Minimal sweep model: mirrors the three gates from Scheduler.acquire_next."""

            def __init__(self, task_statuses: dict[str, str]) -> None:
                self.task_statuses = dict(task_statuses)
                self.acquire_log: list[str] = []

            def acquire_next(self, q: EscalationQueue) -> str | None:
                for tid, status in self.task_statuses.items():
                    if status != 'pending':
                        continue
                    if q.get_by_task(tid, status='pending', level=0):
                        continue
                    if q.has_open_l1(tid):
                        continue
                    return tid
                return None

        sweep = _SweepModel({'A': 'pending', 'B': 'pending'})

        # Simulate two active workflow slots: both tasks "in-progress".
        sweep.task_statuses['A'] = 'in-progress'
        sweep.task_statuses['B'] = 'in-progress'

        # Submit A's parent blocker, B's deduped blocker.
        first = await _blocker(
            server,
            task_id='A',
            agent_role='implementer',
            category='infra_issue',
            summary='fused-memory connection timeout on port 8002',
        )
        parent_id = first['id']

        second = await _blocker(
            server,
            task_id='B',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )
        assert second['status'] == 'dedup_skipped', (
            f'Expected dedup_skipped, got: {second}'
        )

        # Resolve parent.
        queue.resolve(
            parent_id, resolution='infra fixed', resolved_by='steward-test'
        )

        # Model workflow-slot release: both tasks return to pending.
        # (This models "task returns to pending after agent terminates_cleanly".)
        sweep.task_statuses['A'] = 'pending'
        sweep.task_statuses['B'] = 'pending'

        # Run up to 5 sweep ticks; consume each acquired task by setting it in-progress.
        for _ in range(5):
            tid = sweep.acquire_next(queue)
            if tid is not None:
                sweep.acquire_log.append(tid)
                sweep.task_statuses[tid] = 'in-progress'

        assert 'B' in sweep.acquire_log, (
            f'Task B was never re-acquired in 5 sweep ticks after parent resolved. '
            f'acquire_log={sweep.acquire_log}. '
            f'If this fails, the escalation gates were not cleared — see Phase-2 '
            f'finding F2 option B (explicit per-child resume fan-out) in DESIGN.md.'
        )
