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
from unittest.mock import AsyncMock, MagicMock, patch

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

    # Deliberate bypass: Harness.start() / _start_escalation_server (harness.py:2162-2188)
    # is NOT invoked — this fixture exercises only the escalation/harness boundary.
    # We hand-wire only the notify/resolve callbacks here, mirroring the callback
    # registrations in _start_escalation_server but omitting review_checkpoint.escalation_queue
    # wiring and MCP-server bring-up.
    # If _start_escalation_server grows new registrations (e.g. cleanup_supervisor or
    # review_checkpoint hooks), this fixture will silently drift — revisit if that method grows.
    h._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    h._escalation_queue.set_notify_callback(h._on_escalation)
    h._escalation_queue.set_resolve_callback(h._on_escalation_resolved)
    return h


# ---------------------------------------------------------------------------
# Helper (mirror escalation/tests/test_server_dedupe.py:30-33)
# ---------------------------------------------------------------------------


async def _blocker(server, **kwargs: Any) -> dict[str, Any]:
    tool = await server.get_tool('escalate_blocker')
    return await tool.fn(**kwargs)


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
    async def test_dedupe_response_shape_across_task_ids(self, harness: Harness):
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
        self, harness: Harness
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
        # Notify callback must have fired immediately on submit under A.
        assert events['A'].is_set(), (
            'Notify callback must fire under parent task A on submit — '
            '_on_escalation uses escalation.task_id to set the event.'
        )

        # Submit cross-task blocker from task B — dedupes; notify callback NOT fired.
        second = await _blocker(
            server,
            task_id='B',
            agent_role='implementer',
            category='infra_issue',
            summary='Fused-memory  CONNECTION timeout!',
        )
        assert second['status'] == 'dedup_skipped'
        # Dedupe must NOT fire the notify callback under B at submit time.
        assert not events['B'].is_set(), (
            'cross-task dedupe must not fire notify callback under B — '
            'the dedup_skipped path skips the notify fan-out for the child task_id.'
        )

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
        self, harness: Harness
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


# ---------------------------------------------------------------------------
# Step-7 RED tests: Harness._build_task_status_lookup + wiring into create_server
# ---------------------------------------------------------------------------


class TestHarnessTaskStatusLookup:
    """Harness._build_task_status_lookup() contract and create_server wiring.

    Step 7 of task 1366 (AFK A4a): verify the lookup factory and how it is
    injected into create_server via _start_escalation_server().
    """

    @pytest.mark.asyncio
    async def test_build_lookup_returns_async_callable(self, harness: Harness):
        """_build_task_status_lookup() returns a coroutine-function."""
        import inspect

        lookup = harness._build_task_status_lookup()
        assert callable(lookup), '_build_task_status_lookup() must return a callable'
        assert inspect.iscoroutinefunction(lookup), (
            '_build_task_status_lookup() must return an async callable (coroutine function)'
        )

    @pytest.mark.asyncio
    async def test_build_lookup_delegates_to_scheduler(self, harness: Harness):
        """Awaiting the lookup forwards to self.scheduler.get_status(task_id)."""
        # Stub the scheduler's get_status so we can verify delegation.
        harness.scheduler.get_status = AsyncMock(return_value='done')

        lookup = harness._build_task_status_lookup()
        result = await lookup('task-42')

        harness.scheduler.get_status.assert_called_once_with('task-42')
        assert result == 'done', f"Expected 'done', got: {result}"

    @pytest.mark.asyncio
    async def test_build_lookup_returns_none_when_scheduler_returns_none(
        self, harness: Harness,
    ):
        """Awaiting the lookup returns None when scheduler.get_status returns None."""
        harness.scheduler.get_status = AsyncMock(return_value=None)

        lookup = harness._build_task_status_lookup()
        result = await lookup('task-unknown')

        assert result is None

    @pytest.mark.asyncio
    async def test_start_escalation_server_passes_task_status_lookup(
        self, harness: Harness, tmp_path: Path,
    ):
        """_start_escalation_server() calls create_server with non-None task_status_lookup."""
        # Arrange: wire minimal escalation config on the harness's config mock.
        harness.config.escalation.queue_dir = str(tmp_path / 'esc')
        harness.config.escalation.host = '127.0.0.1'
        harness.config.escalation.port = 18100
        harness.config.project_root = tmp_path
        harness.review_checkpoint = None  # skip review_checkpoint wiring

        captured_kwargs: dict = {}

        def _spy_create_server(queue, **kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()  # minimal stand-in for FastMCP server

        # Mock task: done() returns False so the post-start health check passes.
        mock_task = MagicMock()
        mock_task.done.return_value = False

        captured_coros: list = []

        # Close the captured coroutine to prevent it from leaking as an
        # unawaited coroutine — pytest's sys.unraisablehook would otherwise
        # convert the GC-time RuntimeWarning into a PytestUnraisableException-
        # Warning attributed to whichever later test happens to be running
        # (non-deterministic xdist failure mode — task 1468).  Canonical
        # pattern: test_harness_watcher_supervisor.py:140-142.
        def _capture_create_task(coro, *, name=None):
            captured_coros.append(coro)
            coro.close()  # prevent 'coroutine was never awaited' RuntimeWarning during GC
            return mock_task

        with (
            patch('orchestrator.harness.create_server', side_effect=_spy_create_server),
            patch('asyncio.create_task', side_effect=_capture_create_task),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            await harness._start_escalation_server()

        assert 'task_status_lookup' in captured_kwargs, (
            'create_server must be called with task_status_lookup kwarg'
        )
        assert captured_kwargs['task_status_lookup'] is not None, (
            'task_status_lookup passed to create_server must not be None'
        )
        assert captured_coros, 'asyncio.create_task should have been invoked with the _serve coroutine'
        assert captured_coros[0].__qualname__.endswith('_serve'), (
            'asyncio.create_task must be called with the _serve coroutine; '
            f'got {captured_coros[0].__qualname__!r} instead'
        )

    @pytest.mark.asyncio
    async def test_start_escalation_server_lookup_callable_matches_build(
        self, harness: Harness, tmp_path: Path,
    ):
        """The task_status_lookup kwarg is the callable built by _build_task_status_lookup()."""
        harness.config.escalation.queue_dir = str(tmp_path / 'esc')
        harness.config.escalation.host = '127.0.0.1'
        harness.config.escalation.port = 18100
        harness.config.project_root = tmp_path
        harness.review_checkpoint = None
        harness.scheduler.get_status = AsyncMock(return_value='pending')

        captured_lookup: Any = None

        def _spy_create_server(queue, **kwargs):
            nonlocal captured_lookup
            captured_lookup = kwargs.get('task_status_lookup')
            return MagicMock()

        mock_task = MagicMock()
        mock_task.done.return_value = False

        captured_coros: list = []

        # Close the captured coroutine to prevent it from leaking as an
        # unawaited coroutine — pytest's sys.unraisablehook would otherwise
        # convert the GC-time RuntimeWarning into a PytestUnraisableException-
        # Warning attributed to whichever later test happens to be running
        # (non-deterministic xdist failure mode — task 1468).  Canonical
        # pattern: test_harness_watcher_supervisor.py:140-142.
        def _capture_create_task(coro, *, name=None):
            captured_coros.append(coro)
            coro.close()  # prevent 'coroutine was never awaited' RuntimeWarning during GC
            return mock_task

        with (
            patch('orchestrator.harness.create_server', side_effect=_spy_create_server),
            patch('asyncio.create_task', side_effect=_capture_create_task),
            patch('asyncio.sleep', new_callable=AsyncMock),
        ):
            await harness._start_escalation_server()

        assert captured_lookup is not None
        # Verify the captured lookup actually delegates to the scheduler
        result = await captured_lookup('task-99')
        harness.scheduler.get_status.assert_called_once_with('task-99')
        assert result == 'pending'
        assert captured_coros, 'asyncio.create_task should have been invoked with the _serve coroutine'
        assert captured_coros[0].__qualname__.endswith('_serve'), (
            'asyncio.create_task must be called with the _serve coroutine; '
            f'got {captured_coros[0].__qualname__!r} instead'
        )
