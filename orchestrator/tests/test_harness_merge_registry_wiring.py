"""Tests for the single shared InFlightMergeRegistry wiring in Harness.

Verifies that:
(step-9)  Harness.__init__ creates exactly one InFlightMergeRegistry instance.
(step-11) _start_escalation_server injects the SAME instance into create_server.
(step-13) _run_slot injects the SAME instance into TaskWorkflow.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import _init_harness_state_for_test

from orchestrator.harness import Harness
from orchestrator.merge_queue import InFlightMergeRegistry
from orchestrator.scheduler import TaskAssignment


# ---------------------------------------------------------------------------
# Shared harness builder (mirrors test_harness_startup_get_statuses)
# ---------------------------------------------------------------------------


def _build_harness(mock_orch_config) -> Harness:
    """Construct a Harness with heavy constructors patched out."""
    mock_orch_config.max_concurrent_tasks = 2
    mock_orch_config.fused_memory.project_id = 'test'

    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        return Harness(mock_orch_config)


# ---------------------------------------------------------------------------
# step-9: __init__ creates one InFlightMergeRegistry
# ---------------------------------------------------------------------------


class TestHarnessInitCreatesRegistry:
    """Harness.__init__ creates a single InFlightMergeRegistry instance."""

    def test_registry_attribute_created(self, mock_orch_config):
        """_merge_inflight_registry is an InFlightMergeRegistry after __init__."""
        h = _build_harness(mock_orch_config)
        assert isinstance(h._merge_inflight_registry, InFlightMergeRegistry), (
            f'Expected InFlightMergeRegistry, got {type(h._merge_inflight_registry)!r}'
        )


# ---------------------------------------------------------------------------
# step-11: _start_escalation_server injects the shared registry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStartEscalationServerInjectsRegistry:
    """_start_escalation_server passes the shared registry to create_server."""

    async def test_shared_registry_injected_into_create_server(
        self, tmp_path: Path, mock_orch_config,
    ):
        """create_server is called with merge_inflight_registry IS h._merge_inflight_registry.

        RED until step-12 adds the kwarg to the create_server call in harness.py.
        """
        h = _build_harness(mock_orch_config)

        # Set up attributes _start_escalation_server reads from config
        mock_orch_config.escalation.queue_dir = str(tmp_path / 'esc')
        mock_orch_config.escalation.host = '127.0.0.1'
        mock_orch_config.escalation.port = 19999
        # _start_escalation_server wires review_checkpoint if not None
        h.review_checkpoint = None
        h.event_store = None

        with patch('orchestrator.harness.create_server', return_value=MagicMock()) as mock_cs, \
             patch('uvicorn.Server') as mock_uv_cls:
            # Make the uvicorn serve() coroutine return immediately so the
            # background task finishes and doesn't leave a dangling Task.
            mock_uv_cls.return_value.serve = AsyncMock()
            await h._start_escalation_server()

        # Verify the injected kwarg (identity — same object, not just equal)
        call_kwargs = mock_cs.call_args.kwargs
        assert 'merge_inflight_registry' in call_kwargs, (
            f'Expected merge_inflight_registry kwarg in create_server call; '
            f'got kwargs={list(call_kwargs)!r}'
        )
        assert call_kwargs['merge_inflight_registry'] is h._merge_inflight_registry, (
            'Expected the SAME InFlightMergeRegistry instance as h._merge_inflight_registry'
        )

        # Clean up the escalation background task if it's still running
        if hasattr(h, '_escalation_task') and not h._escalation_task.done():
            h._escalation_task.cancel()
            try:
                await h._escalation_task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# step-13: _run_slot injects the shared registry into TaskWorkflow
# ---------------------------------------------------------------------------


@pytest.fixture
def harness_for_registry_run_slot() -> Harness:
    """Harness with all attributes needed to drive _run_slot directly.

    Mirrors harness_for_run_slot in test_harness_cancel_workflow.py, with
    _merge_inflight_registry added for the step-13 registry wiring test.
    """
    h = Harness.__new__(Harness)
    _init_harness_state_for_test(h)
    h._workflow_cancel_events = {}
    h._workflow_cancel_at = {}
    h._workflow_slot_tasks = {}
    h._terminal_cancel_counts = {}
    h._escalation_events = {}
    h._escalation_queue = None
    h._recovered_plans = {}
    h._recovered_sessions = {}
    h._preserved_worktrees = set()
    h.event_store = None
    h.config = None   # type: ignore[assignment]
    h.git_ops = None  # type: ignore[assignment]
    h.briefing = None  # type: ignore[assignment]
    h.mcp = None  # type: ignore[assignment]
    h.usage_gate = None
    h._merge_queue = None  # type: ignore[assignment]
    h._merge_worker = None
    # The shared registry — this is the object we assert is forwarded.
    h._merge_inflight_registry = InFlightMergeRegistry()
    h.cost_store = None
    h._run_store = None
    h._run_id = None
    h.review_checkpoint = None
    h.scheduler = MagicMock()
    h.scheduler.release = MagicMock()
    return h


@pytest.mark.asyncio
class TestRunSlotInjectsRegistryIntoWorkflow:
    """_run_slot passes the shared registry to TaskWorkflow."""

    async def test_shared_registry_passed_to_task_workflow(
        self, harness_for_registry_run_slot: Harness,
    ):
        """TaskWorkflow is constructed with merge_inflight_registry IS h._merge_inflight_registry.

        RED until step-14 adds the kwarg to the TaskWorkflow call in harness.py.
        """
        h = harness_for_registry_run_slot
        tid = '42'
        assignment = TaskAssignment(task_id=tid, task={'title': 't'}, modules=[])
        sem = asyncio.Semaphore(0)

        with patch('orchestrator.harness.TaskWorkflow') as mock_wf_cls:
            async def _wedge() -> None:
                await asyncio.sleep(3600)

            mock_wf = MagicMock()
            mock_wf.run = _wedge
            mock_wf_cls.return_value = mock_wf

            wrapper_task = asyncio.create_task(h._run_slot(assignment, sem))

            # Poll until _run_slot constructs TaskWorkflow (registers in slot tasks).
            for _ in range(50):
                if mock_wf_cls.called:
                    break
                await asyncio.sleep(0.01)

            assert mock_wf_cls.called, 'TaskWorkflow was not constructed by _run_slot'

            # Check the kwarg identity BEFORE cancelling the wedged task
            call_kwargs = mock_wf_cls.call_args.kwargs
            assert 'merge_inflight_registry' in call_kwargs, (
                f'Expected merge_inflight_registry kwarg in TaskWorkflow call; '
                f'got kwargs={list(call_kwargs)!r}'
            )
            assert call_kwargs['merge_inflight_registry'] is h._merge_inflight_registry, (
                'Expected the SAME InFlightMergeRegistry instance as h._merge_inflight_registry'
            )

            # Clean up: hard-cancel the wedged workflow and wait for wrapper_task
            if tid in h._workflow_slot_tasks:
                h.hard_cancel_workflow(tid)
            wrapper_task.cancel()
            try:
                await wrapper_task
            except (asyncio.CancelledError, Exception):
                pass
