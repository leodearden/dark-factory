"""Tests for the single shared InFlightMergeRegistry wiring in Harness.

Verifies that:
(step-9)  Harness.__init__ creates exactly one InFlightMergeRegistry instance.
(step-11) _start_escalation_server injects the SAME instance into create_server.
(step-13) _run_slot injects the SAME instance into TaskWorkflow.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness
from orchestrator.merge_queue import InFlightMergeRegistry


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
