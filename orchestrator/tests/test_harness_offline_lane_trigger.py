"""Tests for the offline deep-test lane's on_post_merge trigger seam (task 1951, β1).

β1 adds an optional ``_offline_lane_notifiee`` slot to Harness and fans out to it
from ``_note_merge_all`` (the SpeculativeMergeWorker's ``on_merge_landed`` callback),
fail-open, alongside the existing service-restart coordinator fan-out.

β2 (the offline lane worker, not yet built) will later set the slot to its own
``on_post_merge`` callback — this module tests ONLY the trigger seam, with a bare
AsyncMock standing in for the not-yet-built worker's notifiee.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.harness import Harness

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def harness(tmp_path: Path, mock_orch_config):
    """Minimal harness with mocked heavy deps, modeled on test_harness_service_restart.py."""
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)

    h.scheduler = MagicMock()
    h.scheduler._dispatched = set()
    h.event_store = MagicMock()
    h.git_ops.get_merge_diff_files = AsyncMock(return_value=([], None))
    return h


# ---------------------------------------------------------------------------
# offline-lane on_post_merge notifiee
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOfflineLaneTrigger:
    """_note_merge_all fans out to the offline-lane notifiee, fail-open."""

    async def test_note_merge_all_invokes_offline_lane_notifiee_with_exact_shas(
        self, harness: Harness
    ):
        """When _offline_lane_notifiee is set, _note_merge_all awaits it with the exact SHAs."""
        harness._offline_lane_notifiee = AsyncMock()

        await harness._note_merge_all('task-1', 'base-sha', 'head-sha')

        harness._offline_lane_notifiee.assert_awaited_once_with(
            'task-1', 'base-sha', 'head-sha'
        )

    async def test_offline_lane_logs_on_post_merge_line(
        self, harness: Harness, caplog
    ):
        """_note_offline_lane logs an operator-visible on_post_merge line with abbreviated SHAs."""
        harness._offline_lane_notifiee = AsyncMock()

        with caplog.at_level(logging.INFO):
            await harness._note_merge_all('t', 'base-sha', 'head-sha')

        assert 'offline-lane: on_post_merge' in caplog.text
        assert 'base-sha'[:12] in caplog.text
        assert 'head-sha'[:12] in caplog.text
