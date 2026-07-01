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
        # Realistic-length (> 12 char) SHAs so the [:12] abbreviation in the
        # log line is actually exercised, rather than being a no-op slice.
        base_sha = 'basedeadbeef0123456789'
        head_sha = 'headfeedface9876543210'

        with caplog.at_level(logging.INFO):
            await harness._note_merge_all('t', base_sha, head_sha)

        assert 'offline-lane: on_post_merge' in caplog.text
        assert base_sha[:12] in caplog.text
        assert head_sha[:12] in caplog.text
        # The truncation actually truncates -- the full untruncated SHAs must
        # not appear in the log line.
        assert base_sha not in caplog.text
        assert head_sha not in caplog.text

    async def test_offline_lane_notifiee_raise_is_swallowed_fail_open(
        self, harness: Harness, caplog
    ):
        """A raising offline-lane notifiee must not block the coordinator fan-out."""
        harness._offline_lane_notifiee = AsyncMock(side_effect=RuntimeError('lane boom'))
        coord_a = MagicMock()
        coord_a.note_merge = AsyncMock()
        coord_b = MagicMock()
        coord_b.note_merge = AsyncMock()
        harness._service_restart_coordinators = [coord_a, coord_b]

        with caplog.at_level(logging.WARNING):
            # Must NOT raise
            await harness._note_merge_all('task-2', 'base2', 'head2')

        coord_a.note_merge.assert_awaited_once_with(
            'task-2', 'base2', 'head2', prefetched_diff=[]
        )
        coord_b.note_merge.assert_awaited_once_with(
            'task-2', 'base2', 'head2', prefetched_diff=[]
        )
        assert 'fail-open' in caplog.text
        assert 'task-2' in caplog.text

    async def test_offline_lane_fires_even_when_git_diff_fetch_fails(
        self, harness: Harness
    ):
        """The offline lane needs only "main advanced", not the changed-file list.

        It must fire even when get_merge_diff_files errors -- unlike the
        service-restart coordinators, which are skipped on a diff-fetch error.
        """
        harness.git_ops.get_merge_diff_files = AsyncMock(
            return_value=([], RuntimeError('git boom'))
        )
        harness._offline_lane_notifiee = AsyncMock()
        coord_a = MagicMock()
        coord_a.note_merge = AsyncMock()
        coord_b = MagicMock()
        coord_b.note_merge = AsyncMock()
        harness._service_restart_coordinators = [coord_a, coord_b]

        await harness._note_merge_all('task-3', 'base3', 'head3')

        harness._offline_lane_notifiee.assert_awaited_once_with(
            'task-3', 'base3', 'head3'
        )
        coord_a.note_merge.assert_not_awaited()
        coord_b.note_merge.assert_not_awaited()

    async def test_offline_lane_noop_when_notifiee_unset(
        self, harness: Harness, caplog
    ):
        """Default production state during β1: _offline_lane_notifiee is None.

        _note_merge_all (and _note_offline_lane) must be a clean no-op for the
        lane -- no raise, no 'offline-lane' log line -- while the
        service-restart coordinator fan-out proceeds exactly as before.
        """
        assert harness._offline_lane_notifiee is None
        coord_a = MagicMock()
        coord_a.note_merge = AsyncMock()
        coord_b = MagicMock()
        coord_b.note_merge = AsyncMock()
        harness._service_restart_coordinators = [coord_a, coord_b]

        with caplog.at_level(logging.INFO):
            # Must NOT raise
            await harness._note_merge_all('task-4', 'base4', 'head4')

        assert 'offline-lane' not in caplog.text
        coord_a.note_merge.assert_awaited_once_with(
            'task-4', 'base4', 'head4', prefetched_diff=[]
        )
        coord_b.note_merge.assert_awaited_once_with(
            'task-4', 'base4', 'head4', prefetched_diff=[]
        )
