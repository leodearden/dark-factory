"""Tests for Harness._file_dirty_tree_escalation (task 2380).

The dirty-tree start-guard used to raise RuntimeError and refuse to start
(see the now-superseded assertion in test_singleton_lock.py).  RCA
2026-07-08: composed with systemd Restart=on-failure + the watchdog revive
loop, that refusal became a silent multi-hour crash-loop (459 aborted runs,
nothing escalated because the process died before it could file anything).

These tests exercise the new deferred helper
``Harness._file_dirty_tree_escalation`` directly against a real
EscalationQueue, mirroring the bare-Harness construction pattern in
test_singleton_lock.py (McpLifecycle/Scheduler/BriefingAssembler patched out
of __init__, git_ops replaced with a MagicMock).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.queue import EscalationQueue

from orchestrator.harness import _DIRTY_TREE_ESCALATION_SENTINEL, Harness


def _build_bare_harness(mock_orch_config, tmp_path: Path) -> Harness:
    with patch('orchestrator.harness.McpLifecycle'), \
         patch('orchestrator.harness.Scheduler'), \
         patch('orchestrator.harness.BriefingAssembler'):
        h = Harness(mock_orch_config)
    h.git_ops = MagicMock()
    h._escalation_queue = EscalationQueue(tmp_path / 'queue')
    return h


class TestFileDirtyTreeEscalation:
    """Unit tests for the deferred dirty-tree escalation helper."""

    @pytest.mark.asyncio
    async def test_dirty_tree_files_born_at_l2_with_file_list(
        self, mock_orch_config, tmp_path,
    ):
        h = _build_bare_harness(mock_orch_config, tmp_path)
        h.git_ops.has_dirty_working_tree = AsyncMock(
            return_value='src/foo.py\nsrc/bar.py',
        )

        await h._file_dirty_tree_escalation(force_dirty_start=False)

        escs = h._escalation_queue.get_by_task(
            _DIRTY_TREE_ESCALATION_SENTINEL, status='pending', level=2,
        )
        assert len(escs) == 1
        esc = escs[0]
        assert esc.severity == 'critical'
        assert esc.level == 2
        assert esc.agent_role.startswith('orchestrator-')
        assert esc.category == 'cleanup_needed'
        assert 'src/foo.py' in esc.detail
        assert 'src/bar.py' in esc.detail

    @pytest.mark.asyncio
    async def test_clean_tree_files_nothing(self, mock_orch_config, tmp_path):
        h = _build_bare_harness(mock_orch_config, tmp_path)
        h.git_ops.has_dirty_working_tree = AsyncMock(return_value='')

        await h._file_dirty_tree_escalation(force_dirty_start=False)

        assert h._escalation_queue.get_pending() == []

    @pytest.mark.asyncio
    async def test_force_dirty_start_skips_escalation(
        self, mock_orch_config, tmp_path,
    ):
        h = _build_bare_harness(mock_orch_config, tmp_path)
        h.git_ops.has_dirty_working_tree = AsyncMock(return_value='src/foo.py')

        await h._file_dirty_tree_escalation(force_dirty_start=True)

        escs = h._escalation_queue.get_by_task(
            _DIRTY_TREE_ESCALATION_SENTINEL, status='pending', level=2,
        )
        assert escs == []

    @pytest.mark.asyncio
    async def test_repeated_dirty_startup_refreshes_instead_of_duplicating(
        self, mock_orch_config, tmp_path,
    ):
        """A restart while still dirty must refresh the L2 in place, not duplicate it."""
        h = _build_bare_harness(mock_orch_config, tmp_path)
        h.git_ops.has_dirty_working_tree = AsyncMock(return_value='src/foo.py')

        await h._file_dirty_tree_escalation(force_dirty_start=False)

        # Simulate a watchdog restart: still dirty, but the file list changed.
        h.git_ops.has_dirty_working_tree = AsyncMock(
            return_value='src/foo.py\nsrc/baz.py',
        )
        await h._file_dirty_tree_escalation(force_dirty_start=False)

        escs = h._escalation_queue.get_by_task(
            _DIRTY_TREE_ESCALATION_SENTINEL, status='pending', level=2,
        )
        assert len(escs) == 1, 'second dirty startup must refresh, not duplicate'
        esc = escs[0]
        assert esc.dedupe_count >= 1
        assert 'src/baz.py' in esc.detail
