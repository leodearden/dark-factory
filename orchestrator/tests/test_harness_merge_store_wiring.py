"""Tests for durable merge queue journal (MergeQueueStore) wiring in Harness (task 1772).

Verifies that:
(step-19) Test 1: Harness.__init__ constructs self._merge_store as a MergeQueueStore
           with path = project_root / 'data' / 'orchestrator' / 'merge_queue.json'.

(step-19) Test 2: _start_merge_worker passes merge_store=harness._merge_store
           to SpeculativeMergeWorker.

(step-19) Test 3: _recover_pending_merges (new method) delegates to
           recover_pending_merges with harness._merge_store, harness._merge_queue,
           harness.git_ops, harness.config, harness.event_store, and
           main_branch/branch_prefix from harness.config.git.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from orchestrator.harness import Harness
from orchestrator.merge_queue_store import MergeQueueStore

# ---------------------------------------------------------------------------
# Shared harness builder — mirrors test_harness_merge_registry_wiring
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
# Test 1 — __init__ creates _merge_store at the expected path
# ---------------------------------------------------------------------------


class TestHarnessInitCreatesMergeStore:
    """Harness.__init__ creates a MergeQueueStore at the canonical path."""

    def test_merge_store_attribute_created(self, mock_orch_config, tmp_path: Path):
        """_merge_store is a MergeQueueStore after __init__.

        RED until step-20 adds:
            self._merge_store = MergeQueueStore(
                self.config.project_root / 'data' / 'orchestrator' / 'merge_queue.json'
            )
        to Harness.__init__.
        """
        h = _build_harness(mock_orch_config)

        assert isinstance(h._merge_store, MergeQueueStore), (
            f'Expected MergeQueueStore, got {type(h._merge_store)!r}'
        )

    def test_merge_store_path(self, mock_orch_config, tmp_path: Path):
        """_merge_store._path == project_root / data / orchestrator / merge_queue.json."""
        h = _build_harness(mock_orch_config)

        expected_path = tmp_path / 'data' / 'orchestrator' / 'merge_queue.json'
        assert h._merge_store._path == expected_path, (
            f'Expected {expected_path!r}, got {h._merge_store._path!r}'
        )


# ---------------------------------------------------------------------------
# Test 2 — _start_merge_worker passes merge_store to SpeculativeMergeWorker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStartMergeWorkerInjectsMergeStore:
    """_start_merge_worker passes merge_store=self._merge_store to SpeculativeMergeWorker."""

    async def test_merge_store_injected(self, mock_orch_config, tmp_path: Path):
        """SpeculativeMergeWorker is constructed with merge_store=harness._merge_store.

        RED until step-20 adds ``merge_store=self._merge_store`` to the
        SpeculativeMergeWorker(...) call in _start_merge_worker.
        """
        h = _build_harness(mock_orch_config)

        # Set up attributes _start_merge_worker reads from self / config.
        mock_orch_config.enabled_verify_runners = []
        h.event_store = None
        h.scheduler = MagicMock()

        # Patch the worker so it doesn't actually run + liveness checks pass.
        mock_worker = MagicMock()
        mock_worker.run = AsyncMock(return_value=None)

        with patch('orchestrator.merge_queue.SpeculativeMergeWorker',
                   return_value=mock_worker) as mock_cls, \
             patch('orchestrator.merge_queue.enforce_merge_liveness_margin'), \
             patch('orchestrator.merge_queue.enforce_persistent_worktree_serial_lane'), \
             patch('orchestrator.harness.StaleServiceRestartCoordinator'), \
             patch('orchestrator.harness.build_train_callback_factory',
                   return_value=MagicMock()):
            await h._start_merge_worker()

        # Locate the SpeculativeMergeWorker constructor call.
        assert mock_cls.called, 'SpeculativeMergeWorker was never constructed'
        call_kwargs = mock_cls.call_args.kwargs

        assert 'merge_store' in call_kwargs, (
            f'merge_store kwarg missing from SpeculativeMergeWorker call; '
            f'got kwargs={list(call_kwargs)!r}'
        )
        assert call_kwargs['merge_store'] is h._merge_store, (
            'Expected merge_store to be the SAME MergeQueueStore as h._merge_store'
        )

        # Clean up the merge-worker background task.
        task = getattr(h, '_merge_worker_task', None)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# Test 3 — _recover_pending_merges delegates to recover_pending_merges
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecoverPendingMergesDelegates:
    """harness._recover_pending_merges() delegates to recover_pending_merges."""

    async def test_delegation_with_correct_args(self, mock_orch_config, tmp_path: Path):
        """_recover_pending_merges calls recover_pending_merges with the right args.

        RED until step-20 adds:
            async def _recover_pending_merges(self):
                await recover_pending_merges(
                    self._merge_store, self._merge_queue, self.git_ops, self.config,
                    event_store=self.event_store,
                    main_branch=self.config.git.main_branch,
                    branch_prefix=self.config.git.branch_prefix,
                )
        and imports recover_pending_merges in harness.py.
        """
        h = _build_harness(mock_orch_config)
        h.event_store = None

        with patch('orchestrator.harness.recover_pending_merges',
                   new=AsyncMock(return_value={'recovered': 0, 'dropped': 0, 'requests': []}),
                   ) as mock_recover:
            await h._recover_pending_merges()

        assert mock_recover.called, '_recover_pending_merges did not call recover_pending_merges'
        call_args = mock_recover.call_args

        # Positional args: store, queue, git_ops, config
        pos = call_args.args
        assert len(pos) >= 4, (
            f'Expected at least 4 positional args; got {len(pos)}: {pos!r}'
        )
        assert pos[0] is h._merge_store, (
            f'First arg (store) should be h._merge_store; got {pos[0]!r}'
        )
        assert pos[1] is h._merge_queue, (
            f'Second arg (queue) should be h._merge_queue; got {pos[1]!r}'
        )
        assert pos[2] is h.git_ops, (
            f'Third arg (git_ops) should be h.git_ops; got {pos[2]!r}'
        )
        assert pos[3] is h.config, (
            f'Fourth arg (config) should be h.config; got {pos[3]!r}'
        )

        # Keyword args
        kw = call_args.kwargs
        assert kw.get('event_store') is h.event_store, (
            f'event_store kwarg should be h.event_store; got {kw.get("event_store")!r}'
        )
        assert kw.get('main_branch') == h.config.git.main_branch, (
            f'main_branch should be {h.config.git.main_branch!r}; '
            f'got {kw.get("main_branch")!r}'
        )
        assert kw.get('branch_prefix') == h.config.git.branch_prefix, (
            f'branch_prefix should be {h.config.git.branch_prefix!r}; '
            f'got {kw.get("branch_prefix")!r}'
        )
