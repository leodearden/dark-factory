"""Tests for SpeculativeMergeWorker retroactive coalescing pass (task γ/1719).

Each TestClass maps to one plan step:
  step-1:  TestConfigAndEventType   — new config knob + new EventType
  step-3:  TestNoOpGuards           — early-return guards of _maybe_coalesce_waiting_singles
  step-5:  TestCoreFormation        — happy-path coalesce: 3 disjoint singles → 1 train
  step-7:  TestExclusionIdempotency — in-flight, detached, and GroupMergeRequest exclusions
  step-9:  TestPartialStackability  — overlap, stack-conflict eject, survivors<2
  step-11: TestDebounce             — signature-based deduplication
  step-13: TestEndToEndWiring       — merger-loop wiring: coalesced train dispatched end-to-end
"""

from __future__ import annotations

import asyncio
import collections
import sys
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _orch_helpers import make_placeholder_future

# ─── Step 1 ─────────────────────────────────────────────────────────────────

class TestConfigAndEventType:
    """step-1 (RED): new config knob and EventType member exist with correct defaults."""

    def test_merge_train_coalesce_enabled_default_false(self):
        """OrchestratorConfig.merge_train_coalesce_enabled defaults to False (OFF by default,
        human-flips after soak — fold-the-decision norm).
        """
        from orchestrator.config import GitConfig, OrchestratorConfig
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cfg = OrchestratorConfig(
                project_root=Path(tmp),
                git=GitConfig(
                    main_branch='main',
                    branch_prefix='task/',
                    remote='origin',
                    worktree_dir='.worktrees',
                ),
            )
            assert cfg.merge_train_coalesce_enabled is False

    def test_event_type_train_coalesced(self):
        """EventType.train_coalesced has value 'train_coalesced'."""
        from orchestrator.event_store import EventType
        assert EventType.train_coalesced == 'train_coalesced'


# ─── helpers shared across steps ────────────────────────────────────────────

def _make_config(tmp_path: Path, *, coalesce_enabled: bool = False) -> 'OrchestratorConfig':
    from orchestrator.config import GitConfig, OrchestratorConfig
    return OrchestratorConfig(
        project_root=tmp_path,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
            push_after_advance=False,
        ),
        merge_train_coalesce_enabled=coalesce_enabled,
    )


def _make_single_req(
    task_id: str,
    *,
    config: 'OrchestratorConfig',
    worktree: Path | None = None,
    lane: Literal['normal', 'high'] = 'normal',
) -> 'MergeRequest':
    from orchestrator.merge_queue import MergeRequest
    loop = asyncio.get_event_loop()
    try:
        future = asyncio.get_running_loop().create_future()
    except RuntimeError:
        future = make_placeholder_future()
    return MergeRequest(
        task_id=task_id,
        branch=f'task/{task_id}',
        worktree=worktree or Path(f'/tmp/wt-{task_id}'),
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
        lane=lane,
    )


def _stub_factory() -> 'TrainCallbackFactory':
    """Return a simple TrainCallbackFactory stub."""
    from orchestrator.merge_queue import TrainCallbacks
    def _factory(train_id: str) -> 'TrainCallbacks':
        return TrainCallbacks(
            status_check=AsyncMock(return_value={}),
            mark_member_done=AsyncMock(),
        )
    return _factory


def _make_worker(
    queue: 'asyncio.Queue',
    config: 'OrchestratorConfig',
    git_ops: 'GitOps | None' = None,
    factory=None,
) -> 'SpeculativeMergeWorker':
    from orchestrator.merge_queue import SpeculativeMergeWorker
    from orchestrator.git_ops import GitOps, GitConfig
    if git_ops is None:
        # stub git_ops — not used in no-op guard tests
        git_ops = MagicMock()
        git_ops.config = config
    worker = SpeculativeMergeWorker(
        git_ops,
        queue,
        train_callback_factory=factory,
    )
    # Wire config onto the worker for the pass to read it.
    worker._config = config
    return worker


# ─── Step 3 ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestNoOpGuards:
    """step-3 (RED): early-return guards of _maybe_coalesce_waiting_singles."""

    async def test_knob_disabled_returns_false_no_mutation(self, tmp_path: Path):
        """(a) merge_train_coalesce_enabled=False → returns False, buffer unchanged."""
        from orchestrator.merge_queue import GroupMergeRequest
        cfg = _make_config(tmp_path, coalesce_enabled=False)
        queue: asyncio.Queue = asyncio.Queue()
        worker = _make_worker(queue, cfg, factory=_stub_factory())

        req1 = _make_single_req('t1', config=cfg)
        req2 = _make_single_req('t2', config=cfg)
        worker._lane_buffers['normal'].append(req1)
        worker._lane_buffers['normal'].append(req2)

        result = await worker._maybe_coalesce_waiting_singles()

        assert result is False, 'should return False when knob is disabled'
        buf = list(worker._lane_buffers['normal'])
        assert buf == [req1, req2], 'buffer must be unchanged'
        # No futures resolved
        assert not req1.result.done()
        assert not req2.result.done()

    async def test_single_candidate_returns_false(self, tmp_path: Path):
        """(b) knob enabled, only 1 candidate → returns False (need ≥2)."""
        cfg = _make_config(tmp_path, coalesce_enabled=True)
        queue: asyncio.Queue = asyncio.Queue()
        worker = _make_worker(queue, cfg, factory=_stub_factory())

        req1 = _make_single_req('t1', config=cfg)
        worker._lane_buffers['normal'].append(req1)

        result = await worker._maybe_coalesce_waiting_singles()

        assert result is False
        assert list(worker._lane_buffers['normal']) == [req1]
        assert not req1.result.done()

    async def test_no_factory_returns_false(self, tmp_path: Path):
        """(c) knob enabled but _train_callback_factory is None → returns False."""
        cfg = _make_config(tmp_path, coalesce_enabled=True)
        queue: asyncio.Queue = asyncio.Queue()
        worker = _make_worker(queue, cfg, factory=None)  # no factory

        req1 = _make_single_req('t1', config=cfg)
        req2 = _make_single_req('t2', config=cfg)
        worker._lane_buffers['normal'].append(req1)
        worker._lane_buffers['normal'].append(req2)

        result = await worker._maybe_coalesce_waiting_singles()

        assert result is False
        assert not req1.result.done()
        assert not req2.result.done()
