"""Tests for the SpeculativeMergeWorker <-> SpeculationController wiring, the
additive ``snapshot()['speculation']`` key, and the permit-conservation
property across live concurrent-verify scenarios (MQ-refactor theta / task
1993).

Steps covered:
  step-9  RED   — worker wiring + additive snapshot key
  step-10 GREEN — wire SpeculationController into __init__ + snapshot()
  step-11 RED   — permit-conservation property test across live scenarios
  step-12 GREEN — _merger_loop delegates all speculation state to the
                  controller

This module reuses the bare-worker git_repo/git_config/git_ops fixtures from
test_merge_queue_request_liveness.py (per-file duplication convention — see
that module's docstring) and, from step-11 onward, the concurrent-verify
scenario helpers from test_merge_queue_concurrent_verify.py. It imports
orchestrator.merge_queue / orchestrator.merge_speculation_controller LOCALLY
inside each test (mirrors test_merge_request_ledger.py's convention) so a
not-yet-wired attribute/key never breaks collection during the RED steps.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run

# ---------------------------------------------------------------------------
# Fixtures (per-file duplication convention — see
# test_merge_queue_request_liveness.py)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    """Initialise a bare git repository with a single commit on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


# ---------------------------------------------------------------------------
# step-9 RED / step-10 GREEN: worker wiring + additive snapshot key
# ---------------------------------------------------------------------------

# The full pre-existing snapshot() key set (must remain present — additive-only
# freeze). See merge_queue.py's snapshot() return dict.
_PRE_EXISTING_SNAPSHOT_KEYS = {
    'entries',
    'depth',
    'head_of_line',
    'verify_in_progress',
    'is_wip_halted',
    'halt_owner_esc_id',
    'occupancy',
    'suffix_conflict_graph',
    'metrics',
    'frozen_prefix',
    'two_layer_invariants',
}


@pytest.mark.asyncio
class TestWorkerWiringAndAdditiveSnapshotKey:
    """SpeculativeMergeWorker <-> SpeculationController wiring (task 1993 step-9).

    RED until step-10 GREEN wires the controller into __init__/snapshot().
    """

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_worker_constructs_speculation_controller_sharing_the_slot(
        self, git_ops: GitOps, depth: int,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker
        from orchestrator.merge_speculation_controller import SpeculationController

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, speculation_depth=depth)

        assert isinstance(worker._speculation_controller, SpeculationController)
        # Same object — so the untouched verifier-side releases (which call
        # self._speculation_slot.release() directly) keep working unchanged.
        assert worker._speculation_controller._slot is worker._speculation_slot
        assert worker._speculation_controller._depth == worker._speculation_depth
        assert worker._speculation_depth == depth

    @pytest.mark.parametrize('depth', [1, 2])
    async def test_snapshot_speculation_key_is_additive_with_initial_values(
        self, git_ops: GitOps, depth: int,
    ) -> None:
        from orchestrator.merge_queue import SpeculativeMergeWorker

        queue: asyncio.Queue = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, queue, speculation_depth=depth)

        snap = worker.snapshot()

        # Additive-only: every pre-existing key stays present.
        assert _PRE_EXISTING_SNAPSHOT_KEYS <= set(snap)

        spec = snap['speculation']
        assert spec['depth'] == depth
        assert spec['held_by_merger'] == 0
        assert spec['slot_available'] == depth
        assert spec['inflight_speculative'] == 0
