"""Tests for SuffixConflictTracker (MQ-refactor task δ=1988): conflict-graph +
bounce-state extraction from SpeculativeMergeWorker into orchestrator.suffix_graph.

Covers:
  step-1 RED  — module surface (SuffixConflictGraph / EMPTY_SUFFIX_CONFLICT_GRAPH /
                SuffixConflictTracker importable from orchestrator.suffix_graph),
                merge_queue shim re-export identity, module logger name, and
                SuffixConflictTracker construction WITHOUT a worker.
  step-3 RED  — SuffixConflictTracker.recompute() without a worker.
  step-5 RED  — SuffixConflictTracker.bounce_conflicting_suffix_items() without
                a worker.
  step-7 RED  — SpeculativeMergeWorker delegates to self._suffix_tracker.
"""

from __future__ import annotations

import asyncio
import collections
from pathlib import Path

import pytest

import orchestrator.merge_queue as merge_queue
import orchestrator.suffix_graph as suffix_graph
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeRequest
from orchestrator.merge_types import MergeBounceRegistry
from orchestrator.suffix_graph import (
    EMPTY_SUFFIX_CONFLICT_GRAPH,
    SuffixConflictGraph,
    SuffixConflictTracker,
)

# ── fixtures (mirrors test_merge_queue_conflict_graph.py) ──────────────────────


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    (repo / 'shared.txt').write_text('line1\nline2\nline3\n')
    (repo / 'disjoint.txt').write_text('aaa\nbbb\nccc\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
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


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future."""
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


def _make_tracker(
    git_ops: GitOps,
    *,
    lane_buffers=None,
    frozen_prefix=None,
    frozen_prefix_tip=None,
) -> SuffixConflictTracker:
    """Build a bare SuffixConflictTracker with trivial default callables.

    No SpeculativeMergeWorker involved — exercises the tracker's own
    unit-testability contract (task δ=1988 design: git_ops passed directly,
    lane buffers / frozen-prefix accessors injected as callables so the
    tracker never needs a worker reference).
    """
    if lane_buffers is None:
        def lane_buffers():
            return {'high': collections.deque(), 'normal': collections.deque()}
    if frozen_prefix is None:
        def frozen_prefix():
            return ()
    if frozen_prefix_tip is None:
        def frozen_prefix_tip(main_sha):
            return main_sha
    return SuffixConflictTracker(
        git_ops,
        lane_buffers=lane_buffers,
        frozen_prefix=frozen_prefix,
        frozen_prefix_tip=frozen_prefix_tip,
    )


# ── step-1: module surface ─────────────────────────────────────────────────────


class TestModuleSurface:
    """orchestrator.suffix_graph exports SuffixConflictGraph,
    EMPTY_SUFFIX_CONFLICT_GRAPH, and SuffixConflictTracker.

    RED until step-2 GREEN creates the module.
    """

    def test_suffix_conflict_graph_importable(self):
        assert SuffixConflictGraph is not None

    def test_empty_suffix_conflict_graph_importable(self):
        assert isinstance(EMPTY_SUFFIX_CONFLICT_GRAPH, SuffixConflictGraph)

    def test_suffix_conflict_tracker_importable(self):
        assert SuffixConflictTracker is not None


# ── step-1: merge_queue shim re-export identity ────────────────────────────────


class TestShimReExportIdentity:
    """orchestrator.merge_queue re-exports the three names via a top-level
    shim; identity (not just equality) must hold so isinstance checks and
    monkeypatches keep working across both import paths.

    RED until step-2 GREEN adds the shim import and deletes the local defs.
    """

    def test_suffix_conflict_graph_identity(self):
        assert merge_queue.SuffixConflictGraph is suffix_graph.SuffixConflictGraph

    def test_empty_suffix_conflict_graph_identity(self):
        assert (
            merge_queue.EMPTY_SUFFIX_CONFLICT_GRAPH
            is suffix_graph.EMPTY_SUFFIX_CONFLICT_GRAPH
        )

    def test_suffix_conflict_tracker_identity(self):
        assert merge_queue.SuffixConflictTracker is suffix_graph.SuffixConflictTracker


# ── step-1: module logger name ─────────────────────────────────────────────────


class TestModuleLogger:
    """suffix_graph's module logger must share merge_queue's logger name so
    existing caplog-based log-message assertions keep matching post-extraction.

    RED until step-2 GREEN sets logger = logging.getLogger('orchestrator.merge_queue').
    """

    def test_logger_name_matches_merge_queue(self):
        assert suffix_graph.logger.name == 'orchestrator.merge_queue'


# ── step-1: tracker construction without a worker ──────────────────────────────


class TestTrackerConstructionWithoutWorker:
    """SuffixConflictTracker is constructable from git_ops + narrow accessor
    callables alone — no SpeculativeMergeWorker required.

    RED until step-2 GREEN adds the class with its state-initialising __init__.
    """

    def test_tracker_initial_graph_is_empty_sentinel(self, git_ops):
        tracker = _make_tracker(git_ops)
        assert tracker.graph is EMPTY_SUFFIX_CONFLICT_GRAPH

    def test_tracker_initial_signature_is_none(self, git_ops):
        tracker = _make_tracker(git_ops)
        assert tracker.signature is None

    def test_tracker_initial_last_known_main_sha_is_none(self, git_ops):
        tracker = _make_tracker(git_ops)
        assert tracker.last_known_main_sha is None

    def test_tracker_bounce_registry_is_merge_bounce_registry(self, git_ops):
        tracker = _make_tracker(git_ops)
        assert isinstance(tracker.bounce_registry, MergeBounceRegistry)
