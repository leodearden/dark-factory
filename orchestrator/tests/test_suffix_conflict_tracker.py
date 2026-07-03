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
from unittest.mock import AsyncMock

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


async def _create_branch_editing(
    repo: Path,
    branch_name: str,
    filename: str,
    content: str,
    base_branch: str = 'main',
) -> str:
    """Create a branch that edits filename with content; return the branch SHA."""
    await _run(['git', 'checkout', '-b', branch_name], cwd=repo)
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Edit {filename} in {branch_name}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    await _run(['git', 'checkout', base_branch], cwd=repo)
    return sha.strip()


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


# ── step-3: recompute() without a worker ────────────────────────────────────────


@pytest.mark.asyncio
class TestTrackerRecomputeWithoutWorker:
    """SuffixConflictTracker.recompute() builds the conflict graph directly
    from a git_ops + lane_buffers callable — no SpeculativeMergeWorker
    involved.

    RED until step-4 GREEN moves the recompute_suffix_conflict_graph() body
    (verbatim) into SuffixConflictTracker.recompute().
    """

    async def test_recompute_builds_edges_and_conflicts_with_main(
        self, git_ops, config, git_repo,
    ):
        """footprint_edges / textual_edges (⊆ footprint) / conflicts_with_main
        all populate correctly, and nodes preserve pick order.
        """
        # A and B both edit shared.txt's line1 -> footprint overlap AND a
        # genuine 3-way textual conflict (textual_edge ⊆ footprint_edge).
        await _run(['git', 'checkout', '-b', 'task/branch-a'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('EDITED-BY-A\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'branch-a edits line1'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        await _run(['git', 'checkout', '-b', 'task/branch-b'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('EDITED-BY-B\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'branch-b edits line1'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        # C forks from the INITIAL main tip and adds conflict.txt; main then
        # advances by adding the SAME path with different content, so C's
        # merge-tree probe against (new) main fails -> conflicts_with_main.
        # conflict.txt is untouched by A/B, so C has no footprint edge with them.
        await _run(['git', 'checkout', '-b', 'task/branch-c'], cwd=git_repo)
        (git_repo / 'conflict.txt').write_text('C-EDIT\n')
        await _run(['git', 'add', 'conflict.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'branch-c adds conflict.txt'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        (git_repo / 'conflict.txt').write_text('MAIN-EDIT\n')
        await _run(['git', 'add', 'conflict.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'main advances conflict.txt'], cwd=git_repo)

        req_a = _make_req('task-a', 'branch-a', config, git_repo)
        req_b = _make_req('task-b', 'branch-b', config, git_repo)
        req_c = _make_req('task-c', 'branch-c', config, git_repo)
        buffers = {
            'high': collections.deque(),
            'normal': collections.deque([req_a, req_b, req_c]),
        }
        tracker = _make_tracker(git_ops, lane_buffers=lambda: buffers)

        await tracker.recompute()

        pair_ab = frozenset({req_a.request_id, req_b.request_id})
        assert pair_ab in tracker.graph.footprint_edges
        assert pair_ab in tracker.graph.textual_edges
        assert tracker.graph.textual_edges <= tracker.graph.footprint_edges
        assert req_c.request_id in tracker.graph.conflicts_with_main
        # nodes are in pick order: high (empty) then normal, FIFO.
        assert tracker.graph.nodes == (
            req_a.request_id, req_b.request_id, req_c.request_id,
        )

    async def test_recompute_sets_signature_and_debounces(
        self, git_ops, config, git_repo,
    ):
        """signature == (ordered_rids, main_sha); a second call with an
        unchanged suffix short-circuits (same graph object returned)."""
        await _create_branch_editing(git_repo, 'task/branch-deb', 'README.md', 'debounce\n')
        req = _make_req('task-deb', 'branch-deb', config, git_repo)
        buffers = {'high': collections.deque(), 'normal': collections.deque([req])}
        tracker = _make_tracker(git_ops, lane_buffers=lambda: buffers)

        await tracker.recompute()
        assert tracker.signature is not None
        ordered_rids, main_sha = tracker.signature
        assert ordered_rids == (req.request_id,)
        assert isinstance(main_sha, str) and main_sha

        first_graph = tracker.graph
        await tracker.recompute()
        assert tracker.graph is first_graph, (
            'Expected the debounce to short-circuit (identical graph object) '
            'when the suffix + main_sha are unchanged.'
        )

    async def test_recompute_empty_suffix_yields_sentinel(self, git_ops, config, git_repo):
        """Empty suffix -> graph is the EMPTY_SUFFIX_CONFLICT_GRAPH sentinel."""
        buffers = {'high': collections.deque(), 'normal': collections.deque()}
        tracker = _make_tracker(git_ops, lane_buffers=lambda: buffers)

        await tracker.recompute()

        assert tracker.graph is EMPTY_SUFFIX_CONFLICT_GRAPH

    async def test_recompute_fail_open_on_get_main_sha_error(
        self, git_ops, config, git_repo,
    ):
        """get_main_sha() raising -> recompute() returns without raising and
        leaves the graph untouched."""
        req = _make_req('task-fo', 'branch-that-does-not-exist', config, git_repo)
        buffers = {'high': collections.deque(), 'normal': collections.deque([req])}
        tracker = _make_tracker(git_ops, lane_buffers=lambda: buffers)
        original_graph = tracker.graph

        git_ops.get_main_sha = AsyncMock(side_effect=RuntimeError('simulated failure'))

        await tracker.recompute()  # must NOT raise

        assert tracker.graph is original_graph, (
            'Expected the graph to be left untouched (fail-open) when '
            'get_main_sha() raises.'
        )
