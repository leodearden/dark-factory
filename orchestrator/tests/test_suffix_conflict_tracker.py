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
from orchestrator.merge_queue import (
    MERGE_BOUNCE_CAP,
    NEEDS_REBASE_REASON_PREFIX,
    GroupMergeRequest,
    MergeOutcome,
    MergeRequest,
    SpeculativeMergeWorker,
)
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
    unit-testability contract (task δ=1988 design, amended per reviewer
    robustness_stale_reference/consistency: git_ops, like lane_buffers and
    the frozen-prefix accessors, is injected as a re-reading callable —
    ``lambda: git_ops`` — rather than a direct GitOps reference, so the
    tracker never needs a worker reference and always observes the
    caller's live GitOps object).
    """
    if lane_buffers is None:
        def _default_lane_buffers():
            return {'high': collections.deque(), 'normal': collections.deque()}
        lane_buffers = _default_lane_buffers
    if frozen_prefix is None:
        def _default_frozen_prefix():
            return ()
        frozen_prefix = _default_frozen_prefix
    if frozen_prefix_tip is None:
        def _default_frozen_prefix_tip(main_sha):
            return main_sha
        frozen_prefix_tip = _default_frozen_prefix_tip
    return SuffixConflictTracker(
        lambda: git_ops,
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


# ── step-5: bounce_conflicting_suffix_items() without a worker ─────────────────


def _make_group_req(
    config: OrchestratorConfig,
    git_repo: Path,
    branch: str = 'tip-task',
) -> GroupMergeRequest:
    """Build a minimal GroupMergeRequest for unit tests (mirrors the helper in
    test_merge_queue_bounce.py).

    Must be called from within an async context (asyncio.get_running_loop()).
    """
    async def _dummy_status_check(tids: list) -> dict:
        return {}

    async def _dummy_mark_done(tid: str, sha: str) -> None:
        pass

    return GroupMergeRequest(
        task_id=branch,
        branch=branch,
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
        merge_first_enqueued_at=None,
        train_id='train-1',
        member_task_ids=[branch],
        tip_branch=branch,
        tip_task_id=branch,
        status_check=_dummy_status_check,
        mark_member_done=_dummy_mark_done,
    )


@pytest.mark.asyncio
class TestTrackerBounceWithoutWorker:
    """SuffixConflictTracker.bounce_conflicting_suffix_items() diverts suffix
    items that conflict with the frozen tip — no SpeculativeMergeWorker
    involved.

    RED until step-6 GREEN moves the _bounce_conflicting_suffix_items() body
    (verbatim) into SuffixConflictTracker.bounce_conflicting_suffix_items().
    """

    async def test_clean_rebase_requeues_item(self, git_ops, config, git_repo):
        """Clean rebase -> item stays in the lane buffer, req.result is left
        untouched, and the debounce signature is invalidated (set to None)."""
        req = _make_req('591', '591', config, git_repo)
        buffers = {'high': collections.deque(), 'normal': collections.deque([req])}
        main_sha = 'deadbeef0000'
        tracker = _make_tracker(
            git_ops, lane_buffers=lambda: buffers,
            frozen_prefix_tip=lambda s: 'FROZENTIP',
        )
        tracker.graph = SuffixConflictGraph(
            nodes=(req.request_id,),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset({req.request_id}),
        )
        tracker.signature = ((req.request_id,), main_sha)

        git_ops.rebase_onto_main = AsyncMock(return_value=True)

        await tracker.bounce_conflicting_suffix_items()

        git_ops.rebase_onto_main.assert_awaited_once()
        call_kwargs = git_ops.rebase_onto_main.call_args
        assert call_kwargs.kwargs.get('onto') == 'FROZENTIP' or (
            len(call_kwargs.args) >= 2 and call_kwargs.args[1] == 'FROZENTIP'
        ), f'Expected onto="FROZENTIP", got {call_kwargs}'

        assert req in buffers['normal'], (
            'Expected req to remain in the lane buffer after a clean rebase (re-queue).'
        )
        assert not req.result.done(), (
            'Expected req.result future to remain pending after a clean rebase.'
        )
        assert tracker.signature is None, (
            'Expected signature to be None after a bounce (debounce invalidated).'
        )

    async def test_real_conflict_escalates(self, git_ops, config, git_repo):
        """Real rebase conflict -> item removed from the lane buffer and
        req.result resolved 'blocked' with NEEDS_REBASE_REASON_PREFIX."""
        req = _make_req('591', '591', config, git_repo)
        buffers = {'high': collections.deque(), 'normal': collections.deque([req])}
        tracker = _make_tracker(
            git_ops, lane_buffers=lambda: buffers,
            frozen_prefix_tip=lambda s: 'FROZENTIP',
        )
        tracker.graph = SuffixConflictGraph(
            nodes=(req.request_id,),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset({req.request_id}),
        )
        tracker.signature = ((req.request_id,), 'deadbeef0000')

        git_ops.rebase_onto_main = AsyncMock(return_value=False)

        await tracker.bounce_conflicting_suffix_items()

        assert req not in buffers['normal'], (
            'Expected req to be removed from the lane buffer after a rebase conflict.'
        )
        assert req.result.done(), 'Expected req.result to be resolved after escalation.'
        outcome: MergeOutcome = req.result.result()
        assert outcome.status == 'blocked', (
            f'Expected status="blocked" but got {outcome.status!r}.'
        )
        assert outcome.reason is not None
        assert outcome.reason.startswith(NEEDS_REBASE_REASON_PREFIX), (
            f'Expected reason to start with NEEDS_REBASE_REASON_PREFIX but got '
            f'{outcome.reason!r}.'
        )
        assert tracker.signature is None

    async def test_cap_exceeded_escalates_without_rebase(self, git_ops, config, git_repo):
        """Bounce cap already exceeded -> escalate WITHOUT calling rebase_onto_main."""
        req = _make_req('591', '591', config, git_repo)
        buffers = {'high': collections.deque(), 'normal': collections.deque([req])}
        tracker = _make_tracker(
            git_ops, lane_buffers=lambda: buffers,
            frozen_prefix_tip=lambda s: 'FROZENTIP',
        )
        tracker.graph = SuffixConflictGraph(
            nodes=(req.request_id,),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset({req.request_id}),
        )
        tracker.signature = ((req.request_id,), 'deadbeef0000')

        # Pre-bump the registry to MERGE_BOUNCE_CAP so the NEXT record_bounce exceeds it.
        for _ in range(MERGE_BOUNCE_CAP):
            tracker.bounce_registry.record_bounce('591')
        assert tracker.bounce_registry.count('591') == MERGE_BOUNCE_CAP

        rebase_spy = AsyncMock(return_value=True)
        git_ops.rebase_onto_main = rebase_spy

        await tracker.bounce_conflicting_suffix_items()

        rebase_spy.assert_not_awaited()
        assert req not in buffers['normal'], (
            'Expected req to be removed from the lane buffer when the bounce cap is exceeded.'
        )
        assert req.result.done(), 'Expected req.result to be resolved after cap escalation.'
        outcome: MergeOutcome = req.result.result()
        assert outcome.status == 'blocked'
        assert outcome.reason is not None
        assert outcome.reason.startswith(NEEDS_REBASE_REASON_PREFIX), (
            f'Expected cap-exceeded reason to start with NEEDS_REBASE_REASON_PREFIX, '
            f'got {outcome.reason!r}.'
        )
        assert tracker.signature is None

    async def test_toctou_reuses_signature_main_sha(self, git_ops, config, git_repo):
        """With signature set, bounce must reuse signature[1] as main_sha and
        must NOT call get_main_sha() (TOCTOU protection)."""
        req = _make_req('591', '591', config, git_repo)
        buffers = {'high': collections.deque(), 'normal': collections.deque([req])}
        main_sha = 'deadbeef0000'
        seen_main_shas: list[str] = []

        def _tip(s):
            seen_main_shas.append(s)
            return 'FROZENTIP'

        tracker = _make_tracker(git_ops, lane_buffers=lambda: buffers, frozen_prefix_tip=_tip)
        tracker.graph = SuffixConflictGraph(
            nodes=(req.request_id,),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset({req.request_id}),
        )
        tracker.signature = ((req.request_id,), main_sha)

        git_ops.rebase_onto_main = AsyncMock(return_value=True)
        get_main_sha_spy = AsyncMock(return_value='SHOULD-NOT-BE-USED')
        git_ops.get_main_sha = get_main_sha_spy

        await tracker.bounce_conflicting_suffix_items()

        get_main_sha_spy.assert_not_awaited()
        assert seen_main_shas == [main_sha], (
            f'Expected frozen_prefix_tip to be called with signature[1]={main_sha!r} '
            f'(TOCTOU reuse), got {seen_main_shas!r}.'
        )

    async def test_group_merge_request_skipped(self, git_ops, config, git_repo):
        """A GroupMergeRequest in conflicts_with_main must NOT be bounced —
        trains keep their existing TRAIN_REBASE_CONFLICT path."""
        grp_req = _make_group_req(config, git_repo, branch='tip-task')
        buffers = {'high': collections.deque(), 'normal': collections.deque([grp_req])}
        tracker = _make_tracker(
            git_ops, lane_buffers=lambda: buffers,
            frozen_prefix_tip=lambda s: 'FROZENTIP',
        )
        tracker.graph = SuffixConflictGraph(
            nodes=(grp_req.request_id,),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset({grp_req.request_id}),
        )
        tracker.signature = ((grp_req.request_id,), 'deadbeef0000')

        await tracker.bounce_conflicting_suffix_items()

        assert grp_req in buffers['normal'], (
            'GroupMergeRequest was removed from the lane buffer — trains must '
            'NOT be bounced by bounce_conflicting_suffix_items().'
        )
        assert not grp_req.result.done(), (
            'GroupMergeRequest future was resolved — trains must NOT be '
            'escalated by bounce_conflicting_suffix_items().'
        )

    async def test_empty_conflicts_with_main_is_a_noop(self, git_ops, config, git_repo):
        """Empty conflicts_with_main -> return immediately; signature untouched."""
        tracker = _make_tracker(git_ops, frozen_prefix_tip=lambda s: 'FROZENTIP')
        sentinel_sig = (('rid-a',), 'abc123')
        tracker.signature = sentinel_sig
        tracker.graph = SuffixConflictGraph(
            nodes=(),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset(),
        )

        await tracker.bounce_conflicting_suffix_items()

        assert tracker.signature == sentinel_sig, (
            'signature was modified even though conflicts_with_main was empty. '
            f'Expected {sentinel_sig!r}, got {tracker.signature!r}.'
        )


# ── step-7: SpeculativeMergeWorker delegates to SuffixConflictTracker ──────────


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring)."""
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


class TestWorkerDelegatesToTracker:
    """SpeculativeMergeWorker delegates its suffix-conflict state to a
    SuffixConflictTracker instance (self._suffix_tracker) via thin property
    wrappers that preserve the worker's original attribute names.

    RED until step-8 GREEN rewires SpeculativeMergeWorker.__init__ to
    construct self._suffix_tracker and replaces the 4 inline attrs with
    delegating properties. (Sync-only class — see TestWorkerMethodDelegation
    for the async method-delegation + recompute-visibility cases; mixing
    sync defs into an @pytest.mark.asyncio class is a hard error here, see
    the "Sync def collected inside an @pytest.mark.asyncio class" filterwarnings entry.)
    """

    def test_worker_has_suffix_tracker(self, git_ops):
        worker = _make_worker(git_ops)
        assert isinstance(worker._suffix_tracker, SuffixConflictTracker)

    def test_read_delegation_graph(self, git_ops):
        worker = _make_worker(git_ops)
        assert worker._suffix_conflict_graph is worker._suffix_tracker.graph

    def test_write_delegation_graph(self, git_ops):
        worker = _make_worker(git_ops)
        g = SuffixConflictGraph(
            nodes=('rid-x',),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset(),
        )
        worker._suffix_conflict_graph = g
        assert worker._suffix_tracker.graph is g, (
            'Writing worker._suffix_conflict_graph must round-trip through '
            'self._suffix_tracker.graph.'
        )

    def test_write_delegation_signature(self, git_ops):
        worker = _make_worker(git_ops)
        sig = (('rid-x',), 'deadbeef')
        worker._suffix_conflict_signature = sig
        assert worker._suffix_tracker.signature == sig
        assert worker._suffix_conflict_signature == sig

    def test_write_delegation_last_known_main_sha(self, git_ops):
        worker = _make_worker(git_ops)
        worker._last_known_main_sha = 'cafef00d'
        assert worker._suffix_tracker.last_known_main_sha == 'cafef00d'
        assert worker._last_known_main_sha == 'cafef00d'

    def test_write_delegation_bounce_registry(self, git_ops):
        worker = _make_worker(git_ops)
        reg = MergeBounceRegistry()
        reg.record_bounce('591')
        worker._bounce_registry = reg
        assert worker._suffix_tracker.bounce_registry is reg
        assert worker._bounce_registry is reg

    def test_snapshot_integrity(self, git_ops):
        worker = _make_worker(git_ops)
        assert (
            worker.snapshot()['suffix_conflict_graph']
            == worker._suffix_tracker.graph.to_snapshot_dict()
        )

    def test_git_ops_reassignment_observed(self, git_ops):
        """amend (reviewer robustness_stale_reference): reassigning
        worker._git_ops after construction must be observed by the tracker
        on its next call — not the GitOps snapshotted at construction time.
        Mirrors the worker._git_ops reassignment pattern already used
        elsewhere (e.g. test_merge_queue.py, test_merge_queue_train_attribution.py).
        """
        worker = _make_worker(git_ops)
        new_git_ops = object()
        worker._git_ops = new_git_ops
        assert worker._suffix_tracker._git_ops() is new_git_ops

    def test_frozen_prefix_reassignment_observed(self, git_ops):
        """amend (reviewer consistency): reassigning worker.frozen_prefix /
        frozen_prefix_tip after construction must be observed by the
        tracker, removing the asymmetry with lane_buffers' existing
        re-reading-lambda semantics."""
        worker = _make_worker(git_ops)
        sentinel_prefix = ('sentinel-rid',)
        worker.frozen_prefix = lambda: sentinel_prefix
        worker.frozen_prefix_tip = lambda main_sha: 'SENTINEL-TIP'
        assert worker._suffix_tracker._frozen_prefix() == sentinel_prefix
        assert worker._suffix_tracker._frozen_prefix_tip('irrelevant') == 'SENTINEL-TIP'


@pytest.mark.asyncio
class TestWorkerMethodDelegation:
    """Async half of TestWorkerDelegatesToTracker: method delegation +
    recompute-visibility (split out because mixing sync defs into an
    @pytest.mark.asyncio class is a hard error in this repo).

    RED until step-8 GREEN rewires recompute_suffix_conflict_graph() /
    _bounce_conflicting_suffix_items() to thin `await self._suffix_tracker...`
    delegators.
    """

    async def test_method_delegation_bounce(self, git_ops):
        """worker._bounce_conflicting_suffix_items() must delegate (await)
        to self._suffix_tracker.bounce_conflicting_suffix_items()."""
        worker = _make_worker(git_ops)
        sentinel_mock = AsyncMock()
        worker._suffix_tracker.bounce_conflicting_suffix_items = sentinel_mock

        await worker._bounce_conflicting_suffix_items()

        sentinel_mock.assert_awaited_once()

    async def test_method_delegation_recompute_visible_both_paths(
        self, git_ops, config, git_repo,
    ):
        """Driving worker.recompute_suffix_conflict_graph() must make the
        resulting graph visible through BOTH the worker's legacy attribute
        name and the tracker's own field (same object)."""
        req = _make_req('591', 'branch-591', config, git_repo)
        worker = _make_worker(git_ops)
        worker._lane_buffers['normal'].append(req)

        await worker.recompute_suffix_conflict_graph()

        assert worker._suffix_conflict_graph is worker._suffix_tracker.graph, (
            'worker._suffix_conflict_graph and worker._suffix_tracker.graph '
            'diverged after recompute_suffix_conflict_graph().'
        )
