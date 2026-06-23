"""Tests for the SuffixConflictGraph (task δ=1889): conflict-graph over the unfrozen suffix.

Covers:
  step-01 RED  — SuffixConflictGraph dataclass fields, helpers, to_snapshot_dict()
  step-03 RED  — snapshot() emits suffix_conflict_graph key; backward-compat
  step-05 RED  — recompute_suffix_conflict_graph() builds footprint_edges from real branches
  step-07 RED  — textual_edges subset of footprint_edges; same-line edits → textual conflict
  step-09 RED  — conflicts_with_main marker per item (user-signal)
  step-11 RED  — debounce/incremental: no re-probe when signature unchanged
  step-13 RED  — _acquire_next_request hook fires recompute
  step-15 RED  — fail-open: probe error → conservative edge; missing ref → skipped cleanly
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import (
    EMPTY_SUFFIX_CONFLICT_GRAPH,
    MergeRequest,
    SpeculativeMergeWorker,
    SuffixConflictGraph,
)

# ── fixtures (mirrors test_merge_queue_finalize_head_visibility.py) ────────────


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


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring)."""
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


# ── step-01: SuffixConflictGraph dataclass ────────────────────────────────────


class TestSuffixConflictGraphDataclass:
    """SuffixConflictGraph is an immutable dataclass with well-typed fields.

    RED until step-02 GREEN adds the dataclass to merge_queue.py.
    """

    def test_empty_graph_module_constant(self):
        """EMPTY_SUFFIX_CONFLICT_GRAPH is a well-formed empty graph."""
        g = EMPTY_SUFFIX_CONFLICT_GRAPH
        assert isinstance(g, SuffixConflictGraph)
        assert g.nodes == ()
        assert g.textual_edges == frozenset()
        assert g.footprint_edges == frozenset()
        assert g.conflicts_with_main == frozenset()

    def test_nodes_are_tuple(self):
        """nodes is a tuple (immutable, ordered)."""
        g = SuffixConflictGraph(
            nodes=('mr-aaa', 'mr-bbb'),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset(),
        )
        assert g.nodes == ('mr-aaa', 'mr-bbb')
        assert isinstance(g.nodes, tuple)

    def test_edges_are_frozenset_of_frozensets(self):
        """textual_edges and footprint_edges are frozensets of 2-element frozensets."""
        edge = frozenset({'mr-aaa', 'mr-bbb'})
        g = SuffixConflictGraph(
            nodes=('mr-aaa', 'mr-bbb'),
            textual_edges=frozenset({edge}),
            footprint_edges=frozenset({edge}),
            conflicts_with_main=frozenset(),
        )
        assert edge in g.textual_edges
        assert edge in g.footprint_edges

    def test_conflicts_with_main_is_frozenset(self):
        """conflicts_with_main is a frozenset of request_ids."""
        g = SuffixConflictGraph(
            nodes=('mr-aaa', 'mr-bbb'),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset({'mr-aaa'}),
        )
        assert 'mr-aaa' in g.conflicts_with_main
        assert 'mr-bbb' not in g.conflicts_with_main

    def test_graph_is_immutable(self):
        """SuffixConflictGraph is frozen (immutable dataclass)."""
        g = EMPTY_SUFFIX_CONFLICT_GRAPH
        with pytest.raises((AttributeError, TypeError)):
            g.nodes = ('mr-x',)  # type: ignore[misc]

    def test_textual_neighbors_empty(self):
        """textual_neighbors returns empty set when no textual edges."""
        g = SuffixConflictGraph(
            nodes=('mr-aaa', 'mr-bbb'),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset(),
        )
        assert g.textual_neighbors('mr-aaa') == frozenset()

    def test_footprint_neighbors_empty(self):
        """footprint_neighbors returns empty set when no footprint edges."""
        g = SuffixConflictGraph(
            nodes=('mr-aaa', 'mr-bbb'),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset(),
        )
        assert g.footprint_neighbors('mr-aaa') == frozenset()

    def test_textual_neighbors_returns_connected_rids(self):
        """textual_neighbors returns the set of RIDs connected by textual edges."""
        edge = frozenset({'mr-aaa', 'mr-bbb'})
        g = SuffixConflictGraph(
            nodes=('mr-aaa', 'mr-bbb', 'mr-ccc'),
            textual_edges=frozenset({edge}),
            footprint_edges=frozenset({edge}),
            conflicts_with_main=frozenset(),
        )
        assert g.textual_neighbors('mr-aaa') == frozenset({'mr-bbb'})
        assert g.textual_neighbors('mr-bbb') == frozenset({'mr-aaa'})
        # mr-ccc is not connected
        assert g.textual_neighbors('mr-ccc') == frozenset()

    def test_footprint_neighbors_returns_connected_rids(self):
        """footprint_neighbors returns the set of RIDs connected by footprint edges."""
        foot_edge_ab = frozenset({'mr-aaa', 'mr-bbb'})
        foot_edge_ac = frozenset({'mr-aaa', 'mr-ccc'})
        g = SuffixConflictGraph(
            nodes=('mr-aaa', 'mr-bbb', 'mr-ccc'),
            textual_edges=frozenset(),
            footprint_edges=frozenset({foot_edge_ab, foot_edge_ac}),
            conflicts_with_main=frozenset(),
        )
        assert g.footprint_neighbors('mr-aaa') == frozenset({'mr-bbb', 'mr-ccc'})
        assert g.footprint_neighbors('mr-bbb') == frozenset({'mr-aaa'})
        assert g.footprint_neighbors('mr-ccc') == frozenset({'mr-aaa'})

    def test_to_snapshot_dict_empty(self):
        """to_snapshot_dict() on empty graph returns stable JSON-safe empty dict."""
        d = EMPTY_SUFFIX_CONFLICT_GRAPH.to_snapshot_dict()
        assert d == {
            'nodes': [],
            'textual_edges': [],
            'footprint_edges': [],
            'conflicts_with_main': [],
        }

    def test_to_snapshot_dict_nodes_as_list(self):
        """to_snapshot_dict() serialises nodes as a list (preserves pick order)."""
        g = SuffixConflictGraph(
            nodes=('mr-zzz', 'mr-aaa'),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset(),
        )
        d = g.to_snapshot_dict()
        assert d['nodes'] == ['mr-zzz', 'mr-aaa']

    def test_to_snapshot_dict_edges_as_sorted_lists(self):
        """to_snapshot_dict() serialises edges as sorted list-of-lists."""
        edge = frozenset({'mr-bbb', 'mr-aaa'})
        g = SuffixConflictGraph(
            nodes=('mr-aaa', 'mr-bbb'),
            textual_edges=frozenset({edge}),
            footprint_edges=frozenset({edge}),
            conflicts_with_main=frozenset(),
        )
        d = g.to_snapshot_dict()
        # Each edge is a sorted list; the outer list is also sorted
        assert d['textual_edges'] == [['mr-aaa', 'mr-bbb']]
        assert d['footprint_edges'] == [['mr-aaa', 'mr-bbb']]

    def test_to_snapshot_dict_conflicts_with_main_as_sorted_list(self):
        """to_snapshot_dict() serialises conflicts_with_main as a sorted list."""
        g = SuffixConflictGraph(
            nodes=('mr-bbb', 'mr-aaa'),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset({'mr-bbb', 'mr-aaa'}),
        )
        d = g.to_snapshot_dict()
        assert d['conflicts_with_main'] == ['mr-aaa', 'mr-bbb']  # sorted

    def test_to_snapshot_dict_is_json_serialisable(self):
        """to_snapshot_dict() output is JSON-serialisable (no frozensets/tuples)."""
        import json
        edge = frozenset({'mr-aaa', 'mr-bbb'})
        g = SuffixConflictGraph(
            nodes=('mr-aaa', 'mr-bbb'),
            textual_edges=frozenset({edge}),
            footprint_edges=frozenset({edge}),
            conflicts_with_main=frozenset({'mr-aaa'}),
        )
        json.dumps(g.to_snapshot_dict())  # must not raise


# ── step-03: snapshot() emits suffix_conflict_graph key ───────────────────────


@pytest.mark.asyncio
class TestSnapshotSuffixConflictGraphKey:
    """snapshot() must include a 'suffix_conflict_graph' key.

    RED until step-04 GREEN adds the key to snapshot() and initialises
    self._suffix_conflict_graph in __init__.
    """

    async def test_snapshot_has_suffix_conflict_graph_key(self, git_ops, config, git_repo):
        """snapshot() includes 'suffix_conflict_graph' (empty by default)."""
        worker = _make_worker(git_ops)
        snap = worker.snapshot()
        assert 'suffix_conflict_graph' in snap

    async def test_snapshot_suffix_conflict_graph_default_is_empty(
        self, git_ops, config, git_repo
    ):
        """Default suffix_conflict_graph in snapshot() equals empty dict."""
        worker = _make_worker(git_ops)
        snap = worker.snapshot()
        assert snap['suffix_conflict_graph'] == EMPTY_SUFFIX_CONFLICT_GRAPH.to_snapshot_dict()

    async def test_snapshot_reflects_stored_graph(self, git_ops, config, git_repo):
        """snapshot() reflects self._suffix_conflict_graph when set manually."""
        worker = _make_worker(git_ops)
        edge = frozenset({'mr-aaa', 'mr-bbb'})
        custom_graph = SuffixConflictGraph(
            nodes=('mr-aaa', 'mr-bbb'),
            textual_edges=frozenset({edge}),
            footprint_edges=frozenset({edge}),
            conflicts_with_main=frozenset({'mr-aaa'}),
        )
        worker._suffix_conflict_graph = custom_graph
        snap = worker.snapshot()
        assert snap['suffix_conflict_graph'] == custom_graph.to_snapshot_dict()

    async def test_snapshot_backward_compat_keys_unchanged(self, git_ops, config, git_repo):
        """Pre-existing snapshot keys (depth, entries, head_of_line) are unchanged."""
        worker = _make_worker(git_ops)
        snap = worker.snapshot()
        # These keys must still be present (backward compatibility)
        assert 'depth' in snap
        assert 'entries' in snap
        assert 'head_of_line' in snap


# ── step-05: recompute builds footprint_edges (real git branches) ─────────────


@pytest.mark.asyncio
class TestRecomputeFootprintEdges:
    """recompute_suffix_conflict_graph() builds footprint_edges via path-set intersection.

    RED until step-06 GREEN implements the method.
    """

    async def _create_branch_editing(
        self,
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

    async def test_footprint_edges_shared_file(self, git_ops, config, git_repo):
        """Two branches editing the same file → footprint_edges contains that pair."""
        # Create branches A and B editing shared.txt, C editing disjoint.txt
        await self._create_branch_editing(git_repo, 'task/branch-a', 'shared.txt', 'edit-a\n')
        await self._create_branch_editing(git_repo, 'task/branch-b', 'shared.txt', 'edit-b\n')
        await self._create_branch_editing(git_repo, 'task/branch-c', 'disjoint.txt', 'edit-c\n')

        worker = _make_worker(git_ops)
        req_a = _make_req('task-a', 'branch-a', config, git_repo)
        req_b = _make_req('task-b', 'branch-b', config, git_repo)
        req_c = _make_req('task-c', 'branch-c', config, git_repo)
        worker._lane_buffers['normal'].append(req_a)
        worker._lane_buffers['normal'].append(req_b)
        worker._lane_buffers['normal'].append(req_c)

        await worker.recompute_suffix_conflict_graph()

        g = worker._suffix_conflict_graph
        # A and B share shared.txt — footprint overlap
        pair_ab = frozenset({req_a.request_id, req_b.request_id})
        assert pair_ab in g.footprint_edges
        # A and C, B and C do NOT share files
        pair_ac = frozenset({req_a.request_id, req_c.request_id})
        pair_bc = frozenset({req_b.request_id, req_c.request_id})
        assert pair_ac not in g.footprint_edges
        assert pair_bc not in g.footprint_edges

    async def test_empty_suffix_yields_empty_graph(self, git_ops, config, git_repo):
        """Empty suffix → EMPTY_SUFFIX_CONFLICT_GRAPH stored."""
        worker = _make_worker(git_ops)
        # No items in lane buffers
        await worker.recompute_suffix_conflict_graph()
        assert worker._suffix_conflict_graph == EMPTY_SUFFIX_CONFLICT_GRAPH

    async def test_nodes_in_pick_order_high_before_normal(
        self, git_ops, config, git_repo
    ):
        """nodes are ordered: high lane before normal, FIFO within each lane."""
        await self._create_branch_editing(git_repo, 'task/branch-hi', 'disjoint.txt', 'hi\n')
        await self._create_branch_editing(git_repo, 'task/branch-n1', 'README.md', 'n1\n')
        await self._create_branch_editing(git_repo, 'task/branch-n2', 'README.md', 'n2\n')

        worker = _make_worker(git_ops)
        req_hi = MergeRequest(
            task_id='task-hi',
            branch='branch-hi',
            worktree=git_repo,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=asyncio.get_running_loop().create_future(),
            lane='high',
        )
        req_n1 = _make_req('task-n1', 'branch-n1', config, git_repo)
        req_n2 = _make_req('task-n2', 'branch-n2', config, git_repo)

        # high lane first, then normal (FIFO)
        worker._lane_buffers['high'].append(req_hi)
        worker._lane_buffers['normal'].append(req_n1)
        worker._lane_buffers['normal'].append(req_n2)

        await worker.recompute_suffix_conflict_graph()

        g = worker._suffix_conflict_graph
        # high must precede normal
        hi_idx = g.nodes.index(req_hi.request_id)
        n1_idx = g.nodes.index(req_n1.request_id)
        n2_idx = g.nodes.index(req_n2.request_id)
        assert hi_idx < n1_idx < n2_idx


# ── step-07: textual_edges subset of footprint_edges ─────────────────────────


@pytest.mark.asyncio
class TestTextualEdgesSubsetOfFootprint:
    """textual_edges ⊆ footprint_edges; same-line edits → textual conflict.

    RED until step-08 GREEN extends recompute to probe textual conflicts.
    """

    async def test_same_line_edit_produces_textual_edge(
        self, git_ops, config, git_repo
    ):
        """Branches editing the SAME line of the same file → textual_edge."""
        # Both branches edit line1 of shared.txt (same line → merge conflict)
        await _run(['git', 'checkout', '-b', 'task/branch-x'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('EDITED-BY-X\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'branch-x edits line1'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        await _run(['git', 'checkout', '-b', 'task/branch-y'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('EDITED-BY-Y\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'branch-y edits line1'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        worker = _make_worker(git_ops)
        req_x = _make_req('task-x', 'branch-x', config, git_repo)
        req_y = _make_req('task-y', 'branch-y', config, git_repo)
        worker._lane_buffers['normal'].append(req_x)
        worker._lane_buffers['normal'].append(req_y)

        await worker.recompute_suffix_conflict_graph()
        g = worker._suffix_conflict_graph

        pair_xy = frozenset({req_x.request_id, req_y.request_id})
        assert pair_xy in g.textual_edges

    async def test_different_line_edit_no_textual_edge(
        self, git_ops, config, git_repo
    ):
        """Branches editing DIFFERENT lines of the same file → footprint overlap but no textual edge.

        Uses a 20-line file so the edited lines (line 1 and line 15) are >6 lines
        apart, preventing git's 3-line context window from creating overlapping
        diff hunks (adjacent-line edits in a small file DO conflict because the
        context windows cover the same region of the file).
        """
        # Build a large file so well-separated edits don't overlap in context
        large_content = '\n'.join(f'original-line-{i}' for i in range(1, 21)) + '\n'
        c_content = large_content.replace('original-line-1\n', 'EDITED-BY-C\n', 1)
        d_content = large_content.replace('original-line-15\n', 'EDITED-BY-D\n', 1)

        # Populate initial large file on main
        (git_repo / 'shared.txt').write_text(large_content)
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'Add large shared.txt'], cwd=git_repo)

        await _run(['git', 'checkout', '-b', 'task/branch-c'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text(c_content)
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'branch-c edits line 1 of 20'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        await _run(['git', 'checkout', '-b', 'task/branch-d'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text(d_content)
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'branch-d edits line 15 of 20'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        worker = _make_worker(git_ops)
        req_c = _make_req('task-c', 'branch-c', config, git_repo)
        req_d = _make_req('task-d', 'branch-d', config, git_repo)
        worker._lane_buffers['normal'].append(req_c)
        worker._lane_buffers['normal'].append(req_d)

        await worker.recompute_suffix_conflict_graph()
        g = worker._suffix_conflict_graph

        pair_cd = frozenset({req_c.request_id, req_d.request_id})
        # footprint overlap (same file)
        assert pair_cd in g.footprint_edges
        # but NOT a textual conflict (different lines → clean merge)
        assert pair_cd not in g.textual_edges

    async def test_textual_edges_subset_of_footprint_edges(
        self, git_ops, config, git_repo
    ):
        """textual_edges ⊆ footprint_edges (contract invariant: textual ⇒ footprint)."""
        # Create X and Y conflicting (same line) + C and D overlapping (diff lines)
        await _run(['git', 'checkout', '-b', 'task/br-inv-x'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('X-edit\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'inv-x'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        await _run(['git', 'checkout', '-b', 'task/br-inv-y'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('Y-edit\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'inv-y'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        worker = _make_worker(git_ops)
        req_x = _make_req('t-inv-x', 'br-inv-x', config, git_repo)
        req_y = _make_req('t-inv-y', 'br-inv-y', config, git_repo)
        worker._lane_buffers['normal'].append(req_x)
        worker._lane_buffers['normal'].append(req_y)

        await worker.recompute_suffix_conflict_graph()
        g = worker._suffix_conflict_graph

        # Every textual edge must also be a footprint edge
        assert g.textual_edges <= g.footprint_edges


# ── step-09: conflicts_with_main (δ user-signal) ─────────────────────────────


@pytest.mark.asyncio
class TestConflictsWithMain:
    """conflicts_with_main marks items that conflict with current main.

    RED until step-10 GREEN extends recompute to probe items vs main.
    """

    async def test_item_conflicting_with_main_is_marked(
        self, git_ops, config, git_repo
    ):
        """Branch that conflicts with main → in conflicts_with_main."""
        # Branch X is forked from the INITIAL main tip and edits line1 of
        # shared.txt.  Main then advances and edits the SAME line differently.
        # merge_tree_conflicts(main_sha, head_x) finds the common ancestor is
        # the initial commit, so both sides modified line1 divergently → CONFLICT.
        # (Branch must be forked BEFORE main advances; a descendant of main
        # fast-forwards cleanly and would never appear in conflicts_with_main.)
        await _run(['git', 'checkout', '-b', 'task/conflict-x'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('X-CONFLICT\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'conflict-x edits line1'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        # Advance main AFTER branching so X and main diverge from a common ancestor
        (git_repo / 'shared.txt').write_text('MAIN-EDIT\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'Main advances: edits line1'], cwd=git_repo)

        # Branch Y edits a different file → clean merge with main
        await _run(['git', 'checkout', '-b', 'task/clean-y'], cwd=git_repo)
        (git_repo / 'disjoint.txt').write_text('Y-edit\nbbb\nccc\n')
        await _run(['git', 'add', 'disjoint.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'clean-y edits disjoint'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        worker = _make_worker(git_ops)
        req_x = _make_req('task-cx', 'conflict-x', config, git_repo)
        req_y = _make_req('task-cy', 'clean-y', config, git_repo)
        worker._lane_buffers['normal'].append(req_x)
        worker._lane_buffers['normal'].append(req_y)

        await worker.recompute_suffix_conflict_graph()
        g = worker._suffix_conflict_graph

        assert req_x.request_id in g.conflicts_with_main
        assert req_y.request_id not in g.conflicts_with_main

    async def test_conflicts_with_main_in_snapshot(
        self, git_ops, config, git_repo
    ):
        """snapshot()['suffix_conflict_graph']['conflicts_with_main'] lists conflicting items."""
        # cx2 is forked from the INITIAL main tip and edits shared.txt line1.
        # Main then advances on the same line → they diverge → conflict with main.
        await _run(['git', 'checkout', '-b', 'task/cx2'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('X2-CONFLICT\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'cx2 conflicts'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        # Advance main AFTER branching
        (git_repo / 'shared.txt').write_text('MAIN-EDIT2\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'Main advances again'], cwd=git_repo)

        worker = _make_worker(git_ops)
        req_x = _make_req('task-cx2', 'cx2', config, git_repo)
        worker._lane_buffers['normal'].append(req_x)

        await worker.recompute_suffix_conflict_graph()
        snap = worker.snapshot()
        assert req_x.request_id in snap['suffix_conflict_graph']['conflicts_with_main']


# ── step-11: debounce / incremental recompute ─────────────────────────────────


@pytest.mark.asyncio
class TestRecomputeDebounce:
    """Recompute is a no-op when signature (suffix request_ids, main_sha) unchanged.

    RED until step-12 GREEN adds the signature short-circuit.
    """

    async def test_no_recompute_when_signature_unchanged(
        self, git_ops, config, git_repo
    ):
        """Second recompute with same suffix+main → no additional probe calls."""
        await _run(['git', 'checkout', '-b', 'task/branch-deb'], cwd=git_repo)
        (git_repo / 'README.md').write_text('debounce-test\n')
        await _run(['git', 'add', 'README.md'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'branch-deb'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        worker = _make_worker(git_ops)
        req = _make_req('task-deb', 'branch-deb', config, git_repo)
        worker._lane_buffers['normal'].append(req)

        call_count = 0
        original_probe = git_ops.merge_tree_conflicts

        async def counting_probe(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return await original_probe(*args, **kwargs)

        git_ops.merge_tree_conflicts = counting_probe

        await worker.recompute_suffix_conflict_graph()
        first_count = call_count

        # Same suffix, same main → no-op
        await worker.recompute_suffix_conflict_graph()
        assert call_count == first_count, (
            f"Debounce failed: call_count went from {first_count} to {call_count}"
        )

    async def test_recompute_fires_on_new_request(
        self, git_ops, config, git_repo
    ):
        """Adding a new request (new request_id) forces a recompute."""
        await _run(['git', 'checkout', '-b', 'task/branch-deb2'], cwd=git_repo)
        (git_repo / 'README.md').write_text('debounce2\n')
        await _run(['git', 'add', 'README.md'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'branch-deb2'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        await _run(['git', 'checkout', '-b', 'task/branch-deb3'], cwd=git_repo)
        (git_repo / 'disjoint.txt').write_text('debounce3\n')
        await _run(['git', 'add', 'disjoint.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'branch-deb3'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        worker = _make_worker(git_ops)
        req1 = _make_req('task-deb2', 'branch-deb2', config, git_repo)
        worker._lane_buffers['normal'].append(req1)

        call_count = 0
        original_probe = git_ops.merge_tree_conflicts

        async def counting_probe(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return await original_probe(*args, **kwargs)

        git_ops.merge_tree_conflicts = counting_probe

        await worker.recompute_suffix_conflict_graph()
        first_count = call_count

        # Add a second request → new suffix → must recompute
        req2 = _make_req('task-deb3', 'branch-deb3', config, git_repo)
        worker._lane_buffers['normal'].append(req2)

        await worker.recompute_suffix_conflict_graph()
        assert call_count > first_count, (
            "Expected recompute after new request but probe count unchanged"
        )


# ── step-13: _acquire_next_request hook ───────────────────────────────────────


@pytest.mark.asyncio
class TestAcquireNextRequestHook:
    """_acquire_next_request fires recompute_suffix_conflict_graph after drain.

    RED until step-14 GREEN adds the hook.
    """

    async def test_acquire_fires_recompute(self, git_ops, config, git_repo):
        """_acquire_next_request → recompute is called, graph contains the request_id."""
        await _run(['git', 'checkout', '-b', 'task/hook-branch'], cwd=git_repo)
        (git_repo / 'README.md').write_text('hook-test\n')
        await _run(['git', 'add', 'README.md'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'hook-branch'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        worker = _make_worker(git_ops)
        queue = worker._queue
        req = _make_req('task-hook', 'hook-branch', config, git_repo)
        await queue.put(req)
        # Put the shutdown sentinel after the real item
        await queue.put(None)  # type: ignore[arg-type]

        result = await worker._acquire_next_request()
        assert result is req
        # After returning, the graph must have the request_id as a node
        g = worker._suffix_conflict_graph
        assert req.request_id in g.nodes


# ── step-15: fail-open error handling ─────────────────────────────────────────


@pytest.mark.asyncio
class TestRecomputeFailOpen:
    """Probe errors → conservative (treat as conflict); missing ref → skipped.

    RED until step-16 GREEN wraps probes in fail-open try/except.
    """

    async def test_probe_error_conservative_textual_edge(
        self, git_ops, config, git_repo
    ):
        """merge_tree_conflicts raising RuntimeError → pair marked as textual edge."""
        await _run(['git', 'checkout', '-b', 'task/fo-a'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('fo-a\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'fo-a'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        await _run(['git', 'checkout', '-b', 'task/fo-b'], cwd=git_repo)
        (git_repo / 'shared.txt').write_text('fo-b\nline2\nline3\n')
        await _run(['git', 'add', 'shared.txt'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'fo-b'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        worker = _make_worker(git_ops)
        req_a = _make_req('task-fo-a', 'fo-a', config, git_repo)
        req_b = _make_req('task-fo-b', 'fo-b', config, git_repo)
        worker._lane_buffers['normal'].append(req_a)
        worker._lane_buffers['normal'].append(req_b)

        async def boom(*args, **kwargs):
            raise RuntimeError('simulated probe failure')

        git_ops.merge_tree_conflicts = boom

        # Must NOT raise
        await worker.recompute_suffix_conflict_graph()
        g = worker._suffix_conflict_graph

        # Pair should be conservatively marked as textual edge (fail-open)
        pair_ab = frozenset({req_a.request_id, req_b.request_id})
        assert pair_ab in g.textual_edges

    async def test_probe_error_conservative_conflicts_with_main(
        self, git_ops, config, git_repo
    ):
        """merge_tree_conflicts raising RuntimeError → item in conflicts_with_main."""
        await _run(['git', 'checkout', '-b', 'task/fo-c'], cwd=git_repo)
        (git_repo / 'README.md').write_text('fo-c\n')
        await _run(['git', 'add', 'README.md'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'fo-c'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        worker = _make_worker(git_ops)
        req_c = _make_req('task-fo-c', 'fo-c', config, git_repo)
        worker._lane_buffers['normal'].append(req_c)

        async def boom(*args, **kwargs):
            raise RuntimeError('simulated main-probe failure')

        git_ops.merge_tree_conflicts = boom

        await worker.recompute_suffix_conflict_graph()
        g = worker._suffix_conflict_graph
        # Conservatively in conflicts_with_main
        assert req_c.request_id in g.conflicts_with_main

    async def test_missing_ref_skipped_without_raising(
        self, git_ops, config, git_repo
    ):
        """Item whose branch ref is gone → skipped without raising."""
        worker = _make_worker(git_ops)
        # Create a req for a branch that doesn't exist (deleted or never pushed)
        req_gone = _make_req('task-gone', 'branch-that-does-not-exist', config, git_repo)
        req_gone_2 = _make_req('task-gone-2', 'branch-that-does-not-exist-2', config, git_repo)
        worker._lane_buffers['normal'].append(req_gone)
        worker._lane_buffers['normal'].append(req_gone_2)

        # Must NOT raise
        await worker.recompute_suffix_conflict_graph()
        # Graph should be produced (empty or with empty edges)
        g = worker._suffix_conflict_graph
        assert isinstance(g, SuffixConflictGraph)


# ── step-17: regression test for masked transient-error bug ──────────────────


@pytest.mark.asyncio
class TestRecomputeTransientErrorGatesSignature:
    """A get_changed_files failure (step-4) must leave the debounce signature
    un-stored so the next tick re-probes instead of short-circuiting on a
    stale/degraded graph.

    Bug: merge_queue.py step-5 previously re-initialised ``_any_probe_error =
    False`` (line 5820) AFTER step-4 set it to True on a get_changed_files
    failure, masking the error and allowing the signature to be cached.

    RED until step-18 GREEN removes the re-initialisation.
    """

    async def test_signature_not_cached_on_get_changed_files_error(
        self, git_ops, config, git_repo
    ):
        """get_changed_files failure → signature stays None → second call re-probes.

        Isolation: ALL merge_tree_conflicts calls SUCCEED (counting spy wraps the
        real method).  The ONLY error source is get_changed_files (step-4 footprint
        resolve).  This verifies that the error flag set in step-4 is not silently
        reset at the start of step-5.
        """
        # Create one real branch so that:
        #   - resolve_branch_sha succeeds (head is valid)
        #   - get_changed_files can be made to fail independently
        #   - merge_tree_conflicts(main_sha, head) still runs (vs-main probe)
        await _run(['git', 'checkout', '-b', 'task/transient-err'], cwd=git_repo)
        (git_repo / 'README.md').write_text('transient-error-test\n')
        await _run(['git', 'add', 'README.md'], cwd=git_repo)
        await _run(['git', 'commit', '-m', 'transient-err branch'], cwd=git_repo)
        await _run(['git', 'checkout', 'main'], cwd=git_repo)

        worker = _make_worker(git_ops)
        req = _make_req('task-transient', 'transient-err', config, git_repo)
        worker._lane_buffers['normal'].append(req)

        # Counting spy: wraps the REAL merge_tree_conflicts so it still succeeds;
        # we just want to observe how many times it is called.
        call_count = 0
        original_probe = git_ops.merge_tree_conflicts

        async def counting_probe(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return await original_probe(*args, **kwargs)

        git_ops.merge_tree_conflicts = counting_probe

        # Inject a transient footprint-probe failure: get_changed_files raises.
        # This is the ONLY error source; merge_tree_conflicts will still succeed.
        async def failing_changed_files(*args, **kwargs):
            raise RuntimeError('transient get_changed_files failure')

        git_ops.get_changed_files = failing_changed_files

        # First call: must NOT raise
        await worker.recompute_suffix_conflict_graph()

        # The vs-main probe must have run (resolve_branch_sha succeeded → valid head)
        assert call_count >= 1, (
            'Expected merge_tree_conflicts to be called at least once '
            '(vs-main probe) but got zero calls'
        )

        # CRITICAL: signature must NOT be cached when a probe error occurred.
        # A transient error leaves the signature un-stored so the next tick
        # re-probes instead of serving a degraded/incomplete graph behind the
        # debounce.
        assert worker._suffix_conflict_signature is None, (
            'Expected _suffix_conflict_signature to be None after a '
            'get_changed_files failure (transient error must not be cached), '
            f'but got {worker._suffix_conflict_signature!r}'
        )

        count_after_first = call_count

        # Second call with same suffix + same main_sha (neither changed).
        # The signature was NOT cached, so the debounce MUST NOT short-circuit;
        # re-probing is required.
        await worker.recompute_suffix_conflict_graph()

        assert call_count > count_after_first, (
            f'Expected re-probe on second call (transient error leaves signature '
            f'un-stored), but call_count stayed at {count_after_first} '
            f'(debounce incorrectly short-circuited on a stale signature)'
        )
