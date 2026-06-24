"""Integration-gate suite λ=1895: full δ/ε/ζ/η/θ/ι pipeline composition.

This file is the G2 boundary-test leaf for the two-layer merge-queue PRD
(plans/two-layer-merge-queue-prd.md §8 Phase 4).  It drives the FULL
pipeline (δ conflict-graph + ε frontier + ζ aging + η bounce + θ breaker
+ ι metrics) through the REAL public methods against a REAL git repo with
the DEFAULT footprint detector.

Fake/instrumented runners only:
  - rebase_onto_main stubbed (AsyncMock)
  - run_scoped_verification + shutil.disk_usage mocked
  - no real ssh/build/merge-to-main

Integration delta over upstream unit suites
-------------------------------------------
The upstream unit suites hand-seed ``_suffix_conflict_graph`` directly.
This suite drives ``recompute_suffix_conflict_graph()`` against REAL git
branches with the DEFAULT footprint detector so the real graph flows into
aging / bounce / frontier.  Scenario 9 replaces the MagicMock worker
(used by the upstream θ test) with a real SpeculativeMergeWorker so the
breaker reads ι's REAL ``landings_total`` from the real ``snapshot()``
and a real landing drives the resume path.

§5.3 invariants asserted in a single run
-----------------------------------------
  I1. Frozen prefix immutable — inflight entries are never reordered or
      re-based out from under an in-flight verify.
  I2. Verify base equals frozen tip — a real-verify dispatch starts only
      against the tip of the frozen prefix (base-chain integrity via
      check_frozen_prefix_invariant).
  I3. Reorder touches only the unfrozen suffix — recompute_suffix_conflict_graph
      updates only _lane_buffers; _inflight order and base_sha are unchanged.
  I4. Liveness / no-starvation — an item in a clique eventually becomes head
      of clique and is picked; disjoint items bypass a blocked clique.

§9 boundary scenarios (DEFAULT detector; scenario 7=κ out of scope)
--------------------------------------------------------------------
  1. Textual conflict bounces disk-free — younger→needs_rebase at recompute
     time; no _merge-* worktree created, no verify slot consumed.
  2. Clean speculative rebase re-queues — work preserved, no agent dispatched,
     merge_first_enqueued_at unchanged.
  3. Real conflict escalates capped — Nth bounce → cap → blocked without rebase
     (1688 thrash-signature backstop).
  4. Aging beats FIFO within a footprint clique (ζ=1891 comparator via real
     recomputed graph, not a hand-seeded one).
  5. Disjoint throughput bypass — disjoint items bypass a blocked clique while
     the frozen prefix is held.
  6. Verify frontier immutable under suffix reorder — inflight order and
     base_sha survive a suffix reorder + recompute; two_layer_invariants == []
     before AND after.
  8. Contract textual⇒overlap — textual_edges ⊆ footprint_edges (DEFAULT
     detector; real branches, real recompute).
  9. Circuit-breaker fires + escalates + resumes — REAL Harness + REAL
     SpeculativeMergeWorker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from escalation.queue import EscalationQueue

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.harness import Harness
from orchestrator.merge_queue import (
    MERGE_BOUNCE_CAP,
    NEEDS_REBASE_REASON_PREFIX,
    InflightEntry,
    MergeOutcome,
    MergeRequest,
    NoLandingsCircuitBreaker,
    SpeculativeItem,
    SpeculativeMergeWorker,
    SuffixConflictGraph,
)
from orchestrator.run_store import RunStore

# ── Module-level sentinel ─────────────────────────────────────────────────────

_SENTINEL_VERIFY_TASK = object()  # noqa: PD901
"""Sentinel for verify_task in pure unit tests.

frozen_prefix() checks ``e.verify_task is not None`` only, so any non-None
object doubles as a verifying-entry marker in tests that do not need a real
asyncio.Task.
"""

# Harness breaker escalation constants (must match harness.py implementation)
_BREAKER_ROLE = 'orchestrator-no-landings-breaker'
_BREAKER_SENTINEL = '__no_landings_breaker__'


# ── Repo seeding ──────────────────────────────────────────────────────────────


async def _setup_repo(repo: Path) -> None:
    """Initialise a git repo with README.md + shared.txt + disjoint.txt on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    (repo / 'shared.txt').write_text('line1\nline2\nline3\n')
    (repo / 'disjoint.txt').write_text('disjoint-line1\ndisjoint-line2\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


# ── Fixtures ──────────────────────────────────────────────────────────────────


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


# ── Request / item / entry builders ──────────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    lane: Literal['normal', 'high'] = 'normal',
    *,
    merge_first_enqueued_at: float | None = 1000.0,
    request_id: str | None = None,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
    The optional *request_id* kwarg lets tests pin a stable identity for
    graph-key assertions; when omitted a fresh UUID is auto-generated.
    """
    kwargs: dict = {}
    if request_id is not None:
        kwargs['request_id'] = request_id
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane=lane,
        merge_first_enqueued_at=merge_first_enqueued_at,
        **kwargs,
    )


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a bare SpeculativeMergeWorker for unit tests (no harness wiring)."""
    return SpeculativeMergeWorker(git_ops, asyncio.Queue())


def _make_fake_item(
    task_id: str,
    *,
    base_sha: str = 'aaaa0000',
    merge_commit: str | None = 'bbbb1111',
    config: OrchestratorConfig,
    git_repo: Path,
) -> tuple[MergeRequest, SpeculativeItem]:
    """Build a (MergeRequest, SpeculativeItem) pair from fake SHAs.

    Does NOT create real git branches — suitable for pure-unit tests that only
    exercise frozen-prefix / frozen_prefix_tip accessor methods without real
    git operations.  Must be called from an async context (for the future).
    """
    req = _make_req(task_id, f'task/{task_id}', config, git_repo)
    fake_merge_result = (
        MergeResult(success=True, merge_commit=merge_commit)
        if merge_commit is not None
        else None
    )
    item = SpeculativeItem(
        request=req,
        merge_result=fake_merge_result,
        merge_wt=None,
        base_sha=base_sha,
        speculative=False,
        skip_verify=False,
    )
    return req, item


def _make_inflight_entry(
    item: SpeculativeItem,
    *,
    verifying: bool = True,
) -> InflightEntry:
    """Build an InflightEntry for unit tests.

    verifying=True  → verify_task set to _SENTINEL_VERIFY_TASK (any non-None
                       object; frozen_prefix() only checks ``is not None``).
    verifying=False → verify_task=None (passthrough entry; excluded from the
                       frozen prefix by §5.3 definition).
    """
    return InflightEntry(
        item=item,
        lease=None,
        verify_task=_SENTINEL_VERIFY_TASK if verifying else None,  # type: ignore[arg-type]
        merge_wt=None,
        was_speculative=False,
        phase='verifying',
    )


# ── Branch / commit helpers ───────────────────────────────────────────────────


async def _create_branch_editing(
    repo: Path,
    branch_name: str,
    filename: str,
    content: str,
    base_branch: str = 'main',
) -> str:
    """Create a branch editing *filename* with *content*; return the branch SHA."""
    await _run(['git', 'checkout', '-b', branch_name], cwd=repo)
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Edit {filename} in {branch_name}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    await _run(['git', 'checkout', base_branch], cwd=repo)
    return sha.strip()


async def _create_frozen_tip_commit(
    repo: Path,
    filename: str,
    content: str,
) -> str:
    """Commit a change directly on main (simulates a frozen-prefix merge commit).

    Assumes the current branch is already 'main' (left by _setup_repo).
    Returns the resulting commit SHA.
    """
    (repo / filename).write_text(content)
    await _run(['git', 'add', filename], cwd=repo)
    await _run(['git', 'commit', '-m', f'Frozen-tip commit: {filename}'], cwd=repo)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    return sha.strip()


# ── Harness factory + real-worker attachment (for scenario 9) ─────────────────


def _make_harness(tmp_path: Path) -> tuple[Harness, MagicMock]:
    """Harness with a real Scheduler + EscalationQueue and a spy RunStore.

    Mirrors test_harness_no_landings_breaker._make_harness.
    Returns (harness, mock_run_store).
    """
    config = OrchestratorConfig(project_root=tmp_path)
    harness = Harness(config)
    mock_run_store = MagicMock(spec=RunStore)
    harness._run_store = mock_run_store
    harness._run_id = 'run-integration-0001'
    harness.event_store = EventStore(tmp_path / 'events.db', 'run-integration-0001')
    harness._escalation_queue = EscalationQueue(tmp_path / 'escalations')
    return harness, mock_run_store


def _attach_real_worker(harness: Harness, git_ops: GitOps) -> SpeculativeMergeWorker:
    """Attach a REAL SpeculativeMergeWorker to the harness as harness._merge_worker.

    Integration delta over the upstream θ test (which used a MagicMock worker):
    the breaker reads ι's REAL ``landings_total`` from the real ``snapshot()``
    and a real landing drives the resume path.

    Returns the attached worker for direct manipulation in tests.
    """
    worker = SpeculativeMergeWorker(
        git_ops,
        asyncio.Queue(),
        escalation_queue=harness._escalation_queue,
    )
    harness._merge_worker = worker
    return worker


def _small_breaker(window: int = 3, margin: int = 1000) -> NoLandingsCircuitBreaker:
    """Return a NoLandingsCircuitBreaker with a small window for test speed."""
    return NoLandingsCircuitBreaker(window_samples=window, disk_margin_bytes=margin)


def _make_falling_disk_iter(
    start: int = 200_000,
    drop: int = 10_000,
):
    """Generate strictly-falling free-bytes values (for breaker-trip scenarios)."""
    current = start
    while True:
        yield current
        current -= drop


# ── step-01 RED: consolidated §5.3 health surface ────────────────────────────


@pytest.mark.asyncio
class TestTwoLayerInvariants:
    """worker.two_layer_invariants(main_sha) -> list[str] is a new consolidated
    §5.3 health method.

    Returns [] when all invariants hold; returns non-empty violation strings
    naming the offending request_id for each broken invariant.

    RED until step-02 GREEN adds the method + snapshot key to SpeculativeMergeWorker.
    """

    async def test_healthy_pipeline_returns_empty_list(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """A well-formed pipeline returns [] — no §5.3 violations detected.

        Well-formed means:
        - Frozen prefix base-chain is intact (base_sha == main_sha for the
          first entry; each subsequent entry's base_sha == predecessor's
          merge_commit).
        - Frozen rids and suffix rids are disjoint.
        - SuffixConflictGraph has textual_edges ⊆ footprint_edges.
        - SuffixConflictGraph conflicts_with_main ⊆ nodes.
        """
        worker = _make_worker(git_ops)
        main_sha = 'main-sha-001'

        # Build a frozen entry chained correctly off main_sha.
        _, item_a = _make_fake_item(
            'task-a', base_sha=main_sha, merge_commit='merge-sha-a',
            config=config, git_repo=git_repo,
        )
        entry_a = _make_inflight_entry(item_a, verifying=True)
        worker._inflight.append(entry_a)

        # Suffix req — different rid, not in frozen prefix.
        req_b = _make_req('task-b', 'task/task-b', config, git_repo)
        worker._lane_buffers['normal'].append(req_b)

        # Build a valid graph: textual_edges ⊆ footprint_edges;
        # conflicts_with_main ⊆ nodes.
        edge_pair = frozenset({req_b.request_id, 'some-other-rid'})
        worker._suffix_conflict_graph = SuffixConflictGraph(
            nodes=(req_b.request_id,),
            textual_edges=frozenset(),      # empty ⊆ any set
            footprint_edges=frozenset({edge_pair}),
            conflicts_with_main=frozenset({req_b.request_id}),
        )

        violations = worker.two_layer_invariants(main_sha)

        assert violations == [], (
            f'Expected [] for a well-formed pipeline but got: {violations!r}'
        )

    async def test_violation_a_broken_base_chain_detected(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """(a) A broken base-chain in the frozen prefix produces a violation str
        naming the offending request_id.
        """
        worker = _make_worker(git_ops)
        main_sha = 'main-sha-001'

        # Frozen entry whose base_sha does NOT match main_sha → chain broken.
        _, item_a = _make_fake_item(
            'task-a', base_sha='wrong-sha', merge_commit='merge-sha-a',
            config=config, git_repo=git_repo,
        )
        entry_a = _make_inflight_entry(item_a, verifying=True)
        worker._inflight.append(entry_a)

        violations = worker.two_layer_invariants(main_sha)

        assert violations, 'Expected a violation for broken base-chain but got []'
        # At least one violation must name the offending rid.
        joined = ' '.join(violations)
        assert item_a.request.request_id in joined, (
            f'Expected offending rid {item_a.request.request_id!r} in violations '
            f'but got: {violations!r}'
        )

    async def test_violation_b_frozen_suffix_overlap_detected(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """(b) A rid present in BOTH the frozen prefix and the unfrozen suffix
        is flagged as a disjointness violation naming that rid.
        """
        worker = _make_worker(git_ops)
        main_sha = 'main-sha-001'

        # Build frozen entry.
        req_a = _make_req('task-a', 'task/task-a', config, git_repo)
        item_a = SpeculativeItem(
            request=req_a,
            merge_result=MergeResult(success=True, merge_commit='merge-sha-a'),
            merge_wt=None,
            base_sha=main_sha,
            speculative=False,
            skip_verify=False,
        )
        entry_a = _make_inflight_entry(item_a, verifying=True)
        worker._inflight.append(entry_a)

        # ALSO put req_a in the suffix — this violates frozen∩suffix disjointness.
        worker._lane_buffers['normal'].append(req_a)

        violations = worker.two_layer_invariants(main_sha)

        assert violations, 'Expected a disjointness violation but got []'
        joined = ' '.join(violations)
        assert req_a.request_id in joined, (
            f'Expected rid {req_a.request_id!r} named in disjointness violation '
            f'but got: {violations!r}'
        )

    async def test_violation_c_textual_not_subset_of_footprint(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """(c) textual_edges containing an edge not in footprint_edges is flagged
        with a violation naming the offending pair (request_ids).
        """
        worker = _make_worker(git_ops)
        main_sha = 'main-sha-001'

        req_x = _make_req('task-x', 'task/task-x', config, git_repo,
                          request_id='rid-x')
        req_y = _make_req('task-y', 'task/task-y', config, git_repo,
                          request_id='rid-y')
        worker._lane_buffers['normal'].extend([req_x, req_y])

        # textual edge {rid-x, rid-y} NOT in footprint_edges → contract violated.
        extra_edge = frozenset({'rid-x', 'rid-y'})
        worker._suffix_conflict_graph = SuffixConflictGraph(
            nodes=('rid-x', 'rid-y'),
            textual_edges=frozenset({extra_edge}),   # NOT a subset of footprint_edges
            footprint_edges=frozenset(),              # empty → textual edge is spurious
            conflicts_with_main=frozenset(),
        )

        violations = worker.two_layer_invariants(main_sha)

        assert violations, (
            'Expected a textual⊈footprint violation but got []'
        )
        joined = ' '.join(violations)
        # At least one of the two rids should appear in the violation message.
        assert 'rid-x' in joined or 'rid-y' in joined, (
            f'Expected rid-x or rid-y in violation but got: {violations!r}'
        )

    async def test_violation_d_conflicts_with_main_missing_from_nodes(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """(d) conflicts_with_main containing a rid absent from nodes is flagged
        as a consistency violation naming that rid.
        """
        worker = _make_worker(git_ops)
        main_sha = 'main-sha-001'

        req_x = _make_req('task-x', 'task/task-x', config, git_repo,
                          request_id='rid-x')
        worker._lane_buffers['normal'].append(req_x)

        # 'rid-ghost' is in conflicts_with_main but NOT in nodes → inconsistency.
        worker._suffix_conflict_graph = SuffixConflictGraph(
            nodes=('rid-x',),
            textual_edges=frozenset(),
            footprint_edges=frozenset(),
            conflicts_with_main=frozenset({'rid-ghost'}),
        )

        violations = worker.two_layer_invariants(main_sha)

        assert violations, (
            'Expected a conflicts_with_main⊄nodes violation but got []'
        )
        joined = ' '.join(violations)
        assert 'rid-ghost' in joined, (
            f'Expected rid-ghost named in violation but got: {violations!r}'
        )

    async def test_snapshot_grows_two_layer_invariants_key(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """snapshot() exposes an additive 'two_layer_invariants' key (list).

        All pre-existing snapshot keys must remain intact (entries, depth,
        head_of_line, suffix_conflict_graph, frozen_prefix, metrics).
        """
        worker = _make_worker(git_ops)

        snap = worker.snapshot()

        # New additive key must be present and be a list.
        assert 'two_layer_invariants' in snap, (
            f"'two_layer_invariants' key not found in snapshot(). "
            f'Keys present: {sorted(snap.keys())!r}'
        )
        assert isinstance(snap['two_layer_invariants'], list), (
            f"Expected 'two_layer_invariants' to be a list, got "
            f"{type(snap['two_layer_invariants'])!r}"
        )

        # All pre-existing keys must still be present (backward-compatible).
        required_keys = {
            'entries', 'depth', 'head_of_line',
            'suffix_conflict_graph', 'frozen_prefix', 'metrics',
        }
        missing = required_keys - snap.keys()
        assert not missing, (
            f'Pre-existing snapshot keys missing after adding two_layer_invariants: '
            f'{missing!r}'
        )


# ── step-03 RED: §9 scenario 1 (textual conflict bounces disk-free)
#                + scenario 8 (contract textual⇒overlap) via REAL pipeline ────


@pytest.mark.asyncio
class TestScenario1And8:
    """§9 scenario 1 + 8: REAL branches, REAL recompute, DEFAULT detector.

    Integration delta: the upstream unit suites hand-seed _suffix_conflict_graph;
    this class drives recompute_suffix_conflict_graph() against REAL git branches
    so the real graph flows into bounce and the textual⇒overlap contract is
    verified end-to-end.

    Scenario 8 (textual⇒overlap contract):
      Two branches editing the SAME line of shared.txt produce a pair that is in
      BOTH footprint_edges (path-overlap) AND textual_edges (3-way conflict), and
      textual_edges ⊆ footprint_edges holds.

    Scenario 1 (textual conflict bounces disk-free):
      The conflicting suffix item is diverted via _bounce_conflicting_suffix_items()
      BEFORE any verify slot: _verifier_queue is empty and no _merge-* worktree
      directory was created.

    RED until the suite lands (the recompute / bounce composition has no known
    gap — these tests exercise the already-deployed wiring at the integration
    boundary; they are expected to go GREEN without production changes).
    """

    async def _setup_real_conflict_branches(
        self,
        git_repo: Path,
        config: OrchestratorConfig,
        git_ops: GitOps,
    ) -> tuple[SpeculativeMergeWorker, MergeRequest, MergeRequest, str]:
        """Create two conflicting branches + a frozen-tip InflightEntry.

        Returns (worker, req_a, req_b, frozen_tip_sha) where:
          req_a — older (merge_first_enqueued_at=100.0), edits shared.txt line2→BRANCH-A
          req_b — younger (merge_first_enqueued_at=300.0), edits shared.txt line2→BRANCH-B
          frozen_tip_sha — SHA of the frozen-tip commit (on a side branch, not main)

        Both branches conflict with each other AND with the frozen tip (all edit
        the same line of shared.txt).  The frozen prefix holds the frozen-tip
        entry so frozen_prefix_tip() → frozen_tip_sha ≠ main_sha.
        """
        # 1. Create the frozen-tip commit on a SIDE branch (NOT on main) so that
        #    main_sha stays at the initial commit M0.  frozen_prefix_tip will then
        #    be this side-branch SHA (derived from the frozen entry's merge_commit).
        frozen_tip_sha = await _create_branch_editing(
            git_repo, 'task/frozen-tip', 'shared.txt',
            'line1\nFROZEN-LINE2\nline3\n',
        )

        # 2. Create suffix branches off M0, both editing line2 differently.
        await _create_branch_editing(
            git_repo, 'task/branch-a', 'shared.txt', 'line1\nBRANCH-A-LINE2\nline3\n',
        )
        await _create_branch_editing(
            git_repo, 'task/branch-b', 'shared.txt', 'line1\nBRANCH-B-LINE2\nline3\n',
        )

        # 3. Build worker.
        worker = _make_worker(git_ops)

        # 4. Populate frozen prefix with the frozen-tip commit as merge_commit.
        #    base_sha = main_sha at initial commit (any real SHA — the chain
        #    integrity check would be tested separately; here we just need
        #    frozen_prefix_tip() to return frozen_tip_sha).
        _, main_sha_raw, _ = await _run(['git', 'rev-parse', 'main'], cwd=git_repo)
        main_sha_at_m0 = main_sha_raw.strip()
        _, item_frozen = _make_fake_item(
            'frozen-task',
            base_sha=main_sha_at_m0,
            merge_commit=frozen_tip_sha,
            config=config,
            git_repo=git_repo,
        )
        entry_frozen = _make_inflight_entry(item_frozen, verifying=True)
        worker._inflight.append(entry_frozen)

        # 5. Put the two suffix requests in the normal lane (A is FIFO-head).
        req_a = _make_req(
            'branch-a', 'branch-a', config, git_repo,
            merge_first_enqueued_at=100.0,
        )
        req_b = _make_req(
            'branch-b', 'branch-b', config, git_repo,
            merge_first_enqueued_at=300.0,
        )
        worker._lane_buffers['normal'].extend([req_a, req_b])

        return worker, req_a, req_b, frozen_tip_sha

    async def test_scenario_8_textual_subset_of_footprint(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """Scenario 8: the conflicting pair is in BOTH textual_edges AND footprint_edges,
        and textual_edges ⊆ footprint_edges holds — verified through the REAL pipeline.
        """
        worker, req_a, req_b, _ = await self._setup_real_conflict_branches(
            git_repo, config, git_ops,
        )

        await worker.recompute_suffix_conflict_graph()

        graph = worker._suffix_conflict_graph
        edge = frozenset({req_a.request_id, req_b.request_id})

        # Both branches edit shared.txt → footprint overlap.
        assert edge in graph.footprint_edges, (
            f'Expected footprint edge {{A,B}} but footprint_edges={graph.footprint_edges!r}'
        )

        # Both branches edit the SAME LINE → textual conflict.
        assert edge in graph.textual_edges, (
            f'Expected textual edge {{A,B}} but textual_edges={graph.textual_edges!r}'
        )

        # Contract: textual_edges ⊆ footprint_edges.
        assert graph.textual_edges <= graph.footprint_edges, (
            f'textual_edges ⊄ footprint_edges! '
            f'textual_edges={graph.textual_edges!r}, '
            f'footprint_edges={graph.footprint_edges!r}'
        )

    async def test_scenario_8_conflicts_with_main_flagged(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """Scenario 8 + 1 setup: after recompute, at least one suffix item is in
        conflicts_with_main (conflicts with the frozen tip).
        """
        worker, req_a, req_b, frozen_tip_sha = await self._setup_real_conflict_branches(
            git_repo, config, git_ops,
        )

        await worker.recompute_suffix_conflict_graph()

        graph = worker._suffix_conflict_graph

        # At least one (likely both) suffix items conflict with the frozen tip.
        conflicting = {req_a.request_id, req_b.request_id} & graph.conflicts_with_main
        assert conflicting, (
            f'Expected at least one suffix item in conflicts_with_main but got '
            f'conflicts_with_main={graph.conflicts_with_main!r}.  '
            f'frozen_prefix_tip should be the frozen-tip SHA; both suffix branches '
            f'edit the same line of shared.txt as the frozen tip.'
        )

    async def test_scenario_1_bounce_before_verify_slot(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """Scenario 1: the conflicting item is bounced BEFORE any verify slot.

        After _bounce_conflicting_suffix_items() (with rebase_onto_main stubbed):
          - _verifier_queue is empty (qsize == 0): no verify slot consumed.
          - No _merge-* worktree directory was created under git_repo.
          - The conflicting req(s) are either re-queued (rebase=True) or
            resolved 'blocked' with NEEDS_REBASE_REASON_PREFIX (rebase=False).
        """
        worker, req_a, req_b, _ = await self._setup_real_conflict_branches(
            git_repo, config, git_ops,
        )

        await worker.recompute_suffix_conflict_graph()

        graph = worker._suffix_conflict_graph
        conflicting_rids = {req_a.request_id, req_b.request_id} & graph.conflicts_with_main
        assert conflicting_rids, 'precondition: at least one item must be in conflicts_with_main'

        # Stub rebase → False: real conflict → item removed, future 'blocked'.
        worker._git_ops.rebase_onto_main = AsyncMock(return_value=False)  # type: ignore[method-assign]

        await worker._bounce_conflicting_suffix_items()

        # SCENARIO 1 assertion A: _verifier_queue is empty — no verify slot consumed.
        assert worker._verifier_queue.qsize() == 0, (
            f'Expected _verifier_queue empty (no verify slot) but qsize='
            f'{worker._verifier_queue.qsize()}'
        )

        # SCENARIO 1 assertion B: no _merge-* worktree created.
        wt_base = git_repo / '.worktrees'
        if wt_base.exists():
            merge_dirs = [
                d for d in wt_base.iterdir()
                if 'merge' in d.name.lower() or d.name.startswith('_merge')
            ]
            assert not merge_dirs, (
                f'Unexpected merge worktree dirs created before verify slot: '
                f'{merge_dirs!r}'
            )

        # SCENARIO 1 assertion C: conflicting reqs are resolved 'blocked'.
        for rid in conflicting_rids:
            req = req_a if rid == req_a.request_id else req_b
            assert req.result.done(), (
                f'Expected req {rid!r} to be resolved after bounce (rebase=False)'
            )
            outcome = req.result.result()
            assert outcome.status == 'blocked', (
                f'Expected status=blocked for bounced req {rid!r}, got {outcome.status!r}'
            )
            assert outcome.reason is not None
            assert outcome.reason.startswith(NEEDS_REBASE_REASON_PREFIX), (
                f'Expected reason to start with NEEDS_REBASE_REASON_PREFIX for {rid!r}, '
                f'got {outcome.reason!r}'
            )

    async def test_scenario_1_two_layer_invariants_holds_before_bounce(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """two_layer_invariants() == [] holds before the bounce (frozen prefix intact).

        The frozen-prefix chain is well-formed even with conflicting suffix items —
        the invariant only cares about the frozen prefix structure, not whether
        suffix items conflict.
        """
        worker, req_a, req_b, _ = await self._setup_real_conflict_branches(
            git_repo, config, git_ops,
        )

        await worker.recompute_suffix_conflict_graph()

        # Get the actual main SHA for the invariant check.
        _, main_sha_raw, _ = await _run(['git', 'rev-parse', 'main'], cwd=git_repo)
        main_sha = main_sha_raw.strip()

        violations = worker.two_layer_invariants(main_sha)
        assert violations == [], (
            f'Expected two_layer_invariants == [] before bounce (frozen prefix is '
            f'well-formed) but got: {violations!r}'
        )
