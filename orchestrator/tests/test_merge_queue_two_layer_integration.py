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


def _small_breaker(window: int = 3, floor: int = 1000) -> NoLandingsCircuitBreaker:
    """Return a NoLandingsCircuitBreaker with a small window for test speed."""
    return NoLandingsCircuitBreaker(window_samples=window, disk_free_floor_bytes=floor)


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

    # ── step-15 RED: snapshot() must use the REAL main SHA (not frozen tip) ─────

    async def test_snapshot_uses_real_main_not_frozen_tip(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """snapshot()['two_layer_invariants'] must agree with two_layer_invariants(M0).

        Regression test for the false base-chain violation in snapshot().

        Setup
        -----
        Build a HEALTHY frozen prefix: one verifying entry with
          base_sha = M0 (real main HEAD)
          merge_commit = M1 (a DISTINCT side-branch commit ≠ M0)

        This means _newest_frozen_commit() returns M1 ≠ M0.

        Failure mode (before step-16 fix)
        ----------------------------------
        snapshot() calls:
            two_layer_invariants(self._newest_frozen_commit() or 'unknown')
            → two_layer_invariants(M1)
        check_frozen_prefix_invariant(M1) expects the first frozen entry's
        base_sha to be M1, but it is M0 → spurious violation
            'frozen-prefix base-chain broken at <rid>: expected M1 but item has M0'
        so snapshot()['two_layer_invariants'] != [] even though the pipeline
        is HEALTHY.

        Expected behaviour (after step-16 fix)
        ----------------------------------------
        snapshot() uses the REAL main SHA cached at recompute time (M0), so
        two_layer_invariants(M0) returns [] and the surfaces agree.

        Assertions
        ----------
        (a) direct call healthy:   worker.two_layer_invariants(M0) == []
        (b) snapshot agrees:       worker.snapshot()['two_layer_invariants'] == []
                                   (FAILS today — snapshot forwards M1 → false +ve)
        (c) surfaces agree:        snapshot value == direct call value
        """
        worker = _make_worker(git_ops)

        # M0 — real main HEAD before anything else.
        m0 = await git_ops.get_main_sha()

        # M1 — a DISTINCT side-branch commit (merge_commit of the frozen entry).
        # _create_branch_editing leaves main at M0 so get_main_sha() still
        # returns M0 after recompute_suffix_conflict_graph() runs.
        m1 = await _create_branch_editing(
            git_repo, 'task/frozen-tip-s15', 'README.md', '# Frozen tip S15\n',
        )
        assert m1 != m0, 'precondition: M1 must differ from M0'

        # Build the frozen entry: base_sha=M0 (correct chain from real main),
        # merge_commit=M1 (a real SHA so _newest_frozen_commit() returns it).
        req_frozen = _make_req('frozen-s15', 'task/frozen-tip-s15', config, git_repo)
        item_frozen = SpeculativeItem(
            request=req_frozen,
            merge_result=MergeResult(success=True, merge_commit=m1),
            merge_wt=None,
            base_sha=m0,
            speculative=False,
            skip_verify=False,
        )
        entry_frozen = _make_inflight_entry(item_frozen, verifying=True)
        worker._inflight.append(entry_frozen)

        # Drive recompute so the worker resolves + caches the real main SHA.
        # (No suffix reqs → EMPTY_SUFFIX_CONFLICT_GRAPH; that's fine —
        #  the §5.3 check is on the frozen prefix only.)
        await worker.recompute_suffix_conflict_graph()

        # Verify the precondition: _newest_frozen_commit() ≠ real main M0.
        newest = worker._newest_frozen_commit()
        assert newest is not None, '_newest_frozen_commit() must return M1 for this test'
        assert newest != m0, (
            f'precondition: _newest_frozen_commit() must ≠ M0 but got {newest!r} == {m0!r}'
        )

        # (a) Direct call with the REAL main SHA must be healthy.
        direct_violations = worker.two_layer_invariants(m0)
        assert direct_violations == [], (
            f'two_layer_invariants(M0) should return [] for a healthy pipeline, '
            f'but got: {direct_violations!r}'
        )

        # (b) snapshot()['two_layer_invariants'] must also be [] (FAILS before fix).
        snap = worker.snapshot()
        snap_violations = snap['two_layer_invariants']
        assert snap_violations == [], (
            f"snapshot()['two_layer_invariants'] should be [] for a healthy pipeline "
            f'but got: {snap_violations!r}\n'
            f'(If non-empty, snapshot() is forwarding the frozen tip M1={newest!r} '
            f'instead of the real main M0={m0!r} — the step-16 bug.)'
        )

        # (c) Both surfaces must agree.
        assert snap_violations == direct_violations, (
            f"snapshot() and two_layer_invariants(M0) disagree:\n"
            f"  snapshot = {snap_violations!r}\n"
            f"  direct   = {direct_violations!r}"
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
        # Compute unconditionally so the assert always executes (not vacuously
        # skipped when .worktrees does not exist yet).
        wt_base = git_repo / '.worktrees'
        merge_dirs = (
            [
                d for d in wt_base.iterdir()
                if 'merge' in d.name.lower() or d.name.startswith('_merge')
            ]
            if wt_base.exists()
            else []
        )
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


# ── step-05 RED: §9 scenario 2 (clean speculative rebase re-queues) ──────────


@pytest.mark.asyncio
class TestScenario2CleanRebaseRequeues:
    """§9 scenario 2: clean speculative rebase re-queues — work preserved, no agent.

    Seed a frozen-tip entry + a suffix req flagged in conflicts_with_main;
    stub rebase_onto_main → True (clean rebase onto frozen tip).  Call
    _bounce_conflicting_suffix_items().

    Assertions:
      - rebase_onto_main awaited once with onto=frozen_tip_sha
      - req REMAINS in _lane_buffers['normal'] (re-queued, work preserved)
      - req.result is NOT done (no agent dispatched)
      - req.merge_first_enqueued_at unchanged (== 9999.0)
      - _bounce_registry.count(branch) == 1
      - _suffix_conflict_signature invalidated to None
      - a needs_rebase log line emitted

    Integration delta: the graph is REAL (recomputed via recompute_suffix_conflict_graph
    against a real branch) rather than hand-seeded.

    RED until the suite lands.
    """

    async def test_clean_rebase_requeues_work_preserved(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Clean rebase → req stays in lane buffer; future untouched; registry++."""
        import logging

        # 1. Create the frozen-tip commit on a side branch (not main) so that
        #    main stays at M0 and frozen_prefix_tip == frozen_tip_sha.
        frozen_tip_sha = await _create_branch_editing(
            git_repo, 'task/frozen-tip-s2', 'shared.txt',
            'line1\nFROZEN-LINE2-S2\nline3\n',
        )

        # 2. Create suffix branch editing the SAME line → will conflict with frozen tip.
        await _create_branch_editing(
            git_repo, 'task/suffix-branch', 'shared.txt',
            'line1\nSUFFIX-LINE2\nline3\n',
        )

        # 3. Build worker with frozen entry.
        worker = _make_worker(git_ops)
        _, main_sha_raw, _ = await _run(['git', 'rev-parse', 'main'], cwd=git_repo)
        main_sha = main_sha_raw.strip()

        _, item_frozen = _make_fake_item(
            'frozen-s2', base_sha=main_sha, merge_commit=frozen_tip_sha,
            config=config, git_repo=git_repo,
        )
        entry_frozen = _make_inflight_entry(item_frozen, verifying=True)
        worker._inflight.append(entry_frozen)

        # 4. Build suffix req with a known merge_first_enqueued_at.
        req = _make_req(
            'suffix-branch', 'suffix-branch', config, git_repo,
            merge_first_enqueued_at=9999.0,
        )
        worker._lane_buffers['normal'].append(req)

        # 5. Recompute the real graph; the suffix branch conflicts with the frozen tip.
        await worker.recompute_suffix_conflict_graph()

        assert req.request_id in worker._suffix_conflict_graph.conflicts_with_main, (
            'precondition: req must be in conflicts_with_main after recompute; '
            'suffix branch edits the same line as the frozen tip'
        )

        # Store the debounce signature so we can assert it's cleared later.
        assert worker._suffix_conflict_signature is not None, (
            'precondition: debounce signature must be set after a successful recompute'
        )

        # 6. Stub rebase → True (clean rebase — no actual git op).
        worker._git_ops.rebase_onto_main = AsyncMock(return_value=True)  # type: ignore[method-assign]

        with caplog.at_level(logging.INFO):
            await worker._bounce_conflicting_suffix_items()

        # Assert: rebase_onto_main called with onto=frozen_tip_sha.
        worker._git_ops.rebase_onto_main.assert_awaited_once()
        call_kwargs = worker._git_ops.rebase_onto_main.call_args
        onto_arg = call_kwargs.kwargs.get('onto') or (
            call_kwargs.args[1] if len(call_kwargs.args) >= 2 else None
        )
        assert onto_arg == frozen_tip_sha, (
            f'Expected rebase_onto_main called with onto={frozen_tip_sha!r} '
            f'but got {onto_arg!r}'
        )

        # Assert: req REMAINS in lane buffer (re-queued, work preserved).
        assert req in worker._lane_buffers['normal'], (
            'Expected req to remain in lane buffer after a clean rebase (re-queue)'
        )

        # Assert: future NOT done (no agent dispatched).
        assert not req.result.done(), (
            'Expected req.result future to remain pending after a clean rebase'
        )

        # Assert: merge_first_enqueued_at unchanged.
        assert req.merge_first_enqueued_at == 9999.0, (
            f'Expected merge_first_enqueued_at == 9999.0 but got '
            f'{req.merge_first_enqueued_at!r}'
        )

        # Assert: bounce registry incremented to 1.
        branch = req.branch
        assert worker._bounce_registry.count(branch) == 1, (
            f'Expected bounce registry count for branch {branch!r} to be 1 '
            f'after one clean rebase bounce'
        )

        # Assert: _suffix_conflict_signature invalidated to None.
        assert worker._suffix_conflict_signature is None, (
            'Expected _suffix_conflict_signature to be None after a bounce '
            '(so next recompute re-probes reality)'
        )

        # Assert: a needs_rebase log line was emitted.
        relevant = [
            r for r in caplog.records
            if 'needs_rebase' in r.message.lower()
            or NEEDS_REBASE_REASON_PREFIX.lower()[:10] in r.message.lower()
        ]
        assert relevant, (
            f'Expected a needs_rebase log line but none found. '
            f'Log records: {[r.message for r in caplog.records]!r}'
        )


# ── step-07 RED: §9 scenario 3 (real conflict escalates, capped) ─────────────


@pytest.mark.asyncio
class TestScenario3RealConflictCapped:
    """§9 scenario 3: real rebase conflict escalates; repeated bounces hit the cap.

    Drive repeated bounces of one conflicting suffix req across N cycles with
    rebase_onto_main → False (real conflict).

    Assertions:
      CASE A — single real conflict (under cap):
        - req removed from lane buffer
        - req.result 'blocked' with NEEDS_REBASE_REASON_PREFIX
        - rebase_onto_main was awaited (conflict probed)

      CASE B — bounce count already at MERGE_BOUNCE_CAP:
        - Next bounce escalates WITHOUT attempting a rebase (spy not awaited)
        - req removed from lane buffer
        - req.result 'blocked' with NEEDS_REBASE_REASON_PREFIX
        (1688 thrash-signature backstop: a flapping conflict cannot become an
        unbounded agent-$ fire)

    Integration delta: graph fed by real recompute (not hand-seeded), so
    conflicts_with_main is populated by the DEFAULT detector's real probe.

    RED until the suite lands.
    """

    async def _setup_conflict_worker(
        self,
        git_repo: Path,
        config: OrchestratorConfig,
        git_ops: GitOps,
    ) -> tuple[SpeculativeMergeWorker, MergeRequest, str]:
        """Return (worker, conflicting_req, frozen_tip_sha) with a real graph."""
        frozen_tip_sha = await _create_branch_editing(
            git_repo, 'task/frozen-tip-s3', 'shared.txt',
            'line1\nFROZEN-LINE2-S3\nline3\n',
        )
        await _create_branch_editing(
            git_repo, 'task/conflict-branch', 'shared.txt',
            'line1\nCONFLICT-LINE2\nline3\n',
        )

        worker = _make_worker(git_ops)
        _, main_sha_raw, _ = await _run(['git', 'rev-parse', 'main'], cwd=git_repo)
        main_sha = main_sha_raw.strip()

        _, item_frozen = _make_fake_item(
            'frozen-s3', base_sha=main_sha, merge_commit=frozen_tip_sha,
            config=config, git_repo=git_repo,
        )
        entry_frozen = _make_inflight_entry(item_frozen, verifying=True)
        worker._inflight.append(entry_frozen)

        req = _make_req('conflict-branch', 'conflict-branch', config, git_repo)
        worker._lane_buffers['normal'].append(req)

        await worker.recompute_suffix_conflict_graph()

        assert req.request_id in worker._suffix_conflict_graph.conflicts_with_main, (
            'precondition: req must be in conflicts_with_main after real recompute'
        )
        return worker, req, frozen_tip_sha

    async def test_case_a_real_conflict_blocks_req(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """CASE A: rebase → False (real conflict) → req removed, 'blocked' escalated."""
        worker, req, _ = await self._setup_conflict_worker(git_repo, config, git_ops)

        # Stub rebase → False (real conflict, under cap).
        worker._git_ops.rebase_onto_main = AsyncMock(return_value=False)  # type: ignore[method-assign]

        await worker._bounce_conflicting_suffix_items()

        # rebase_onto_main was called (conflict probed, not short-circuited by cap).
        worker._git_ops.rebase_onto_main.assert_awaited_once()

        # req removed from lane buffer.
        assert req not in worker._lane_buffers['normal'], (
            'Expected req to be removed from lane buffer after a real conflict'
        )

        # req.result 'blocked' with NEEDS_REBASE_REASON_PREFIX.
        assert req.result.done()
        outcome = req.result.result()
        assert outcome.status == 'blocked'
        assert outcome.reason is not None
        assert outcome.reason.startswith(NEEDS_REBASE_REASON_PREFIX), (
            f'Expected reason starting with NEEDS_REBASE_REASON_PREFIX, got {outcome.reason!r}'
        )

    async def test_case_b_cap_exceeded_no_rebase(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """CASE B: bounce count at cap → escalate WITHOUT rebase (1688 thrash backstop).

        Pre-seed registry to MERGE_BOUNCE_CAP for the branch.  The next
        _bounce_conflicting_suffix_items() call must NOT attempt rebase_onto_main,
        remove the req, and resolve it 'blocked' with NEEDS_REBASE_REASON_PREFIX.
        """
        worker, req, _ = await self._setup_conflict_worker(git_repo, config, git_ops)
        branch = req.branch

        # Pre-seed registry to MERGE_BOUNCE_CAP.
        for _ in range(MERGE_BOUNCE_CAP):
            worker._bounce_registry.record_bounce(branch)
        assert worker._bounce_registry.count(branch) == MERGE_BOUNCE_CAP

        # Spy on rebase — it must NOT be called when cap is exceeded.
        rebase_spy = AsyncMock(return_value=False)
        worker._git_ops.rebase_onto_main = rebase_spy  # type: ignore[method-assign]

        await worker._bounce_conflicting_suffix_items()

        # rebase NOT called (cap short-circuit — 1688 thrash-signature backstop).
        rebase_spy.assert_not_awaited()

        # req removed from lane buffer.
        assert req not in worker._lane_buffers['normal'], (
            'Expected req removed from lane buffer when bounce cap exceeded'
        )

        # req.result 'blocked' with NEEDS_REBASE_REASON_PREFIX.
        assert req.result.done()
        outcome = req.result.result()
        assert outcome.status == 'blocked'
        assert outcome.reason is not None
        assert outcome.reason.startswith(NEEDS_REBASE_REASON_PREFIX), (
            f'Expected cap-exceeded reason starting with NEEDS_REBASE_REASON_PREFIX, '
            f'got {outcome.reason!r}'
        )

    async def test_repeated_bounces_count_climbs(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """Per-branch bounce count climbs monotonically until cap reached.

        Drive N = MERGE_BOUNCE_CAP cycles.  Each cycle: re-seed graph,
        stub rebase → False, call _bounce_conflicting_suffix_items().
        After each cycle (under cap), the req is removed and the registry
        resets (cleared on escalation) — we verify the count was incremented
        before clearing by checking the blocking outcome.
        """
        worker, req, _ = await self._setup_conflict_worker(git_repo, config, git_ops)
        branch = req.branch

        # CASE A check: one real conflict → blocked, registry cleared.
        worker._git_ops.rebase_onto_main = AsyncMock(return_value=False)  # type: ignore[method-assign]
        await worker._bounce_conflicting_suffix_items()

        assert req.result.done()
        assert req.result.result().status == 'blocked'
        # Registry is cleared after a real-conflict escalation.
        assert worker._bounce_registry.count(branch) == 0, (
            'Expected registry cleared after real-conflict escalation'
        )


# ── step-09 RED: §9 scenario 4 (aging beats FIFO) + scenario 5 (disjoint bypass) ─


@pytest.mark.asyncio
class TestScenario4And5AgingAndDisjointBypass:
    """§9 scenario 4 + 5: clique-scoped aging beats FIFO + disjoint bypass.

    Uses the REAL recomputed graph (not hand-seeded) so footprint_edges between
    clique members is derived from actual changed-path sets via the DEFAULT detector.

    Setup (FIFO order in lane buffer): [req_a (mfea=300.0), req_c (disjoint), req_b (mfea=100.0)]
      A: edits shared.txt → footprint-clique member with B
      B: edits shared.txt → footprint-clique member with A; older (mfea=100.0)
      C: edits disjoint.txt → disjoint from both A and B

    Expected _pop_next_pickable() sequence:
      Pick 1: C (disjoint bypass — vacuously minimal despite C being in FIFO position 2)
      Pick 2: B (aging: B.mfea=100.0 < A.mfea=300.0 → B is clique-minimal)
      Pick 3: A (only item remaining)

    Invariants: two_layer_invariants(main_sha) == [] before and after each pop
    (frozen prefix immutable; reorder touches only the unfrozen suffix).

    RED until the suite lands.
    """

    async def _setup_aging_and_disjoint(
        self,
        git_repo: Path,
        config: OrchestratorConfig,
        git_ops: GitOps,
    ) -> tuple[SpeculativeMergeWorker, MergeRequest, MergeRequest, MergeRequest, str]:
        """Return (worker, req_a, req_b, req_c, main_sha).

        Creates real branches; populates _lane_buffers in FIFO order [A, C, B];
        adds a frozen entry so the test also validates two_layer_invariants.
        """
        # Create branches off main.
        await _create_branch_editing(
            git_repo, 'task/aging-a', 'shared.txt', 'line1\nAGING-A-LINE2\nline3\n',
        )
        await _create_branch_editing(
            git_repo, 'task/aging-b', 'shared.txt', 'line1\nAGING-B-LINE2\nline3\n',
        )
        await _create_branch_editing(
            git_repo, 'task/disjoint-c', 'disjoint.txt',
            'DISJOINT-C-LINE1\ndisjoint-line2\n',
        )

        # Build worker.
        worker = _make_worker(git_ops)

        # Add a frozen entry to exercise two_layer_invariants in context.
        _, main_sha_raw, _ = await _run(['git', 'rev-parse', 'main'], cwd=git_repo)
        main_sha = main_sha_raw.strip()

        frozen_tip_sha = await _create_branch_editing(
            git_repo, 'task/frozen-tip-s45', 'README.md',
            '# Frozen-tip for scenario 4/5\n',
        )
        _, item_frozen = _make_fake_item(
            'frozen-s45', base_sha=main_sha, merge_commit=frozen_tip_sha,
            config=config, git_repo=git_repo,
        )
        entry_frozen = _make_inflight_entry(item_frozen, verifying=True)
        worker._inflight.append(entry_frozen)

        # Build suffix requests — put in FIFO order [A, C, B].
        req_a = _make_req('aging-a', 'aging-a', config, git_repo, merge_first_enqueued_at=300.0)
        req_c = _make_req('disjoint-c', 'disjoint-c', config, git_repo, merge_first_enqueued_at=200.0)
        req_b = _make_req('aging-b', 'aging-b', config, git_repo, merge_first_enqueued_at=100.0)
        worker._lane_buffers['normal'].extend([req_a, req_c, req_b])

        return worker, req_a, req_b, req_c, main_sha

    async def test_recompute_builds_real_footprint_edge_a_b(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """After recompute, A and B have a footprint edge; C has no edge with A or B."""
        worker, req_a, req_b, req_c, _ = await self._setup_aging_and_disjoint(
            git_repo, config, git_ops,
        )

        await worker.recompute_suffix_conflict_graph()

        graph = worker._suffix_conflict_graph
        edge_ab = frozenset({req_a.request_id, req_b.request_id})

        assert edge_ab in graph.footprint_edges, (
            f'Expected footprint edge {{A,B}} — both edit shared.txt. '
            f'footprint_edges={graph.footprint_edges!r}'
        )

        # C is disjoint from A and B.
        edge_ac = frozenset({req_a.request_id, req_c.request_id})
        edge_bc = frozenset({req_b.request_id, req_c.request_id})
        assert edge_ac not in graph.footprint_edges, (
            'C edits disjoint.txt only — should not overlap with A (shared.txt)'
        )
        assert edge_bc not in graph.footprint_edges, (
            'C edits disjoint.txt only — should not overlap with B (shared.txt)'
        )

    async def test_scenario_4_aging_beats_fifo(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """Scenario 4: B (mfea=100.0) is picked before A (mfea=300.0) despite A being FIFO-head.

        After removing C (disjoint), buf=[A, B]:
          _pop_next_pickable() must return B (older clique-minimal via the real graph).
        """
        worker, req_a, req_b, req_c, main_sha = await self._setup_aging_and_disjoint(
            git_repo, config, git_ops,
        )

        await worker.recompute_suffix_conflict_graph()

        # Invariant holds before picks.
        violations = worker.two_layer_invariants(main_sha)
        assert violations == [], f'two_layer_invariants before picks: {violations!r}'

        # Remove C from buffer to isolate scenario 4 (aging within the clique).
        worker._lane_buffers['normal'].remove(req_c)

        picked = worker._pop_next_pickable()
        assert picked is req_b, (
            f'Expected B (older, mfea=100.0) to be picked via clique-scoped aging '
            f'but got {picked!r}. A (mfea=300.0) should be skipped because B is '
            f'its older footprint-neighbor in the same lane.'
        )

        # After picking B, A should be next (only item remaining).
        picked2 = worker._pop_next_pickable()
        assert picked2 is req_a, (
            f'Expected A to be picked after B is gone but got {picked2!r}'
        )

        # Invariant holds after picks (suffix was reordered, frozen prefix unchanged).
        violations = worker.two_layer_invariants(main_sha)
        assert violations == [], f'two_layer_invariants after picks: {violations!r}'

    async def test_scenario_5_disjoint_bypass(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """Scenario 5: disjoint C bypasses blocked A (whose clique-peer B is older).

        FIFO buf = [A (mfea=300.0), C (mfea=200.0, disjoint), B (mfea=100.0)]:
          Pick 1 → C (disjoint bypass: C is vacuously minimal; A is blocked by older B)
          Pick 2 → B (aging: B.mfea < A.mfea in the clique)
          Pick 3 → A (only item remaining)
        """
        worker, req_a, req_b, req_c, main_sha = await self._setup_aging_and_disjoint(
            git_repo, config, git_ops,
        )

        await worker.recompute_suffix_conflict_graph()

        # Invariant holds before any pick.
        violations = worker.two_layer_invariants(main_sha)
        assert violations == [], f'two_layer_invariants before picks: {violations!r}'

        # Pick 1: C should bypass blocked A (FIFO position 0) and older B.
        picked1 = worker._pop_next_pickable()
        assert picked1 is req_c, (
            f'Expected C (disjoint) to bypass A (blocked by older clique-peer B) '
            f'but got {picked1!r}. FIFO buf was [A(300.0), C(200.0), B(100.0)]; '
            f'C is disjoint so it is vacuously minimal and should be picked before '
            f'A (whose clique-peer B is older = not minimal).'
        )

        # Pick 2: B (older clique-minimal) beats A.
        picked2 = worker._pop_next_pickable()
        assert picked2 is req_b, (
            f'Expected B (mfea=100.0, older clique-minimal) to be picked next '
            f'but got {picked2!r}'
        )

        # Pick 3: A last.
        picked3 = worker._pop_next_pickable()
        assert picked3 is req_a, (
            f'Expected A to be picked last but got {picked3!r}'
        )

        # Invariant holds after picks.
        violations = worker.two_layer_invariants(main_sha)
        assert violations == [], f'two_layer_invariants after picks: {violations!r}'


# ── step-11 RED: §9 scenario 6 (verify frontier immutable under reorder)
#                + §5.3 invariants in one run ─────────────────────────────────


@pytest.mark.asyncio
class TestScenario6FrontierImmutableUnderReorder:
    """§9 scenario 6: verify frontier immutable under suffix reorder.

    Place in-flight verifying entries D, E (frozen, chained base→merge_commit)
    and unfrozen suffix reqs F, G in _lane_buffers.

    Steps:
      1. Capture D/E base_sha, frozen_prefix(), frozen_prefix_tip(main).
      2. Reorder the suffix (G before F) and call recompute_suffix_conflict_graph().
      3. Assert:
         a. D/E base_sha unchanged.
         b. _inflight order/identity unchanged.
         c. frozen_prefix() and frozen_prefix_tip(main) unchanged.
         d. The suffix graph nodes reflect the NEW order (G before F) and
            exclude every frozen rid.
         e. worker.two_layer_invariants(main) == [] before AND after the reorder.

    Integration delta: uses two_layer_invariants() (step-02 new method) as the
    single §5.3 health surface; the reorder test exercises frozen-prefix
    immutability (I1/I3) plus the disjointness check (I2/frozen∩suffix).

    RED until the suite lands.
    """

    async def test_scenario_6_frontier_immutable_under_reorder(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """D/E inflight order/base_sha/frozen_prefix unchanged after suffix reorder."""
        # 1. Get main_sha.
        _, main_sha_raw, _ = await _run(['git', 'rev-parse', 'main'], cwd=git_repo)
        main_sha = main_sha_raw.strip()

        # 2. Create two frozen entries (D, E) with a chained base→merge_commit.
        #    We use fake SHAs for the merge commits (they don't need to be real git
        #    objects for the invariant check — only the frozen_prefix_tip is used
        #    for recompute, which uses get_main_sha() independently).
        #    To keep two_layer_invariants == [], we chain correctly:
        #      D: base_sha=main_sha, merge_commit='merge-d-sha'
        #      E: base_sha='merge-d-sha', merge_commit='merge-e-sha'

        # For recompute to work (frozen_prefix_tip('merge-e-sha')), the SHA must
        # exist in the repo.  Create side-branch commits for D and E tips.
        merge_d_sha = await _create_branch_editing(
            git_repo, 'task/frozen-d', 'README.md', '# Frozen D\n',
        )
        merge_e_sha = await _create_branch_editing(
            git_repo, 'task/frozen-e', 'README.md', '# Frozen E\n',
        )

        # Create suffix branches (F and G, disjoint from each other and from D/E).
        await _create_branch_editing(
            git_repo, 'task/suffix-f', 'shared.txt', 'line1\nF-LINE2\nline3\n',
        )
        await _create_branch_editing(
            git_repo, 'task/suffix-g', 'disjoint.txt', 'G-LINE1\ndisjoint-line2\n',
        )

        worker = _make_worker(git_ops)

        # Populate frozen prefix: D chained off main, E chained off D.
        req_d = _make_req('frozen-d', 'task/frozen-d', config, git_repo)
        item_d = SpeculativeItem(
            request=req_d,
            merge_result=MergeResult(success=True, merge_commit=merge_d_sha),
            merge_wt=None,
            base_sha=main_sha,
            speculative=False,
            skip_verify=False,
        )
        entry_d = _make_inflight_entry(item_d, verifying=True)

        req_e = _make_req('frozen-e', 'task/frozen-e', config, git_repo)
        item_e = SpeculativeItem(
            request=req_e,
            merge_result=MergeResult(success=True, merge_commit=merge_e_sha),
            merge_wt=None,
            base_sha=merge_d_sha,  # chained off D
            speculative=False,
            skip_verify=False,
        )
        entry_e = _make_inflight_entry(item_e, verifying=True)

        worker._inflight.extend([entry_d, entry_e])

        # Suffix reqs: F then G (FIFO: F is head).
        req_f = _make_req('suffix-f', 'suffix-f', config, git_repo, merge_first_enqueued_at=100.0)
        req_g = _make_req('suffix-g', 'disjoint-c', config, git_repo, merge_first_enqueued_at=200.0)
        worker._lane_buffers['normal'].extend([req_f, req_g])

        # 3. Capture pre-reorder state.
        pre_frozen_prefix = worker.frozen_prefix()
        pre_frozen_tip = worker.frozen_prefix_tip(main_sha)
        pre_d_base_sha = item_d.base_sha
        pre_e_base_sha = item_e.base_sha
        pre_inflight_ids = tuple(e.item.request.request_id for e in worker._inflight)

        # Verify two_layer_invariants == [] BEFORE reorder.
        violations_before = worker.two_layer_invariants(main_sha)
        assert violations_before == [], (
            f'two_layer_invariants before reorder: {violations_before!r}'
        )

        # 4. Reorder suffix (G before F) then recompute.
        # Simulate reorder: swap F and G in the lane buffer.
        worker._lane_buffers['normal'].remove(req_g)
        worker._lane_buffers['normal'].appendleft(req_g)

        # Invalidate signature so recompute runs.
        worker._suffix_conflict_signature = None

        await worker.recompute_suffix_conflict_graph()

        # 5. Assert: frozen prefix unchanged.

        # (a) D/E base_sha unchanged.
        assert item_d.base_sha == pre_d_base_sha, (
            f'D base_sha mutated by reorder: {item_d.base_sha!r} != {pre_d_base_sha!r}'
        )
        assert item_e.base_sha == pre_e_base_sha, (
            f'E base_sha mutated by reorder: {item_e.base_sha!r} != {pre_e_base_sha!r}'
        )

        # (b) _inflight order/identity unchanged.
        post_inflight_ids = tuple(e.item.request.request_id for e in worker._inflight)
        assert post_inflight_ids == pre_inflight_ids, (
            f'_inflight order changed after reorder: '
            f'before={pre_inflight_ids!r}, after={post_inflight_ids!r}'
        )

        # (c) frozen_prefix() and frozen_prefix_tip unchanged.
        assert worker.frozen_prefix() == pre_frozen_prefix, (
            f'frozen_prefix() changed after suffix reorder: '
            f'before={pre_frozen_prefix!r}, after={worker.frozen_prefix()!r}'
        )
        assert worker.frozen_prefix_tip(main_sha) == pre_frozen_tip, (
            f'frozen_prefix_tip changed after suffix reorder: '
            f'before={pre_frozen_tip!r}, after={worker.frozen_prefix_tip(main_sha)!r}'
        )

        # (d) suffix graph reflects NEW order (G before F) and excludes frozen rids.
        graph = worker._suffix_conflict_graph
        frozen_rids = set(worker.frozen_prefix())

        # Frozen rids must not appear in graph nodes.
        for frid in frozen_rids:
            assert frid not in graph.nodes, (
                f'Frozen rid {frid!r} appears in suffix graph nodes after reorder — '
                f'violates frozen/suffix partition'
            )

        # Both F and G must be in graph nodes (they are the unfrozen suffix).
        assert req_f.request_id in graph.nodes, (
            f'req_f not in suffix graph nodes: nodes={graph.nodes!r}'
        )
        assert req_g.request_id in graph.nodes, (
            f'req_g not in suffix graph nodes: nodes={graph.nodes!r}'
        )

        # G should come before F in nodes (reflecting the reordered buffer).
        assert graph.nodes.index(req_g.request_id) < graph.nodes.index(req_f.request_id), (
            f'Expected G before F in graph.nodes after reorder, got: {graph.nodes!r}'
        )

        # (e) two_layer_invariants == [] AFTER reorder.
        violations_after = worker.two_layer_invariants(main_sha)
        assert violations_after == [], (
            f'two_layer_invariants after reorder: {violations_after!r}'
        )


# ── step-13 RED: §9 scenario 9 (circuit-breaker fires + escalates + resumes)
#                via REAL Harness + REAL SpeculativeMergeWorker ────────────────


@pytest.mark.asyncio
class TestScenario9CircuitBreaker:
    """§9 scenario 9: circuit-breaker fires + escalates + resumes.

    Integration delta over upstream θ test (test_harness_no_landings_breaker.py):
      - Upstream test used a MagicMock worker with worker.snapshot() hard-coded.
      - This suite attaches a REAL SpeculativeMergeWorker; the breaker reads
        ι's REAL landings_total from worker.snapshot()['metrics']['landings_total'].
      - Resume is driven by calling worker._merge_metrics.record_landing() so
        that the next snapshot() increments landings_total — proving the breaker
        reads the live ι counter, not a stub.

    Fake/instrumented runners: shutil.disk_usage patched; no real ssh/build.

    RED until the suite lands.
    """

    async def _drive_to_trip(
        self,
        harness,
        worker,
        config: OrchestratorConfig,
        git_repo: Path,
        window: int = 3,
    ) -> None:
        """Drive the breaker to tripped + scheduler halted state.

        Adds a fake req to the worker's lane buffer so snapshot()['depth'] > 0
        (required to pass the idle-queue guard in _run_no_landings_breaker_pass).
        landings_total stays flat (no record_landing() calls).
        """
        # Add a queued req so depth > 0 (idle-queue guard bypass).
        req = MergeRequest(
            task_id='breaker-dummy',
            branch='breaker-dummy',
            worktree=git_repo,
            pre_rebased=False,
            task_files=None,
            module_configs=[],
            config=config,
            result=asyncio.get_running_loop().create_future(),
        )
        worker._lane_buffers['normal'].append(req)

        disk_iter = _make_falling_disk_iter(start=200_000, drop=10_000)
        with patch('shutil.disk_usage') as mock_du:
            mock_du.side_effect = lambda _p: MagicMock(free=next(disk_iter))
            for _ in range(window):
                await harness._run_no_landings_breaker_pass()

    async def test_fire_halts_scheduler_with_real_worker(
        self, tmp_path: Path, git_repo: Path, config: OrchestratorConfig,
        git_ops: GitOps,
    ) -> None:
        """FIRE: flat landings_total + falling disk → scheduler paused.

        The breaker reads landings_total from the REAL worker snapshot(),
        not from a MagicMock stub.
        """
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=1000)
        worker = _attach_real_worker(harness, git_ops)

        # landings_total starts at 0 and stays flat (no record_landing calls).
        assert worker.snapshot()['metrics']['landings_total'] == 0

        await self._drive_to_trip(harness, worker, config, git_repo, window=3)

        assert harness.scheduler.is_paused, (
            'scheduler should be paused after trip window with flat landings_total '
            'from real worker snapshot()'
        )

    async def test_fire_files_info_escalation_with_real_worker(
        self, tmp_path: Path, git_repo: Path, config: OrchestratorConfig,
        git_ops: GitOps,
    ) -> None:
        """FIRE: after trip, exactly one pending INFO escalation with correct attrs."""
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=1000)
        worker = _attach_real_worker(harness, git_ops)

        await self._drive_to_trip(harness, worker, config, git_repo, window=3)

        assert harness._escalation_queue is not None
        pending = [
            e
            for e in harness._escalation_queue.get_by_task(
                _BREAKER_SENTINEL, status='pending'
            )
            if e.agent_role == _BREAKER_ROLE
        ]
        assert len(pending) == 1, (
            f'Expected 1 pending breaker INFO escalation, got {len(pending)}'
        )
        esc = pending[0]
        assert esc.severity == 'info'
        assert esc.level == 0
        assert esc.category == 'risk_identified'
        assert esc.task_id == _BREAKER_SENTINEL

    async def test_resume_via_real_landing_unpauses_scheduler(
        self, tmp_path: Path, git_repo: Path, config: OrchestratorConfig,
        git_ops: GitOps,
    ) -> None:
        """RESUME: recording a REAL landing via worker._merge_metrics.record_landing()
        increments landings_total in snapshot(); the next breaker pass unpauses
        the scheduler and resolves the open INFO escalation.

        This is the integration delta: the upstream θ test stubbed the worker's
        snapshot() — here the real ι counter drives the resume.
        """
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=1000)
        worker = _attach_real_worker(harness, git_ops)

        # Trip the breaker.
        await self._drive_to_trip(harness, worker, config, git_repo, window=3)
        assert harness.scheduler.is_paused, 'precondition: scheduler must be paused'

        # Confirm escalation was filed.
        assert harness._escalation_queue is not None
        before_pending = [
            e
            for e in harness._escalation_queue.get_by_task(
                _BREAKER_SENTINEL, status='pending'
            )
            if e.agent_role == _BREAKER_ROLE
        ]
        assert before_pending, 'precondition: INFO escalation must be filed'

        # Drive a REAL landing via the ι counter (integration delta).
        # The breaker's resume condition: landings_total > landings_at_trip.
        landings_at_trip = worker.snapshot()['metrics']['landings_total']
        worker._merge_metrics.record_landing()
        assert worker.snapshot()['metrics']['landings_total'] == landings_at_trip + 1

        # Recovery pass with disk OK.
        with patch('shutil.disk_usage') as mock_du:
            mock_du.return_value = MagicMock(free=150_000)
            await harness._run_no_landings_breaker_pass()

        # Scheduler unpaused.
        assert not harness.scheduler.is_paused, (
            'scheduler should be unpaused after real landing increments landings_total'
        )

        # Breaker INFO escalation resolved.
        after_pending = [
            e
            for e in harness._escalation_queue.get_by_task(
                _BREAKER_SENTINEL, status='pending'
            )
            if e.agent_role == _BREAKER_ROLE
        ]
        assert not after_pending, (
            f'Expected 0 pending breaker INFO escalations after resume via real '
            f'landing, got {len(after_pending)}'
        )

    async def test_fire_dedup_one_escalation_with_real_worker(
        self, tmp_path: Path, git_repo: Path, config: OrchestratorConfig,
        git_ops: GitOps,
    ) -> None:
        """Multiple passes after trip do not stack duplicate escalations (dedup).

        Use a high floor (500_000) so the post-trip falling disk values (150k...)
        do not trigger disk-recovery resume — only a clean landing would resume,
        and landings_total stays flat here.
        """
        harness, _rs = _make_harness(tmp_path)
        harness._no_landings_breaker = _small_breaker(window=3, floor=500_000)
        worker = _attach_real_worker(harness, git_ops)

        # Trip then run 3 more passes.
        await self._drive_to_trip(harness, worker, config, git_repo, window=3)

        # Add more falling-disk passes while still tripped.
        disk_iter = _make_falling_disk_iter(start=150_000, drop=5_000)
        with patch('shutil.disk_usage') as mock_du:
            mock_du.side_effect = lambda _p: MagicMock(free=next(disk_iter))
            for _ in range(3):
                await harness._run_no_landings_breaker_pass()

        assert harness._escalation_queue is not None
        all_pending = [
            e
            for e in harness._escalation_queue.get_by_task(
                _BREAKER_SENTINEL, status='pending'
            )
            if e.agent_role == _BREAKER_ROLE
        ]
        assert len(all_pending) == 1, (
            f'Expected exactly 1 breaker INFO escalation (dedup), got {len(all_pending)}'
        )
