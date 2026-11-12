"""Tests for the frozen-prefix / verify-frontier invariant (task ε=1890).

Covers:
  prereq-1     — scaffolding: fixtures + helpers (this file, no test logic)
  step-01 RED  — frozen_prefix() + unfrozen_suffix() accessors
  step-03 RED  — frozen_prefix_tip(main_sha)
  step-05 RED  — check_frozen_prefix_invariant(main_sha) → list[str]
  step-07 RED  — snapshot() exposes additive 'frozen_prefix' key
  step-09 RED  — HEADLINE property: in-flight immutable under suffix reorder;
                 recompute excludes frozen rids (exclusion filter)
  step-11 RED  — the dispatch-time verify-base guard (task 5302: now
                 _verify_base_divergence_at_dispatch, the §5.3 detector)
  3206 step-1  — the PRD §5.3 RE-MERGE CARVE-OUT at the dispatch guard: a
                 recovery re-merge that fell to main is exempt (task 8524,
                 plans/merge-lane-policy-prd.md D1); the carve-out is narrow
                 (an ordinary mismatch still warns exactly once).
  3206 step-5  — the surviving WARNING keeps its CALL-TIME CONTEXT (task_id,
                 request_id, both shas, verify_depth), which task 1999 records
                 as available on no other §5.3 surface.  Data only: the
                 message's wording is deliberately NOT pinned here.
  3206 step-7d — FIVE-SITE PARITY: every `_remerge` return path stamps the
                 recovery marker.

CHARACTERIZATION-TEST NOTE (task 3206). The §5.3 verify-base rule is a
DETECTOR and is never promoted to hard control-flow enforcement (esc-3206-3;
PRD §5.3 and §10; measured precision as an enforcement predicate was 0/4).
The 3206 tests above are the executable half of that fence: they are expected
to pass immediately, and their purpose is to make a future SILENT flip to
enforcement impossible — each fails the moment the guard changes control
flow. If they go red, re-derive the evidence in PRD §5.3 before "fixing" them.

INVARIANT (§5.3): frozen prefix = {items currently verifying} ∪ {landed}.
An item once it enters verify is immutable (never reordered / re-based out
from under an in-flight verify).  A verify may only start against a base that
is the tip of the frozen prefix (no verify against a speculative-only base).
The unfrozen suffix may be reordered / inserted freely; any reorder recomputes
the merge-tree for the affected suffix ONLY.

Mapping:
  {verifying} = self._inflight entries with verify_task is not None
                + self._finalizing_head when phase in {'verifying',
                  'gate_reverify', 'finalizing'}
  {landed}    = main (its SHA reflects landed items)
  unfrozen    = the intake's lane buffers (pick order high→normal, FIFO)
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import pytest
from _merge_lane_fakes import (
    FakeClock,
    FakeVerifier,
    RecordingEscalations,
    lane_scene_config,
    make_lane,
)

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_lane.remerge_base import RemergeClaim
from orchestrator.merge_lane.types import (
    InflightStatus,
    NoHost,
    QueuedBranch,
    RemergeBase,
    RemergeReason,
    SpecPermit,
    VerifyBaseDivergence,
)
from orchestrator.merge_lane.worker import (
    DecidedItem,
    InflightEntry,
    ItemLifecycleState,
    MergeOutcome,
    MergeRequest,
    RealMergeItem,
    SpeculativeItem,
    SpeculativeMergeWorker,
)

# ── Module-level sentinel for verify_task in pure unit tests ─────────────────
#
# _frozen_inflight_entries() only checks `e.verify_task is not None`, so any
# non-None object doubles as a verifying-entry sentinel in tests that do not
# need a real asyncio.Task (no event-loop required for the accessor methods
# themselves).  The type annotation is unenforced at runtime.
_SENTINEL_VERIFY_TASK = object()  # noqa: PD901 (not a dataframe)


# ── Fixtures (mirrored from test_merge_queue_conflict_graph.py) ───────────────


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
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
    return lane_scene_config(git_repo, git_config)


# ── Module-level helpers ──────────────────────────────────────────────────────


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    lane: Literal['normal', 'high'] = 'normal',
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
    """
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane=lane,
    )


def _make_worker(git_ops: GitOps) -> SpeculativeMergeWorker:
    """Build a lane on the fakes for unit tests (no harness wiring).

    The verifier and clock are injected rather than left to the production
    adapters, so no row here shells out or reads the wall clock -- which is
    also what keeps a snapshot's derived ``age_secs`` stable across the
    before/after comparisons below.
    """
    return make_lane(git_ops, verifier=FakeVerifier(), clock=FakeClock())


def _make_fake_item(
    task_id: str,
    *,
    base_sha: str = 'aaaa0000',
    merge_commit: str | None = 'bbbb1111',
    config: OrchestratorConfig,
    git_repo: Path,
) -> tuple[MergeRequest, SpeculativeItem]:
    """Build a (MergeRequest, SpeculativeItem) pair from fake SHAs.

    Does NOT create real git branches — suitable for pure-unit tests that
    only exercise the accessor methods (frozen_prefix, frozen_prefix_tip,
    check_frozen_prefix_invariant) without needing actual git operations.
    Must be called from an async context (for MergeRequest.result future).
    """
    req = _make_req(task_id, f'task/{task_id}', config, git_repo)
    item: SpeculativeItem
    if merge_commit is not None:
        item = RealMergeItem(
            request=req,
            merge_result=MergeResult(success=True, merge_commit=merge_commit),
            merge_wt=Path('/fake/merge-wt'),
            base_sha=base_sha,
            speculative=False,
        )
    else:
        item = DecidedItem(
            request=req,
            immediate_outcome=MergeOutcome('conflict'),
            base_sha=base_sha,
            speculative=False,
        )
    return req, item


def _make_inflight_entry(
    item: SpeculativeItem,
    *,
    verifying: bool = True,
) -> InflightEntry:
    """Build an InflightEntry for unit tests.

    verifying=True  → verify_task set to _SENTINEL_VERIFY_TASK (any non-None
                       object; frozen_prefix() only checks `is not None`).
    verifying=False → verify_task=None (passthrough entry; excluded from the
                       frozen prefix by §5.3 definition).
    """
    return InflightEntry(
        item=item,
        lease=None,
        verify_task=_SENTINEL_VERIFY_TASK if verifying else None,  # type: ignore[arg-type]
        merge_wt=None,
        was_speculative=False,
    )


# ── The lane-state seam, and the public observations that replace it ─────────
#
# §5.3's accessors are pure functions of the lane's in-flight region and its
# lane buffers, and the rows below need regions the live pipeline cannot
# produce: a broken base chain, one rid in BOTH regions, a passthrough entry
# carrying no merge commit, a finalize head parked mid-phase.  The lane
# exposes no public seam for admitting such a region, so the five helpers
# below are the ONLY places in this file that name a lane internal -- one per
# internal, so the whole file moves the day a seam lands (task 5032 filed an
# escalate_info naming them, for PRD task δ).  Everything a row OBSERVES goes
# through the public accessors and snapshot() instead.


def _seed_frozen(worker: SpeculativeMergeWorker, *entries: InflightEntry) -> None:
    """Admit *entries* to the lane's in-flight region, oldest first."""
    worker._inflight.extend(entries)


def _seed_suffix(
    worker: SpeculativeMergeWorker,
    *reqs: MergeRequest,
    lane: Literal['normal', 'high'] = 'normal',
    replace: bool = False,
) -> None:
    """Admit *reqs* to *lane*'s buffer; *replace* reorders the lane wholesale."""
    intake = worker._intake
    kept = () if replace else intake.buffered(lane)
    intake.replace_lane(lane, (*kept, *reqs))


def _seed_finalizing_head(
    worker: SpeculativeMergeWorker,
    entry: InflightEntry,
    *,
    phase: ItemLifecycleState = ItemLifecycleState.VERIFYING,
    then: ItemLifecycleState | None = None,
) -> None:
    """Park *entry* as the finalize head at *phase*, then move it to *then*.

    The head is derived from the lifecycle registry, so registering the entry
    is what puts it at the submission-order head of the frozen prefix.
    """
    worker._register_item(entry, initial=phase)
    if then is not None:
        worker._note_transition(entry.item.request.request_id, phase, then)


def _guard_warn(
    worker: SpeculativeMergeWorker, item: SpeculativeItem, main_sha: str,
) -> VerifyBaseDivergence | None:
    """Call the §5.3 dispatch-time verify-base detector directly."""
    return worker._verify_base_divergence_at_dispatch(item, main_sha)


async def _dispatch(
    worker: SpeculativeMergeWorker,
    git_ops: GitOps,
    item: SpeculativeItem,
    get_main_sha: Callable[..., Awaitable[str]],
) -> InflightEntry | NoHost:
    """Drive *item* through the lane's dispatch step while git's main read is
    *get_main_sha*.  The verify the dispatch launches is a no-op, so a row
    observes only what dispatch itself decided."""
    import unittest.mock as mock

    async def _noop_verify(_item, _lease, **_kwargs):
        return None

    with (
        mock.patch.object(git_ops, 'get_main_sha', side_effect=get_main_sha),
        mock.patch.object(worker, '_run_inflight_verify', _noop_verify),
    ):
        return await worker._dispatch_item(item)


def _assert_divergence(
    result: VerifyBaseDivergence | None,
    item: SpeculativeItem,
    expected_tip: str,
    verify_depth: int,
) -> None:
    """*result* records *item* dispatched off *expected_tip* at *verify_depth*."""
    assert result is not None, 'expected a VerifyBaseDivergence, got None'
    assert (
        result.request_id, result.base_sha, result.expected_tip, result.verify_depth,
    ) == (item.request.request_id, item.base_sha, expected_tip, verify_depth)


_INFLIGHT_STATES = frozenset(
    {'dispatching', 'verifying', 'gate_reverify', 'finalizing', 'passthrough'},
)


def _inflight_region(worker: SpeculativeMergeWorker) -> list[dict]:
    """The in-flight region as the lane publishes it, submission order first.

    The public twin of the private deque: what ``snapshot()`` reports for
    every entry past the queue, which is what an immutability assertion
    actually needs to pin (membership, order and each entry's published
    state), rather than the identity of the objects behind them.
    """
    return [
        entry for entry in worker.snapshot()['entries']
        if entry['state'] in _INFLIGHT_STATES
    ]


def _suffix_graph_nodes(worker: SpeculativeMergeWorker) -> list[str]:
    """The suffix-conflict graph's nodes in pick order, from ``snapshot()``."""
    return worker.snapshot()['suffix_conflict_graph']['nodes']


# ── step-01 RED: frozen_prefix() + unfrozen_suffix() ─────────────────────────


@pytest.mark.asyncio
class TestFrozenAndUnfrozenAccessors:
    """frozen_prefix() and unfrozen_suffix() partition the pipeline state.

    RED until step-02 GREEN adds the methods to SpeculativeMergeWorker.
    """

    async def test_bare_worker_frozen_prefix_empty(
        self, git_ops: GitOps,
    ) -> None:
        """A bare worker has no in-flight entries → frozen_prefix() == ()."""
        worker = _make_worker(git_ops)
        assert worker.frozen_prefix() == ()

    async def test_bare_worker_unfrozen_suffix_empty(
        self, git_ops: GitOps,
    ) -> None:
        """A bare worker has no lane-buffer items → unfrozen_suffix() == ()."""
        worker = _make_worker(git_ops)
        assert worker.unfrozen_suffix() == ()

    async def test_two_verifying_inflight_in_frozen_prefix(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Two verifying InflightEntry items → frozen_prefix() == (rid_a, rid_b)."""
        worker = _make_worker(git_ops)
        _, item_a = _make_fake_item('t-a', base_sha='aaa0', merge_commit='aaa1',
                                    config=config, git_repo=git_repo)
        _, item_b = _make_fake_item('t-b', base_sha='aaa1', merge_commit='aaa2',
                                    config=config, git_repo=git_repo)
        entry_a = _make_inflight_entry(item_a, verifying=True)
        entry_b = _make_inflight_entry(item_b, verifying=True)
        _seed_frozen(worker, entry_a)
        _seed_frozen(worker, entry_b)

        fp = worker.frozen_prefix()
        assert fp == (item_a.request.request_id, item_b.request.request_id)

    async def test_passthrough_excluded_from_frozen_prefix(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A passthrough entry (verify_task=None) is EXCLUDED from frozen_prefix()."""
        worker = _make_worker(git_ops)
        _, item_a = _make_fake_item('t-a', base_sha='aaa0', merge_commit='aaa1',
                                    config=config, git_repo=git_repo)
        _, item_pass = _make_fake_item('t-pass', base_sha='aaa1', merge_commit=None,
                                       config=config, git_repo=git_repo)
        entry_a = _make_inflight_entry(item_a, verifying=True)
        entry_pass = _make_inflight_entry(item_pass, verifying=False)  # passthrough
        _seed_frozen(worker, entry_a)
        _seed_frozen(worker, entry_pass)

        fp = worker.frozen_prefix()
        assert item_a.request.request_id in fp
        assert item_pass.request.request_id not in fp

    async def test_lane_buffers_appear_in_unfrozen_suffix(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Requests in the lane buffers appear in unfrozen_suffix(), high before normal."""
        worker = _make_worker(git_ops)
        req_hi = _make_req('t-hi', 'task/t-hi', config, git_repo, lane='high')
        req_n1 = _make_req('t-n1', 'task/t-n1', config, git_repo)
        req_n2 = _make_req('t-n2', 'task/t-n2', config, git_repo)
        _seed_suffix(worker, req_hi, lane='high')
        _seed_suffix(worker, req_n1)
        _seed_suffix(worker, req_n2)

        us = worker.unfrozen_suffix()
        # high lane before normal
        hi_idx = us.index(req_hi.request_id)
        n1_idx = us.index(req_n1.request_id)
        n2_idx = us.index(req_n2.request_id)
        assert hi_idx < n1_idx < n2_idx

    async def test_frozen_and_unfrozen_are_disjoint(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """frozen_prefix() and unfrozen_suffix() must be disjoint sets."""
        worker = _make_worker(git_ops)
        _, item_a = _make_fake_item('t-a', base_sha='aaa0', merge_commit='aaa1',
                                    config=config, git_repo=git_repo)
        _, item_b = _make_fake_item('t-b', config=config, git_repo=git_repo)
        entry_a = _make_inflight_entry(item_a, verifying=True)
        _seed_frozen(worker, entry_a)
        req_b = item_b.request
        _seed_suffix(worker, req_b)

        fp = set(worker.frozen_prefix())
        us = set(worker.unfrozen_suffix())
        assert fp & us == set(), f'frozen ∩ unfrozen should be empty, got {fp & us}'


# ── step-03 RED: frozen_prefix_tip(main_sha) ─────────────────────────────────


@pytest.mark.asyncio
class TestFrozenPrefixTip:
    """frozen_prefix_tip(main_sha) returns the newest frozen merge_commit.

    RED until step-04 GREEN adds _newest_frozen_commit + frozen_prefix_tip.
    """

    async def test_empty_inflight_returns_main_sha(
        self, git_ops: GitOps,
    ) -> None:
        """Empty frozen prefix → frozen_prefix_tip returns main_sha unchanged."""
        worker = _make_worker(git_ops)
        assert worker.frozen_prefix_tip('mainsha0') == 'mainsha0'

    async def test_single_verifying_entry_returns_merge_commit(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """One verifying inflight entry → tip is its merge_commit."""
        worker = _make_worker(git_ops)
        _, item = _make_fake_item('t-x', base_sha='main0', merge_commit='commitX',
                                  config=config, git_repo=git_repo)
        entry = _make_inflight_entry(item, verifying=True)
        _seed_frozen(worker, entry)
        assert worker.frozen_prefix_tip('main0') == 'commitX'

    async def test_two_stacked_entries_returns_newest(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Two stacked entries: tip is the second (newest/top-of-stack) merge_commit."""
        worker = _make_worker(git_ops)
        _, item_d = _make_fake_item('t-d', base_sha='main0', merge_commit='commitX',
                                    config=config, git_repo=git_repo)
        _, item_e = _make_fake_item('t-e', base_sha='commitX', merge_commit='commitY',
                                    config=config, git_repo=git_repo)
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))
        _seed_frozen(worker, _make_inflight_entry(item_e, verifying=True))

        # Deque order: D (left/head) → E (right/tail); tip is E (last/newest)
        assert worker.frozen_prefix_tip('main0') == 'commitY'

    async def test_passthrough_entry_skipped(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Entry with no merge_commit (passthrough/conflict) skipped in tip walk."""
        worker = _make_worker(git_ops)
        # A verifying entry with a real commit
        _, item_v = _make_fake_item('t-v', base_sha='main0', merge_commit='commitV',
                                    config=config, git_repo=git_repo)
        # A passthrough entry (merge_result=None → no merge_commit)
        _, item_p = _make_fake_item('t-p', base_sha='main0', merge_commit=None,
                                    config=config, git_repo=git_repo)
        _seed_frozen(worker, _make_inflight_entry(item_v, verifying=True))
        _seed_frozen(worker, _make_inflight_entry(item_p, verifying=False))

        # The passthrough is not frozen; tip is still the verifying entry's commit
        assert worker.frozen_prefix_tip('main0') == 'commitV'


# ── step-05 RED: check_frozen_prefix_invariant(main_sha) ─────────────────────


@pytest.mark.asyncio
class TestCheckFrozenPrefixInvariant:
    """check_frozen_prefix_invariant returns [] when healthy, violations when broken.

    RED until step-06 GREEN implements the method.
    """

    async def test_well_formed_stack_returns_empty(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """D.base==main_sha, D.merge_commit==Dc; E.base==Dc, E.merge_commit==Ec → []."""
        worker = _make_worker(git_ops)
        main_sha = 'main000'
        _, item_d = _make_fake_item('t-d', base_sha=main_sha, merge_commit='commitDc',
                                    config=config, git_repo=git_repo)
        _, item_e = _make_fake_item('t-e', base_sha='commitDc', merge_commit='commitEc',
                                    config=config, git_repo=git_repo)
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))
        _seed_frozen(worker, _make_inflight_entry(item_e, verifying=True))

        violations = worker.check_frozen_prefix_invariant(main_sha)
        assert violations == [], f'expected no violations, got: {violations}'

    async def test_broken_base_chain_returns_violation(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """E.base_sha is a speculative-only SHA (not D's merge_commit) → non-empty violations."""
        worker = _make_worker(git_ops)
        main_sha = 'main000'
        _, item_d = _make_fake_item('t-d', base_sha=main_sha, merge_commit='commitDc',
                                    config=config, git_repo=git_repo)
        # E's base_sha is wrong: it should be commitDc but is some other SHA
        _, item_e = _make_fake_item('t-e', base_sha='wrong_speculative_sha',
                                    merge_commit='commitEc',
                                    config=config, git_repo=git_repo)
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))
        _seed_frozen(worker, _make_inflight_entry(item_e, verifying=True))

        violations = worker.check_frozen_prefix_invariant(main_sha)
        assert len(violations) > 0, 'expected a base-chain violation'
        # The violation message should name the offending rid
        assert item_e.request.request_id in ' '.join(violations)

    async def test_lone_head_off_main_is_intact(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A lone head whose base is not main_sha, and not a recorded dead
        commit, breaks nothing: INV-3 is chain-intact, not base == main (task 5302).
        """
        worker = _make_worker(git_ops)
        _, item_d = _make_fake_item('t-d', base_sha='older-main', merge_commit='commitDc',
                                    config=config, git_repo=git_repo)
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))

        assert worker.check_frozen_prefix_invariant('main000') == []

    async def test_disjointness_violation(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A request_id in BOTH frozen prefix and unfrozen suffix → disjointness violation."""
        worker = _make_worker(git_ops)
        main_sha = 'main000'
        req, item_d = _make_fake_item('t-d', base_sha=main_sha, merge_commit='commitDc',
                                      config=config, git_repo=git_repo)
        entry = _make_inflight_entry(item_d, verifying=True)
        _seed_frozen(worker, entry)
        # Same request also in lane buffer — violates frozen/suffix disjointness
        _seed_suffix(worker, req)

        violations = worker.check_frozen_prefix_invariant(main_sha)
        assert len(violations) > 0, 'expected a disjointness violation'
        assert req.request_id in ' '.join(violations)


# ── step-09 RED: HEADLINE property test ──────────────────────────────────────


@pytest.mark.asyncio
class TestFrozenPrefixPropertyTest:
    """HEADLINE (§5.3): in-flight verify immutable under suffix reorder/insert;
    recompute excludes frozen rids from the suffix-conflict-graph nodes.

    RED until step-10 GREEN adds the frozen-rid exclusion filter to
    recompute_suffix_conflict_graph().  The immutability assertion (a) already
    holds (emergent from δ's lane-buffers-only recompute); only the exclusion
    assertion (b) fails RED.
    """

    async def test_inflight_immutable_under_suffix_reorder(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """(a) In-flight entries are unchanged after suffix reorder + recompute.

        Setup: D/E in _inflight (verifying); F/G in the normal lane buffer.
        Capture state, reverse F/G, recompute, assert:
          - D/E base_sha UNCHANGED
          - D/E inflight identity/order UNCHANGED
          - frozen_prefix() and frozen_prefix_tip(main) UNCHANGED
          - check_frozen_prefix_invariant(main) == []
          - suffix_conflict_graph.nodes equals NEW order (G before F)
          - No frozen rid (D/E) in suffix_conflict_graph.nodes
        """
        main_sha = 'fake-main-sha'
        worker = _make_worker(git_ops)

        # Build D and E as fake inflight items (pure unit — no real git merges
        # needed for the immutability assertion; base_sha / merge_commit are
        # constructed to form a well-chained stack)
        _, item_d = _make_fake_item('t-d', base_sha=main_sha, merge_commit='sha-Dc',
                                    config=config, git_repo=git_repo)
        _, item_e = _make_fake_item('t-e', base_sha='sha-Dc', merge_commit='sha-Ec',
                                    config=config, git_repo=git_repo)
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))
        _seed_frozen(worker, _make_inflight_entry(item_e, verifying=True))

        # F and G are lane-buffer items (simple MergeRequest objects; fake branches
        # that recompute_suffix_conflict_graph will fail-open on for edges/footprint,
        # but their request_ids still appear in nodes which is what we assert).
        req_f = _make_req('t-f', 'task/t-f', config, git_repo)
        req_g = _make_req('t-g', 'task/t-g', config, git_repo)
        _seed_suffix(worker, req_f)
        _seed_suffix(worker, req_g)

        # ── Capture pre-reorder state ─────────────────────────────────────────
        pre_d_base = item_d.base_sha
        pre_e_base = item_e.base_sha
        pre_frozen = worker.frozen_prefix()
        pre_tip = worker.frozen_prefix_tip(main_sha)
        pre_region = _inflight_region(worker)

        # ── Reorder: F, G → G, F ─────────────────────────────────────────────
        _seed_suffix(worker, req_g, req_f, replace=True)
        await worker.recompute_suffix_conflict_graph()

        # ── Assert inflight immutability ──────────────────────────────────────
        assert item_d.base_sha == pre_d_base, 'D.base_sha mutated by reorder'
        assert item_e.base_sha == pre_e_base, 'E.base_sha mutated by reorder'
        assert _inflight_region(worker) == pre_region, (
            'in-flight region changed: membership, order or a published state'
        )
        assert worker.frozen_prefix() == pre_frozen, 'frozen_prefix() changed'
        assert worker.frozen_prefix_tip(main_sha) == pre_tip, 'frozen_prefix_tip changed'
        assert worker.check_frozen_prefix_invariant(main_sha) == [], (
            'invariant broken after reorder'
        )

        # ── Assert suffix graph reflects new order (G before F) ──────────────
        g_nodes = _suffix_graph_nodes(worker)
        assert req_g.request_id in g_nodes, 'G missing from suffix graph nodes'
        assert req_f.request_id in g_nodes, 'F missing from suffix graph nodes'
        assert g_nodes.index(req_g.request_id) < g_nodes.index(req_f.request_id), (
            'suffix graph does not reflect new G-before-F order'
        )

        # ── Assert no frozen rid appears in suffix graph nodes ────────────────
        # NOTE: this loop is trivially satisfied here because D/E are only ever
        # in _inflight and are never pushed into the lane buffers, so they could
        # not appear in the suffix graph regardless of the exclusion filter.
        # The genuine RED→GREEN coverage for the exclusion filter lives in
        # test_frozen_rid_excluded_from_suffix_graph below.
        frozen_rids = set(worker.frozen_prefix())
        for frozen_rid in frozen_rids:
            assert frozen_rid not in g_nodes, (
                f'frozen rid {frozen_rid!r} appeared in suffix_conflict_graph.nodes'
            )

    async def test_frozen_rid_excluded_from_suffix_graph(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """(b) Exclusion filter: a frozen rid pushed into the lane buffers is excluded.

        This is the genuine RED→GREEN delta: before step-10 adds the exclusion
        filter, a frozen request that is also in the lane buffers appears in
        suffix_conflict_graph.nodes.  After step-10, it is excluded.

        Setup: item D in _inflight (verifying); also D's request pushed into
        the lane buffers.  Recompute → D's rid must NOT appear in nodes.
        """
        main_sha = 'fake-main-sha'
        worker = _make_worker(git_ops)

        req_d, item_d = _make_fake_item('t-d-excl', base_sha=main_sha,
                                        merge_commit='sha-Dc-excl',
                                        config=config, git_repo=git_repo)
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))

        # Intentionally push the SAME request into the lane buffers (simulates the
        # bug this filter guards against: a frozen rid leaking into the suffix).
        _seed_suffix(worker, req_d)

        # Also add an innocent item so the suffix is non-empty (gives
        # recompute something to build without the early-empty-suffix return)
        req_innocent = _make_req('t-innocent', 'task/t-innocent', config, git_repo)
        _seed_suffix(worker, req_innocent)

        await worker.recompute_suffix_conflict_graph()

        g_nodes = _suffix_graph_nodes(worker)
        assert req_d.request_id not in g_nodes, (
            f'frozen rid {req_d.request_id!r} appeared in suffix graph nodes '
            f'(exclusion filter missing)'
        )
        assert req_innocent.request_id in g_nodes, 'innocent item disappeared'


# ── step-07 RED: snapshot() exposes additive 'frozen_prefix' key ─────────────


@pytest.mark.asyncio
class TestSnapshotFrozenPrefixKey:
    """snapshot() grows an additive 'frozen_prefix' key; existing keys unchanged.

    RED until step-08 GREEN adds the key to snapshot().
    """

    async def test_bare_worker_frozen_prefix_key_shape(
        self, git_ops: GitOps,
    ) -> None:
        """Bare worker snapshot() contains frozen_prefix with empty/zero values."""
        worker = _make_worker(git_ops)
        snap = worker.snapshot()
        assert 'frozen_prefix' in snap, "snapshot() is missing 'frozen_prefix' key"
        fp = snap['frozen_prefix']
        assert fp == {
            'request_ids': [],
            'tip_merge_commit': None,
            'verify_depth': 0,
        }, f"unexpected frozen_prefix shape: {fp}"

    async def test_two_verifying_inflight_snapshot_shape(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Two verifying in-flight items → frozen_prefix reflects them."""
        worker = _make_worker(git_ops)
        _, item_d = _make_fake_item('t-d', base_sha='main0', merge_commit='Dc',
                                    config=config, git_repo=git_repo)
        _, item_e = _make_fake_item('t-e', base_sha='Dc', merge_commit='Ec',
                                    config=config, git_repo=git_repo)
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))
        _seed_frozen(worker, _make_inflight_entry(item_e, verifying=True))

        snap = worker.snapshot()
        fp = snap['frozen_prefix']
        assert fp['request_ids'] == [
            item_d.request.request_id,
            item_e.request.request_id,
        ]
        assert fp['tip_merge_commit'] == 'Ec'
        assert fp['verify_depth'] == 2

    async def test_snapshot_backward_compat_all_prior_keys_present(
        self, git_ops: GitOps,
    ) -> None:
        """All pre-existing snapshot keys are still present (backward-compatible)."""
        worker = _make_worker(git_ops)
        snap = worker.snapshot()
        required_keys = {
            'entries',
            'depth',
            'head_of_line',
            'verify_in_progress',
            'occupancy',
            'is_wip_halted',
            'halt_owner_esc_id',
            'suffix_conflict_graph',  # δ/1889
            'metrics',               # ι/1894
            'frozen_prefix',         # ε/1890 — the new key
        }
        missing = required_keys - snap.keys()
        assert not missing, f'snapshot() is missing keys: {missing}'

    async def test_metrics_key_unaffected_by_frozen_prefix(
        self, git_ops: GitOps,
    ) -> None:
        """'metrics' key still carries its whole documented shape."""
        worker = _make_worker(git_ops)
        snap = worker.snapshot()
        assert 'metrics' in snap
        assert snap['metrics'].keys() == {
            'retries_per_landing',
            'drift_at_detection',
            'landings_total',
            'retries_total',
        }, f'unexpected metrics shape: {snap["metrics"]}'


# ── step-11 RED: the dispatch-time verify-base guard ──────────────────────────


@pytest.mark.asyncio
class TestWarnIfVerifyBaseNotFrozenTip:
    """The dispatch guard warns when verify base != frozen tip, and returns the
    :class:`VerifyBaseDivergence` record (task 5302); on a match it returns None.

    It never mutates lane state and never raises.
    """

    async def test_speculative_only_base_logs_warning(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Candidate base_sha != frozen_prefix_tip → exactly one WARNING logged."""
        import logging

        worker = _make_worker(git_ops)

        # Frozen predecessor: merge_commit == 'sha-Dc'
        _, item_d = _make_fake_item(
            't-d', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))

        # Candidate item: base_sha is a speculative-only SHA (not the frozen tip)
        req_e, item_e = _make_fake_item(
            't-e', base_sha='wrong-speculative-base', merge_commit='sha-Ec',
            config=config, git_repo=git_repo,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            result = _guard_warn(worker, item_e, 'main0')

        _assert_divergence(result, item_e, 'sha-Dc', 1)

        # Exactly one WARNING mentioning the rid and 'frozen'
        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
        ]
        assert len(warnings) == 1, (
            f'expected exactly 1 WARNING, got {len(warnings)}: {[r.message for r in warnings]}'
        )
        warning_text = warnings[0].getMessage()
        assert req_e.request_id in warning_text or item_e.request.request_id in warning_text, (
            f'WARNING does not mention the candidate rid: {warning_text!r}'
        )
        assert 'frozen' in warning_text.lower(), (
            f'WARNING does not mention "frozen": {warning_text!r}'
        )

    async def test_base_matches_frozen_tip_no_warning(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Candidate base_sha == frozen_prefix_tip(main) → NO warning logged."""
        import logging

        worker = _make_worker(git_ops)

        # Frozen predecessor: merge_commit == 'sha-Dc'
        _, item_d = _make_fake_item(
            't-d', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))

        # Candidate item: base_sha == frozen_prefix_tip (correct)
        _, item_e = _make_fake_item(
            't-e', base_sha='sha-Dc', merge_commit='sha-Ec',
            config=config, git_repo=git_repo,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            result = _guard_warn(worker, item_e, 'main0')

        assert result is None
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0, (
            f'expected no WARNINGs, got: {[r.message for r in warnings]}'
        )

    async def test_empty_frozen_prefix_base_matches_main_no_warning(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Empty frozen prefix, candidate base_sha == main_sha → NO warning."""
        import logging

        worker = _make_worker(git_ops)
        # No in-flight entries → frozen_prefix_tip('main0') == 'main0'

        _, item_d = _make_fake_item(
            't-d', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            result = _guard_warn(worker, item_d, 'main0')

        assert result is None
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0, (
            f'expected no WARNINGs, got: {[r.message for r in warnings]}'
        )

    async def test_method_does_not_mutate_state(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
    ) -> None:
        """The dispatch guard never mutates worker state."""
        worker = _make_worker(git_ops)
        _, item_d = _make_fake_item(
            't-d', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))

        # Candidate with wrong base (triggers warning path)
        _, item_e = _make_fake_item(
            't-e', base_sha='wrong-base', merge_commit='sha-Ec',
            config=config, git_repo=git_repo,
        )

        # Capture state before
        pre_frozen = worker.frozen_prefix()
        pre_region = _inflight_region(worker)
        pre_tip = worker.frozen_prefix_tip('main0')

        _guard_warn(worker, item_e, 'main0')

        # State must be entirely unchanged
        assert worker.frozen_prefix() == pre_frozen
        assert _inflight_region(worker) == pre_region
        assert worker.frozen_prefix_tip('main0') == pre_tip


# ── task 3206 step-1 RED: §5.3 re-merge carve-out at the DISPATCH guard ──────


def _mark_recovery(item: SpeculativeItem) -> SpeculativeItem:
    """Stamp *item* as a recovery re-merge that fell to main, the anchor
    ``_remerge`` gives it (plans/merge-lane-policy-prd.md D1)."""
    return dataclasses.replace(item, remerge_base=RemergeBase(None, RemergeReason.NO_TIP))


@pytest.mark.asyncio
class TestRemergeCarveOutAtDispatchGuard:
    """A recovery re-merge that fell to main is exempt from the dispatch-time
    guard, and nothing else is (plans/merge-lane-policy-prd.md D1)."""

    async def test_a_fell_to_main_item_suppresses_warning(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """CARVE-OUT: a recovery item with base != frozen tip logs NO warning."""
        import logging

        worker = _make_worker(git_ops)

        # A frozen predecessor exists, so frozen_prefix_tip('main0') == 'sha-Dc'
        # — i.e. the guard would otherwise fire on the candidate below.
        _, item_d = _make_fake_item(
            't-d', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))

        # A recovery re-merge that fell to main (here 'main0'), not onto the
        # frozen tip 'sha-Dc'.
        _, item_r = _make_fake_item(
            't-recovery', base_sha='main0', merge_commit='sha-Rc',
            config=config, git_repo=git_repo,
        )
        item_r = _mark_recovery(item_r)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            result = _guard_warn(worker, item_r, 'main0')

        assert result is None
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 0, (
            f'§5.3 re-merge carve-out (task 3206): a recovery item must not warn, '
            f'got: {[r.getMessage() for r in warnings]}'
        )

    async def test_non_recovery_item_still_warns(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """CONTROL: the carve-out is narrow — an ordinary item still warns once.

        Same base/tip mismatch as the carve-out test above, minus the marker.
        Guards against the carve-out swallowing genuine class-(d) violations.
        """
        import logging

        worker = _make_worker(git_ops)

        _, item_d = _make_fake_item(
            't-d', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))

        _, item_e = _make_fake_item(
            't-e', base_sha='main0', merge_commit='sha-Ec',
            config=config, git_repo=git_repo,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            result = _guard_warn(worker, item_e, 'main0')

        _assert_divergence(result, item_e, 'sha-Dc', 1)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, (
            f'expected exactly 1 WARNING for a non-recovery mismatch, got '
            f'{len(warnings)}: {[r.getMessage() for r in warnings]}'
        )

    async def test_healthy_base_never_warns_regardless_of_marker(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """CONTROL: base == frozen tip → no warning, marker or not (unchanged path)."""
        import logging

        worker = _make_worker(git_ops)

        _, item_d = _make_fake_item(
            't-d', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))

        for marked in (False, True):
            caplog.clear()
            _, item_e = _make_fake_item(
                f't-e-{marked}', base_sha='sha-Dc', merge_commit='sha-Ec',
                config=config, git_repo=git_repo,
            )
            if marked:
                item_e = _mark_recovery(item_e)

            with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
                assert _guard_warn(worker, item_e, 'main0') is None

            warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
            assert len(warnings) == 0, (
                f'healthy base (marked={marked}) must never warn, got: '
                f'{[r.getMessage() for r in warnings]}'
            )


# ── task 3206: the surviving WARNING keeps its call-time context ─────────────


@pytest.mark.asyncio
class TestGuardWarningCarriesCallTimeContext:
    """The §5.3 WARNING must keep its call-time context (task 3206).

    Once the re-merge carve-out removes class (b), the only benign class left
    is the class-(a) PHANTOM FROZEN TIP: a stranded finalize head makes
    `frozen_prefix_tip()` return a dead merge commit, so a dispatch whose base
    is perfectly correct still mismatches.  Triaging that from the log needs
    the ids and both shas — which is what this class pins.

    SCOPE — DATA, NOT PROSE.  Two sibling tests here previously asserted the
    message's WORDING too (that it says 'stale'/'stranded'/'phantom', cites
    3082, says 'advisory', cites 3206 / §5.3).  They were deleted on review:
    they pinned cosmetic detail — a task number, an adjective — inside an
    ~800-character log paragraph, went red on any harmless reword, and covered
    no failure mode not already covered elsewhere.  Do not re-add them, and do
    not reintroduce them as looser substring matching.

    That narrative is NOT gone, it simply lives where it belongs: the
    `logger.warning(...)` message itself, the `_verify_base_divergence_at_dispatch`
    docstring, and PRD §5.3 — all of which still carry it.  The enforcement
    fence they were reaching for is held by CONTROL-FLOW assertions that go red
    the moment the guard refuses a dispatch: `TestDispatchItemGuardWiring` and
    `TestRemergeReturnPathParity` below, plus
    `TestAdvisoryContractRegressionPins` (dispatch returns a live InflightEntry,
    verify task launched, host lease not released — including the measured
    phantom-tip 5687 / 5830 shape) in test_merge_queue_verify_base_invariant.py.
    """

    async def test_warning_carries_call_time_context(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """(1) task_id, request_id, actual base, expected tip and verify_depth.

        The WARNING and the ``VerifyBaseDivergence`` record carry the same
        call-time context (task 5302).  The record, in
        ``snapshot()['verify_base_divergences']``, is the durable copy; the
        WARNING is the journal's, so a reword must not drop it either.

        Every needle is DERIVED from the fixture's own request/item/worker
        state rather than re-typed as a short literal, and verify_depth is
        pinned as the RENDERED PAIR ``verify_depth=1``.  A bare ``'1'`` would
        be satisfied vacuously — the message body itself contains 'ε=1890',
        '§5.3' and 'esc-3206-3' — so it would pin the label while letting
        the %d argument be dropped or reordered against the other three.
        """
        import logging

        worker = _make_worker(git_ops)

        _, item_d = _make_fake_item(
            't-d', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))

        req_e, item_e = _make_fake_item(
            't-e', base_sha='live-main-tip', merge_commit='sha-Ec',
            config=config, git_repo=git_repo,
        )

        # Ground the expected verify_depth in the FIXTURE (exactly one frozen
        # entry), not in the implementation, so the rendered pair asserted
        # below is a genuine value pin.
        assert len(worker.frozen_prefix()) == 1
        expected_tip = worker.frozen_prefix_tip('main0')

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            _guard_warn(worker, item_e, 'main0')

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        text = warnings[0].getMessage()

        for needle, what in (
            (item_e.request.task_id, 'task_id'),
            (req_e.request_id, 'request_id'),
            (item_e.base_sha, 'actual base'),
            (expected_tip, 'expected frozen tip'),
            ('verify_depth=1', 'verify_depth label AND value'),
        ):
            assert needle in text, (
                f'WARNING lost its {what} ({needle!r}): the journal copy of the '
                f'context the VerifyBaseDivergence record carries. Got: {text!r}'
            )


# ── Amendment: _finalizing_head branch coverage (review suggestion) ───────────


@pytest.mark.asyncio
class TestFinalizingHeadInFrozenPrefix:
    """_finalizing_head is included in the frozen prefix when its phase is
    verifying/gate_reverify/finalizing and it is not vacated (task 5302), and
    excluded otherwise.

    These tests exercise the _frozen_inflight_entries() branch for
    self._finalizing_head that was untested by the original step-01/02 tests.

    The _finalizing_head (set by _finalize_inflight) is the submission-order
    head of the pipeline.  When it holds a qualifying phase it appears FIRST
    in frozen_prefix() — older than all _inflight entries.
    """

    async def test_verifying_finalizing_head_appears_first(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """_finalizing_head with phase='verifying' appears FIRST in frozen_prefix().

        Submission order: finalizing_head < _inflight head.  frozen_prefix()
        must list the finalizing_head rid before any _inflight rid.
        """
        worker = _make_worker(git_ops)

        _, item_d = _make_fake_item('t-d', base_sha='main0', merge_commit='sha-Dc',
                                    config=config, git_repo=git_repo)
        _, item_e = _make_fake_item('t-e', base_sha='sha-Dc', merge_commit='sha-Ec',
                                    config=config, git_repo=git_repo)
        entry_d = _make_inflight_entry(item_d, verifying=True)  # phase='verifying'
        entry_e = _make_inflight_entry(item_e, verifying=True)

        _seed_finalizing_head(worker, entry_d)
        _seed_frozen(worker, entry_e)

        fp = worker.frozen_prefix()
        assert len(fp) == 2, f'expected 2 frozen entries, got {fp!r}'
        assert fp[0] == item_d.request.request_id, (
            f'expected _finalizing_head rid first in frozen_prefix(), got {fp!r}'
        )
        assert fp[1] == item_e.request.request_id

    async def test_verifying_finalizing_head_tip_is_inflight_newest(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """frozen_prefix_tip() returns the newest _inflight commit, not the
        finalizing_head commit.

        _frozen_inflight_entries() returns [finalizing_head, ...inflight].
        _newest_frozen_commit() iterates in REVERSE → inflight entries are
        newer than the finalizing_head.  The tip must be the newest inflight
        commit (sha-Ec), proving the finalizing_head is treated as the OLDEST.
        """
        worker = _make_worker(git_ops)

        _, item_d = _make_fake_item('t-d', base_sha='main0', merge_commit='sha-Dc',
                                    config=config, git_repo=git_repo)
        _, item_e = _make_fake_item('t-e', base_sha='sha-Dc', merge_commit='sha-Ec',
                                    config=config, git_repo=git_repo)
        entry_d = _make_inflight_entry(item_d, verifying=True)
        entry_e = _make_inflight_entry(item_e, verifying=True)

        _seed_finalizing_head(worker, entry_d)
        _seed_frozen(worker, entry_e)

        tip = worker.frozen_prefix_tip('main0')
        assert tip == 'sha-Ec', (
            f"expected newest inflight commit 'sha-Ec', got {tip!r}; "
            "the finalizing_head (sha-Dc) is the OLDEST entry and must not be "
            "returned as the tip when a newer _inflight entry exists"
        )

    async def test_passthrough_phase_finalizing_head_excluded(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """_finalizing_head with a non-qualifying phase is excluded from frozen_prefix().

        Only phases in {'verifying', 'gate_reverify', 'finalizing'} qualify.
        A 'passthrough'-phase _finalizing_head must not appear in the frozen prefix
        or affect frozen_prefix_tip().
        """
        worker = _make_worker(git_ops)

        _, item_p = _make_fake_item('t-p', base_sha='main0', merge_commit='sha-Pc',
                                    config=config, git_repo=git_repo)
        entry_p = _make_inflight_entry(item_p, verifying=True)
        _seed_finalizing_head(
            worker, entry_p, phase=ItemLifecycleState.DISPATCHING,  # non-qualifying
        )

        fp = worker.frozen_prefix()
        assert fp == (), (
            f"passthrough-phase _finalizing_head must not appear in frozen_prefix(), "
            f"got {fp!r}"
        )
        # frozen_prefix_tip falls back to main_sha when no qualifying entries exist
        assert worker.frozen_prefix_tip('main-sentinel') == 'main-sentinel'

    async def test_gate_reverify_finalizing_head_included(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """_finalizing_head with phase='gate_reverify' is included in frozen_prefix().

        gate_reverify is the third qualifying phase alongside verifying and finalizing.
        """
        worker = _make_worker(git_ops)

        _, item_g = _make_fake_item('t-g', base_sha='main0', merge_commit='sha-Gc',
                                    config=config, git_repo=git_repo)
        entry_g = _make_inflight_entry(item_g, verifying=True)
        # gate_reverify: a qualifying phase distinct from 'verifying'
        _seed_finalizing_head(
            worker, entry_g, then=ItemLifecycleState.GATE_REVERIFY,
        )

        fp = worker.frozen_prefix()
        assert item_g.request.request_id in fp, (
            f"gate_reverify-phase _finalizing_head missing from frozen_prefix(): {fp!r}"
        )

    @pytest.mark.parametrize('status', [InflightStatus.REQUEUED, InflightStatus.DROPPED])
    @pytest.mark.parametrize(
        'then', [None, ItemLifecycleState.FINALIZING], ids=['verifying', 'finalizing'],
    )
    async def test_vacated_finalizing_head_is_not_frozen(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        status: InflightStatus,
        then: ItemLifecycleState | None,
    ) -> None:
        """A vacated finalize head is not frozen, whatever its phase (task 5302).

        Its verify requeued or dropped the request and released the lease
        (task 4582); only the head-of-line finalize remains.  Counting it is
        the PRD §5.3 class-(a) phantom frozen tip.
        """
        worker = _make_worker(git_ops)
        _, item_h = _make_fake_item('t-h', base_sha='main0', merge_commit='phantom-c',
                                    config=config, git_repo=git_repo)
        entry_h = dataclasses.replace(_make_inflight_entry(item_h), status=status)
        _seed_finalizing_head(worker, entry_h, then=then)

        assert worker.frozen_prefix() == ()
        assert worker.snapshot()['frozen_prefix']['request_ids'] == []
        assert worker.frozen_prefix_tip('main0') == 'main0'

    async def test_vacated_finalizing_head_yields_to_live_deque_entry(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Behind a vacated finalize head, the live deque entry is the whole prefix."""
        worker = _make_worker(git_ops)
        _, item_h = _make_fake_item('t-h', base_sha='main0', merge_commit='phantom-c',
                                    config=config, git_repo=git_repo)
        _, item_e = _make_fake_item('t-e', base_sha='main0', merge_commit='sha-Ec',
                                    config=config, git_repo=git_repo)
        entry_h = dataclasses.replace(
            _make_inflight_entry(item_h), status=InflightStatus.REQUEUED,
        )
        _seed_finalizing_head(worker, entry_h)
        _seed_frozen(worker, _make_inflight_entry(item_e))

        assert worker.frozen_prefix() == (item_e.request.request_id,)
        assert worker.frozen_prefix_tip('main0') == 'sha-Ec'

    async def test_phantom_finalize_head_does_not_poison_the_dispatch_guard(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Replay of PRD §5.3 class (a): 2026-08-08, tasks 5687/5830.

        A requeued head's dead merge commit stayed the frozen tip, so a later
        dispatch on correct main warned.  With the head vacated the tip is
        main, and the same dispatch is silent.
        """
        import logging

        worker = _make_worker(git_ops)
        _, item_h = _make_fake_item('t-5687', base_sha='main0', merge_commit='phantom-c',
                                    config=config, git_repo=git_repo)
        entry_h = dataclasses.replace(
            _make_inflight_entry(item_h), status=InflightStatus.REQUEUED,
        )
        _seed_finalizing_head(worker, entry_h, then=ItemLifecycleState.FINALIZING)
        _, item_c = _make_fake_item('t-5830', base_sha='main0', merge_commit='sha-Cc',
                                    config=config, git_repo=git_repo)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            _guard_warn(worker, item_c, 'main0')

        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == []

    async def test_live_finalizing_head_is_still_frozen(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """CONTROL: a finalize head that was not vacated stays frozen."""
        worker = _make_worker(git_ops)
        _, item_h = _make_fake_item('t-h', base_sha='main0', merge_commit='sha-Hc',
                                    config=config, git_repo=git_repo)
        _seed_finalizing_head(
            worker, _make_inflight_entry(item_h), then=ItemLifecycleState.FINALIZING,
        )

        assert worker.frozen_prefix() == (item_h.request.request_id,)
        assert worker.frozen_prefix_tip('main0') == 'sha-Hc'


# ── Amendment: _dispatch_item guard wiring tests (review suggestion) ──────────


@pytest.mark.asyncio
class TestDispatchItemGuardWiring:
    """The §5.3 guard in _dispatch_item is fail-open and fires on the real dispatch path.

    These tests anchor the integration seam: the guard is wired into
    _dispatch_item (not just directly callable), and a get_main_sha() error
    in the guard is swallowed without blocking verify dispatch.

    Setup: items use speculative=True so Mechanism 2 does NOT call get_main_sha
    in the chain-remerge block — isolating the guard's own try/except as the
    only code path that touches get_main_sha.
    """

    async def test_dispatch_guard_fail_open_on_get_main_sha_error(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """get_main_sha() raising in the guard does not block verify dispatch (fail-open).

        The try/except around the guard in _dispatch_item must swallow any
        Exception so a transient git error never prevents a verify from launching.
        Confirms the integration wiring: _dispatch_item returns a verifying
        InflightEntry even when the guard's git call fails.
        """
        worker = _make_worker(git_ops)

        # Build an item that passes all _dispatch_item early-bailout checks:
        #   - not abandoned (future not cancelled)
        #   - not operator-halted (_operator_halt not set by default)
        #   - no immediate_outcome
        #   - speculative=True → Mechanism 2 won't call get_main_sha in remerge block
        #   - merge_wt=git_repo → LocalRunner factory assertion passes
        req, item = _make_fake_item(
            't-dispatch-fail-open', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )
        item = dataclasses.replace(item, speculative=True, merge_wt=git_repo)

        async def _raise_on_get_main(*_args, **_kwargs):
            raise RuntimeError('simulated transient git error')

        result = await _dispatch(worker, git_ops, item, _raise_on_get_main)

        # Fail-open: dispatch must return a verifying InflightEntry.
        assert isinstance(result, InflightEntry), (
            f'_dispatch_item returned {result!r}; expected a verifying InflightEntry '
            'even when get_main_sha() raises inside the §5.3 guard (fail-open)'
        )
        assert result.verify_task is not None, (
            'dispatch returned an entry with no verify task: the guard refused '
            'the dispatch instead of failing open'
        )

        # Clean up the background verify task to prevent event-loop warnings.
        if result.verify_task is not None:
            result.verify_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await result.verify_task

    async def test_dispatch_guard_warns_on_non_tip_base(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Guard fires a WARNING on the real _dispatch_item path when base_sha ≠ frozen tip.

        Proves the guard is wired into _dispatch_item, not just directly callable.
        Setup: frozen predecessor with merge_commit='sha-Dc'; dispatch item
        with base_sha='wrong-base' → the wired guard must log a WARNING naming
        the rid, confirming the integration seam is active.
        """
        import logging

        worker = _make_worker(git_ops)

        # Frozen predecessor in _inflight: its merge_commit becomes the frozen tip.
        _, item_d = _make_fake_item(
            't-d-frozen', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )
        _seed_frozen(worker, _make_inflight_entry(item_d, verifying=True))

        # Dispatch item: base_sha='wrong-base' does not match frozen tip 'sha-Dc'.
        req_e, item_e = _make_fake_item(
            't-e-dispatch', base_sha='wrong-base', merge_commit='sha-Ec',
            config=config, git_repo=git_repo,
        )
        item_e = dataclasses.replace(item_e, speculative=True, merge_wt=git_repo)

        # get_main_sha returns a valid sha so the guard call proceeds rather
        # than raising.  The mismatch 'wrong-base' vs frozen tip 'sha-Dc'
        # triggers the warning.
        async def _return_main(*_args, **_kwargs) -> str:
            return 'fake-main'

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            result = await _dispatch(worker, git_ops, item_e, _return_main)

        # Guard must have fired at least one WARNING naming the dispatched rid.
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) >= 1, (
            f'expected ≥1 WARNING from the _dispatch_item guard (integration seam), '
            f'got {len(warnings)}: {[r.message for r in warnings]}'
        )
        warning_text = ' '.join(r.getMessage() for r in warnings)
        assert (
            req_e.request_id in warning_text
            or item_e.request.request_id in warning_text
        ), (
            f'WARNING does not mention the dispatched rid {req_e.request_id!r}: '
            f'{warning_text!r}'
        )
        assert 'frozen' in warning_text.lower(), (
            f'WARNING does not mention "frozen": {warning_text!r}'
        )

        # Dispatch must still succeed — the guard is purely observational.
        assert isinstance(result, InflightEntry), 'dispatch must succeed even when guard warns'
        assert result.verify_task is not None, (
            'guard is observational: it must not withhold the verify task'
        )

        # Clean up the background verify task.
        if result.verify_task is not None:
            result.verify_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await result.verify_task


# ── task 5302: the dispatch-time verify-base detector key ────────────────────


_DIVERGENCE_CATEGORY = 'merge_verify_base_divergence'


async def _dispatch_on_main(
    worker: SpeculativeMergeWorker,
    git_ops: GitOps,
    item: SpeculativeItem,
    main_sha: str,
) -> InflightEntry:
    """Dispatch *item* for real-verify while main reads *main_sha*, then admit
    the entry to the in-flight region as the verifier loop does.

    The verify itself is a no-op (:func:`_dispatch`).
    """
    async def _return_main(*_args, **_kwargs) -> str:
        return main_sha

    entry = await _dispatch(worker, git_ops, item, _return_main)
    assert isinstance(entry, InflightEntry) and entry.verify_task is not None, 'dispatch was refused'
    _seed_frozen(worker, entry)
    return entry


class _FailingEscalations(RecordingEscalations):
    """An escalation queue whose every filing raises."""

    def submit(self, escalation: Any) -> None:
        raise RuntimeError('simulated escalation-queue outage')


async def _settle(*entries: InflightEntry) -> None:
    """Await each entry's no-op verify task so the loop closes clean."""
    for entry in entries:
        if entry.verify_task is not None:
            entry.verify_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await entry.verify_task


@pytest.mark.asyncio
class TestVerifyBaseDetectorKey:
    """A divergent dispatch is published and escalated, never refused (task 5302).

    The record computed at dispatch is the one ``snapshot()`` reports under
    ``verify_base_divergences``, and its first hit files one L1.  Items are
    speculative, as in :class:`TestDispatchItemGuardWiring`, so Mechanism 2
    reads no main of its own.
    """

    def _lane(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        escalations: RecordingEscalations | None = None,
    ) -> tuple[SpeculativeMergeWorker, RecordingEscalations]:
        """A lane whose frozen prefix is one verifying head with merge commit 'sha-Dc'."""
        escalations = RecordingEscalations() if escalations is None else escalations
        worker = make_lane(
            git_ops, verifier=FakeVerifier(), clock=FakeClock(), escalation_queue=escalations,
        )
        _, item_d = _make_fake_item('t-d-frozen', base_sha='fake-main', merge_commit='sha-Dc',
                                    config=config, git_repo=git_repo)
        _seed_frozen(worker, _make_inflight_entry(item_d))
        return worker, escalations

    def _candidate(
        self, task_id: str, base_sha: str, config: OrchestratorConfig, git_repo: Path,
    ) -> SpeculativeItem:
        _, item = _make_fake_item(task_id, base_sha=base_sha, merge_commit=f'sha-{task_id}',
                                  config=config, git_repo=git_repo)
        return dataclasses.replace(item, speculative=True, merge_wt=git_repo)

    async def test_divergent_dispatch_is_published_and_escalated(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        worker, escalations = self._lane(git_ops, config, git_repo)
        item_e = self._candidate('t-e', 'wrong-base', config, git_repo)

        entry = await _dispatch_on_main(worker, git_ops, item_e, 'fake-main')

        published = worker.snapshot()['verify_base_divergences']
        assert [d['request_id'] for d in published] == [item_e.request.request_id]
        assert published[0]['expected_tip'] == 'sha-Dc'
        filed = [e for e in escalations.filed if e.category == _DIVERGENCE_CATEGORY]
        assert len(filed) == 1 and filed[0].level == 1
        await _settle(entry)

    async def test_a_failed_filing_neither_blocks_dispatch_nor_goes_silent(
        self,
        git_ops: GitOps,
        config: OrchestratorConfig,
        git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        worker, _ = self._lane(git_ops, config, git_repo, _FailingEscalations())
        item_e = self._candidate('t-e', 'wrong-base', config, git_repo)

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
            entry = await _dispatch_on_main(worker, git_ops, item_e, 'fake-main')

        published = worker.snapshot()['verify_base_divergences']
        assert [d['request_id'] for d in published] == [item_e.request.request_id]
        failed = [r for r in caplog.records if 'failed to file' in r.getMessage()]
        assert len(failed) == 1 and failed[0].levelno == logging.WARNING, failed
        assert failed[0].exc_info is not None, 'the filing failure lost its traceback'
        await _settle(entry)

    async def test_divergent_successor_reads_as_a_chain_link_break_not_a_verify_base_string(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        worker, _ = self._lane(git_ops, config, git_repo)
        item_e = self._candidate('t-e', 'wrong-base', config, git_repo)

        entry = await _dispatch_on_main(worker, git_ops, item_e, 'fake-main')

        assert worker.snapshot()['two_layer_invariants'] == [
            f'frozen-prefix base-chain broken at {item_e.request.request_id}: '
            f"expected base 'sha-Dc' but item has 'wrong-base'"
        ]
        await _settle(entry)

    async def test_open_l1_dedups_a_second_divergence(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """The second detection goes through the guard directly: the lane's one
        local host is still leased to the first verify, so a second full
        dispatch would be parked before reaching the guard."""
        worker, escalations = self._lane(git_ops, config, git_repo)
        first = await _dispatch_on_main(
            worker, git_ops, self._candidate('t-e', 'wrong-base', config, git_repo), 'fake-main',
        )
        escalations.open_it()

        item_f = self._candidate('t-f', 'wrong-base-2', config, git_repo)
        _assert_divergence(_guard_warn(worker, item_f, 'fake-main'), item_f, 'sha-t-e', 2)

        assert len([e for e in escalations.filed if e.category == _DIVERGENCE_CATEGORY]) == 1
        await _settle(first)

    async def test_dispatch_on_the_tip_records_nothing(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        worker, escalations = self._lane(git_ops, config, git_repo)
        item_e = self._candidate('t-e', 'sha-Dc', config, git_repo)

        entry = await _dispatch_on_main(worker, git_ops, item_e, 'fake-main')

        assert worker.snapshot()['verify_base_divergences'] == []
        assert escalations.filed == []
        await _settle(entry)


# ── task 3206 step-7(d): FIVE-SITE PARITY for the _remerge recovery marker ───


class TestRemergeReturnPathParity:
    """Every item `_remerge` produces carries the claim it was given.

    `_remerge` is the SINGLE PRODUCER of recovery items for all five consumer
    paths (`_verifier_loop` head-failure cascade, `_void_and_remerge` INV-3
    void, `_finalize_inflight` RUNNER_UNAVAILABLE, and `_dispatch_item`'s
    dead-base-straggler and Mechanism-2 sites).  Its single exit stamps the
    claim's anchor as `remerge_base` and the claim's permit on whatever
    `_remerge_inner` returned, so the speculative <=> permit invariant
    (plans/merge-lane-policy-prd.md §6) holds for every present and future
    return path of the body, and the §5.3 detector reads the anchor rather
    than a separately maintained flag (task 8524).
    """

    @pytest.mark.asyncio
    async def test_remerge_stamps_the_claim_on_every_exit(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Whatever `_remerge_inner` returns, `_remerge` returns it stamped
        with the claim, for both union arms and both kinds of anchor."""
        import unittest.mock as mock

        worker = _make_worker(git_ops)
        claims = (
            RemergeClaim(RemergeBase(None, RemergeReason.NO_TIP), None),
            RemergeClaim(RemergeBase('sha-tip', RemergeReason.LIVE_TIP), SpecPermit()),
        )
        for claim in claims:
            for label, merge_commit in (('real', 'sha-Rc'), ('decided', None)):
                _, unmarked = _make_fake_item(
                    f't-recovery-{label}', base_sha='real-main',
                    merge_commit=merge_commit, config=config, git_repo=git_repo,
                )
                assert unmarked.remerge_base is None, 'fixture must start unmarked'
                anchors: list[RemergeBase] = []

                # Patch on the INSTANCE, not the class: an instance attribute
                # is not re-bound, so the defaults survive the call.
                async def _body(
                    _req: object, _mono: object, anchor: RemergeBase,
                    _item: SpeculativeItem = unmarked, _seen: list[RemergeBase] = anchors,
                ) -> SpeculativeItem:
                    _seen.append(anchor)
                    return _item

                with mock.patch.object(worker, '_remerge_inner', _body):
                    out = await worker._remerge(unmarked.request, None, claim=claim)

                assert anchors == [claim.anchor], f'{label}: the body merges onto the claim'
                assert out.remerge_base == claim.anchor, f'{label}: anchor not stamped'
                assert out.permit is claim.permit, f'{label}: permit not handed over'
                assert dataclasses.replace(out, remerge_base=None, permit=None) == unmarked, (
                    f'_remerge altered the {label} item beyond the claim'
                )

    def test_remerge_has_a_single_exit(self) -> None:
        """The stamp above is only load-bearing while `_remerge` has ONE exit.

        Deliberately the whole of the structural check: not what the exits
        construct, only that there is exactly one, so no path can bypass the
        stamp.  If you need another return in `_remerge`, put it in
        `_remerge_inner` instead.
        """
        import ast
        import inspect
        import textwrap

        from orchestrator.merge_lane.worker import SpeculativeMergeWorker

        src = textwrap.dedent(inspect.getsource(SpeculativeMergeWorker._remerge))
        returns = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Return)]

        assert len(returns) == 1, (
            f'_remerge must have exactly ONE exit -- it is the single chokepoint '
            f'that stamps the claim (remerge_base + permit, task 8524); found '
            f'{len(returns)}. Move the new return into _remerge_inner.'
        )

    @pytest.mark.asyncio
    async def test_anchor_defaults_to_none_on_both_item_variants(
        self, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """An item built without an anchor carries none: only `_remerge`
        stamps one (task 8524)."""
        _, real = _make_fake_item('t-unanchored-real', config=config, git_repo=git_repo)
        _, decided = _make_fake_item(
            't-unanchored-decided', merge_commit=None, config=config, git_repo=git_repo,
        )

        assert isinstance(real, RealMergeItem)
        assert isinstance(decided, DecidedItem)
        assert real.remerge_base is None
        assert decided.remerge_base is None

    @pytest.mark.asyncio
    async def test_the_recovery_marker_is_a_fall_to_main(
        self, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        _, item = _make_fake_item('t-mark', config=config, git_repo=git_repo)
        marked = _mark_recovery(item)
        assert marked.remerge_base == RemergeBase(None, RemergeReason.NO_TIP)
