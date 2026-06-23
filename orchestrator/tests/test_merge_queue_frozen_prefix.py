"""Tests for the frozen-prefix / verify-frontier invariant (task ε=1890).

Covers:
  prereq-1     — scaffolding: fixtures + helpers (this file, no test logic)
  step-01 RED  — frozen_prefix() + unfrozen_suffix() accessors
  step-03 RED  — frozen_prefix_tip(main_sha)
  step-05 RED  — check_frozen_prefix_invariant(main_sha) → list[str]
  step-07 RED  — snapshot() exposes additive 'frozen_prefix' key
  step-09 RED  — HEADLINE property: in-flight immutable under suffix reorder;
                 recompute excludes frozen rids (exclusion filter)
  step-11 RED  — _warn_if_verify_base_not_frozen_tip log-only guard

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
  unfrozen    = self._lane_buffers (pick order high→normal, FIFO)
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from pathlib import Path
from typing import Literal

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    InflightEntry,
    MergeRequest,
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
    return OrchestratorConfig(project_root=git_repo, git=git_config)


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
        branch=branch,
        worktree=git_repo,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane=lane,
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

    Does NOT create real git branches — suitable for pure-unit tests that
    only exercise the accessor methods (frozen_prefix, frozen_prefix_tip,
    check_frozen_prefix_invariant) without needing actual git operations.
    Must be called from an async context (for MergeRequest.result future).
    """
    req = _make_req(task_id, f'task/{task_id}', config, git_repo)
    fake_merge_result = MergeResult(
        success=True,
        merge_commit=merge_commit,
    ) if merge_commit is not None else None
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
        phase='verifying',
    )


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
        worker._inflight.append(entry_a)
        worker._inflight.append(entry_b)

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
        worker._inflight.append(entry_a)
        worker._inflight.append(entry_pass)

        fp = worker.frozen_prefix()
        assert item_a.request.request_id in fp
        assert item_pass.request.request_id not in fp

    async def test_lane_buffers_appear_in_unfrozen_suffix(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Requests in _lane_buffers appear in unfrozen_suffix(), high before normal."""
        worker = _make_worker(git_ops)
        req_hi = _make_req('t-hi', 'task/t-hi', config, git_repo, lane='high')
        req_n1 = _make_req('t-n1', 'task/t-n1', config, git_repo)
        req_n2 = _make_req('t-n2', 'task/t-n2', config, git_repo)
        worker._lane_buffers['high'].append(req_hi)
        worker._lane_buffers['normal'].append(req_n1)
        worker._lane_buffers['normal'].append(req_n2)

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
        worker._inflight.append(entry_a)
        req_b = item_b.request
        worker._lane_buffers['normal'].append(req_b)

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
        worker._inflight.append(entry)
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
        worker._inflight.append(_make_inflight_entry(item_d, verifying=True))
        worker._inflight.append(_make_inflight_entry(item_e, verifying=True))

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
        worker._inflight.append(_make_inflight_entry(item_v, verifying=True))
        worker._inflight.append(_make_inflight_entry(item_p, verifying=False))

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
        worker._inflight.append(_make_inflight_entry(item_d, verifying=True))
        worker._inflight.append(_make_inflight_entry(item_e, verifying=True))

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
        worker._inflight.append(_make_inflight_entry(item_d, verifying=True))
        worker._inflight.append(_make_inflight_entry(item_e, verifying=True))

        violations = worker.check_frozen_prefix_invariant(main_sha)
        assert len(violations) > 0, 'expected a base-chain violation'
        # The violation message should name the offending rid
        assert item_e.request.request_id in ' '.join(violations)

    async def test_disjointness_violation(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A request_id in BOTH frozen prefix and unfrozen suffix → disjointness violation."""
        worker = _make_worker(git_ops)
        main_sha = 'main000'
        req, item_d = _make_fake_item('t-d', base_sha=main_sha, merge_commit='commitDc',
                                      config=config, git_repo=git_repo)
        entry = _make_inflight_entry(item_d, verifying=True)
        worker._inflight.append(entry)
        # Same request also in lane buffer — violates frozen/suffix disjointness
        worker._lane_buffers['normal'].append(req)

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
    holds (emergent from δ's _lane_buffers-only recompute); only the exclusion
    assertion (b) fails RED.
    """

    async def test_inflight_immutable_under_suffix_reorder(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """(a) In-flight entries are unchanged after suffix reorder + recompute.

        Setup: D/E in _inflight (verifying); F/G in _lane_buffers['normal'].
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
        worker._inflight.append(_make_inflight_entry(item_d, verifying=True))
        worker._inflight.append(_make_inflight_entry(item_e, verifying=True))

        # F and G are lane-buffer items (simple MergeRequest objects; fake branches
        # that recompute_suffix_conflict_graph will fail-open on for edges/footprint,
        # but their request_ids still appear in nodes which is what we assert).
        req_f = _make_req('t-f', 'task/t-f', config, git_repo)
        req_g = _make_req('t-g', 'task/t-g', config, git_repo)
        worker._lane_buffers['normal'].append(req_f)
        worker._lane_buffers['normal'].append(req_g)

        # ── Capture pre-reorder state ─────────────────────────────────────────
        pre_d_base = item_d.base_sha
        pre_e_base = item_e.base_sha
        pre_frozen = worker.frozen_prefix()
        pre_tip = worker.frozen_prefix_tip(main_sha)
        pre_inflight = list(worker._inflight)  # shallow copy for identity check

        # ── Reorder: F, G → G, F ─────────────────────────────────────────────
        worker._lane_buffers['normal'].clear()
        worker._lane_buffers['normal'].append(req_g)
        worker._lane_buffers['normal'].append(req_f)
        await worker.recompute_suffix_conflict_graph()

        # ── Assert inflight immutability ──────────────────────────────────────
        assert item_d.base_sha == pre_d_base, 'D.base_sha mutated by reorder'
        assert item_e.base_sha == pre_e_base, 'E.base_sha mutated by reorder'
        assert list(worker._inflight) == pre_inflight, '_inflight order/identity changed'
        assert worker.frozen_prefix() == pre_frozen, 'frozen_prefix() changed'
        assert worker.frozen_prefix_tip(main_sha) == pre_tip, 'frozen_prefix_tip changed'
        assert worker.check_frozen_prefix_invariant(main_sha) == [], (
            'invariant broken after reorder'
        )

        # ── Assert suffix graph reflects new order (G before F) ──────────────
        g_after = worker._suffix_conflict_graph
        g_nodes = g_after.nodes
        assert req_g.request_id in g_nodes, 'G missing from suffix graph nodes'
        assert req_f.request_id in g_nodes, 'F missing from suffix graph nodes'
        assert g_nodes.index(req_g.request_id) < g_nodes.index(req_f.request_id), (
            'suffix graph does not reflect new G-before-F order'
        )

        # ── Assert no frozen rid appears in suffix graph nodes ────────────────
        # NOTE: this loop is trivially satisfied here because D/E are only ever
        # in _inflight and are never pushed into _lane_buffers, so they could
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
        """(b) Exclusion filter: a frozen rid pushed into _lane_buffers is excluded.

        This is the genuine RED→GREEN delta: before step-10 adds the exclusion
        filter, a frozen request that is also in _lane_buffers appears in
        suffix_conflict_graph.nodes.  After step-10, it is excluded.

        Setup: item D in _inflight (verifying); also D's request pushed into
        _lane_buffers.  Recompute → D's rid must NOT appear in nodes.
        """
        main_sha = 'fake-main-sha'
        worker = _make_worker(git_ops)

        req_d, item_d = _make_fake_item('t-d-excl', base_sha=main_sha,
                                        merge_commit='sha-Dc-excl',
                                        config=config, git_repo=git_repo)
        worker._inflight.append(_make_inflight_entry(item_d, verifying=True))

        # Intentionally push the SAME request into _lane_buffers (simulates the
        # bug this filter guards against: a frozen rid leaking into the suffix).
        worker._lane_buffers['normal'].append(req_d)

        # Also add an innocent item so the suffix is non-empty (gives
        # recompute something to build without the early-empty-suffix return)
        req_innocent = _make_req('t-innocent', 'task/t-innocent', config, git_repo)
        worker._lane_buffers['normal'].append(req_innocent)

        await worker.recompute_suffix_conflict_graph()

        g_nodes = worker._suffix_conflict_graph.nodes
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
        worker._inflight.append(_make_inflight_entry(item_d, verifying=True))
        worker._inflight.append(_make_inflight_entry(item_e, verifying=True))

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
        """'metrics' key is present and equals _merge_metrics.as_snapshot()."""
        worker = _make_worker(git_ops)
        snap = worker.snapshot()
        assert 'metrics' in snap
        assert snap['metrics'] == worker._merge_metrics.as_snapshot()


# ── step-11 RED: _warn_if_verify_base_not_frozen_tip log-only guard ───────────


@pytest.mark.asyncio
class TestWarnIfVerifyBaseNotFrozenTip:
    """_warn_if_verify_base_not_frozen_tip warns when verify base != frozen tip.

    The method is log-only: it never mutates state, never raises, returns None.

    RED until step-12 GREEN adds _warn_if_verify_base_not_frozen_tip.
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
        worker._inflight.append(_make_inflight_entry(item_d, verifying=True))

        # Candidate item: base_sha is a speculative-only SHA (not the frozen tip)
        req_e, item_e = _make_fake_item(
            't-e', base_sha='wrong-speculative-base', merge_commit='sha-Ec',
            config=config, git_repo=git_repo,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = worker._warn_if_verify_base_not_frozen_tip(item_e, 'main0')

        # Returns None (log-only, no side effect)
        assert result is None

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
        worker._inflight.append(_make_inflight_entry(item_d, verifying=True))

        # Candidate item: base_sha == frozen_prefix_tip (correct)
        _, item_e = _make_fake_item(
            't-e', base_sha='sha-Dc', merge_commit='sha-Ec',
            config=config, git_repo=git_repo,
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = worker._warn_if_verify_base_not_frozen_tip(item_e, 'main0')

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

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            result = worker._warn_if_verify_base_not_frozen_tip(item_d, 'main0')

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
        """_warn_if_verify_base_not_frozen_tip never mutates worker state."""
        worker = _make_worker(git_ops)
        _, item_d = _make_fake_item(
            't-d', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item_d, verifying=True))

        # Candidate with wrong base (triggers warning path)
        _, item_e = _make_fake_item(
            't-e', base_sha='wrong-base', merge_commit='sha-Ec',
            config=config, git_repo=git_repo,
        )

        # Capture state before
        pre_frozen = worker.frozen_prefix()
        pre_inflight_len = len(worker._inflight)
        pre_tip = worker.frozen_prefix_tip('main0')

        worker._warn_if_verify_base_not_frozen_tip(item_e, 'main0')

        # State must be entirely unchanged
        assert worker.frozen_prefix() == pre_frozen
        assert len(worker._inflight) == pre_inflight_len
        assert worker.frozen_prefix_tip('main0') == pre_tip


# ── Amendment: _finalizing_head branch coverage (review suggestion) ───────────


@pytest.mark.asyncio
class TestFinalizingHeadInFrozenPrefix:
    """_finalizing_head is included in the frozen prefix when its phase is
    verifying/gate_reverify/finalizing, and excluded for other phases.

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

        worker._finalizing_head = entry_d
        worker._inflight.append(entry_e)

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

        worker._finalizing_head = entry_d
        worker._inflight.append(entry_e)

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
        entry_p.phase = 'passthrough'  # override to a non-qualifying phase

        worker._finalizing_head = entry_p

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
        entry_g.phase = 'gate_reverify'  # a qualifying phase distinct from 'verifying'

        worker._finalizing_head = entry_g

        fp = worker.frozen_prefix()
        assert item_g.request.request_id in fp, (
            f"gate_reverify-phase _finalizing_head missing from frozen_prefix(): {fp!r}"
        )


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
        import unittest.mock as mock

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

        async def _noop_verify(_item, _lease):
            return None

        with (
            mock.patch.object(git_ops, 'get_main_sha', side_effect=_raise_on_get_main),
            mock.patch.object(worker, '_run_inflight_verify', _noop_verify),
        ):
            result = await worker._dispatch_item(item)

        # Fail-open: dispatch must return a verifying InflightEntry, not None.
        assert result is not None, (
            '_dispatch_item returned None; expected a verifying InflightEntry '
            'even when get_main_sha() raises inside the §5.3 guard (fail-open)'
        )
        assert isinstance(result, InflightEntry)
        assert result.phase == 'verifying', f'expected phase=verifying, got {result.phase!r}'

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
        import unittest.mock as mock

        worker = _make_worker(git_ops)

        # Frozen predecessor in _inflight: its merge_commit becomes the frozen tip.
        _, item_d = _make_fake_item(
            't-d-frozen', base_sha='main0', merge_commit='sha-Dc',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item_d, verifying=True))

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

        async def _noop_verify(_item, _lease):
            return None

        with (
            mock.patch.object(git_ops, 'get_main_sha', side_effect=_return_main),
            mock.patch.object(worker, '_run_inflight_verify', _noop_verify),
            caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'),
        ):
            result = await worker._dispatch_item(item_e)

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
        assert result is not None, 'dispatch must succeed even when guard warns'
        assert result.phase == 'verifying'

        # Clean up the background verify task.
        if result.verify_task is not None:
            result.verify_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await result.verify_task
