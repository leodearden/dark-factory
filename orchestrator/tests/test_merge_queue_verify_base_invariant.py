"""Tests for promoting the ε=1890 verify-base guard into two_layer_invariants (I5, task 1999).

Covers:
  step-01 RED — the verify-base sub-check composed into two_layer_invariants():
                a frozen entry whose base_sha is not the expected frozen-tip
                base produces a violation on that PUBLIC surface and on
                snapshot()['two_layer_invariants']. A HEALTHY multi-entry
                chained frozen prefix produces NO such violation (guards
                against a naive "every entry == newest tip" implementation).
  3206 step-3  — the PRD §5.3 RE-MERGE CARVE-OUT applies to this surface too:
                a frozen entry carrying `remerge_recovery=True` produces no
                verify-base⊄frozen-tip violation, the carve-out is
                emission-only (the shared `_frozen_base_chain` walk still
                advances past the entry, so successors are not falsely
                flagged), and BOTH §5.3 surfaces decide identically on the
                same inputs — the anti-drift parity the shared-generator
                design exists to protect.
  3206 step-7  — REGRESSION PINS for the ADVISORY contract: a phantom frozen
                tip never refuses a dispatch (the measured 5687 / 5830 shape),
                the WARNING accompanies a COMPLETED dispatch, the mismatch
                path leaks no host lease, and the healthy path is unchanged.

CHARACTERIZATION-TEST NOTE (task 3206). The §5.3 verify-base rule is ADVISORY
BY DECISION and is never promoted to hard control-flow enforcement (PRD §5.3
and §10). The step-7 pins are expected to pass immediately; their purpose is
to make a future SILENT flip to enforcement impossible — each fails the moment
the guard changes control flow. If they go red, re-derive the evidence in PRD
§5.3 before "fixing" them.

HOW THIS MODULE OBSERVES (task 5026). Sub-check (iii) is selected out of the
composed public surface by its own distinct marker (:data:`_VERIFY_BASE_MARKER`)
rather than by calling the per-entry helper behind it, and the §5.3 main-SHA
cache is read back through its only consumer (:func:`_cached_main_sha_violations`)
rather than off the field. Production commits to both of those surfaces in
merge_queue.py's own docstrings; neither is incidental prose.

See _warn_if_verify_base_not_frozen_tip (merge_queue.py) for the log-only
dispatch-time guard this promotes to snapshot granularity, and
check_frozen_prefix_invariant for the base-chain math this mirrors.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_queue import (
    DecidedItem,
    InflightEntry,
    InflightVerifyResult,
    ItemLifecycleState,
    MergeOutcome,
    MergeRequest,
    QueuedBranch,
    RealMergeItem,
    SpeculativeItem,
    SpeculativeMergeWorker,
)
from orchestrator.verify_runner import HostLease

# ── Module-level sentinel for verify_task in pure unit tests ─────────────────
#
# frozen_prefix() only checks `e.verify_task is not None`, so any non-None
# object doubles as a verifying-entry marker (mirrors test_merge_queue_frozen_prefix.py).
_SENTINEL_VERIFY_TASK = object()  # noqa: PD901


# ── Fixtures (mirrored from test_merge_queue_frozen_prefix.py) ───────────────


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


# ── Module-level helpers (mirrored from test_merge_queue_frozen_prefix.py) ───


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
    exercise the accessor/invariant methods, mirroring
    test_merge_queue_frozen_prefix.py's helper of the same name.
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


def _mark_recovery(item: SpeculativeItem) -> SpeculativeItem:
    """Stamp the §5.3 re-merge recovery marker (task 3206) on *item*.

    Mirrors what `_remerge` does in production — it is the single producer of
    recovery items for all five consumer paths, so the marker is set once at
    the producer rather than threaded through each consumer.  Mirrors the
    helper of the same name in test_merge_queue_frozen_prefix.py.
    """
    return dataclasses.replace(item, remerge_recovery=True)


def _make_inflight_entry(
    item: SpeculativeItem,
    *,
    verifying: bool = True,
) -> InflightEntry:
    """Build an InflightEntry for unit tests (verifying=True → frozen)."""
    return InflightEntry(
        item=item,
        lease=None,
        verify_task=_SENTINEL_VERIFY_TASK if verifying else None,  # type: ignore[arg-type]
        merge_wt=None,
        was_speculative=False,
    )


# ── The §5.3 verify-base sub-check, read through its PUBLIC surface ──────────
#
# two_layer_invariants() composes three sub-checks plus the inherited
# base-chain walk, and sub-check (iii) — the ε=1890 verify-base⊄frozen-tip
# promotion — is the one this module is about.  Production words it distinctly
# from check_frozen_prefix_invariant's 'frozen-prefix base-chain broken …' for
# exactly this reason: its docstring commits sub-check (iii) to standing "as
# its own named verify-frontier assertion target".  So the marker below is a
# contract this module is entitled to rely on, not incidental prose — the
# per-entry chained-base predicate is reachable publicly and needs no reach
# into _verify_base_frozen_tip_violations().
_VERIFY_BASE_MARKER = 'verify-base⊄frozen-tip'


def _verify_base_violations(worker: SpeculativeMergeWorker, main_sha: str) -> list[str]:
    """Return only sub-check (iii)'s violations from the public composed surface."""
    return [v for v in worker.two_layer_invariants(main_sha) if _VERIFY_BASE_MARKER in v]


def _install_frozen_head(
    worker: SpeculativeMergeWorker, config: OrchestratorConfig, git_repo: Path,
    *, task_id: str, base_sha: str, merge_commit: str,
) -> SpeculativeItem:
    """Register a frozen head entry on *worker* and return its item.

    Registered at FINALIZING — a qualifying phase, so `_frozen_inflight_entries()`
    counts it and it takes part in the `_frozen_base_chain` walk both §5.3
    surfaces share.  The file's single in-flight-state seed: there is no public
    constructor for in-flight state (the residual named for task 5446).
    """
    _, head_item = _make_fake_item(
        task_id, base_sha=base_sha, merge_commit=merge_commit,
        config=config, git_repo=git_repo,
    )
    worker._register_item(
        _make_inflight_entry(head_item, verifying=True),
        initial=ItemLifecycleState.FINALIZING,
    )
    return head_item


def _stranded_finalize_head(
    worker: SpeculativeMergeWorker, config: OrchestratorConfig, git_repo: Path,
    *, dead_commit: str,
) -> SpeculativeItem:
    """Install a stranded (phantom) finalize head whose merge_commit is dead.

    Reproduces the measured 2026-08-08 shape (the task-3082 class): an entry
    registered at FINALIZING but NOT present in `_inflight` is exactly what
    `_finalizing_head_entry()` returns, and `_entry_phase` reports
    'finalizing', so `frozen_prefix_tip(main)` returns *dead_commit* instead of
    the live main tip.
    """
    head_item = _install_frozen_head(
        worker, config, git_repo,
        task_id='t-phantom', base_sha='some-older-main', merge_commit=dead_commit,
    )
    assert worker.frozen_prefix_tip('LIVEMAIN') == dead_commit, (
        'fixture precondition: the phantom head must poison frozen_prefix_tip'
    )
    return head_item


# ── Reading the §5.3 main-SHA cache back through its only public consumer ────
#
# snapshot()['two_layer_invariants'] composes sub-check (iii) against the
# CACHED main SHA (merge_queue.py's `self._last_known_main_sha or 'unknown'`),
# never against an argument — so a frozen head whose base is a SHA under no
# test's control puts the cached value, verbatim, in the expected-base half of
# the violation.  That makes the cache's exact value assertable without
# reading the field.  Install the probe AFTER the code path under test: it is
# a pure state write, so it cannot perturb what it measures.
_PROBE_BASE = 'PROBE-BASE-NEVER-A-MAIN-SHA'


def _cached_main_sha_violations(
    worker: SpeculativeMergeWorker, config: OrchestratorConfig, git_repo: Path,
) -> list[str]:
    """Return the probe head's verify-base violation(s), which name the cached SHA."""
    _install_frozen_head(
        worker, config, git_repo,
        task_id='t-cache-probe', base_sha=_PROBE_BASE, merge_commit='c-probe',
    )
    return [
        v for v in worker.snapshot()['two_layer_invariants']
        if _VERIFY_BASE_MARKER in v
    ]


# ── step-01 RED: _verify_base_frozen_tip_violations() composed into two_layer_invariants ──


@pytest.mark.asyncio
class TestVerifyBaseFrozenTipPromotion:
    """Promote the ε=1890 dispatch-time guard predicate to snapshot granularity (task 1999 I5).

    RED until step-02 GREEN adds _verify_base_frozen_tip_violations() and
    composes it into two_layer_invariants() as sub-check (iii).
    """

    async def test_stale_verify_base_produces_distinct_violation(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A frozen head entry whose base_sha is NOT main_sha → dedicated verify-base violation.

        Read through :meth:`two_layer_invariants` selected on sub-check
        (iii)'s own marker (see :data:`_VERIFY_BASE_MARKER`), so the assertion
        names the composed public surface the operator dashboard reads rather
        than reaching for the per-entry helper behind it.  Selecting on the
        marker cannot confuse sub-check (iii) with the inherited base-chain
        walk: production words the two distinctly on purpose, and
        check_frozen_prefix_invariant's 'frozen-prefix base-chain broken …'
        shares no substring with it.  ``snapshot()['two_layer_invariants']``
        is checked separately because it composes against the CACHED main SHA
        (λ=1895's real-main-not-frozen-tip convention), not against an
        argument.
        """
        worker = _make_worker(git_ops)
        main_sha = 'M0'
        _, item = _make_fake_item(
            't-stale', base_sha='deadbeef', merge_commit='c1',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item, verifying=True))
        rid = item.request.request_id

        violations = _verify_base_violations(worker, main_sha)
        assert violations, (
            f'expected two_layer_invariants() to flag {rid!r} with a verify-base '
            f'sub-check (iii) violation, got: {worker.two_layer_invariants(main_sha)}'
        )
        assert all(rid in v for v in violations), (
            f'expected every violation to name {rid!r}, got: {violations}'
        )

        # snapshot()['two_layer_invariants'] must surface the same violation(s) —
        # set the cached main SHA so snapshot() computes against real main.
        worker._last_known_main_sha = main_sha
        snap_violations = worker.snapshot()['two_layer_invariants']
        assert all(v in snap_violations for v in violations), (
            f'expected snapshot()["two_layer_invariants"] to surface the same '
            f'verify-base violation(s), got: {snap_violations}'
        )

    async def test_healthy_chained_prefix_has_no_verify_base_violation(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A HEALTHY 2-entry chained frozen prefix must NOT trip the verify-base check.

        entry0.base==main_sha, entry0.merge_commit==C1; entry1.base==C1,
        entry1.merge_commit==C2.  Guards against a naive "every entry.base_sha
        == frozen_prefix_tip(main_sha)" implementation, which would falsely
        flag entry0 (base=main_sha != newest tip C2).
        """
        worker = _make_worker(git_ops)
        main_sha = 'M0'
        _, item_0 = _make_fake_item(
            't-0', base_sha=main_sha, merge_commit='C1',
            config=config, git_repo=git_repo,
        )
        _, item_1 = _make_fake_item(
            't-1', base_sha='C1', merge_commit='C2',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item_0, verifying=True))
        worker._inflight.append(_make_inflight_entry(item_1, verifying=True))

        # The whole composed surface must be clean, which subsumes sub-check
        # (iii) being clean — a naive "every entry == newest tip" implementation
        # would put entry0's violation in here.
        violations = worker.two_layer_invariants(main_sha)
        assert violations == [], (
            f'expected a healthy chained frozen prefix to have no violations, got: {violations}'
        )


# ── task 3206 step-3 RED: §5.3 re-merge carve-out at the SNAPSHOT surface ────


@pytest.mark.asyncio
class TestRemergeCarveOutAtSnapshotSurface:
    """The snapshot §5.3 surface honours the same re-merge carve-out (task 3206).

    §5.3 has TWO surfaces: the dispatch-time guard
    (:meth:`_warn_if_verify_base_not_frozen_tip`) and this snapshot-granularity
    promotion composed into two_layer_invariants() as sub-check (iii) (task
    1999 I5).  The PRD §5.3 re-merge carve-out is a property of the RULE, not
    of one surface, so BOTH must apply it — otherwise a recovery re-merge is
    silent at dispatch but still reported as a violation in the health
    snapshot.  That is exactly the silent-disagreement failure the shared
    `_frozen_base_chain` generator exists to prevent.

    Asserted on BOTH composed surfaces — two_layer_invariants(main_sha) and
    snapshot()['two_layer_invariants'] — selected by sub-check (iii)'s own
    marker, per this module's established convention.

    RED until step-4 GREEN skips marked entries at violation-emission time.
    """

    async def test_recovery_entry_produces_no_verify_base_violation(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """CARVE-OUT: a frozen entry marked remerge_recovery emits no violation.

        Same shape as test_stale_verify_base_produces_distinct_violation above
        (base_sha != the chained expected base), plus the marker — so the ONLY
        difference between violation and silence is the carve-out itself.
        """
        worker = _make_worker(git_ops)
        main_sha = 'M0'
        _, item = _make_fake_item(
            't-recovery', base_sha='deadbeef', merge_commit='c1',
            config=config, git_repo=git_repo,
        )
        item = _mark_recovery(item)
        worker._inflight.append(_make_inflight_entry(item, verifying=True))

        rid = item.request.request_id
        assert _verify_base_violations(worker, main_sha) == [], (
            '§5.3 re-merge carve-out (task 3206): a recovery entry must not '
            'produce a verify-base⊄frozen-tip violation'
        )

        # …and it must be absent from the SNAPSHOT surface too, not merely
        # from the argument-taking one (a carve-out that the snapshot path
        # re-derived for itself would still leak into operator health).
        worker._last_known_main_sha = main_sha
        snap_violations = worker.snapshot()['two_layer_invariants']
        assert not [v for v in snap_violations if _VERIFY_BASE_MARKER in v and rid in v], (
            f'expected no verify-base violation for the recovery entry in '
            f'snapshot()["two_layer_invariants"], got: {snap_violations}'
        )

    async def test_unmarked_entry_still_produces_exactly_one_violation(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """CONTROL: without the marker, step-01's behaviour is preserved exactly.

        The carve-out must be narrow — it may not swallow a genuine violation.
        """
        worker = _make_worker(git_ops)
        main_sha = 'M0'
        _, item = _make_fake_item(
            't-stale', base_sha='deadbeef', merge_commit='c1',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item, verifying=True))
        rid = item.request.request_id

        fired = _verify_base_violations(worker, main_sha)
        assert len(fired) == 1, (
            f'expected exactly 1 verify-base violation for an unmarked stale '
            f'entry, got: {fired}'
        )
        assert rid in fired[0], f'violation must name {rid!r}, got: {fired[0]!r}'

    async def test_recovery_entry_still_advances_the_chain_for_successors(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """The carve-out suppresses EMISSION only — never the chain walk itself.

        `_frozen_base_chain` is shared with check_frozen_prefix_invariant, and
        each entry's merge_commit advances the expected base for its
        successors.  Dropping a recovery entry from the WALK would corrupt the
        expected base of every entry behind it, converting one suppressed
        violation into a cascade of false ones.

        Here entry0 is a recovery entry (base != main, suppressed) whose
        merge_commit is C1; entry1 is a HEALTHY successor based on C1.  entry1
        must stay clean — which is only true if entry0 still advanced the chain.
        """
        worker = _make_worker(git_ops)
        main_sha = 'M0'
        _, item_0 = _make_fake_item(
            't-recovery', base_sha='some-other-main', merge_commit='C1',
            config=config, git_repo=git_repo,
        )
        item_0 = _mark_recovery(item_0)
        _, item_1 = _make_fake_item(
            't-successor', base_sha='C1', merge_commit='C2',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item_0, verifying=True))
        worker._inflight.append(_make_inflight_entry(item_1, verifying=True))

        assert _verify_base_violations(worker, main_sha) == [], (
            'recovery entry suppressed, healthy successor clean — a chain '
            'corrupted by filtering the recovery entry OUT of the walk would '
            'falsely flag the successor'
        )

        # check_frozen_prefix_invariant shares the same generator and is
        # deliberately NOT carved out: it still sees the recovery entry's own
        # base-chain break, proving the walk was not filtered.
        chain_violations = worker.check_frozen_prefix_invariant(main_sha)
        assert [v for v in chain_violations if item_0.request.request_id in v], (
            f'expected check_frozen_prefix_invariant to still see the recovery '
            f'entry in the shared walk, got: {chain_violations}'
        )
        assert not [v for v in chain_violations if item_1.request.request_id in v], (
            f'the healthy successor must NOT be flagged — chain advanced past '
            f'the recovery entry, got: {chain_violations}'
        )

    async def test_two_surfaces_agree_on_the_same_inputs(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """ANTI-DRIFT PARITY: both §5.3 surfaces decide identically.

        For the identical (item, main_sha) inputs the dispatch guard's decision
        to WARN and the snapshot surface's decision to report a VIOLATION must
        agree: both fire for the unmarked case, both stay silent for the
        marked one.  This is the explicit anti-drift assertion the
        shared-generator design exists to protect — a carve-out landed on only
        one surface passes that surface's own tests but fails here.
        """
        import logging

        main_sha = 'M0'
        for marked, want_fire in ((False, True), (True, False)):
            worker = _make_worker(git_ops)
            _, item = _make_fake_item(
                f't-parity-{marked}', base_sha='deadbeef', merge_commit='c1',
                config=config, git_repo=git_repo,
            )
            if marked:
                item = _mark_recovery(item)

            # Snapshot surface: the entry IS the frozen head, expected == main.
            worker._inflight.append(_make_inflight_entry(item, verifying=True))
            snapshot_fired = bool(_verify_base_violations(worker, main_sha))

            # Dispatch surface: same item, same main, evaluated as the
            # candidate against an EMPTY frozen prefix (so frozen_prefix_tip
            # == main_sha) — the same expected-vs-actual comparison the
            # snapshot surface makes for the head entry.
            dispatch_worker = _make_worker(git_ops)
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
                dispatch_worker._warn_if_verify_base_not_frozen_tip(item, main_sha)
            dispatch_fired = any(
                r.levelno >= logging.WARNING for r in caplog.records
            )

            assert dispatch_fired == snapshot_fired == want_fire, (
                f'§5.3 surfaces disagree for marked={marked}: '
                f'dispatch_fired={dispatch_fired}, snapshot_fired={snapshot_fired}, '
                f'expected both={want_fire}'
            )


# ── DEFECT 2 (task 2357) RED: dequeue/dispatch-time refresh of _last_known_main_sha ──


def _fake_local_allocator() -> MagicMock:
    """A MagicMock HostAllocator stub with a free local slot and a patched
    ``acquire`` that returns a fake local HostLease directly — bypasses the
    real LocalRunner factory closure entirely (merge_wt realness is
    irrelevant to this guard-refresh test), mirroring the established
    pattern at test_merge_queue_concurrent_verify.py:499.
    """
    allocator = MagicMock()
    allocator.free_host_count.return_value = 1
    allocator.acquire = AsyncMock(
        return_value=HostLease(name='local', runner=MagicMock(), is_local=True)
    )
    return allocator


@pytest.mark.asyncio
class TestDispatchRefreshesLastKnownMainSha:
    """The ε=1890 §5.3 dispatch guard in `_dispatch_item` must refresh the
    cached main SHA from its own already-fetched fresh `get_main_sha()`
    result, so `snapshot()['two_layer_invariants']` never lags the live
    dispatch guard's view of main (task 2357 DEFECT 2).

    RED until the GREEN step adds the one-line cache write inside the
    guard's try-block (merge_queue.py `_dispatch_item`, ~10291-10295).
    """

    async def test_dispatch_refreshes_last_known_main_sha(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A dispatch's fresh get_main_sha() must refresh the §5.3 main-SHA cache.

        The refreshed value is read back through the cache's only consumer —
        see :func:`_cached_main_sha_violations` — so the assertion names the
        fresh SHA exactly, not merely "no longer stale" (which the companion
        test below covers from the other direction).
        """
        old_main = 'OLD00000'
        new_main = 'NEW11111'
        git_ops.get_main_sha = AsyncMock(return_value=new_main)  # type: ignore[method-assign]
        worker = _make_worker(git_ops)
        worker._last_known_main_sha = old_main

        # Non-speculative item whose base_sha == new_main so Mechanism 2's
        # main_advanced chain-remerge guard (merge_queue.py:10046) does not fire.
        _, item = _make_fake_item(
            't-dispatch', base_sha=new_main, merge_commit='c-dispatch',
            config=config, git_repo=git_repo,
        )
        assert isinstance(item, RealMergeItem)  # narrow union arm for item.merge_wt (pyright)

        worker._host_allocator = _fake_local_allocator()
        worker._run_inflight_verify = AsyncMock(  # type: ignore[method-assign]
            return_value=InflightVerifyResult(outcome=None, merge_wt=item.merge_wt)
        )

        entry = await worker._dispatch_item(item)

        assert entry is not None, 'dispatch should succeed with a free local slot'
        cached = _cached_main_sha_violations(worker, config, git_repo)
        assert len(cached) == 1 and new_main in cached[0] and old_main not in cached[0], (
            f'expected _dispatch_item to refresh the §5.3 cache to the fresh '
            f'ε=1890 guard SHA {new_main!r}, got: {cached}'
        )

    async def test_dispatch_refresh_closes_snapshot_false_positive(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Companion snapshot check: a stale cache produces a false §5.3
        positive for a healthy frozen head before dispatch; the dispatch
        guard's refresh closes it afterward — with the merger otherwise
        idle (no recompute_suffix_conflict_graph call anywhere in this test).
        """
        old_main = 'OLD00000'
        new_main = 'NEW11111'
        git_ops.get_main_sha = AsyncMock(return_value=new_main)  # type: ignore[method-assign]
        worker = _make_worker(git_ops)
        worker._last_known_main_sha = old_main

        # A healthy frozen head already in _inflight, based on new_main.
        _, head_item = _make_fake_item(
            't-head', base_sha=new_main, merge_commit='c-head',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(head_item, verifying=True))

        # Before dispatch: cache is stale (old_main) → false positive.
        assert worker.snapshot()['two_layer_invariants'] != [], (
            'expected a false positive while _last_known_main_sha is stale'
        )

        _, item = _make_fake_item(
            't-dispatch2', base_sha=new_main, merge_commit='c-dispatch2',
            config=config, git_repo=git_repo,
        )
        assert isinstance(item, RealMergeItem)  # narrow union arm for item.merge_wt (pyright)
        worker._host_allocator = _fake_local_allocator()
        worker._run_inflight_verify = AsyncMock(  # type: ignore[method-assign]
            return_value=InflightVerifyResult(outcome=None, merge_wt=item.merge_wt)
        )

        await worker._dispatch_item(item)

        # After dispatch: cache refreshed → the pre-existing head_item's false
        # positive is gone (the newly-dispatched item is not itself appended
        # to _inflight here — that is the caller's job on a successful dispatch).
        violations = worker.snapshot()['two_layer_invariants']
        assert violations == [], (
            f'expected the false positive to clear once _last_known_main_sha '
            f'is refreshed by dispatch, got: {violations!r}'
        )

    async def test_dispatch_fail_open_leaves_cache_unchanged(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A transient get_main_sha() error at the guard must leave the §5.3
        main-SHA cache untouched — fail-open, never overwrite a good cache
        with a bad read — while dispatch still succeeds.
        """
        old_main = 'OLD00000'
        # First call = Mechanism 2 staleness check (succeeds); second call =
        # the ε=1890 guard's own fetch (fails) — isolates the fail-open
        # behaviour to the guard's try/except without disturbing Mechanism 2.
        git_ops.get_main_sha = AsyncMock(  # type: ignore[method-assign]
            side_effect=[old_main, RuntimeError('simulated transient git error')]
        )
        worker = _make_worker(git_ops)
        worker._last_known_main_sha = old_main

        # base_sha == old_main so Mechanism 2's staleness check (which fires
        # first and uses its own get_main_sha() call) does not trigger a remerge.
        _, item = _make_fake_item(
            't-failopen', base_sha=old_main, merge_commit='c-failopen',
            config=config, git_repo=git_repo,
        )
        assert isinstance(item, RealMergeItem)  # narrow union arm for item.merge_wt (pyright)
        worker._host_allocator = _fake_local_allocator()
        worker._run_inflight_verify = AsyncMock(  # type: ignore[method-assign]
            return_value=InflightVerifyResult(outcome=None, merge_wt=item.merge_wt)
        )

        entry = await worker._dispatch_item(item)

        assert entry is not None, 'dispatch must still succeed despite the guard error'
        cached = _cached_main_sha_violations(worker, config, git_repo)
        assert len(cached) == 1 and old_main in cached[0], (
            f'a get_main_sha() error at the guard must leave the cache unchanged '
            f'at {old_main!r} (fail-open), got: {cached}'
        )


# ── DEFECT 2 (task 2357) regression lock: no-over-correction ─────────────────


@pytest.mark.asyncio
class TestNoOverCorrectionRegressionLock:
    """The DEFECT 2 refresh (dequeue + land, steps 2/4) must not blind the
    underlying §5.3 invariant: genuine drift is still caught, and a healthy
    stack with a fresh cache is genuinely clean.

    Expected GREEN immediately (the refresh only corrects the cached SHA
    fed into an unchanged invariant surface); RED would indicate a refresh
    step wrongly mutated the invariant logic itself — the over-correction
    guard.
    """

    async def test_genuine_drift_still_caught(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A frozen head whose base_sha is NOT a FRESH main_sha still trips
        two_layer_invariants()' verify-base sub-check — proving the refresh
        corrects the cached SHA rather than blinding the check itself.
        """
        worker = _make_worker(git_ops)
        fresh_main = 'FRESH-MAIN-001'

        _, item = _make_fake_item(
            't-drift', base_sha='STALE-DRIFT-BASE', merge_commit='c-drift',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item, verifying=True))
        rid = item.request.request_id

        violations = _verify_base_violations(worker, fresh_main)
        assert violations, (
            f'expected genuine drift (base_sha != fresh main_sha) to still be '
            f'flagged against a FRESH main_sha, got: '
            f'{worker.two_layer_invariants(fresh_main)}'
        )
        assert all(rid in v for v in violations), (
            f'expected every violation to name {rid!r}, got: {violations}'
        )

    async def test_healthy_chained_stack_with_fresh_cache_is_clean(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A healthy multi-entry chained frozen stack (head base==real main;
        each successor base==predecessor merge_commit) with
        _last_known_main_sha set to the real current tip yields
        two_layer_invariants() == [] — the false positive is genuinely gone,
        not just papered over.
        """
        worker = _make_worker(git_ops)
        real_main = await git_ops.get_main_sha()

        _, item_0 = _make_fake_item(
            't-chain-0', base_sha=real_main, merge_commit='c-chain-1',
            config=config, git_repo=git_repo,
        )
        _, item_1 = _make_fake_item(
            't-chain-1', base_sha='c-chain-1', merge_commit='c-chain-2',
            config=config, git_repo=git_repo,
        )
        _, item_2 = _make_fake_item(
            't-chain-2', base_sha='c-chain-2', merge_commit='c-chain-3',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item_0, verifying=True))
        worker._inflight.append(_make_inflight_entry(item_1, verifying=True))
        worker._inflight.append(_make_inflight_entry(item_2, verifying=True))

        # The dequeue/land refresh (steps 2/4) is what keeps this field equal
        # to the real tip in production; set it directly here to isolate the
        # invariant-surface assertion from the refresh call sites themselves
        # (those are covered by TestDispatchRefreshesLastKnownMainSha and
        # test_merge_queue_two_layer_integration.py's land tests).
        worker._last_known_main_sha = real_main

        assert worker.two_layer_invariants(real_main) == [], (
            'expected a healthy chained frozen stack to have no violations '
            'against the real main tip'
        )

        snap_violations = worker.snapshot()['two_layer_invariants']
        assert snap_violations == [], (
            f'expected snapshot() to be clean once _last_known_main_sha is the '
            f'real tip for a healthy chained stack, got: {snap_violations!r}'
        )


# ── task 3206 step-7: regression pins for the ADVISORY contract ──────────────
#
# CHARACTERIZATION TESTS. Most of these pass immediately after steps 2/4/6 —
# that is the point. Their job is not to drive new behaviour but to make a
# future SILENT flip to enforcement impossible: each one FAILS the moment the
# §5.3 guard changes control flow. Task 3206 adjudicated that the guard STAYS
# ADVISORY (PRD §5.3, §10); these are the executable half of that fence, the
# PRD prose being the other half.


@pytest.mark.asyncio
class TestAdvisoryContractRegressionPins:
    """The §5.3 guard is advisory: it may log, but it may never refuse.

    Task regression items (1), (4), (5) plus the lease-leak trap. If a future
    change makes `_warn_if_verify_base_not_frozen_tip` (or the dispatch site
    around it) refuse a dispatch, every test here goes red — which is the
    designed outcome, not a flake to be silenced. Re-derive the evidence in
    PRD §5.3 first.
    """

    async def test_phantom_frozen_tip_does_not_block_dispatch(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """(a) THE 3082 COUPLING PIN — the headline regression (task item 4).

        A stranded finalize head makes `frozen_prefix_tip()` return a DEAD
        merge commit, so an unrelated item dispatching against the LIVE MAIN
        TIP mismatches it.  That dispatch must still go through: the verify
        task is launched, a live InflightEntry is returned, and the host lease
        is NOT released.

        Production instances (orchestrator-reify, both `verify_depth=1`):
          2026-08-08 00:09:07  task 5687 (mr-441b5c13)  base 9d08d3d3
          2026-08-08 01:35:00  task 5830 (mr-ad16d177)  base 2b903602
        both mismatched a phantom expected-tip 95908e44 that persisted 86
        minutes.  5687 then PASSED and LANDED — 'verify end (merge=2b903602,
        passed=True)' → 'Advanced main to 2b903602' at 00:59:55.  Under hard
        enforcement both would have been REFUSED; this test is what makes that
        regression loud.
        """
        live_main = 'LIVEMAIN'
        git_ops.get_main_sha = AsyncMock(return_value=live_main)  # type: ignore[method-assign]
        worker = _make_worker(git_ops)
        _stranded_finalize_head(
            worker, config, git_repo, dead_commit='DEADPHANTOM',
        )

        # The unrelated dispatch: its base IS the live main tip — correct by
        # every measure except the poisoned frozen tip.
        _, item = _make_fake_item(
            't-victim', base_sha=live_main, merge_commit='c-victim',
            config=config, git_repo=git_repo,
        )
        assert isinstance(item, RealMergeItem)  # narrow for item.merge_wt (pyright)

        allocator = _fake_local_allocator()
        worker._host_allocator = allocator
        worker._run_inflight_verify = AsyncMock(  # type: ignore[method-assign]
            return_value=InflightVerifyResult(outcome=None, merge_wt=item.merge_wt)
        )

        entry = await worker._dispatch_item(item)

        assert entry is not None, (
            'ADVISORY CONTRACT VIOLATED (task 3206, PRD §5.3): a phantom frozen '
            'tip must never refuse a dispatch whose base is the live main tip. '
            'This is the measured 5687 / 5830 shape — that dispatch passed and '
            'landed in production.'
        )
        assert entry.verify_task is not None, (
            'the real-verify task must still be launched, not skipped'
        )
        assert entry.passthrough_outcome is None, (
            'the item must not be short-circuited into a passthrough outcome'
        )

    async def test_advisory_warns_and_still_dispatches(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """(b) ADVISORY PROCEEDS (task item 1) — WARNING *and* dispatch, together.

        Asserted in one test on purpose: nobody can keep the log and drop the
        flow, or keep the flow and silently drop the log.
        """
        import logging

        live_main = 'LIVEMAIN'
        git_ops.get_main_sha = AsyncMock(return_value=live_main)  # type: ignore[method-assign]
        worker = _make_worker(git_ops)
        _stranded_finalize_head(
            worker, config, git_repo, dead_commit='DEADPHANTOM',
        )

        _, item = _make_fake_item(
            't-victim2', base_sha=live_main, merge_commit='c-victim2',
            config=config, git_repo=git_repo,
        )
        assert isinstance(item, RealMergeItem)
        worker._host_allocator = _fake_local_allocator()
        worker._run_inflight_verify = AsyncMock(  # type: ignore[method-assign]
            return_value=InflightVerifyResult(outcome=None, merge_wt=item.merge_wt)
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            entry = await worker._dispatch_item(item)

        guard_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and '§5.3 guard' in r.getMessage()
        ]
        assert len(guard_warnings) == 1, (
            f'expected exactly one §5.3 guard WARNING, got: '
            f'{[r.getMessage() for r in guard_warnings]}'
        )
        assert entry is not None, (
            'the guard WARNING must accompany a COMPLETED dispatch — logging '
            'without proceeding is enforcement by another name'
        )

    async def test_mismatch_path_does_not_leak_the_host_lease(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """(c) NO LEASE LEAK — the trap any future enforcement must solve.

        The guard sits AFTER `allocator.acquire()` in `_dispatch_item`, so a
        bare `return None` inserted at the guard would leak the host lease:
        acquired, never released, never attached to a returned entry.  Pin
        both halves — the lease is acquired and NOT released/cancelled, and it
        is handed to the returned entry rather than dropped on the floor.
        """
        live_main = 'LIVEMAIN'
        git_ops.get_main_sha = AsyncMock(return_value=live_main)  # type: ignore[method-assign]
        worker = _make_worker(git_ops)
        _stranded_finalize_head(
            worker, config, git_repo, dead_commit='DEADPHANTOM',
        )

        _, item = _make_fake_item(
            't-victim3', base_sha=live_main, merge_commit='c-victim3',
            config=config, git_repo=git_repo,
        )
        assert isinstance(item, RealMergeItem)
        allocator = _fake_local_allocator()
        worker._host_allocator = allocator
        worker._run_inflight_verify = AsyncMock(  # type: ignore[method-assign]
            return_value=InflightVerifyResult(outcome=None, merge_wt=item.merge_wt)
        )

        entry = await worker._dispatch_item(item)

        assert allocator.acquire.await_count == 1, (
            'fixture precondition: the dispatch must have acquired a lease'
        )
        for leaked in ('release', 'cancel_and_release', 'quarantine_and_release'):
            assert not getattr(allocator, leaked).called, (
                f'HOST LEASE LEAK / premature release: allocator.{leaked} was '
                f'called on the §5.3 mismatch path. The guard runs AFTER '
                f'acquire(); an enforcement flip that returns early here must '
                f'hand the lease back explicitly (task 3206).'
            )
        assert entry is not None and entry.lease is not None, (
            'the acquired lease must be attached to the returned entry, not dropped'
        )

    async def test_healthy_base_dispatches_with_no_warning(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """(e) HEALTHY UNAFFECTED (task item 5) — base == frozen tip.

        No WARNING, dispatch completes exactly as before. Pins that steps
        2/4/6 changed nothing on the healthy path.
        """
        import logging

        live_main = 'LIVEMAIN'
        git_ops.get_main_sha = AsyncMock(return_value=live_main)  # type: ignore[method-assign]
        worker = _make_worker(git_ops)

        _, item = _make_fake_item(
            't-healthy', base_sha=live_main, merge_commit='c-healthy',
            config=config, git_repo=git_repo,
        )
        assert isinstance(item, RealMergeItem)
        worker._host_allocator = _fake_local_allocator()
        worker._run_inflight_verify = AsyncMock(  # type: ignore[method-assign]
            return_value=InflightVerifyResult(outcome=None, merge_wt=item.merge_wt)
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_queue'):
            entry = await worker._dispatch_item(item)

        assert entry is not None, 'a healthy dispatch must complete'
        guard_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and '§5.3 guard' in r.getMessage()
        ]
        assert guard_warnings == [], (
            f'healthy base == frozen tip must emit no §5.3 WARNING, got: '
            f'{[r.getMessage() for r in guard_warnings]}'
        )
