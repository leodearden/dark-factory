"""The §5.3 verify-base rule after its split out of two_layer_invariants (task 5302).

Task 1999 (I5) promoted the ε=1890 dispatch-time verify-base guard into
two_layer_invariants() as sub-check (iii).  Task 5302 found it re-judged a
DISPATCH-time property against POST-dispatch main, so any move of main under
an in-flight head read as a violation (the 2026-08-12 and 2026-08-28
recurrences, PRD §5.3).  The rule is now a detector, evaluated once at
dispatch and published in snapshot()['verify_base_divergences'] (covered in
test_merge_queue_frozen_prefix.py and test_merge_lane_verify_base_detector.py);
two_layer_invariants reports INV-3 chain-intact instead.

Covers:
  task 5302     — no two_layer_invariants() or snapshot() output carries the
                  'verify-base⊄frozen-tip' marker; a frozen head off current
                  main that is not a recorded dead commit is no violation.
  3206 step-3   — the PRD §5.3 RE-MERGE CARVE-OUT after the split: the
                  dispatch log and record agree, and in the chain walk a
                  recovery entry's link break is annotated and still advances
                  the chain for its successors.
  2357 DEFECT 2 — dispatch refreshes the cached main SHA, fail-open; a stale
                  cache no longer raises a false positive at all.
  3206 step-7   — REGRESSION PINS for the no-refusal contract: a phantom frozen
                  tip never refuses a dispatch (the measured 5687 / 5830 shape),
                  the WARNING accompanies a COMPLETED dispatch, the mismatch
                  path leaks no host lease, and the healthy path is unchanged.

CHARACTERIZATION-TEST NOTE (task 3206). The §5.3 verify-base rule is a
DETECTOR and is never promoted to hard control-flow enforcement (esc-3206-3;
PRD §5.3 and §10). The step-7 pins make a future SILENT flip to enforcement
impossible — each fails the moment the guard changes control flow. If they go
red, re-derive the evidence in PRD §5.3 before "fixing" them.

See merge_lane/worker.py::SpeculativeMergeWorker::_verify_base_divergence_at_dispatch
for the dispatch-time detector, and check_frozen_prefix_invariant for the INV-3
chain walk two_layer_invariants reports.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock

import pytest
from _merge_lane_fakes import VerifyTasks, lane_scene_config
from _merge_lane_fakes import verifies as verifies
from test_merge_queue_two_layer_integration import _make_merged_item

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, MergeResult, _run
from orchestrator.merge_lane.dead_base_ledger import DeadBaseKiller, DeadBaseSite
from orchestrator.merge_lane.types import (
    InflightStatus,
    RemergeBase,
    RemergeReason,
    VerifyBaseDivergenceKind,
)
from orchestrator.merge_lane.worker import (
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
    return lane_scene_config(git_repo, git_config)


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
    helper of the same name in test_merge_queue_frozen_prefix.py: since task
    8524 that is a recovery re-merge that fell to main.
    """
    return dataclasses.replace(item, remerge_base=RemergeBase(None, RemergeReason.NO_TIP))


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


def _install_frozen_head(
    worker: SpeculativeMergeWorker, config: OrchestratorConfig, git_repo: Path,
    *, task_id: str, base_sha: str, merge_commit: str,
) -> SpeculativeItem:
    """Register a frozen head entry on *worker* and return its item.

    Registered at FINALIZING — a qualifying phase, so `_frozen_inflight_entries()`
    counts it and it heads the INV-3 chain walk two_layer_invariants reports.
    The file's single in-flight-state seed: there is no public constructor for
    in-flight state (the residual named for task 5446).
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
    the live main tip.  The entry is not vacated: this is the stranding path
    task 5302's liveness rule cannot see, so the pins below stay meaningful.
    """
    head_item = _install_frozen_head(
        worker, config, git_repo,
        task_id='t-phantom', base_sha='some-older-main', merge_commit=dead_commit,
    )
    assert worker.frozen_prefix_tip('LIVEMAIN') == dead_commit, (
        'fixture precondition: the phantom head must poison frozen_prefix_tip'
    )
    return head_item


def _cached_main_sha(worker: SpeculativeMergeWorker) -> str | None:
    """The main SHA snapshot() feeds two_layer_invariants.

    Read off the intake: since task 5302 no report echoes the cached SHA
    verbatim, so its consumer can no longer stand in for the field.
    """
    return worker._intake.last_known_main_sha


# ── task 5302: two_layer_invariants reports INV-3, not the verify-base rule ──


@pytest.mark.asyncio
class TestVerifyBaseIsNotAnInvariant:
    """A frozen head off current main is intact unless its base is a recorded
    dead commit (INV-3, plans/merge-verdict-integrity-prd.md §1).

    Main moving under an in-flight verify is not a violation.  A verify that
    STARTED off the frozen tip is the dispatch-time detector's record, in
    ``snapshot()['verify_base_divergences']``, not this field's.
    """

    async def test_head_off_current_main_is_no_violation(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """The old head-mismatch and genuine-drift rows, flipped (task 5302)."""
        worker = _make_worker(git_ops)
        main_sha = 'M0'
        _, item = _make_fake_item(
            't-drift', base_sha='deadbeef', merge_commit='c1',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(item, verifying=True))
        worker._intake.note_main_sha(main_sha)

        assert worker.two_layer_invariants(main_sha) == []
        assert worker.snapshot()['two_layer_invariants'] == []

    async def test_healthy_chained_prefix_has_no_violation(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A HEALTHY 2-entry chained frozen prefix has no violation at all.

        entry0.base==main_sha, entry0.merge_commit==C1; entry1.base==C1,
        entry1.merge_commit==C2.  Guards against a naive "every entry.base_sha
        == frozen_prefix_tip(main_sha)" check, which would flag entry0.
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
        worker._inflight.extend([
            _make_inflight_entry(item_0, verifying=True),
            _make_inflight_entry(item_1, verifying=True),
        ])

        violations = worker.two_layer_invariants(main_sha)
        assert violations == [], (
            f'expected a healthy chained frozen prefix to have no violations, got: {violations}'
        )


# ── task 3206: the §5.3 re-merge carve-out, after the task-5302 split ────────


@pytest.mark.asyncio
class TestRemergeCarveOutAfterTheSplit:
    """Where the PRD §5.3 re-merge carve-out still shows (task 3206, task 5302).

    At dispatch it suppresses the detector, and the WARNING and the record
    decide together.  In the INV-3 chain walk, a recovery entry behind a
    frozen predecessor really does break the successor link: it is reported,
    annotated 'recovery re-merge (expected)', and still advances the chain, so
    its own successor is not falsely flagged.
    """

    def _three_entry_prefix(
        self, worker: SpeculativeMergeWorker, config: OrchestratorConfig, git_repo: Path,
        *, marked: bool,
    ) -> tuple[SpeculativeItem, SpeculativeItem]:
        """Seed head on 'M0' → middle on real main (recovery iff *marked*) → tail on it."""
        _, head = _make_fake_item(
            't-head', base_sha='M0', merge_commit='C0', config=config, git_repo=git_repo,
        )
        _, middle = _make_fake_item(
            't-middle', base_sha='real-main', merge_commit='C1',
            config=config, git_repo=git_repo,
        )
        if marked:
            middle = _mark_recovery(middle)
        _, tail = _make_fake_item(
            't-tail', base_sha='C1', merge_commit='C2', config=config, git_repo=git_repo,
        )
        worker._inflight.extend(_make_inflight_entry(i) for i in (head, middle, tail))
        return middle, tail

    async def test_recovery_link_break_is_annotated_and_advances_the_chain(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        worker = _make_worker(git_ops)
        middle, tail = self._three_entry_prefix(worker, config, git_repo, marked=True)

        violations = worker.two_layer_invariants('M0')

        assert len(violations) == 1, (
            f'expected exactly the annotated middle link: the healthy tail must '
            f'NOT be flagged, as the chain advanced past the recovery entry, and '
            f'no snapshot-time verify-base check may report, got: {violations}'
        )
        assert violations[0].startswith(
            f'frozen-prefix base-chain broken at {middle.request.request_id}: '
            f"expected base 'C0' but item has 'real-main' "
            f'[recovery re-merge fell to main (no_tip):'
        ), violations
        assert worker.check_frozen_prefix_invariant('M0') == violations

    async def test_unmarked_link_break_is_not_annotated(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """CONTROL: the annotation is narrow — genuine damage reads as damage."""
        worker = _make_worker(git_ops)
        middle, _ = self._three_entry_prefix(worker, config, git_repo, marked=False)

        flagged = [
            v for v in worker.check_frozen_prefix_invariant('M0')
            if middle.request.request_id in v
        ]
        assert len(flagged) == 1 and 'recovery re-merge' not in flagged[0], flagged

    async def test_dispatch_log_and_record_agree(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """ANTI-DRIFT: the WARNING fires exactly when a record is returned.

        One computation feeds the log, the L1 and
        ``snapshot()['verify_base_divergences']``, so for the same inputs they
        decide together: both for an unmarked off-tip item, neither for a
        recovery one.
        """
        import logging

        for marked, want_fire in ((False, True), (True, False)):
            _, item = _make_fake_item(
                f't-parity-{marked}', base_sha='deadbeef', merge_commit='c1',
                config=config, git_repo=git_repo,
            )
            if marked:
                item = _mark_recovery(item)
            dispatch_worker = _make_worker(git_ops)
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
                record = dispatch_worker._verify_base_divergence_at_dispatch(item, 'M0')
            logged = any(r.levelno >= logging.WARNING for r in caplog.records)

            assert logged == (record is not None) == want_fire, (
                f'dispatch log and record disagree for marked={marked}: '
                f'logged={logged}, record={record!r}, expected both={want_fire}'
            )


# ── DEFECT 2 (task 2357): dequeue/dispatch-time refresh of last_known_main_sha ──


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
    """The ε=1890 §5.3 dispatch guard in `_dispatch_item` refreshes the cached
    main SHA from its own already-fetched fresh `get_main_sha()` result
    (task 2357 DEFECT 2), so the main snapshot() feeds two_layer_invariants
    never lags the dispatch's view.

    Since task 5302 a stale cache can no longer raise a false positive (the
    head arm does not compare with main), but INV-3's dead-link check still
    short-circuits on base == main, so the refresh keeps its purpose.
    """

    async def test_dispatch_refreshes_last_known_main_sha(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A dispatch's fresh get_main_sha() must refresh the §5.3 main-SHA cache."""
        old_main = 'OLD00000'
        new_main = 'NEW11111'
        git_ops.get_main_sha = AsyncMock(return_value=new_main)  # type: ignore[method-assign]
        worker = _make_worker(git_ops)
        worker._intake.note_main_sha(old_main)

        # Non-speculative item whose base_sha == new_main so Mechanism 2's
        # main_advanced chain-remerge guard does not fire.
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
        assert _cached_main_sha(worker) == new_main

    async def test_stale_cache_raises_no_snapshot_false_positive(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Task 2357's stale-cache false positive is structurally gone (task 5302).

        A stale cache used to make a healthy frozen head read as a §5.3
        violation until a dispatch refreshed it.  The snapshot is now clean
        before the dispatch as well as after it.
        """
        old_main = 'OLD00000'
        new_main = 'NEW11111'
        git_ops.get_main_sha = AsyncMock(return_value=new_main)  # type: ignore[method-assign]
        worker = _make_worker(git_ops)
        worker._intake.note_main_sha(old_main)

        _, head_item = _make_fake_item(
            't-head', base_sha=new_main, merge_commit='c-head',
            config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(head_item, verifying=True))

        assert worker.snapshot()['two_layer_invariants'] == [], (
            'a stale last_known_main_sha must not make a non-dead head a violation'
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

        violations = worker.snapshot()['two_layer_invariants']
        assert violations == [], f'expected a clean snapshot after dispatch, got: {violations!r}'

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
        worker._intake.note_main_sha(old_main)

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
        assert _cached_main_sha(worker) == old_main


# ── DEFECT 2 (task 2357) regression lock: no-over-correction ─────────────────


@pytest.mark.asyncio
class TestNoOverCorrectionRegressionLock:
    """A healthy stack with a fresh cache is genuinely clean (task 2357).

    The genuine-drift row this class used to carry is gone with the task-5302
    split: post-dispatch drift is not a violation, and a dispatch-time
    divergence is the detector key's job.
    """

    async def test_healthy_chained_stack_with_fresh_cache_is_clean(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A healthy multi-entry chained frozen stack (head base==real main;
        each successor base==predecessor merge_commit) with
        last_known_main_sha set to the real current tip yields
        two_layer_invariants() == [] on both surfaces.
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
        worker._inflight.extend(
            _make_inflight_entry(i, verifying=True) for i in (item_0, item_1, item_2)
        )
        worker._intake.note_main_sha(real_main)

        assert worker.two_layer_invariants(real_main) == [], (
            'expected a healthy chained frozen stack to have no violations '
            'against the real main tip'
        )

        snap_violations = worker.snapshot()['two_layer_invariants']
        assert snap_violations == [], (
            f'expected snapshot() to be clean once last_known_main_sha is the '
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
    change makes `_verify_base_divergence_at_dispatch` (or the dispatch site
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

        assert isinstance(entry, InflightEntry), (
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

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
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
        assert isinstance(entry, InflightEntry) and entry.lease is not None, (
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

        with caplog.at_level(logging.WARNING, logger='orchestrator.merge_lane'):
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


# ── task 8524: a re-anchor's own detector class (plans/merge-lane-policy-prd.md D12) ──

_VB_DEAD = 'd' * 40


async def _straggler_dispatch(
    git_ops: GitOps, config: OrchestratorConfig, monkeypatch: pytest.MonkeyPatch,
    verifies: VerifyTasks, *, tip_status: InflightStatus | None,
) -> tuple[SpeculativeMergeWorker, InflightEntry, str]:
    """Dispatch a dead-based straggler that holds a permit, past one frontier
    entry whose verify is running and whose status is *tip_status*.  Returns
    the worker, the dispatched entry and that frontier entry's commit."""
    worker = _make_worker(git_ops)
    worker._host_allocator = _fake_local_allocator()
    monkeypatch.setattr(worker, '_cleanup_owned_merge_worktree', AsyncMock())
    monkeypatch.setattr(worker, '_run_inflight_verify', AsyncMock(
        return_value=InflightVerifyResult(outcome=None, merge_wt=Path('/fake/merge-wt')),
    ))
    _, tip_item = await _make_merged_item(git_ops, config, 'task/vb-tip', 'vb_tip.py', 't=1\n')
    worker._inflight.append(InflightEntry(
        item=tip_item, lease=None, verify_task=verifies.pending(),
        merge_wt=None, was_speculative=False, status=tip_status,
    ))
    worker._record_dead_base(_VB_DEAD, killer=DeadBaseKiller(DeadBaseSite.HEAD_FAILED, 'vb'))
    _, built = await _make_merged_item(git_ops, config, 'task/vb-strag', 'vb_s.py', 's=1\n')
    permit = await worker._speculation_ledger.acquire()
    straggler = dataclasses.replace(built, base_sha=_VB_DEAD, speculative=True, permit=permit)
    entry = await worker._dispatch_item(straggler)
    assert isinstance(entry, InflightEntry)
    tip = tip_item.merge_result.merge_commit
    assert tip is not None
    return worker, entry, tip


@pytest.mark.asyncio
class TestReanchorDivergenceAtTheWorker:
    """A recovery re-merge stacked on a live tip is judged like any stacked
    dispatch; a hit on one is ``reanchor_onto_dead_tip``, and one that fell to
    main stays exempt (plans/merge-lane-policy-prd.md D1, D12; §7 row 3)."""

    async def test_a_vacated_tip_owner_sends_the_reanchor_to_main_with_no_divergence(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        """Boundary row 3: the 3082 shape, a tip whose verify requeued."""
        worker, entry, _ = await _straggler_dispatch(
            git_ops, config, monkeypatch, verifies, tip_status=InflightStatus.REQUEUED,
        )

        assert entry.item.remerge_base == RemergeBase(None, RemergeReason.TIP_VACATED)
        worker._inflight.append(entry)
        assert worker.verify_base_divergences() == ()

    async def test_a_live_tip_reanchor_on_the_frozen_tip_records_no_divergence(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        _, entry, tip = await _straggler_dispatch(
            git_ops, config, monkeypatch, verifies, tip_status=None,
        )

        assert entry.item.remerge_base == RemergeBase(tip, RemergeReason.LIVE_TIP)
        assert entry.verify_base_divergence is None

    async def test_a_reanchor_whose_tip_owner_vacated_before_dispatch_is_its_own_class(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        worker = _make_worker(git_ops)
        _, owner = _make_fake_item(
            't-owner', base_sha='M0', merge_commit='C-tip', config=config, git_repo=git_repo,
        )
        worker._inflight.append(
            dataclasses.replace(_make_inflight_entry(owner), status=InflightStatus.REQUEUED),
        )
        _, item = _make_fake_item(
            't-reanchored', base_sha='C-tip', merge_commit='C-r', config=config, git_repo=git_repo,
        )
        item = dataclasses.replace(item, remerge_base=RemergeBase('C-tip', RemergeReason.LIVE_TIP))

        record = worker._verify_base_divergence_at_dispatch(item, 'M0')

        assert record is not None
        assert record.kind is VerifyBaseDivergenceKind.REANCHOR_ONTO_DEAD_TIP
        assert (record.base_sha, record.expected_tip) == ('C-tip', 'M0')
