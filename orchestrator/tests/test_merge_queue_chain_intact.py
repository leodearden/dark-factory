"""INV-3 chain-intact verdict enforcement (PRD plans/merge-verdict-integrity-prd.md leaf γ).

A verdict (PASS or FAIL) is adoptable iff, at adoption time, its verified
tree still equals ``current-main ∘ S`` for an ordered sequence S of
STILL-LIVE pipeline items (none failed/ejected/superseded/re-merged since
verify dispatch).  A broken-chain verdict is VOID: emit ``verdict_voided``
naming the dead link, re-merge + re-verify the item, and a FAIL from a dead
base NEVER blocks the task.  Deliberately verify-depth-agnostic: chain-intact,
NOT base==main.

This file is the boundary-test leaf (PRD §7).  It reuses the merge-queue unit
harness by bare module-name sibling import (no ``tests/__init__.py``):
  - ``_make_fake_item`` / ``_make_inflight_entry`` / ``_make_merged_item`` —
    pure builders (test_merge_queue_two_layer_integration).  ``_make_worker`` is
    LOCAL to this file: the sibling's forwards no collaborators.

Step-1 (this commit) covers the pure predicate + recorder foundation:
  * ``_chain_dead_link(item, main_sha)`` — the single shared chain-intact predicate.
  * ``_record_dead_base(*shas, killer)`` — the dead-commit ledger's write path
    (its bounded FIFO is pinned by test_merge_lane_dead_base_ledger.py).
Adoption / dispatch / cascade enforcement land in later steps.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from _merge_lane_fakes import VerifyTasks, lane_scene_config
from _merge_lane_fakes import verifies as verifies

# Bare sibling-import of the pure builders (no tests/__init__.py — tests dir is
# on sys.path; mirrors test_concurrent_verify_boundary importing from
# test_merge_queue_concurrent_verify).
from test_merge_queue_concurrent_verify import _one_remerge, _record_remerges
from test_merge_queue_two_layer_integration import (
    _make_fake_item,
    _make_inflight_entry,
    _make_merged_item,
)
from test_merge_queue_verify_base_invariant import _fake_local_allocator
from test_merge_speculation import _LateArrivalFakeEventStore

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventStore, EventType
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_lane.dead_base_ledger import DeadBaseKiller, DeadBaseSite
from orchestrator.merge_lane.remerge_base import RemergeClaim
from orchestrator.merge_lane.speculation_controller import PermitLedger, SpeculationController
from orchestrator.merge_lane.types import (
    CapPermit,
    InflightStatus,
    ItemLifecycleState,
    MergeRequest,
    NoHost,
    RemergeBase,
    RemergeReason,
    SpecPermit,
)
from orchestrator.merge_lane.worker import (
    DecidedItem,
    InflightEntry,
    InflightVerifyResult,
    MergeOutcome,
    RealMergeItem,
    SpeculativeItem,
    SpeculativeMergeWorker,
)
from orchestrator.verify_runner import HostLease

# ── Fixtures (mirrored from test_merge_queue_verify_base_invariant.py) ────────


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


# ── Fake SHAs + entry/allocator builders (mirror TestFinalizeInflightNonPass) ─

# 40-char fake SHAs, mutually distinct so base==main / base∈dead-set are
# unambiguous.
_DEAD = 'd' * 40
_KILLER = DeadBaseKiller(DeadBaseSite.HEAD_FAILED, 'chain-intact-killer')
_MAIN = 'a' * 40
# A LIVE deep predecessor merge_commit — != main and NOT in the dead set.  A
# verdict on this base is verify-depth-agnostic INTACT (never voided).
_LIVE_DEEP = 'e' * 40


def _make_worker(
    git_ops: GitOps, *, event_store: EventStore | None = None,
) -> SpeculativeMergeWorker:
    """Build a bare worker, optionally carrying an injected event store.

    ``event_store=`` is a real parameter of
    ``merge_queue.py::SpeculativeMergeWorker.__init__``, so the tests here hold
    the store they injected and read their speculative events straight off it.
    Local to this file rather than the sibling ``_make_worker``
    (test_merge_queue_two_layer_integration), which forwards nothing.
    """
    return SpeculativeMergeWorker(git_ops, asyncio.Queue(), event_store=event_store)


def _awaiting_host(worker: SpeculativeMergeWorker) -> list[str]:
    """Task ids of the items parked for re-dispatch, read from ``snapshot()``.

    ``merge_queue.py::SpeculativeMergeWorker.snapshot`` enumerates the
    front-priority re-dispatch queue as entries in state ``'awaiting_host'`` —
    the public view of the parking a voided verdict's re-merged replacement
    lands in.
    """
    return [
        entry['task_id'] for entry in worker.snapshot()['entries']
        if entry['state'] == 'awaiting_host'
    ]


def _make_mock_allocator() -> MagicMock:
    """A MagicMock allocator with async release/cancel_and_release (no real hosts)."""
    alloc = MagicMock()
    alloc.release = AsyncMock()
    alloc.cancel_and_release = AsyncMock()
    return alloc


def _make_fail_entry(
    item: RealMergeItem, lease: HostLease, fail_outcome: MergeOutcome, merge_wt_val: Path,
) -> InflightEntry:
    """InflightEntry whose verify_task resolves to a FAIL (outcome is not None)."""

    async def _fake_fail_verify() -> InflightVerifyResult:
        return InflightVerifyResult(outcome=fail_outcome, merge_wt=merge_wt_val, status=None)

    return InflightEntry(
        item=item,
        lease=lease,
        verify_task=asyncio.ensure_future(_fake_fail_verify()),
        merge_wt=merge_wt_val,
        was_speculative=False,
    )


def _make_pass_entry(
    item: RealMergeItem, lease: HostLease, merge_wt_val: Path,
) -> InflightEntry:
    """InflightEntry whose verify_task resolves to a PASS (outcome is None)."""

    async def _fake_pass_verify() -> InflightVerifyResult:
        return InflightVerifyResult(outcome=None, merge_wt=merge_wt_val, status=None)

    return InflightEntry(
        item=item,
        lease=lease,
        verify_task=asyncio.ensure_future(_fake_pass_verify()),
        merge_wt=merge_wt_val,
        was_speculative=False,
    )


def _frontier_entry(
    item: RealMergeItem,
    verify_task: asyncio.Task[InflightVerifyResult],
    *,
    status: InflightStatus | None = None,
) -> InflightEntry:
    """A verify-frontier entry whose verify is *verify_task* (the re-anchor
    decision reads its liveness)."""
    return InflightEntry(
        item=item,
        lease=None,
        verify_task=verify_task,
        merge_wt=None,
        was_speculative=False,
        status=status,
    )


def _remerge_claim(remerge: AsyncMock) -> RemergeClaim:
    """The claim the last awaited ``_remerge`` call was handed (task 8524)."""
    assert remerge.await_args is not None
    claim = remerge.await_args.kwargs.get('claim')
    assert isinstance(claim, RemergeClaim), remerge.await_args
    return claim


# ── _chain_dead_link predicate ───────────────────────────────────────────────


@pytest.mark.asyncio
class TestChainDeadLinkPredicate:
    """The single shared chain-intact predicate: a chain is dead iff the item's
    ``base_sha`` is a known-dead commit AND is not current main.  Depth-agnostic:
    only a DEAD base breaks the chain; a live deep base does not."""

    async def test_dead_base_not_main_returns_base(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """base_sha in the dead set and != main → broken chain, return the dead link."""
        worker = _make_worker(git_ops)
        _, item = _make_fake_item(
            't1', base_sha='deadcommit', merge_commit='mc1', config=config, git_repo=git_repo,
        )
        worker._record_dead_base('deadcommit', killer=_KILLER)
        assert worker._chain_dead_link(item, 'livemain') == 'deadcommit'

    async def test_base_equals_main_returns_none(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """base_sha == main → the base LANDED, chain intact, even if once recorded dead."""
        worker = _make_worker(git_ops)
        _, item = _make_fake_item(
            't2', base_sha='mainsha', merge_commit='mc2', config=config, git_repo=git_repo,
        )
        worker._record_dead_base('mainsha', killer=_KILLER)
        assert worker._chain_dead_link(item, 'mainsha') is None

    async def test_live_deep_base_returns_none(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """Verify-depth-agnostic: a live predecessor merge_commit (NOT dead) is intact."""
        worker = _make_worker(git_ops)
        _, item = _make_fake_item(
            't3', base_sha='livedeep', merge_commit='mc3', config=config, git_repo=git_repo,
        )
        # livedeep is NOT in the dead set → a deep multi-merge chain is intact.
        assert worker._chain_dead_link(item, 'livemain') is None

    async def test_decided_item_returns_none(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A DecidedItem/passthrough has no verified tree to adopt → always None."""
        worker = _make_worker(git_ops)
        _, item = _make_fake_item(
            't4', base_sha='deadcommit', merge_commit=None, config=config, git_repo=git_repo,
        )
        worker._record_dead_base('deadcommit', killer=_KILLER)
        assert worker._chain_dead_link(item, 'livemain') is None


# ── Adoption-time FAIL-void (the 5260 incident) + one-shot / no-livelock ──────


@pytest.mark.asyncio
class TestAdoptionFailVoid:
    """A FAIL verdict verified against a DEAD base is VOID, not adopted: the item
    is re-merged + re-parked, ``verdict_voided`` is emitted, and the phantom FAIL
    NEVER resolves (blocks) the request.  A FAIL on a LIVE (==main) base is a
    genuine failure and IS adopted normally (one-shot; cannot livelock)."""

    async def test_fail_from_dead_base_is_voided_and_never_blocks(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        # A real merged item, then point its base at a known-DEAD commit.
        _, item = await _make_merged_item(git_ops, config, 'task/void-fail', 'vf.py', 'x=1\n')
        item = dataclasses.replace(item, base_sha=_DEAD)
        req = item.request
        main_sha = await git_ops.get_main_sha()   # the REAL tip the chain check reads

        store = _LateArrivalFakeEventStore()
        worker = _make_worker(git_ops, event_store=store)
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._host_allocator = _make_mock_allocator()
        worker._record_dead_base(_DEAD, killer=_KILLER)

        # The re-merged (live-based) replacement _remerge would produce.
        _, remerged = _make_fake_item(
            'remerged-fail', base_sha=main_sha, merge_commit='rmc00000',
            config=config, git_repo=git_repo,
        )
        worker._remerge = AsyncMock(return_value=remerged)  # type: ignore[method-assign]

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = _make_fail_entry(
            item, lease, MergeOutcome('blocked', reason='verify fail'), item.merge_wt,
        )

        result = await worker._finalize_inflight(entry)

        assert result is False
        worker._remerge.assert_awaited_once()          # item re-merged against real main
        assert 'remerged-fail' in _awaiting_host(worker)  # re-parked for re-verify
        assert not req.result.done()                   # phantom FAIL never blocked the task
        assert worker._n_failed is False               # voided ≠ failed (no false cascade)
        voided = store.speculative_events(EventType.verdict_voided)
        assert len(voided) == 1
        assert voided[0]['data']['dead_link'] == _DEAD
        assert voided[0]['data']['reason'] == 'chain_dead'
        assert voided[0]['data']['point'] == 'adoption'

    async def test_fail_on_live_main_base_is_adopted_no_livelock(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        # base_sha == current main → chain intact even though an UNRELATED dead
        # commit sits in the ledger; a genuine FAIL IS adopted (one-shot).
        _, item = await _make_merged_item(git_ops, config, 'task/live-fail', 'lf.py', 'y=2\n')
        main_sha = await git_ops.get_main_sha()   # the REAL tip the chain check reads
        item = dataclasses.replace(item, base_sha=main_sha)
        req = item.request

        store = _LateArrivalFakeEventStore()
        worker = _make_worker(git_ops, event_store=store)
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._host_allocator = _make_mock_allocator()
        worker._record_dead_base(_DEAD, killer=_KILLER)  # unrelated dead commit
        worker._remerge = AsyncMock()  # type: ignore[method-assign]

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        fail_outcome = MergeOutcome('blocked', reason='genuine verify fail')
        entry = _make_fail_entry(item, lease, fail_outcome, item.merge_wt)

        result = await worker._finalize_inflight(entry)

        assert result is False
        worker._remerge.assert_not_awaited()           # NOT voided → not re-merged
        assert _awaiting_host(worker) == []            # nothing re-parked
        assert req.result.done()                       # genuine FAIL IS adopted (blocks)
        assert req.result.result().status == 'blocked'
        assert worker._n_failed is True
        assert store.speculative_events(EventType.verdict_voided) == []


# ── Adoption-time PASS-void + depth-agnostic non-regression ───────────────────


@pytest.mark.asyncio
class TestAdoptionPassVoid:
    """A PASS verdict verified against a DEAD base is VOID too: the dead-based
    tree is NEVER advanced onto main — the item is re-merged + re-parked and
    ``verdict_voided`` is emitted.  A PASS on a LIVE DEEP base (a predecessor
    merge_commit that is NOT dead) is INTACT and adopted through the normal CAS
    path (verify-depth-agnostic: chain-intact, NOT base==main)."""

    async def test_pass_from_dead_base_is_voided_and_never_advances_main(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        # A real merged item, then point its base at a known-DEAD commit.
        _, item = await _make_merged_item(git_ops, config, 'task/void-pass', 'vp.py', 'p=1\n')
        item = dataclasses.replace(item, base_sha=_DEAD)
        req = item.request
        # The REAL tip, both as the chain check's live main and as the fact the
        # 'never advanced' assertion is made against — no advance_main spy needed:
        # the dead-based tree not landing IS main's tip being unmoved.  The void
        # fires before the CAS loop, so MAX_CAS_RETRIES never comes into play.
        main_before = await git_ops.get_main_sha()

        store = _LateArrivalFakeEventStore()
        worker = _make_worker(git_ops, event_store=store)
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._host_allocator = _make_mock_allocator()
        worker._record_dead_base(_DEAD, killer=_KILLER)

        _, remerged = _make_fake_item(
            'remerged-pass', base_sha=main_before, merge_commit='rmc11111',
            config=config, git_repo=git_repo,
        )
        worker._remerge = AsyncMock(return_value=remerged)  # type: ignore[method-assign]

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = _make_pass_entry(item, lease, item.merge_wt)

        result = await worker._finalize_inflight(entry)

        assert result is False
        assert await git_ops.get_main_sha() == main_before  # dead tree never advanced
        worker._remerge.assert_awaited_once()              # re-merged against real main
        assert 'remerged-pass' in _awaiting_host(worker)   # re-parked for re-verify
        assert not req.result.done()                       # PASS void leaves req unresolved
        assert worker._n_failed is False                   # voided ≠ failed
        voided = store.speculative_events(EventType.verdict_voided)
        assert len(voided) == 1
        assert voided[0]['data']['dead_link'] == _DEAD
        assert voided[0]['data']['reason'] == 'chain_dead'
        assert voided[0]['data']['point'] == 'adoption'

    async def test_pass_on_live_deep_base_is_adopted_not_voided(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        # base_sha = a LIVE deep predecessor merge_commit — != main and NOT in
        # the dead set.  Depth-agnostic: a deep intact chain is adopted (reaches
        # the normal CAS advance path), NOT voided.  This must hold both before
        # and after the PASS-branch void is wired (a non-regression guard).
        _, item = await _make_merged_item(git_ops, config, 'task/deep-pass', 'dp.py', 'd=3\n')
        item = dataclasses.replace(item, base_sha=_LIVE_DEEP)
        req = item.request

        main_before = await git_ops.get_main_sha()

        store = _LateArrivalFakeEventStore()
        worker = _make_worker(git_ops, event_store=store)
        # Exit the CAS fast; we only assert the void branch is SKIPPED.
        worker.MAX_CAS_RETRIES = 0  # type: ignore[assignment]
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._host_allocator = _make_mock_allocator()
        worker._record_dead_base(_DEAD, killer=_KILLER)  # unrelated dead commit; _LIVE_DEEP is NOT in it
        worker._remerge = AsyncMock()  # type: ignore[method-assign]

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = _make_pass_entry(item, lease, item.merge_wt)

        await worker._finalize_inflight(entry)

        # NOT voided: the deep intact chain proceeds to the normal CAS path.
        worker._remerge.assert_not_awaited()
        assert _awaiting_host(worker) == []
        assert store.speculative_events(EventType.verdict_voided) == []
        # Proof the normal PASS/CAS path WAS reached, not voided.  The REAL CAS
        # ran: it compared-and-swapped against _LIVE_DEEP, which is not main's
        # tip, exhausted its 0-retry budget and RESOLVED the request.  A void
        # takes neither step — the dead-base sibling above asserts the exact
        # opposite pair (request unresolved, plus a verdict_voided event).
        assert req.result.done(), (
            'the CAS path must have been entered and resolved the request; '
            'a voided verdict leaves it unresolved'
        )
        assert req.result.result().status == 'blocked'
        assert await git_ops.get_main_sha() == main_before, (
            'the CAS expected-ref is the item\'s own _LIVE_DEEP base, which is '
            'not main, so the swap correctly refuses to move main'
        )


# ── Dispatch-time / host-acquisition dead-base re-check (enforcement (a)) ──────


@pytest.mark.asyncio
class TestDispatchDeadBaseRecheck:
    """A built-awaiting-host straggler whose base went DEAD is re-checked PER-ITEM
    at host-acquisition — regardless of the GLOBAL ``_has_inflight_verify`` flag
    (the exact 5260 gap: the old gate skipped the staleness re-merge whenever ANY
    other verify was in flight).  A doomed item is discarded + re-merged against
    real main instead of acquiring a host and burning a 40-min verify on a dead
    base.  An item on a LIVE base dispatches normally under the same state."""

    async def test_dead_base_rechecked_at_dispatch_despite_inflight_verify(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        main_sha = await git_ops.get_main_sha()   # the REAL tip the re-check reads
        store = _LateArrivalFakeEventStore()
        worker = _make_worker(git_ops, event_store=store)
        worker._record_dead_base(_DEAD, killer=_KILLER)
        worker._host_allocator = _fake_local_allocator()  # one free local slot
        worker._cleanup_owned_merge_worktree = AsyncMock()  # type: ignore[method-assign]
        worker._run_inflight_verify = AsyncMock(  # type: ignore[method-assign]
            return_value=InflightVerifyResult(outcome=None, merge_wt=Path('/fake/merge-wt')),
        )
        # _remerge returns a DecidedItem so the post-remerge passthrough returns
        # WITHOUT ever launching a verify (proves no verify burned on the dead base).
        _, remerged = _make_fake_item(
            'remerged-dispatch', base_sha=main_sha, merge_commit=None,
            config=config, git_repo=git_repo,
        )
        remerge = AsyncMock(return_value=remerged)
        monkeypatch.setattr(worker, '_remerge', remerge)

        # A head entry WITH a running verify → _has_inflight_verify is True (the
        # exact 5260 condition the old global-flag gate mis-handles).
        _, head = _make_fake_item(
            'head', base_sha=main_sha, merge_commit='headc', config=config, git_repo=git_repo,
        )
        assert isinstance(head, RealMergeItem)
        worker._inflight.append(_frontier_entry(head, verifies.pending()))

        # The built-awaiting-host straggler: base_sha points at a DEAD commit.
        _, item = _make_fake_item(
            'straggler', base_sha=_DEAD, merge_commit='stragc', config=config, git_repo=git_repo,
        )
        assert isinstance(item, RealMergeItem)

        await worker._dispatch_item(item)

        remerge.assert_awaited_once()          # re-merged despite _has_inflight_verify True
        # The live head's commit is the tip the replacement is claimed onto.
        assert _remerge_claim(remerge).anchor == RemergeBase('headc', RemergeReason.LIVE_TIP)
        worker._run_inflight_verify.assert_not_called()  # NO verify burned on the dead base
        voided = store.speculative_events(EventType.verdict_voided)
        assert len(voided) == 1
        assert voided[0]['data']['dead_link'] == _DEAD
        assert voided[0]['data']['reason'] == 'chain_dead'
        assert voided[0]['data']['point'] == 'dispatch'

    async def test_live_base_dispatches_normally_under_inflight_verify(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        # Control: base_sha == current main → chain intact; the item must dispatch
        # normally (verify launched, no remerge, no void) even though an UNRELATED
        # dead commit sits in the ledger and _has_inflight_verify is True.
        main_sha = await git_ops.get_main_sha()   # the REAL tip the re-check reads
        store = _LateArrivalFakeEventStore()
        worker = _make_worker(git_ops, event_store=store)
        worker._record_dead_base(_DEAD, killer=_KILLER)  # unrelated dead commit
        allocator = _fake_local_allocator()
        allocator.release = AsyncMock()  # the straggler's verdict frees its host
        worker._host_allocator = allocator
        worker._cleanup_owned_merge_worktree = AsyncMock()  # type: ignore[method-assign]
        worker._remerge = AsyncMock()  # type: ignore[method-assign]
        worker._run_inflight_verify = AsyncMock(  # type: ignore[method-assign]
            return_value=InflightVerifyResult(outcome=None, merge_wt=Path('/fake/merge-wt')),
        )

        _, head = _make_fake_item(
            'head2', base_sha=main_sha, merge_commit='headc2', config=config, git_repo=git_repo,
        )
        worker._inflight.append(_make_inflight_entry(head, verifying=True))

        _, item = _make_fake_item(
            'live-straggler', base_sha=main_sha, merge_commit='livec', config=config,
            git_repo=git_repo,
        )
        assert isinstance(item, RealMergeItem)

        entry = await worker._dispatch_item(item)
        assert isinstance(entry, InflightEntry) and entry.verify_task is not None
        await entry.verify_task

        worker._remerge.assert_not_awaited()             # intact chain → no remerge
        worker._run_inflight_verify.assert_called_once()  # verify launched normally
        assert store.speculative_events(EventType.verdict_voided) == []


# ── Head-failure cascade dangling-successor-edge population (enforcement (c)) ──

# Fake SHAs for the 5260 cascade topology, mutually distinct from the module
# constants above so base==main / base∈dead-set are unambiguous:
_HEAD_C = 'c' * 40   # the FAILED head's merge commit — dead once the head fails.
_DOWN_C = 'f' * 40   # the downstream speculative's OLD merge commit — dead once
#                      the cascade re-merges it away (a straggler stacked on it
#                      is left with a DANGLING successor edge).
_STRAG_C = '9' * 40  # a built-awaiting-host straggler's own merge commit.


def _make_remerge_to_decided_stub() -> AsyncMock:
    """AsyncMock for ``worker._remerge`` returning a DecidedItem passthrough.

    A re-merged item that is a :class:`DecidedItem` dispatches as a passthrough
    (verify_task=None) — no host is acquired and ``_run_inflight_verify`` is
    NEVER called — so the cascade / dispatch re-check can drive to a clean rest
    without a real repo or a real verify.  Bound to the request passed to
    ``_remerge`` so lifecycle/registry bookkeeping stays coherent.
    """

    def _side(
        request: object, started_monotonic: object = None, claim: object = None,
    ) -> DecidedItem:
        return DecidedItem(
            request=request,  # type: ignore[arg-type]
            immediate_outcome=MergeOutcome('conflict', reason='remerged-stub'),
            base_sha=_MAIN,
            speculative=False,
        )

    return AsyncMock(side_effect=_side)


async def _drive_cascade_recording(
    worker: object, until: Callable[[], bool], *, timeout_s: float = 5.0,
) -> None:
    """Drive ``_verifier_loop`` until *until* holds -- the head-failure cascade
    has re-merged what the caller is waiting on.

    The cascade is INLINE in ``_verifier_loop`` (not a callable method), so it is
    driven by pre-populating ``worker._inflight`` with a FAILED head + a
    downstream entry (both with ALREADY-RESOLVED verify_tasks so DISPATCH-FILL
    breaks straight to FINALIZE-HEAD) and running the loop as a task.  The loop
    is cancelled as soon as *until* holds — once the cascade has awaited
    ``_remerge``, step-10's ``_record_dead_base`` calls (which run BEFORE that
    remerge) have populated the ledger.  A persistent verifier-getter, if one
    was armed, is cancelled too so no pending task leaks.
    """
    loop_task = asyncio.ensure_future(worker._verifier_loop())  # type: ignore[attr-defined]
    try:
        for _ in range(int(timeout_s / 0.01)):
            if until():
                break
            await asyncio.sleep(0.01)
    finally:
        loop_task.cancel()
        with contextlib.suppress(BaseException):
            await loop_task
        _pvg = getattr(worker, '_pending_verifier_get', None)
        if _pvg is not None:
            _pvg.cancel()
            with contextlib.suppress(BaseException):
                await _pvg


@pytest.mark.asyncio
class TestCascadeStragglerCaughtComposition:
    """The headline 5260 end-to-end.

    The cascade's dead-base recording (c) — it records each invalidated
    predecessor commit as dead, so a built-awaiting-host straggler parked on the
    re-dispatch queue, which is invisible to the cascade and left with a
    DANGLING successor edge, becomes catchable at its next dispatch — composes
    with the dispatch-time dead-base re-check (a) so that straggler is CAUGHT:
    re-merged against real main, NOT verified for 43 min against a base the
    cascade already re-merged away.

    Two phases in one deterministic drive (no fragile host-availability timing):
      · Phase 1 — drive the head-failure cascade so the downstream's OLD merge
        commit DOWN_C is recorded dead (enforcement (c)).
      · Phase 2 — the straggler (base_sha == DOWN_C) finally gets a host and
        dispatches; the INV-3 dispatch re-check (enforcement (a)) reads the
        ledger DOWN_C landed in and VOIDs it.

    RED: phase 1 records nothing → in phase 2 ``_chain_dead_link`` returns None →
    the straggler burns a verify against the dead base (no ``verdict_voided``,
    ``_run_inflight_verify`` called).  GREEN (step-10): the straggler is caught.
    ``two_layer_invariants`` stays empty throughout: INV-3 is the invariant it
    reports over the frozen prefix, and the caught straggler never sits there.
    """

    async def test_straggler_on_dead_downstream_base_caught_at_dispatch(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        main_sha = await git_ops.get_main_sha()   # the REAL tip the chain check reads
        store = _LateArrivalFakeEventStore()
        worker = _make_worker(git_ops, event_store=store)
        worker._host_allocator = _make_mock_allocator()
        worker._release_or_cleanup = AsyncMock()  # type: ignore[method-assign]
        worker._cleanup_owned_merge_worktree = AsyncMock()  # type: ignore[method-assign]
        worker._abort_remote_verify = AsyncMock()  # type: ignore[method-assign]
        cascade_remerge = _make_remerge_to_decided_stub()
        monkeypatch.setattr(worker, '_remerge', cascade_remerge)
        worker._run_inflight_verify = AsyncMock(  # type: ignore[method-assign]
            return_value=InflightVerifyResult(outcome=None, merge_wt=Path('/fake/merge-wt')),
        )

        # ── Phase 1: drive the cascade so DOWN_C is recorded dead ─────────────
        _, head_item = _make_fake_item(
            'comp-head', base_sha=main_sha, merge_commit=_HEAD_C,
            config=config, git_repo=git_repo,
        )
        assert isinstance(head_item, RealMergeItem)
        head_entry = _make_fail_entry(
            head_item, HostLease(name='local', runner=MagicMock(), is_local=True),
            MergeOutcome('blocked', reason='head verify fail'), head_item.merge_wt,
        )
        _, down_item = _make_fake_item(
            'comp-down', base_sha=_HEAD_C, merge_commit=_DOWN_C,
            config=config, git_repo=git_repo,
        )
        assert isinstance(down_item, RealMergeItem)
        down_entry = _make_pass_entry(
            down_item, HostLease(name='remote', runner=MagicMock(), is_local=False),
            down_item.merge_wt,
        )
        worker._inflight.append(head_entry)
        worker._inflight.append(down_entry)
        await asyncio.sleep(0.05)
        await _drive_cascade_recording(worker, lambda: cascade_remerge.await_count >= 1)

        # Precondition for phase 2, and both ends of enforcement point (c): the
        # cascade records the FAILED head's own merge commit dead as well as the
        # downstream entries' — a straggler stacked directly on the head is
        # caught at dispatch only if the head's commit is in this set too.
        dead = worker._dead_base_commits
        assert _DOWN_C in dead, "the downstream entry's old merge commit must be dead"
        assert _HEAD_C in dead, "the FAILED head's own merge commit must be dead"

        # ── Phase 2: the built-awaiting-host straggler finally dispatches ─────
        # Its base_sha is DOWN_C — the commit the cascade re-merged away.  Give it
        # a free host slot (host-acquisition time).  The injected store stays the
        # constructor's; phase 2's events are read as the DELTA past this mark,
        # which isolates them exactly as swapping in a fresh store did.
        voided_before = len(store.speculative_events(EventType.verdict_voided))
        worker._host_allocator = _fake_local_allocator()
        remerge = _make_remerge_to_decided_stub()  # a fresh await_count
        monkeypatch.setattr(worker, '_remerge', remerge)
        _, straggler = _make_fake_item(
            'comp-straggler', base_sha=_DOWN_C, merge_commit=_STRAG_C,
            config=config, git_repo=git_repo,
        )
        assert isinstance(straggler, RealMergeItem)

        await worker._dispatch_item(straggler)

        # Caught at dispatch: re-merged under a claim, NO verify burned.
        remerge.assert_awaited_once()
        _remerge_claim(remerge)
        worker._run_inflight_verify.assert_not_called()
        voided = store.speculative_events(EventType.verdict_voided)[voided_before:]
        assert len(voided) == 1
        assert voided[0]['data']['dead_link'] == _DOWN_C
        assert voided[0]['data']['reason'] == 'chain_dead'
        assert voided[0]['data']['point'] == 'dispatch'

        # INV-3, the invariant two_layer_invariants reports, reads clean.
        assert worker.two_layer_invariants(main_sha) == []


# ── task 8524: a recovery re-merge lands on the base its claim names ─────────


def _ledger(worker: SpeculativeMergeWorker) -> PermitLedger[SpecPermit]:
    return worker._speculation_ledger


async def _remerge(
    worker: SpeculativeMergeWorker, req: MergeRequest, claim: RemergeClaim,
) -> SpeculativeItem:
    return await worker._remerge(req, None, claim=claim)


@pytest.mark.asyncio
class TestRemergeOntoTheClaimedBase:
    """``_remerge`` merges onto the claim's anchor and hands the replacement
    the claim's permit (plans/merge-lane-policy-prd.md D1)."""

    async def test_a_live_tip_claim_stacks_the_replacement_on_the_tip(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        worker = _make_worker(git_ops)
        _, item_a = await _make_merged_item(git_ops, config, 'task/tip-a', 'ta.py', 'a=1\n')
        req_b, _ = await _make_merged_item(git_ops, config, 'task/tip-b', 'tb.py', 'b=1\n')
        tip = item_a.merge_result.merge_commit
        assert tip is not None
        permit = await _ledger(worker).acquire()
        claim = RemergeClaim(RemergeBase(tip, RemergeReason.LIVE_TIP), permit)

        out = await _remerge(worker, req_b, claim)

        assert isinstance(out, RealMergeItem)
        assert out.base_sha == tip
        assert out.speculative is True
        assert out.permit is permit
        assert out.remerge_base == claim.anchor
        commit = out.merge_result.merge_commit
        assert commit is not None
        rc, _, _ = await _run(['git', 'merge-base', '--is-ancestor', tip, commit], cwd=git_repo)
        assert rc == 0, "the replacement's tree must contain the tip's"

    async def test_a_main_claim_merges_onto_main(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        worker = _make_worker(git_ops)
        req, _ = await _make_merged_item(git_ops, config, 'task/main-claim', 'mc.py', 'm=1\n')
        claim = RemergeClaim(RemergeBase(None, RemergeReason.TIP_DEAD), None)

        out = await _remerge(worker, req, claim)

        assert isinstance(out, RealMergeItem)
        assert out.base_sha == await git_ops.get_main_sha()
        assert out.speculative is False
        assert out.permit is None
        assert out.remerge_base == claim.anchor

    async def test_a_failed_remerge_releases_the_claimed_permit(
        self, git_ops: GitOps, config: OrchestratorConfig,
    ) -> None:
        import unittest.mock as mock

        worker = _make_worker(git_ops)
        req, _ = await _make_merged_item(git_ops, config, 'task/raise', 'rs.py', 'r=1\n')
        permit = await _ledger(worker).acquire()
        claim = RemergeClaim(RemergeBase('sha-tip', RemergeReason.LIVE_TIP), permit)

        with (
            mock.patch.object(
                worker, '_remerge_inner', AsyncMock(side_effect=RuntimeError('merge blew up')),
            ),
            pytest.raises(RuntimeError, match='merge blew up'),
        ):
            await _remerge(worker, req, claim)

        assert permit.released is True
        assert _ledger(worker).live == frozenset()


# ── task 8524: the dispatch sites re-anchor (consumers 4 and 5) ──────────────


def _dispatching_worker(
    git_ops: GitOps, store: _LateArrivalFakeEventStore, monkeypatch: pytest.MonkeyPatch,
) -> SpeculativeMergeWorker:
    """A bare worker with one free local host, whose verify never runs."""
    worker = _make_worker(git_ops, event_store=store)
    worker._host_allocator = _fake_local_allocator()
    monkeypatch.setattr(worker, '_cleanup_owned_merge_worktree', AsyncMock())
    monkeypatch.setattr(worker, '_run_inflight_verify', AsyncMock(
        return_value=InflightVerifyResult(outcome=None, merge_wt=Path('/fake/merge-wt')),
    ))
    return worker


async def _live_tip(
    worker: SpeculativeMergeWorker, git_ops: GitOps, config: OrchestratorConfig,
    verifies: VerifyTasks, *, status: InflightStatus | None = None,
) -> str:
    """Seed the verify frontier with one real merge whose verify is running;
    return its merge commit."""
    _, tip_item = await _make_merged_item(git_ops, config, 'task/disp-tip', 'disp_tip.py', 't=1\n')
    worker._inflight.append(_frontier_entry(tip_item, verifies.pending(), status=status))
    tip = tip_item.merge_result.merge_commit
    assert tip is not None
    return tip


async def _dead_based_straggler(
    worker: SpeculativeMergeWorker, git_ops: GitOps, config: OrchestratorConfig,
    filename: str = 'disp_s.py',
) -> tuple[RealMergeItem, SpecPermit]:
    """A speculative straggler adding *filename*, built on a recorded-dead
    commit and holding a permit."""
    worker._record_dead_base(_DEAD, killer=_KILLER)
    _, built = await _make_merged_item(git_ops, config, 'task/disp-strag', filename, 's=1\n')
    permit = await _ledger(worker).acquire()
    straggler = dataclasses.replace(built, base_sha=_DEAD, speculative=True, permit=permit)
    return straggler, permit


async def _dispatch(worker: SpeculativeMergeWorker, item: RealMergeItem) -> InflightEntry | NoHost:
    return await worker._dispatch_item(item)


def _the_row(store: _LateArrivalFakeEventStore, event_type: EventType) -> dict[str, object]:
    rows = store.speculative_events(event_type)
    assert len(rows) == 1, rows
    return rows[0]['data']


def _replacement(entry: InflightEntry | NoHost) -> RealMergeItem:
    assert isinstance(entry, InflightEntry), entry
    assert isinstance(entry.item, RealMergeItem), entry.item
    return entry.item


@pytest.mark.asyncio
class TestDispatchSitesReanchor:
    """A dead-base straggler (consumer 4) and a Mechanism-2 stale discard
    (consumer 5) re-merge onto the claimed base, and the triggering row names
    the anchor (plans/merge-lane-policy-prd.md D1, §7 rows 1 and 3)."""

    async def test_a_dead_based_straggler_restacks_on_the_live_tip(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        store = _LateArrivalFakeEventStore()
        worker = _dispatching_worker(git_ops, store, monkeypatch)
        tip = await _live_tip(worker, git_ops, config, verifies)
        straggler, permit = await _dead_based_straggler(worker, git_ops, config)

        entry = await _dispatch(worker, straggler)

        replacement = _replacement(entry)
        assert replacement.base_sha == tip
        assert replacement.speculative is True
        assert replacement.remerge_base == RemergeBase(tip, RemergeReason.LIVE_TIP)
        assert isinstance(entry, InflightEntry) and entry.permit is permit
        row = _the_row(store, EventType.verdict_voided)
        assert row['point'] == 'dispatch'
        assert (row['anchor_reason'], row['anchor_base']) == ('live_tip', tip)

    async def test_with_no_live_tip_the_straggler_falls_to_main_and_frees_its_permit(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = _LateArrivalFakeEventStore()
        worker = _dispatching_worker(git_ops, store, monkeypatch)
        straggler, permit = await _dead_based_straggler(worker, git_ops, config)

        entry = await _dispatch(worker, straggler)

        replacement = _replacement(entry)
        assert replacement.base_sha == await git_ops.get_main_sha()
        assert replacement.speculative is False
        assert replacement.remerge_base == RemergeBase(None, RemergeReason.NO_TIP)
        assert permit.released is True
        assert permit not in _ledger(worker).live
        assert isinstance(entry, InflightEntry) and entry.permit is None
        row = _the_row(store, EventType.verdict_voided)
        assert (row['anchor_reason'], row['anchor_base']) == ('no_tip', None)

    async def test_a_hostless_dispatch_parks_the_replacement_not_the_original(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        """Driven through ``_verifier_loop``: the re-merge falls to main and
        frees the permit, then the free host is taken before the acquire.  The
        replacement is parked, and it re-dispatches as-is, holding no
        speculation slot."""
        store = _LateArrivalFakeEventStore()
        worker = _dispatching_worker(git_ops, store, monkeypatch)
        straggler, permit = await _dead_based_straggler(worker, git_ops, config)
        allocator = _fake_local_allocator()
        lease = allocator.acquire.return_value

        def _slot_taken_meanwhile(*_args: object, **_kwargs: object) -> None:
            allocator.free_host_count.return_value = 0

        allocator.acquire.side_effect = _slot_taken_meanwhile
        monkeypatch.setattr(worker, '_host_allocator', allocator)
        worker._register_item(straggler, ItemLifecycleState.REDISPATCH_PARKED)
        worker._intake.request_redispatch(straggler)

        await _drive_cascade_recording(worker, lambda: allocator.acquire.await_count >= 1)

        parked = _parked(worker, straggler.request.task_id)
        assert isinstance(parked, RealMergeItem) and parked is not straggler
        assert permit.released is True
        assert parked.base_sha == await git_ops.get_main_sha()
        assert (parked.speculative, parked.permit) == (False, None)
        assert parked.remerge_base == RemergeBase(None, RemergeReason.NO_TIP)
        assert _awaiting_host(worker) == [straggler.request.task_id]
        allocator.acquire.side_effect = None
        allocator.acquire.return_value = lease
        allocator.free_host_count.return_value = 1
        await _live_tip(worker, git_ops, config, verifies)
        entry = await _dispatch(worker, parked)
        assert isinstance(entry, InflightEntry) and entry.item is parked
        assert entry.permit is None and not _ledger(worker).live

    async def test_losing_the_host_race_parks_an_ordinary_item_without_its_cap(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-speculative item holding a merger-ahead cap finds a free host,
        then loses it before the acquire.  It parks with the cap released and
        cleared, rather than raising out of NoHost."""
        worker = _dispatching_worker(git_ops, _LateArrivalFakeEventStore(), monkeypatch)
        allocator = _fake_local_allocator()
        allocator.acquire.return_value = None
        monkeypatch.setattr(worker, '_host_allocator', allocator)
        cap_ledger = PermitLedger(asyncio.Semaphore(1), 1, token_factory=CapPermit)
        monkeypatch.setattr(worker, '_merge_ahead_ledger', cap_ledger)
        _, built = await _make_merged_item(git_ops, config, 'task/no-host', 'nh.py', 'n=1\n')
        item = dataclasses.replace(built, cap_permit=await cap_ledger.acquire())

        outcome = await _dispatch(worker, item)

        assert isinstance(outcome, NoHost), outcome
        assert outcome.item.cap_permit is None
        assert outcome.item.request is item.request
        assert cap_ledger.live == frozenset()

    async def test_a_launch_that_raises_after_a_re_anchor_frees_the_replacement(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        """A live-tip claim took a fresh slot for the replacement, then the
        host acquire raised.  The caller resolves only the item it handed in,
        so the dispatch itself frees the replacement's permit and worktree."""
        store = _LateArrivalFakeEventStore()
        worker = _dispatching_worker(git_ops, store, monkeypatch)
        await _live_tip(worker, git_ops, config, verifies)
        straggler, permit = await _dead_based_straggler(worker, git_ops, config)
        _ledger(worker).release(permit)
        allocator = _fake_local_allocator()
        allocator.acquire.side_effect = RuntimeError('acquire blew up')
        monkeypatch.setattr(worker, '_host_allocator', allocator)
        cleanup = AsyncMock()
        monkeypatch.setattr(worker, '_cleanup_owned_merge_worktree', cleanup)

        with pytest.raises(RuntimeError, match='acquire blew up'):
            await _dispatch(worker, straggler)

        assert _ledger(worker).live == frozenset()
        cleaned = [call.args[0] for call in cleanup.await_args_list]
        assert len(cleaned) == 2, cleaned
        assert cleaned[0] == straggler.merge_wt
        assert cleaned[1] != straggler.merge_wt

    async def test_a_launch_that_raises_moves_the_merger_off_the_orphaned_replacement(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        """The merger held the straggler's commit, so the re-merge re-pointed
        it at the replacement; the launch then raised.  The replacement's
        commit is recorded dead and the merger moves back to the live tip,
        rather than stacking its next request on a commit nothing will land."""
        worker = _dispatching_worker(git_ops, _LateArrivalFakeEventStore(), monkeypatch)
        tip = await _live_tip(worker, git_ops, config, verifies)
        straggler, permit = await _dead_based_straggler(worker, git_ops, config)
        _ledger(worker).release(permit)
        await _merger_holding(worker, config, git_repo, _straggler_commit(straggler))
        allocator = _fake_local_allocator()
        allocator.acquire.side_effect = RuntimeError('acquire blew up')
        monkeypatch.setattr(worker, '_host_allocator', allocator)
        cleanup = AsyncMock()
        monkeypatch.setattr(worker, '_cleanup_owned_merge_worktree', cleanup)

        with pytest.raises(RuntimeError, match='acquire blew up'):
            await _dispatch(worker, straggler)

        replacement_wt = cleanup.await_args_list[-1].args[0]
        rc, orphan, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=replacement_wt)
        assert rc == 0
        assert orphan.strip() in worker._dead_base_commits
        assert _speculation(worker)['spec_base'] == tip

    async def test_a_re_anchor_that_conflicts_with_the_live_tip_is_a_terminal_conflict(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        """A straggler whose branch clashes with the live tip's resolves as a
        terminal conflict, as the merger's own speculative merges onto an
        unverified base do; its permit rides the passthrough to finalize."""
        store = _LateArrivalFakeEventStore()
        worker = _dispatching_worker(git_ops, store, monkeypatch)
        tip = await _live_tip(worker, git_ops, config, verifies)
        straggler, permit = await _dead_based_straggler(
            worker, git_ops, config, filename='disp_tip.py',
        )

        entry = await _dispatch(worker, straggler)

        assert isinstance(entry, InflightEntry) and isinstance(entry.item, DecidedItem)
        assert entry.passthrough_outcome is not None
        assert entry.passthrough_outcome.status == 'conflict'
        assert entry.item.remerge_base == RemergeBase(tip, RemergeReason.LIVE_TIP)
        assert entry.permit is permit

    async def test_a_released_held_permit_is_never_reused_by_a_live_tip_claim(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        store = _LateArrivalFakeEventStore()
        worker = _dispatching_worker(git_ops, store, monkeypatch)
        tip = await _live_tip(worker, git_ops, config, verifies)
        straggler, permit = await _dead_based_straggler(worker, git_ops, config)
        _ledger(worker).release(permit)

        entry = await _dispatch(worker, straggler)

        assert _replacement(entry).remerge_base == RemergeBase(tip, RemergeReason.LIVE_TIP)
        assert isinstance(entry, InflightEntry) and entry.permit is not permit
        assert entry.permit in _ledger(worker).live

    async def test_a_vacated_tip_owner_sends_the_straggler_to_main(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        """Boundary row 3 at the worker: the 3082 shape, a frozen tip whose
        verify requeued its request."""
        store = _LateArrivalFakeEventStore()
        worker = _dispatching_worker(git_ops, store, monkeypatch)
        await _live_tip(worker, git_ops, config, verifies, status=InflightStatus.REQUEUED)
        straggler, permit = await _dead_based_straggler(worker, git_ops, config)

        entry = await _dispatch(worker, straggler)

        replacement = _replacement(entry)
        assert replacement.base_sha == await git_ops.get_main_sha()
        assert replacement.remerge_base == RemergeBase(None, RemergeReason.TIP_VACATED)
        assert permit.released is True
        assert isinstance(entry, InflightEntry) and entry.permit is None
        row = _the_row(store, EventType.verdict_voided)
        assert row['anchor_reason'] == 'tip_vacated'

    async def test_a_mechanism_2_discard_row_names_the_anchor(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        store = _LateArrivalFakeEventStore()
        worker = _dispatching_worker(git_ops, store, monkeypatch)
        _, built = await _make_merged_item(git_ops, config, 'task/disp-stale', 'disp_m.py', 'm=1\n')
        stale = dataclasses.replace(built, base_sha=_LIVE_DEEP)  # main moved since the build

        entry = await _dispatch(worker, stale)

        assert _replacement(entry).remerge_base == RemergeBase(None, RemergeReason.NO_TIP)
        row = _the_row(store, EventType.speculative_discard)
        assert row['reason'] == 'main_advanced'
        assert (row['anchor_reason'], row['anchor_base']) == ('no_tip', None)


# ── task 8524: the adoption-time void re-anchors (consumer 2) ────────────────


def _parked(worker: SpeculativeMergeWorker, task_id: str) -> SpeculativeItem:
    """The one item parked for re-dispatch under *task_id*."""
    parked = [i for i in worker._intake.pending_redispatch() if i.request.task_id == task_id]
    assert len(parked) == 1, parked
    return parked[0]


async def _voided_entry(
    worker: SpeculativeMergeWorker, git_ops: GitOps, config: OrchestratorConfig,
) -> tuple[InflightEntry, SpecPermit]:
    """A FAILed verdict built on a recorded-dead base, whose entry holds a permit."""
    _, built = await _make_merged_item(git_ops, config, 'task/adopt-x', 'adopt_x.py', 'x=1\n')
    permit = await _ledger(worker).acquire()
    item = dataclasses.replace(built, base_sha=_DEAD, permit=permit)
    worker._register_owned_merge_worktree(item.merge_wt)
    worker._host_allocator = _make_mock_allocator()
    worker._record_dead_base(_DEAD, killer=_KILLER)
    lease = HostLease(name='local', runner=MagicMock(), is_local=True)
    entry = _make_fail_entry(item, lease, MergeOutcome('blocked', reason='verify fail'), item.merge_wt)
    return dataclasses.replace(entry, permit=permit), permit


async def _finalize(worker: SpeculativeMergeWorker, entry: InflightEntry) -> bool:
    return await worker._finalize_inflight(entry)


def _stub_cascade_teardown(worker: SpeculativeMergeWorker, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the head-failure cascade run with no worktree, remote host or free
    verify host: its re-merged items park instead of dispatching."""
    allocator = _make_mock_allocator()
    allocator.free_host_count.return_value = 0
    monkeypatch.setattr(worker, '_host_allocator', allocator)
    for name in ('_release_or_cleanup', '_cleanup_owned_merge_worktree', '_abort_remote_verify'):
        monkeypatch.setattr(worker, name, AsyncMock())


@pytest.mark.asyncio
class TestAdoptionVoidReanchor:
    """A voided verdict's replacement re-anchors only on a tip that outlives
    the head-failure cascade its void sets off; with none it falls to main and
    frees the entry's permit (plans/merge-lane-policy-prd.md D1)."""

    async def test_an_unrelated_inflight_entry_dies_in_the_cascade_so_the_replacement_falls_to_main(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        """A voided head returns False, so the head-failure cascade tears down
        every entry behind it: an unrelated W behind the voided X is no tip
        for X's replacement to stack on."""
        store = _LateArrivalFakeEventStore()
        worker = _make_worker(git_ops, event_store=store)
        entry, permit = await _voided_entry(worker, git_ops, config)
        worker._inflight.append(entry)
        w_commit = await _live_tip(worker, git_ops, config, verifies)
        _stub_cascade_teardown(worker, monkeypatch)
        remerges = _record_remerges(worker, monkeypatch)

        await _drive_cascade_recording(
            worker, lambda: {'task/adopt-x', 'task/disp-tip'} <= {c[0] for c in remerges},
        )

        claim, replacement = _one_remerge(remerges, 'task/adopt-x')
        assert claim.anchor == RemergeBase(None, RemergeReason.TIP_DEAD)
        assert claim.anchor.tip != w_commit
        assert isinstance(replacement, RealMergeItem)
        assert replacement.base_sha == await git_ops.get_main_sha()
        assert replacement.speculative is False
        assert replacement.permit is None
        assert permit.released is True
        adoption_rows = [
            row['data'] for row in store.speculative_events(EventType.verdict_voided)
            if row['data']['point'] == 'adoption'
        ]
        assert [(r['anchor_reason'], r['anchor_base']) for r in adoption_rows] == [
            ('tip_dead', None),
        ]
        assert w_commit in worker._dead_base_commits
        assert worker.speculation_accounting_violations() == []

    async def test_a_successor_built_on_the_voided_commit_sends_it_to_main(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        verifies: VerifyTasks,
    ) -> None:
        """The circularity ground: the only tip on offer is stacked on the
        commit being voided, so it dies with it."""
        store = _LateArrivalFakeEventStore()
        worker = _make_worker(git_ops, event_store=store)
        entry, permit = await _voided_entry(worker, git_ops, config)
        assert isinstance(entry.item, RealMergeItem)
        _, successor = _make_fake_item(
            'adopt-y', base_sha=entry.item.merge_result.merge_commit or '',
            merge_commit='7' * 40, config=config, git_repo=git_repo,
        )
        assert isinstance(successor, RealMergeItem)
        worker._inflight.append(_frontier_entry(successor, verifies.pending()))

        assert await _finalize(worker, entry) is False

        replacement = _parked(worker, 'task/adopt-x')
        assert replacement.base_sha == await git_ops.get_main_sha()
        assert replacement.speculative is False
        assert replacement.remerge_base == RemergeBase(None, RemergeReason.TIP_DEAD)
        assert replacement.permit is None
        assert permit.released is True
        row = _the_row(store, EventType.verdict_voided)
        assert (row['anchor_reason'], row['anchor_base']) == ('tip_dead', None)


async def _runner_unavailable_entry(
    worker: SpeculativeMergeWorker, git_ops: GitOps, config: OrchestratorConfig,
    verifies: VerifyTasks,
) -> tuple[InflightEntry, SpecPermit]:
    """An entry whose remote verify host died mid-verify, holding a permit."""
    _, built = await _make_merged_item(git_ops, config, 'task/ru-x', 'ru_x.py', 'x=1\n')
    permit = await _ledger(worker).acquire()
    item = dataclasses.replace(built, permit=permit)
    unavailable = InflightVerifyResult(
        outcome=None, merge_wt=item.merge_wt,
        status=InflightStatus.RUNNER_UNAVAILABLE, reason='ssh timeout',
    )
    entry = InflightEntry(
        item=item, lease=None, verify_task=await verifies.finished(unavailable),
        merge_wt=item.merge_wt, was_speculative=False, permit=permit,
    )
    return entry, permit


@pytest.mark.asyncio
class TestRunnerUnavailableReanchor:
    """A RUNNER_UNAVAILABLE re-merge decides against the frontier that
    survives the head-failure cascade its False return sets off
    (plans/merge-lane-policy-prd.md D1)."""

    async def test_an_unrelated_inflight_entry_is_no_tip_for_the_replacement(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        worker = _make_worker(git_ops)
        monkeypatch.setattr(worker, '_release_or_cleanup', AsyncMock())
        await _live_tip(worker, git_ops, config, verifies)
        entry, permit = await _runner_unavailable_entry(worker, git_ops, config, verifies)

        assert await _finalize(worker, entry) is False

        replacement = _parked(worker, 'task/ru-x')
        assert replacement.base_sha == await git_ops.get_main_sha()
        assert replacement.remerge_base == RemergeBase(None, RemergeReason.TIP_DEAD)
        assert replacement.permit is None
        assert permit.released is True
        assert _ledger(worker).live == frozenset()


# ── task 8524: a dead base re-anchors the merger's held base (D2) ────────────

_TIP_T = '3' * 40       # a live frontier entry's merge commit
_HELD_DEAD = '4' * 40   # the base the merger holds, about to be recorded dead


def _controller(worker: SpeculativeMergeWorker) -> SpeculationController:
    return worker._speculation_controller


async def _merger_holding(
    worker: SpeculativeMergeWorker, config: OrchestratorConfig, git_repo: Path, base: str,
) -> None:
    """Put the merger in state A: a look-ahead found, its permit retained,
    stacked on *base*."""
    controller = _controller(worker)
    await controller.acquire_for_lookahead()
    req, _ = _make_fake_item('merger-next', config=config, git_repo=git_repo)
    controller.on_lookahead_found(req, base)


def _seed_live_tip(
    worker: SpeculativeMergeWorker, config: OrchestratorConfig, git_repo: Path,
    verifies: VerifyTasks,
) -> None:
    _, item = _make_fake_item(
        'merger-tip', base_sha=_MAIN, merge_commit=_TIP_T, config=config, git_repo=git_repo,
    )
    assert isinstance(item, RealMergeItem)
    worker._inflight.append(_frontier_entry(item, verifies.pending()))


def _speculation(worker: SpeculativeMergeWorker) -> dict[str, object]:
    return worker.snapshot()['speculation']


@pytest.mark.asyncio
class TestDeadBaseReanchorsTheMerger:
    """Recording a dead base re-points the merger's held base at the live
    frontier tip, or lets the permit go (plans/merge-lane-policy-prd.md D2)."""

    async def test_a_dead_held_base_moves_to_the_live_tip(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        verifies: VerifyTasks,
    ) -> None:
        worker = _make_worker(git_ops)
        _seed_live_tip(worker, config, git_repo, verifies)
        await _merger_holding(worker, config, git_repo, _HELD_DEAD)

        worker._record_dead_base(_HELD_DEAD, killer=_KILLER)

        spec = _speculation(worker)
        assert spec['spec_base'] == _TIP_T
        assert spec['held_by_merger'] == 1

    async def test_with_no_live_tip_the_held_permit_is_released(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        worker = _make_worker(git_ops)
        await _merger_holding(worker, config, git_repo, _HELD_DEAD)
        assert _speculation(worker)['slot_available'] == 0

        worker._record_dead_base(_HELD_DEAD, killer=_KILLER)

        spec = _speculation(worker)
        assert spec['spec_base'] is None
        assert spec['held_by_merger'] == 0
        assert spec['slot_available'] == spec['depth']

    async def test_recording_an_item_commit_dead_reaches_the_same_hook(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        verifies: VerifyTasks,
    ) -> None:
        worker = _make_worker(git_ops)
        _seed_live_tip(worker, config, git_repo, verifies)
        await _merger_holding(worker, config, git_repo, _HELD_DEAD)
        _, held = _make_fake_item(
            'merger-held', base_sha=_MAIN, merge_commit=_HELD_DEAD,
            config=config, git_repo=git_repo,
        )

        worker._record_item_commit_dead(held, DeadBaseSite.DISPATCH_VOID)

        assert _speculation(worker)['spec_base'] == _TIP_T

    async def test_an_idle_merger_is_untouched(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        verifies: VerifyTasks,
    ) -> None:
        worker = _make_worker(git_ops)
        _seed_live_tip(worker, config, git_repo, verifies)
        before = _speculation(worker)

        worker._record_dead_base(_HELD_DEAD, killer=_KILLER)

        assert _speculation(worker) == before

    async def test_a_void_does_not_repoint_the_merger_at_a_doomed_inflight_entry(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        """The merger holds a look-ahead stacked on X, and X's verdict is voided.
        The unrelated W behind X dies in the cascade X's void sets off, so the
        merger's held base goes nowhere and its permit is let go."""
        # Depth 2: room for X's permit and the merger's.
        worker = SpeculativeMergeWorker(git_ops, asyncio.Queue(), speculation_depth=2)
        monkeypatch.setattr(worker, '_release_or_cleanup', AsyncMock())
        await _live_tip(worker, git_ops, config, verifies)
        entry, _ = await _voided_entry(worker, git_ops, config)
        assert isinstance(entry.item, RealMergeItem)
        x_commit = entry.item.merge_result.merge_commit
        assert x_commit is not None
        await _merger_holding(worker, config, git_repo, x_commit)

        assert await _finalize(worker, entry) is False

        spec = _speculation(worker)
        assert spec['spec_base'] is None
        assert spec['held_by_merger'] == 0


def _straggler_commit(straggler: RealMergeItem) -> str:
    commit = straggler.merge_result.merge_commit
    assert commit is not None
    return commit


@pytest.mark.asyncio
class TestADispatchVoidHandsTheMergerTheReplacement:
    """A merger holding a dispatch-voided item's commit stacks after that
    item's replacement, not beside it on the tip the replacement was built
    on (plans/merge-lane-policy-prd.md D2)."""

    async def test_a_pending_merger_re_points_at_the_replacement(
        self, git_ops: GitOps, config: OrchestratorConfig,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        worker = _dispatching_worker(git_ops, _LateArrivalFakeEventStore(), monkeypatch)
        tip = await _live_tip(worker, git_ops, config, verifies)
        straggler, _ = await _dead_based_straggler(worker, git_ops, config)
        _controller(worker).on_lookahead_pending(
            _straggler_commit(straggler), straggler.request,
        )

        replacement = _replacement(await _dispatch(worker, straggler))

        spec = _speculation(worker)
        assert replacement.base_sha == tip
        assert spec['pending_spec_base'] == replacement.merge_result.merge_commit
        assert spec['pending_spec_base'] != tip
        assert spec['pending_predecessor_task_id'] == replacement.request.task_id

    async def test_a_merger_in_state_a_re_points_at_the_replacement(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        worker = _dispatching_worker(git_ops, _LateArrivalFakeEventStore(), monkeypatch)
        await _live_tip(worker, git_ops, config, verifies)
        straggler, _ = await _dead_based_straggler(worker, git_ops, config)
        next_req, _ = _make_fake_item('merger-next', config=config, git_repo=git_repo)
        _controller(worker).on_lookahead_found(next_req, _straggler_commit(straggler))

        replacement = _replacement(await _dispatch(worker, straggler))

        assert _speculation(worker)['spec_base'] == replacement.merge_result.merge_commit

    async def test_a_raising_re_merge_still_re_points_the_merger_at_the_live_tip(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
        monkeypatch: pytest.MonkeyPatch, verifies: VerifyTasks,
    ) -> None:
        worker = _dispatching_worker(git_ops, _LateArrivalFakeEventStore(), monkeypatch)
        tip = await _live_tip(worker, git_ops, config, verifies)
        straggler, _ = await _dead_based_straggler(worker, git_ops, config)
        next_req, _ = _make_fake_item('merger-next', config=config, git_repo=git_repo)
        _controller(worker).on_lookahead_found(next_req, _straggler_commit(straggler))
        monkeypatch.setattr(
            worker, '_remerge_inner', AsyncMock(side_effect=RuntimeError('merge blew up')),
        )

        with pytest.raises(RuntimeError, match='merge blew up'):
            await _dispatch(worker, straggler)

        assert _speculation(worker)['spec_base'] == tip
