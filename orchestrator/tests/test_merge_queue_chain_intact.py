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
  - ``_make_worker`` / ``_make_fake_item`` — pure builders (test_merge_queue_two_layer_integration).

Step-1 (this commit) covers the pure predicate + recorder foundation:
  * ``_chain_dead_link(item, main_sha)`` — the single shared chain-intact predicate.
  * ``_record_dead_base(*shas)`` — the bounded-FIFO dead-commit ledger.
Adoption / dispatch / cascade enforcement land in later steps.
"""

from __future__ import annotations

import asyncio
import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.event_store import EventType
from orchestrator.git_ops import AdvanceOutcome, GitOps, _run
from orchestrator.merge_queue import (
    InflightEntry,
    InflightVerifyResult,
    MergeOutcome,
    RealMergeItem,
)
from orchestrator.verify_runner import HostLease

# Bare sibling-import of the pure builders (no tests/__init__.py — tests dir is
# on sys.path; mirrors test_concurrent_verify_boundary importing from
# test_merge_queue_concurrent_verify).
from test_merge_queue_two_layer_integration import (
    _make_fake_item,
    _make_merged_item,
    _make_worker,
)
from test_merge_speculation import _LateArrivalFakeEventStore

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
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# ── Fake SHAs + entry/allocator builders (mirror TestFinalizeInflightNonPass) ─

# 40-char fake SHAs, mutually distinct so base==main / base∈dead-set are
# unambiguous.
_DEAD = 'd' * 40
_MAIN = 'a' * 40
# A LIVE deep predecessor merge_commit — != main and NOT in the dead set.  A
# verdict on this base is verify-depth-agnostic INTACT (never voided).
_LIVE_DEEP = 'e' * 40


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
        worker._dead_base_commits.add('deadcommit')
        assert worker._chain_dead_link(item, 'livemain') == 'deadcommit'

    async def test_base_equals_main_returns_none(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """base_sha == main → the base LANDED, chain intact, even if once recorded dead."""
        worker = _make_worker(git_ops)
        _, item = _make_fake_item(
            't2', base_sha='mainsha', merge_commit='mc2', config=config, git_repo=git_repo,
        )
        worker._dead_base_commits.add('mainsha')
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
        worker._dead_base_commits.add('deadcommit')
        assert worker._chain_dead_link(item, 'livemain') is None


# ── _record_dead_base bounded FIFO ledger ────────────────────────────────────


class TestRecordDeadBase:
    """The dead-commit ledger is a bounded FIFO: past the cap the oldest recorded
    commit is evicted; empty/blank/duplicate SHAs are no-ops."""

    def test_record_and_bounded_fifo_eviction(self, git_ops: GitOps) -> None:
        worker = _make_worker(git_ops)
        worker._dead_base_commits_cap = 3
        # cap+1 distinct SHAs → oldest (c0) evicted; size clamps to cap.
        worker._record_dead_base('c0', 'c1', 'c2', 'c3')
        assert len(worker._dead_base_commits) == 3
        assert 'c0' not in worker._dead_base_commits
        assert {'c1', 'c2', 'c3'} <= worker._dead_base_commits

    def test_record_skips_empty_blank_and_dupes(self, git_ops: GitOps) -> None:
        worker = _make_worker(git_ops)
        # '' and '  ' (strips to empty) are skipped; the duplicate 'x' collapses.
        worker._record_dead_base('', 'x', 'x', '  ')
        assert worker._dead_base_commits == {'x'}


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

        worker = _make_worker(git_ops)
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._host_allocator = _make_mock_allocator()
        worker._event_store = _LateArrivalFakeEventStore()
        worker._dead_base_commits.add(_DEAD)
        worker._git_ops.get_main_sha = AsyncMock(return_value=_MAIN)  # type: ignore[method-assign]

        # The re-merged (live-based) replacement _remerge would produce.
        _, remerged = _make_fake_item(
            'remerged-fail', base_sha=_MAIN, merge_commit='rmc00000',
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
        assert remerged in worker._redispatch          # re-parked for re-verify
        assert not req.result.done()                   # phantom FAIL never blocked the task
        assert worker._n_failed is False               # voided ≠ failed (no false cascade)
        voided = worker._event_store.speculative_events(EventType.verdict_voided)
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
        item = dataclasses.replace(item, base_sha=_MAIN)
        req = item.request

        worker = _make_worker(git_ops)
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._host_allocator = _make_mock_allocator()
        worker._event_store = _LateArrivalFakeEventStore()
        worker._dead_base_commits.add(_DEAD)  # unrelated dead commit
        worker._git_ops.get_main_sha = AsyncMock(return_value=_MAIN)  # type: ignore[method-assign]
        worker._remerge = AsyncMock()  # type: ignore[method-assign]

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        fail_outcome = MergeOutcome('blocked', reason='genuine verify fail')
        entry = _make_fail_entry(item, lease, fail_outcome, item.merge_wt)

        result = await worker._finalize_inflight(entry)

        assert result is False
        worker._remerge.assert_not_awaited()           # NOT voided → not re-merged
        assert not worker._redispatch                  # nothing re-parked
        assert req.result.done()                       # genuine FAIL IS adopted (blocks)
        assert req.result.result().status == 'blocked'
        assert worker._n_failed is True
        assert worker._event_store.speculative_events(EventType.verdict_voided) == []


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

        worker = _make_worker(git_ops)
        # Today's RED run has no PASS-branch void yet, so it falls through to the
        # CAS loop; MAX_CAS_RETRIES=0 makes that exit immediately (no real advance
        # — advance_main is mocked to cas_failed) so the RED assertions below fail
        # cleanly rather than looping.
        worker.MAX_CAS_RETRIES = 0
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._host_allocator = _make_mock_allocator()
        worker._event_store = _LateArrivalFakeEventStore()
        worker._dead_base_commits.add(_DEAD)
        worker._git_ops.get_main_sha = AsyncMock(return_value=_MAIN)  # type: ignore[method-assign]
        # advance_main spy — a dead-based tree must NEVER be advanced onto main.
        worker._git_ops.advance_main = AsyncMock(  # type: ignore[method-assign]
            return_value=AdvanceOutcome(result='cas_failed'),
        )

        _, remerged = _make_fake_item(
            'remerged-pass', base_sha=_MAIN, merge_commit='rmc11111',
            config=config, git_repo=git_repo,
        )
        worker._remerge = AsyncMock(return_value=remerged)  # type: ignore[method-assign]

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = _make_pass_entry(item, lease, item.merge_wt)

        result = await worker._finalize_inflight(entry)

        assert result is False
        worker._git_ops.advance_main.assert_not_awaited()  # dead tree never advanced
        worker._remerge.assert_awaited_once()              # re-merged against real main
        assert remerged in worker._redispatch              # re-parked for re-verify
        assert not req.result.done()                       # PASS void leaves req unresolved
        assert worker._n_failed is False                   # voided ≠ failed
        voided = worker._event_store.speculative_events(EventType.verdict_voided)
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

        worker = _make_worker(git_ops)
        worker.MAX_CAS_RETRIES = 0  # exit the CAS fast; we only assert the void branch is SKIPPED
        worker._register_owned_merge_worktree(item.merge_wt)
        worker._host_allocator = _make_mock_allocator()
        worker._event_store = _LateArrivalFakeEventStore()
        worker._dead_base_commits.add(_DEAD)  # unrelated dead commit; _LIVE_DEEP is NOT in it
        worker._git_ops.get_main_sha = AsyncMock(return_value=_MAIN)  # type: ignore[method-assign]
        worker._git_ops.advance_main = AsyncMock(  # type: ignore[method-assign]
            return_value=AdvanceOutcome(result='cas_failed'),
        )
        worker._remerge = AsyncMock()  # type: ignore[method-assign]

        lease = HostLease(name='local', runner=MagicMock(), is_local=True)
        entry = _make_pass_entry(item, lease, item.merge_wt)

        await worker._finalize_inflight(entry)

        # NOT voided: the deep intact chain proceeds to the normal CAS path.
        worker._remerge.assert_not_awaited()
        assert not worker._redispatch
        assert worker._event_store.speculative_events(EventType.verdict_voided) == []
        # Proof the normal PASS/CAS path WAS reached (advance attempted), not voided.
        worker._git_ops.advance_main.assert_awaited()
