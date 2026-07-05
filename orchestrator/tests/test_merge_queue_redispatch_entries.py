"""Tests for _redispatch observability in snapshot()['entries'] (task 2068).

Follow-up from orphan L1 esc-2063-2, discovered while characterizing task 2063
(I4 speculation-permit conservation). snapshot()['entries'] is built from
_finalizing_head, _inflight, _remerging_item, _verifier_queue, _inflight_req,
_lane_buffers, and _queue — but never iterates self._redispatch. A request
parked on _redispatch (a speculative item awaiting a host when hosts <
speculation_depth, or a cascade-remerged item awaiting re-dispatch) was
therefore absent from snapshot()['entries'] and invisible to dashboard/
heartbeat depth+occupancy while it waits.

Task 2063 fixed only the conservation COUNT (_inflight_speculative_count());
this closes the matching entries-visibility gap.

FIX: snapshot() gains a new section (1c) iterating self._redispatch,
mirroring the existing _remerging_item transient-window entry, emitting a
'state': 'awaiting_host' entry per parked item.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeRequest, SpeculativeMergeWorker
from orchestrator.merge_types import DecidedItem, MergeOutcome, RealMergeItem

# ── fixtures (mirrors test_merge_queue_finalize_head_visibility.py) ─────────


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


@pytest.mark.asyncio
class TestSnapshotRedispatchEntries:
    """snapshot()['entries'] must surface items parked on self._redispatch.

    RED until snapshot() adds a section iterating self._redispatch.
    """

    async def test_redispatch_parked_item_appears_in_entries(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A speculative item awaiting a host on _redispatch must show up in
        snapshot()['entries'] with its request_id/branch/state so dashboard
        depth+occupancy reflect it while it waits.

        RED: snapshot() never reads self._redispatch, so the item is absent
        from entries and depth undercounts by one.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req = _make_req('rd-parked', 'task/rd-parked', config, git_repo)
        item = RealMergeItem(
            request=req,
            merge_result=None,  # type: ignore[arg-type]
            merge_wt=git_repo,
            base_sha='deadbeef',
            speculative=True,
        )
        worker._redispatch.append(item)

        snap = worker.snapshot()

        entry_task_ids = [e['task_id'] for e in snap['entries']]
        assert 'rd-parked' in entry_task_ids, (
            f"Expected 'rd-parked' in snapshot entries, got {entry_task_ids}. "
            "RED: snapshot() never iterates self._redispatch."
        )
        assert snap['depth'] == 1, (
            f"Expected depth==1 (the redispatch-parked item), got {snap['depth']}."
        )

        rd_entry = snap['entries'][0]
        assert rd_entry['task_id'] == 'rd-parked'
        assert rd_entry['branch'] == 'task/rd-parked'
        assert rd_entry['request_id'] == req.request_id
        assert rd_entry['state'] == 'awaiting_host', (
            f"Expected state=='awaiting_host', got {rd_entry['state']!r}."
        )

    async def test_no_redispatch_items_unchanged(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """With an empty _redispatch, snapshot() is unaffected (regression guard)."""
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        snap = worker.snapshot()

        assert snap['entries'] == []
        assert snap['depth'] == 0

    async def test_redispatch_entries_precede_awaiting_verify_entries(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """_redispatch is front-priority (drained by the verifier loop's
        DISPATCH-FILL ahead of _verifier_queue), so section 1c must place its
        entries BEFORE any awaiting_verify entry in snapshot()['entries'] —
        presence alone isn't enough.

        Guards the positional invariant documented in snapshot()'s section 1c
        comment: a future refactor that moved 1c below section 2 would keep
        the redispatch entry present (and depth correct) while silently
        breaking the head-of-line / position semantics dashboard consumers
        rely on. Without this test, that regression would pass both
        test_redispatch_parked_item_appears_in_entries (only _redispatch is
        populated there) and the depth/entries-presence checks.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        rd_req = _make_req('rd-parked', 'task/rd-parked', config, git_repo)
        rd_item = RealMergeItem(
            request=rd_req,
            merge_result=None,  # type: ignore[arg-type]
            merge_wt=git_repo,
            base_sha='deadbeef',
            speculative=True,
        )
        worker._redispatch.append(rd_item)

        av_req = _make_req('av-waiting', 'task/av-waiting', config, git_repo)
        av_item = RealMergeItem(
            request=av_req,
            merge_result=None,  # type: ignore[arg-type]
            merge_wt=git_repo,
            base_sha='deadbeef',
            speculative=False,
        )
        worker._verifier_queue.put_nowait(av_item)

        snap = worker.snapshot()

        entry_task_ids = [e['task_id'] for e in snap['entries']]
        assert 'rd-parked' in entry_task_ids and 'av-waiting' in entry_task_ids, (
            f"Expected both the redispatch and awaiting_verify entries present, "
            f"got {entry_task_ids}."
        )
        rd_index = entry_task_ids.index('rd-parked')
        av_index = entry_task_ids.index('av-waiting')
        assert rd_index < av_index, (
            f"Expected the redispatch entry (index {rd_index}) to precede the "
            f"awaiting_verify entry (index {av_index}) since _redispatch is "
            f"front-priority over _verifier_queue. Got order: {entry_task_ids}."
        )

    async def test_redispatch_decided_item_has_none_worktree(
        self, git_ops: GitOps, config: OrchestratorConfig, git_repo: Path,
    ) -> None:
        """A DecidedItem parked on _redispatch must surface with worktree=None.

        _redispatch is typed deque[SpeculativeItem] (RealMergeItem |
        DecidedItem), and this isn't just type-possible: _remerge() returns
        SpeculativeItem and can resolve to a DecidedItem (e.g. an
        already-merged/conflict outcome hit during a cascade-remerge), and
        both cascade-remerge call sites push its result straight onto
        _redispatch (append/appendleft). A DecidedItem never owns a merge
        worktree — item_merge_wt's DecidedItem arm always returns None — so
        section 1c's worktree_path=item_merge_wt(_rd_item) must resolve to
        None for this case rather than raising or defaulting incorrectly.
        """
        q: asyncio.Queue[MergeRequest] = asyncio.Queue()
        worker = SpeculativeMergeWorker(git_ops, q)

        req = _make_req('rd-decided', 'task/rd-decided', config, git_repo)
        item = DecidedItem(
            request=req,
            immediate_outcome=MergeOutcome('blocked', reason='test-filler'),
            base_sha='deadbeef',
            speculative=False,
        )
        worker._redispatch.append(item)

        snap = worker.snapshot()

        entry_task_ids = [e['task_id'] for e in snap['entries']]
        assert 'rd-decided' in entry_task_ids, (
            f"Expected 'rd-decided' in snapshot entries, got {entry_task_ids}."
        )
        rd_entry = snap['entries'][0]
        assert rd_entry['state'] == 'awaiting_host'
        assert rd_entry['worktree'] is None, (
            "Expected worktree is None for a DecidedItem parked on _redispatch "
            f"(it never owns a merge worktree), got {rd_entry['worktree']!r}."
        )
