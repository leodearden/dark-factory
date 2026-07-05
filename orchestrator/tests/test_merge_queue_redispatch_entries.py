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
from orchestrator.merge_types import RealMergeItem

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
