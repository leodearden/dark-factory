"""The verify's dispatch-time base facts, read off the lane's public census.

Each entry of ``MergeLane.snapshot()['entries']`` carries ``verify_base``:
``None`` until that entry's verify has resolved the facts the merge-skew
classifier attributes a failure against, then
``{'main_sha': ..., 'merge_base_sha': ...}``. This file pins the task-2357
claim at that public seam: the verify classifies against the FROZEN
merge-time main, never a fresh read of main.

Drives a real ``MergeLane`` over a real tmp git repo through the façade only;
git runs through module-local ``subprocess`` helpers.
"""
from __future__ import annotations

import asyncio
import dataclasses
import subprocess
from pathlib import Path

import pytest
from _merge_lane_fakes import (
    FakeVerifier,
    fails,
    lane_entry,
    lane_scene_config,
    make_lane,
    running_lane,
)
from _orch_helpers import MERGE_GATE_BARRIER_TIMEOUT, MERGE_RESULT_TIMEOUT, wait_responsive

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps
from orchestrator.merge_lane import MergeRequest, QueuedBranch


def _git(repo: Path, *args: str) -> str:
    """Run ``git *args`` in *repo*; return its stripped stdout."""
    return subprocess.run(
        ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit_empty_on_main(repo: Path, message: str) -> str:
    """Commit an empty commit on the checked-out main of *repo*; return its SHA.

    ``--allow-empty`` rather than staging anything: the repo root holds the
    lane's untracked ``.worktrees/`` directory, which must never be committed.
    """
    _git(repo, 'commit', '--allow-empty', '-q', '-m', message)
    return _git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real repo with one commit (C0) on main."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@test.com')
    _git(repo, 'config', 'user.name', 'Test')
    (repo / 'README.md').write_text('# Test\n')
    _git(repo, 'add', 'README.md')
    _git(repo, 'commit', '-q', '-m', 'C0')
    return repo


def _git_config(*, persistent_merge_worktree: bool) -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        persistent_merge_worktree=persistent_merge_worktree,
    )


def _make_request(
    task_id: str, worktree: Path, config: OrchestratorConfig,
) -> MergeRequest:
    """A ``MergeRequest`` for *task_id* whose future is on the running loop."""
    return MergeRequest(
        task_id=task_id,
        branch=QueuedBranch.parse(task_id, config.git.branch_prefix),
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        lane='normal',
    )


class _OutsideLandingDuringWarmSwap(GitOps):
    """Real ``GitOps`` that lands one outside commit on main mid-verify.

    Main must move AFTER the merge captured the item's base SHA but BEFORE
    the verify resolves its base facts, or a regression that re-reads main
    would read the same SHA and pass. On a single-host lane the LOCAL warm
    swap -- ``reset_persistent_merge_worktree`` -- is the one collaborator
    call in that window, so the first call commits an "outside landing" onto
    main and then delegates: the single-host stand-in for a landing racing a
    verify's setup. Overriding a PUBLIC method keeps the swap genuinely on
    disk and adds no patch target, as
    test_merge_speculation.py::_AdvanceFailingGitOps does for advance_main.
    """

    outside_landing: str | None = None

    async def reset_persistent_merge_worktree(self, merge_commit: str) -> Path:
        if self.outside_landing is None:
            self.outside_landing = _commit_empty_on_main(
                self.project_root, 'outside landing',
            )
        return await super().reset_persistent_merge_worktree(merge_commit)


class TestVerifyBaseOnTheSnapshot:
    """The verify classifies against the frozen merge-time main (task 2383 β / 2357).

    ``main_sha`` is the item's base SHA as captured when it merged, never a
    fresh read of main at verify time, and ``merge_base_sha`` is the
    merge-base of that SHA with the merged branch tip. Read off the public
    ``snapshot()`` census rather than a stubbed ``_run_post_merge_verify``.

    Companion pins of the same claim's downstream composition:
    test_merge_queue_disposition_wiring.py::TestRunPostMergeVerifyRealMainHeadFilter
    and test_merge_queue_disposition_wiring.py::TestRunInflightVerifyFreezesDispatchTimeMainSha.
    """

    @pytest.mark.asyncio
    async def test_a_queued_entry_carries_no_verify_base(
        self, git_repo: Path,
    ) -> None:
        git_config = _git_config(persistent_merge_worktree=False)
        config = lane_scene_config(git_repo, git_config)
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        lane = make_lane(GitOps(git_config, git_repo), queue)
        req = _make_request('queued', git_repo, config)

        queue.put_nowait(req)

        entry = lane_entry(lane, req.request_id)
        assert entry is not None
        assert entry['verify_base'] is None

    @pytest.mark.asyncio
    async def test_verify_base_is_the_frozen_merge_time_main_not_a_fresh_read(
        self, git_repo: Path,
    ) -> None:
        git_config = _git_config(persistent_merge_worktree=True)
        git_ops = _OutsideLandingDuringWarmSwap(git_config, git_repo)
        config = lane_scene_config(git_repo, git_config)

        fork = _git(git_repo, 'rev-parse', 'HEAD')
        branch_wt = (await git_ops.create_worktree('frozen')).path
        (branch_wt / 'frozen.py').write_text('x = 1\n')
        await git_ops.commit(branch_wt, 'Add frozen.py')
        base = _commit_empty_on_main(git_repo, 'C1')

        req = _make_request('frozen', branch_wt, config)
        gate = asyncio.Event()
        verifier = FakeVerifier(
            default=dataclasses.replace(
                fails(category='test_failure', summary='red'), release=gate,
            ),
        )
        queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
        lane = make_lane(git_ops, queue, verifier=verifier)

        async with running_lane(lane) as run:
            await queue.put(req)
            await wait_responsive(
                verifier.await_entry(1),
                timeout=MERGE_GATE_BARRIER_TIMEOUT,
                label='frozen: verify parked in run_scoped',
            )
            entry = lane_entry(lane, req.request_id)
            gate.set()
            outcome = await run.outcome(req, timeout=MERGE_RESULT_TIMEOUT)

        assert git_ops.outside_landing is not None
        assert await git_ops.get_main_sha() == git_ops.outside_landing != base, (
            'the fixture must move main between the merge and the verify, or '
            'a fresh read of main would be indistinguishable from the frozen base'
        )
        assert entry is not None
        assert entry['state'] == 'verifying'
        assert entry['verify_base'] == {'main_sha': base, 'merge_base_sha': fork}, (
            f'main_sha must be the FROZEN merge-time base {base} (task 2357) -- '
            f'the outside landing SHA {git_ops.outside_landing} there means the '
            f'verify re-read main -- and merge_base_sha its merge-base {fork} '
            f'with the merged branch tip'
        )
        assert outcome.status == 'blocked', outcome
