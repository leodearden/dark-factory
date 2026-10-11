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
from typing import Any

import pytest
from _merge_lane_fakes import (
    FakeClock,
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
from orchestrator.merge_lane import MergeOutcome, MergeRequest, QueuedBranch


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


async def _branch_with_one_commit(git_ops: GitOps, task_id: str) -> Path:
    """A task worktree forked at current main, one commit ahead; its path."""
    branch_wt = (await git_ops.create_worktree(task_id)).path
    (branch_wt / f'{task_id}.py').write_text('x = 1\n')
    await git_ops.commit(branch_wt, f'Add {task_id}.py')
    return branch_wt


async def _entry_while_verifying(
    git_ops: GitOps, req: MergeRequest,
) -> tuple[dict[str, Any] | None, MergeOutcome]:
    """Submit *req* to a lane over *git_ops*, read its ``snapshot()`` entry
    while its verify is parked in ``run_scoped``, then release the verify RED
    so no ``advance_main`` runs; return that entry and the outcome.

    Each abort-poll of the parked verify charges the lane's clock a full
    poll interval, so ``FakeClock``'s default ``wait_cap`` grants the verify
    only ~11s of real time before the no-progress budget calls it dead -- a
    loaded host spends that on the merge's git work alone, and the requeued
    item then re-merges onto whatever main has become. A 1s cap keeps that
    window far past the barrier timeouts.
    """
    gate = asyncio.Event()
    verifier = FakeVerifier(
        default=dataclasses.replace(
            fails(category='test_failure', summary='red'), release=gate,
        ),
    )
    queue: asyncio.Queue[MergeRequest] = asyncio.Queue()
    lane = make_lane(git_ops, queue, verifier=verifier, clock=FakeClock(wait_cap=1.0))

    async with running_lane(lane) as run:
        await queue.put(req)
        await wait_responsive(
            verifier.await_entry(1),
            timeout=MERGE_GATE_BARRIER_TIMEOUT,
            label=f'{req.task_id}: verify parked in run_scoped',
        )
        entry = lane_entry(lane, req.request_id)
        gate.set()
        outcome = await run.outcome(req, timeout=MERGE_RESULT_TIMEOUT)
    return entry, outcome


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
    merge-base of that SHA with the merged branch tip, or ``None`` when that
    cannot be resolved. Read off the public ``snapshot()`` census rather than
    a stubbed ``_run_post_merge_verify``.

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
        branch_wt = await _branch_with_one_commit(git_ops, 'frozen')
        base = _commit_empty_on_main(git_repo, 'C1')

        entry, outcome = await _entry_while_verifying(
            git_ops, _make_request('frozen', branch_wt, config),
        )

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

    @pytest.mark.asyncio
    async def test_an_unresolvable_merge_base_publishes_none(
        self, git_repo: Path, tmp_path: Path,
    ) -> None:
        """``merge_base_sha`` degrades to ``None`` (I3, fail-open) when
        ``git merge-base`` cannot resolve the pair, and ``main_sha`` is still
        the frozen base. The request's project root -- where the merge-base
        runs -- is an unrelated repo that holds neither SHA.
        """
        git_config = _git_config(persistent_merge_worktree=False)
        git_ops = GitOps(git_config, git_repo)
        branch_wt = await _branch_with_one_commit(git_ops, 'nobase')
        base = _git(git_repo, 'rev-parse', 'HEAD')
        unrelated = tmp_path / 'unrelated'
        unrelated.mkdir()
        _git(unrelated, 'init', '-q', '-b', 'main')
        config = lane_scene_config(unrelated, git_config)

        entry, outcome = await _entry_while_verifying(
            git_ops, _make_request('nobase', branch_wt, config),
        )

        assert entry is not None
        assert entry['state'] == 'verifying'
        assert entry['verify_base'] == {'main_sha': base, 'merge_base_sha': None}
        assert outcome.status == 'blocked', outcome
