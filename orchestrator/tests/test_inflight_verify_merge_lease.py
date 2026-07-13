"""Tests that _run_post_merge_verify holds the merge-verify lease across the
LOCAL in-process verify dispatch on the persistent _merge-verify worktree
(task 2315, BUG 1, step-17/18).

Incident: the persistent ``_merge-verify`` worktree was clobbered while a
verify was still running in it. The merge-verify flock + holder-pgid lease
(task 2306) already exists and is recorded by the host verify-merge CLI
(``cli.py:444-512``) for a REMOTE dispatch — but the LOCAL in-process
dispatch (this module's ``_run_post_merge_verify``, ``runner=None``) never
recorded it, so the step-13/14 reset guard and the step-15/16 GC-reclaim
defer never actually engaged for a local verify. This module closes that
gap: the lease must be held for the duration of the local dispatch on the
persistent warm lane, and released once the span completes — but ONLY for
that specific combination (local + persistent-worktree knob ON + merge_wt
is actually the persistent path); every other combination must stay
byte-identical (no lease recorded).
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _orch_helpers import make_placeholder_future

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeOutcome, MergeRequest, _run_post_merge_verify
from orchestrator.verify import VerifyResult

# ---------------------------------------------------------------------------
# Real-git fixtures (mirroring test_merge_queue_persistent_worktree.py /
# test_merge_queue_concurrent_verify.py)
# ---------------------------------------------------------------------------


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _head_sha(repo: Path) -> str:
    rc, out, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0
    return out.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_ops(git_repo: Path) -> GitOps:
    """A real-git-backed GitOps.  Its OWN GitConfig's persistent_merge_worktree
    knob is irrelevant to the lease-gating decision under test (only
    req.config.git.persistent_merge_worktree is consulted — see
    test_merge_queue_persistent_worktree.py's git_ops/_make_persistent_config
    split, which establishes the same precedent)."""
    git_config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(git_config, git_repo)


def _make_config(git_repo: Path, *, persistent: bool) -> OrchestratorConfig:
    git_config = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        persistent_merge_worktree=persistent,
    )
    return OrchestratorConfig(project_root=git_repo, git=git_config)


def _make_request(config: OrchestratorConfig, worktree: Path) -> MergeRequest:
    try:
        future: asyncio.Future[MergeOutcome] = asyncio.get_running_loop().create_future()
    except RuntimeError:
        future = make_placeholder_future()
    return MergeRequest(
        task_id='lease-t1',
        branch='task/lease-t1',
        worktree=worktree,
        pre_rebased=False,
        task_files=None,
        module_configs=[],
        config=config,
        result=future,
    )


def _mock_verify_result(passed: bool) -> VerifyResult:
    return VerifyResult(
        passed=passed,
        test_output='ok' if passed else 'FAILED',
        lint_output='',
        type_output='',
        summary='ok' if passed else 'fail',
        category='' if passed else 'test_failure',
    )


# ---------------------------------------------------------------------------
# step-17/18
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestInflightVerifyMergeLease:
    """_run_post_merge_verify holds the merge-verify lease for the duration
    of the LOCAL in-process verify dispatch on the persistent _merge-verify
    worktree — so the reset guard (step-13/14) and the GC-reclaim defer
    (step-15/16) actually engage for local verifies.  The remote/CLI path
    already records this lease (cli.py:444-512)."""

    async def test_lease_held_during_local_dispatch_on_persistent_worktree(
        self, git_ops: GitOps, git_repo: Path, tmp_path: Path,
    ):
        config = _make_config(git_repo, persistent=True)
        merge_wt = await git_ops.reset_persistent_merge_worktree(await _head_sha(git_repo))
        req = _make_request(config, tmp_path)

        dispatch_observed_lease: list[bool] = []

        async def _side(*args, **kwargs):
            dispatch_observed_lease.append(git_ops._merge_verify_lease_active())
            return _mock_verify_result(True)

        with patch(
            'orchestrator.merge_queue.VerifyRunnerPool.dispatch',
            new=AsyncMock(side_effect=_side),
        ):
            outcome = await _run_post_merge_verify(
                git_ops, req, merge_wt=merge_wt,
                timeouts={}, enospc_retries={},
                max_timeouts=3, max_enospc=1,
                runner=None,
            )

        assert outcome is None, f'expected PASS (None outcome), got {outcome}'
        assert dispatch_observed_lease == [True], (
            'the merge-verify lease must be held at dispatch time for the '
            'local in-process verify on the persistent warm lane'
        )
        assert git_ops._merge_verify_lease_active() is False, (
            'the lease must be released once the verify span completes'
        )

    async def test_lease_not_held_for_ephemeral_worktree(
        self, git_ops: GitOps, git_repo: Path, tmp_path: Path,
    ):
        """Control A: an ephemeral (non-persistent-path) merge_wt must NOT
        record the lease — it is not the shared warm lane the guards protect."""
        config = _make_config(git_repo, persistent=True)
        merge_wt = git_ops.worktree_base / '_merge-abc123'
        req = _make_request(config, tmp_path)

        dispatch_observed_lease: list[bool] = []

        async def _side(*args, **kwargs):
            dispatch_observed_lease.append(git_ops._merge_verify_lease_active())
            return _mock_verify_result(True)

        with patch(
            'orchestrator.merge_queue.VerifyRunnerPool.dispatch',
            new=AsyncMock(side_effect=_side),
        ):
            outcome = await _run_post_merge_verify(
                git_ops, req, merge_wt=merge_wt,
                timeouts={}, enospc_retries={},
                max_timeouts=3, max_enospc=1,
                runner=None,
            )

        assert outcome is None
        assert dispatch_observed_lease == [False], (
            'an ephemeral (non-warm-lane) merge_wt must not record a lease'
        )

    async def test_lease_not_held_when_knob_off(
        self, git_ops: GitOps, git_repo: Path, tmp_path: Path,
    ):
        """Control B: knob off (persistent_merge_worktree=False) → never
        record a lease even when merge_wt equals the nominal persistent
        path."""
        config = _make_config(git_repo, persistent=False)
        merge_wt = git_ops.persistent_merge_worktree_path
        req = _make_request(config, tmp_path)

        dispatch_observed_lease: list[bool] = []

        async def _side(*args, **kwargs):
            dispatch_observed_lease.append(git_ops._merge_verify_lease_active())
            return _mock_verify_result(True)

        with patch(
            'orchestrator.merge_queue.VerifyRunnerPool.dispatch',
            new=AsyncMock(side_effect=_side),
        ):
            outcome = await _run_post_merge_verify(
                git_ops, req, merge_wt=merge_wt,
                timeouts={}, enospc_retries={},
                max_timeouts=3, max_enospc=1,
                runner=None,
            )

        assert outcome is None
        assert dispatch_observed_lease == [False], (
            'persistent_merge_worktree=False must never record a lease, '
            'even against the nominal persistent path'
        )
