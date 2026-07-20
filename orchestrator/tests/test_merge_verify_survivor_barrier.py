"""Tests for GitOps.reap_merge_verify_survivors — task 2828, limb 1
(the STARTUP SURVIVOR BARRIER).

Restart-collateral RCA: on orchestrator restart the merge_verify flock dies
with the process and the holder-pgid lease goes stale (its pgid is dead →
``_merge_verify_lease_active()`` is fail-OPEN → False).  But the PREVIOUS
run's verify subtree (bash → cargo → rustc, ``setsid``'d / reparented to
init) can still be ALIVE under ``<worktree_base>/_merge-verify``.  The merge
worker's first ``reset_persistent_merge_worktree`` then ``git reset --hard`` +
``git clean -xfd``s that tree, clobbering the live build out from under
itself — neither the flock holder nor the lease-pgid guard sees the survivor
(both died with the parent).

``reap_merge_verify_survivors`` is the one-time startup barrier that scans
/proc for process groups whose cwd / an open fd / an mmap'd path falls
at-or-under the ``_merge-verify`` subtree and reaps them BEFORE the worker
loop (hence before the first reset) runs.  These tests drive:

  (a) knob off            → returns True, never scans (no fixed lane to protect)
  (b) knob on, clear tree → returns True
  (c) knob on, live       → reaps a real straggler; own pgid is excluded
  (d) residual            → returns False + a loud ERROR (fail-open barrier)
  (e) integration         → reap-then-reset round-trip on a real git worktree
                            (the induced-restart approximation)

RED before step-08: ``reap_merge_verify_survivors`` is not yet defined on
GitOps, and git_ops does not yet import the scan/reap primitives.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from shared.proc_group import scan_process_groups_under_path as _real_scan

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run

# ---------------------------------------------------------------------------
# Helpers (mirroring test_merge_verify_lease_guard.py's GitOps fixtures and
# test_proc_group.py's real-subprocess planting + bounded reap-poll).
# ---------------------------------------------------------------------------


def _git_config(**overrides) -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
        **overrides,
    )


def _git_ops(tmp_path: Path, **config_overrides) -> GitOps:
    """A pure-FS GitOps (the barrier's scan/reap never shells out to git)."""
    project_root = tmp_path / 'project'
    project_root.mkdir(exist_ok=True)
    return GitOps(_git_config(**config_overrides), project_root)


async def _spawn_sleeper_in(cwd) -> asyncio.subprocess.Process:
    """Spawn a real ``sleep 30`` leading its own process group, with cwd *cwd*.

    ``start_new_session=True`` makes the child the leader of a fresh process
    group (``pgid == pid``); *cwd* becomes its working directory so
    ``/proc/<pid>/cwd`` points at (a path under) the scan root — exactly the
    shape of an orphaned cargo/rustc survivor.
    """
    return await asyncio.create_subprocess_exec(
        'sleep', '30',
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=str(cwd),
        start_new_session=True,
    )


async def _pgid_gone_within(pgid: int, timeout: float = 5.0, step: float = 0.1) -> bool:
    """Poll until a process group is fully reaped by the kernel (bounded).

    ProcessLookupError (ESRCH) → gone.  PermissionError (EPERM) means the pgid
    was recycled to another user, i.e. the group we planted is definitively
    gone; treat both as success.
    """
    iterations = max(1, int(timeout / step))
    for _ in range(iterations):
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            return True
        await asyncio.sleep(step)
    return False


def _kill_group(pgid: int) -> None:
    """Best-effort SIGKILL of an entire process group (test cleanup)."""
    with contextlib.suppress(ProcessLookupError, OSError):
        os.killpg(pgid, signal.SIGKILL)


async def _setup_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


async def _get_merge_commit(git_ops: GitOps, branch_name: str, filename: str) -> str:
    """Create a feature branch, commit a file, merge to main, return merge_commit."""
    wt_info = await git_ops.create_worktree(branch_name)
    (wt_info.path / filename).write_text(f'{branch_name} = True\n')
    await git_ops.commit(wt_info.path, f'Add {filename}')
    result = await git_ops.merge_to_main(wt_info.path, branch_name)
    assert result.success and result.merge_commit
    return result.merge_commit


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A real temporary git repository with an initial commit."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def real_git_ops(git_repo: Path) -> GitOps:
    """A real-git-backed GitOps for the reap-then-reset integration round-trip."""
    return GitOps(_git_config(), git_repo)


@pytest.mark.asyncio
class TestReapMergeVerifySurvivors:
    """GitOps.reap_merge_verify_survivors (task 2828, step-07/08)."""

    async def test_knob_off_returns_true_and_never_scans(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Knob off: no fixed ``_merge-verify`` lane exists, so there is
        nothing to protect — the barrier returns True immediately and must
        never pay for a /proc scan."""
        git_ops = _git_ops(tmp_path, persistent_merge_worktree=False)

        scan_spy = MagicMock(return_value=set())
        monkeypatch.setattr(
            'orchestrator.git_ops.scan_process_groups_under_path',
            scan_spy,
            raising=False,
        )

        result = await git_ops.reap_merge_verify_survivors()

        assert result is True, 'knob off → no lane to protect → True'
        scan_spy.assert_not_called()

    async def test_knob_on_no_survivors_returns_true(self, tmp_path: Path):
        """Knob on but nothing references the (never-created) ``_merge-verify``
        subtree → the real scan finds nothing → returns True."""
        git_ops = _git_ops(tmp_path, persistent_merge_worktree=True)
        assert not git_ops.persistent_merge_worktree_path.exists()

        result = await git_ops.reap_merge_verify_survivors()

        assert result is True

    @pytest.mark.timeout(20)
    async def test_knob_on_reaps_real_survivor_and_excludes_own_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ):
        """Knob on with a real live survivor under the lane: the barrier reaps
        it, returns True, and passes its OWN process group to the scan's
        ``exclude_pgids`` so it never signals the orchestrator itself.

        The scan is wrapped with a delegating spy (records ``exclude_pgids``,
        then calls the real primitive) so the genuine reap still happens.
        """
        git_ops = _git_ops(tmp_path, persistent_merge_worktree=True)
        root = git_ops.persistent_merge_worktree_path
        work = root / 'target'
        work.mkdir(parents=True)

        captured: dict[str, frozenset[int]] = {}

        def spy_scan(scan_root, *, exclude_pgids=frozenset()):
            captured['exclude'] = frozenset(exclude_pgids)
            return _real_scan(scan_root, exclude_pgids=exclude_pgids)

        monkeypatch.setattr(
            'orchestrator.git_ops.scan_process_groups_under_path',
            spy_scan,
            raising=False,
        )

        proc = await _spawn_sleeper_in(work)
        pgid = proc.pid
        try:
            result = await git_ops.reap_merge_verify_survivors(grace_secs=5.0)

            assert result is True, 'a reaped survivor leaves a clear tree → True'
            assert await _pgid_gone_within(pgid), (
                f'planted survivor pgid {pgid} must be reaped by the barrier'
            )
            assert os.getpgrp() in captured.get('exclude', frozenset()), (
                "the orchestrator's own process group must be excluded from "
                'the scan so the barrier never signals itself'
            )
        finally:
            _kill_group(pgid)
            with contextlib.suppress(Exception):
                await proc.wait()

    @pytest.mark.timeout(20)
    async def test_residual_survivor_returns_false_and_logs_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """A non-clearable residual (reap left a group alive) is fail-open:
        returns False and logs a loud ERROR naming the leftover group, rather
        than silently reporting success.

        ``reap_process_groups`` is stubbed to a no-op, so the planted group
        survives and the re-scan still finds it.
        """
        git_ops = _git_ops(tmp_path, persistent_merge_worktree=True)
        root = git_ops.persistent_merge_worktree_path
        work = root / 'target'
        work.mkdir(parents=True)

        def noop_reap(pgids, **_kwargs):
            return {p: 'survived' for p in pgids}

        monkeypatch.setattr(
            'orchestrator.git_ops.reap_process_groups', noop_reap, raising=False,
        )

        proc = await _spawn_sleeper_in(work)
        pgid = proc.pid
        try:
            with caplog.at_level(logging.ERROR, logger='orchestrator.git_ops'):
                result = await git_ops.reap_merge_verify_survivors()

            assert result is False, (
                'a non-clearable residual must return False (fail-open at the '
                'harness), never silently True'
            )
            error_records = [
                r for r in caplog.records if r.levelno >= logging.ERROR
            ]
            assert error_records, 'a residual survivor must be logged at ERROR'
            assert any(str(pgid) in r.getMessage() for r in error_records), (
                f'the residual pgid {pgid} must be named in the ERROR log; got '
                f'{[r.getMessage() for r in error_records]}'
            )
        finally:
            _kill_group(pgid)
            with contextlib.suppress(Exception):
                await proc.wait()

    @pytest.mark.timeout(30)
    async def test_integration_reap_then_reset_succeeds_on_clear_tree(
        self, real_git_ops: GitOps,
    ):
        """INTEGRATION (induced-restart approximation, real git): create a real
        persistent ``_merge-verify`` worktree, plant a straggler whose cwd is
        under it (a survivor of the "previous run"), reap it, then assert a
        subsequent ``reset_persistent_merge_worktree`` succeeds on the
        now-clear tree — the very reset the barrier protects."""
        real_git_ops.config.persistent_merge_worktree = True

        # Create the persistent _merge-verify worktree for real (create-once).
        merge_commit_a = await _get_merge_commit(real_git_ops, 'surv-a', 'sa.py')
        await real_git_ops.reset_persistent_merge_worktree(merge_commit_a)
        root = real_git_ops.persistent_merge_worktree_path
        assert root.exists()

        # Plant a straggler from a "previous run" with cwd under the tree.
        proc = await _spawn_sleeper_in(root)
        pgid = proc.pid
        try:
            ok = await real_git_ops.reap_merge_verify_survivors()
            assert ok is True
            assert await _pgid_gone_within(pgid), (
                'the induced-restart straggler must be reaped before the next '
                'reset can clobber the live tree'
            )

            # The reset the barrier protects must now succeed on the clear tree.
            merge_commit_b = await _get_merge_commit(real_git_ops, 'surv-b', 'sb.py')
            warm_path = await real_git_ops.reset_persistent_merge_worktree(
                merge_commit_b,
            )
            assert warm_path == real_git_ops.persistent_merge_worktree_path
            _, head_sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=warm_path)
            assert head_sha.strip() == merge_commit_b.strip(), (
                'after the survivor is reaped, reset_persistent_merge_worktree '
                'must advance HEAD on the now-clear tree'
            )
        finally:
            _kill_group(pgid)
            with contextlib.suppress(Exception):
                await proc.wait()
