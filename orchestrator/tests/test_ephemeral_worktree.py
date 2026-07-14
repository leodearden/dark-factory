"""Tests for GitOps.ephemeral_worktree() — task θ (verify-plan PRD).

GitOps.ephemeral_worktree(kind, sha) is the single extraction point for the
two main-tip probes' (verify_failure_is_preexisting_on_main, run_main_tip_sweep
in orchestrator/src/orchestrator/verify.py) copy-pasted worktree lifecycle:
mint a path under worktree_base with a kind-specific prefix, retry
`git worktree add --detach` on lock contention, and GUARANTEE scoped cleanup
(`git worktree remove --force` + rmtree) with NO `git worktree prune` ever
issued (DD5 — the comment-only invariant that failed in the 2026-07-04
warm-lane broad-prune registration-wipe incident, df 2097-2100).

Test coverage:
  step-1: WorktreeKind enum + E2 PROTECTED_PREFIXES registration
  step-3: GitOps.ephemeral_worktree CM behavior (naming, retry, cleanup,
          raise-on-add-failure)
  step-5: E1 — both probes route through the CM and NEVER issue
          ['git', 'worktree', 'prune']
  amend-1: no-leak-on-add-failure regression, both probes' fail-safe
           `except EphemeralWorktreeError` handlers, and the
           EphemeralWorktreeError -> BlockDisposition table row
  task 2507 step-1: flock-liveness guard — TestEphemeralWorktreeFlock (lock
           held across the CM body and acquired BEFORE `git worktree add`,
           modelling reify warm-lane-gc.sh's `flock -n` orphan-removal
           contender with an independent Python OFD; released+unlinked on
           normal exit and on a body exception; no lock-file leak when
           `git worktree add` exhausts its retries)
  task 2507 step-3: flock-denied path — TestEphemeralWorktreeFlockDenied (a
           LOCK_NB-denied acquire raises EphemeralWorktreeError BEFORE any
           `git worktree add`, the CM body never runs, and a pre-existing
           foreign-held lock file is never unlinked)
"""
from __future__ import annotations

import asyncio
import fcntl
import logging
import os
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, call, patch

import pytest

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import PROTECTED_PREFIXES, GitOps, WarmBaseHealth, WorktreeKind
from orchestrator.verify import VerifyResult

# ---------------------------------------------------------------------------
# step-1: WorktreeKind enum + E2 registration
# ---------------------------------------------------------------------------


class TestWorktreeKindRegistration:
    """step-1: WorktreeKind enum exists, its values are the probe/sweep
    prefixes, and both are registered in PROTECTED_PREFIXES /
    GitOps.protected_prefixes() (E2) with a non-empty owner.

    RED today: WorktreeKind does not exist yet, so every test here fails on
    the (deliberately in-test, not module-level) import.
    """

    def test_worktree_kind_import_and_values(self) -> None:
        from orchestrator.git_ops import WorktreeKind

        assert WorktreeKind.MAIN_PROBE.value == '_mainprobe-'
        assert WorktreeKind.MAIN_SWEEP.value == '_mainsweep-'

    def test_worktree_kind_values_registered_in_module_prefixes(self) -> None:
        from orchestrator.git_ops import WorktreeKind

        for kind in (WorktreeKind.MAIN_PROBE, WorktreeKind.MAIN_SWEEP):
            assert kind.value in PROTECTED_PREFIXES, (
                f'expected {kind.value!r} to be a key in PROTECTED_PREFIXES; '
                f'got keys={list(PROTECTED_PREFIXES)!r}'
            )
            owner = PROTECTED_PREFIXES[kind.value]
            assert isinstance(owner, str) and owner, (
                f'expected a non-empty owner tag for {kind.value!r}; got {owner!r}'
            )

    def test_worktree_kind_values_registered_in_instance_protected_prefixes(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.git_ops import WorktreeKind

        git_ops = GitOps(GitConfig(), tmp_path)
        registry = git_ops.protected_prefixes()

        for kind in (WorktreeKind.MAIN_PROBE, WorktreeKind.MAIN_SWEEP):
            assert kind.value in registry, (
                f'expected {kind.value!r} in protected_prefixes(); '
                f'got keys={list(registry)!r}'
            )
            assert registry[kind.value] == PROTECTED_PREFIXES[kind.value], (
                f'expected protected_prefixes() owner for {kind.value!r} to '
                f'match the module registry owner; got '
                f'{registry[kind.value]!r} != {PROTECTED_PREFIXES[kind.value]!r}'
            )


# ---------------------------------------------------------------------------
# step-3: GitOps.ephemeral_worktree CM behavior
# ---------------------------------------------------------------------------

MAIN_SHA = 'c' * 40


def _make_fake_run(add_rcs: list[int], calls: list[list[str]]):
    """Fake ``orchestrator.git_ops._run`` recording every argv into *calls*.

    ``git worktree add`` return codes are consumed in order from *add_rcs*
    (the last entry repeats once exhausted). A successful add mkdirs the
    ``--detach`` target, mirroring what real ``git worktree add`` does, so
    a later unconditional ``shutil.rmtree`` has something real on disk to
    remove. Every other command (e.g. ``git worktree remove``) always
    succeeds.
    """
    state = {'add_calls': 0}

    async def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if 'worktree' in cmd and 'add' in cmd:
            idx = state['add_calls']
            rc = add_rcs[idx] if idx < len(add_rcs) else add_rcs[-1]
            state['add_calls'] += 1
            if rc == 0:
                detach_idx = cmd.index('--detach')
                Path(cmd[detach_idx + 1]).mkdir(parents=True, exist_ok=True)
            return (rc, '', '' if rc == 0 else 'lock contention')
        return (0, '', '')

    return _fake_run


class TestEphemeralWorktreeNamingAndAdd:
    """step-3 (a)/(b): yielded path naming + ``git worktree add`` argv.

    RED today: GitOps has no ``ephemeral_worktree`` method.
    """

    def test_yields_path_under_worktree_base_with_mainprobe_prefix(
        self, tmp_path: Path,
    ) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []

        async def _body() -> Path:
            async with git_ops.ephemeral_worktree(WorktreeKind.MAIN_PROBE, MAIN_SHA) as p:
                return p

        with patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)):
            yielded = asyncio.run(_body())

        assert yielded.parent == git_ops.worktree_base, (
            f'expected {yielded!r} to be a direct child of worktree_base={git_ops.worktree_base}'
        )
        assert yielded.name.startswith('_mainprobe-'), (
            f"expected name to start with '_mainprobe-'; got {yielded.name!r}"
        )

    def test_yields_path_under_worktree_base_with_mainsweep_prefix(
        self, tmp_path: Path,
    ) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []

        async def _body() -> Path:
            async with git_ops.ephemeral_worktree(WorktreeKind.MAIN_SWEEP, MAIN_SHA) as p:
                return p

        with patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)):
            yielded = asyncio.run(_body())

        assert yielded.parent == git_ops.worktree_base, (
            f'expected {yielded!r} to be a direct child of worktree_base={git_ops.worktree_base}'
        )
        assert yielded.name.startswith('_mainsweep-'), (
            f"expected name to start with '_mainsweep-'; got {yielded.name!r}"
        )

    def test_issues_worktree_add_detach_with_sha(self, tmp_path: Path) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []

        async def _body() -> None:
            async with git_ops.ephemeral_worktree(WorktreeKind.MAIN_PROBE, MAIN_SHA):
                pass

        with patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)):
            asyncio.run(_body())

        add_calls = [c for c in calls if 'worktree' in c and 'add' in c]
        assert len(add_calls) == 1, f'expected exactly 1 add call; got {add_calls}'
        add_cmd = add_calls[0]
        assert '--detach' in add_cmd, f'expected --detach in add argv: {add_cmd}'
        assert MAIN_SHA in add_cmd, f'expected {MAIN_SHA!r} in add argv: {add_cmd}'


class TestEphemeralWorktreeCleanup:
    """step-3 (c)/(d): guaranteed scoped cleanup, no prune — on both normal
    exit and a body-raised exception.

    RED today: GitOps has no ``ephemeral_worktree`` method.
    """

    def test_normal_exit_removes_scoped_and_never_prunes(self, tmp_path: Path) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []

        async def _body() -> Path:
            async with git_ops.ephemeral_worktree(WorktreeKind.MAIN_PROBE, MAIN_SHA) as p:
                assert p.exists(), 'expected the minted path to exist inside the CM body'
                return p

        with patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)):
            p = asyncio.run(_body())

        remove_calls = [c for c in calls if 'worktree' in c and 'remove' in c]
        assert len(remove_calls) == 1, f'expected exactly 1 remove call; got {remove_calls}'
        assert '--force' in remove_calls[0], f'expected --force in remove argv: {remove_calls[0]}'
        assert str(p) in remove_calls[0], f'expected {p} in remove argv: {remove_calls[0]}'
        assert not any(
            'git' in c and 'worktree' in c and 'prune' in c for c in calls
        ), f'CM must NEVER issue a prune argv (DD5/E1); got calls={calls}'
        assert not p.exists(), (
            'expected the unconditional shutil.rmtree to remove the minted '
            'directory after the CM exits'
        )

    def test_cleanup_runs_and_never_prunes_when_body_raises(
        self, tmp_path: Path,
    ) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        captured: dict[str, Path] = {}

        async def _body() -> None:
            async with git_ops.ephemeral_worktree(WorktreeKind.MAIN_PROBE, MAIN_SHA) as p:
                captured['path'] = p
                raise RuntimeError('simulated body crash')

        with (
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)),
            pytest.raises(RuntimeError, match='simulated body crash'),
        ):
            asyncio.run(_body())

        remove_calls = [c for c in calls if 'worktree' in c and 'remove' in c]
        assert len(remove_calls) == 1, (
            f'expected cleanup to still issue a remove when the body raises; got {remove_calls}'
        )
        assert not any(
            'git' in c and 'worktree' in c and 'prune' in c for c in calls
        ), f'CM must NEVER issue a prune argv (DD5/E1); got calls={calls}'
        assert not captured['path'].exists(), (
            'expected cleanup (rmtree) to still run when the body raises'
        )


class TestEphemeralWorktreeRetry:
    """step-3 (e)/(f): retry-on-lock-contention + raise-after-exhaustion.

    RED today: GitOps has no ``ephemeral_worktree`` method and
    ``EphemeralWorktreeError`` does not exist.
    """

    def test_retries_add_and_enters_body_on_eventual_success(
        self, tmp_path: Path,
    ) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        entered = False

        async def _body() -> None:
            nonlocal entered
            async with git_ops.ephemeral_worktree(WorktreeKind.MAIN_PROBE, MAIN_SHA):
                entered = True

        with (
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([1, 1, 0], calls)),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
        ):
            asyncio.run(_body())

        add_calls = [c for c in calls if 'worktree' in c and 'add' in c]
        assert len(add_calls) == 3, f'expected exactly 3 add attempts; got {len(add_calls)}'
        # All 3 attempts must target the SAME minted path (retry reuses it;
        # it does not mint a fresh path per attempt).
        detach_targets = {c[c.index('--detach') + 1] for c in add_calls}
        assert len(detach_targets) == 1, (
            f'expected all retries to target the same path; got {detach_targets}'
        )
        assert entered, 'expected the CM body to run once add eventually succeeds'
        # asyncio.sleep is patched so the 2 backoff waits between the 3
        # attempts don't add real wall-clock time to the suite; still pin
        # the linear-backoff durations (0.5 * (attempt + 1)) requested for
        # the 2 failed attempts.
        assert mock_sleep.await_args_list == [call(0.5), call(1.0)], (
            f'expected backoff sleeps of 0.5s then 1.0s; got {mock_sleep.await_args_list}'
        )

    def test_raises_ephemeral_worktree_error_after_exhausting_retries(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.git_ops import EphemeralWorktreeError

        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        entered = False

        async def _body() -> None:
            nonlocal entered
            async with git_ops.ephemeral_worktree(WorktreeKind.MAIN_PROBE, MAIN_SHA):
                entered = True  # must never run

        with (
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([1, 1, 1], calls)),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
            pytest.raises(EphemeralWorktreeError),
        ):
            asyncio.run(_body())

        add_calls = [c for c in calls if 'worktree' in c and 'add' in c]
        assert len(add_calls) == 3, f'expected exactly 3 add attempts; got {len(add_calls)}'
        assert not entered, 'expected the CM body to NEVER run when add exhausts retries'
        remove_calls = [c for c in calls if 'worktree' in c and 'remove' in c]
        assert not remove_calls, (
            f'expected NO git worktree remove when add never succeeded; got {remove_calls}'
        )
        # asyncio.sleep is patched to avoid 1.5s of real backoff delay; the
        # CM only sleeps BETWEEN attempts (2 sleeps for 3 attempts, none
        # after the final exhausted attempt).
        assert mock_sleep.await_args_list == [call(0.5), call(1.0)], (
            f'expected backoff sleeps of 0.5s then 1.0s; got {mock_sleep.await_args_list}'
        )

    def test_add_failure_leaves_no_leaked_directory_under_worktree_base(
        self, tmp_path: Path,
    ) -> None:
        """Amendment (reviewer_comprehensive robustness_resource_leak,
        git_ops.py:1188): a failed ``git worktree add`` can still leave a
        partial/empty directory at tmp_path — real git creates the target
        directory early during add, before it can fail — and because this
        prefix is PROTECTED_PREFIXES-registered, the reaper will NEVER
        reclaim it. The CM must clean up tmp_path itself before raising, or
        the directory leaks under worktree_base permanently. Restores the
        pre-extraction probes' unconditional 'rmtree in case ... the
        worktree add never ran' cleanup guarantee.
        """
        from orchestrator.git_ops import EphemeralWorktreeError

        git_ops = GitOps(GitConfig(), tmp_path)
        entered = False

        async def _fake_run_leaves_partial_dir(cmd, **kwargs):
            if 'worktree' in cmd and 'add' in cmd:
                # Simulate real `git worktree add` creating its target
                # directory before it fails on lock contention.
                detach_idx = cmd.index('--detach')
                Path(cmd[detach_idx + 1]).mkdir(parents=True, exist_ok=True)
                return (1, '', 'lock contention')
            return (0, '', '')  # pragma: no cover — no other command expected

        async def _body() -> None:
            nonlocal entered
            async with git_ops.ephemeral_worktree(WorktreeKind.MAIN_PROBE, MAIN_SHA):
                entered = True  # must never run

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run_leaves_partial_dir),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock) as mock_sleep,
            pytest.raises(EphemeralWorktreeError),
        ):
            asyncio.run(_body())

        assert not entered, 'expected the CM body to NEVER run when add exhausts retries'
        leaked = list(git_ops.worktree_base.glob('_mainprobe-*'))
        assert not leaked, (
            f'expected no leaked _mainprobe-* directory under worktree_base '
            f'after add exhausts retries; found {leaked}'
        )
        # asyncio.sleep is patched to avoid 1.5s of real backoff delay.
        assert mock_sleep.await_args_list == [call(0.5), call(1.0)], (
            f'expected backoff sleeps of 0.5s then 1.0s; got {mock_sleep.await_args_list}'
        )


# ---------------------------------------------------------------------------
# task 2507 step-1: flock-liveness guard on GitOps.ephemeral_worktree
# ---------------------------------------------------------------------------


class TestEphemeralWorktreeFlock:
    """task 2507 step-1: ephemeral_worktree acquires a sibling ``<name>.lock``
    flock BEFORE ``git worktree add`` and holds it across the whole worktree
    lifetime, releasing it (and unlinking the file, only when THIS call was
    the one to acquire it) on every exit path.

    Models reify warm-lane-gc.sh's Pass-2 orphan-removal contender (``exec
    8>"$orphan_lock"; flock -n 8``, gc.sh:564-574) with an independent
    Python open-file-description: a second ``os.open`` + ``fcntl.flock(
    LOCK_EX|LOCK_NB)`` on the SAME lock path. flock is a kernel advisory
    lock keyed on the inode, so this reproduces gc.sh's preserve-while-held
    / reclaim-after-release contract hermetically, with no dependency on
    the external reify repo (which may be absent in CI).

    RED today: ephemeral_worktree takes no lock at all, so no sibling
    ``.lock`` file is ever created and a second-OFD ``LOCK_NB`` acquire
    always SUCCEEDS instead of raising ``BlockingIOError``.
    """

    @pytest.mark.parametrize('kind', [WorktreeKind.MAIN_PROBE, WorktreeKind.MAIN_SWEEP])
    def test_lock_held_during_body_and_released_after_normal_exit(
        self, tmp_path: Path, kind: WorktreeKind,
    ) -> None:
        """(a) + (c): while the CM body is running, the sibling lock file
        exists and is held exclusively (an independent second OFD's
        non-blocking LOCK_EX acquire raises BlockingIOError — models gc.sh's
        contender). After NORMAL exit, the lock file is unlinked and a
        fresh LOCK_EX|LOCK_NB acquire on a recreated file now succeeds."""
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        captured: dict[str, Path] = {}

        async def _body() -> None:
            async with git_ops.ephemeral_worktree(kind, MAIN_SHA) as p:
                lock_path = p.parent / f'{p.name}.lock'
                captured['lock_path'] = lock_path
                assert lock_path.exists(), (
                    f'expected sibling lock file {lock_path} to exist while '
                    f'the CM body is running'
                )
                fd2 = os.open(lock_path, os.O_RDWR)
                try:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(fd2)

        with patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)):
            asyncio.run(_body())

        lock_path = captured['lock_path']
        assert not lock_path.exists(), (
            f'expected the sibling lock file {lock_path} to be unlinked '
            f'after normal CM exit'
        )
        # A fresh independent OFD can now acquire the (recreated) lock
        # file's LOCK_EX|LOCK_NB without contention.
        fd2 = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)  # must not raise
        finally:
            os.close(fd2)

    @pytest.mark.parametrize('kind', [WorktreeKind.MAIN_PROBE, WorktreeKind.MAIN_SWEEP])
    def test_lock_acquired_before_worktree_add(
        self, tmp_path: Path, kind: WorktreeKind,
    ) -> None:
        """(b): the lock is already held BEFORE ``git worktree add`` runs.
        A fake ``_run`` that, on the ``worktree add`` argv, tries a
        second-OFD LOCK_EX|LOCK_NB acquire on the derived lock path BEFORE
        creating the ``--detach`` target records whether it was denied —
        i.e. the lock was present+held before the worktree dir exists."""
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        observed: dict[str, bool] = {}

        async def _fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            if 'worktree' in cmd and 'add' in cmd:
                detach_idx = cmd.index('--detach')
                target = Path(cmd[detach_idx + 1])
                assert not target.exists(), (
                    'expected the worktree target to not exist yet at the '
                    'point the lock contention is probed'
                )
                lock_path = target.parent / f'{target.name}.lock'
                fd2 = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
                try:
                    fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    observed['denied_before_add'] = True
                else:
                    observed['denied_before_add'] = False
                finally:
                    os.close(fd2)
                target.mkdir(parents=True, exist_ok=True)
                return (0, '', '')
            return (0, '', '')

        async def _body() -> None:
            async with git_ops.ephemeral_worktree(kind, MAIN_SHA):
                pass

        with patch('orchestrator.git_ops._run', side_effect=_fake_run):
            asyncio.run(_body())

        assert observed.get('denied_before_add') is True, (
            'expected the sibling lock to already be held (LOCK_NB denied) '
            'by the time `git worktree add` runs — the lock must be '
            'acquired BEFORE the worktree exists on disk'
        )

    @pytest.mark.parametrize('kind', [WorktreeKind.MAIN_PROBE, WorktreeKind.MAIN_SWEEP])
    def test_lock_unlinked_after_body_exception(
        self, tmp_path: Path, kind: WorktreeKind,
    ) -> None:
        """(d): on a body-EXCEPTION exit the lock file is still unlinked
        (the finally ran)."""
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        captured: dict[str, Path] = {}

        async def _body() -> None:
            async with git_ops.ephemeral_worktree(kind, MAIN_SHA) as p:
                captured['lock_path'] = p.parent / f'{p.name}.lock'
                raise RuntimeError('simulated body crash')

        with (
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)),
            pytest.raises(RuntimeError, match='simulated body crash'),
        ):
            asyncio.run(_body())

        lock_path = captured['lock_path']
        assert not lock_path.exists(), (
            f'expected the sibling lock file {lock_path} to be unlinked '
            f'even when the CM body raises (finally ran)'
        )

    @pytest.mark.parametrize('kind', [WorktreeKind.MAIN_PROBE, WorktreeKind.MAIN_SWEEP])
    def test_no_lock_file_leaks_when_add_exhausts_retries(
        self, tmp_path: Path, kind: WorktreeKind,
    ) -> None:
        """(e): when ``git worktree add`` fails on every retry (->
        EphemeralWorktreeError), no sibling ``<kind.value><hex>.lock`` file
        leaks under worktree_base either — mirrors the existing
        no-leak-directory regression (amend-1,
        test_add_failure_leaves_no_leaked_directory_under_worktree_base) but
        for the lock file this task adds. A regression here would also flip
        that existing test's glob('_mainprobe-*') no-leak assertion, since
        the sibling lock name matches the same glob.
        """
        from orchestrator.git_ops import EphemeralWorktreeError

        git_ops = GitOps(GitConfig(), tmp_path)

        async def _fake_run_leaves_partial_dir(cmd, **kwargs):
            if 'worktree' in cmd and 'add' in cmd:
                # Simulate real `git worktree add` creating its target
                # directory before it fails on lock contention.
                detach_idx = cmd.index('--detach')
                Path(cmd[detach_idx + 1]).mkdir(parents=True, exist_ok=True)
                return (1, '', 'lock contention')
            return (0, '', '')  # pragma: no cover — no other command expected

        async def _body() -> None:
            async with git_ops.ephemeral_worktree(kind, MAIN_SHA):
                pass  # pragma: no cover — must never run

        with (
            patch('orchestrator.git_ops._run', side_effect=_fake_run_leaves_partial_dir),
            patch('orchestrator.git_ops.asyncio.sleep', new_callable=AsyncMock),
            pytest.raises(EphemeralWorktreeError),
        ):
            asyncio.run(_body())

        leaked_locks = list(git_ops.worktree_base.glob(f'{kind.value}*.lock'))
        assert not leaked_locks, (
            f'expected no leaked {kind.value}*.lock file under worktree_base '
            f'after add exhausts retries; found {leaked_locks}'
        )


# ---------------------------------------------------------------------------
# task 2507 step-3: flock-denied path raises EphemeralWorktreeError
# ---------------------------------------------------------------------------


class TestEphemeralWorktreeFlockDenied:
    """task 2507 step-3: a LOCK_NB-denied flock acquire (another live
    consumer already holds the sibling lock — should never happen in
    practice, since ephemeral_worktree mints a fresh uuid4 hex per call,
    but must fail safe if it ever does) converts to
    ``EphemeralWorktreeError`` BEFORE any ``git worktree add`` is
    attempted: the CM body never runs, and — critically — a lock file we
    did not acquire is NEVER unlinked out from under its real holder.

    Patches ``orchestrator.git_ops.fcntl.flock`` directly to raise
    ``BlockingIOError``, modelling the denial without needing a real
    second holder process.

    RED today: step-2 leaves the ``fcntl.flock`` acquire un-wrapped, so
    the raised ``BlockingIOError`` propagates as a bare
    ``BlockingIOError`` instead of ``EphemeralWorktreeError``.
    """

    def test_flock_denied_raises_ephemeral_worktree_error_before_add(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator.git_ops import EphemeralWorktreeError

        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        entered = False
        fixed_uuid = uuid.UUID('11111111-2222-3333-4444-555555555555')

        # Derive the EXACT lock path ephemeral_worktree will compute (by
        # patching uuid4 to the same fixed value), so a foreign-held lock
        # file can be pre-created there and asserted untouched afterward.
        expected_name = f'{WorktreeKind.MAIN_PROBE.value}{fixed_uuid.hex[:8]}'
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        lock_path = git_ops.worktree_base / f'{expected_name}.lock'
        lock_path.write_text('foreign holder')

        async def _body() -> None:
            nonlocal entered
            async with git_ops.ephemeral_worktree(WorktreeKind.MAIN_PROBE, MAIN_SHA):
                entered = True  # must never run

        with (
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)),
            patch('orchestrator.git_ops.uuid.uuid4', return_value=fixed_uuid),
            patch(
                'orchestrator.git_ops.fcntl.flock',
                side_effect=BlockingIOError(11, 'Resource temporarily unavailable'),
            ),
            pytest.raises(EphemeralWorktreeError),
        ):
            asyncio.run(_body())

        assert not entered, (
            'expected the CM body to NEVER run when the flock acquire is denied'
        )
        add_calls = [c for c in calls if 'worktree' in c and 'add' in c]
        assert not add_calls, (
            f'expected NO git worktree add when the flock acquire is denied; '
            f'got {add_calls}'
        )
        assert lock_path.exists(), (
            'expected the pre-existing foreign-held lock file to survive the '
            'failure untouched — we must never unlink a lock we did not acquire'
        )
        assert lock_path.read_text() == 'foreign holder', (
            'expected the foreign lock file contents to be untouched'
        )


# ---------------------------------------------------------------------------
# step-5: E1 — both probes route through the CM; the CM never prunes
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        project_root=tmp_path,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )


_PASSING_RESULT = VerifyResult(
    passed=True, test_output='', lint_output='', type_output='',
    summary='all checks passed',
)


class TestBothProbesRouteThroughEphemeralWorktree:
    """step-5 (E1 — the hard-leaf, load-bearing incident-prevention signal
    this task exists to add): both main-tip probes
    (verify_failure_is_preexisting_on_main, run_main_tip_sweep — both in
    verify.py) must build their throwaway worktree by calling
    GitOps.ephemeral_worktree(), and neither may EVER issue
    ``['git', 'worktree', 'prune']`` (DD5) — only a scoped
    ``git worktree remove --force``.

    RED today: both probes still build their own worktree inline (bespoke
    path build + retry-add + finally-cleanup) and never call
    ``git_ops.ephemeral_worktree`` — the spy sees 0 calls.
    """

    def test_verify_failure_is_preexisting_on_main_routes_through_cm(
        self, tmp_path: Path,
    ) -> None:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()

        git_ops = GitOps(config.git, config.project_root)
        git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        # A category/cause_hint combo unique to this test so its key can
        # never collide with another test's entry in the process-wide
        # (module-level) verify._PROBE_CACHE.
        failing_result = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='theta_e1_mainprobe_signal',
            cause_hint='task theta E1 ephemeral_worktree routing signal (mainprobe)',
            category='task_theta_e1_mainprobe',
        )

        calls: list[list[str]] = []

        with (
            patch.object(verify_module, 'run_scoped_verification',
                         new=AsyncMock(return_value=_PASSING_RESULT)),
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)),
            patch.object(
                git_ops, 'ephemeral_worktree', wraps=git_ops.ephemeral_worktree,
            ) as spied_cm,
        ):
            asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], failing_result, git_ops,
                )
            )

        assert spied_cm.call_count >= 1, (
            'expected verify_failure_is_preexisting_on_main to build its probe '
            'worktree via git_ops.ephemeral_worktree(); got 0 calls'
        )
        assert not any(
            'worktree' in c and 'prune' in c for c in calls
        ), f'probe must NEVER issue a prune argv (DD5/E1); got calls={calls}'
        assert any(
            'worktree' in c and 'remove' in c and '--force' in c for c in calls
        ), f'expected a scoped "git worktree remove --force" argv; got calls={calls}'

    def test_run_main_tip_sweep_routes_through_cm(self, tmp_path: Path) -> None:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = GitOps(config.git, config.project_root)
        git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        calls: list[list[str]] = []

        async def _fake_full_verify(project_root, cfg, **kwargs) -> VerifyResult:
            return _PASSING_RESULT

        with (
            patch.object(verify_module, 'run_full_verification', side_effect=_fake_full_verify),
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)),
            patch.object(
                git_ops, 'ephemeral_worktree', wraps=git_ops.ephemeral_worktree,
            ) as spied_cm,
        ):
            result = asyncio.run(verify_module.run_main_tip_sweep(config, git_ops))

        assert result is not None, f'expected a (sha, VerifyResult) tuple, got {result!r}'
        assert spied_cm.call_count >= 1, (
            'expected run_main_tip_sweep to build its sweep worktree via '
            'git_ops.ephemeral_worktree(); got 0 calls'
        )
        assert not any(
            'worktree' in c and 'prune' in c for c in calls
        ), f'probe must NEVER issue a prune argv (DD5/E1); got calls={calls}'
        assert any(
            'worktree' in c and 'remove' in c and '--force' in c for c in calls
        ), f'expected a scoped "git worktree remove --force" argv; got calls={calls}'


# ---------------------------------------------------------------------------
# amend-1: both probes' fail-safe `except EphemeralWorktreeError` handlers
# ---------------------------------------------------------------------------


class TestBothProbesFailSafeOnEphemeralWorktreeError:
    """Amendment (reviewer_comprehensive test_coverage): the CM's raise is
    covered by TestEphemeralWorktreeRetry, and CM-routing on the happy path
    is covered by TestBothProbesRouteThroughEphemeralWorktree, but no test
    previously drove an actual EphemeralWorktreeError THROUGH a probe
    caller. This pins the fail-safe ``except EphemeralWorktreeError``
    handler each probe carries — previously an inline early ``return``
    before ``ephemeral_worktree`` existed — so a regression there (e.g. the
    exception propagating uncaught) is caught here rather than in
    production.
    """

    def test_verify_failure_is_preexisting_on_main_fails_safe_when_add_exhausts_retries(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()

        git_ops = GitOps(config.git, config.project_root)
        git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        # A category/cause_hint combo unique to this test so its key can
        # never collide with another test's entry in the process-wide
        # (module-level) verify._PROBE_CACHE.
        failing_result = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='theta_amend_addfail_mainprobe_signal',
            cause_hint='task theta amendment add-exhaustion fail-safe signal (mainprobe)',
            category='task_theta_amend_addfail_mainprobe',
        )

        calls: list[list[str]] = []

        with (
            patch.object(verify_module, 'run_scoped_verification',
                         new=AsyncMock(return_value=_PASSING_RESULT)),
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([1, 1, 1], calls)),
            caplog.at_level(logging.WARNING, logger='orchestrator.verify'),
        ):
            result = asyncio.run(
                verify_module.verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], failing_result, git_ops,
                )
            )

        assert result == (False, ''), (
            f"expected the fail-safe sentinel (False, '') when "
            f'ephemeral_worktree exhausts add retries; got {result!r}'
        )
        add_calls = [c for c in calls if 'worktree' in c and 'add' in c]
        assert len(add_calls) == 3, f'expected 3 add attempts; got {len(add_calls)}'
        assert not any(
            'worktree' in c and 'remove' in c for c in calls
        ), f'expected no remove call when add never succeeded; got {calls}'
        assert any(
            'contagion guard disabled' in r.getMessage() for r in caplog.records
        ), (
            f'expected a WARNING noting the contagion guard was disabled; '
            f'got: {[r.getMessage() for r in caplog.records]}'
        )

    def test_run_main_tip_sweep_fails_safe_when_add_exhausts_retries(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        from orchestrator import verify as verify_module

        config = _make_config(tmp_path)
        git_ops = GitOps(config.git, config.project_root)
        git_ops.get_main_sha = AsyncMock(return_value=MAIN_SHA)  # type: ignore[method-assign]
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        calls: list[list[str]] = []
        full_verify_called = False

        async def _fake_full_verify(project_root, cfg, **kwargs) -> VerifyResult:
            nonlocal full_verify_called
            full_verify_called = True  # must never run
            return _PASSING_RESULT

        with (
            patch.object(verify_module, 'run_full_verification', side_effect=_fake_full_verify),
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([1, 1, 1], calls)),
            caplog.at_level(logging.WARNING, logger='orchestrator.verify'),
        ):
            result = asyncio.run(verify_module.run_main_tip_sweep(config, git_ops))

        assert result is None, (
            f'expected the fail-safe sentinel None when ephemeral_worktree '
            f'exhausts add retries; got {result!r}'
        )
        assert not full_verify_called, (
            'expected run_full_verification to NEVER run when add exhausts retries'
        )
        add_calls = [c for c in calls if 'worktree' in c and 'add' in c]
        assert len(add_calls) == 3, f'expected 3 add attempts; got {len(add_calls)}'
        assert not any(
            'worktree' in c and 'remove' in c for c in calls
        ), f'expected no remove call when add never succeeded; got {calls}'
        assert any(
            'sweep skipped for this tick' in r.getMessage() for r in caplog.records
        ), (
            f'expected a WARNING noting the sweep was skipped; got: '
            f'{[r.getMessage() for r in caplog.records]}'
        )


# ---------------------------------------------------------------------------
# amend-1: EphemeralWorktreeError -> BlockDisposition table row
# ---------------------------------------------------------------------------


class TestEphemeralWorktreeErrorDisposition:
    """Amendment (reviewer_comprehensive test_coverage): EphemeralWorktreeError
    gained an explicit ``workflow_types._disposition_table()`` row (BD-2
    completeness — commit cced1c0adb, resolving esc-2147-3) but the row's
    resolved BlockDisposition was untested. The canonical home for
    classify_failure()-table pins is test_block_disposition.py's
    TestClassifyFailureKnownRows, which sits outside this task's locked
    module scope (git_ops.py/verify.py + the 4 verify/ephemeral-worktree
    test files) — this is a scoped-in stand-in pinning the same contract
    for the row this task added.
    """

    def test_classify_failure_resolves_the_expected_disposition(self) -> None:
        from orchestrator.git_ops import EphemeralWorktreeError
        from orchestrator.unblock_types import BlockClass
        from orchestrator.verify_categories import FailureCategory
        from orchestrator.workflow_types import RequeueKind, classify_failure

        disposition = classify_failure(EphemeralWorktreeError('add failed after retries'))

        assert disposition.category is FailureCategory.NONE
        assert disposition.escalate_to_human is True
        assert disposition.requeue_kind is RequeueKind.BLOCK
        assert disposition.counts_against_requeue_cap is True
        assert disposition.block_class is BlockClass.AGENT_FAILURE
        assert disposition.reason_prefix == 'Ephemeral probe worktree add failed after retries'


# ---------------------------------------------------------------------------
# task 2567 step-1: GitOps._seed_warm_lane gains take_lane_lock
# ---------------------------------------------------------------------------


class TestSeedWarmLaneLaneLock:
    """task 2567 step-1: pins the new ``take_lane_lock`` param on
    ``GitOps._seed_warm_lane`` — the deadlock-avoidance seam for
    ``ephemeral_worktree``'s upcoming CM-routed seed (task 2567's
    LOAD-BEARING correctness fix: the CM already holds ``<lane_dir>.lock``
    for its entire lifetime, so a naive seed call would self-deadlock
    against the SAME lock path and time out after
    ``_SEED_WARM_LANE_LOCK_WAIT_SECS``).

    RED today: ``_seed_warm_lane`` has no ``take_lane_lock`` kwarg, so any
    call passing it raises ``TypeError``.
    """

    @staticmethod
    def _make_plain_base_and_script(git_ops: GitOps, lane: Path) -> None:
        base = git_ops.warm_lane_base_target_path
        base.mkdir(parents=True, exist_ok=True)
        (base / 'sentinel').write_text('base content')
        scripts_dir = lane / 'scripts'
        scripts_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / 'seed-warm-lane.sh').write_text('#!/usr/bin/env bash\nexit 0\n')

    def test_take_lane_lock_false_omits_outer_flock(self, tmp_path: Path) -> None:
        """(a): take_lane_lock=False -> argv[0] is the script path and
        '<lane_dir>.lock' appears nowhere in argv."""
        git_ops = GitOps(GitConfig(), tmp_path)
        lane = tmp_path / '_lane-0'
        self._make_plain_base_and_script(git_ops, lane)
        calls: list[list[str]] = []

        async def _fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return (0, '', '')

        with patch('orchestrator.git_ops._run', side_effect=_fake_run):
            rc = asyncio.run(
                git_ops._seed_warm_lane(lane, '--fresh-checkout', take_lane_lock=False)
            )

        assert rc == 0, f'expected rc=0 from the fake script; got {rc}'
        assert len(calls) == 1, f'expected exactly 1 subprocess call; got {calls}'
        cmd = calls[0]
        script_path = str(lane / 'scripts' / 'seed-warm-lane.sh')
        lane_lock = str(lane) + '.lock'
        assert cmd[0] == script_path, (
            f'expected argv[0] to be the script path when take_lane_lock=False; got {cmd!r}'
        )
        assert lane_lock not in cmd, (
            f'expected {lane_lock!r} to be absent from argv; got {cmd!r}'
        )

    def test_default_take_lane_lock_true_includes_outer_flock(self, tmp_path: Path) -> None:
        """(b): default (take_lane_lock=True) -> argv begins with
        'flock -x -w 30 -E 124 <lane_dir>.lock' then the script."""
        git_ops = GitOps(GitConfig(), tmp_path)
        lane = tmp_path / '_lane-0'
        self._make_plain_base_and_script(git_ops, lane)
        calls: list[list[str]] = []

        async def _fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return (0, '', '')

        with patch('orchestrator.git_ops._run', side_effect=_fake_run):
            rc = asyncio.run(git_ops._seed_warm_lane(lane, '--fresh-checkout'))

        assert rc == 0, f'expected rc=0 from the fake script; got {rc}'
        assert len(calls) == 1, f'expected exactly 1 subprocess call; got {calls}'
        cmd = calls[0]
        lane_lock = str(lane) + '.lock'
        script_path = str(lane / 'scripts' / 'seed-warm-lane.sh')
        assert cmd[:6] == ['flock', '-x', '-w', '30', '-E', '124'], (
            f'expected the outer flock -x -w 30 -E 124 prefix; got {cmd!r}'
        )
        assert cmd[6] == lane_lock, (
            f'expected the lane lock path right after the flock flags; got {cmd!r}'
        )
        assert cmd[7] == script_path, (
            f'expected the script path right after the outer flock argv; got {cmd!r}'
        )

    def test_symlink_base_take_lane_lock_false_keeps_inner_gen_flock(
        self, tmp_path: Path,
    ) -> None:
        """(c): a symlink base (target -> .gen.N) with take_lane_lock=False
        still includes the INNER 'flock -s <gen>.lock' wrapper — only the
        OUTER lane lock is dropped."""
        base_dir = tmp_path / 'bases'
        base_dir.mkdir()
        gen_dir = base_dir / '.gen.0'
        gen_dir.mkdir()
        (gen_dir / 'sentinel').write_text('gen content')
        target_symlink = base_dir / 'target'
        target_symlink.symlink_to('.gen.0')
        (base_dir / '.gen.0.lock').touch()

        config = GitConfig(warm_lane_base_target_dir=str(target_symlink))
        git_ops = GitOps(config, tmp_path)
        lane = tmp_path / '_lane-1'
        scripts_dir = lane / 'scripts'
        scripts_dir.mkdir(parents=True)
        (scripts_dir / 'seed-warm-lane.sh').write_text('#!/usr/bin/env bash\nexit 0\n')

        calls: list[list[str]] = []

        async def _fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            return (0, '', '')

        with patch('orchestrator.git_ops._run', side_effect=_fake_run):
            rc = asyncio.run(
                git_ops._seed_warm_lane(lane, '--fresh-checkout', take_lane_lock=False)
            )

        assert rc == 0, f'expected rc=0 from the fake script; got {rc}'
        assert len(calls) == 1, f'expected exactly 1 subprocess call; got {calls}'
        cmd = calls[0]
        lane_lock = str(lane) + '.lock'
        gen_lock = str(gen_dir) + '.lock'
        assert lane_lock not in cmd, (
            f'expected the OUTER lane lock to be absent from argv; got {cmd!r}'
        )
        assert cmd[:3] == ['flock', '-s', gen_lock], (
            f'expected the INNER flock -s <gen_dir>.lock prefix to remain; got {cmd!r}'
        )


# ---------------------------------------------------------------------------
# task 2567 step-3: GitOps.ephemeral_worktree gains warm_seed
# ---------------------------------------------------------------------------


class TestEphemeralWorktreeWarmSeed:
    """task 2567 step-3: pins the new ``warm_seed`` param on
    ``ephemeral_worktree`` — the reflink-seed integration point for the
    main-health probe. When True and the warm-lane CoW seed base is
    resolvable, the CM seeds the minted worktree's ``target/`` from the
    shared base (via ``_seed_warm_lane(..., take_lane_lock=False)``, since
    the CM already holds ``<lane_dir>.lock``) before the body runs. Any
    seed fault is fail-soft (proceed COLD, body still runs); a
    non-resolvable base skips the seed subprocess entirely; default False
    keeps ``run_main_tip_sweep`` (MAIN_SWEEP) byte-identical.

    RED today: ``ephemeral_worktree`` has no ``warm_seed`` kwarg, so any
    call passing it raises ``TypeError``.
    """

    def test_warm_seed_true_and_base_ok_seeds_fresh_checkout_without_lane_lock(
        self, tmp_path: Path,
    ) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        entered_paths: list[Path] = []

        async def _body() -> None:
            async with git_ops.ephemeral_worktree(
                WorktreeKind.MAIN_PROBE, MAIN_SHA, warm_seed=True,
            ) as p:
                entered_paths.append(p)
                assert p.exists(), 'expected the minted path to exist inside the CM body'

        with (
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)),
            patch.object(git_ops, '_seed_warm_lane', new=AsyncMock(return_value=0)) as mock_seed,
            patch.object(git_ops, '_warm_lane_base_resolvable', return_value=WarmBaseHealth.OK),
        ):
            asyncio.run(_body())

        assert len(entered_paths) == 1, 'expected the CM body to run exactly once'
        mock_seed.assert_awaited_once_with(
            entered_paths[0], '--fresh-checkout', take_lane_lock=False,
        )

    @pytest.mark.parametrize('seed_rc', [75, 127])
    def test_warm_seed_fails_soft_on_nonzero_seed_rc(
        self, tmp_path: Path, seed_rc: int,
    ) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        entered = False

        async def _body() -> None:
            nonlocal entered
            async with git_ops.ephemeral_worktree(
                WorktreeKind.MAIN_PROBE, MAIN_SHA, warm_seed=True,
            ):
                entered = True

        with (
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)),
            patch.object(git_ops, '_seed_warm_lane', new=AsyncMock(return_value=seed_rc)),
            patch.object(git_ops, '_warm_lane_base_resolvable', return_value=WarmBaseHealth.OK),
        ):
            asyncio.run(_body())  # must not raise

        assert entered, 'expected the CM body to still run (fail-soft) on a seed fault'
        remove_calls = [c for c in calls if 'worktree' in c and 'remove' in c]
        assert len(remove_calls) == 1, f'expected cleanup remove to still run; got {remove_calls}'

    @pytest.mark.parametrize('raiser', ['_warm_lane_base_resolvable', '_seed_warm_lane'])
    def test_warm_seed_gate_exception_is_structural_fail_soft(
        self, tmp_path: Path, raiser: str,
    ) -> None:
        """task 2567 amendment: an unexpected exception escaping the
        warm-seed gate (not just a non-zero seed rc) must not propagate
        out of the CM and must not skip the scoped ``git worktree remove
        --force`` cleanup. The gate is wrapped in its own broad
        ``except Exception`` so this is structural, not merely dependent
        on ``_warm_lane_base_resolvable``/``_seed_warm_lane`` each
        catching everything internally today.
        """
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        entered = False

        async def _body() -> None:
            nonlocal entered
            async with git_ops.ephemeral_worktree(
                WorktreeKind.MAIN_PROBE, MAIN_SHA, warm_seed=True,
            ):
                entered = True

        patches = [
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)),
            patch.object(git_ops, '_warm_lane_base_resolvable', return_value=WarmBaseHealth.OK),
            patch.object(git_ops, '_seed_warm_lane', new=AsyncMock(return_value=0)),
        ]
        if raiser == '_warm_lane_base_resolvable':
            patches[1] = patch.object(
                git_ops, '_warm_lane_base_resolvable', side_effect=RuntimeError('boom'),
            )
        else:
            patches[2] = patch.object(
                git_ops, '_seed_warm_lane', new=AsyncMock(side_effect=RuntimeError('boom')),
            )

        with patches[0], patches[1], patches[2]:
            asyncio.run(_body())  # must not raise

        assert entered, 'expected the CM body to still run (fail-soft) on a gate exception'
        remove_calls = [c for c in calls if 'worktree' in c and 'remove' in c]
        assert len(remove_calls) == 1, f'expected cleanup remove to still run; got {remove_calls}'

    @pytest.mark.parametrize('health', [WarmBaseHealth.ABSENT, WarmBaseHealth.INDETERMINATE])
    def test_warm_seed_skipped_when_base_not_ok(
        self, tmp_path: Path, health: WarmBaseHealth,
    ) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        entered = False

        async def _body() -> None:
            nonlocal entered
            async with git_ops.ephemeral_worktree(
                WorktreeKind.MAIN_PROBE, MAIN_SHA, warm_seed=True,
            ):
                entered = True

        with (
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)),
            patch.object(git_ops, '_seed_warm_lane', new=AsyncMock(return_value=0)) as mock_seed,
            patch.object(git_ops, '_warm_lane_base_resolvable', return_value=health),
        ):
            asyncio.run(_body())

        mock_seed.assert_not_awaited()
        assert entered, 'expected the CM body to still run cold when the base is not OK'

    @pytest.mark.parametrize('kind', [WorktreeKind.MAIN_PROBE, WorktreeKind.MAIN_SWEEP])
    def test_default_warm_seed_false_never_seeds(
        self, tmp_path: Path, kind: WorktreeKind,
    ) -> None:
        git_ops = GitOps(GitConfig(), tmp_path)
        calls: list[list[str]] = []
        entered = False

        async def _body() -> None:
            nonlocal entered
            async with git_ops.ephemeral_worktree(kind, MAIN_SHA):
                entered = True

        with (
            patch('orchestrator.git_ops._run', side_effect=_make_fake_run([0], calls)),
            patch.object(git_ops, '_seed_warm_lane', new=AsyncMock(return_value=0)) as mock_seed,
            patch.object(git_ops, '_warm_lane_base_resolvable', return_value=WarmBaseHealth.OK),
        ):
            asyncio.run(_body())

        mock_seed.assert_not_awaited()
        assert entered, 'expected the CM body to run normally'
