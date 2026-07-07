"""Tests for GitOps.reap_interactive_worktrees — task δ (2012).

Crash-safety reaper for the ``_iact-*`` interactive-worktree band that task α
(2010) mints via ``GitOps.create_interactive_worktree``.  Mirrors
``prune_stale_merge_worktrees``'s shape (enumerate registered worktrees under
``worktree_base``, ``git worktree remove --force`` + a single
``git worktree prune``) but with a δ-specific per-worktree reap predicate:
landed-on-main (merge marker), TTL-idle (stamp ``created_at`` for
no-unmerged-work worktrees, HEAD commit time for live ones), and
disk-pressure eviction (idle-only).

Fixtures mirror test_interactive_worktree.py's real-temp-git-repo pattern
(``_init_repo``/``_commit_file``) — no seed script is needed since reaper
tests don't care about ``warm=True/False``.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from orchestrator.artifacts import TaskArtifacts
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, ReapedInteractiveWorktree, _run


def _stamp_path(worktree: Path) -> Path:
    """Resolve the interactive.json stamp at its new .task-meta location
    (W11 gamma .task -> .task-meta relocation; the stamp is a SIBLING of the
    worktree under ``<worktree_base>/.task-meta/<name>``, not inside it)."""
    return (
        TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
        / 'interactive.json'
    )

# ---------------------------------------------------------------------------
# Repo fixture + helpers (mirrors test_interactive_worktree.py)
# ---------------------------------------------------------------------------


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def iw_git_repo(tmp_path: Path) -> Path:
    """Temporary git repo with an initial commit for interactive-worktree-reaper tests."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _commit_file(repo: Path, name: str, content: str, message: str) -> str:
    """Write+commit a file on the repo's current branch; return the new commit SHA."""
    (repo / name).write_text(content)
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', message], cwd=repo)
    rc, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=repo)
    assert rc == 0, f'rev-parse HEAD failed after committing {name!r}'
    return sha.strip()


def _read_stamp(path: Path) -> dict:
    return json.loads(_stamp_path(path).read_text())


def _write_stamp(path: Path, stamp: dict) -> None:
    stamp_path = _stamp_path(path)
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    stamp_path.write_text(json.dumps(stamp))


def _backdate_stamp(path: Path, created_at: datetime) -> None:
    """Rewrite the ``.task/interactive.json`` stamp's ``created_at`` field."""
    stamp = _read_stamp(path)
    stamp['created_at'] = created_at.isoformat()
    _write_stamp(path, stamp)


async def _registered_worktree_paths(repo: Path) -> set[str]:
    """Return the set of registered worktree paths (resolved) via `git worktree list`."""
    rc, out, _ = await _run(['git', 'worktree', 'list', '--porcelain'], cwd=repo)
    assert rc == 0, 'git worktree list --porcelain failed'
    paths = set()
    for line in out.splitlines():
        if line.startswith('worktree '):
            paths.add(str(Path(line[len('worktree '):].strip()).resolve()))
    return paths


# ---------------------------------------------------------------------------
# Step-01: core TTL rule
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapInteractiveWorktreesTtlRule:
    """RED — reap_interactive_worktrees: TTL-idle (no-unmerged) vs within-TTL-live."""

    async def test_ttl_expired_idle_reaped_recent_live_preserved(
        self, iw_git_repo: Path,
    ) -> None:
        """A: no commits, stamp older than ttl -> reaped.

        B: recent own-commit (unmerged, beyond main), stamp within ttl ->
        preserved.
        """
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info_a = await git_ops.create_interactive_worktree('a')
        info_b = await git_ops.create_interactive_worktree('b')

        now = datetime.now(UTC)
        # A: backdate its stamp to well beyond the TTL; no own-commits (HEAD
        # == main tip at creation time, never advanced).
        _backdate_stamp(
            info_a.path,
            now - timedelta(seconds=config.interactive_worktree_ttl + 3600),
        )
        # B: add a recent own-commit (HEAD now beyond main); its stamp is
        # already fresh (written at create time, moments ago).
        await _commit_file(info_b.path, 'work.txt', 'work\n', 'wip on b')

        reaped = await git_ops.reap_interactive_worktrees(now=now)

        assert all(isinstance(r, ReapedInteractiveWorktree) for r in reaped), (
            f'expected every entry to be a ReapedInteractiveWorktree, got {reaped!r}'
        )

        registered = await _registered_worktree_paths(iw_git_repo)
        assert str(info_a.path.resolve()) not in registered, (
            f'expected A ({info_a.path}) to be removed from `git worktree list`; '
            f'registered: {registered}'
        )
        assert str(info_b.path.resolve()) in registered, (
            f'expected B ({info_b.path}) to remain registered; registered: {registered}'
        )

        reaped_paths = {str(r.path.resolve()) for r in reaped}
        assert str(info_a.path.resolve()) in reaped_paths, (
            f'expected a record for A in the returned list; got {reaped!r}'
        )
        assert str(info_b.path.resolve()) not in reaped_paths, (
            f'expected NO record for B in the returned list; got {reaped!r}'
        )
        a_record = next(
            r for r in reaped if r.path.resolve() == info_a.path.resolve()
        )
        assert a_record.branch == info_a.branch, (
            f'expected reaped record branch {info_a.branch!r}, got {a_record.branch!r}'
        )
        assert a_record.slug == 'a', f"expected reaped record slug 'a', got {a_record.slug!r}"
        assert a_record.reason, 'expected a non-empty reason on the reaped record'


# ---------------------------------------------------------------------------
# Amendment: a `git rev-parse HEAD` failure must defer the reap decision,
# never be conflated with "no commits beyond main" (I2 data-loss guard).
# worktree_head_beyond_main returns None for both cases; the reaper must
# independently re-confirm HEAD is readable before treating None as safe.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapInteractiveWorktreesHeadUnreadable:
    """A HEAD-unreadable candidate is retained this sweep, not force-removed."""

    async def test_head_unreadable_defers_reap_despite_expired_ttl(
        self, iw_git_repo: Path, caplog,
    ) -> None:
        """X: no own-commits, stamp older than ttl (would normally reap as
        'ttl_idle') but HEAD is unreadable this sweep -> PRESERVED + WARN.
        """
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info_x = await git_ops.create_interactive_worktree('unreadable-x')
        now = datetime.now(UTC)
        _backdate_stamp(
            info_x.path,
            now - timedelta(seconds=config.interactive_worktree_ttl + 3600),
        )

        git_ops._worktree_head_readable = AsyncMock(return_value=False)

        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            reaped = await git_ops.reap_interactive_worktrees(now=now)

        registered = await _registered_worktree_paths(iw_git_repo)
        assert str(info_x.path.resolve()) in registered, (
            f'expected X ({info_x.path}) to be PRESERVED when HEAD could '
            f'not be read this sweep, despite an expired stamp; '
            f'registered: {registered}'
        )
        x_paths = {r.path.resolve() for r in reaped}
        assert info_x.path.resolve() not in x_paths, (
            f'expected NO reaped record for X; got {reaped!r}'
        )

        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'unreadable-x' in r.getMessage()
        ]
        assert warnings, (
            f'expected a WARNING naming the HEAD-unreadable worktree (X); '
            f'all warnings: {[r.getMessage() for r in caplog.records]}'
        )


# ---------------------------------------------------------------------------
# Amendment: the disk-guard subprocess is deferred until a candidate
# actually reaches the idle-and-within-TTL branch, not invoked unconditionally
# at the top of every sweep.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapInteractiveWorktreesDiskGuardLaziness:
    """The disk-guard check is skipped when no candidate needs it, and
    invoked at most once per sweep when one does."""

    async def test_disk_guard_not_invoked_when_no_iact_worktrees(
        self, iw_git_repo: Path,
    ) -> None:
        """No _iact-* worktrees registered -> the disk-guard is never called."""
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)
        git_ops._run_warm_lane_disk_guard = AsyncMock(return_value=0)

        reaped = await git_ops.reap_interactive_worktrees(now=datetime.now(UTC))

        assert reaped == []
        git_ops._run_warm_lane_disk_guard.assert_not_awaited()

    async def test_disk_guard_not_invoked_when_only_live_candidate(
        self, iw_git_repo: Path,
    ) -> None:
        """Only a live (unmerged-commit), within-ttl worktree is present ->
        the idle/disk-pressure branch is never reached, so the guard is
        never called."""
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)
        git_ops._run_warm_lane_disk_guard = AsyncMock(return_value=0)

        info_live = await git_ops.create_interactive_worktree('live-only')
        await _commit_file(info_live.path, 'work.txt', 'work\n', 'wip')

        reaped = await git_ops.reap_interactive_worktrees(now=datetime.now(UTC))

        assert reaped == []
        git_ops._run_warm_lane_disk_guard.assert_not_awaited()

    async def test_disk_guard_invoked_once_for_multiple_idle_candidates(
        self, iw_git_repo: Path,
    ) -> None:
        """Two idle, within-ttl candidates both reach the disk-pressure
        branch -> the guard is invoked exactly once (result reused), not
        once per candidate."""
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)
        git_ops._run_warm_lane_disk_guard = AsyncMock(return_value=0)

        await git_ops.create_interactive_worktree('idle-one')
        await git_ops.create_interactive_worktree('idle-two')

        reaped = await git_ops.reap_interactive_worktrees(now=datetime.now(UTC))

        assert reaped == []
        git_ops._run_warm_lane_disk_guard.assert_awaited_once()


# ---------------------------------------------------------------------------
# Step-03: landed-on-main detection + fresh-worktree preservation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapInteractiveWorktreesLanded:
    """RED — a merge marker on main reaps regardless of TTL; a fresh
    zero-commit worktree at the main tip must NOT be mistaken for 'landed'
    (guards against an is_ancestor false-positive)."""

    async def test_landed_branch_reaped_fresh_worktree_preserved(
        self, iw_git_repo: Path,
    ) -> None:
        """C: own-commit + merged into main (canonical subject) -> reaped as
        'landed' despite being within ttl. D: fresh zero-commit worktree at
        the (now-advanced) main tip, within ttl -> preserved."""
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info_c = await git_ops.create_interactive_worktree('landed-c')
        await _commit_file(info_c.path, 'work.txt', 'work\n', 'wip on c')

        # Merge task/landed-c into main using the canonical subject, run from
        # the main checkout (project_root). The branch stays checked out in
        # info_c.path throughout, exactly as a live interactive session would
        # leave it — merging FROM a branch checked out elsewhere is legal.
        merge_rc, _, merge_err = await _run(
            [
                'git', 'merge', '--no-ff',
                '-m', f'Merge {info_c.branch} into {config.main_branch}',
                info_c.branch,
            ],
            cwd=iw_git_repo,
        )
        assert merge_rc == 0, (
            f'setup: merge of {info_c.branch} into main failed: {merge_err}'
        )

        # D is created AFTER the merge, so its "main tip" already includes
        # C's merge commit -- zero own-commits, HEAD == main tip exactly like
        # the is_ancestor false-positive shape documented on
        # worktree_head_beyond_main / find_task_citation_commit.
        info_d = await git_ops.create_interactive_worktree('fresh-d')

        now = datetime.now(UTC)
        reaped = await git_ops.reap_interactive_worktrees(now=now)

        registered = await _registered_worktree_paths(iw_git_repo)
        assert str(info_c.path.resolve()) not in registered, (
            f'expected C ({info_c.path}) to be removed as landed; '
            f'registered: {registered}'
        )
        assert str(info_d.path.resolve()) in registered, (
            f'expected D ({info_d.path}) to remain (fresh, zero commits, '
            f'within ttl); registered: {registered}'
        )

        c_record = next(
            (r for r in reaped if r.path.resolve() == info_c.path.resolve()), None,
        )
        assert c_record is not None, f'expected a reaped record for C; got {reaped!r}'
        assert c_record.reason == 'landed', (
            f"expected C's reap reason to be 'landed', got {c_record.reason!r}"
        )
        d_paths = {r.path.resolve() for r in reaped}
        assert info_d.path.resolve() not in d_paths, (
            f'expected NO reaped record for D; got {reaped!r}'
        )


# ---------------------------------------------------------------------------
# Step-05: disk-pressure eviction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapInteractiveWorktreesDiskPressure:
    """RED — disk pressure evicts idle _iact-* worktrees but never live ones."""

    async def test_disk_pressure_evicts_idle_preserves_live(
        self, iw_git_repo: Path,
    ) -> None:
        """E: idle, no own-commits, within ttl -> evicted under pressure.

        F: recent own-commit (unmerged), within ttl -> PRESERVED even under
        pressure (unmerged work is never dropped on pressure alone).
        """
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info_e = await git_ops.create_interactive_worktree('idle-e')
        info_f = await git_ops.create_interactive_worktree('live-f')
        await _commit_file(info_f.path, 'work.txt', 'work\n', 'wip on f')

        now = datetime.now(UTC)
        git_ops._run_warm_lane_disk_guard = AsyncMock(return_value=75)
        reaped = await git_ops.reap_interactive_worktrees(now=now)

        registered = await _registered_worktree_paths(iw_git_repo)
        assert str(info_e.path.resolve()) not in registered, (
            f'expected E ({info_e.path}) to be evicted under disk pressure; '
            f'registered: {registered}'
        )
        assert str(info_f.path.resolve()) in registered, (
            f'expected F ({info_f.path}) to be PRESERVED under disk pressure '
            f'(unmerged work is never dropped on pressure alone); '
            f'registered: {registered}'
        )

        e_record = next(
            (r for r in reaped if r.path.resolve() == info_e.path.resolve()), None,
        )
        assert e_record is not None, f'expected a reaped record for E; got {reaped!r}'
        assert e_record.reason == 'disk_pressure', (
            f"expected E's reap reason to be 'disk_pressure', got {e_record.reason!r}"
        )
        f_paths = {r.path.resolve() for r in reaped}
        assert info_f.path.resolve() not in f_paths, (
            f'expected NO reaped record for F; got {reaped!r}'
        )

    async def test_control_no_pressure_preserves_idle_worktree(
        self, iw_git_repo: Path,
    ) -> None:
        """Control: with the guard mocked to 0 (no pressure), an idle
        within-ttl worktree is preserved — proves the eviction above is
        pressure-driven, not an unrelated TTL/landed rule."""
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info_e = await git_ops.create_interactive_worktree('idle-e2')

        now = datetime.now(UTC)
        git_ops._run_warm_lane_disk_guard = AsyncMock(return_value=0)
        reaped = await git_ops.reap_interactive_worktrees(now=now)

        registered = await _registered_worktree_paths(iw_git_repo)
        assert str(info_e.path.resolve()) in registered, (
            f'expected E ({info_e.path}) to be preserved with no disk '
            f'pressure; registered: {registered}'
        )
        assert reaped == [], (
            f'expected nothing reaped under no pressure; got {reaped!r}'
        )


# ---------------------------------------------------------------------------
# Step-07: fail-soft stamp handling + band isolation (I1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReapInteractiveWorktreesFailSoftStamp:
    """RED — a missing/corrupt ``.task/interactive.json`` stamp must never
    become an indefinite leak (I2) NOR a cause of silently dropped unmerged
    work."""

    async def test_missing_stamp_reaped_corrupt_with_unmerged_preserved(
        self, iw_git_repo: Path, caplog,
    ) -> None:
        """G: stamp deleted, no own-commits -> reaped ('stale_no_stamp') + WARN.

        H: stamp corrupted (invalid JSON), WITH an unmerged own-commit ->
        PRESERVED — a corrupt stamp must never cause unmerged work to be
        silently dropped.
        """
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        info_g = await git_ops.create_interactive_worktree('no-stamp-g')
        _stamp_path(info_g.path).unlink()

        info_h = await git_ops.create_interactive_worktree('corrupt-h')
        _stamp_path(info_h.path).write_text('{not valid json')
        await _commit_file(info_h.path, 'work.txt', 'work\n', 'wip on h')

        now = datetime.now(UTC)
        with caplog.at_level(logging.WARNING, logger='orchestrator.git_ops'):
            reaped = await git_ops.reap_interactive_worktrees(now=now)

        registered = await _registered_worktree_paths(iw_git_repo)
        assert str(info_g.path.resolve()) not in registered, (
            f'expected G ({info_g.path}) to be reaped despite its missing '
            f'stamp; registered: {registered}'
        )
        assert str(info_h.path.resolve()) in registered, (
            f'expected H ({info_h.path}) to be PRESERVED — a corrupt stamp '
            f'must never cause unmerged work to be silently dropped; '
            f'registered: {registered}'
        )

        g_record = next(
            (r for r in reaped if r.path.resolve() == info_g.path.resolve()), None,
        )
        assert g_record is not None, f'expected a reaped record for G; got {reaped!r}'
        assert g_record.reason == 'stale_no_stamp', (
            f"expected G's reap reason to be 'stale_no_stamp', got {g_record.reason!r}"
        )
        h_paths = {r.path.resolve() for r in reaped}
        assert info_h.path.resolve() not in h_paths, (
            f'expected NO reaped record for H; got {reaped!r}'
        )

        warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'no-stamp-g' in r.getMessage()
        ]
        assert warnings, (
            f'expected a WARNING naming the missing-stamp worktree (G); '
            f'all warnings: {[r.getMessage() for r in caplog.records]}'
        )


@pytest.mark.asyncio
class TestReapInteractiveWorktreesBandIsolation:
    """RED — the reaper must never enumerate, evaluate, or remove worktrees
    outside the ``_iact-*`` band (isolation invariant I1)."""

    async def test_non_iact_sibling_worktrees_never_touched(
        self, iw_git_repo: Path,
    ) -> None:
        """``_lane-*``, ``_merge-*``, and plain numeric task-id worktrees
        registered alongside the ``_iact-*`` band must return no records and
        must never be removed."""
        config = GitConfig()
        git_ops = GitOps(config, iw_git_repo)

        worktree_base = git_ops.worktree_base
        sibling_paths = {
            worktree_base / '_lane-x': 'lane-branch-x',
            worktree_base / '_merge-x': 'merge-branch-x',
            worktree_base / '123': 'task-branch-123',
        }
        for path, branch in sibling_paths.items():
            rc, _, err = await _run(
                ['git', 'worktree', 'add', '-b', branch, str(path), 'main'],
                cwd=iw_git_repo,
            )
            assert rc == 0, f'setup: failed to register {path}: {err}'

        # A genuine _iact-* candidate too, so the sweep does real work and
        # isn't a no-op that would vacuously "preserve" the siblings.
        info_i = await git_ops.create_interactive_worktree('real-i')

        now = datetime.now(UTC)
        reaped = await git_ops.reap_interactive_worktrees(now=now)

        registered = await _registered_worktree_paths(iw_git_repo)
        for path in sibling_paths:
            assert str(path.resolve()) in registered, (
                f'expected {path} to remain registered (outside the '
                f'_iact-* band); registered: {registered}'
            )
        assert str(info_i.path.resolve()) in registered, (
            f'expected I ({info_i.path}) to remain (fresh, within ttl); '
            f'registered: {registered}'
        )

        reaped_paths = {r.path.resolve() for r in reaped}
        for path in sibling_paths:
            assert path.resolve() not in reaped_paths, (
                f'expected NO reaped record for {path}; got {reaped!r}'
            )
