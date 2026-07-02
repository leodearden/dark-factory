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
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, ReapedInteractiveWorktree, _run

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
    return json.loads((path / '.task' / 'interactive.json').read_text())


def _write_stamp(path: Path, stamp: dict) -> None:
    (path / '.task' / 'interactive.json').write_text(json.dumps(stamp))


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
