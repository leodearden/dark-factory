"""Tests for git_ops.materialize_member_solo (train un-stack helper).

Step-1 (RED): Real-git temp-repo tests asserting that materialize_member_solo
correctly un-stacks a member's own delta onto current main.

Fixture: main + initial commit; task/b1 adds a.txt; task/b2 is stacked
on task/b1 and adds b.txt — so task/b2's tree contains both a.txt and b.txt.

Assertions:
  - b2 un-stacked: diff vs main contains ONLY b.txt (NOT a.txt)
  - b1 un-stacked (anchor, predecessor_ref='main'): only a.txt in diff
  - conflict case: returns None, NO dangling _solo-* worktree or branch
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, WorktreeInfo, _run

# ---------------------------------------------------------------------------
# Helpers: repo setup
# ---------------------------------------------------------------------------


async def _init_repo(path: Path) -> None:
    """git init -b main + configure identity."""
    await _run(['git', 'init', '-b', 'main'], cwd=path)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=path)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=path)


async def _commit_all(path: Path, msg: str) -> str:
    """Stage all and commit; return commit SHA."""
    await _run(['git', 'add', '-A'], cwd=path)
    await _run(['git', 'commit', '-m', msg], cwd=path)
    _, sha, _ = await _run(['git', 'rev-parse', 'HEAD'], cwd=path)
    return sha.strip()


def _make_git_ops(repo: Path) -> GitOps:
    """Return a GitOps pointing at *repo* with default test GitConfig."""
    cfg = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(cfg, repo)


async def _add_stacked_branch(
    git_ops: GitOps,
    name: str,
    start_ref: str,
    *,
    writes: dict[str, str],
) -> None:
    """Create worktree + branch task/<name> from start_ref, write files, commit.

    *writes* maps file name → content.  Files are written (potentially
    overwriting content inherited from start_ref) and staged before
    committing.
    """
    full_branch = f'task/{name}'
    wt = git_ops.worktree_base / name
    wt.parent.mkdir(parents=True, exist_ok=True)
    await _run(
        ['git', 'worktree', 'add', '-b', full_branch, str(wt), start_ref],
        cwd=git_ops.project_root,
    )
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=wt)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=wt)
    for fname, content in writes.items():
        (wt / fname).write_text(content)
    await _run(['git', 'add', '-A'], cwd=wt)
    await _run(['git', 'commit', '-m', f'{name}: write files'], cwd=wt)


async def _diff_files_vs_main(git_ops: GitOps, branch: str) -> set[str]:
    """Return the set of file names that differ between main and *branch*."""
    _, diff_out, _ = await _run(
        ['git', 'diff', '--name-only', 'main', branch],
        cwd=git_ops.project_root,
    )
    return {f.strip() for f in diff_out.splitlines() if f.strip()}


async def _worktree_names(git_ops: GitOps) -> set[str]:
    """Return the set of registered worktree directory names."""
    _, out, _ = await _run(
        ['git', 'worktree', 'list', '--porcelain'],
        cwd=git_ops.project_root,
    )
    names: set[str] = set()
    for line in out.splitlines():
        if line.startswith('worktree '):
            wt_path = Path(line[len('worktree '):].strip())
            names.add(wt_path.name)
    return names


async def _branch_names(git_ops: GitOps, pattern: str) -> set[str]:
    """Return branches matching *pattern* (git branch --list <pattern>)."""
    _, out, _ = await _run(
        ['git', 'branch', '--list', pattern],
        cwd=git_ops.project_root,
    )
    return {b.strip().lstrip('* ') for b in out.splitlines() if b.strip()}


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMaterializeMemberSolo:
    """Real-git tests for materialize_member_solo."""

    async def test_b2_unstacked_contains_only_own_delta(self, tmp_path: Path) -> None:
        """b2 un-stacked onto main: diff vs main shows ONLY b.txt, NOT a.txt."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)

        # Initial commit on main (README only)
        (repo / 'README.md').write_text('initial\n')
        await _commit_all(repo, 'initial')

        git_ops = _make_git_ops(repo)

        # task/b1 (anchor): adds a.txt
        await _add_stacked_branch(git_ops, 'b1', 'main', writes={'a.txt': 'b1 content\n'})

        # task/b2 (stacked on b1): adds b.txt — cumulative tree has a.txt + b.txt
        await _add_stacked_branch(git_ops, 'b2', 'task/b1', writes={'b.txt': 'b2 content\n'})

        # Pre-condition: task/b2 carries BOTH a.txt and b.txt (cumulative stack)
        cumulative_files = await _diff_files_vs_main(git_ops, 'task/b2')
        assert 'a.txt' in cumulative_files, 'pre-condition: task/b2 should carry a.txt'
        assert 'b.txt' in cumulative_files, 'pre-condition: task/b2 should carry b.txt'

        # ---- actual test ----
        result = await git_ops.materialize_member_solo('b2', predecessor_ref='task/b1')

        assert result is not None, 'materialize_member_solo should succeed for non-conflicting b2'
        assert isinstance(result, WorktreeInfo)

        # Solo branch must show only b.txt (b2's own delta), NOT a.txt
        solo_files = await _diff_files_vs_main(git_ops, '_solo-b2')
        assert 'b.txt' in solo_files, 'b2 solo should contain b.txt'
        assert 'a.txt' not in solo_files, 'b2 solo must NOT contain a.txt (b1 delta)'

        # WorktreeInfo.base_commit should equal the rebased tip SHA
        _, head_sha, _ = await _run(
            ['git', 'rev-parse', '_solo-b2'],
            cwd=repo,
        )
        assert result.base_commit == head_sha.strip()

        # The solo worktree path should exist
        assert result.path.is_dir()

        # Clean up
        await git_ops.cleanup_merge_worktree(result.path)

    async def test_b1_anchor_contains_only_own_delta(self, tmp_path: Path) -> None:
        """b1 (anchor, predecessor_ref='main'): diff vs main shows only a.txt."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)

        (repo / 'README.md').write_text('initial\n')
        await _commit_all(repo, 'initial')

        git_ops = _make_git_ops(repo)
        await _add_stacked_branch(git_ops, 'b1', 'main', writes={'a.txt': 'b1 content\n'})

        result = await git_ops.materialize_member_solo('b1', predecessor_ref='main')

        assert result is not None
        assert isinstance(result, WorktreeInfo)

        solo_files = await _diff_files_vs_main(git_ops, '_solo-b1')
        assert 'a.txt' in solo_files, 'b1 solo should contain a.txt'
        assert len(solo_files) == 1, f'b1 solo should differ only in a.txt, got {solo_files}'

        # Clean up
        await git_ops.cleanup_merge_worktree(result.path)

    async def test_conflict_returns_none_no_dangling_worktree(self, tmp_path: Path) -> None:
        """Rebase conflict during un-stack → None; no _solo-* worktree or branch left."""
        repo = tmp_path / 'repo'
        repo.mkdir()
        await _init_repo(repo)

        # Initial commit: shared.txt = "original"
        (repo / 'shared.txt').write_text('original\n')
        (repo / 'README.md').write_text('initial\n')
        await _commit_all(repo, 'initial')

        git_ops = _make_git_ops(repo)

        # task/cpred: shared.txt = "pred" (changes "original" → "pred")
        await _add_stacked_branch(
            git_ops, 'cpred', 'main',
            writes={'shared.txt': 'pred\n'},
        )

        # task/cmember (stacked on cpred): shared.txt = "member" (changes "pred" → "member")
        # The member's own delta: patch "-pred +member"
        # Applied to main (shared.txt = "original") → conflict!
        await _add_stacked_branch(
            git_ops, 'cmember', 'task/cpred',
            writes={'shared.txt': 'member\n'},
        )

        # Record which worktrees exist before the call
        before = await _worktree_names(git_ops)

        result = await git_ops.materialize_member_solo('cmember', predecessor_ref='task/cpred')

        assert result is None, 'conflict should return None'

        # No dangling _solo-* worktrees should have been added
        after = await _worktree_names(git_ops)
        solo_added = {w for w in (after - before) if w.startswith('_solo-')}
        assert not solo_added, f'Dangling _solo- worktrees left: {solo_added}'

        # No dangling _solo-* branches
        leftover_branches = await _branch_names(git_ops, '_solo-*')
        assert not leftover_branches, f'Dangling _solo- branches left: {leftover_branches}'
