"""GitOps-level wiring of rebase-drop reporting (task 4669).

Every rebase that rewrites a real task branch must report, and record, the
commits it drops because the target already carries their change:
GitOps.rebase_onto_main (requeue, pre-merge, train, stacking),
GitOps.rebase_preserving_task_commits (its guarded wrapper), and the 5618
un-stack, GitOps.unstack_from_unlanded_base.

Real temp git repositories throughout.  Branches live at
``git_ops.worktree_base / <id>`` on ``task/<id>``.  Records are read through
the public RebaseDropLedger; everything else is asserted through git plumbing.

Base fixture (``repo`` fixture):
  main: index.html ('v=43'), a.txt
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio

from orchestrator.branch_stack import UnstackOutcome
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.rebase_drops import (
    REBASE_DROP_REF_NAMESPACE,
    DroppedCommit,
    RebaseDropLedger,
)

# ---------------------------------------------------------------------------
# Helpers (copied locally from test_git_ops_unstack.py, per-file convention)
# ---------------------------------------------------------------------------

_OWN_A = 'feat(7): own a (task 7)'
_BUMP = 'chore(7): bump 43 -> 44 (task 7)'
_OWN_B = 'test(7): own b (task 7)'


async def _git(cwd: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=cwd)
    assert rc == 0, f'git {args} failed: {err}'
    return out


async def _init_repo(path: Path) -> None:
    await _git(path, 'init', '-b', 'main')
    await _git(path, 'config', 'user.email', 'test@test.com')
    await _git(path, 'config', 'user.name', 'Test')


async def _commit_files(path: Path, writes: dict[str, str], msg: str) -> str:
    """Write and commit exactly *writes*; a bare ``git add -A`` in the main
    checkout would stage the nested ``.worktrees/*`` checkouts as gitlinks."""
    for name, content in writes.items():
        (path / name).write_text(content)
    await _git(path, 'add', '--', *writes)
    await _git(path, 'commit', '-m', msg)
    return await _rev_parse(path, 'HEAD')


async def _rev_parse(cwd: Path, ref: str) -> str:
    return (await _git(cwd, 'rev-parse', '--verify', ref)).strip()


def _make_git_ops(repo: Path) -> GitOps:
    cfg = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(cfg, repo)


@dataclass(frozen=True)
class Repo:
    root: Path
    git_ops: GitOps

    @property
    def ledger(self) -> RebaseDropLedger:
        return RebaseDropLedger(self.git_ops.project_root, _run)

    async def add_branch(self, name: str, start_ref: str = 'main') -> Path:
        """Create worktree + branch task/<name> at *start_ref*; return the worktree."""
        wt = self.git_ops.worktree_base / name
        wt.parent.mkdir(parents=True, exist_ok=True)
        await _git(self.root, 'worktree', 'add', '-b', f'task/{name}', str(wt), start_ref)
        return wt

    async def advance_main(self, writes: dict[str, str], msg: str = 'main advances') -> str:
        return await _commit_files(self.root, writes, msg)

    async def subjects(self, branch: str) -> list[str]:
        out = await _git(self.root, 'log', '--reverse', '--format=%s', f'main..{branch}')
        return out.splitlines()

    async def dropped_refs(self) -> list[str]:
        out = await _git(
            self.root, 'for-each-ref', '--format=%(refname)', REBASE_DROP_REF_NAMESPACE,
        )
        return out.split()


@pytest_asyncio.fixture
async def repo(tmp_path: Path) -> Repo:
    root = tmp_path / 'repo'
    root.mkdir()
    await _init_repo(root)
    await _commit_files(root, {'index.html': 'v=43\n', 'a.txt': 'a\n'}, 'initial')
    return Repo(root=root, git_ops=_make_git_ops(root))


@dataclass(frozen=True)
class Task7:
    wt: Path
    bump: str


async def _shape_3490(repo: Repo) -> Task7:
    """task/7: own a, bump 43 -> 44, own b; main then carries the bump plus m.txt."""
    wt = await repo.add_branch('7')
    await _commit_files(wt, {'a.txt': 'a\nown\n'}, _OWN_A)
    bump = await _commit_files(wt, {'index.html': 'v=44\n'}, _BUMP)
    await _commit_files(wt, {'b.txt': 'b\n'}, _OWN_B)
    await repo.advance_main({'index.html': 'v=44\n', 'm.txt': 'm\n'}, 'main: bump and more')
    return Task7(wt=wt, bump=bump)


# ---------------------------------------------------------------------------
# GitOps.rebase_onto_main / rebase_preserving_task_commits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRebaseOntoMainRecordsDrops:
    async def test_rebase_onto_main_records_the_dropped_bump(self, repo: Repo) -> None:
        task = await _shape_3490(repo)

        assert await repo.git_ops.rebase_onto_main(task.wt) is True

        assert await repo.ledger.dropped_of('task/7') == (
            DroppedCommit(sha=task.bump, subject=_BUMP, paths=('index.html',)),
        )
        assert await repo.subjects('task/7') == [_OWN_A, _OWN_B]

    async def test_rebase_preserving_task_commits_records_a_partial_drop(
        self, repo: Repo,
    ) -> None:
        task = await _shape_3490(repo)

        assert await repo.git_ops.rebase_preserving_task_commits(task.wt) is True

        assert await repo.ledger.dropped_of('task/7') == (
            DroppedCommit(sha=task.bump, subject=_BUMP, paths=('index.html',)),
        )
        assert await repo.subjects('task/7') == [_OWN_A, _OWN_B]

    async def test_a_clean_rebase_records_nothing(self, repo: Repo) -> None:
        wt = await repo.add_branch('7')
        await _commit_files(wt, {'a.txt': 'a\nown\n'}, _OWN_A)
        await _commit_files(wt, {'index.html': 'v=44\n'}, _BUMP)
        await repo.advance_main({'m.txt': 'm\n'})

        assert await repo.git_ops.rebase_onto_main(wt) is True

        assert await repo.dropped_refs() == []
        assert await repo.subjects('task/7') == [_OWN_A, _BUMP]


# ---------------------------------------------------------------------------
# GitOps.unstack_from_unlanded_base (through branch_stack.rebase_own_delta)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUnstackRecordsDrops:
    async def test_unstack_records_only_the_dropped_own_commit(self, repo: Repo) -> None:
        p_wt = await repo.add_branch('P')
        await _commit_files(p_wt, {'p.txt': 'p\n'}, 'feat(P): p1')
        await _commit_files(p_wt, {'p2.txt': 'p2\n'}, 'feat(P): p2')
        m_wt = await repo.add_branch('M')
        await _commit_files(m_wt, {'m.txt': 'm\n'}, 'feat(M): own m')
        await _commit_files(m_wt, {'index.html': 'v=44\n'}, 'chore(M): bump 43 -> 44')
        stacked = await repo.git_ops.stack_train_branches(['P', 'M'])
        assert stacked.survivors == ['P', 'M']
        stacked_bump = await _rev_parse(repo.root, 'task/M')
        await repo.advance_main({'index.html': 'v=44\n', 'x.txt': 'x\n'}, 'main: bump')

        result = await repo.git_ops.unstack_from_unlanded_base('task/M')

        assert result.outcome is UnstackOutcome.UNSTACKED
        assert await repo.ledger.dropped_of('task/M') == (
            DroppedCommit(
                sha=stacked_bump, subject='chore(M): bump 43 -> 44', paths=('index.html',),
            ),
        )
        assert await repo.ledger.dropped_of('task/P') == ()
        assert await repo.subjects('task/M') == ['feat(M): own m']
