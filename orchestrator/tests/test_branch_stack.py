"""Tests for orchestrator.branch_stack: stack-base provenance (task 5618).

A branch the train stacker rebased onto an unlanded predecessor carries the
predecessor's commits below its own.  branch_stack records the base each
branch was stacked onto (StackBaseLedger) and computes which commits of the
branch are therefore not its own (foreign_commit_cut).

Every test runs against a REAL temp git repository; the git runner is
orchestrator.git_ops._run, injected exactly as production injects it.

Fixture shape (``stacked`` fixture):
  main: base.txt
  task/P: from main, two commits (p1.txt, then p2.txt)
  task/M: worktree created at P's tip, two own commits (m1.txt, m2.txt)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio

from orchestrator.branch_stack import (
    StackBaseLedger,
    StackInspectionError,
    base_owners,
    foreign_commit_cut,
)
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run

# ---------------------------------------------------------------------------
# Helpers (copied locally from test_git_ops_train_solo.py, per-file convention)
# ---------------------------------------------------------------------------


async def _init_repo(path: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=path)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=path)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=path)


async def _commit_all(path: Path, msg: str) -> str:
    await _run(['git', 'add', '-A'], cwd=path)
    await _run(['git', 'commit', '-m', msg], cwd=path)
    return await _rev_parse(path, 'HEAD')


async def _commit_file(path: Path, name: str, content: str) -> str:
    (path / name).write_text(content)
    return await _commit_all(path, f'write {name}')


async def _rev_parse(cwd: Path, ref: str) -> str:
    rc, out, err = await _run(['git', 'rev-parse', '--verify', ref], cwd=cwd)
    assert rc == 0, err
    return out.strip()


def _make_git_ops(repo: Path) -> GitOps:
    cfg = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(cfg, repo)


async def _add_branch(git_ops: GitOps, name: str, start_ref: str) -> Path:
    """Create worktree + branch task/<name> at *start_ref*; return the worktree."""
    wt = git_ops.worktree_base / name
    wt.parent.mkdir(parents=True, exist_ok=True)
    rc, _, err = await _run(
        ['git', 'worktree', 'add', '-b', f'task/{name}', str(wt), start_ref],
        cwd=git_ops.project_root,
    )
    assert rc == 0, err
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=wt)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=wt)
    return wt


async def _git(cwd: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=cwd)
    assert rc == 0, f'git {args} failed: {err}'
    return out


@dataclass(frozen=True)
class Stacked:
    repo: Path
    git_ops: GitOps
    p_first: str
    p_tip: str
    m_tip: str

    @property
    def m_wt(self) -> Path:
        return self.git_ops.worktree_base / 'M'

    @property
    def p_wt(self) -> Path:
        return self.git_ops.worktree_base / 'P'


@pytest_asyncio.fixture
async def stacked(tmp_path: Path) -> Stacked:
    repo = tmp_path / 'repo'
    repo.mkdir()
    await _init_repo(repo)
    await _commit_file(repo, 'base.txt', 'base\n')
    git_ops = _make_git_ops(repo)

    p_wt = await _add_branch(git_ops, 'P', 'main')
    p_first = await _commit_file(p_wt, 'p1.txt', 'p1\n')
    p_tip = await _commit_file(p_wt, 'p2.txt', 'p2\n')

    m_wt = await _add_branch(git_ops, 'M', p_tip)
    await _commit_file(m_wt, 'm1.txt', 'm1\n')
    m_tip = await _commit_file(m_wt, 'm2.txt', 'm2\n')

    return Stacked(
        repo=repo, git_ops=git_ops, p_first=p_first, p_tip=p_tip, m_tip=m_tip,
    )


async def _cut(s: Stacked, *, tip: str, base: str) -> str | None:
    main_sha = await _rev_parse(s.repo, 'main')
    return await foreign_commit_cut(
        _run, s.repo, tip=tip, main_ref=main_sha, base=base,
    )


# ---------------------------------------------------------------------------
# StackBaseLedger
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStackBaseLedger:
    async def test_record_then_base_of_round_trips(self, stacked: Stacked) -> None:
        ledger = StackBaseLedger(stacked.repo, _run)

        await ledger.record('task/M', stacked.p_tip)

        assert await ledger.base_of('task/M') == stacked.p_tip
        show_ref = await _git(stacked.repo, 'show-ref')
        private_refs = [
            line.split()[1]
            for line in show_ref.splitlines()
            if 'refs/dark-factory/' in line
        ]
        assert private_refs == ['refs/dark-factory/stack-base/task/M']
        assert ledger.ref_for('task/M') == 'refs/dark-factory/stack-base/task/M'

    async def test_base_of_unrecorded_branch_is_none(self, stacked: Stacked) -> None:
        ledger = StackBaseLedger(stacked.repo, _run)

        assert await ledger.base_of('task/M') is None

    async def test_forget_clears_and_is_idempotent(self, stacked: Stacked) -> None:
        ledger = StackBaseLedger(stacked.repo, _run)
        await ledger.record('task/M', stacked.p_tip)

        await ledger.forget('task/M')
        assert await ledger.base_of('task/M') is None

        await ledger.forget('task/M')
        assert await ledger.base_of('task/M') is None

    async def test_record_survives_base_branch_deletion(self, stacked: Stacked) -> None:
        ledger = StackBaseLedger(stacked.repo, _run)
        await ledger.record('task/M', stacked.p_tip)

        await _git(stacked.repo, 'worktree', 'remove', '--force', str(stacked.p_wt))
        await _git(stacked.repo, 'branch', '-D', 'task/P')

        assert await ledger.base_of('task/M') == stacked.p_tip
        object_type = await _git(stacked.repo, 'cat-file', '-t', stacked.p_tip)
        assert object_type.strip() == 'commit'


# ---------------------------------------------------------------------------
# foreign_commit_cut
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestForeignCommitCut:
    async def test_intact_stack_cuts_at_base_tip(self, stacked: Stacked) -> None:
        cut = await _cut(stacked, tip=stacked.m_tip, base=stacked.p_tip)

        assert cut == stacked.p_tip

    async def test_replayed_stack_cuts_at_replayed_copy_of_base_tip(
        self, stacked: Stacked,
    ) -> None:
        await _commit_file(stacked.repo, 'unrelated.txt', 'main moved\n')
        await _git(stacked.m_wt, 'rebase', 'main')
        replayed_tip = await _rev_parse(stacked.repo, 'task/M')

        cut = await _cut(stacked, tip=replayed_tip, base=stacked.p_tip)

        ordered = (
            await _git(
                stacked.repo, 'rev-list', '--reverse', '--topo-order', 'main..task/M',
            )
        ).split()
        assert cut == ordered[1]
        assert cut != stacked.p_tip

    async def test_branch_with_no_foreign_commits_has_no_cut(
        self, stacked: Stacked,
    ) -> None:
        x_wt = await _add_branch(stacked.git_ops, 'X', 'main')
        await _commit_file(x_wt, 'x1.txt', 'x1\n')
        x_tip = await _commit_file(x_wt, 'x2.txt', 'x2\n')

        assert await _cut(stacked, tip=x_tip, base=stacked.p_tip) is None

    async def test_landed_base_has_no_cut(self, stacked: Stacked) -> None:
        await _git(stacked.repo, 'merge', '--no-ff', '-m', 'land P', 'task/P')

        assert await _cut(stacked, tip=stacked.m_tip, base=stacked.p_tip) is None

    async def test_foreign_looking_commit_after_own_commit_never_extends_cut(
        self, stacked: Stacked,
    ) -> None:
        x_wt = await _add_branch(stacked.git_ops, 'X', 'main')
        await _commit_file(x_wt, 'x1.txt', 'x1\n')
        await _git(x_wt, 'cherry-pick', stacked.p_first)
        x_tip = await _rev_parse(x_wt, 'HEAD')

        assert await _cut(stacked, tip=x_tip, base=stacked.p_tip) is None

    async def test_unresolvable_tip_raises_inspection_error(
        self, stacked: Stacked,
    ) -> None:
        with pytest.raises(StackInspectionError):
            await _cut(stacked, tip='deadbeef' * 5, base=stacked.p_tip)


# ---------------------------------------------------------------------------
# base_owners
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestBaseOwners:
    async def test_names_the_branch_pointing_at_the_base(
        self, stacked: Stacked,
    ) -> None:
        assert await base_owners(_run, stacked.repo, stacked.p_tip) == ('task/P',)

    async def test_empty_once_the_base_branch_moved(self, stacked: Stacked) -> None:
        await _commit_file(stacked.p_wt, 'p3.txt', 'p3\n')

        assert await base_owners(_run, stacked.repo, stacked.p_tip) == ()
