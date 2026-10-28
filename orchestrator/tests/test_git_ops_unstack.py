"""GitOps-level tests for stack-base recording and un-stacking (task 5618).

stack_train_branches records the base each member was stacked onto, and
GitOps.unstack_from_unlanded_base strips a never-landed base's commits from a
member so the member carries only its own delta.  The train's anchor is
un-stacked too, before anything is stacked onto it (task 6133).
stack_train_branches also reports each member's tip from just before
stacking, and
GitOps.restore_pre_stack_tip puts a member back on it (task 4953).

Real temp git repositories throughout.  Members live at
``git_ops.worktree_base / <id>`` on branch ``task/<id>``, the layout
stack_train_branches expects.  Stack records are read through the public
StackBaseLedger; everything else is asserted through git plumbing.

Base fixture (``repo`` fixture):
  main: shared.txt ('v0'), other.txt ('v0')
  task/P: from main, edits shared.txt to 'p' and adds p_only.txt
  task/M: from main, adds m.txt
"""

from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest
import pytest_asyncio

from orchestrator.branch_stack import (
    STACKED_ON_UNLANDED_BASE_REASON_PREFIX,
    PreStackTip,
    StackBaseLedger,
    TipRestoreOutcome,
    UnstackOutcome,
    UnstackResult,
)
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run

# ---------------------------------------------------------------------------
# Helpers (copied locally from test_git_ops_train_solo.py, per-file convention)
# ---------------------------------------------------------------------------


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


async def _add_member(
    git_ops: GitOps, name: str, start_ref: str, writes: dict[str, str],
) -> Path:
    """Create worktree + branch task/<name> from *start_ref* with one commit."""
    wt = git_ops.worktree_base / name
    wt.parent.mkdir(parents=True, exist_ok=True)
    await _git(
        git_ops.project_root, 'worktree', 'add', '-b', f'task/{name}', str(wt), start_ref,
    )
    await _git(wt, 'config', 'user.email', 'test@test.com')
    await _git(wt, 'config', 'user.name', 'Test')
    await _commit_files(wt, writes, f'{name}: write files')
    return wt


async def _diff_vs_main(repo: Path, branch: str) -> set[str]:
    out = await _git(repo, 'diff', '--name-only', f'main...{branch}')
    return {line.strip() for line in out.splitlines() if line.strip()}


async def _git_path(wt: Path, name: str) -> Path:
    path = Path((await _git(wt, 'rev-parse', '--git-path', name)).strip())
    return path if path.is_absolute() else wt / path


async def _assert_clean_and_not_rebasing(wt: Path) -> None:
    assert (await _git(wt, 'status', '--porcelain')).strip() == ''
    assert not (await _git_path(wt, 'rebase-merge')).exists()


def _fail_git(
    monkeypatch: pytest.MonkeyPatch, where: Path, prefix: str, *, after: int = 0,
) -> None:
    """Put a ``git`` first on PATH that exits 128 for each call whose arguments
    start with *prefix*, once *after* such calls have run; every other call
    reaches the real git.  ``run_git`` copies ``os.environ`` at each spawn, so
    GitOps sees it (precedent: test_git_fixtures.py::_install_git_spawn_counter)."""
    real_git = shutil.which('git')
    assert real_git is not None
    bin_dir = where / 'bin'
    bin_dir.mkdir(parents=True)
    seen = shlex.quote(str(where / 'matched'))
    shim = bin_dir / 'git'
    shim.write_text(
        '#!/bin/sh\n'
        f'case "$*" in {shlex.quote(prefix)}*)\n'
        f'  printf x >> {seen}\n'
        f'  [ $(wc -c < {seen}) -gt {after} ] && exit 128 ;;\n'
        'esac\n'
        f'exec {shlex.quote(real_git)} "$@"\n'
    )
    shim.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')


@dataclass(frozen=True)
class Repo:
    root: Path
    git_ops: GitOps

    @property
    def ledger(self) -> StackBaseLedger:
        return StackBaseLedger(self.git_ops.project_root, _run)

    def wt(self, name: str) -> Path:
        return self.git_ops.worktree_base / name

    async def sha(self, ref: str) -> str:
        return await _rev_parse(self.root, ref)

    async def advance_main(self, writes: dict[str, str]) -> str:
        return await _commit_files(self.root, writes, 'main advances')


@pytest_asyncio.fixture
async def repo(tmp_path: Path) -> Repo:
    root = tmp_path / 'repo'
    root.mkdir()
    await _init_repo(root)
    await _commit_files(root, {'shared.txt': 'v0\n', 'other.txt': 'v0\n'}, 'initial')
    return Repo(root=root, git_ops=_make_git_ops(root))


async def _add_p_and_m(repo: Repo, *, m_writes: dict[str, str] | None = None) -> None:
    await _add_member(
        repo.git_ops, 'P', 'main', {'shared.txt': 'p\n', 'p_only.txt': 'p\n'},
    )
    await _add_member(repo.git_ops, 'M', 'main', m_writes or {'m.txt': 'm\n'})


async def _stack_p_and_m(repo: Repo, *, m_writes: dict[str, str] | None = None) -> None:
    await _add_p_and_m(repo, m_writes=m_writes)
    result = await repo.git_ops.stack_train_branches(['P', 'M'])
    assert result.survivors == ['P', 'M']


# ---------------------------------------------------------------------------
# stack_train_branches records the base
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStackRecordsBase:
    async def test_each_non_anchor_member_records_its_predecessor_tip(
        self, repo: Repo,
    ) -> None:
        for name in ('A', 'B', 'C'):
            await _add_member(repo.git_ops, name, 'main', {f'{name}.txt': f'{name}\n'})

        result = await repo.git_ops.stack_train_branches(['A', 'B', 'C'])

        assert result.survivors == ['A', 'B', 'C']
        assert await repo.ledger.base_of('task/B') == await repo.sha('task/A')
        assert await repo.ledger.base_of('task/C') == await repo.sha('task/B')
        assert await repo.ledger.base_of('task/A') is None

    async def test_member_ejected_for_conflict_has_no_record(self, repo: Repo) -> None:
        await _add_member(repo.git_ops, 'A', 'main', {'foo.txt': 'alpha\n'})
        await _add_member(repo.git_ops, 'B', 'main', {'foo.txt': 'beta\n'})

        result = await repo.git_ops.stack_train_branches(['A', 'B'])

        assert result.ejected == ['B']
        assert await repo.ledger.base_of('task/B') is None

    async def test_stacking_prunes_records_of_deleted_branches(self, repo: Repo) -> None:
        await _stack_p_and_m(repo)
        await _git(repo.root, 'worktree', 'remove', '--force', str(repo.wt('M')))
        await _git(repo.root, 'branch', '-D', 'task/M')
        for name in ('A', 'B'):
            await _add_member(repo.git_ops, name, 'main', {f'{name}.txt': f'{name}\n'})

        result = await repo.git_ops.stack_train_branches(['A', 'B'])

        assert result.survivors == ['A', 'B']
        assert await repo.ledger.base_of('task/M') is None
        assert await repo.ledger.base_of('task/B') == await repo.sha('task/A')

    async def test_forget_stack_bases_clears_each_members_record(
        self, repo: Repo,
    ) -> None:
        for name in ('A', 'B', 'C'):
            await _add_member(repo.git_ops, name, 'main', {f'{name}.txt': f'{name}\n'})
        await repo.git_ops.stack_train_branches(['A', 'B', 'C'])

        await repo.git_ops.forget_stack_bases(['A', 'B', 'C'])

        for name in ('A', 'B', 'C'):
            assert await repo.ledger.base_of(f'task/{name}') is None

    async def test_restack_sheds_the_earlier_unlanded_base(self, repo: Repo) -> None:
        await _add_member(repo.git_ops, 'P1', 'main', {'p1.txt': 'p1\n'})
        await _add_member(repo.git_ops, 'P2', 'main', {'p2.txt': 'p2\n'})
        await _add_member(repo.git_ops, 'M', 'main', {'m.txt': 'm\n'})
        first = await repo.git_ops.stack_train_branches(['P1', 'M'])
        assert first.survivors == ['P1', 'M']

        second = await repo.git_ops.stack_train_branches(['P2', 'M'])

        assert second.survivors == ['P2', 'M']
        assert await _diff_vs_main(repo.root, 'task/M') == {'p2.txt', 'm.txt'}
        assert await repo.ledger.base_of('task/M') == await repo.sha('task/P2')


# ---------------------------------------------------------------------------
# stack_train_branches un-stacks the anchor too (task 6133)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStackUnstacksTheAnchor:
    async def test_anchor_stacked_on_an_unlanded_base_is_stacked_from_its_own_delta(
        self, repo: Repo,
    ) -> None:
        await _stack_p_and_m(repo)
        await _add_member(repo.git_ops, 'X', 'main', {'x.txt': 'x\n'})

        result = await repo.git_ops.stack_train_branches(['M', 'X'])

        assert result.survivors == ['M', 'X']
        assert await _diff_vs_main(repo.root, 'task/M') == {'m.txt'}
        assert await _diff_vs_main(repo.root, 'task/X') == {'m.txt', 'x.txt'}
        assert await repo.ledger.base_of('task/M') is None
        assert await repo.ledger.base_of('task/X') == await repo.sha('task/M')

    async def test_anchor_that_cannot_be_unstacked_abandons_the_train(
        self, repo: Repo,
    ) -> None:
        await _stack_p_and_m(repo)
        await _add_member(repo.git_ops, 'X', 'main', {'x.txt': 'x\n'})
        (repo.wt('M') / 'm.txt').write_text('uncommitted edit\n')
        m_before = await repo.sha('task/M')
        x_before = await repo.sha('task/X')

        result = await repo.git_ops.stack_train_branches(['M', 'X'])

        assert (result.survivors, result.ejected, result.pre_stack_tips) == (
            [], ['M'], (),
        )
        assert await repo.sha('task/X') == x_before
        assert await repo.sha('task/M') == m_before
        assert await repo.ledger.base_of('task/X') is None
        assert await repo.ledger.base_of('task/M') == await repo.sha('task/P')
        assert (repo.wt('M') / 'm.txt').read_text() == 'uncommitted edit\n'

    async def test_anchor_whose_own_delta_conflicts_abandons_the_train(
        self, repo: Repo,
    ) -> None:
        await _stack_p_and_m(repo, m_writes={'m.txt': 'm\n', 'other.txt': 'm\n'})
        await repo.advance_main({'shared.txt': 'main\n', 'other.txt': 'main\n'})
        await _add_member(repo.git_ops, 'X', 'main', {'x.txt': 'x\n'})
        m_before = await repo.sha('task/M')
        x_before = await repo.sha('task/X')

        result = await repo.git_ops.stack_train_branches(['M', 'X'])

        assert (result.survivors, result.ejected) == ([], ['M'])
        assert await repo.sha('task/X') == x_before
        assert await repo.sha('task/M') == m_before
        assert await repo.ledger.base_of('task/X') is None


# ---------------------------------------------------------------------------
# GitOps.unstack_from_unlanded_base
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUnstackFromUnlandedBase:
    async def test_branch_without_record_is_not_stacked(self, repo: Repo) -> None:
        await _add_p_and_m(repo)
        tip_before = await repo.sha('task/M')

        result = await repo.git_ops.unstack_from_unlanded_base('task/M')

        assert result.outcome is UnstackOutcome.NOT_STACKED
        assert await repo.sha('task/M') == tip_before

    async def test_unlanded_predecessor_is_stripped(self, repo: Repo) -> None:
        await _stack_p_and_m(repo)
        p_tip = await repo.sha('task/P')

        result = await repo.git_ops.unstack_from_unlanded_base('task/M')

        assert result.outcome is UnstackOutcome.UNSTACKED
        assert await _diff_vs_main(repo.root, 'task/M') == {'m.txt'}
        assert await repo.git_ops.is_ancestor('main', 'task/M')
        m_wt = repo.wt('M')
        assert await _rev_parse(m_wt, 'HEAD') == await repo.sha('task/M')
        assert (await _git(m_wt, 'symbolic-ref', 'HEAD')).strip() == 'refs/heads/task/M'
        await _assert_clean_and_not_rebasing(m_wt)
        assert await repo.ledger.base_of('task/M') is None
        assert result.base == p_tip
        assert result.base_owners == ('task/P',)

    async def test_landed_predecessor_is_not_stacked_and_clears_record(
        self, repo: Repo,
    ) -> None:
        await _stack_p_and_m(repo)
        await _git(repo.root, 'merge', '--no-ff', '-m', 'land P', 'task/P')
        tip_before = await repo.sha('task/M')

        result = await repo.git_ops.unstack_from_unlanded_base('task/M')

        assert result.outcome is UnstackOutcome.NOT_STACKED
        assert await repo.sha('task/M') == tip_before
        assert await repo.ledger.base_of('task/M') is None

    async def test_replayed_stack_is_stripped(self, repo: Repo) -> None:
        await _stack_p_and_m(repo)
        await repo.advance_main({'unrelated.txt': 'main moved\n'})
        await _git(repo.wt('M'), 'rebase', 'main')

        result = await repo.git_ops.unstack_from_unlanded_base('task/M')

        assert result.outcome is UnstackOutcome.UNSTACKED
        assert await _diff_vs_main(repo.root, 'task/M') == {'m.txt'}
        count = await _git(repo.root, 'rev-list', '--count', 'main..task/M')
        assert count.strip() == '1'

    async def test_own_conflict_is_reported_with_attribution(self, repo: Repo) -> None:
        await _stack_p_and_m(repo, m_writes={'m.txt': 'm\n', 'other.txt': 'm\n'})
        await repo.advance_main({'shared.txt': 'main\n', 'other.txt': 'main\n'})
        tip_before = await repo.sha('task/M')

        result = await repo.git_ops.unstack_from_unlanded_base('task/M')

        assert result.outcome is UnstackOutcome.CONFLICT
        assert await repo.sha('task/M') == tip_before
        await _assert_clean_and_not_rebasing(repo.wt('M'))
        assert result.conflicted_paths == ('other.txt',)
        assert await repo.ledger.base_of('task/M') == await repo.sha('task/P')
        assert result.stops_merge is True
        reason = result.merge_block_reason()
        assert reason.startswith(STACKED_ON_UNLANDED_BASE_REASON_PREFIX)
        assert 'task/P' in reason
        assert 'other.txt' in reason
        assert f'git rebase --onto main {result.cut} task/M' in reason
        assert 'shared.txt' not in reason

    async def test_rebase_failure_without_a_conflict_blocks(self, repo: Repo) -> None:
        await _stack_p_and_m(repo)
        tip_before = await repo.sha('task/M')
        (await _git_path(repo.wt('M'), 'index.lock')).touch()

        result = await repo.git_ops.unstack_from_unlanded_base('task/M')

        assert result.outcome is UnstackOutcome.BLOCKED
        assert result.conflicted_paths == ()
        assert 'index.lock' in result.detail
        assert await repo.sha('task/M') == tip_before
        assert await repo.ledger.base_of('task/M') == await repo.sha('task/P')
        reason = result.merge_block_reason()
        assert 'index.lock' in reason
        assert 'conflicts in' not in reason

    async def test_dirty_worktree_blocks(self, repo: Repo) -> None:
        await _stack_p_and_m(repo)
        tip_before = await repo.sha('task/M')
        (repo.wt('M') / 'm.txt').write_text('uncommitted edit\n')

        result = await repo.git_ops.unstack_from_unlanded_base('task/M')

        assert result.outcome is UnstackOutcome.BLOCKED
        assert await repo.sha('task/M') == tip_before
        assert await repo.ledger.base_of('task/M') is not None
        assert (repo.wt('M') / 'm.txt').read_text() == 'uncommitted edit\n'

    async def test_branch_held_by_no_worktree_blocks(self, repo: Repo) -> None:
        await _stack_p_and_m(repo)
        tip_before = await repo.sha('task/M')
        await _git(repo.root, 'worktree', 'remove', '--force', str(repo.wt('M')))

        result = await repo.git_ops.unstack_from_unlanded_base('task/M')

        assert result.outcome is UnstackOutcome.BLOCKED
        assert await repo.sha('task/M') == tip_before


# ---------------------------------------------------------------------------
# stack_train_branches reports each member's pre-stack tip (task 4953)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStackReportsPreStackTips:
    async def test_each_stacked_member_reports_its_tip_from_before_stacking(
        self, repo: Repo,
    ) -> None:
        await _add_p_and_m(repo)
        m_before = await repo.sha('task/M')

        result = await repo.git_ops.stack_train_branches(['P', 'M'])

        assert result.survivors == ['P', 'M']
        assert result.pre_stack_tips == (
            PreStackTip(member_id='M', sha=m_before, stacked_sha=await repo.sha('task/M')),
        )

    async def test_member_whose_tip_does_not_resolve_is_ejected_unmoved(
        self, repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _add_p_and_m(repo)
        m_before = await repo.sha('task/M')
        _fail_git(monkeypatch, tmp_path / 'shim', 'rev-parse --verify refs/heads/task/M')

        result = await repo.git_ops.stack_train_branches(['P', 'M'])

        assert (result.survivors, result.ejected, result.pre_stack_tips) == (['P'], ['M'], ())
        assert await repo.sha('task/M') == m_before

    async def test_member_whose_stacked_tip_does_not_resolve_is_ejected_with_its_base(
        self, repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        await _add_p_and_m(repo)
        _fail_git(
            monkeypatch, tmp_path / 'shim', 'rev-parse --verify refs/heads/task/M', after=1,
        )

        result = await repo.git_ops.stack_train_branches(['P', 'M'])

        assert (result.survivors, result.ejected, result.pre_stack_tips) == (['P'], ['M'], ())
        assert await repo.ledger.base_of('task/M') == await repo.sha('task/P')

    async def test_member_ejected_for_conflict_reports_no_tip(self, repo: Repo) -> None:
        await _add_member(repo.git_ops, 'A', 'main', {'foo.txt': 'alpha\n'})
        await _add_member(repo.git_ops, 'B', 'main', {'foo.txt': 'beta\n'})

        result = await repo.git_ops.stack_train_branches(['A', 'B'])

        assert result.ejected == ['B']
        assert result.pre_stack_tips == ()


# ---------------------------------------------------------------------------
# GitOps.restore_pre_stack_tip (task 4953)
# ---------------------------------------------------------------------------


async def _stacked_m(repo: Repo) -> PreStackTip:
    """task/M stacked onto task/P; the tip M had before stacking."""
    await _add_p_and_m(repo)
    result = await repo.git_ops.stack_train_branches(['P', 'M'])
    (tip,) = result.pre_stack_tips
    return tip


@pytest.mark.asyncio
class TestRestorePreStackTip:
    async def test_a_stacked_member_is_put_back_on_its_pre_stack_tip(
        self, repo: Repo,
    ) -> None:
        tip = await _stacked_m(repo)
        stacked = await repo.sha('task/M')

        restore = await repo.git_ops.restore_pre_stack_tip(tip)

        assert restore.outcome is TipRestoreOutcome.RESTORED
        assert (restore.branch, restore.found_sha, restore.pre_stack_sha) == (
            'task/M', stacked, tip.sha,
        )
        assert await repo.sha('task/M') == tip.sha
        m_wt = repo.wt('M')
        assert await _rev_parse(m_wt, 'HEAD') == tip.sha
        assert (await _git(m_wt, 'status', '--porcelain')).strip() == ''
        assert not (m_wt / 'p_only.txt').exists()
        assert (m_wt / 'shared.txt').read_text() == 'v0\n'
        assert await repo.ledger.base_of('task/M') is None

    async def test_a_member_already_on_its_pre_stack_tip_is_unchanged(
        self, repo: Repo,
    ) -> None:
        await _add_p_and_m(repo)
        m_tip = await repo.sha('task/M')
        tip = PreStackTip(member_id='M', sha=m_tip, stacked_sha=m_tip)

        restore = await repo.git_ops.restore_pre_stack_tip(tip)

        assert restore.outcome is TipRestoreOutcome.UNCHANGED
        assert await repo.sha('task/M') == tip.sha

    async def test_a_dirty_worktree_is_left_alone(self, repo: Repo) -> None:
        tip = await _stacked_m(repo)
        stacked = await repo.sha('task/M')
        (repo.wt('M') / 'm.txt').write_text('uncommitted edit\n')

        restore = await repo.git_ops.restore_pre_stack_tip(tip)

        assert restore.outcome is TipRestoreOutcome.DIRTY
        assert await repo.sha('task/M') == stacked
        assert (repo.wt('M') / 'm.txt').read_text() == 'uncommitted edit\n'

    async def test_a_stacked_tip_already_on_main_is_left_alone(self, repo: Repo) -> None:
        tip = await _stacked_m(repo)
        stacked = await repo.sha('task/M')
        await _git(repo.root, 'merge', '--no-ff', '-m', 'land M', 'task/M')

        restore = await repo.git_ops.restore_pre_stack_tip(tip)

        assert restore.outcome is TipRestoreOutcome.LANDED
        assert await repo.sha('task/M') == stacked

    async def test_a_deleted_branch_is_missing(self, repo: Repo) -> None:
        tip = await _stacked_m(repo)
        await _git(repo.root, 'worktree', 'remove', '--force', str(repo.wt('M')))
        await _git(repo.root, 'branch', '-D', 'task/M')

        restore = await repo.git_ops.restore_pre_stack_tip(tip)

        assert restore.outcome is TipRestoreOutcome.MISSING
        assert restore.found_sha is None

    async def test_a_branch_no_worktree_holds_is_restored_by_its_ref(
        self, repo: Repo,
    ) -> None:
        tip = await _stacked_m(repo)
        await _git(repo.root, 'worktree', 'remove', '--force', str(repo.wt('M')))

        restore = await repo.git_ops.restore_pre_stack_tip(tip)

        assert restore.outcome is TipRestoreOutcome.RESTORED
        assert restore.worktree is None
        assert await repo.sha('task/M') == tip.sha

    async def test_a_stacked_tip_the_lane_rebased_onto_main_is_restored(
        self, repo: Repo,
    ) -> None:
        tip = await _stacked_m(repo)
        await repo.advance_main({'other.txt': 'main moved\n'})
        await _git(repo.wt('M'), 'rebase', 'main')

        restore = await repo.git_ops.restore_pre_stack_tip(tip)

        assert restore.outcome is TipRestoreOutcome.RESTORED
        assert await repo.sha('task/M') == tip.sha

    async def test_a_commit_its_train_did_not_stack_is_never_discarded(
        self, repo: Repo,
    ) -> None:
        tip = await _stacked_m(repo)
        await repo.advance_main({'other.txt': 'main moved\n'})
        await _git(repo.wt('M'), 'rebase', 'main')
        agent_commit = await _commit_files(repo.wt('M'), {'agent.txt': 'agent\n'}, 'agent')

        restore = await repo.git_ops.restore_pre_stack_tip(tip)

        assert restore.outcome is TipRestoreOutcome.MOVED
        assert agent_commit in restore.detail
        assert await repo.sha('task/M') == agent_commit
        assert await repo.ledger.base_of('task/M') == await repo.sha('task/P')

    async def test_a_branch_moved_while_restoring_is_left_where_it_is(
        self, repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tip = await _stacked_m(repo)
        stacked = await repo.sha('task/M')
        _fail_git(monkeypatch, tmp_path / 'shim', 'update-ref refs/heads/task/M ')

        restore = await repo.git_ops.restore_pre_stack_tip(tip)

        assert restore.outcome is TipRestoreOutcome.MOVED
        assert await repo.sha('task/M') == stacked
        assert await _rev_parse(repo.wt('M'), 'HEAD') == stacked
        assert await repo.ledger.base_of('task/M') == await repo.sha('task/P')

    async def test_a_worktree_deleted_under_the_restore_leaves_the_branch(
        self, repo: Repo,
    ) -> None:
        tip = await _stacked_m(repo)
        stacked = await repo.sha('task/M')
        shutil.rmtree(repo.wt('M'))

        restore = await repo.git_ops.restore_pre_stack_tip(tip)

        assert restore.outcome is TipRestoreOutcome.MOVED
        assert await repo.sha('task/M') == stacked

    async def test_a_failed_worktree_reset_puts_the_branch_back_on_its_stacked_tip(
        self, repo: Repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tip = await _stacked_m(repo)
        stacked = await repo.sha('task/M')
        _fail_git(monkeypatch, tmp_path / 'shim', 'reset --hard')

        restore = await repo.git_ops.restore_pre_stack_tip(tip)

        assert restore.outcome is TipRestoreOutcome.RESET_FAILED
        assert restore.worktree == repo.wt('M')
        assert await repo.sha('task/M') == stacked
        assert await _rev_parse(repo.wt('M'), 'HEAD') == stacked
        assert (await _git(repo.wt('M'), 'status', '--porcelain')).strip() == ''
        assert await repo.ledger.base_of('task/M') == await repo.sha('task/P')


class TestUnstackResultInvariant:
    def test_unstacked_without_base_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            UnstackResult(
                outcome=UnstackOutcome.UNSTACKED, branch='task/X', main_branch='main',
            )

    def test_conflict_without_a_conflicted_path_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            UnstackResult(
                outcome=UnstackOutcome.CONFLICT,
                branch='task/X',
                main_branch='main',
                base='a' * 40,
                cut='b' * 40,
            )

    def test_not_stacked_has_no_block_reason(self) -> None:
        result = UnstackResult(
            outcome=UnstackOutcome.NOT_STACKED, branch='task/X', main_branch='main',
        )

        with pytest.raises(ValueError):
            result.merge_block_reason()
