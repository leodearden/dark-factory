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

from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest
import pytest_asyncio
from _git_fault_shim import fail_git

from orchestrator.branch_stack import (
    StackBaseLedger,
    StackBaseRead,
    StackInspectionError,
    StackRecordState,
    StackTarget,
    base_owners,
    foreign_commit_cut,
    rebase_own_delta,
)
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.ref_ledger import GitRunner

# ---------------------------------------------------------------------------
# Helpers (copied locally from test_git_ops_train_solo.py, per-file convention)
# ---------------------------------------------------------------------------


async def _init_repo(path: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=path)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=path)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=path)


async def _commit_file(path: Path, name: str, content: str) -> str:
    """Write and commit *name* only; a bare ``git add -A`` in the main
    checkout would stage the nested ``.worktrees/*`` checkouts as gitlinks."""
    (path / name).write_text(content)
    await _run(['git', 'add', '--', name], cwd=path)
    await _run(['git', 'commit', '-m', f'write {name}'], cwd=path)
    return await _rev_parse(path, 'HEAD')


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

    async def test_prune_orphans_drops_only_records_of_deleted_branches(
        self, stacked: Stacked,
    ) -> None:
        ledger = StackBaseLedger(stacked.repo, _run)
        await ledger.record('task/M', stacked.p_tip)
        await ledger.record('task/P', stacked.p_first)
        await _git(stacked.repo, 'worktree', 'remove', '--force', str(stacked.m_wt))
        await _git(stacked.repo, 'branch', '-D', 'task/M')

        assert await ledger.prune_orphans() == ('task/M',)

        assert await ledger.base_of('task/M') is None
        assert await ledger.base_of('task/P') == stacked.p_first

    async def test_move_branch_rekeys_only_that_branchs_record(
        self, stacked: Stacked,
    ) -> None:
        ledger = StackBaseLedger(stacked.repo, _run)
        await ledger.record('task/M', stacked.p_tip)
        await ledger.record('task/M0', stacked.p_first)

        await ledger.move_branch('task/M', 'task/M-20261117T000000Z')

        assert await ledger.base_of('task/M') is None
        assert await ledger.base_of('task/M-20261117T000000Z') == stacked.p_tip
        assert await ledger.base_of('task/M0') == stacked.p_first

    async def test_prune_orphans_with_no_records_drops_nothing(
        self, stacked: Stacked,
    ) -> None:
        assert await StackBaseLedger(stacked.repo, _run).prune_orphans() == ()

    async def test_record_survives_base_branch_deletion(self, stacked: Stacked) -> None:
        ledger = StackBaseLedger(stacked.repo, _run)
        await ledger.record('task/M', stacked.p_tip)

        await _git(stacked.repo, 'worktree', 'remove', '--force', str(stacked.p_wt))
        await _git(stacked.repo, 'branch', '-D', 'task/P')

        assert await ledger.base_of('task/M') == stacked.p_tip
        object_type = await _git(stacked.repo, 'cat-file', '-t', stacked.p_tip)
        assert object_type.strip() == 'commit'


# ---------------------------------------------------------------------------
# StackBaseLedger.read / ensure_record, StackBaseRead, StackTarget (task 10473)
# ---------------------------------------------------------------------------


def _runner_failing(*prefix: str, seen: list[list[str]] | None = None) -> GitRunner:
    """The real runner, except that a git call whose arguments start with
    *prefix* answers rc 128; every call is appended to *seen* when given."""

    async def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
        if seen is not None:
            seen.append(list(cmd))
        if prefix and tuple(cmd[1:1 + len(prefix)]) == prefix:
            return 128, '', 'fatal: injected failure'
        return await _run(cmd, cwd=cwd)

    return run


@pytest.mark.asyncio
class TestStackBaseRead:
    async def test_a_branch_with_no_record_reads_absent(self, stacked: Stacked) -> None:
        read = await StackBaseLedger(stacked.repo, _run).read('task/M')

        assert read.state is StackRecordState.ABSENT
        assert read.base is None

    async def test_a_recorded_branch_reads_its_base(self, stacked: Stacked) -> None:
        ledger = StackBaseLedger(stacked.repo, _run)
        await ledger.record('task/M', stacked.p_tip)

        read = await ledger.read('task/M')

        assert read.state is StackRecordState.RECORDED
        assert read.base == stacked.p_tip
        assert read.names(stacked.p_tip)
        assert not read.names(stacked.p_first)

    async def test_a_ledger_outside_any_repository_is_unreadable(
        self, tmp_path: Path,
    ) -> None:
        outside = tmp_path / 'not-a-repo'
        outside.mkdir()
        ledger = StackBaseLedger(outside, _run)

        read = await ledger.read('task/M')

        assert read.state is StackRecordState.UNREADABLE
        assert read.base is None
        assert read.detail
        assert await ledger.base_of('task/M') is None

    async def test_a_failing_ref_listing_is_unreadable_not_absent(
        self, stacked: Stacked,
    ) -> None:
        await StackBaseLedger(stacked.repo, _run).record('task/M', stacked.p_tip)
        ledger = StackBaseLedger(stacked.repo, _runner_failing('for-each-ref'))

        read = await ledger.read('task/M')

        assert read.state is StackRecordState.UNREADABLE
        assert read.base is None

    async def test_base_of_answers_none_for_absent_and_the_base_once_recorded(
        self, stacked: Stacked,
    ) -> None:
        ledger = StackBaseLedger(stacked.repo, _run)
        assert await ledger.base_of('task/M') is None

        await ledger.record('task/M', stacked.p_tip)

        assert await ledger.base_of('task/M') == stacked.p_tip

    async def test_ensure_record_writes_an_absent_record(self, stacked: Stacked) -> None:
        ledger = StackBaseLedger(stacked.repo, _run)

        read = await ledger.ensure_record('task/M', stacked.p_tip)

        assert read == StackBaseRead(StackRecordState.RECORDED, stacked.p_tip)
        assert await ledger.base_of('task/M') == stacked.p_tip

    async def test_ensure_record_replaces_a_record_naming_another_base(
        self, stacked: Stacked,
    ) -> None:
        ledger = StackBaseLedger(stacked.repo, _run)
        await ledger.record('task/M', stacked.p_first)

        read = await ledger.ensure_record('task/M', stacked.p_tip)

        assert read.names(stacked.p_tip)

    async def test_ensure_record_writes_nothing_when_the_record_names_the_base(
        self, stacked: Stacked,
    ) -> None:
        await StackBaseLedger(stacked.repo, _run).record('task/M', stacked.p_tip)
        seen: list[list[str]] = []
        ledger = StackBaseLedger(stacked.repo, _runner_failing(seen=seen))

        read = await ledger.ensure_record('task/M', stacked.p_tip)

        assert read.state is StackRecordState.RECORDED
        assert read.base == stacked.p_tip
        assert not [cmd for cmd in seen if 'update-ref' in cmd]

    async def test_ensure_record_whose_write_fails_reads_back_without_the_base(
        self, stacked: Stacked,
    ) -> None:
        ledger = StackBaseLedger(stacked.repo, _runner_failing('update-ref'))

        read = await ledger.ensure_record('task/M', stacked.p_tip)

        assert not read.names(stacked.p_tip)
        assert read.state is StackRecordState.ABSENT


class TestStackBaseReadInvariants:
    def test_recorded_without_a_base_is_rejected(self) -> None:
        with pytest.raises(ValueError) as raised:
            StackBaseRead(StackRecordState.RECORDED)

        assert 'recorded' in str(raised.value)
        assert 'None' in str(raised.value)

    @pytest.mark.parametrize('state', [StackRecordState.ABSENT, StackRecordState.UNREADABLE])
    def test_a_base_without_a_record_is_rejected(self, state: StackRecordState) -> None:
        with pytest.raises(ValueError) as raised:
            StackBaseRead(state, 'a' * 40)

        assert str(state) in str(raised.value)
        assert 'a' * 40 in str(raised.value)


class TestStackTarget:
    def test_holds_a_branch_and_a_full_sha(self) -> None:
        target = StackTarget('task/P', 'a' * 40)

        assert (target.branch, target.commit) == ('task/P', 'a' * 40)

    def test_an_empty_branch_is_rejected(self) -> None:
        with pytest.raises(ValueError) as raised:
            StackTarget('', 'a' * 40)

        assert "''" in str(raised.value)

    @pytest.mark.parametrize('commit', ['abc123', 'main', 'g' * 40, 'A' * 40, ''])
    def test_a_commit_that_is_not_a_full_hex_sha_is_rejected(self, commit: str) -> None:
        with pytest.raises(ValueError) as raised:
            StackTarget('task/P', commit)

        assert repr(commit) in str(raised.value)

    @pytest.mark.parametrize('field_name', ['branch', 'commit'])
    def test_is_frozen(self, field_name: str) -> None:
        target = StackTarget('task/P', 'a' * 40)

        with pytest.raises(FrozenInstanceError):
            setattr(target, field_name, 'b' * 40)


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


# ---------------------------------------------------------------------------
# rebase_own_delta reports whether its abort failed (task 7540, via 10473)
# ---------------------------------------------------------------------------

#: The argv prefix of guarded_abort's ``rebase --abort`` (RECOVERY_GIT's flags).
_GUARDED_REBASE_ABORT = '-c rerere.enabled=false rebase --abort'


async def _conflicting_own_commit(s: Stacked) -> tuple[Path, str, str]:
    """task/X from main rewrites base.txt; main then rewrites it differently.

    Returns X's worktree, X's fork point and main's new tip."""
    x_wt = await _add_branch(s.git_ops, 'X', 'main')
    fork = await _rev_parse(s.repo, 'main')
    await _commit_file(x_wt, 'base.txt', 'x\n')
    onto = await _commit_file(s.repo, 'base.txt', 'main moved\n')
    return x_wt, fork, onto


async def _rebase_merge_dir(wt: Path) -> Path:
    path = Path((await _git(wt, 'rev-parse', '--git-path', 'rebase-merge')).strip())
    return path if path.is_absolute() else wt / path


@pytest.mark.asyncio
class TestRebaseOwnDeltaReportsItsAbort:
    async def test_a_conflict_whose_abort_held_is_not_an_abort_failure(
        self, stacked: Stacked,
    ) -> None:
        x_wt, fork, onto = await _conflicting_own_commit(stacked)

        result = await rebase_own_delta(_run, x_wt, onto=onto, cut=fork)

        assert not result.ok
        assert result.conflicted_paths == ('base.txt',)
        assert result.abort_failed is False
        assert (await _git(x_wt, 'status', '--porcelain')).strip() == ''
        assert not (await _rebase_merge_dir(x_wt)).exists()

    async def test_a_failed_abort_is_reported(
        self, stacked: Stacked, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        x_wt, fork, onto = await _conflicting_own_commit(stacked)
        fail_git(monkeypatch, tmp_path / 'shim' / 'bin', f'{_GUARDED_REBASE_ABORT}*')

        result = await rebase_own_delta(_run, x_wt, onto=onto, cut=fork)

        assert not result.ok
        assert result.abort_failed is True
        assert (await _rebase_merge_dir(x_wt)).exists()
        await _git(x_wt, 'rebase', '--abort')

    async def test_a_rebase_git_refused_to_start_is_not_an_abort_failure(
        self, stacked: Stacked,
    ) -> None:
        x_wt, fork, onto = await _conflicting_own_commit(stacked)
        (x_wt / 'base.txt').write_text('dirty\n')

        result = await rebase_own_delta(_run, x_wt, onto=onto, cut=fork)

        assert not result.ok
        assert result.conflicted_paths == ()
        assert result.abort_failed is False
        assert (x_wt / 'base.txt').read_text() == 'dirty\n'
