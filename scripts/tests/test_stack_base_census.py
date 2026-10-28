"""stack_base_census: task branches stacked on an unlanded sibling before task
5618's stack-base record existed (task 6125).

Real temp git repositories throughout.  Members live at
``<repo>/.worktrees/<name>`` on branch ``task/<name>``, the layout
orchestrator/tests/test_git_ops_unstack.py::_add_member uses, so GitOps finds
the worktree that holds a branch.  Every rebase runs inside the member's own
worktree, so the branch reflog records ``rebase (finish): refs/heads/task/<name>
onto <sha>`` exactly as the orchestrator's stacker leaves it.
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import stack_base_census as census
from committed_git_repo import committed_repo
from orchestrator.branch_stack import StackBaseLedger
from shared.git_async import run_git

# ---------------------------------------------------------------------------
# Real-git fixture (sync helpers, local to this file)
# ---------------------------------------------------------------------------


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ['git', *args], cwd=cwd, capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, f'git {args} failed: {proc.stderr}'
    return proc.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo on ``main`` with one seed commit and a repo-local identity."""
    root = tmp_path / 'repo'
    committed_repo(root, {'seed.txt': 'seed\n'})
    git(root, 'config', 'user.email', 'test@test.com')
    git(root, 'config', 'user.name', 'Test')
    git(root, 'config', 'commit.gpgsign', 'false')
    with (root / '.git' / 'info' / 'exclude').open('a') as exclude:
        exclude.write('.worktrees/\n')
    return root


def commit(wt: Path, files: Mapping[str, str], msg: str) -> str:
    """Write and commit exactly *files* in *wt*; return the new HEAD."""
    for rel, text in files.items():
        (wt / rel).write_text(text)
    git(wt, 'add', '--', *files)
    git(wt, 'commit', '-m', msg)
    return rev(wt, 'HEAD')


def add_member(repo: Path, name: str, start_ref: str, files: Mapping[str, str]) -> Path:
    """Worktree ``.worktrees/<name>`` on new branch ``task/<name>`` from *start_ref*,
    with one commit of *files*."""
    wt = repo / '.worktrees' / name
    git(repo, 'worktree', 'add', '-b', f'task/{name}', str(wt), start_ref)
    commit(wt, files, f'{name}: write files')
    return wt


def rebase(wt: Path, *args: str) -> None:
    git(wt, 'rebase', *args)


def rev(repo: Path, ref: str) -> str:
    return git(repo, 'rev-parse', '--verify', f'{ref}^{{commit}}')


def advance_main(repo: Path, files: Mapping[str, str]) -> str:
    """Commit *files* on main in the root checkout; return main's new tip."""
    return commit(repo, files, 'main: advance')


async def census_of(repo: Path, statuses: Mapping[int, str] | None = None):
    return await census.run_census(repo, main_branch='main', statuses=statuses or {})


def findings_of(report) -> dict[str, census.BranchFinding]:
    return {finding.branch: finding for finding in report.findings}


async def _runner(cmd, cwd=None, **_kwargs) -> tuple[int, str, str]:
    result = await run_git(cmd, cwd)
    return result.returncode, result.stdout, result.stderr


def ledger_of(repo: Path) -> StackBaseLedger:
    """Task 5618's ledger over *repo*, through its public constructor."""
    return StackBaseLedger(repo, _runner)


def stacked_on_p(repo: Path, member: str = 'M') -> tuple[Path, Path]:
    """task/P and task/<member> branch from main; <member> is rebased onto P."""
    p_wt = add_member(repo, 'P', 'main', {'p.txt': 'p\n'})
    m_wt = add_member(repo, member, 'main', {f'{member.lower()}.txt': 'm\n'})
    rebase(m_wt, 'task/P')
    return p_wt, m_wt


# ---------------------------------------------------------------------------
# step-1: the core census
# ---------------------------------------------------------------------------


async def test_a_member_rebased_onto_an_unlanded_sibling_is_confirmed(repo: Path) -> None:
    stacked_on_p(repo)
    p_tip = rev(repo, 'task/P')

    report = await census_of(repo)

    finding = findings_of(report)['task/M']
    assert finding.verdict is census.Verdict.CONFIRMED
    assert finding.base == p_tip
    assert finding.contained_siblings == ('task/P',)
    assert 'task/P' in finding.base_owners
    assert finding.cut == p_tip
    assert report.main_sha == rev(repo, 'main')
    assert report.unlanded_branches == 2


async def test_a_member_created_on_a_siblings_tip_is_confirmed(repo: Path) -> None:
    add_member(repo, 'P', 'main', {'p.txt': 'p\n'})
    p_tip = rev(repo, 'task/P')
    add_member(repo, 'C', p_tip, {'c.txt': 'c\n'})
    assert git(repo, 'reflog', 'show', '--format=%gs', 'refs/heads/task/C').splitlines()[-1] == (
        f'branch: Created from {p_tip}'
    )

    report = await census_of(repo)

    finding = findings_of(report)['task/C']
    assert finding.verdict is census.Verdict.CONFIRMED
    assert finding.base == p_tip
    assert finding.contained_siblings == ('task/P',)
    assert report.unlanded_branches == 2


async def test_a_member_only_ever_rebased_onto_main_has_no_finding(repo: Path) -> None:
    add_member(repo, 'P', 'main', {'p.txt': 'p\n'})
    m_wt = add_member(repo, 'M', 'main', {'m.txt': 'm\n'})
    advance_main(repo, {'unrelated.txt': 'u\n'})
    rebase(m_wt, 'main')

    report = await census_of(repo)

    assert 'task/M' not in findings_of(report)
    assert report.findings == ()
    assert report.unlanded_branches == 2


async def test_a_member_whose_base_later_landed_has_no_finding(repo: Path) -> None:
    stacked_on_p(repo)
    git(repo, 'merge', '--ff-only', 'task/P')

    report = await census_of(repo)

    assert 'task/M' not in findings_of(report)
    assert report.findings == ()
    assert report.unlanded_branches == 1


async def test_landed_branches_are_not_scanned(repo: Path) -> None:
    add_member(repo, 'L', 'main', {'l.txt': 'l\n'})
    git(repo, 'merge', '--ff-only', 'task/L')
    add_member(repo, 'U', 'main', {'u.txt': 'u\n'})

    report = await census_of(repo)

    assert 'task/L' not in findings_of(report)
    assert report.findings == ()
    assert report.unlanded_branches == 1
    assert report.main_sha == rev(repo, 'main')


# ---------------------------------------------------------------------------
# step-3: adjudication edges
# ---------------------------------------------------------------------------


async def test_a_stack_replayed_onto_main_is_confirmed_from_the_reflog_alone(repo: Path) -> None:
    _, m_wt = stacked_on_p(repo)
    p_tip = rev(repo, 'task/P')
    advance_main(repo, {'unrelated.txt': 'u\n'})
    rebase(m_wt, 'main')
    own = {rev(repo, 'task/M')}
    replayed = [sha for sha in git(repo, 'rev-list', 'main..task/M').split() if sha not in own]

    finding = findings_of(await census_of(repo))['task/M']

    assert finding.contained_siblings == ()
    assert finding.verdict is census.Verdict.CONFIRMED
    assert finding.base == p_tip
    assert finding.cut == replayed[0]


async def test_a_member_unstacked_by_hand_is_already_clean(repo: Path) -> None:
    _, m_wt = stacked_on_p(repo)
    rebase(m_wt, '--onto', 'main', rev(repo, 'task/P'), 'task/M')

    finding = findings_of(await census_of(repo))['task/M']

    assert finding.verdict is census.Verdict.ALREADY_CLEAN
    assert finding.cut is None


async def test_a_rebase_onto_the_branchs_own_history_is_not_a_stacking_event(
    repo: Path,
) -> None:
    m_wt = add_member(repo, 'M', 'main', {'m1.txt': '1\n'})
    m1 = rev(repo, 'task/M')
    m2 = commit(m_wt, {'m2.txt': '2\n'}, 'M: second')
    commit(m_wt, {'m3.txt': '3\n'}, 'M: third')
    rebase(m_wt, '--onto', m1, m2)

    report = await census_of(repo)

    assert 'task/M' not in findings_of(report)


async def test_a_base_whose_patches_landed_on_main_has_no_finding(repo: Path) -> None:
    stacked_on_p(repo)
    advance_main(repo, {'unrelated.txt': 'u\n'})  # a new parent, so the pick is a new SHA
    git(repo, 'cherry-pick', rev(repo, 'task/P'))

    report = await census_of(repo)

    assert report.findings == ()
    assert report.unlanded_branches == 2


async def test_a_base_with_no_live_carrier_is_stranded(repo: Path) -> None:
    stacked_on_p(repo)
    p_tip = rev(repo, 'task/P')
    git(repo, 'worktree', 'remove', '--force', str(repo / '.worktrees' / 'P'))
    git(repo, 'branch', '-D', 'task/P')

    finding = findings_of(await census_of(repo))['task/M']

    assert finding.verdict is census.Verdict.STRANDED_BASE
    assert finding.base == p_tip
    assert finding.base_owners == ()


async def test_a_contained_sibling_without_a_reflog_stacking_event_is_unadjudicated(
    repo: Path,
) -> None:
    m_wt = add_member(repo, 'M', 'main', {'m1.txt': '1\n'})
    m1 = rev(repo, 'task/M')
    commit(m_wt, {'m2.txt': '2\n'}, 'M: second')
    git(repo, 'branch', 'task/S', m1)

    finding = findings_of(await census_of(repo))['task/M']

    assert finding.verdict is census.Verdict.UNADJUDICATED
    assert finding.contained_siblings == ('task/S',)
    assert 'task/S' in finding.detail


async def test_a_branch_with_a_stack_base_record_is_recorded(repo: Path) -> None:
    stacked_on_p(repo)
    p_tip = rev(repo, 'task/P')
    await ledger_of(repo).record('task/M', p_tip)

    finding = findings_of(await census_of(repo))['task/M']

    assert finding.verdict is census.Verdict.RECORDED
    assert finding.base == p_tip


async def test_a_residual_older_base_is_unadjudicated(repo: Path) -> None:
    add_member(repo, 'P1', 'main', {'p1.txt': 'p1\n'})
    m_wt = add_member(repo, 'M', 'main', {'m.txt': 'm\n'})
    rebase(m_wt, 'task/P1')
    advance_main(repo, {'unrelated.txt': 'u\n'})
    rebase(m_wt, 'main')
    add_member(repo, 'P2', 'main', {'p2.txt': 'p2\n'})
    rebase(m_wt, 'task/P2')
    p1_tip = rev(repo, 'task/P1')

    finding = findings_of(await census_of(repo))['task/M']

    assert finding.verdict is census.Verdict.UNADJUDICATED
    assert finding.base == rev(repo, 'task/P2')
    assert p1_tip in finding.detail


# ---------------------------------------------------------------------------
# step-5: task-status gating and the store reader
# ---------------------------------------------------------------------------


async def test_a_stacked_branch_of_a_terminal_task_is_inert(repo: Path) -> None:
    stacked_on_p(repo, member='101')

    finding = findings_of(await census_of(repo, statuses={101: 'done'}))['task/101']

    assert finding.verdict is census.Verdict.INERT
    assert finding.cut is None
    assert finding.base_owners == ()
    assert finding.task_status == 'done'
    assert finding.base == rev(repo, 'task/P')


async def test_a_branch_without_a_task_row_is_adjudicated_as_live(repo: Path) -> None:
    stacked_on_p(repo, member='101')

    finding = findings_of(await census_of(repo, statuses={}))['task/101']

    assert finding.verdict is census.Verdict.CONFIRMED
    assert finding.task_status is None


async def test_a_live_task_status_is_carried_on_its_finding(repo: Path) -> None:
    stacked_on_p(repo, member='101')

    finding = findings_of(await census_of(repo, statuses={101: 'pending'}))['task/101']

    assert finding.verdict is census.Verdict.CONFIRMED
    assert finding.task_status == 'pending'


def test_load_task_statuses_reads_the_master_tag_read_only(make_tasks_db) -> None:
    db_path = make_tasks_db([
        {'id': 7, 'status': 'pending'},
        {'id': 8, 'status': 'done'},
        {'id': 9, 'tag': 'feature', 'status': 'cancelled'},
    ])
    before = db_path.read_bytes()

    assert census.load_task_statuses(db_path) == {7: 'pending', 8: 'done'}
    assert db_path.read_bytes() == before


async def test_a_non_numeric_task_branch_has_no_task_id(repo: Path) -> None:
    stacked_on_p(repo, member='foo')
    add_member(repo, '102', rev(repo, 'task/P'), {'102.txt': '102\n'})

    findings = findings_of(await census_of(repo, statuses={102: 'cancelled'}))

    assert findings['task/foo'].task_status is None
    assert findings['task/foo'].verdict is census.Verdict.CONFIRMED
    assert findings['task/102'].verdict is census.Verdict.INERT
