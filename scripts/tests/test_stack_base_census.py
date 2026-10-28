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

import asyncio
import dataclasses
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import stack_base_census as census
from committed_git_repo import committed_repo
from orchestrator.branch_stack import StackBaseLedger, UnstackOutcome
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps
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


async def test_a_sibling_at_the_same_commit_is_not_contained(repo: Path) -> None:
    add_member(repo, 'P', 'main', {'p.txt': 'p\n'})
    p_tip = rev(repo, 'task/P')
    git(repo, 'branch', 'task/C', p_tip)

    findings = findings_of(await census_of(repo))

    assert 'task/P' not in findings
    assert findings['task/C'].contained_siblings == ()
    assert findings['task/C'].verdict is census.Verdict.CONFIRMED
    assert findings['task/C'].base == p_tip

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


# ---------------------------------------------------------------------------
# step-7: adoption
# ---------------------------------------------------------------------------


def stack_base_refs(repo: Path) -> list[str]:
    return git(repo, 'for-each-ref', '--format=%(refname)', 'refs/dark-factory/stack-base/').split()


def make_git_ops(repo: Path) -> GitOps:
    """GitOps as orchestrator/tests/test_git_ops_unstack.py::_make_git_ops builds it."""
    cfg = GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )
    return GitOps(cfg, repo)


async def test_adopting_a_confirmed_finding_lets_5618_unstack_it(repo: Path) -> None:
    stacked_on_p(repo)
    p_tip = rev(repo, 'task/P')
    report = await census_of(repo)

    outcomes = await census.adopt_confirmed(repo, report)

    assert [(o.branch, o.base, o.written) for o in outcomes] == [('task/M', p_tip, True)]
    assert await ledger_of(repo).base_of('task/M') == p_tip

    result = await make_git_ops(repo).unstack_from_unlanded_base('task/M')

    assert result.outcome is UnstackOutcome.UNSTACKED
    assert git(repo, 'diff', '--name-only', 'main...task/M').split() == ['m.txt']


async def test_only_confirmed_findings_are_adopted(repo: Path) -> None:
    add_member(repo, 'P', 'main', {'p.txt': 'p\n'})
    p_tip = rev(repo, 'task/P')
    confirmed_wt = add_member(repo, 'C', 'main', {'c.txt': 'c\n'})
    rebase(confirmed_wt, 'task/P')
    clean_wt = add_member(repo, 'K', 'main', {'k.txt': 'k\n'})
    rebase(clean_wt, 'task/P')
    rebase(clean_wt, '--onto', 'main', p_tip, 'task/K')
    add_member(repo, 'Q', 'main', {'q.txt': 'q\n'})
    stranded_wt = add_member(repo, 'S', 'main', {'s.txt': 's\n'})
    rebase(stranded_wt, 'task/Q')
    git(repo, 'worktree', 'remove', '--force', str(repo / '.worktrees' / 'Q'))
    git(repo, 'branch', '-D', 'task/Q')
    unadjudicated_wt = add_member(repo, 'U', 'main', {'u1.txt': '1\n'})
    u1 = rev(repo, 'task/U')
    commit(unadjudicated_wt, {'u2.txt': '2\n'}, 'U: second')
    git(repo, 'update-ref', 'refs/heads/task/X', u1)
    report = await census_of(repo)
    assert {f.branch: f.verdict for f in report.findings} == {
        'task/C': census.Verdict.CONFIRMED,
        'task/K': census.Verdict.ALREADY_CLEAN,
        'task/S': census.Verdict.STRANDED_BASE,
        'task/U': census.Verdict.UNADJUDICATED,
    }

    outcomes = await census.adopt_confirmed(repo, report)

    assert [(o.branch, o.written) for o in outcomes] == [('task/C', True)]
    assert stack_base_refs(repo) == ['refs/dark-factory/stack-base/task/C']


async def test_a_record_that_does_not_read_back_is_reported_unwritten(repo: Path) -> None:
    stacked_on_p(repo)
    p_tip = rev(repo, 'task/P')
    report = await census_of(repo)
    lock = repo / '.git' / 'refs' / 'dark-factory' / 'stack-base' / 'task' / 'M.lock'
    lock.parent.mkdir(parents=True)
    lock.write_text('')

    outcomes = await census.adopt_confirmed(repo, report)

    assert len(outcomes) == 1
    assert outcomes[0].written is False
    assert 'task/M' in outcomes[0].detail
    assert p_tip in outcomes[0].detail
    assert stack_base_refs(repo) == []


# ---------------------------------------------------------------------------
# step-9: the CLI
# ---------------------------------------------------------------------------


def stack_member(repo: Path, name: str, onto: str) -> Path:
    """task/<name> branches from main, commits, and is rebased onto *onto*."""
    wt = add_member(repo, name, 'main', {f'{name.lower()}.txt': f'{name}\n'})
    rebase(wt, onto)
    return wt


def run_cli(repo: Path, tasks_db: Path, *extra: str) -> int:
    return census.main(['--project-root', str(repo), '--tasks-db', str(tasks_db), *extra])


def test_the_default_run_prints_a_markdown_report_and_writes_nothing(
    repo: Path, make_tasks_db, capsys: pytest.CaptureFixture[str],
) -> None:
    add_member(repo, 'P', 'main', {'p.txt': 'p\n'})
    p_tip = rev(repo, 'task/P')
    stack_member(repo, 'M', 'task/P')
    stack_member(repo, '105', 'task/P')
    u_wt = add_member(repo, 'U', 'main', {'u1.txt': '1\n'})
    git(repo, 'update-ref', 'refs/heads/task/X', rev(repo, 'task/U'))
    commit(u_wt, {'u2.txt': '2\n'}, 'U: second')

    assert run_cli(repo, make_tasks_db([{'id': 105, 'status': 'done'}])) == 0

    out = capsys.readouterr().out
    assert rev(repo, 'main') in out
    assert 'unlanded task branches: 5' in out
    assert '### task/M: confirmed' in out
    assert '### task/U: unadjudicated' in out
    assert f'base: `{p_tip}`' in out
    assert f'cut: `{p_tip}`' in out
    assert f'git rebase --onto main {p_tip} task/M' in out
    assert 'task/X' in out
    assert '| already_clean | 0 |' in out
    assert '| recorded | 0 |' in out
    assert '| inert | 1 |' in out
    assert stack_base_refs(repo) == []


def test_json_output_round_trips_every_finding(
    repo: Path, make_tasks_db, capsys: pytest.CaptureFixture[str],
) -> None:
    add_member(repo, 'P', 'main', {'p.txt': 'p\n'})
    p_tip = rev(repo, 'task/P')
    stack_member(repo, 'M', 'task/P')
    clean_wt = stack_member(repo, 'K', 'task/P')
    rebase(clean_wt, '--onto', 'main', p_tip, 'task/K')
    stack_member(repo, '105', 'task/P')
    tasks_db = make_tasks_db([{'id': 105, 'status': 'done'}])

    assert run_cli(repo, tasks_db, '--json') == 0

    payload = json.loads(capsys.readouterr().out)
    assert {f['branch']: f['verdict'] for f in payload['findings']} == {
        'task/M': 'confirmed',
        'task/K': 'already_clean',
        'task/105': 'inert',
    }
    expected = asyncio.run(census.run_census(
        repo, main_branch='main', statuses=census.load_task_statuses(tasks_db),
    ))
    assert payload['findings'] == [
        json.loads(json.dumps(dataclasses.asdict(finding))) for finding in expected.findings
    ]
    assert payload['main_sha'] == expected.main_sha
    assert payload['unlanded_branches'] == expected.unlanded_branches


def adoption_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith('adopt ')]


def test_adopt_confirmed_exits_zero_when_every_record_reads_back(
    repo: Path, make_tasks_db, capsys: pytest.CaptureFixture[str],
) -> None:
    stacked_on_p(repo)
    p_tip = rev(repo, 'task/P')

    assert run_cli(repo, make_tasks_db([]), '--adopt-confirmed') == 0

    lines = adoption_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert lines[0].startswith(f'adopt task/M at {p_tip}: written')
    assert stack_base_refs(repo) == ['refs/dark-factory/stack-base/task/M']


def test_adopt_confirmed_exits_one_when_a_record_does_not(
    repo: Path, make_tasks_db, capsys: pytest.CaptureFixture[str],
) -> None:
    stacked_on_p(repo)
    p_tip = rev(repo, 'task/P')
    lock = repo / '.git' / 'refs' / 'dark-factory' / 'stack-base' / 'task' / 'M.lock'
    lock.parent.mkdir(parents=True)
    lock.write_text('')

    assert run_cli(repo, make_tasks_db([]), '--adopt-confirmed') == 1

    lines = adoption_lines(capsys.readouterr().out)
    assert len(lines) == 1
    assert lines[0].startswith(f'adopt task/M at {p_tip}: NOT WRITTEN')
    assert stack_base_refs(repo) == []


def test_a_missing_tasks_db_warns_and_treats_every_branch_as_live(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    stacked_on_p(repo, member='105')
    missing = tmp_path / 'absent' / 'tasks.db'

    assert run_cli(repo, missing, '--json') == 0

    captured = capsys.readouterr()
    assert str(missing) in captured.err
    payload = json.loads(captured.out)
    assert {f['branch']: f['verdict'] for f in payload['findings']} == {'task/105': 'confirmed'}


def test_a_project_root_that_is_not_a_repo_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    not_a_repo = tmp_path / 'plain'
    not_a_repo.mkdir()

    assert run_cli(not_a_repo, tmp_path / 'tasks.db') == 2

    assert 'git rev-parse' in capsys.readouterr().err
