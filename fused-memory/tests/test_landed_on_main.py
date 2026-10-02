"""Behaviour of ``services/landed_on_main.py::probe_landing`` against REAL git.

Every fixture is a tmp repository built with plain git commands: real
``--no-ff`` merge markers, real cherry-picked twins (cherry-pick keeps the
author date, author email and subject, and mints a new sha), and real
committer dates pinned through ``GIT_AUTHOR_DATE``/``GIT_COMMITTER_DATE``. The
verdict IS git semantics, so a canned ``run_git`` fake would only pin our
assumptions about git. The one wrapper in this file (the hoist test) delegates
to the real ``run_git`` and only counts calls.
"""

from __future__ import annotations

import logging
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from _fm_helpers import _init_git_repo

from fused_memory.services import landed_on_main as landed_module
from fused_memory.services.landed_on_main import (
    LandingEvidence,
    LandingQuery,
    LandingVerdict,
    probe_landing,
)

#: A fixed, whole-second instant every dated commit is placed around.
T0 = datetime(2026, 9, 1, 12, 0, 0, tzinfo=UTC)


def _git(repo: Path, *args: str, when: datetime | None = None) -> str:
    env = dict(os.environ)
    if when is not None:
        stamp = f'@{int(when.timestamp())} +0000'
        env['GIT_AUTHOR_DATE'] = stamp
        env['GIT_COMMITTER_DATE'] = stamp
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def _committer_only(repo: Path, *args: str, when: datetime) -> str:
    """Run *args* with only the COMMITTER date pinned (cherry-pick keeps the author's)."""
    env = dict(os.environ)
    env['GIT_COMMITTER_DATE'] = f'@{int(when.timestamp())} +0000'
    return subprocess.run(
        ['git', '-C', str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def _commit(repo: Path, name: str, subject: str, *, when: datetime) -> str:
    (repo / name).write_text(f'{name}\n')
    _git(repo, 'add', name)
    _git(repo, 'commit', '-q', '-m', subject, when=when)
    return _git(repo, 'rev-parse', 'HEAD')


def _branch_with_commits(repo: Path, task_id: str, count: int, *, when: datetime) -> list[str]:
    """Create ``task/<id>`` at main's tip carrying *count* own commits; return their shas."""
    _git(repo, 'checkout', '-q', '-b', f'task/{task_id}', 'main')
    shas = [
        _commit(
            repo,
            f'{task_id}-{n}.txt',
            f'task {task_id} change {n}',
            when=when + timedelta(minutes=n),
        )
        for n in range(count)
    ]
    _git(repo, 'checkout', '-q', 'main')
    return shas


def _merge_and_delete(repo: Path, task_id: str, *, when: datetime) -> str:
    branch = f'task/{task_id}'
    _git(repo, 'merge', '-q', '--no-ff', '-m', f'Merge {branch} into main', branch, when=when)
    merge_sha = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'branch', '-q', '-D', branch)
    return merge_sha


def _cherry_pick_onto_main(repo: Path, shas: list[str], *, when: datetime) -> None:
    for sha in shas:
        _committer_only(repo, 'cherry-pick', sha, when=when)


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # Stop git discovering any repository above tmp_path (the non-git case
    # must be genuinely non-git wherever pytest puts its tmp tree).
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
    root = tmp_path / 'repo'
    root.mkdir()
    _init_git_repo(root)
    return root


async def _probe_one(repo: Path, task_id: str, reopened_at: datetime | None = None) -> LandingVerdict:
    verdicts = await probe_landing(str(repo), [LandingQuery(task_id, reopened_at=reopened_at)])
    return verdicts[task_id]


@pytest.mark.asyncio
async def test_deleted_branch_with_merge_marker_is_landed(repo: Path) -> None:
    _branch_with_commits(repo, '101', 1, when=T0)
    merge_sha = _merge_and_delete(repo, '101', when=T0 + timedelta(hours=1))

    verdict = await _probe_one(repo, '101')

    assert verdict == LandingVerdict(
        landed=True, evidence=LandingEvidence.MERGE_MARKER, commit=merge_sha,
    ), 'a deleted branch whose merge marker is on main is the acceptance landed shape'


@pytest.mark.asyncio
async def test_branch_with_commits_off_main_is_not_landed(repo: Path) -> None:
    _branch_with_commits(repo, '102', 1, when=T0)

    verdict = await _probe_one(repo, '102')

    assert verdict.landed is False, 'unmerged own commits with no marker are genuine WIP off main'
    assert verdict.evidence is None


@pytest.mark.asyncio
async def test_rebased_twins_on_main_are_landed(repo: Path) -> None:
    shas = _branch_with_commits(repo, '103', 2, when=T0)
    _cherry_pick_onto_main(repo, shas, when=T0 + timedelta(hours=2))

    verdict = await _probe_one(repo, '103')

    assert verdict.landed is True, (
        'the task-3838 shape: every own commit was rebased onto main under a new sha'
    )
    assert verdict.evidence is LandingEvidence.REBASED_TWINS


@pytest.mark.asyncio
async def test_partially_twinned_branch_is_not_landed(repo: Path) -> None:
    shas = _branch_with_commits(repo, '104', 2, when=T0)
    _cherry_pick_onto_main(repo, shas[:1], when=T0 + timedelta(hours=2))

    verdict = await _probe_one(repo, '104')

    assert verdict.landed is False, 'one own commit has no twin on main, so work is still off main'


@pytest.mark.asyncio
async def test_bare_branch_without_marker_is_not_landed(repo: Path) -> None:
    _git(repo, 'branch', 'task/105', 'main')

    verdict = await _probe_one(repo, '105')

    assert verdict.landed is False, 'a branch with zero own commits and no marker never landed anything'


@pytest.mark.asyncio
async def test_never_dispatched_task_is_not_landed(repo: Path) -> None:
    verdict = await _probe_one(repo, '106')

    assert verdict.landed is False, 'no branch and no marker is no evidence of landing'


@pytest.mark.asyncio
async def test_merge_marker_older_than_reopen_is_stale(repo: Path) -> None:
    _branch_with_commits(repo, '107', 1, when=T0 - timedelta(hours=1))
    _merge_and_delete(repo, '107', when=T0)

    after_reopen = await _probe_one(repo, '107', reopened_at=T0 + timedelta(hours=1))
    before_reopen = await _probe_one(repo, '107', reopened_at=T0 - timedelta(hours=1))

    assert after_reopen.landed is False, (
        'a marker committed before the task was reopened is the previous run, so the reopen stays actionable'
    )
    assert before_reopen.landed is True, 'a marker committed after the reopen is fresh evidence'


@pytest.mark.asyncio
async def test_rebased_twin_older_than_reopen_is_stale(repo: Path) -> None:
    shas = _branch_with_commits(repo, '108', 2, when=T0 - timedelta(hours=3))
    _cherry_pick_onto_main(repo, shas, when=T0)

    after_reopen = await _probe_one(repo, '108', reopened_at=T0 + timedelta(hours=1))
    before_reopen = await _probe_one(repo, '108', reopened_at=T0 - timedelta(hours=1))

    assert after_reopen.landed is False, 'twins committed before the reopen do not land the reopened work'
    assert before_reopen.landed is True, 'twins committed after the reopen are fresh evidence'
    assert before_reopen.evidence is LandingEvidence.REBASED_TWINS


def _branch_holding_a_fix_from_main(repo: Path, task_id: str, *, behind_the_fork: bool) -> None:
    """``task/<id>`` whose only own commit is a cherry-pick of a commit on main.

    The fix lands on main either before the branch forks (so the pick is
    redundant) or after it (the usual pull-in of a fix the task needs).
    """
    if behind_the_fork:
        fix = _commit(repo, 'fix.txt', 'fix a shared bug', when=T0)
        _git(repo, 'checkout', '-q', '-b', f'task/{task_id}', 'main')
    else:
        _git(repo, 'branch', f'task/{task_id}', 'main')
        fix = _commit(repo, 'fix.txt', 'fix a shared bug', when=T0)
        _git(repo, 'checkout', '-q', f'task/{task_id}')
    _committer_only(
        repo, 'cherry-pick', '--keep-redundant-commits', fix, when=T0 + timedelta(hours=1),
    )
    _git(repo, 'checkout', '-q', 'main')


@pytest.mark.asyncio
@pytest.mark.parametrize('behind_the_fork', [False, True], ids=['after-the-fork', 'behind-the-fork'])
async def test_a_branch_holding_only_a_fix_from_main_is_not_landed(
    repo: Path, behind_the_fork: bool,
) -> None:
    _branch_holding_a_fix_from_main(repo, '110', behind_the_fork=behind_the_fork)

    verdict = await _probe_one(repo, '110')

    assert verdict.landed is False, (
        "main holds the original the branch copied, not a copy of the branch's own work"
    )


@pytest.mark.asyncio
async def test_a_fix_pulled_from_main_does_not_hide_work_that_landed(repo: Path) -> None:
    _branch_holding_a_fix_from_main(repo, '111', behind_the_fork=False)
    _git(repo, 'checkout', '-q', 'task/111')
    work = _commit(repo, '111.txt', 'task 111 work', when=T0 + timedelta(hours=2))
    _git(repo, 'checkout', '-q', 'main')
    _cherry_pick_onto_main(repo, [work], when=T0 + timedelta(hours=3))

    verdict = await _probe_one(repo, '111')

    assert verdict.landed is True, 'the task work itself was rebased onto main'
    assert verdict.evidence is LandingEvidence.REBASED_TWINS


@pytest.mark.asyncio
async def test_marker_for_task_10_does_not_land_task_1(repo: Path) -> None:
    _branch_with_commits(repo, '10', 1, when=T0)
    _merge_and_delete(repo, '10', when=T0 + timedelta(hours=1))

    verdicts = await probe_landing(str(repo), [LandingQuery('1'), LandingQuery('10')])

    assert verdicts['1'].landed is False, "'Merge task/10 into main' must not be read as task 1's marker"
    assert verdicts['10'].landed is True


@pytest.mark.asyncio
async def test_marker_text_in_a_commit_body_is_not_a_marker(repo: Path) -> None:
    (repo / 'revert.txt').write_text('revert\n')
    _git(repo, 'add', 'revert.txt')
    _git(
        repo, 'commit', '-q',
        '-m', 'Revert "Merge task/109 into main"',
        '-m', 'Merge task/109 into main',
        when=T0,
    )

    verdict = await _probe_one(repo, '109')

    assert verdict.landed is False, 'only a commit whose SUBJECT is the merge subject is a marker'


@pytest.mark.asyncio
async def test_non_git_root_is_unknown_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
    root = tmp_path / 'plain'
    root.mkdir()

    with caplog.at_level(logging.WARNING, logger=landed_module.__name__):
        verdicts = await probe_landing(str(root), [LandingQuery('1'), LandingQuery('2')])

    assert {task_id: v.landed for task_id, v in verdicts.items()} == {'1': None, '2': None}, (
        'a failed probe is unknown, never "not landed"'
    )
    warnings = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.WARNING and r.name == landed_module.__name__
    ]
    assert any(
        str(root) in message and ('branch listing' in message or 'merge-marker scan' in message)
        for message in warnings
    ), f'expected a WARNING naming the failed probe and {root}; got {warnings!r}'


@pytest.mark.asyncio
@pytest.mark.parametrize('query_count', [1, 5])
async def test_shared_scans_run_once_per_probe_call(
    repo: Path, monkeypatch: pytest.MonkeyPatch, query_count: int,
) -> None:
    task_ids = [str(300 + n) for n in range(query_count)]
    for task_id in task_ids:
        shas = _branch_with_commits(repo, task_id, 1, when=T0)
        _cherry_pick_onto_main(repo, shas, when=T0 + timedelta(hours=1))

    calls: list[tuple[str, ...]] = []
    real_run_git = landed_module.run_git

    async def counting_run_git(cmd, *args, **kwargs):
        calls.append(tuple(cmd))
        return await real_run_git(cmd, *args, **kwargs)

    monkeypatch.setattr(landed_module, 'run_git', counting_run_git)

    verdicts = await probe_landing(str(repo), [LandingQuery(t) for t in task_ids])

    assert all(verdicts[t].landed is True for t in task_ids)
    branch_listings = [c for c in calls if 'for-each-ref' in c]
    marker_scans = [c for c in calls if any(a.startswith('--grep') for a in c)]
    twin_scans = [c for c in calls if any(a.startswith('--since') for a in c)]
    assert len(branch_listings) == 1, f'branch listing must be hoisted; got {branch_listings!r}'
    assert len(marker_scans) == 1, f'marker scan must be hoisted; got {marker_scans!r}'
    assert len(twin_scans) <= 1, f'main twin scan must be hoisted; got {twin_scans!r}'
