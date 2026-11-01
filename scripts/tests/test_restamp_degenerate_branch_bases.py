"""restamp_degenerate_branch_bases: which pending tasks get their branch base re-stamped (task 7332).

Real temp git repositories throughout. Branches are moved the way the
orchestrator moves them: ``git branch -f`` onto a later main commit, with the
recorded base left at the creation point.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import pytest
import restamp_degenerate_branch_bases as restamp
from committed_git_repo import committed_repo
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps


def git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ['git', *args], cwd=cwd, capture_output=True, text=True, timeout=60, check=False,
    )
    assert proc.returncode == 0, f'git {args} failed: {proc.stderr}'
    return proc.stdout.strip()


def commit_on(repo: Path, name: str, msg: str) -> str:
    (repo / name).write_text(f'{name}\n')
    git(repo, 'add', '--', name)
    git(repo, 'commit', '-m', msg)
    return git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / 'repo'
    committed_repo(root, {'seed.txt': 'seed\n'})
    git(root, 'config', 'user.email', 'test@test.com')
    git(root, 'config', 'user.name', 'Test')
    git(root, 'config', 'commit.gpgsign', 'false')
    return root


@pytest.fixture
def git_ops(repo: Path) -> GitOps:
    return GitOps(GitConfig(push_after_advance=False), repo)


def judge(git_ops: GitOps, task_id: int, metadata: dict[str, Any]) -> restamp.BranchVerdict:
    return asyncio.run(restamp.judge_branch(git_ops, task_id, metadata))


def test_a_zero_commit_branch_moved_onto_a_later_main_is_eligible(repo: Path, git_ops: GitOps):
    creation_point = git(repo, 'rev-parse', 'HEAD')
    later_main = commit_on(repo, 'other.txt', 'feat(900): unrelated work')
    git(repo, 'branch', 'task/101', later_main)

    verdict = judge(git_ops, 101, {'branch_base_sha': creation_point})

    assert verdict.eligible
    assert verdict.tip == later_main


def test_a_branch_with_no_recorded_base_on_main_is_eligible(repo: Path, git_ops: GitOps):
    git(repo, 'branch', 'task/101')

    assert judge(git_ops, 101, {}).eligible


def test_a_missing_branch_is_excluded(git_ops: GitOps):
    assert judge(git_ops, 101, {}).exclusion is restamp.Exclusion.BRANCH_ABSENT


def test_a_branch_already_at_its_recorded_base_is_excluded(repo: Path, git_ops: GitOps):
    git(repo, 'branch', 'task/101')

    verdict = judge(git_ops, 101, {'branch_base_sha': git(repo, 'rev-parse', 'HEAD')})

    assert verdict.exclusion is restamp.Exclusion.ALREADY_DEGENERATE


def test_a_branch_with_a_commit_of_its_own_is_excluded(repo: Path, git_ops: GitOps):
    base = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'checkout', '-q', '-b', 'task/101')
    commit_on(repo, 'work.txt', 'impl(101): work')
    git(repo, 'checkout', '-q', 'main')

    verdict = judge(git_ops, 101, {'branch_base_sha': base})

    assert verdict.exclusion is restamp.Exclusion.COMMITS_BEYOND_MAIN


def test_a_branch_merged_under_a_marker_is_excluded(repo: Path, git_ops: GitOps):
    base = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'checkout', '-q', '-b', 'task/101')
    commit_on(repo, 'work.txt', 'work without a citation')
    git(repo, 'checkout', '-q', 'main')
    git(repo, 'merge', '--no-ff', 'task/101', '-m', 'Merge task/101 into main')

    verdict = judge(git_ops, 101, {'branch_base_sha': base})

    assert verdict.exclusion is restamp.Exclusion.MERGE_MARKER


def test_a_branch_whose_task_is_cited_on_main_is_excluded(repo: Path, git_ops: GitOps):
    base = git(repo, 'rev-parse', 'HEAD')
    commit_on(repo, 'work.txt', 'impl(101): landed by hand')
    git(repo, 'branch', 'task/101')

    verdict = judge(git_ops, 101, {'branch_base_sha': base})

    assert verdict.exclusion is restamp.Exclusion.CITATION


def test_the_report_counts_every_reason_and_lists_the_eligible_ids(repo: Path, git_ops: GitOps):
    creation_point = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'branch', 'task/102')
    later_main = commit_on(repo, 'other.txt', 'feat(900): unrelated work')
    git(repo, 'branch', 'task/101', later_main)
    pending = [
        (101, {'branch_base_sha': creation_point}),
        (102, {'branch_base_sha': creation_point}),
        (103, {}),
    ]

    verdicts = asyncio.run(restamp.census(git_ops, pending))
    report = restamp.render_report(repo, later_main, verdicts)

    assert 'pending scanned: 3' in report
    assert 'with a task branch: 2' in report
    assert 'eligible: 1' in report
    assert 'already_degenerate: 1' in report
    assert 'branch_absent: 1' in report
    assert 'eligible ids: 101' in report


class FakeFusedMemory(restamp.RestampClient):
    """Answers update_task with *update_reply* and get_task from what was written."""

    def __init__(self, update_reply: dict | None = None):
        super().__init__('http://unused')
        self.update_reply = update_reply
        self.stored: dict[str, str] = {}
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict, *, timeout: float | None = None) -> dict:
        self.calls.append((name, arguments))
        if name == 'update_task':
            if self.update_reply is not None:
                return self.update_reply
            self.stored[arguments['id']] = arguments['metadata']
            return {'id': int(arguments['id'])}
        return {'id': int(arguments['id']), 'metadata': self.stored.get(arguments['id'], '{}')}


def _moved_branch(repo: Path) -> tuple[str, str]:
    creation_point = git(repo, 'rev-parse', 'HEAD')
    later_main = commit_on(repo, 'other.txt', 'feat(900): unrelated work')
    git(repo, 'branch', 'task/101', later_main)
    return creation_point, later_main


def test_apply_writes_the_tip_in_merge_mode_and_confirms_it(repo: Path, git_ops: GitOps):
    creation_point, later_main = _moved_branch(repo)
    pending = {101: {'branch_base_sha': creation_point}}
    verdicts = asyncio.run(restamp.census(git_ops, pending.items()))
    client = FakeFusedMemory()

    failures = asyncio.run(restamp.apply_restamps(client, git_ops, repo, pending, verdicts))

    assert failures == 0
    name, arguments = client.calls[0]
    assert name == 'update_task'
    assert arguments['id'] == '101' and arguments['metadata_mode'] == 'merge'
    assert client.stored['101'] == f'{{"branch_base_sha": "{later_main}"}}'
    assert client.calls[1][0] == 'get_task'


def test_apply_counts_an_empty_reply_as_a_failure(repo: Path, git_ops: GitOps):
    creation_point, _ = _moved_branch(repo)
    pending = {101: {'branch_base_sha': creation_point}}
    verdicts = asyncio.run(restamp.census(git_ops, pending.items()))

    failures = asyncio.run(restamp.apply_restamps(
        FakeFusedMemory(update_reply={}), git_ops, repo, pending, verdicts,
    ))

    assert failures == 1
