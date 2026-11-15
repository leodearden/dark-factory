"""restamp_degenerate_branch_bases: which pending tasks get their branch base re-stamped (task 7332).

Real temp git repositories throughout. Branches are moved the way the
orchestrator moves them: ``git branch -f`` onto a later main commit, with the
recorded base left at the creation point.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import restamp_degenerate_branch_bases as restamp
from committed_git_repo import committed_repo
from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, GitQueryFailed
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)


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


def test_an_uncompilable_citation_pattern_is_a_git_error_not_eligible(repo: Path):
    creation_point = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'branch', 'task/101', commit_on(repo, 'other.txt', 'feat(900): unrelated'))
    broken = GitOps(GitConfig(push_after_advance=False, commit_citation_pattern='('), repo)

    verdict = judge(broken, 101, {'branch_base_sha': creation_point})

    assert verdict.exclusion is restamp.Exclusion.GIT_ERROR


def test_a_failed_marker_scan_is_a_git_error_not_eligible(
    repo: Path, git_ops: GitOps, monkeypatch: pytest.MonkeyPatch,
):
    creation_point = git(repo, 'rev-parse', 'HEAD')
    git(repo, 'branch', 'task/101', commit_on(repo, 'other.txt', 'feat(900): unrelated'))

    async def unreadable(branch: str) -> str | None:
        raise GitQueryFailed(f'scan for {branch} failed')

    monkeypatch.setattr(git_ops, 'find_merge_marker_checked', unreadable)

    verdict = judge(git_ops, 101, {'branch_base_sha': creation_point})

    assert verdict.exclusion is restamp.Exclusion.GIT_ERROR


class FakeFusedMemory(restamp.RestampClient):
    """An in-memory task store behind the two tools the script calls."""

    def __init__(self, tasks: dict[int, dict], *, update_reply: dict | None = None):
        super().__init__('http://unused')
        self.tasks = tasks
        self.update_reply = update_reply
        self.calls: list[tuple[str, dict]] = []

    async def __aenter__(self) -> FakeFusedMemory:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def call_tool(self, name: str, arguments: dict, *, timeout: float | None = None) -> dict:
        self.calls.append((name, arguments))
        task = self.tasks[int(arguments['id'])]
        if name == 'update_task':
            if self.update_reply is not None:
                return self.update_reply
            task['metadata'] = {**task['metadata'], **json.loads(arguments['metadata'])}
        return {'id': int(arguments['id']), **task}


def _moved_branch(repo: Path) -> tuple[str, str]:
    creation_point = git(repo, 'rev-parse', 'HEAD')
    later_main = commit_on(repo, 'other.txt', 'feat(900): unrelated work')
    git(repo, 'branch', 'task/101', later_main)
    return creation_point, later_main


def _apply(client: FakeFusedMemory, git_ops: GitOps, repo: Path, pending: dict) -> int:
    verdicts = asyncio.run(restamp.census(git_ops, pending.items()))
    eligible = [v for v in verdicts if v.eligible]
    return asyncio.run(restamp.apply_restamps(client, git_ops, repo, eligible))


def test_apply_writes_the_tip_in_merge_mode_and_confirms_it(repo: Path, git_ops: GitOps):
    creation_point, later_main = _moved_branch(repo)
    client = FakeFusedMemory({101: {'status': 'pending', 'metadata': {'branch_base_sha': creation_point}}})

    failures = _apply(client, git_ops, repo, {101: {'branch_base_sha': creation_point}})

    assert failures == 0
    assert [name for name, _ in client.calls] == ['get_task', 'update_task', 'get_task']
    update = client.calls[1][1]
    assert update['id'] == '101' and update['metadata_mode'] == 'merge'
    assert client.tasks[101]['metadata']['branch_base_sha'] == later_main


@pytest.mark.parametrize(
    'live',
    [
        {'status': 'in-progress', 'metadata': 'census'},
        {'status': 'pending', 'metadata': 'moved'},
    ],
    ids=['no-longer-pending', 'base-changed'],
)
def test_apply_skips_a_task_touched_since_the_census(repo: Path, git_ops: GitOps, live: dict):
    creation_point, _ = _moved_branch(repo)
    live_base = creation_point if live['metadata'] == 'census' else 'f' * 40
    client = FakeFusedMemory({101: {'status': live['status'], 'metadata': {'branch_base_sha': live_base}}})

    failures = _apply(client, git_ops, repo, {101: {'branch_base_sha': creation_point}})

    assert failures == 0
    assert [name for name, _ in client.calls] == ['get_task']


def test_apply_counts_an_empty_reply_as_a_failure(repo: Path, git_ops: GitOps):
    creation_point, _ = _moved_branch(repo)
    client = FakeFusedMemory(
        {101: {'status': 'pending', 'metadata': {'branch_base_sha': creation_point}}},
        update_reply={},
    )

    assert _apply(client, git_ops, repo, {101: {'branch_base_sha': creation_point}}) == 1


# ── main_async: the default never dials MCP; exit codes 0 / 1 / 2 ──────────


@pytest.fixture
def project(repo: Path, make_tasks_db) -> Path:
    creation_point, _ = _moved_branch(repo)
    (repo / '.taskmaster' / 'tasks').mkdir(parents=True)
    make_tasks_db(
        [{'id': 101, 'status': 'pending', 'metadata': {'branch_base_sha': creation_point}}],
        directory=repo / '.taskmaster' / 'tasks',
    )
    (repo / restamp.CONFIG_FILENAME).write_text(f'project_root: {repo}\n')
    return repo


def _args(root: Path, *, apply: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        project_root=str(root), config=None, server_url='http://unused', check=not apply, apply=apply,
    )


def test_check_reports_without_dialling_mcp(
    project: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
):
    def no_client(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail('--check must never construct an MCP client')

    monkeypatch.setattr(restamp, 'RestampClient', no_client)

    assert asyncio.run(restamp.main_async(_args(project))) == 0
    assert 'eligible ids: 101' in capsys.readouterr().out


def test_an_unreadable_task_store_exits_2(repo: Path):
    (repo / restamp.CONFIG_FILENAME).write_text(f'project_root: {repo}\n')

    assert asyncio.run(restamp.main_async(_args(repo))) == 2


def test_a_failed_write_under_apply_exits_1(project: Path, monkeypatch: pytest.MonkeyPatch):
    creation_point = git(project, 'rev-parse', 'HEAD~1')
    refusing = FakeFusedMemory(
        {101: {'status': 'pending', 'metadata': {'branch_base_sha': creation_point}}},
        update_reply={'error': 'refused', 'error_type': 'Test'},
    )
    monkeypatch.setattr(restamp, 'RestampClient', lambda _url: refusing)

    assert asyncio.run(restamp.main_async(_args(project, apply=True))) == 1
