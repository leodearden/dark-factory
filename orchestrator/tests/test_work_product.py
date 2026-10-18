"""work_product: the one-git-child task-branch probe and the continuity predicate
(task 3820; plans/scheduler-dispatch-scoring-and-lock-layer-prd.md C3).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _git_fixtures import README_SEED, _git, build_repo

from orchestrator.work_product import (
    BranchProbeError,
    BranchWork,
    parse_branch_work,
    read_branch_work,
)

MAIN_TIP_AT = 1_760_000_000
TASK_TIP_AT = 1_760_003_600


def _committer_date(monkeypatch: pytest.MonkeyPatch, epoch: int) -> None:
    monkeypatch.setenv('GIT_COMMITTER_DATE', f'{epoch} +0000')


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _committer_date(monkeypatch, MAIN_TIP_AT)
    return build_repo(tmp_path / 'repo', README_SEED)


def _branch_with_commits(
    repo: Path, monkeypatch: pytest.MonkeyPatch, branch: str, *, commits: int, at: int = TASK_TIP_AT,
) -> None:
    _committer_date(monkeypatch, at)
    _git(repo, 'checkout', '-q', '-b', branch, 'main')
    for n in range(commits):
        _git(repo, 'commit', '-q', '--allow-empty', '-m', f'{branch} work {n}')
    _git(repo, 'checkout', '-q', 'main')


def _read(project_root: Path, **kwargs) -> dict[str, BranchWork]:
    return read_branch_work(project_root, main_branch='main', branch_prefix='task/', **kwargs)


class TestReadBranchWork:
    def test_branches_are_keyed_by_task_id_with_commits_ahead_and_tip_time(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/7', commits=2)
        _git(repo, 'branch', 'task/8', 'main')

        assert _read(repo) == {
            '7': BranchWork(commits_ahead=2, tip_committed_at=TASK_TIP_AT),
            '8': BranchWork(commits_ahead=0, tip_committed_at=MAIN_TIP_AT),
        }

    def test_a_same_named_tag_does_not_change_the_keys(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/7', commits=1)
        _git(repo, 'tag', 'task/7', 'main')

        assert set(_read(repo)) == {'7'}

    def test_suffixed_and_nested_branches_keep_their_whole_suffix(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/9-retry', commits=1)
        _branch_with_commits(repo, monkeypatch, 'task/nested/4', commits=1)

        assert set(_read(repo)) == {'9-retry', 'nested/4'}

    def test_a_repo_without_task_branches_reads_as_empty(self, repo):
        assert _read(repo) == {}

    def test_a_missing_main_branch_is_a_probe_error_quoting_git(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/7', commits=1)

        with pytest.raises(BranchProbeError) as excinfo:
            read_branch_work(repo, main_branch='trunk', branch_prefix='task/')

        assert excinfo.value.returncode == 128
        rendered = str(excinfo.value)
        assert 'failed to find' in rendered
        assert 'for-each-ref' in rendered

    def test_a_subdirectory_of_a_repo_is_not_read_as_that_repo(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/7', commits=1)
        subdir = repo / 'sub'
        subdir.mkdir()

        with pytest.raises(BranchProbeError):
            _read(subdir)

    def test_one_read_forks_exactly_one_ceilinged_for_each_ref(self, repo, monkeypatch):
        _branch_with_commits(repo, monkeypatch, 'task/7', commits=1)
        calls: list[tuple[tuple, dict]] = []

        def counting_run(*args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.run(*args, **kwargs)

        _read(repo, run=counting_run)

        assert len(calls) == 1
        args, kwargs = calls[0]
        argv = list(args[0] if args else kwargs['args'])
        assert argv[0] == 'git'
        assert 'for-each-ref' in argv
        assert any('%(ahead-behind:main)' in arg for arg in argv)
        assert 'refs/heads/task/' in argv
        assert kwargs['env']['GIT_CEILING_DIRECTORIES'] == str(repo.resolve().parent)

    def test_a_timed_out_git_is_a_probe_error(self, repo):
        def hung_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0] if args else kwargs['args'], timeout=60.0)

        with pytest.raises(BranchProbeError, match='timed out'):
            _read(repo, run=hung_run)

    def test_a_missing_git_binary_is_a_probe_error(self, repo):
        def no_git(*args, **kwargs):
            raise FileNotFoundError(2, 'No such file or directory', 'git')

        with pytest.raises(BranchProbeError):
            _read(repo, run=no_git)


class TestParseBranchWork:
    def test_blank_lines_are_ignored(self):
        stdout = (
            'refs/heads/task/7 2 0 1760003600\n'
            '\n'
            '   \n'
            'refs/heads/task/8 0 3 1760000000\n'
        )

        assert parse_branch_work(stdout, branch_prefix='task/') == {
            '7': BranchWork(commits_ahead=2, tip_committed_at=1760003600),
            '8': BranchWork(commits_ahead=0, tip_committed_at=1760000000),
        }

    @pytest.mark.parametrize(
        'line',
        [
            'refs/heads/task/7 x 0 1760003600',
            'refs/heads/task/7 2 0',
            'refs/heads/task/7 2 0 1760003600 extra',
            'refs/heads/feature/7 2 0 1760003600',
            'refs/tags/task/7 2 0 1760003600',
        ],
    )
    def test_an_unparseable_line_is_a_probe_error_naming_it(self, line):
        with pytest.raises(BranchProbeError) as excinfo:
            parse_branch_work(f'refs/heads/task/1 1 0 1760000000\n{line}\n', branch_prefix='task/')

        assert line in str(excinfo.value)
