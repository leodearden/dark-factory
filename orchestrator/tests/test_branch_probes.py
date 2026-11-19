"""orchestrator.mcp.branch_probes against real git repos.

Only a timeout and a missing git binary are faked, at the stdlib boundary:
neither can be produced cheaply from a real fixture.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from _git_fixtures import seed_repo
from orchestrator.mcp.branch_probes import (
    Reachability,
    resolve_main_sha,
    sha_branch_reachability,
    sha_exists_on_branch,
)


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def _commit_file(repo: Path, name: str, content: str, message: str) -> str:
    (repo / name).write_text(content)
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-m', message)
    return _git(repo, 'rev-parse', 'HEAD')


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    return seed_repo(tmp_path / 'repo')


@pytest.fixture()
def side_branch_sha(repo: Path) -> str:
    """A real commit object that lives only on a side branch, with main checked out."""
    _git(repo, 'switch', '-c', 'side')
    sha = _commit_file(repo, 'side.py', 'side\n', 'feat: on a side branch')
    _git(repo, 'switch', 'main')
    return sha


@pytest.fixture()
def no_repo(tmp_path: Path) -> Path:
    outside = tmp_path / 'outside'
    outside.mkdir()
    return outside


FABRICATED_SHA = 'deadbeef' * 5


class TestShaBranchReachability:
    def test_head_is_reachable(self, repo):
        assert sha_branch_reachability(repo, _git(repo, 'rev-parse', 'HEAD')) is (
            Reachability.REACHABLE
        )

    def test_a_side_branch_commit_is_unreachable(self, repo, side_branch_sha):
        assert sha_branch_reachability(repo, side_branch_sha) is Reachability.UNREACHABLE

    def test_a_fabricated_sha_is_unreachable(self, repo):
        assert sha_branch_reachability(repo, FABRICATED_SHA) is Reachability.UNREACHABLE

    def test_a_directory_in_no_repository_is_unknown(self, no_repo):
        assert sha_branch_reachability(no_repo, FABRICATED_SHA) is Reachability.UNKNOWN

    @pytest.mark.parametrize('failing_probe', ['rev-parse', 'merge-base'])
    @pytest.mark.parametrize(
        'exc',
        [subprocess.TimeoutExpired(cmd=['git'], timeout=10), OSError('no git binary')],
        ids=['timeout', 'oserror'],
    )
    def test_either_probe_blowing_up_is_unknown(
        self, repo, monkeypatch, failing_probe, exc,
    ):
        """An infra fault in either probe is never a verdict about the sha."""
        head = _git(repo, 'rev-parse', 'HEAD')
        real_run = subprocess.run

        def _fake_run(argv, **kwargs):
            if failing_probe in argv:
                raise exc
            return real_run(argv, **kwargs)

        monkeypatch.setattr(subprocess, 'run', _fake_run)

        assert sha_branch_reachability(repo, head) is Reachability.UNKNOWN


class TestShaExistsOnBranch:
    def test_head_exists_on_the_branch(self, repo):
        assert sha_exists_on_branch(repo, _git(repo, 'rev-parse', 'HEAD')) is True

    def test_a_side_branch_commit_does_not(self, repo, side_branch_sha):
        assert sha_exists_on_branch(repo, side_branch_sha) is False

    def test_no_repository_fails_closed(self, no_repo):
        assert sha_exists_on_branch(no_repo, FABRICATED_SHA) is False


class TestResolveMainSha:
    def test_returns_main_head(self, repo):
        assert resolve_main_sha(repo) == _git(repo, 'rev-parse', 'main')

    def test_outside_any_repository_is_empty(self, no_repo):
        assert resolve_main_sha(no_repo) == ''
