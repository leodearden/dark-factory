"""replay_deep_stack_study indexes exactly the landing merges the merge writer emits."""
from __future__ import annotations

import subprocess
from pathlib import Path

from replay_deep_stack_study import merge_commit_index


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True, capture_output=True)
    _git(repo, 'config', 'user.email', 'replay-study@example.com')
    _git(repo, 'config', 'user.name', 'Replay Study Test')
    _git(repo, 'config', 'commit.gpgsign', 'false')
    _git(repo, 'commit', '-q', '--allow-empty', '--no-verify', '-m', 'seed')
    return repo


def _merge(repo: Path, branch: str, subject: str) -> str:
    _git(repo, 'checkout', '-q', '-B', branch, 'main')
    _git(repo, 'commit', '-q', '--allow-empty', '--no-verify', '-m', f'work on {branch}')
    _git(repo, 'checkout', '-q', 'main')
    _git(repo, 'merge', '-q', '--no-ff', '--no-verify', '-m', subject, branch)
    return _git(repo, 'rev-parse', 'HEAD')


def _commit(repo: Path, *messages: str) -> None:
    flags = [part for message in messages for part in ('-m', message)]
    _git(repo, 'commit', '-q', '--allow-empty', '--no-verify', *flags)


def test_merge_commit_index_holds_only_whole_landing_subjects_oldest_first(tmp_path):
    repo = _repo(tmp_path)
    first_landing = _merge(repo, 'task/5', 'Merge task/5 into main')
    _merge(repo, 'task/5', 'Merge task/5 into main')
    _merge(repo, 'task/7', 'Merge task/7 into main (retry)')
    _commit(repo, 'Revert "Merge task/9 into main"')
    _commit(repo, 'Land eleven', 'Merge task/11 into main')

    assert merge_commit_index(repo) == {'5': first_landing}
