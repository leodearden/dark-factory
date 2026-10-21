"""A throwaway git repository under tmp_path, with explicit commit times and no dependence on host git config."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from df_pytest_isolation import git_redirect_env

_IDENTITY = ('-c', 'user.name=t', '-c', 'user.email=t@example.invalid', '-c', 'commit.gpgsign=false')


def _env(**extra: str) -> dict[str, str]:
    redirecting = set(git_redirect_env(os.environ))
    env = {key: value for key, value in os.environ.items() if key not in redirecting}
    return {**env, 'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1', **extra}


def _dated(when: datetime) -> dict[str, str]:
    stamp = when.isoformat()
    return {'GIT_AUTHOR_DATE': stamp, 'GIT_COMMITTER_DATE': stamp}


@dataclass(frozen=True)
class Repo:
    root: Path

    def git(self, *args: str, env: dict[str, str] | None = None) -> str:
        completed = subprocess.run(
            ['git', *args],
            cwd=self.root,
            env=env or _env(),
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout

    def write(self, relpath: str, content: bytes | str) -> Path:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            content = content.encode()
        path.write_bytes(content)
        return path

    def symlink(self, relpath: str, target: str) -> None:
        path = self.root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target)

    def gitlink(self, relpath: str, sha: str) -> None:
        """A submodule entry pointing at `sha`, staged directly; its empty directory keeps `add -A` from dropping it."""
        (self.root / relpath).mkdir(parents=True, exist_ok=True)
        self.git('update-index', '--add', '--cacheinfo', f'160000,{sha},{relpath}')

    def commit(self, message: str, *, when: datetime) -> str:
        self.git('add', '-A')
        self.git(*_IDENTITY, 'commit', '-q', '--no-verify', '-m', message, env=_env(**_dated(when)))
        return self.rev('HEAD')

    def branch(self, name: str, start: str = 'main') -> None:
        self.git('branch', name, start)

    def checkout(self, name: str) -> None:
        self.git('checkout', '-q', name)

    def merge_no_ff(self, branch: str, message: str, *, when: datetime) -> str:
        self.git(
            *_IDENTITY, 'merge', '-q', '--no-ff', '--no-verify', '-m', message, branch,
            env=_env(**_dated(when)),
        )
        return self.rev('HEAD')

    def rev(self, ref: str) -> str:
        return self.git('rev-parse', '--verify', ref).strip()


def make_repo(path: Path) -> Repo:
    path.mkdir(parents=True, exist_ok=True)
    repo = Repo(path)
    repo.git('init', '-q', '-b', 'main')
    return repo
