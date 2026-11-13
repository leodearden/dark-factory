"""Real throwaway git repos for the polarity-lint tests, because the 2x2 is a measurement of a tree."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ['git', '-C', str(root), *args], check=True, capture_output=True, text=True
    )


def init_git_repo(root: Path, files: dict[str, str]) -> Path:
    """Initialize a real repo at *root* on branch main, seeded with *files*
    and committed, so ``HEAD`` resolves.

    ``HEAD`` — not a historical SHA — is the whole point: at AUTHORING time
    the task has not been implemented yet, so HEAD *is* the pre-task tree
    (design decision #2). These fixtures therefore stand in for "the repo as
    the author sees it at ``commit_planning`` time".
    """
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['git', 'init', '-b', 'main', str(root)], check=True, capture_output=True, text=True
    )
    run_git(root, 'config', 'user.email', 'polarity-test@example.com')
    run_git(root, 'config', 'user.name', 'Polarity Test')
    run_git(root, 'config', 'commit.gpgsign', 'false')
    for rel, body in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding='utf-8')
    run_git(root, 'add', '-A')
    run_git(root, 'commit', '--no-verify', '-m', 'seed the authoring tree')
    return root


def commit_all(root: Path, message: str) -> None:
    run_git(root, 'add', '-A')
    run_git(root, 'commit', '--no-verify', '-q', '-m', message)
