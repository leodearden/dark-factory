"""Real-git helpers the merge screen's test modules share.

The fixtures resolve ``repo`` from the requesting module, so each module
seeds its own repository. A consumer imports a fixture by redundant alias
(``from _screen_fixtures import git_ops as git_ops``), because pytest finds
fixtures only in the requesting module's namespace and its conftests.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from _orch_helpers import git_env_with_ceiling

from orchestrator.config import GitConfig
from orchestrator.git_ops import GitOps, ScreenLane
from orchestrator.verify_cancel import lane_lock_path


def head(cwd: Path) -> str:
    """The sha HEAD names in the checkout at *cwd*."""
    return subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=cwd, env=git_env_with_ceiling(cwd),
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def lane_git_config() -> GitConfig:
    return GitConfig(
        main_branch='main', branch_prefix='task/', remote='origin',
        worktree_dir='.worktrees', push_after_advance=False,
    )


@pytest.fixture
def git_ops(repo: Path) -> GitOps:
    return GitOps(lane_git_config(), repo)


@pytest.fixture
def held_merger_lane(git_ops: GitOps) -> Iterator[Path]:
    """Hold the ``_screen-merger`` lane's flock, as warm-lane-gc would; yields the lock path."""
    lock = lane_lock_path(git_ops.worktree_base / ScreenLane.MERGER.value)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock
    finally:
        os.close(fd)
