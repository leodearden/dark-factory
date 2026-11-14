"""Git checkouts for the markup-sink suites to resolve a project root from.

Every task runs in a linked worktree that ``git worktree remove`` deletes, so a
record a lane files must land in its MAIN checkout. The plan-tools and
verdict-tools suites both pin that resolution against the checkout built here.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def git(*args: str, cwd: Path) -> None:
    """Run one git command in *cwd*, failing loudly."""
    subprocess.run(
        ['git', *args], cwd=str(cwd), check=True, capture_output=True, text=True,
    )


def main_checkout_with_lane(root: Path) -> tuple[Path, Path]:
    """``(checkout, lane)``: a committed repo under *root* and one linked worktree of it."""
    checkout = root / 'repo'
    checkout.mkdir()
    git('init', cwd=checkout)
    git('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit',
        '--allow-empty', '-m', 'root', cwd=checkout)
    lane = root / 'lane'
    git('worktree', 'add', str(lane), '-b', 'task/1', cwd=checkout)
    return checkout, lane
