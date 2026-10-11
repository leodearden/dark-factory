"""Readers for the LME eval's committed artifacts: the repo's git index and each arm's run dir.

Contract tests over those artifacts read files plus ``git`` only, so they run in the merge
lane's default selection; ``git`` skips the calling test where no git working tree exists.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[3]


def git(*args: str) -> subprocess.CompletedProcess[str]:
    if shutil.which('git') is None:
        pytest.skip('git is not available; cannot check the committed index')
    inside = subprocess.run(
        ['git', 'rev-parse', '--is-inside-work-tree'],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    if inside.returncode != 0:
        pytest.skip('not a git working tree; cannot check the committed index')
    return subprocess.run(
        ['git', *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )


def committed_run_dir(evidence: Path, arm_id: str) -> Path:
    runs = evidence / 'runs' / arm_id
    stamps = sorted(path for path in runs.iterdir() if path.is_dir()) if runs.is_dir() else []
    assert len(stamps) == 1, f'{arm_id} must have exactly one committed stamp dir: {stamps}'
    return stamps[0]
