"""Real git checkouts left in the states a project_root hygiene probe reads.

Both state builders edit the ``README.md`` that
``_git_fixtures.py::README_SEED`` commits, so they act on a checkout seeded by
``_git_fixtures.py::seed_repo`` (or a linked worktree of one).

Imported by bare module name (``from _checkout_hygiene_fixtures import ...``),
matching the flat-test-helper convention of ``_orch_helpers.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _orch_helpers import git_env_with_ceiling
from df_pytest_isolation import git_redirect_env


def run_git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run git in *cwd* under the test ceiling, with every repo-redirecting variable dropped."""
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    return subprocess.run(
        ['git', *args], cwd=cwd, env=env, check=check, capture_output=True, text=True,
    )


def leave_unmerged(checkout: Path, side: str) -> None:
    """Leave README.md unmerged (UU) in *checkout*'s index via a real conflicting merge.

    Commits README.md on a new branch *side* and on the branch *checkout* had
    checked out, then merges *side* into the latter and asserts the merge stopped.
    """
    run_git(checkout, 'checkout', '-q', '-b', side)
    (checkout / 'README.md').write_text(f'# {side}\n')
    run_git(checkout, 'commit', '-qam', side)
    run_git(checkout, 'checkout', '-q', '-')
    (checkout / 'README.md').write_text('# ours\n')
    run_git(checkout, 'commit', '-qam', 'ours')
    merge = run_git(checkout, 'merge', side, check=False)
    assert merge.returncode != 0, merge.stdout


def dirty_tracked_file(checkout: Path) -> None:
    """Leave tracked, uncommitted WIP in *checkout*: README.md modified, nothing staged."""
    (checkout / 'README.md').write_text('# uncommitted WIP\n')
