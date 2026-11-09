"""The fixture project a task-role verify test runs against: a seeded repo and its spec.

The spec's module and task file name what the seed commits, so the two live together.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from _git_fixtures import RepoSeed, seed_repo

from orchestrator.task_verify_spec import TaskVerifySpec
from orchestrator.verify_spec_shared import VerifyCommand

VERIFY_MODULE = 'mod'
VERIFY_TASK_FILE = f'{VERIFY_MODULE}/test_x.py'
VERIFY_REPO_SEED = RepoSeed(
    files=((VERIFY_TASK_FILE, '# placeholder\n'),), message='initial commit',
)


def seed_verify_repo(tmp_path: Path) -> tuple[Path, str]:
    """Seed ``tmp_path/repo`` with :data:`VERIFY_REPO_SEED`; return it and its HEAD sha."""
    repo = seed_repo(tmp_path / 'repo', VERIFY_REPO_SEED)
    head_sha = subprocess.run(
        ['git', '-C', str(repo), 'rev-parse', 'HEAD'],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return repo, head_sha


def seed_task_branch_repo(
    tmp_path: Path, *, modules: tuple[str, ...] = (VERIFY_MODULE,),
) -> tuple[Path, str, str]:
    """Seed main with ``<module>/test_x.py`` per module and a ``.task/``-ignoring
    ``.gitignore``, then check out branch ``task`` one commit ahead touching every
    module file; return the repo, main's sha and the task tip.
    """
    files = tuple((f'{module}/test_x.py', '# placeholder\n') for module in modules)
    repo = seed_repo(
        tmp_path / 'repo',
        RepoSeed(files=(*files, ('.gitignore', '.task/\n')), message='initial commit'),
    )
    base = _git(repo, 'rev-parse', 'HEAD')
    _git(repo, 'checkout', '-q', '-b', 'task')
    for path, _text in files:
        (repo / path).write_text('# changed on the task branch\n')
    _git(repo, 'commit', '-q', '-am', 'task change')
    return repo, base, _git(repo, 'rev-parse', 'HEAD')


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(repo), *args], check=True, capture_output=True, text=True,
    ).stdout.strip()


def task_verify_spec(
    sha: str, test_command: str, *, preprovision: str = '', tip_sha: str | None = None,
) -> TaskVerifySpec:
    """A spec running *test_command* on the seeded module, tipped at *sha* unless *tip_sha* is given."""
    return TaskVerifySpec(
        verify_commands=(VerifyCommand(VERIFY_MODULE, test_command=test_command),),
        global_verify_command=None,
        task_files=(VERIFY_TASK_FILE,),
        force_workspace=False,
        verify_env={},
        scope_cargo=False,
        verify_cold_preprovision_command=preprovision,
        attempt_id=1,
        task_id='t1',
        tip_sha=sha if tip_sha is None else tip_sha,
    )
