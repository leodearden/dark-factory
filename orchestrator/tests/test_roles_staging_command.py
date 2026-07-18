"""Behavioral + regression-guard tests for the mandated git staging command.

Task 2745 (refile of 2721): the implementer-mandated staging command
`git add -- . ':!.task'` names the gitignored `.task` path directly in a
pathspec. That trips git's "paths are ignored by one of your .gitignore
files" advice AND makes git exit 1 -- even though staging still succeeds
and correctly excludes `.task/`. Agents kept re-deriving this exit-1 as a
"gotcha" from scratch because the command itself was the friction, not
missing documentation.

The fix drops the redundant `:!.task` exclusion: `.task/` is already
gitignored at the repo root, so a plain `git add -- .` excludes it for
free, with exit 0. `MANDATED_STAGING_COMMAND` is the single source of
truth every staging-rules role prompt must cite, so a partial fix (some
roles missed) can't happen again.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from orchestrator.agents.roles import MANDATED_STAGING_COMMAND


def _init_repo_with_gitignored_task_dir(repo: Path) -> None:
    """Create a throwaway repo with a gitignored, populated `.task/` dir.

    Mirrors a real worktree's shape: a repo-root `.gitignore` excluding
    `.task/`, a populated `.task/` (plan.json + iterations.jsonl), and a
    tracked source file with a pending (uncommitted) change to stage.
    """
    subprocess.run(['git', 'init', '-b', 'main'], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ['git', 'config', 'user.email', 'test@test.com'], cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True, capture_output=True)

    (repo / '.gitignore').write_text('.task/\n')
    src_dir = repo / 'src'
    src_dir.mkdir()
    (src_dir / 'mod.py').write_text("print('initial')\n")
    subprocess.run(
        ['git', 'add', '.gitignore', 'src/mod.py'], cwd=repo, check=True, capture_output=True,
    )
    subprocess.run(['git', 'commit', '-m', 'initial'], cwd=repo, check=True, capture_output=True)

    task_dir = repo / '.task'
    task_dir.mkdir()
    (task_dir / 'plan.json').write_text('{"plan": true}\n')
    (task_dir / 'iterations.jsonl').write_text('{"iteration": 1}\n')

    # A real pending change to the tracked source file, so there's something
    # genuine to stage.
    (src_dir / 'mod.py').write_text("print('changed')\n")


def _staged_paths(repo: Path) -> list[str]:
    result = subprocess.run(
        ['git', 'diff', '--cached', '--name-only'],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def test_mandated_command_exits_zero_and_excludes_task_dir(tmp_path: Path) -> None:
    """The mandated command stages tracked changes and excludes `.task/`, exit 0."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo_with_gitignored_task_dir(repo)

    result = subprocess.run(
        shlex.split(MANDATED_STAGING_COMMAND), cwd=repo, capture_output=True, text=True,
    )

    assert result.returncode == 0, (
        f'mandated staging command exited non-zero: stdout={result.stdout!r} stderr={result.stderr!r}'
    )
    staged = _staged_paths(repo)
    assert 'src/mod.py' in staged, staged
    assert all(not path.startswith('.task/') for path in staged), staged


def test_legacy_exclusion_form_exits_one_root_cause(tmp_path: Path) -> None:
    """Root-cause-as-spec: the retired `:!.task` exclusion form still exits 1.

    This pins the actual root cause (naming a gitignored path in a
    pathspec) rather than a guess -- if git's behavior here ever changes,
    this test (not just intuition) will catch it.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    _init_repo_with_gitignored_task_dir(repo)

    result = subprocess.run(
        ['git', 'add', '--', '.', ':!.task'], cwd=repo, capture_output=True, text=True,
    )

    assert result.returncode == 1, (
        'expected the legacy `:!.task` form to reproduce the exit-1 root cause; got '
        f'returncode={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}'
    )
