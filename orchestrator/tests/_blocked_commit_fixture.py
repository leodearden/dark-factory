"""A real ``git commit --only`` held open inside its pre-commit hook.

This is the shape of the reify incident behind task 5229: a healthy commit on
main whose pre-commit hook runs for many minutes.  While the hook runs,
``.git/index.lock`` exists, NO process holds a file descriptor on it (git
closes the lockfile fd before running hooks), and the committing ``git``
process is alive with its cwd at the work-tree toplevel.  Tests that need that
shape get it from here rather than from a hand-made lock file, which would
have no process behind it at all.
"""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

from _orch_helpers import git_env_with_ceiling
from df_pytest_isolation import git_redirect_env

_READY_MARKER = 'HOOK-READY'
_HOOK_SCRIPT = f"""#!/bin/sh
echo "{_READY_MARKER} $PPID $$" >&2
exec sleep 3600
"""
_EXIT_TIMEOUT_S = 30.0


def hermetic_git_env(repo: Path) -> dict[str, str]:
    """The env for a git child in *repo*: discovery fenced at its parent, no redirects inherited."""
    env = git_env_with_ceiling(repo)
    for key in git_redirect_env(env):
        del env[key]
    return env


def _write_hook(hooks_dir: Path) -> None:
    hook = hooks_dir / 'pre-commit'
    hook.write_text(_HOOK_SCRIPT, encoding='utf-8')
    hook.chmod(0o755)


def _await_hook_ready(proc: subprocess.Popen[str]) -> tuple[int, int]:
    """``(git pid, hook pid)`` once the hook announces itself; raise if git exits first."""
    assert proc.stderr is not None
    seen: list[str] = []
    for line in proc.stderr:
        if line.startswith(_READY_MARKER):
            _marker, git_pid, hook_pid = line.split()
            return int(git_pid), int(hook_pid)
        seen.append(line)
    proc.kill()
    proc.wait(timeout=_EXIT_TIMEOUT_S)
    raise RuntimeError(
        f'git commit exited (rc={proc.returncode}) before its pre-commit hook ran: '
        f'{"".join(seen).strip()!r}'
    )


def _release(proc: subprocess.Popen[str], hook_pid: int | None) -> None:
    """End the hook by its pid so git aborts the commit and rolls back its index.lock."""
    if hook_pid is not None:
        with contextlib.suppress(ProcessLookupError):
            os.kill(hook_pid, signal.SIGTERM)
    try:
        proc.communicate(timeout=_EXIT_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=_EXIT_TIMEOUT_S)


@contextlib.contextmanager
def commit_blocked_in_pre_commit_hook(repo: Path, path: str) -> Iterator[int]:
    """Run ``git commit --only <path>`` in *repo* and yield its pid while its hook blocks.

    *path* must be tracked and modified in *repo*: ``--only`` needs a change to
    commit, and git refuses before the hook ever runs otherwise.  On exit the
    hook is terminated, so the commit aborts and git removes the index.lock.
    """
    with tempfile.TemporaryDirectory(prefix='blocked-commit-hooks-') as hooks_dir:
        _write_hook(Path(hooks_dir))
        proc = subprocess.Popen(
            ['git', '-c', f'core.hooksPath={hooks_dir}', 'commit', '-q',
             '--only', path, '-m', 'blocked in its pre-commit hook'],
            cwd=repo, env=hermetic_git_env(repo),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True,
        )
        hook_pid: int | None = None
        try:
            git_pid, hook_pid = _await_hook_ready(proc)
            yield git_pid
        finally:
            _release(proc, hook_pid)
