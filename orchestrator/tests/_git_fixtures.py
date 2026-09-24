"""Pristine committed git repos for tests, copied rather than rebuilt.

Each distinct :class:`RepoSeed` is built ONCE per session into a template by
:func:`build_repo`; every test gets a ``copytree`` of it.  A copied index is
stat-dirty to plumbing (inode and ctime differ, so ``git diff-files`` reports
every file) until one ``git update-index --refresh`` rewrites it — porcelain
hides this by refreshing silently, which is why the refresh is not optional.

The session instance is installed by
``orchestrator/tests/conftest.py::pristine_repo_templates``.
"""
from __future__ import annotations

import contextlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from _orch_helpers import git_env_with_ceiling
from df_pytest_isolation import git_redirect_env


@dataclass(frozen=True)
class RepoSeed:
    """The initial commit's contents: ``(relative path, text)`` pairs plus a message."""

    files: tuple[tuple[str, str], ...]
    message: str

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError('a RepoSeed must commit at least one file')
        for relpath, _text in self.files:
            path = PurePosixPath(relpath)
            if not path.parts or path.is_absolute() or '..' in path.parts:
                raise ValueError(f'seed path must stay inside the repo: {relpath!r}')


README_SEED = RepoSeed(files=(('README.md', '# Test\n'),), message='Initial commit')


def _git_env(cwd: Path) -> dict[str, str]:
    env = git_env_with_ceiling(cwd)
    for key in git_redirect_env(env):
        del env[key]
    return env


def _git(cwd: Path, *args: str) -> None:
    try:
        subprocess.run(
            ['git', *args], cwd=cwd, env=_git_env(cwd),
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f'git {" ".join(args)} failed in {cwd}: {exc.stderr.strip()}'
        ) from exc


def build_repo(dest: Path, seed: RepoSeed) -> Path:
    """The one reference recipe.  Files already in *dest* join the initial commit."""
    dest.mkdir(parents=True, exist_ok=True)
    _git(dest, 'init', '-b', 'main')
    _git(dest, 'config', 'user.email', 'test@test.com')
    _git(dest, 'config', 'user.name', 'Test')
    for relpath, text in seed.files:
        target = dest / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    _git(dest, 'add', '-A')
    _git(dest, 'commit', '-m', seed.message)
    return dest


class RepoTemplates:
    """One committed template per seed under *root*, built on first use."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._templates: dict[RepoSeed, Path] = {}

    @property
    def root(self) -> Path:
        return self._root

    def seed(self, dest: Path, seed: RepoSeed = README_SEED) -> Path:
        """Make *dest* a pristine repo for *seed*.

        A non-empty *dest* is built in place instead of copied into, so its
        existing files land in the initial commit exactly as ``git add -A``
        would have put them there.
        """
        if dest.exists() and any(dest.iterdir()):
            return build_repo(dest, seed)
        shutil.copytree(self._template(seed), dest, symlinks=True, dirs_exist_ok=True)
        _git(dest, 'update-index', '--refresh')
        return dest

    def _template(self, seed: RepoSeed) -> Path:
        if seed not in self._templates:
            self._root.mkdir(parents=True, exist_ok=True)
            scratch = Path(tempfile.mkdtemp(dir=self._root, prefix='seed-'))
            self._templates[seed] = build_repo(scratch / 'repo', seed)
        return self._templates[seed]


_session: RepoTemplates | None = None


@contextlib.contextmanager
def session_templates(root: Path) -> Iterator[RepoTemplates]:
    """Install a process-wide :class:`RepoTemplates` at *root*; restore the previous on exit."""
    global _session
    previous = _session
    templates = RepoTemplates(root)
    _session = templates
    try:
        yield templates
    finally:
        _session = previous


def seed_repo(dest: Path, seed: RepoSeed = README_SEED) -> Path:
    """Make *dest* a pristine repo for *seed* from the session's templates."""
    if _session is None:
        raise RuntimeError(
            'no session RepoTemplates is installed; '
            'orchestrator/tests/conftest.py::pristine_repo_templates installs one'
        )
    return _session.seed(dest, seed)
