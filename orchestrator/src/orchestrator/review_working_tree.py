"""Describe the uncommitted state of the checkout a review's Phase 1 measured."""

from __future__ import annotations

import enum
import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from shared.git_async import run_git

logger = logging.getLogger(__name__)

_GIT_QUERY_TIMEOUT_SECS = 30.0

_UNTRACKED_QUERY = (
    'git', '--no-optional-locks', 'ls-files', '--others', '--exclude-standard', '-z',
)
_UNCOMMITTED_QUERY = (
    'git', '--no-optional-locks', 'diff', '--name-only', '--no-renames', '-z', 'HEAD', '--',
)


class TreeChange(enum.Enum):
    UNTRACKED = 'untracked'
    UNCOMMITTED = 'uncommitted'


def _reject(problem: str, paths: Iterable[str]) -> None:
    offending = sorted(set(paths))
    if offending:
        raise ValueError(f'{problem}: {", ".join(offending)}')


@dataclass(frozen=True)
class WorkingTreeState:
    """Repo-relative POSIX paths, sorted, that make the checkout differ from HEAD."""

    untracked: tuple[str, ...]
    uncommitted: tuple[str, ...]

    def __post_init__(self) -> None:
        _reject(
            'working-tree paths must be repo-relative',
            (p for p in self.untracked + self.uncommitted if PurePosixPath(p).is_absolute()),
        )
        _reject(
            'a path cannot be both untracked and uncommitted',
            set(self.untracked) & set(self.uncommitted),
        )

    @property
    def is_clean(self) -> bool:
        return not self.untracked and not self.uncommitted

    def changes(self) -> Iterator[tuple[str, TreeChange]]:
        for path in self.untracked:
            yield path, TreeChange.UNTRACKED
        for path in self.uncommitted:
            yield path, TreeChange.UNCOMMITTED


async def read_working_tree_state(project_root: Path) -> WorkingTreeState | None:
    """The checkout's state, or None (logged) when git cannot describe it."""
    untracked = await _query_paths(project_root, _UNTRACKED_QUERY)
    if untracked is None:
        return None
    uncommitted = await _query_paths(project_root, _UNCOMMITTED_QUERY)
    if uncommitted is None:
        return None
    return WorkingTreeState(
        untracked=tuple(sorted(untracked - uncommitted)),
        uncommitted=tuple(sorted(uncommitted)),
    )


async def _query_paths(project_root: Path, query: tuple[str, ...]) -> set[str] | None:
    try:
        result = await run_git(query, cwd=project_root, timeout=_GIT_QUERY_TIMEOUT_SECS)
    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            'Working-tree read skipped for %s: `%s` raised %r',
            project_root, ' '.join(query), exc,
        )
        return None
    if not result.ok:
        logger.warning(
            'Working-tree read skipped for %s: `%s` rc=%s stderr=%s',
            project_root, ' '.join(query), result.returncode, result.stderr,
        )
        return None
    return {path for path in result.stdout.split('\0') if path}
