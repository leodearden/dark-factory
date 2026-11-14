"""Which watched Python modules has the mainline removed since a given moment.

Answers one git question for scripts/orchestrator-watchdog.py's staleness pass
(task 7198): a module removed from main after an orchestrator unit started arms
a ModuleNotFoundError in that unit the first time it imports the module lazily
(task 7098). The answer is the watched ``.py`` files present on the mainline as
of the moment and absent at HEAD, plus when the first mainline commit removing
one of them landed.

Pure in its inputs: the caller names the repository, the moment, the watched
paths and the logger, so nothing here reads a clock, the environment or a
module global. Stdlib-only, like its importer. Every failure is reported
through ``log.warning`` and answers "no removals": a probe that cannot read the
history must never be taken to have found one.

Import-resolution contract: this stays a flat sibling in scripts/, which a
directly executed script finds because its own directory is sys.path[0].
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

GIT_TIMEOUT_SECS = 5

_NO_ANSWER = "reporting no module removals"


class Warns(Protocol):
    def warning(self, msg: str) -> None: ...


@dataclass(frozen=True)
class ModuleRemovals:
    """Watched modules removed since a moment, and when the first removal landed on main."""

    paths: tuple[str, ...]
    first_landed: int


def removals_since(
    repo_dir: str, since: float, watched: Sequence[str], log: Warns
) -> ModuleRemovals | None:
    """Return the watched .py files the mainline of *repo_dir* has removed since *since*.

    The baseline is the first-parent commit HEAD's history stood at as of
    *since*. A rename reads as the removal of its old path, and so does a
    module turned into a package (``foo.py`` to ``foo/__init__.py``). A
    side-branch removal is dated by the merge that brought it onto main.
    Returns None when nothing was removed, and after a WARNING when git could
    not answer.
    """
    base = _mainline_commit_at(repo_dir, since, log)
    if base is None:
        return None
    removed = _removed_paths(repo_dir, base, watched, log)
    if not removed:
        return None
    first_landed = _first_removal_epoch(repo_dir, base, removed, log)
    if first_landed is None:
        return None
    return ModuleRemovals(removed, first_landed)


def _git_stdout(repo_dir: str, args: list[str], purpose: str, log: Warns) -> str | None:
    """Return the stdout of ``git -C <repo_dir> <args>``, or None after a WARNING naming *purpose*."""
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECS,
        )
    except Exception as exc:
        log.warning(f"{purpose}: git raised {exc!r}; {_NO_ANSWER}")
        return None
    if result.returncode != 0:
        log.warning(
            f"{purpose}: git exited {result.returncode}: {result.stderr.strip()}; {_NO_ANSWER}"
        )
        return None
    return result.stdout


def _mainline_commit_at(repo_dir: str, epoch: float, log: Warns) -> str | None:
    stdout = _git_stdout(
        repo_dir,
        ["rev-list", "-1", "--first-parent", f"--before=@{int(epoch)}", "HEAD"],
        "module-removal probe: mainline commit at the baseline",
        log,
    )
    if stdout is None:
        return None
    commit = stdout.strip()
    if not commit:
        log.warning(
            f"module-removal probe: no mainline commit at or before {int(epoch)}; {_NO_ANSWER}"
        )
        return None
    return commit


def _removed_paths(
    repo_dir: str, base: str, watched: Sequence[str], log: Warns
) -> tuple[str, ...]:
    stdout = _git_stdout(
        repo_dir,
        ["diff", "--no-renames", "--name-only", "-z", "--diff-filter=D", base, "HEAD", "--", *watched],
        "module-removal probe: watched modules removed since the baseline",
        log,
    )
    if stdout is None:
        return ()
    return tuple(path for path in stdout.split("\0") if path.endswith(".py"))


def _first_removal_epoch(
    repo_dir: str, base: str, removed: tuple[str, ...], log: Warns
) -> int | None:
    stdout = _git_stdout(
        repo_dir,
        [
            "log",
            "--first-parent",
            "--no-renames",
            "--diff-filter=D",
            "--format=%ct",
            f"{base}..HEAD",
            "--",
            *removed,
        ],
        "module-removal probe: landing time of the first removal",
        log,
    )
    if stdout is None:
        return None
    try:
        return min(int(epoch) for epoch in stdout.split())
    except ValueError:
        log.warning(
            f"module-removal probe: no landing time for {list(removed)} in {stdout!r}; "
            f"{_NO_ANSWER}"
        )
        return None
