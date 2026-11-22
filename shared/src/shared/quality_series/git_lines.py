"""Reading git under a timeout with typed failures, down to one change's landed lines by path class.

Path classes and the module key are plans/quality-dashboard-prd.md decision 6 and open question 4.
"""

from __future__ import annotations

import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

__all__ = [
    'GitObjectMissing',
    'GitTimeout',
    'GitUnavailable',
    'LandedLines',
    'ModuleLines',
    'PathClass',
    'git_output',
    'landed_lines',
    'module_key',
    'path_class',
]


class GitUnavailable(Exception):
    """git is not installed, the directory is not a repository, or git failed for another reason."""


class GitObjectMissing(Exception):
    """A named commit does not exist in the repository; the lines are permanently unmeasurable."""


class GitTimeout(Exception):
    """git did not answer within the caller's budget."""


class PathClass(StrEnum):
    SRC = 'src'
    TEST = 'test'
    DOC = 'doc'
    LOCK = 'lock'
    DATA = 'data'


_LOCK_NAMES = frozenset({'uv.lock', 'package-lock.json'})
_DOC_DIRS = frozenset({'plans', 'docs'})
_CODE_SUFFIXES = frozenset(
    {'.py', '.rs', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.sh', '.html', '.css', '.toml'}
)


def _is_test(path: PurePosixPath) -> bool:
    name = path.name
    return (
        'tests' in path.parts[:-1]
        or name.startswith(('test_', 'conftest'))
        or name.endswith('_test.py')
    )


def path_class(path: str) -> PathClass:
    """The first class that claims the path, in the order lock, doc, data, test, src."""
    pure = PurePosixPath(path)
    suffix = pure.suffix.lower()
    if pure.name in _LOCK_NAMES or suffix == '.lock':
        return PathClass.LOCK
    if suffix == '.md' or _DOC_DIRS.intersection(pure.parts[:-1]):
        return PathClass.DOC
    if suffix not in _CODE_SUFFIXES:
        return PathClass.DATA
    if _is_test(pure):
        return PathClass.TEST
    return PathClass.SRC


def module_key(path: str) -> str:
    """``<pkg>/src/<pkg2>/<rest>`` keys as ``<pkg>/<rest[0]>``; else two components at depth >= 3, else one."""
    parts = PurePosixPath(path).parts
    if len(parts) == 0:
        raise ValueError(f'module_key needs a non-empty path, got {path!r}')
    if len(parts) >= 4 and parts[1] == 'src':
        return f'{parts[0]}/{parts[3]}'
    if len(parts) >= 3:
        return f'{parts[0]}/{parts[1]}'
    return parts[0]


@dataclass(frozen=True)
class ModuleLines:
    module: str
    src_lines: int


@dataclass(frozen=True)
class LandedLines:
    """Added + deleted lines; code_lines = src_lines + test_lines; modules sorted by lines then name."""

    code_lines: int
    src_lines: int
    test_lines: int
    modules: tuple[ModuleLines, ...]


def _run_git(
    repo: str | os.PathLike[str], args: list[str], timeout_s: float
) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ['git', '-C', os.fspath(repo), *args],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise GitTimeout(f'git {args[0]} in {os.fspath(repo)} exceeded {timeout_s}s') from exc
    except FileNotFoundError as exc:
        raise GitUnavailable(f'no git binary to run in {os.fspath(repo)}') from exc


def _classify_failure(
    repo: str | os.PathLike[str], args: list[str], objects: tuple[str, ...], stderr: bytes,
    timeout_s: float,
) -> Exception:
    if _run_git(repo, ['rev-parse', '--git-dir'], timeout_s).returncode != 0:
        return GitUnavailable(f'{os.fspath(repo)} is not a git repository')
    for name in objects:
        if _run_git(repo, ['cat-file', '-e', f'{name}^{{commit}}'], timeout_s).returncode != 0:
            return GitObjectMissing(f'commit {name} is not in {os.fspath(repo)}')
    return GitUnavailable(
        f'git {" ".join(args)} failed in {os.fspath(repo)}: {stderr.decode(errors="replace")}'
    )


def git_output(
    repo: str | os.PathLike[str], args: list[str], timeout_s: float,
    *, objects: tuple[str, ...] = (),
) -> bytes:
    """stdout of one ``git -C repo <args>`` bounded by ``timeout_s``.

    Raises GitTimeout, GitUnavailable, or GitObjectMissing when one of ``objects`` (the commits
    the command names) does not exist; only the failure path makes the probe calls that tell
    those apart, so no git message text is parsed.
    """
    if not timeout_s > 0:
        raise ValueError(f'a git call needs a positive timeout (timeout_s={timeout_s!r})')
    result = _run_git(repo, args, timeout_s)
    if result.returncode != 0:
        raise _classify_failure(repo, args, objects, result.stderr, timeout_s)
    return result.stdout


def _numstat(
    repo: str | os.PathLike[str], base_sha: str, sha: str, timeout_s: float
) -> list[tuple[str, int]]:
    stdout = git_output(
        repo, ['diff', '--numstat', '--no-renames', '-z', base_sha, sha], timeout_s,
        objects=(base_sha, sha),
    )
    churn: list[tuple[str, int]] = []
    for record in stdout.decode(errors='surrogateescape').split('\0'):
        if not record:
            continue
        added, deleted, path = record.split('\t', 2)
        binary = added == '-' or deleted == '-'
        churn.append((path, 0 if binary else int(added) + int(deleted)))
    return churn


def landed_lines(
    repo: str | os.PathLike[str], base_sha: str, sha: str, timeout_s: float
) -> LandedLines:
    """Churn of ``base_sha..sha`` by path class, under one git call (see git_output for errors).

    A path with no line churn (binary, or a mode-only change) touches no class and no module.
    """
    by_class: Counter[PathClass] = Counter()
    by_module: Counter[str] = Counter()
    for path, lines in _numstat(repo, base_sha, sha, timeout_s):
        if lines == 0:
            continue
        cls = path_class(path)
        by_class[cls] += lines
        if cls is PathClass.SRC:
            by_module[module_key(path)] += lines
    modules = sorted(by_module.items(), key=lambda item: (-item[1], item[0]))
    return LandedLines(
        code_lines=by_class[PathClass.SRC] + by_class[PathClass.TEST],
        src_lines=by_class[PathClass.SRC],
        test_lines=by_class[PathClass.TEST],
        modules=tuple(ModuleLines(module, lines) for module, lines in modules),
    )
