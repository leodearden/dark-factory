"""Read-only reads of a dispatched git tree, by content address, never from the
working copy (INV-3)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

from shared.git_async import run_git

GIT_READ_TIMEOUT_SECS = 60.0

_STDERR_TAIL_CHARS = 500
_OBJECT_LINE = re.compile(r'([0-9a-f]{40}|[0-9a-f]{64}) (blob|tree|commit)')


class TreeReadError(RuntimeError):
    """git could not answer a read of the dispatched tree."""


async def tree_of(repo: Path, rev: str) -> str:
    """The tree object id *rev* names; a tree id names itself."""
    return await _git(repo, ['rev-parse', '--verify', f'{rev}^{{tree}}'])


async def root_blobs(repo: Path, tree: str) -> tuple[tuple[str, str], ...]:
    """Every root-level file of *tree* as ``(name, blob oid)``, sorted by name."""
    listing = await _git(repo, ['ls-tree', '-z', tree])
    blobs = []
    for record in filter(None, listing.split('\0')):
        meta, _tab, name = record.partition('\t')
        _mode, kind, oid = meta.split(' ')
        if kind == 'blob':
            blobs.append((name, oid))
    return tuple(sorted(blobs))


async def directories(repo: Path, tree: str) -> tuple[str, ...]:
    """Every directory of *tree*, recursively, as a tree-root-relative path."""
    listing = await _git(repo, ['ls-tree', '-r', '-d', '--name-only', '-z', tree])
    return tuple(filter(None, listing.split('\0')))


async def object_ids(repo: Path, tree: str, paths: Iterable[str]) -> dict[str, str | None]:
    """{path: object id} for each of *paths* in *tree*, None where the path is
    absent. One ``git cat-file --batch-check`` whatever the path count."""
    wanted = list(dict.fromkeys(paths))
    for path in wanted:
        if '\n' in path:
            raise TreeReadError(f'cannot batch-check a path containing a newline: {path!r}')
    if not wanted:
        return {}
    answer = await _git(
        repo, ['cat-file', '--batch-check=%(objectname) %(objecttype)'],
        input_text=''.join(f'{tree}:{path}\n' for path in wanted),
    )
    lines = answer.split('\n')
    if len(lines) != len(wanted):
        raise TreeReadError(
            f'git cat-file --batch-check answered {len(lines)} lines for {len(wanted)} paths of {tree}',
        )
    return {path: _object_id(tree, path, line) for path, line in zip(wanted, lines, strict=True)}


async def blob_text(repo: Path, tree: str, path: str) -> str | None:
    """The text of the file at *path* in *tree*, stripped of surrounding
    whitespace as every ``run_git`` stdout is; None when the path is absent."""
    if (await object_ids(repo, tree, [path]))[path] is None:
        return None
    return await _git(repo, ['cat-file', 'blob', f'{tree}:{path}'])


def _object_id(tree: str, path: str, line: str) -> str | None:
    if line == f'{tree}:{path} missing':
        return None
    match = _OBJECT_LINE.fullmatch(line)
    if match is None:
        raise TreeReadError(f'git cat-file --batch-check answered {line!r} for {path!r} in {tree}')
    return match.group(1)


async def _git(repo: Path, args: Sequence[str], *, input_text: str | None = None) -> str:
    argv = ['git', *args]
    command = ' '.join(argv)
    try:
        result = await run_git(argv, cwd=repo, input_text=input_text, timeout=GIT_READ_TIMEOUT_SECS)
    except (OSError, UnicodeDecodeError) as exc:
        raise TreeReadError(f'{command} could not run in {repo}: {exc}') from exc
    if not result.ok:
        raise TreeReadError(
            f'{command} failed in {repo} (rc={result.returncode}, timed_out={result.timed_out}): '
            f'{result.stderr[-_STDERR_TAIL_CHARS:]}',
        )
    return result.stdout
