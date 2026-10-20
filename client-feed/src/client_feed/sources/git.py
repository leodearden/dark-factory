"""Read-only git plumbing confined to one root: every call runs in the root, under an environment that cannot redirect it."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import IO, Final

from client_feed.contract import Branch

MAIN_BRANCH: Final = 'main'
TASK_BRANCH_PREFIX: Final = 'task/'
GIT_TIMEOUT_SECS: Final = 300

_BRANCH_REFS: Final = 'refs/heads/'
_MAIN_REF: Final = f'{_BRANCH_REFS}{MAIN_BRANCH}'
_OBJECT_SHA: Final = re.compile(r'[0-9a-f]{40}|[0-9a-f]{64}')
_DECIMAL: Final = re.compile(r'[0-9]+')
_COMMIT_MARK: Final = b'\x01'
_FIRST_PARENT_DIFF: Final = '--diff-merges=first-parent'

# Mirrors df_pytest_isolation.py::_GIT_REDIRECT_ENV: the variables that make git act on a
# repository or config other than the one at cwd.
_REDIRECTING_ENV: Final = frozenset(
    {
        'GIT_DIR',
        'GIT_WORK_TREE',
        'GIT_INDEX_FILE',
        'GIT_COMMON_DIR',
        'GIT_OBJECT_DIRECTORY',
        'GIT_ALTERNATE_OBJECT_DIRECTORIES',
        'GIT_NAMESPACE',
        'GIT_CONFIG_GLOBAL',
        'GIT_CONFIG_SYSTEM',
        'GIT_CONFIG_COUNT',
        'GIT_CONFIG_PARAMETERS',
    }
)
_REDIRECTING_ENV_PREFIXES: Final = ('GIT_CONFIG_KEY_', 'GIT_CONFIG_VALUE_')


class GitError(Exception):
    """A git command that failed; its str names no path and carries no stderr."""

    def __init__(self, *, subcommand: str, returncode: int | None, stderr: str) -> None:
        self.subcommand = subcommand
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f'git {subcommand} failed (rc {returncode})')


def _git_env() -> dict[str, str]:
    kept = {
        key: value
        for key, value in os.environ.items()
        if key not in _REDIRECTING_ENV and not key.startswith(_REDIRECTING_ENV_PREFIXES)
    }
    return {**kept, 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': os.devnull}


def _run(
    root: Path,
    *args: str,
    input_bytes: bytes | None = None,
    ok_returncodes: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[bytes]:
    """The one spawn point for a one-shot git command."""
    subcommand = args[0]
    try:
        completed = subprocess.run(
            ['git', *args],
            input=input_bytes,
            capture_output=True,
            timeout=GIT_TIMEOUT_SECS,
            check=False,
            cwd=root,
            env=_git_env(),
        )
    except subprocess.TimeoutExpired as exc:
        raise GitError(
            subcommand=subcommand, returncode=None, stderr=f'timed out after {GIT_TIMEOUT_SECS}s'
        ) from exc
    except OSError as exc:
        raise GitError(subcommand=subcommand, returncode=None, stderr=str(exc)) from exc
    if completed.returncode not in ok_returncodes:
        raise GitError(
            subcommand=subcommand,
            returncode=completed.returncode,
            stderr=completed.stderr.decode(errors='replace'),
        )
    return completed


def _require_sha(value: str) -> str:
    if _OBJECT_SHA.fullmatch(value) is None:
        raise ValueError(f'expected a 40- or 64-hex object sha, got {value!r}')
    return value


def repository_toplevel(root: Path) -> Path | None:
    """The top of the work tree containing `root`, or None when `root` is in none."""
    if not root.is_dir():
        return None
    completed = _run(root, 'rev-parse', '--show-toplevel', ok_returncodes=(0, 128))
    if completed.returncode != 0:
        return None
    return Path(os.fsdecode(completed.stdout.rstrip(b'\n')))


def main_head(root: Path) -> str | None:
    """refs/heads/main's commit, or None before main's first commit."""
    completed = _run(
        root, 'rev-parse', '--verify', '-q', f'{_MAIN_REF}^{{commit}}', ok_returncodes=(0, 1)
    )
    return completed.stdout.decode().strip() if completed.returncode == 0 else None


class EntryKind(StrEnum):
    FILE = 'file'
    SYMLINK = 'symlink'
    SUBMODULE = 'submodule'


_MODE_KINDS: Final = {
    '100644': EntryKind.FILE,
    '100755': EntryKind.FILE,
    '120000': EntryKind.SYMLINK,
    '160000': EntryKind.SUBMODULE,
}


@dataclass(frozen=True)
class TreeEntry:
    """One non-tree entry of a commit's tree; `size` is the blob's bytes, None for a submodule."""

    path: str
    kind: EntryKind
    object_sha: str
    size: int | None


def _tree_entry(record: bytes) -> TreeEntry:
    meta, _, raw_path = record.partition(b'\t')
    mode, _, sha, size = meta.decode().split()
    path = os.fsdecode(raw_path)
    kind = _MODE_KINDS.get(mode)
    if kind is None:
        raise ValueError(f'unsupported tree entry mode {mode} at {path!r}')
    return TreeEntry(path=path, kind=kind, object_sha=sha, size=None if size == '-' else int(size))


def tree_entries(root: Path, commit: str) -> tuple[TreeEntry, ...]:
    """Every file, symlink and submodule entry of `commit`'s whole tree."""
    output = _run(root, 'ls-tree', '-r', '-l', '-z', '--full-tree', _require_sha(commit)).stdout
    return tuple(_tree_entry(record) for record in output.split(b'\0') if record)


def _cat_file_error(reason: str, process: subprocess.Popen[bytes]) -> GitError:
    return GitError(subcommand='cat-file', returncode=process.poll(), stderr=reason)


def _pipes(process: subprocess.Popen[bytes]) -> tuple[IO[bytes], IO[bytes]]:
    if process.stdin is None or process.stdout is None:
        raise _cat_file_error('cat-file started without pipes', process)
    return process.stdin, process.stdout


def _read_blob(process: subprocess.Popen[bytes], sha: str) -> bytes:
    request, reply = _pipes(process)
    try:
        request.write(_require_sha(sha).encode() + b'\n')
        request.flush()
    except OSError as exc:
        raise _cat_file_error(f'request for {sha} not accepted', process) from exc
    header = reply.readline().split()
    if len(header) != 3 or header[1] != b'blob':
        raise _cat_file_error(f'object {sha} is not a readable blob', process)
    size = int(header[2])
    content = reply.read(size)
    if len(content) != size or reply.read(1) != b'\n':
        raise _cat_file_error(f'object {sha} truncated', process)
    return content


def _finish(process: subprocess.Popen[bytes]) -> None:
    try:
        _, stderr = process.communicate(timeout=GIT_TIMEOUT_SECS)
    except subprocess.TimeoutExpired as exc:
        _abandon(process)
        raise _cat_file_error(f'timed out after {GIT_TIMEOUT_SECS}s', process) from exc
    if process.returncode != 0:
        raise GitError(
            subcommand='cat-file',
            returncode=process.returncode,
            stderr=stderr.decode(errors='replace'),
        )


def _abandon(process: subprocess.Popen[bytes]) -> None:
    process.kill()
    process.communicate()


@contextlib.contextmanager
def blob_reader(root: Path) -> Iterator[Callable[[str], bytes]]:
    """`read(blob_sha) -> bytes` over one `git cat-file --batch` session: the only way client_feed reads file content."""
    try:
        process = subprocess.Popen(
            ['git', 'cat-file', '--batch'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=root,
            env=_git_env(),
        )
    except OSError as exc:
        raise GitError(subcommand='cat-file', returncode=None, stderr=str(exc)) from exc
    try:
        yield lambda sha: _read_blob(process, sha)
    except BaseException:
        _abandon(process)
        raise
    _finish(process)


def _task_id(branch: str) -> int | None:
    suffix = branch.removeprefix(TASK_BRANCH_PREFIX)
    if _DECIMAL.fullmatch(suffix) is None or int(suffix) < 1:
        return None
    return int(suffix)


def _branch(record: bytes) -> Branch:
    refname, unix, ahead_behind = record.decode().split('\0')
    name = refname.removeprefix(_BRANCH_REFS)
    return Branch(
        name=name,
        task_id=_task_id(name),
        ahead=int(ahead_behind.split()[0]),
        last_commit_at=datetime.fromtimestamp(int(unix), UTC),
    )


def task_branches(root: Path) -> tuple[Branch, ...]:
    """Every refs/heads/task/* branch, by name, with its count of commits main does not have."""
    output = _run(
        root,
        'for-each-ref',
        f'--format=%(refname)%00%(committerdate:unix)%00%(ahead-behind:{_MAIN_REF})',
        f'{_BRANCH_REFS}{TASK_BRANCH_PREFIX}',
    ).stdout
    branches = (_branch(record) for record in output.split(b'\n') if record)
    return tuple(sorted(branches, key=lambda branch: branch.name))


@dataclass(frozen=True)
class MergeStats:
    """What a commit landed on main: its diff against its first parent."""

    files_changed: int
    insertions: int
    deletions: int


def _numstat_lines(count: bytes) -> int:
    return 0 if count == b'-' else int(count)


def merge_stats(root: Path, commit: str) -> MergeStats:
    """The commit's numstat against its first parent; a rename is one file and a binary file counts no lines."""
    output = _run(
        root,
        'diff-tree', '-r', '-z', '--numstat', '-M', '--root', _FIRST_PARENT_DIFF, '--no-commit-id',
        _require_sha(commit),
    ).stdout
    tokens = output.split(b'\0')
    files = insertions = deletions = 0
    index = 0
    while index < len(tokens):
        token = tokens[index]
        index += 1
        if not token:
            continue
        added, deleted, path = token.split(b'\t', 2)
        if not path:
            index += 2
        files += 1
        insertions += _numstat_lines(added)
        deletions += _numstat_lines(deleted)
    return MergeStats(files_changed=files, insertions=insertions, deletions=deletions)


def generated_paths(root: Path, commit: str, paths: Sequence[str]) -> frozenset[str]:
    """The paths that `commit`'s own attributes mark linguist-generated."""
    source = _require_sha(commit)
    if not paths:
        return frozenset()
    output = _run(
        root,
        'check-attr', '--source', source, '-z', '--stdin', 'linguist-generated',
        input_bytes=b''.join(os.fsencode(path) + b'\0' for path in paths),
    ).stdout
    fields = output.split(b'\0')
    triples = zip(fields[0::3], fields[1::3], fields[2::3], strict=False)
    return frozenset(os.fsdecode(path) for path, _, value in triples if value in (b'set', b'true'))


def is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    """Whether `ancestor` is reachable from `descendant`; an unknown commit is no ancestor."""
    completed = _run(
        root,
        'merge-base', '--is-ancestor', _require_sha(ancestor), _require_sha(descendant),
        ok_returncodes=(0, 1, 128),
    )
    return completed.returncode == 0


def _commit_changes(record: bytes) -> tuple[datetime, list[str]]:
    header, _, names = record.partition(b'\0')
    paths = names.removeprefix(b'\n').split(b'\0')
    return datetime.fromtimestamp(int(header), UTC), [os.fsdecode(path) for path in paths if path]


def last_changes(root: Path, head: str, *, since: str | None) -> dict[str, datetime]:
    """For each path changed on `head`'s first-parent chain (after `since`), its newest change's committer time."""
    revisions = _require_sha(head) if since is None else f'{_require_sha(since)}..{head}'
    output = _run(
        root,
        'log', '--first-parent', _FIRST_PARENT_DIFF, '--no-renames', '--format=%x01%ct',
        '--name-only', '-z',
        revisions, '--',
    ).stdout
    changed: dict[str, datetime] = {}
    for record in output.split(_COMMIT_MARK)[1:]:
        when, paths = _commit_changes(record)
        for path in paths:
            changed.setdefault(path, when)
    return changed
