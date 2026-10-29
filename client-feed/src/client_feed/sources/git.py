"""Read-only git plumbing confined to one root: every call runs in the root, under an environment that cannot redirect it."""

from __future__ import annotations

import contextlib
import itertools
import os
import re
import subprocess
import threading
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
_PARSE_FAILURE: Final = 'output not understood'

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
    """A git command that failed or whose output could not be parsed; its str names no path and carries no stderr."""

    def __init__(self, *, subcommand: str, returncode: int | None, stderr: str) -> None:
        self.subcommand = subcommand
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f'git {subcommand} failed (rc {returncode})')

    @property
    def reason(self) -> str:
        """git's own first line of complaint, else the str; it may name paths, so it is for the operator only."""
        lines = self.stderr.strip().splitlines()
        return lines[0] if lines else str(self)


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


@contextlib.contextmanager
def _parsing(subcommand: str, *, returncode: int | None = 0) -> Iterator[None]:
    """A ValueError raised while parsing `subcommand`'s output becomes that command's GitError, its detail in stderr."""
    try:
        yield
    except ValueError as exc:
        raise GitError(
            subcommand=subcommand, returncode=returncode, stderr=f'{_PARSE_FAILURE}: {exc}'
        ) from exc


def _require_sha(value: str) -> str:
    if _OBJECT_SHA.fullmatch(value) is None:
        raise ValueError(f'expected a 40- or 64-hex object sha, got {value!r}')
    return value


def repository_toplevel(root: Path) -> Path:
    """The top of the work tree containing `root`; a GitError whose reason says why there is none usable.

    A missing root, a root in no repository and a repository git refuses (dubious ownership: host
    config, and so safe.directory, is off here) all fail alike, each with git's or the OS's own reason.
    """
    completed = _run(root, 'rev-parse', '--show-toplevel')
    return Path(os.fsdecode(completed.stdout.rstrip(b'\n')))


def _peeled_commit(root: Path, rev: str) -> str | None:
    completed = _run(root, 'rev-parse', '--verify', '-q', f'{rev}^{{commit}}', ok_returncodes=(0, 1))
    return completed.stdout.decode().strip() if completed.returncode == 0 else None


def main_head(root: Path) -> str | None:
    """refs/heads/main's commit, or None before main's first commit."""
    return _peeled_commit(root, _MAIN_REF)


def has_commit(root: Path, commit: str) -> bool:
    """Whether the repository holds `commit` as a commit object; a GitError, never False, when git cannot answer."""
    return _peeled_commit(root, _require_sha(commit)) is not None


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
    with _parsing('ls-tree'):
        return tuple(_tree_entry(record) for record in output.split(b'\0') if record)


def _cat_file_error(reason: str, process: subprocess.Popen[bytes]) -> GitError:
    return GitError(subcommand='cat-file', returncode=process.poll(), stderr=reason)


def _pipes(process: subprocess.Popen[bytes]) -> tuple[IO[bytes], IO[bytes]]:
    if process.stdin is None or process.stdout is None:
        raise _cat_file_error('cat-file started without pipes', process)
    return process.stdin, process.stdout


@dataclass(frozen=True)
class _CatFile:
    """One `git cat-file --batch` session, and how long git has to answer each read and to exit at the end."""

    process: subprocess.Popen[bytes]
    timeout_secs: float


def _timed_out(session: _CatFile) -> GitError:
    return GitError(
        subcommand='cat-file', returncode=None, stderr=f'timed out after {session.timeout_secs}s'
    )


@contextlib.contextmanager
def _deadline(session: _CatFile) -> Iterator[None]:
    """The block, its session killed once it outlasts the timeout; a killed session's block is a timeout, whatever the pipe gave."""
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        session.process.kill()

    timer = threading.Timer(session.timeout_secs, expire)
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        timer.join()
        if expired.is_set():
            raise _timed_out(session)


def _read_blob(session: _CatFile, sha: str) -> bytes:
    with _deadline(session):
        return _exchange(session.process, sha)


def _exchange(process: subprocess.Popen[bytes], sha: str) -> bytes:
    request, reply = _pipes(process)
    try:
        request.write(_require_sha(sha).encode() + b'\n')
        request.flush()
    except OSError as exc:
        raise _cat_file_error(f'request for {sha} not accepted', process) from exc
    header = reply.readline().split()
    if len(header) != 3 or header[1] != b'blob':
        raise _cat_file_error(f'object {sha} is not a readable blob', process)
    with _parsing('cat-file', returncode=process.poll()):
        size = int(header[2])
    content = reply.read(size)
    if len(content) != size or reply.read(1) != b'\n':
        raise _cat_file_error(f'object {sha} truncated', process)
    return content


def _finish(session: _CatFile) -> None:
    process = session.process
    try:
        _, stderr = process.communicate(timeout=session.timeout_secs)
    except subprocess.TimeoutExpired as exc:
        _abandon(process)
        raise _timed_out(session) from exc
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
def blob_reader(
    root: Path, *, timeout_secs: float = GIT_TIMEOUT_SECS
) -> Iterator[Callable[[str], bytes]]:
    """`read(blob_sha) -> bytes` over one `git cat-file --batch` session: the only way client_feed reads file content.

    git has `timeout_secs` to answer each read, so the caller's work between reads never counts against it.
    """
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
    session = _CatFile(process=process, timeout_secs=timeout_secs)
    try:
        yield lambda sha: _read_blob(session, sha)
    except BaseException:
        _abandon(process)
        raise
    _finish(session)


def _task_id(branch: str) -> int | None:
    suffix = branch.removeprefix(TASK_BRANCH_PREFIX)
    if _DECIMAL.fullmatch(suffix) is None or int(suffix) < 1:
        return None
    return int(suffix)


def _branch(record: bytes) -> Branch | None:
    """One for-each-ref record; None for a ref whose name is not UTF-8, which no feed record can carry."""
    raw_refname, unix, ahead_behind = record.split(b'\0')
    try:
        refname = raw_refname.decode()
    except UnicodeDecodeError:
        return None
    name = refname.removeprefix(_BRANCH_REFS)
    return Branch(
        name=name,
        task_id=_task_id(name),
        ahead=int(ahead_behind.split()[0]),
        last_commit_at=datetime.fromtimestamp(int(unix), UTC),
    )


def task_branches(root: Path) -> tuple[Branch, ...]:
    """Every refs/heads/task/* branch with a UTF-8 name, by name, with its count of commits main does not have."""
    output = _run(
        root,
        'for-each-ref',
        f'--format=%(refname)%00%(committerdate:unix)%00%(ahead-behind:{_MAIN_REF})',
        f'{_BRANCH_REFS}{TASK_BRANCH_PREFIX}',
    ).stdout
    with _parsing('for-each-ref'):
        branches = [_branch(record) for record in output.split(b'\n') if record]
    return tuple(sorted(filter(None, branches), key=lambda branch: branch.name))


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
    with _parsing('diff-tree'):
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
    head = _require_sha(head)
    revisions = head if since is None else f'{_require_sha(since)}..{head}'
    output = _run(
        root,
        'log', '--first-parent', _FIRST_PARENT_DIFF, '--no-renames', '--format=%x01%ct',
        '--name-only', '-z',
        revisions, '--',
    ).stdout
    changed: dict[str, datetime] = {}
    with _parsing('log'):
        for record in output.split(_COMMIT_MARK)[1:]:
            when, paths = _commit_changes(record)
            for path in paths:
                changed.setdefault(path, when)
    return changed


class ChangeStatus(StrEnum):
    ADDED = 'A'
    DELETED = 'D'
    MODIFIED = 'M'
    RENAMED = 'R'
    TYPE_CHANGED = 'T'


_ABSENT_MODE: Final = '000000'
_ABSENT_SHA: Final = re.compile(r'0{40}|0{64}')
_HUNK_MARK: Final = b'@@'
_ADDED_MARK: Final = b'+'


@dataclass(frozen=True)
class PathChange:
    """One path's change between two trees; each side absent (None) where the path did not exist."""

    status: ChangeStatus
    old_path: str | None
    new_path: str | None
    old_blob: str | None
    new_blob: str | None
    old_kind: EntryKind | None
    new_kind: EntryKind | None


def _change_status(letter: str) -> ChangeStatus:
    try:
        return ChangeStatus(letter.rstrip('0123456789'))
    except ValueError as exc:
        raise ValueError(f'unsupported change status {letter}') from exc


def _side_kind(mode: str) -> EntryKind | None:
    if mode == _ABSENT_MODE:
        return None
    kind = _MODE_KINDS.get(mode)
    if kind is None:
        raise ValueError(f'unsupported tree entry mode {mode} in a change')
    return kind


def _side_blob(sha: str) -> str | None:
    return None if _ABSENT_SHA.fullmatch(sha) else sha


def _record_path(tokens: Iterator[bytes], meta: bytes) -> str:
    path = next(tokens, None)
    if path is None:
        raise ValueError(f'raw change record {meta!r} ends before its path')
    return os.fsdecode(path)


def _raw_change(meta: bytes, tokens: Iterator[bytes]) -> PathChange:
    """One `--raw -z` record from its ':<modes> <shas> <status>' token, its one path (two for a rename) taken from `tokens`."""
    old_mode, new_mode, old_sha, new_sha, letter = meta.removeprefix(b':').decode().split()
    status = _change_status(letter)
    first = _record_path(tokens, meta)
    second = _record_path(tokens, meta) if status is ChangeStatus.RENAMED else first
    return PathChange(
        status=status,
        old_path=None if status is ChangeStatus.ADDED else first,
        new_path=None if status is ChangeStatus.DELETED else second,
        old_blob=_side_blob(old_sha),
        new_blob=_side_blob(new_sha),
        old_kind=_side_kind(old_mode),
        new_kind=_side_kind(new_mode),
    )


def _raw_changes(tokens: Iterator[bytes]) -> tuple[PathChange, ...]:
    return tuple(_raw_change(meta, tokens) for meta in tokens if meta)


def tree_changes(root: Path, old: str, new: str) -> tuple[PathChange, ...]:
    """Every path that differs between two commits' trees, renames paired by git's default detection."""
    output = _run(
        root, 'diff-tree', '-r', '-z', '-M', '--raw', '--no-abbrev', _require_sha(old), _require_sha(new)
    ).stdout
    with _parsing('diff-tree'):
        return _raw_changes(iter(output.split(b'\0')))


def _change_sets(tokens: Iterator[bytes]) -> tuple[tuple[PathChange, ...], ...]:
    """A `log --raw -z` stream split per commit: a commit mark opens a set, and each record consumes its own paths."""
    sets: list[list[PathChange]] = []
    for token in tokens:
        if token.startswith(_COMMIT_MARK):
            sets.append([])
        elif token and not sets:
            raise ValueError('raw change record before any commit')
        elif token:
            sets[-1].append(_raw_change(token.removeprefix(b'\n'), tokens))
    return tuple(tuple(changes) for changes in sets)


def first_parent_changes(root: Path, *, since: str, head: str) -> tuple[tuple[PathChange, ...], ...]:
    """Each first-parent commit after `since` through `head`, oldest first, as its changes against its first parent."""
    output = _run(
        root,
        'log', '--first-parent', _FIRST_PARENT_DIFF, '-M', '--raw', '-z', '--no-abbrev', '--reverse',
        '--format=%x01%H',
        f'{_require_sha(since)}..{_require_sha(head)}', '--',
    ).stdout
    with _parsing('log'):
        return _change_sets(iter(output.split(b'\0')))


def added_lines(root: Path, old_blob: str, new_blob: str) -> bytes:
    """The lines `new_blob` adds or changes against `old_blob`, joined by newlines; nothing for identical blobs."""
    output = _run(
        root,
        'diff', '-U0', '-a', '--no-color', '--no-ext-diff', '--no-textconv',
        _require_sha(old_blob), _require_sha(new_blob),
        ok_returncodes=(0, 1),
    ).stdout
    lines = iter(output.split(b'\n'))
    hunks = itertools.dropwhile(lambda line: not line.startswith(_HUNK_MARK), lines)
    return b'\n'.join(line[1:] for line in hunks if line.startswith(_ADDED_MARK))
