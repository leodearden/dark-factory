"""Read-only git plumbing confined to one root: every call runs in the root, under an environment that cannot redirect it."""

from __future__ import annotations

import contextlib
import os
import re
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO, Final

MAIN_BRANCH: Final = 'main'
TASK_BRANCH_PREFIX: Final = 'task/'
GIT_TIMEOUT_SECS: Final = 300

_MAIN_REF: Final = f'refs/heads/{MAIN_BRANCH}'
_OBJECT_SHA: Final = re.compile(r'[0-9a-f]{40}|[0-9a-f]{64}')

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
