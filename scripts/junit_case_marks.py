"""Resolve a junit row, at the tree its run verified, to the timeout mark pytest-timeout read for it.

:class:`CaseMarks` owns one ``git cat-file --batch`` process over a repo and
reads source as ``<sha>:<path>``, so a row is judged against the code its own
run had, never against today's checkout.

A junit ``classname`` is dotted and relative to the leg's rootdir.  Its module
is the LONGEST dotted prefix naming a ``.py`` file, looked for first under the
leg's prefix directory and then under the repo root; the remaining parts are
the class chain.  The test name loses a trailing ``@<xdist group>`` and then
its ``[<params>]``.  The answer is
``shared.testing_timeout_markers.closest_timeout_site`` over that module's
sites, so it sees what that function sees and no more.  A row whose tree or
module cannot be read is an :class:`Unresolved` carrying the reason.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Protocol

from shared.testing_timeout_markers import TimeoutSite, closest_timeout_site, timeout_marker_sites

__all__ = ['CaseMarks', 'JunitRow', 'Unresolved']

_XDIST_GROUP_SUFFIX = re.compile(r'@[^@\[\]]*$')


class JunitRow(Protocol):
    """The two junit attributes a row is resolved by; ``verify_log_junits.JunitCase`` is one."""

    @property
    def classname(self) -> str: ...

    @property
    def name(self) -> str: ...


@dataclass(frozen=True)
class Unresolved:
    """A row whose closest mark could not be read, and why."""

    reason: str


@dataclass(frozen=True)
class _Object:
    kind: str
    oid: str


@dataclass(frozen=True)
class _Module:
    path: str
    blob: str
    class_chain: tuple[str, ...]


_NOT_FOUND = (' missing', ' ambiguous')


class CaseMarks:
    """Closest timeout marks of junit rows at their own trees, over one ``git cat-file --batch``.

    Object lookups are cached by name and parsed marks by blob id, so a module
    shared by many runs' trees is parsed once; no source text is kept.
    """

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._process: subprocess.Popen[bytes] | None = None
        self._objects: dict[str, _Object | None] = {}
        self._sites: dict[str, tuple[TimeoutSite, ...] | None] = {}

    def __enter__(self) -> CaseMarks:
        self._process = subprocess.Popen(
            ['git', '-C', str(self._repo_root), 'cat-file', '--batch'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        _pipe(process.stdin).close()
        process.wait()
        _pipe(process.stdout).close()

    def commit_exists(self, sha: str) -> bool:
        """True iff *sha* names a commit in the repo."""
        found = self._lookup(f'{sha}^{{commit}}')
        return found is not None and found.kind == 'commit'

    def closest(self, tree_sha: str, leg_prefix: str, case: JunitRow) -> TimeoutSite | Unresolved | None:
        """The closest timeout mark of *case* at *tree_sha*, None when it has none."""
        if not self.commit_exists(tree_sha):
            return Unresolved(f'tree {tree_sha} is not a commit in {self._repo_root}')
        module = self._module(tree_sha, leg_prefix, case.classname)
        if module is None:
            return Unresolved(
                f'no module for classname {case.classname!r} under {leg_prefix!r} or the repo '
                f'root at {tree_sha}'
            )
        sites = self._sites.get(module.blob)
        if sites is None:
            return Unresolved(f'{module.path} at {tree_sha} does not parse')
        return closest_timeout_site(sites, module.class_chain, _function_name(case.name))

    def _module(self, tree_sha: str, leg_prefix: str, classname: str) -> _Module | None:
        parts = tuple(part for part in classname.split('.') if part)
        for base in _bases(leg_prefix):
            for length in range(len(parts), 0, -1):
                path = f'{base}{"/".join(parts[:length])}.py'
                found = self._lookup(f'{tree_sha}:{path}')
                if found is not None and found.kind == 'blob':
                    return _Module(path, found.oid, parts[length:])
        return None

    def _lookup(self, name: str) -> _Object | None:
        if name not in self._objects:
            self._objects[name] = self._query(name)
        return self._objects[name]

    def _query(self, name: str) -> _Object | None:
        if self._process is None:
            raise RuntimeError('CaseMarks is used outside its `with` block')
        if '\n' in name:
            return None
        stdin, stdout = _pipe(self._process.stdin), _pipe(self._process.stdout)
        stdin.write(name.encode('utf-8') + b'\n')
        stdin.flush()
        header = stdout.readline().decode('utf-8').rstrip('\n')
        if header.endswith(_NOT_FOUND):
            return None
        oid, kind, size = header.rsplit(' ', 2)
        content = stdout.read(int(size))
        stdout.read(1)
        if kind == 'blob' and oid not in self._sites:
            self._sites[oid] = _parsed_sites(content)
        return _Object(kind, oid)


def _parsed_sites(content: bytes) -> tuple[TimeoutSite, ...] | None:
    try:
        tree = ast.parse(content.decode('utf-8'))
    except (SyntaxError, ValueError, UnicodeDecodeError):
        return None
    return timeout_marker_sites(tree, {})


def _pipe(stream: IO[bytes] | None) -> IO[bytes]:
    if stream is None:
        raise RuntimeError('git cat-file --batch was started without pipes')
    return stream


def _bases(leg_prefix: str) -> tuple[str, ...]:
    prefix = leg_prefix.strip('/')
    return (f'{prefix}/', '') if prefix else ('',)


def _function_name(test_name: str) -> str:
    return _XDIST_GROUP_SUFFIX.sub('', test_name).split('[', 1)[0]
