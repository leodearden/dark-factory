"""Which merge changes qualify for the narrow merge gate (rationale: docs/narrow-merge-gate.md)."""
from __future__ import annotations

import ast
import copy
import functools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

ChangeStatus = Literal['A', 'M', 'D']
ReadPair = Callable[[str], tuple[str | None, str | None]]
CommentPredicate = Callable[[str, str], bool]

_CHANGE_STATUSES: frozenset[str] = frozenset({'A', 'M', 'D'})
_REGULAR_FILE_MODES: frozenset[str] = frozenset({'100644', '100755'})
_UNNORMALISED_SEGMENTS: frozenset[str] = frozenset({'', '.', '..'})
_CODING_COOKIE = re.compile(r'^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)')
_DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


class MergeGateResolution(StrEnum):
    """Which command set a merge verify runs: the full gate, or one of the two narrow ones."""

    FULL = 'full_gate'
    DOCS = 'docs_gate'
    COMMENTS = 'comments_gate'

    @property
    def is_narrow(self) -> bool:
        return self is not MergeGateResolution.FULL


class MissReason(StrEnum):
    """Why a merge change took the full gate instead of a narrow one."""

    DISABLED = 'disabled'
    EMPTY_DELTA = 'empty_delta'
    DELTA_UNREADABLE = 'delta_unreadable'
    DELETED = 'deleted'
    MODE_CHANGED = 'mode_changed'
    NOT_ALLOWLISTED = 'not_allowlisted'
    CODE_CHANGED = 'code_changed'
    NO_COMMENT_PREDICATE = 'no_comment_predicate'
    UNDECODABLE = 'undecodable'
    EVIDENCE_ABSENT = 'evidence_absent'
    FORCED_FULL_GATE = 'forced_full_gate'
    EMPTY_COMMAND_SET = 'empty_command_set'


@dataclass(frozen=True)
class ChangedPath:
    """One path of a tree delta, as `git diff --raw --no-renames` reports it (modes are 6-digit octal strings)."""

    path: str
    status: ChangeStatus
    old_mode: str
    new_mode: str

    def __post_init__(self) -> None:
        if self.status not in _CHANGE_STATUSES:
            raise ValueError(f'ChangedPath status must be one of A, M, D; got {self.status!r} for {self.path!r}')


@dataclass(frozen=True)
class GateMiss:
    """The first reason a change missed the narrow gate, and the path it is about when there is one."""

    reason: MissReason
    path: str | None = None


@dataclass(frozen=True)
class GateClassification:
    """A change's gate: a narrow one carries no miss, the full one always carries the miss that sent it there."""

    resolution: MergeGateResolution
    miss: GateMiss | None = None

    def __post_init__(self) -> None:
        if self.resolution.is_narrow and self.miss is not None:
            raise ValueError(f'a {self.resolution.value} classification must carry miss=None, got {self.miss!r}')
        if not self.resolution.is_narrow and self.miss is None:
            raise ValueError(f'a {self.resolution.value} classification must carry a miss, got miss=None')


def full_gate(reason: MissReason, path: str | None = None) -> GateClassification:
    """The full-gate classification for *reason*, about *path* when there is one."""
    return GateClassification(MergeGateResolution.FULL, GateMiss(reason, path))


def matches_path_glob(path: str, pattern: str) -> bool:
    """Whether repo-relative *path* matches *pattern*: `*` and `?` stay within one segment, a `**` segment spans any depth."""
    return _compiled_glob(pattern).fullmatch(path) is not None


@functools.lru_cache(maxsize=256)
def _compiled_glob(pattern: str) -> re.Pattern[str]:
    segments = pattern.split('/')
    last = len(segments) - 1
    return re.compile(''.join(_segment_regex(segment, is_last=i == last) for i, segment in enumerate(segments)))


def _segment_regex(segment: str, *, is_last: bool) -> str:
    if segment == '**':
        return '.+' if is_last else '(?:[^/]+/)*'
    body = ''.join(_char_regex(char) for char in segment)
    return body if is_last else body + '/'


def _char_regex(char: str) -> str:
    if char == '*':
        return '[^/]*'
    if char == '?':
        return '[^/]'
    return re.escape(char)


def classify_docs_change(changes: Sequence[ChangedPath], docs_paths: Sequence[str]) -> GateClassification:
    """DOCS when every changed path is a present, regular, allowlisted prose file; otherwise FULL with the first miss."""
    if not changes:
        return full_gate(MissReason.EMPTY_DELTA)
    for change in sorted(changes, key=lambda c: c.path):
        miss = shape_miss(change) or _allowlist_miss(change.path, docs_paths)
        if miss is not None:
            return GateClassification(MergeGateResolution.FULL, miss)
    return GateClassification(MergeGateResolution.DOCS)


def is_docs_path(path: str, docs_paths: Sequence[str]) -> bool:
    """Whether *path* matches any of the *docs_paths* globs."""
    return any(matches_path_glob(path, pattern) for pattern in docs_paths)


def shape_miss(change: ChangedPath) -> GateMiss | None:
    """The miss for a path no narrow gate may take whatever its content: unnormalised, deleted, or not a plain file."""
    if _is_unnormalised(change.path):
        return GateMiss(MissReason.NOT_ALLOWLISTED, change.path)
    if change.status == 'D':
        return GateMiss(MissReason.DELETED, change.path)
    if not _is_regular_unchanged_mode(change):
        return GateMiss(MissReason.MODE_CHANGED, change.path)
    return None


def _allowlist_miss(path: str, docs_paths: Sequence[str]) -> GateMiss | None:
    return None if is_docs_path(path, docs_paths) else GateMiss(MissReason.NOT_ALLOWLISTED, path)


def _is_unnormalised(path: str) -> bool:
    return path.startswith('/') or any(segment in _UNNORMALISED_SEGMENTS for segment in path.split('/'))


def _is_regular_unchanged_mode(change: ChangedPath) -> bool:
    if change.new_mode not in _REGULAR_FILE_MODES:
        return False
    return change.status == 'A' or change.old_mode == change.new_mode


def python_comments_only(base_text: str, head_text: str) -> bool:
    """Whether two Python sources differ only in comments and docstrings, with the same shebang and coding cookie."""
    if _interpreter_header(base_text) != _interpreter_header(head_text):
        return False
    try:
        base_tree, head_tree = ast.parse(base_text), ast.parse(head_text)
    except SyntaxError:
        return False
    return _docstring_free_dump(base_tree) == _docstring_free_dump(head_tree)


def _interpreter_header(text: str) -> tuple[str | None, str | None]:
    first_lines = text.splitlines()[:2]
    shebang = first_lines[0] if first_lines and first_lines[0].startswith('#!') else None
    cookies = (match.group(1) for line in first_lines if (match := _CODING_COOKIE.match(line)))
    return shebang, next(cookies, None)


def _docstring_free_dump(tree: ast.Module) -> str:
    stripped = copy.deepcopy(tree)
    for node in ast.walk(stripped):
        if isinstance(node, _DOCSTRING_OWNERS) and _starts_with_docstring(node.body):
            node.body = node.body[1:]
    return ast.dump(stripped, include_attributes=False)


def _starts_with_docstring(body: Sequence[ast.stmt]) -> bool:
    first = body[0] if body else None
    return isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)


_COMMENT_PREDICATES: Mapping[str, CommentPredicate] = {'.py': python_comments_only}


def classify_change(changes: Sequence[ChangedPath], docs_paths: Sequence[str], read_pair: ReadPair) -> GateClassification:
    """DOCS when every path is prose, COMMENTS when the rest differ only in comments; otherwise FULL with the first miss.

    *read_pair* gives a path's (base, head) text, None for a side that is absent or not UTF-8; docs paths are never read.
    """
    if not changes:
        return full_gate(MissReason.EMPTY_DELTA)
    resolution = MergeGateResolution.DOCS
    for change in sorted(changes, key=lambda c: c.path):
        verdict = _path_verdict(change, docs_paths, read_pair)
        if isinstance(verdict, GateMiss):
            return GateClassification(MergeGateResolution.FULL, verdict)
        if verdict is MergeGateResolution.COMMENTS:
            resolution = MergeGateResolution.COMMENTS
    return GateClassification(resolution)


def _path_verdict(change: ChangedPath, docs_paths: Sequence[str], read_pair: ReadPair) -> GateMiss | MergeGateResolution:
    shape = shape_miss(change)
    if shape is not None:
        return shape
    if is_docs_path(change.path, docs_paths):
        return MergeGateResolution.DOCS
    predicate = _COMMENT_PREDICATES.get(PurePosixPath(change.path).suffix)
    if predicate is None:
        return GateMiss(MissReason.NO_COMMENT_PREDICATE, change.path)
    if change.status == 'A':
        return GateMiss(MissReason.CODE_CHANGED, change.path)
    base_text, head_text = read_pair(change.path)
    if base_text is None or head_text is None:
        return GateMiss(MissReason.UNDECODABLE, change.path)
    if predicate(base_text, head_text):
        return MergeGateResolution.COMMENTS
    return GateMiss(MissReason.CODE_CHANGED, change.path)
