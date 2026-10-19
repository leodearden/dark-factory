"""Static enumeration of ``pytest.mark.defect`` refs: which collected test node carries which ref.

plans/defect-tests-prd.md decision 8 makes this static scan the authoritative binding of a defect
ref to its node ids, and decision 14 makes completeness part of every answer: a file whose refs
cannot all be read statically is incomplete, never clean. Collection mirrors pytest's DEFAULT
rules only, so a project that overrides them is a known, fail-safe limit.
"""

from __future__ import annotations

import ast
import posixpath
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from shared.defects import MalformedDefectRef, NodeId, parse_ref
from shared.pytest_mark_grammar import marker_name, pytestmark_elements

__all__ = [
    'Carrier',
    'DisablingMark',
    'EnumerationGap',
    'FileEnumeration',
    'GapReason',
    'NonStaticMark',
    'enumerate_source',
]

_DEFECT_MARKER = 'defect'
_TEST_FUNCTION_PREFIX = 'test'
_TEST_CLASS_PREFIX = 'Test'
_CONSTRUCTORS = frozenset({'__init__', '__new__'})


class DisablingMark(StrEnum):
    """A pytest mark that statically keeps a carrier from running as a guard; values are pytest's names."""

    SKIP = 'skip'
    SKIPIF = 'skipif'
    XFAIL = 'xfail'


class GapReason(StrEnum):
    SYNTAX_ERROR = 'syntax_error'
    UNREADABLE_REF = 'unreadable_ref'
    MALFORMED_REF = 'malformed_ref'
    UNREADABLE_FILE = 'unreadable_file'


@dataclass(frozen=True)
class EnumerationGap:
    """Why a file's refs cannot all be seen, at the source line that hides them when one is known."""

    reason: GapReason
    line: int | None


@dataclass(frozen=True)
class Carrier:
    node_id: NodeId
    disabled_by: frozenset[DisablingMark]

    @property
    def runnable(self) -> bool:
        return not self.disabled_by


@dataclass(frozen=True)
class NonStaticMark:
    """A readable defect mark that reaches no statically collected item, so it carries nothing."""

    ref: str
    line: int


@dataclass(frozen=True)
class FileEnumeration:
    """One file's carriers by ref, its non-static marks, and the gaps that make it incomplete."""

    path: str
    carriers: Mapping[str, tuple[Carrier, ...]]
    non_static: tuple[NonStaticMark, ...] = ()
    gaps: tuple[EnumerationGap, ...] = ()

    def __post_init__(self) -> None:
        if not _is_tree_relative(self.path):
            raise ValueError(f'FileEnumeration path {self.path!r} is not a normalised tree-relative POSIX path')
        carriers = dict(self.carriers)
        for ref in carriers:
            parse_ref(ref)
        object.__setattr__(self, 'carriers', MappingProxyType(carriers))

    @property
    def complete(self) -> bool:
        return not self.gaps


def _is_tree_relative(path: str) -> bool:
    return (
        bool(path)
        and path != '.'
        and not path.startswith('/')
        and posixpath.normpath(path) == path
        and path.split('/', 1)[0] != '..'
    )


def _module_relative(path: str, module_root: str) -> str:
    """*path* made relative to *module_root*, which is '.' or a directory prefix of it."""
    if not _is_tree_relative(path) or not (module_root == '.' or _is_tree_relative(module_root)):
        raise ValueError(
            f'path {path!r} and module_root {module_root!r} must be normalised tree-relative '
            "POSIX paths; module_root may also be '.'"
        )
    if module_root == '.':
        return path
    prefix = f'{module_root}/'
    if not path.startswith(prefix):
        raise ValueError(f'path {path!r} is not inside module_root {module_root!r}')
    return path[len(prefix):]


def _read_ref(mark: ast.expr) -> str | EnumerationGap:
    """The ref one defect mark names: exactly one positional str literal, no keywords, in the grammar."""
    if not (isinstance(mark, ast.Call) and len(mark.args) == 1 and not mark.keywords):
        return EnumerationGap(GapReason.UNREADABLE_REF, mark.lineno)
    argument = mark.args[0]
    if not (isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
        return EnumerationGap(GapReason.UNREADABLE_REF, mark.lineno)
    try:
        return parse_ref(argument.value).ref
    except MalformedDefectRef:
        return EnumerationGap(GapReason.MALFORMED_REF, mark.lineno)


def _defect_marks(tree: ast.Module) -> list[ast.expr]:
    """Every defect-mark expression in *tree*, by position: each ``defect(...)`` call and each bare mark."""
    found = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call | ast.Attribute) and marker_name(node) == _DEFECT_MARKER
    ]
    called = {id(node.func) for node in found if isinstance(node, ast.Call)}
    marks = [node for node in found if id(node) not in called]
    return sorted(marks, key=lambda mark: (mark.lineno, mark.col_offset))


@dataclass(frozen=True)
class _ScopeMarks:
    """The marks that reach a scope from it and every scope enclosing it."""

    defect_marks: tuple[ast.expr, ...] = ()

    def with_elements(self, elements: Sequence[ast.expr]) -> _ScopeMarks:
        defect_marks = tuple(element for element in elements if marker_name(element) == _DEFECT_MARKER)
        return _ScopeMarks(self.defect_marks + defect_marks)


_Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _winning_definitions(body: Sequence[ast.stmt]) -> list[_Definition]:
    """The scope's direct-child definitions, the last one of each name winning its first's place."""
    definitions: dict[str, _Definition] = {}
    for statement in body:
        if isinstance(statement, _Definition):
            definitions[statement.name] = statement
    return list(definitions.values())


def _is_test_class(node: ast.ClassDef) -> bool:
    if not node.name.startswith(_TEST_CLASS_PREFIX):
        return False
    return not any(
        isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef) and statement.name in _CONSTRUCTORS
        for statement in node.body
    )


def _test_items(
    body: Sequence[ast.stmt], inherited: _ScopeMarks, class_chain: tuple[str, ...]
) -> Iterator[tuple[str, _ScopeMarks]]:
    """Each item pytest's default collection takes from a scope *body*, named, with the marks reaching it."""
    scope = inherited.with_elements(pytestmark_elements(body))
    for definition in _winning_definitions(body):
        marks = scope.with_elements(definition.decorator_list)
        if isinstance(definition, ast.ClassDef):
            if _is_test_class(definition):
                yield from _test_items(definition.body, marks, (*class_chain, definition.name))
        elif definition.name.startswith(_TEST_FUNCTION_PREFIX):
            yield '::'.join((*class_chain, definition.name)), marks


_Reading = str | EnumerationGap


def _carriers(
    items: Iterator[tuple[str, _ScopeMarks]],
    readings: Mapping[int, _Reading],
    node_of: Callable[[str], NodeId],
) -> tuple[dict[str, tuple[Carrier, ...]], frozenset[int]]:
    """The carriers by ref the static *items* yield, and the ids of the census marks they reach."""
    carriers: dict[str, dict[Carrier, None]] = {}
    reached: set[int] = set()
    for name, marks in items:
        for mark in marks.defect_marks:
            reached.add(id(mark))
            reading = readings[id(mark)]
            if isinstance(reading, str):
                carriers.setdefault(reading, {})[Carrier(node_of(name), frozenset())] = None
    return {ref: tuple(item_carriers) for ref, item_carriers in carriers.items()}, frozenset(reached)


def _non_static(
    marks: Sequence[ast.expr], readings: Mapping[int, _Reading], reached: frozenset[int]
) -> tuple[NonStaticMark, ...]:
    return tuple(
        NonStaticMark(reading, mark.lineno)
        for mark in marks
        if id(mark) not in reached and isinstance(reading := readings[id(mark)], str)
    )


def _enumerate_module(path: str, node_of: Callable[[str], NodeId], tree: ast.Module) -> FileEnumeration:
    marks = _defect_marks(tree)
    readings = {id(mark): _read_ref(mark) for mark in marks}
    carriers, reached = _carriers(_test_items(tree.body, _ScopeMarks(), ()), readings, node_of)
    gaps = tuple(reading for reading in readings.values() if isinstance(reading, EnumerationGap))
    return FileEnumeration(path, carriers, non_static=_non_static(marks, readings, reached), gaps=gaps)


def enumerate_source(path: str, source: str | bytes, *, module_root: str) -> FileEnumeration:
    """Enumerate the defect refs in one file's *source*, at tree-relative POSIX *path*.

    *module_root* is '.' or a directory prefix of *path*; each carrier's NodeId is qualified by it,
    with *path* made relative to it. A source that does not parse is one SYNTAX_ERROR gap.
    """
    relative_path = _module_relative(path, module_root)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as error:
        gap = EnumerationGap(GapReason.SYNTAX_ERROR, getattr(error, 'lineno', None))
        return FileEnumeration(path, {}, gaps=(gap,))

    def node_of(name: str) -> NodeId:
        return NodeId(module_root=module_root, path=relative_path, name=name)

    return _enumerate_module(path, node_of, tree)
