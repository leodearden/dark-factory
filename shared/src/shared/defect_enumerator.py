"""Static enumeration of ``pytest.mark.defect`` refs: which collected test node carries which ref.

plans/defect-tests-prd.md decision 8 makes this static scan the authoritative binding of a defect
ref to its node ids, and decision 14 makes completeness part of every answer: a file whose refs
cannot all be read statically is incomplete, never clean. Collection mirrors pytest's DEFAULT
rules only, so a project that overrides them is a known, fail-safe limit.
"""

from __future__ import annotations

import argparse
import ast
import os
import posixpath
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatchcase
from pathlib import Path
from types import MappingProxyType

from shared.cli_boundary import LoudArgumentParser, run_cli
from shared.defects import MalformedDefectRef, NodeId, parse_ref
from shared.pytest_mark_grammar import marker_name, pytestmark_elements

__all__ = [
    'Carrier',
    'DisablingMark',
    'EnumerationGap',
    'FileEnumeration',
    'GapReason',
    'NonStaticMark',
    'TreeEnumeration',
    'enumerate_source',
    'enumerate_tree',
    'is_test_file',
    'main',
]

_DEFECT_MARKER = 'defect'
_CONSTRUCTORS = frozenset({'__init__', '__new__'})

# pytest 9.0.3's defaults: python_files, python_functions and python_classes (_pytest/python.py),
# and norecursedirs (_pytest/main.py). All matching is case-sensitive.
_PYTHON_FILES = ('test_*.py', '*_test.py')
_TEST_FUNCTION_PREFIX = 'test'
_TEST_CLASS_PREFIX = 'Test'
_NORECURSE_DIRS = ('*.egg', '.*', '_darcs', 'build', 'CVS', 'dist', 'node_modules', 'venv', '{arch}')


class DisablingMark(StrEnum):
    """A pytest mark that statically keeps a carrier from running as a guard; values are pytest's names."""

    SKIP = 'skip'
    SKIPIF = 'skipif'
    XFAIL = 'xfail'


_DISABLING_MARKS = frozenset(DisablingMark)


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
    disabled_by: frozenset[DisablingMark] = frozenset()

    def with_elements(self, elements: Sequence[ast.expr]) -> _ScopeMarks:
        named = [(element, marker_name(element)) for element in elements]
        defect_marks = tuple(element for element, name in named if name == _DEFECT_MARKER)
        disabled_by = frozenset(DisablingMark(name) for _, name in named if name in _DISABLING_MARKS)
        return _ScopeMarks(self.defect_marks + defect_marks, self.disabled_by | disabled_by)


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
                carriers.setdefault(reading, {})[Carrier(node_of(name), marks.disabled_by)] = None
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


@dataclass(frozen=True)
class TreeEnumeration:
    """An aggregation of FileEnumerations, one per distinct path and held in path order."""

    files: tuple[FileEnumeration, ...]

    def __post_init__(self) -> None:
        counts = Counter(file.path for file in self.files)
        duplicated = sorted(path for path, count in counts.items() if count > 1)
        if duplicated:
            raise ValueError(f'TreeEnumeration holds more than one enumeration of {duplicated}')
        object.__setattr__(self, 'files', tuple(sorted(self.files, key=lambda file: file.path)))

    @property
    def carriers(self) -> Mapping[str, tuple[Carrier, ...]]:
        """Every file's carriers merged per ref: refs sorted, carriers in path then source order."""
        merged: dict[str, list[Carrier]] = {}
        for file in self.files:
            for ref, carriers in file.carriers.items():
                merged.setdefault(ref, []).extend(carriers)
        return MappingProxyType({ref: tuple(merged[ref]) for ref in sorted(merged)})

    @property
    def runnable_refs(self) -> frozenset[str]:
        return frozenset(
            ref for ref, carriers in self.carriers.items() if any(carrier.runnable for carrier in carriers)
        )

    @property
    def non_static_refs(self) -> frozenset[str]:
        return frozenset(mark.ref for file in self.files for mark in file.non_static)

    @property
    def incomplete_files(self) -> tuple[str, ...]:
        return tuple(file.path for file in self.files if not file.complete)

    @property
    def complete(self) -> bool:
        return not self.incomplete_files


def is_test_file(path: str) -> bool:
    """Whether pytest's default python_files collects the file at *path*."""
    name = posixpath.basename(path)
    return any(fnmatchcase(name, pattern) for pattern in _PYTHON_FILES)


def _is_norecurse(directory_name: str) -> bool:
    return any(fnmatchcase(directory_name, pattern) for pattern in _NORECURSE_DIRS)


def _refuse_unlistable(error: OSError) -> None:
    raise error


def _test_file_paths(root: Path) -> list[str]:
    """Tree-relative POSIX paths of every test file under *root*, sorted; an unlistable directory raises."""
    paths: list[str] = []
    for directory, directory_names, file_names in os.walk(root, onerror=_refuse_unlistable, followlinks=False):
        directory_names[:] = [name for name in directory_names if not _is_norecurse(name)]
        relative = Path(directory).relative_to(root)
        paths.extend((relative / name).as_posix() for name in file_names if is_test_file(name))
    return sorted(paths)


def _enumerate_file(root: Path, path: str, module_root: str) -> FileEnumeration:
    try:
        source = (root / path).read_bytes()
    except OSError:
        return FileEnumeration(path, {}, gaps=(EnumerationGap(GapReason.UNREADABLE_FILE, None),))
    return enumerate_source(path, source, module_root=module_root)


def enumerate_tree(root: Path, *, module_root_of: Callable[[str], str]) -> TreeEnumeration:
    """Enumerate every test file pytest's default rules would collect under *root*.

    *module_root_of* maps each tree-relative path to the module root that qualifies its NodeIds.
    A file that cannot be read is an UNREADABLE_FILE gap, so the walk always covers the rest.
    """
    return TreeEnumeration(tuple(
        _enumerate_file(root, path, module_root_of(path)) for path in _test_file_paths(root)
    ))


def _directory(value: str) -> Path:
    path = Path(value)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f'{value!r} is not a directory')
    return path


def _build_parser() -> LoudArgumentParser:
    parser = LoudArgumentParser(
        prog='python -m shared.defect_enumerator',
        description=(
            'List the pytest.mark.defect refs each test file under TREE carries, and whether every '
            'file could be read completely. Exits 1 when any file is incomplete.'
        ),
    )
    parser.add_argument('tree', type=_directory, help='the directory to walk')
    return parser


def _tree_root_module(path: str) -> str:
    return '.'


def _gap_text(gap: EnumerationGap) -> str:
    return f'{gap.reason}' if gap.line is None else f'{gap.reason} at line {gap.line}'


def _file_header(file: FileEnumeration) -> str:
    if file.complete:
        return f'{file.path}: complete'
    return f'{file.path}: incomplete ({"; ".join(_gap_text(gap) for gap in file.gaps)})'


def _carrier_text(ref: str, carrier: Carrier) -> str:
    node_id = carrier.node_id
    text = f'  {ref}  {node_id.module_root}/{node_id.path}::{node_id.name}'
    if carrier.disabled_by:
        text += f' [disabled: {", ".join(sorted(carrier.disabled_by))}]'
    return text


def _render(tree: TreeEnumeration) -> Iterator[str]:
    for file in tree.files:
        yield _file_header(file)
        for ref, carriers in file.carriers.items():
            for carrier in carriers:
                yield _carrier_text(ref, carrier)
        for mark in file.non_static:
            yield f'  {mark.ref}  not static (line {mark.line})'
    yield (
        f'{len(tree.files)} files, {len(tree.incomplete_files)} incomplete, '
        f'{len(tree.carriers)} refs carried, {len(tree.non_static_refs)} refs not static'
    )


def main(argv: list[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    tree = enumerate_tree(arguments.tree, module_root_of=_tree_root_module)
    for line in _render(tree):
        print(line)
    return 0 if tree.complete else 1


if __name__ == '__main__':
    sys.exit(run_cli(main))
