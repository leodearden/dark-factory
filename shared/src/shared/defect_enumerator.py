"""Static enumeration of ``pytest.mark.defect`` refs: which collected test node carries which ref.

plans/defect-tests-prd.md decision 8 makes this static scan the authoritative binding of a defect
ref to its node ids, and decision 14 makes completeness part of every answer: a file whose refs
cannot all be read statically is incomplete, never clean. Collection mirrors pytest's DEFAULT
rules only, so a project that overrides them is a known, fail-safe limit. A Test class's bases are
resolved only where the module defines them, so test methods, marks, constructors and ``__test__``
inherited from an imported base are all invisible. A ``pytestmark`` that cannot be read statically
disables its carriers as UNRESOLVED, but a decorator naming no mark is taken for an ordinary
decorator, so a skip applied by an imported decorator helper is invisible.
"""

from __future__ import annotations

import argparse
import ast
import os
import posixpath
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fnmatch import fnmatchcase
from functools import cached_property
from pathlib import Path
from types import MappingProxyType

from shared.cli_boundary import LoudArgumentParser, run_cli
from shared.defects import MalformedDefectRef, NodeId, parse_ref
from shared.pytest_mark_grammar import marker_name, pytestmark_elements, scope_binding

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
_MARK_NAMESPACE = 'mark'
_CONSTRUCTORS = frozenset({'__init__', '__new__'})
_TEST_FLAG = '__test__'

# pytest 9.0.3's defaults: python_files, python_functions and python_classes (_pytest/python.py),
# and norecursedirs and the never-collected virtualenvs of _in_venv (_pytest/main.py). All
# matching is case-sensitive.
_PYTHON_FILES = ('test_*.py', '*_test.py')
_TEST_FUNCTION_PREFIX = 'test'
_TEST_CLASS_PREFIX = 'Test'
_NORECURSE_DIRS = ('*.egg', '.*', '_darcs', 'build', 'CVS', 'dist', 'node_modules', 'venv', '{arch}')
_VIRTUALENV_MARKERS = ('pyvenv.cfg', 'conda-meta/history')


class DisablingMark(StrEnum):
    """A pytest mark that statically keeps a carrier from running as a guard; values are pytest's names.

    UNRESOLVED is a ``pytestmark`` reaching the carrier that cannot be read statically and so may skip it.
    """

    SKIP = 'skip'
    SKIPIF = 'skipif'
    XFAIL = 'xfail'
    UNRESOLVED = 'unresolved'


_DISABLING_MARKS = frozenset({DisablingMark.SKIP, DisablingMark.SKIPIF, DisablingMark.XFAIL})
_PYTESTMARK = 'pytestmark'


class GapReason(StrEnum):
    """Why a tree entry's refs are unseen; a *_DIRECTORY reason's entry is a directory, not a file.

    REVISITED_DIRECTORY is a directory a symlink leads back to after the walk has already listed it.
    """

    SYNTAX_ERROR = 'syntax_error'
    UNREADABLE_REF = 'unreadable_ref'
    MALFORMED_REF = 'malformed_ref'
    UNREADABLE_FILE = 'unreadable_file'
    UNLISTABLE_DIRECTORY = 'unlistable_directory'
    REVISITED_DIRECTORY = 'revisited_directory'


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
    """The ref one defect mark names: ``pytest.mark.defect``, one positional str literal, no keywords, in the grammar."""
    if marker_name(mark) != _DEFECT_MARKER:
        return EnumerationGap(GapReason.UNREADABLE_REF, mark.lineno)
    if not (isinstance(mark, ast.Call) and len(mark.args) == 1 and not mark.keywords):
        return EnumerationGap(GapReason.UNREADABLE_REF, mark.lineno)
    argument = mark.args[0]
    if not (isinstance(argument, ast.Constant) and isinstance(argument.value, str)):
        return EnumerationGap(GapReason.UNREADABLE_REF, mark.lineno)
    try:
        return parse_ref(argument.value).ref
    except MalformedDefectRef:
        return EnumerationGap(GapReason.MALFORMED_REF, mark.lineno)


def _is_mark_namespace(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == _MARK_NAMESPACE


def _mark_namespace_aliases(nodes: Iterable[ast.AST]) -> frozenset[str]:
    """Names bound to pytest's mark namespace by ``from pytest import mark`` or ``NAME = <x>.mark``."""
    aliases: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.module == 'pytest':
            aliases.update(alias.asname or alias.name for alias in node.names if alias.name == _MARK_NAMESPACE)
        elif isinstance(node, ast.Assign) and _is_mark_namespace(node.value):
            aliases.update(target.id for target in node.targets if isinstance(target, ast.Name))
    return frozenset(aliases)


def _spelled_marker(element: ast.expr, aliases: frozenset[str]) -> str | None:
    """The marker *element* names, called or bare: NAME in ``<x>.mark.NAME``, or on an *aliases* name."""
    if isinstance(element, ast.Call):
        element = element.func
    if not isinstance(element, ast.Attribute):
        return None
    owner = element.value
    if _is_mark_namespace(owner) or (isinstance(owner, ast.Name) and owner.id in aliases):
        return element.attr
    return None


def _defect_marks(nodes: Sequence[ast.AST], aliases: frozenset[str]) -> list[ast.expr]:
    """Every defect-mark expression among a module's *nodes*, by position: each spelling's call, or the bare spelling.

    Spellings through an alias of pytest or its mark namespace are included, so that
    :func:`_read_ref` makes each one a gap rather than letting it vanish.
    """
    marks: dict[int, ast.expr] = {
        id(node): node
        for node in nodes
        if isinstance(node, ast.Attribute) and _spelled_marker(node, aliases) == _DEFECT_MARKER
    }
    for node in nodes:
        if isinstance(node, ast.Call) and id(node.func) in marks:
            marks[id(node.func)] = node
    return sorted(marks.values(), key=lambda mark: (mark.lineno, mark.col_offset))


def _as_disabling(name: str | None) -> DisablingMark | None:
    return DisablingMark(name) if name in _DISABLING_MARKS else None


_Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _scope_nodes(body: Sequence[ast.stmt]) -> Iterator[ast.AST]:
    """Every node a scope *body* runs in that scope itself: nested definitions' and lambdas' bodies excluded."""
    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _Definition | ast.Lambda):
            stack.extend(ast.iter_child_nodes(node))


def _binding_targets(statement: ast.stmt) -> list[ast.expr]:
    if isinstance(statement, ast.Assign):
        return statement.targets
    if isinstance(statement, ast.AnnAssign):
        return [statement.target]
    return []


def _pytestmark_is_static(body: Sequence[ast.stmt]) -> bool:
    """Whether a scope names ``pytestmark`` only as the target of its direct-child plain or annotated bindings.

    A binding under ``if``/``try``/``with``, an augmented one, an ``append``/``extend`` or any other use
    leaves the value pytest reads unknowable statically.
    """
    direct_targets = {id(target) for statement in body for target in _binding_targets(statement)}
    return all(
        id(node) in direct_targets
        for node in _scope_nodes(body)
        if isinstance(node, ast.Name) and node.id == _PYTESTMARK
    )


@dataclass(frozen=True)
class _MarkReader:
    """Which marker a module's mark element names, through its aliases and its names bound to one mark.

    A decorator naming no mark is taken for an ordinary decorator, so a skip spelled by an imported
    helper there is invisible; a ``pytestmark`` element holds only marks, so one naming no known
    marker there is UNRESOLVED.
    """

    aliases: frozenset[str]
    bound: Mapping[str, str]

    @classmethod
    def of_module(cls, nodes: Iterable[ast.AST], aliases: frozenset[str]) -> _MarkReader:
        bound: dict[str, str] = {}
        for node in nodes:
            if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
                marker = _spelled_marker(node.value, aliases)
                if marker is not None:
                    targets = _binding_targets(node)
                    bound.update((target.id, marker) for target in targets if isinstance(target, ast.Name))
        return cls(aliases, MappingProxyType(bound))

    def _marker(self, element: ast.expr) -> str | None:
        spelled = element.func if isinstance(element, ast.Call) else element
        if isinstance(spelled, ast.Name):
            return self.bound.get(spelled.id)
        return _spelled_marker(spelled, self.aliases)

    def decorator_disabling(self, decorators: Sequence[ast.expr]) -> frozenset[DisablingMark]:
        return frozenset(filter(None, (_as_disabling(self._marker(decorator)) for decorator in decorators)))

    def pytestmark_disabling(self, body: Sequence[ast.stmt]) -> frozenset[DisablingMark]:
        markers = [self._marker(element) for element in pytestmark_elements(body)]
        disabled_by = {_as_disabling(marker) for marker in markers}
        if None in markers or not _pytestmark_is_static(body):
            disabled_by.add(DisablingMark.UNRESOLVED)
        return frozenset(filter(None, disabled_by))


@dataclass(frozen=True)
class _ScopeMarks:
    """The marks that reach a scope from it and every scope enclosing it."""

    defect_marks: tuple[ast.expr, ...] = ()
    disabled_by: frozenset[DisablingMark] = frozenset()

    def with_elements(
        self, elements: Sequence[ast.expr], disabled_by: frozenset[DisablingMark]
    ) -> _ScopeMarks:
        defect_marks = tuple(element for element in elements if marker_name(element) == _DEFECT_MARKER)
        return _ScopeMarks(self.defect_marks + defect_marks, self.disabled_by | disabled_by)


def _winning_definitions(body: Sequence[ast.stmt]) -> list[_Definition]:
    """The scope's direct-child definitions, the last one of each name winning its first's place."""
    definitions: dict[str, _Definition] = {}
    for statement in body:
        if isinstance(statement, _Definition):
            definitions[statement.name] = statement
    return list(definitions.values())


def _classes(definitions: Iterable[_Definition]) -> dict[str, ast.ClassDef]:
    return {definition.name: definition for definition in definitions if isinstance(definition, ast.ClassDef)}


def _depth_first(
    node: ast.ClassDef, classes: Mapping[str, ast.ClassDef], visiting: frozenset[str]
) -> Iterator[ast.ClassDef]:
    yield node
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id not in visiting and base.id in classes:
            yield from _depth_first(classes[base.id], classes, visiting | {base.id})


def _lineage(node: ast.ClassDef, classes: Mapping[str, ast.ClassDef]) -> tuple[ast.ClassDef, ...]:
    """*node* and each base *classes* defines, once each in MRO order: the static stand-in for its MRO.

    Each class keeps its last depth-first place, so a diamond's shared base follows both its
    subclasses, as C3 places it.
    """
    walk = tuple(_depth_first(node, classes, frozenset({node.name})))
    return tuple(reversed(dict.fromkeys(reversed(walk))))


def _defines_constructor(node: ast.ClassDef) -> bool:
    return any(
        isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef) and statement.name in _CONSTRUCTORS
        for statement in node.body
    )


def _test_flag(bodies: Iterable[Sequence[ast.stmt]]) -> bool:
    """pytest's ``__test__``: the first of *bodies* to bind it decides, and only a truthy constant collects."""
    for body in bodies:
        value = scope_binding(body, _TEST_FLAG)
        if value is not None:
            return isinstance(value, ast.Constant) and bool(value.value)
    return True


def _is_collected(lineage: tuple[ast.ClassDef, ...]) -> bool:
    """Whether pytest collects a ``Test*`` class of *lineage*: no constructor in it and a truthy ``__test__``."""
    if any(_defines_constructor(ancestor) for ancestor in lineage):
        return False
    return _test_flag(ancestor.body for ancestor in lineage)


def _lineage_marks(
    lineage: tuple[ast.ClassDef, ...], scope: _ScopeMarks, reader: _MarkReader
) -> _ScopeMarks:
    """*scope* plus every lineage class's own marks: its pytestmark binding and its decorators."""
    for ancestor in lineage:
        disabled_by = reader.pytestmark_disabling(ancestor.body) | reader.decorator_disabling(ancestor.decorator_list)
        scope = scope.with_elements([*pytestmark_elements(ancestor.body), *ancestor.decorator_list], disabled_by)
    return scope


@dataclass(frozen=True)
class _Module:
    """What every scope of one module resolves against: its classes and its mark reader."""

    classes: Mapping[str, ast.ClassDef]
    reader: _MarkReader


@dataclass(frozen=True)
class _Member:
    """A definition a scope holds, with the classes its bases resolve among."""

    definition: _Definition
    classes: Mapping[str, ast.ClassDef]


def _members(bodies: Iterable[Sequence[ast.stmt]], module_classes: Mapping[str, ast.ClassDef]) -> list[_Member]:
    """The definitions a scope merged from *bodies* holds: the first body to define a name keeps it.

    A member's bases resolve among the module's classes and those of the body defining it.
    """
    members: dict[str, _Member] = {}
    for body in bodies:
        definitions = _winning_definitions(body)
        classes = {**module_classes, **_classes(definitions)}
        for definition in definitions:
            members.setdefault(definition.name, _Member(definition, classes))
    return list(members.values())


_Item = tuple[str, _ScopeMarks]


def _scope_items(
    members: Iterable[_Member],
    scope: _ScopeMarks,
    class_chain: tuple[str, ...],
    module: _Module,
) -> Iterator[_Item]:
    """Each item pytest's default collection takes from a scope's *members*, named, with the marks reaching it."""
    for member in members:
        definition = member.definition
        if isinstance(definition, ast.ClassDef):
            yield from _class_items(definition, member.classes, scope, class_chain, module)
        elif definition.name.startswith(_TEST_FUNCTION_PREFIX):
            name = '::'.join((*class_chain, definition.name))
            decorators = definition.decorator_list
            yield name, scope.with_elements(decorators, module.reader.decorator_disabling(decorators))


def _class_items(
    node: ast.ClassDef,
    classes: Mapping[str, ast.ClassDef],
    enclosing: _ScopeMarks,
    class_chain: tuple[str, ...],
    module: _Module,
) -> Iterator[_Item]:
    """The items of a collected Test class: its lineage's members and marks, named under its own chain."""
    if not node.name.startswith(_TEST_CLASS_PREFIX):
        return
    lineage = _lineage(node, classes)
    if not _is_collected(lineage):
        return
    members = _members((ancestor.body for ancestor in lineage), module.classes)
    scope = _lineage_marks(lineage, enclosing, module.reader)
    yield from _scope_items(members, scope, (*class_chain, node.name), module)


def _module_items(body: Sequence[ast.stmt], reader: _MarkReader) -> Iterator[_Item]:
    if not _test_flag((body,)):
        return iter(())
    module = _Module(_classes(_winning_definitions(body)), reader)
    scope = _ScopeMarks().with_elements(pytestmark_elements(body), reader.pytestmark_disabling(body))
    return _scope_items(_members((body,), module.classes), scope, (), module)


_Reading = str | EnumerationGap


def _carriers(
    items: Iterator[_Item],
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
    nodes = list(ast.walk(tree))
    aliases = _mark_namespace_aliases(nodes)
    marks = _defect_marks(nodes, aliases)
    readings = {id(mark): _read_ref(mark) for mark in marks}
    reader = _MarkReader.of_module(nodes, aliases)
    carriers, reached = _carriers(_module_items(tree.body, reader), readings, node_of)
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

    @cached_property
    def carriers(self) -> Mapping[str, tuple[Carrier, ...]]:
        """Every file's carriers merged per ref: refs sorted, carriers in path then source order."""
        merged: dict[str, list[Carrier]] = {}
        for file in self.files:
            for ref, carriers in file.carriers.items():
                merged.setdefault(ref, []).extend(carriers)
        return MappingProxyType({ref: tuple(merged[ref]) for ref in sorted(merged)})

    @cached_property
    def runnable_refs(self) -> frozenset[str]:
        return frozenset(
            ref for ref, carriers in self.carriers.items() if any(carrier.runnable for carrier in carriers)
        )

    @cached_property
    def non_static_refs(self) -> frozenset[str]:
        return frozenset(mark.ref for file in self.files for mark in file.non_static)

    @cached_property
    def incomplete_files(self) -> tuple[str, ...]:
        """The path of every incomplete entry, an unlistable directory's included."""
        return tuple(file.path for file in self.files if not file.complete)

    @property
    def complete(self) -> bool:
        return not self.incomplete_files


def is_test_file(path: str) -> bool:
    """Whether pytest's default python_files collects the file at *path*."""
    name = posixpath.basename(path)
    return any(fnmatchcase(name, pattern) for pattern in _PYTHON_FILES)


def _is_virtualenv(directory: Path) -> bool:
    try:
        return any((directory / marker).is_file() for marker in _VIRTUALENV_MARKERS)
    except OSError:
        return False


def _is_pruned(directory: Path) -> bool:
    return any(fnmatchcase(directory.name, pattern) for pattern in _NORECURSE_DIRS) or _is_virtualenv(directory)


@dataclass(frozen=True)
class _Walk:
    """The tree-relative POSIX paths of a walk's test files, and of each directory it could not cover."""

    test_files: tuple[str, ...]
    directory_gaps: tuple[tuple[str, GapReason], ...]


def _walk(root: Path) -> _Walk:
    """Every test file under *root*, following directory symlinks as pytest does; both parts sorted.

    A subdirectory that cannot be listed is UNLISTABLE_DIRECTORY, and one whose real path was
    already listed (a symlink cycle or a second link to it) is REVISITED_DIRECTORY rather than
    walked twice. Only an unlistable *root* raises.
    """
    test_files: list[str] = []
    directory_gaps: list[tuple[str, GapReason]] = []
    listed: set[str] = set()

    def note_unlistable(error: OSError) -> None:
        directory = Path(error.filename).relative_to(root)
        if directory == Path():
            raise error
        directory_gaps.append((directory.as_posix(), GapReason.UNLISTABLE_DIRECTORY))

    for directory, directory_names, file_names in os.walk(root, onerror=note_unlistable, followlinks=True):
        current = Path(directory)
        relative = current.relative_to(root)
        real = os.path.realpath(current)
        if real in listed:
            directory_names[:] = []
            directory_gaps.append((relative.as_posix(), GapReason.REVISITED_DIRECTORY))
            continue
        listed.add(real)
        directory_names[:] = sorted(name for name in directory_names if not _is_pruned(current / name))
        test_files.extend((relative / name).as_posix() for name in file_names if is_test_file(name))
    return _Walk(tuple(sorted(test_files)), tuple(sorted(directory_gaps)))


def _enumerate_file(root: Path, path: str, module_root: str) -> FileEnumeration:
    try:
        source = (root / path).read_bytes()
    except OSError:
        return FileEnumeration(path, {}, gaps=(EnumerationGap(GapReason.UNREADABLE_FILE, None),))
    return enumerate_source(path, source, module_root=module_root)


def _directory_gap(path: str, reason: GapReason) -> FileEnumeration:
    return FileEnumeration(path, {}, gaps=(EnumerationGap(reason, None),))


def enumerate_tree(root: Path, *, module_root_of: Callable[[str], str]) -> TreeEnumeration:
    """Enumerate every test file pytest's default rules would collect under *root*.

    *module_root_of* maps each tree-relative path to the module root that qualifies its NodeIds.
    Directory symlinks are followed. A file that cannot be read is an UNREADABLE_FILE gap, and a
    subdirectory that cannot be listed or that a symlink revisits is a directory gap, so the walk
    always covers the rest; only an unlistable *root* raises.
    """
    walk = _walk(root)
    files = (_enumerate_file(root, path, module_root_of(path)) for path in walk.test_files)
    return TreeEnumeration((*files, *(_directory_gap(path, reason) for path, reason in walk.directory_gaps)))


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
