"""Part 2 of the suite census for Python trees (task 5414): how test files pin implementation detail.

Whole-tree and report-only: every tracked ``.py`` file under a ``tests``
directory, grouped by package (its first path segment), measured with the
merge-lane ratchet's own measures wherever one exists. The output is a count
per package, never a ranking.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from source_measures import (
    MetricsError,
    PatchCall,
    file_size_measures_in_tree,
    patch_calls_in_tree,
    private_reads_in_tree,
    src_module_name,
    tracked_files,
)

_PROSE_MIN_WORDS = 6
_QUOTE_MIN_CHARS = 3
_PROSE_COMPARE_OPS = (ast.In, ast.NotIn, ast.Eq, ast.NotEq)
_UNREADABLE = (SyntaxError, ValueError, MetricsError, OSError)
_SCRIPT_DIRS = ('scripts', 'scripts/legibility')
_TOTAL = 'total'

_ADDITIVE = (
    'test_files', 'lines', 'prose_lines', 'test_functions', 'private_reads',
    'tests_with_private_patch', 'private_patch_sites_outside_tests',
    'prose_assertion_sites', 'tests_with_prose_assertion',
)


@dataclass(frozen=True)
class PackageRow:
    package: str
    test_files: int
    lines: int
    prose_lines: int
    test_functions: int
    private_reads: int
    tests_with_private_patch: int
    private_patch_sites_outside_tests: int
    distinct_private_targets: frozenset[str]
    prose_assertion_sites: int
    tests_with_prose_assertion: int
    exact_dup_groups: int
    exact_dup_redundant: int
    structural_dup_groups: int
    structural_dup_redundant: int
    largest_structural_group: tuple[int, str] | None


@dataclass(frozen=True)
class PythonPinningCensus:
    rows: tuple[PackageRow, ...]
    totals: PackageRow
    unreadable: tuple[str, ...]
    complete: bool


# ---------------------------------------------------------------------------
# What is first-party, and where its modules live.

@dataclass(frozen=True)
class _FirstParty:
    roots: frozenset[str]
    modules: MappingProxyType[str, str]

    @classmethod
    def of(cls, tracked: Sequence[str]) -> _FirstParty:
        modules: dict[str, str] = {}
        for path in tracked:
            name = src_module_name(path)
            if name is not None:
                modules[name.removesuffix('.__init__')] = path
            elif path.rpartition('/')[0] in _SCRIPT_DIRS:
                modules[path.rpartition('/')[2].removesuffix('.py')] = path
        roots = {
            name.split('.', 1)[0] for name, path in modules.items()
            if path.endswith('/__init__.py') and '.' not in name
        }
        roots.update(
            name for name, path in modules.items() if path.rpartition('/')[0] in _SCRIPT_DIRS
        )
        return cls(roots=frozenset(roots), modules=MappingProxyType(modules))

    def owns(self, dotted: str) -> bool:
        return dotted.split('.', 1)[0] in self.roots


class _ProseIndex:
    """Module-level prose string constants of first-party modules, parsed once per census."""

    def __init__(self, tree_root: Path, first_party: _FirstParty) -> None:
        self._tree_root = tree_root
        self._first_party = first_party
        self._cache: dict[str, tuple[str, ...]] = {}
        self.unreadable: set[str] = set()

    def constants(self, modules: Iterable[str]) -> tuple[str, ...]:
        return tuple(
            constant for module in sorted(modules) for constant in self._of(module)
        )

    def _of(self, module: str) -> tuple[str, ...]:
        path = self._first_party.modules[module]
        if path not in self._cache:
            try:
                tree = ast.parse((self._tree_root / path).read_text(encoding='utf-8'))
            except _UNREADABLE:
                self.unreadable.add(path)
                tree = ast.Module(body=[], type_ignores=[])
            self._cache[path] = _prose_constants(tree)
        return self._cache[path]


def _prose_constants(tree: ast.Module) -> tuple[str, ...]:
    values = (
        node.value for node in tree.body
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name))
        or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name))
    )
    return tuple(
        value.value for value in values
        if isinstance(value, ast.Constant) and isinstance(value.value, str)
        and len(value.value.split()) >= _PROSE_MIN_WORDS
    )


def _imported_modules(tree: ast.Module, first_party: _FirstParty) -> set[str]:
    named: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            named.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            named.add(node.module)
            named.update(f'{node.module}.{alias.name}' for alias in node.names)
    return {name for name in named if name in first_party.modules}


def _import_bindings(tree: ast.Module) -> dict[str, str]:
    """Module-level names bound by an import, to the dotted name each binds."""
    bindings: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split('.', 1)[0]
                bindings[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f'{node.module}.{alias.name}'
    return bindings


# ---------------------------------------------------------------------------
# Private patches.

def _is_private_name(name: str) -> bool:
    return name.startswith('_') and not name.startswith('__')


def _root_name(expr: ast.expr | None) -> str | None:
    while isinstance(expr, ast.Attribute | ast.Subscript | ast.Call):
        expr = expr.func if isinstance(expr, ast.Call) else expr.value
    return expr.id if isinstance(expr, ast.Name) else None


@dataclass(frozen=True)
class _PatchJudge:
    """Which patch calls in one file target a first-party private name."""

    first_party: _FirstParty
    bindings: Mapping[str, str]

    def private_calls(self, node: ast.AST) -> list[PatchCall]:
        return [call for call in patch_calls_in_tree(node) if self._is_private(call)]

    def target(self, call: PatchCall) -> str | None:
        if call.target is not None:
            return call.target
        if isinstance(call.receiver, ast.Name):
            module = self.bindings.get(call.receiver.id)
            if module in self.first_party.modules:
                return f'{module}.{call.attribute}'
        return None

    def _is_private(self, call: PatchCall) -> bool:
        if call.target is not None:
            head, *rest = call.target.split('.')
            return head in self.first_party.roots and any(map(_is_private_name, rest))
        bound = self.bindings.get(_root_name(call.receiver) or '')
        foreign = bound is not None and not self.first_party.owns(bound)
        return _is_private_name(call.attribute or '') and not foreign


# ---------------------------------------------------------------------------
# Test functions, prose-constant assertions and duplicate bodies.

@dataclass(frozen=True)
class _TestFunction:
    qualname: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    class_decorators: tuple[ast.expr, ...]


def _is_test_module(path: str) -> bool:
    name = path.rpartition('/')[2]
    return name.startswith('test_') or name.endswith('_test.py')


def _test_functions(
    tree: ast.Module, path: str,
) -> tuple[list[_TestFunction], list[ast.ClassDef]]:
    functions: list[_TestFunction] = []
    classes: list[ast.ClassDef] = []

    def visit(body: Sequence[ast.stmt], prefix: str, decorators: tuple[ast.expr, ...]) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith('test'):
                functions.append(_TestFunction(f'{prefix}::{node.name}', node, decorators))
            elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                classes.append(node)
                visit(node.body, f'{prefix}::{node.name}', (*decorators, *node.decorator_list))

    visit(tree.body, path, ())
    return functions, classes


def _quotes_prose(operand: ast.expr, constants: Sequence[str]) -> bool:
    return (
        isinstance(operand, ast.Constant) and isinstance(operand.value, str)
        and len(operand.value) >= _QUOTE_MIN_CHARS
        and any(operand.value in constant for constant in constants)
    )


def _is_prose_assertion(node: ast.Assert, constants: Sequence[str]) -> bool:
    return any(
        isinstance(compare, ast.Compare)
        and any(isinstance(op, _PROSE_COMPARE_OPS) for op in compare.ops)
        and any(_quotes_prose(operand, constants) for operand in (compare.left, *compare.comparators))
        for compare in ast.walk(node.test)
    )


def _prose_assertions(node: ast.AST, constants: Sequence[str]) -> int:
    if not constants:
        return 0
    return sum(
        1 for child in ast.walk(node)
        if isinstance(child, ast.Assert) and _is_prose_assertion(child, constants)
    )


class _EraseNames(ast.NodeTransformer):
    """Erases identifiers and constant values, leaving the body's shape."""

    def visit_Name(self, node: ast.Name) -> ast.Name:
        node.id = '_'
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        node.value = None
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        self.generic_visit(node)
        node.attr = '_'
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.keyword:
        self.generic_visit(node)
        if node.arg is not None:
            node.arg = '_'
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        self.generic_visit(node)
        node.arg = '_'
        return node

    def _erase_definition(self, node: ast.AST) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            node.name = '_'
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _erase_definition


def _digest(dump: str) -> bytes:
    return hashlib.blake2b(dump.encode(), digest_size=16).digest()


def _duplicate_keys(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[bytes, bytes]:
    """(exact, structural) keys of a test body: its name and decorators never count."""
    clone = copy.deepcopy(node)
    clone.decorator_list = []
    clone.name = '_'
    exact = _digest(ast.dump(clone))
    return exact, _digest(ast.dump(_EraseNames().visit(clone)))


# ---------------------------------------------------------------------------
# One file, then one package.

@dataclass(frozen=True)
class _FileMeasures:
    test_files: int
    lines: int
    prose_lines: int
    test_functions: int
    private_reads: int
    tests_with_private_patch: int
    private_patch_sites_outside_tests: int
    prose_assertion_sites: int
    tests_with_prose_assertion: int
    distinct_private_targets: frozenset[str]
    exact_keys: tuple[tuple[bytes, str], ...]
    structural_keys: tuple[tuple[bytes, str], ...]


def _measure_file(
    tree_root: Path, path: str, first_party: _FirstParty, prose: _ProseIndex,
) -> _FileMeasures:
    source = (tree_root / path).read_text(encoding='utf-8')
    module = ast.parse(source)
    size = file_size_measures_in_tree(source, module, path=path)
    judge = _PatchJudge(first_party, _import_bindings(module))
    constants = prose.constants(_imported_modules(module, first_party))
    functions, classes = _test_functions(module, path) if _is_test_module(path) else ([], [])
    file_private = judge.private_calls(module)
    own = [len(judge.private_calls(f.node)) for f in functions]
    inherited = [sum(len(judge.private_calls(d)) for d in f.class_decorators) for f in functions]
    in_tests = sum(own) + sum(
        len(judge.private_calls(d)) for cls in classes for d in cls.decorator_list
    )
    keys = [_duplicate_keys(f.node) for f in functions]
    return _FileMeasures(
        test_files=1, lines=size.lines, prose_lines=size.prose_lines,
        test_functions=len(functions), private_reads=private_reads_in_tree(module),
        tests_with_private_patch=sum(1 for o, i in zip(own, inherited, strict=True) if o or i),
        private_patch_sites_outside_tests=len(file_private) - in_tests,
        prose_assertion_sites=_prose_assertions(module, constants),
        tests_with_prose_assertion=sum(1 for f in functions if _prose_assertions(f.node, constants)),
        distinct_private_targets=frozenset(
            target for call in file_private if (target := judge.target(call)) is not None
        ),
        exact_keys=tuple((exact, f.qualname) for (exact, _), f in zip(keys, functions, strict=True)),
        structural_keys=tuple((shape, f.qualname) for (_, shape), f in zip(keys, functions, strict=True)),
    )


def _duplicate_groups(keyed: Iterable[tuple[bytes, str]]) -> list[list[str]]:
    members: dict[bytes, list[str]] = {}
    for key, qualname in keyed:
        members.setdefault(key, []).append(qualname)
    return [group for group in members.values() if len(group) > 1]


def _largest(groups: Iterable[list[str]]) -> tuple[int, str] | None:
    return min(((len(g), g[0]) for g in groups), key=lambda s: (-s[0], s[1]), default=None)


def _package_row(package: str, files: Sequence[_FileMeasures]) -> PackageRow:
    exact = _duplicate_groups(key for f in files for key in f.exact_keys)
    structural = _duplicate_groups(key for f in files for key in f.structural_keys)
    return PackageRow(
        package=package,
        **{name: sum(getattr(f, name) for f in files) for name in _ADDITIVE},
        distinct_private_targets=frozenset().union(*(f.distinct_private_targets for f in files)),
        exact_dup_groups=len(exact), exact_dup_redundant=sum(len(g) - 1 for g in exact),
        structural_dup_groups=len(structural),
        structural_dup_redundant=sum(len(g) - 1 for g in structural),
        largest_structural_group=_largest(structural),
    )


def _totals(rows: Sequence[PackageRow]) -> PackageRow:
    largest = [row.largest_structural_group for row in rows if row.largest_structural_group]
    return PackageRow(
        package=_TOTAL,
        **{name: sum(getattr(row, name) for row in rows) for name in _ADDITIVE},
        distinct_private_targets=frozenset().union(*(row.distinct_private_targets for row in rows)),
        exact_dup_groups=sum(row.exact_dup_groups for row in rows),
        exact_dup_redundant=sum(row.exact_dup_redundant for row in rows),
        structural_dup_groups=sum(row.structural_dup_groups for row in rows),
        structural_dup_redundant=sum(row.structural_dup_redundant for row in rows),
        largest_structural_group=min(largest, key=lambda s: (-s[0], s[1]), default=None),
    )


def _in_test_tree(path: str) -> bool:
    return 'tests' in path.split('/')[:-1]


def measure_python_tree(tree_root: Path) -> PythonPinningCensus:
    tracked = tracked_files(tree_root, '*.py')
    first_party = _FirstParty.of(tracked)
    prose = _ProseIndex(tree_root, first_party)
    by_package: defaultdict[str, list[_FileMeasures]] = defaultdict(list)
    unreadable: set[str] = set()
    for path in filter(_in_test_tree, tracked):
        try:
            by_package[path.split('/', 1)[0]].append(
                _measure_file(tree_root, path, first_party, prose)
            )
        except _UNREADABLE:
            unreadable.add(path)
    unreadable |= prose.unreadable
    rows = tuple(_package_row(package, files) for package, files in sorted(by_package.items()))
    return PythonPinningCensus(
        rows=rows, totals=_totals(rows), unreadable=tuple(sorted(unreadable)),
        complete=not unreadable,
    )


# ---------------------------------------------------------------------------
# Rendering.

_DEFINITIONS = """\
Every tracked `.py` file with a `tests` directory in its path is a test file;
its package is its first path segment. Rows are in package-name order and are
not ranked.

- **test fns**: module-level `test*` functions and `test*` methods of `Test*`
  classes (nested `Test*` classes included), in `test_*.py` / `*_test.py` files.
- **lines / prose lines**: the merge-lane ratchet's `file_size_measures`
  (prose = docstring or comment lines).
- **private reads**: the ratchet's `private_reads` (single-underscore attribute
  accesses, reads and writes, except on bare `self`/`cls`), over every test file.
- **private-patch tests**: test fns holding a patch call (the ratchet's patch
  shapes) whose string target starts with a first-party name and has a later
  `_private` segment, or whose object-form attribute is `_private` and whose
  receiver is not bound to a non-first-party import. A `Test*` class decorator
  counts for each of its test methods. Receivers are not type-resolved, so the
  object form over-counts.
- **patch sites outside tests**: the same calls in fixtures, helpers and module
  scope, counted here rather than attributed to the tests that use them.
- **distinct targets**: string targets plus object-form targets whose receiver
  is a first-party module import.
- **prose asserts**: `assert` statements comparing (`in`, `not in`, `==`, `!=`)
  a string of 3+ characters that occurs inside a module-level string constant
  of 6+ words in a first-party module the file imports. These are candidates
  for hand sampling, not confirmed prose pins.
- **exact / structural duplicates**: test fns with identical AST bodies
  (name and decorators ignored); structural also erases identifiers, attribute
  names, constant values, keyword and parameter names. Grouped within a
  package; `groups/redundant` where redundant = members beyond the first."""


def _cells(row: PackageRow) -> tuple[object, ...]:
    largest = row.largest_structural_group
    return (
        row.package, row.test_files, row.lines, row.prose_lines, row.test_functions,
        row.private_reads, row.tests_with_private_patch, row.private_patch_sites_outside_tests,
        len(row.distinct_private_targets), row.prose_assertion_sites,
        row.tests_with_prose_assertion,
        f'{row.exact_dup_groups}/{row.exact_dup_redundant}',
        f'{row.structural_dup_groups}/{row.structural_dup_redundant}',
        f'{largest[0]}: `{largest[1]}`' if largest else '—',
    )


_HEADER = (
    'package', 'test files', 'lines', 'prose lines', 'test fns', 'private reads',
    'private-patch tests', 'patch sites outside tests', 'distinct targets',
    'prose asserts', 'tests with prose asserts', 'exact dups', 'structural dups',
    'largest structural group',
)


def _row_line(cells: Iterable[object]) -> str:
    return '| ' + ' | '.join(str(cell).replace('|', '\\|') for cell in cells) + ' |'


def render_markdown(census: PythonPinningCensus) -> str:
    table = [_row_line(_HEADER), _row_line('---' for _ in _HEADER)]
    table.extend(_row_line(_cells(row)) for row in (*census.rows, census.totals))
    unreadable = '\n'.join(f'- `{path}`' for path in census.unreadable) or '- none'
    return (
        f'### Python pinning and duplication\n\n{_DEFINITIONS}\n\n' + '\n'.join(table)
        + f'\n\nUnreadable files ({len(census.unreadable)}; the census is '
        + ('complete' if census.complete else 'INCOMPLETE') + f'):\n\n{unreadable}\n'
    )
