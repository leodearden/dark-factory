"""Static census of the escalation categories a source tree files.

Walks every non-test ``.py`` under the given roots and resolves the category
passed to ``Escalation``, ``has_open_l1``, ``find_terminal_by_citation`` and
every function that forwards its own ``category`` parameter to one of them,
following values through constants, imports, local and tuple-unpacked
assignments, and the returns of functions defined in the same module. It
cannot see fully dynamic values (subscripts, calls into other modules); only
the submit-time check in ``escalation.categories`` catches those.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from escalation.categories import EscalationCategory

_SKIPPED_DIRS = frozenset({'tests', '.venv', 'node_modules', '__pycache__'})
_SEEDS: Mapping[str, int | None] = {
    'Escalation': 4,
    'has_open_l1': None,
    'find_terminal_by_citation': 1,
}
_ESCALATION_CLASS = 'escalation.models.Escalation'
_REGISTRY_CLASS = 'escalation.categories.EscalationCategory'
_MAX_DEPTH = 16

_Function = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class CensusSite:
    path: Path
    line: int
    value: str


@dataclass(frozen=True)
class CensusResult:
    sites: tuple[CensusSite, ...]
    wrappers: frozenset[str]
    parsed_paths: tuple[Path, ...]
    parse_failures: tuple[str, ...]

    @property
    def files_parsed(self) -> int:
        return len(self.parsed_paths)


@dataclass(frozen=True)
class _ModuleIndex:
    path: Path
    module: str | None
    constants: Mapping[str, ast.expr]
    class_names: frozenset[str]
    class_constants: Mapping[str, ast.expr]
    imports: Mapping[str, str]
    functions: tuple[_Function, ...]
    calls: tuple[tuple[_Function | None, ast.Call], ...]


@dataclass(frozen=True)
class _CategoryParam:
    index: int | None
    default: ast.expr | None


@dataclass(frozen=True)
class _Bound:
    """What a local name is bound to: *value* itself, or its *element*-th item."""

    value: ast.expr
    element: int | None = None


@dataclass(frozen=True)
class _Wrapper:
    owner: _ModuleIndex
    function: _Function
    param: _CategoryParam


def _iter_sources(roots: Sequence[Path]) -> Iterator[tuple[Path, str | None, bool]]:
    for root in roots:
        for path in sorted(root.rglob('*.py')):
            rel = path.relative_to(root).parts
            if _SKIPPED_DIRS.intersection(rel[:-1]) or path.name.startswith('test_'):
                continue
            yield path, _module_name((root.name, *rel)), path.name == '__init__.py'


def _module_name(parts: tuple[str, ...]) -> str | None:
    if 'src' not in parts[:-1]:
        return None
    dotted = [*parts[parts.index('src') + 1:-1], parts[-1].removesuffix('.py')]
    if dotted[-1] == '__init__':
        dotted.pop()
    return '.'.join(dotted) or None


def _absolute_module(index_module: str | None, is_package: bool, node: ast.ImportFrom) -> str:
    if node.level == 0 or index_module is None:
        return node.module or ''
    package = index_module.split('.')
    if not is_package:
        package.pop()
    package = package[: len(package) - (node.level - 1)]
    return '.'.join([*package, *([node.module] if node.module else [])])


def _assigned(statement: ast.stmt) -> Iterator[tuple[str, ast.expr]]:
    if isinstance(statement, ast.Assign):
        for target in statement.targets:
            if isinstance(target, ast.Name):
                yield target.id, statement.value
    elif (isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name)
          and statement.value is not None):
        yield statement.target.id, statement.value


def _import_bindings(node: ast.Import | ast.ImportFrom, module: str | None,
                     is_package: bool) -> Iterator[tuple[str, str]]:
    if isinstance(node, ast.ImportFrom):
        base = _absolute_module(module, is_package, node)
        for alias in node.names:
            yield alias.asname or alias.name, f'{base}.{alias.name}'
        return
    for alias in node.names:
        head = alias.name.split('.')[0]
        yield (alias.asname, alias.name) if alias.asname else (head, head)


def _index(path: Path, module: str | None, is_package: bool, tree: ast.Module) -> _ModuleIndex:
    """One traversal; every call is paired with its innermost enclosing function."""
    classes: list[ast.ClassDef] = []
    functions: list[_Function] = []
    imports: dict[str, str] = {}
    calls: list[tuple[_Function | None, ast.Call]] = []
    pending: list[tuple[ast.AST, _Function | None]] = [(tree, None)]
    while pending:
        node, function = pending.pop()
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call):
                calls.append((function, child))
            elif isinstance(child, ast.ClassDef):
                classes.append(child)
            elif isinstance(child, _Function):
                functions.append(child)
            elif isinstance(child, ast.Import | ast.ImportFrom):
                imports.update(_import_bindings(child, module, is_package))
            pending.append((child, child if isinstance(child, _Function) else function))
    return _ModuleIndex(
        path=path,
        module=module,
        constants=dict(pair for stmt in tree.body for pair in _assigned(stmt)),
        class_names=frozenset(cls.name for cls in classes),
        class_constants=dict(pair for cls in classes for stmt in cls.body for pair in _assigned(stmt)),
        imports=imports,
        functions=tuple(functions),
        calls=tuple(calls),
    )


def _resolve(expr: ast.expr, index: _ModuleIndex, function: _Function | None,
             modules: Mapping[str, _ModuleIndex], depth: int = 0) -> frozenset[str]:
    if depth > _MAX_DEPTH:
        return frozenset()
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return frozenset({expr.value})
    if isinstance(expr, ast.Name):
        return _resolve_name(expr.id, index, function, modules, depth + 1)
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        return _resolve_attribute(expr.value.id, expr.attr, index, modules, depth + 1)
    if isinstance(expr, ast.Call):
        return _resolve_returns(expr, None, index, modules, depth + 1)
    return frozenset()


def _local_bindings(function: _Function, name: str) -> list[_Bound]:
    bindings: list[_Bound] = []
    for node in ast.walk(function):
        if isinstance(node, ast.stmt):
            bindings.extend(_Bound(value) for bound, value in _assigned(node) if bound == name)
        if isinstance(node, ast.Assign):
            bindings.extend(
                _Bound(node.value, i)
                for target in node.targets if isinstance(target, ast.Tuple)
                for i, item in enumerate(target.elts)
                if isinstance(item, ast.Name) and item.id == name
            )
    return bindings


def _resolve_bound(bound: _Bound, index: _ModuleIndex, function: _Function | None,
                   modules: Mapping[str, _ModuleIndex], depth: int) -> frozenset[str]:
    if bound.element is None:
        return _resolve(bound.value, index, function, modules, depth)
    if isinstance(bound.value, ast.Tuple) and bound.element < len(bound.value.elts):
        return _resolve(bound.value.elts[bound.element], index, function, modules, depth)
    if isinstance(bound.value, ast.Call):
        return _resolve_returns(bound.value, bound.element, index, modules, depth + 1)
    return frozenset()


def _resolve_returns(call: ast.Call, element: int | None, index: _ModuleIndex,
                     modules: Mapping[str, _ModuleIndex], depth: int) -> frozenset[str]:
    name = _callee_name(call)
    returned = [(fn, node.value) for fn in index.functions if fn.name == name
                for node in ast.walk(fn) if isinstance(node, ast.Return) and node.value is not None]
    return frozenset().union(*(
        _resolve_bound(_Bound(value, element), index, fn, modules, depth) for fn, value in returned
    ))


def _resolve_name(name: str, index: _ModuleIndex, function: _Function | None,
                  modules: Mapping[str, _ModuleIndex], depth: int) -> frozenset[str]:
    local = _local_bindings(function, name) if function is not None else []
    if local:
        return frozenset().union(*(_resolve_bound(b, index, function, modules, depth) for b in local))
    if name in index.constants:
        return _resolve(index.constants[name], index, None, modules, depth)
    owner, _, attr = index.imports.get(name, '').rpartition('.')
    if owner in modules:
        return _resolve_name(attr, modules[owner], None, modules, depth + 1)
    return frozenset()


def _resolve_attribute(base: str, attr: str, index: _ModuleIndex,
                       modules: Mapping[str, _ModuleIndex], depth: int) -> frozenset[str]:
    target = index.imports.get(base, '')
    if base == 'EscalationCategory' or target == _REGISTRY_CLASS:
        member = EscalationCategory.__members__.get(attr)
        return frozenset({member.value}) if member is not None else frozenset()
    if base in ('self', 'cls') or base in index.class_names:
        value = index.class_constants.get(attr)
        return frozenset() if value is None else _resolve(value, index, None, modules, depth)
    if target in modules:
        return _resolve_name(attr, modules[target], None, modules, depth)
    owner, _, cls = target.rpartition('.')
    if owner in modules and cls in modules[owner].class_names:
        return _resolve_attribute(cls, attr, modules[owner], modules, depth + 1)
    return frozenset()


def _callee_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _category_argument(call: ast.Call, position: int | None) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == 'category':
            return keyword.value
    if position is None or len(call.args) <= position:
        return None
    if any(isinstance(arg, ast.Starred) for arg in call.args[: position + 1]):
        return None
    return call.args[position]


def _known_call_category(call: ast.Call, known: Mapping[str, int | None]) -> ast.expr | None:
    name = _callee_name(call)
    if name is None or name not in known:
        return None
    return _category_argument(call, known[name])


def _category_param(function: _Function) -> _CategoryParam | None:
    args = function.args
    positional = [*args.posonlyargs, *args.args]
    offset = 1 if positional and positional[0].arg in ('self', 'cls') else 0
    first_default = len(positional) - len(args.defaults)
    for i, arg in enumerate(positional):
        if arg.arg == 'category':
            default = args.defaults[i - first_default] if i >= first_default else None
            return _CategoryParam(index=i - offset, default=default)
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        if arg.arg == 'category':
            return _CategoryParam(index=None, default=default)
    return None


def _forwards_category(function: _Function, known: Mapping[str, int | None]) -> bool:
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            argument = _known_call_category(node, known)
            if isinstance(argument, ast.Name) and argument.id == 'category':
                return True
    return False


def _discover_wrappers(indexes: Sequence[_ModuleIndex]) -> tuple[dict[str, int | None], list[_Wrapper]]:
    """Fixed point from the seeds. Wrappers are keyed by bare function name, so a
    same-named def anywhere in the tree is treated as the wrapper: an
    over-approximation that can only add sites, never hide one."""
    known: dict[str, int | None] = dict(_SEEDS)
    for index in indexes:
        known.update({alias: _SEEDS['Escalation'] for alias, target in index.imports.items()
                      if target == _ESCALATION_CLASS})
    pending = [_Wrapper(index, fn, param) for index in indexes for fn in index.functions
               if (param := _category_param(fn)) is not None]
    wrappers: list[_Wrapper] = []
    progressed = True
    while progressed:
        progressed = False
        for candidate in [c for c in pending if _forwards_category(c.function, known)]:
            pending.remove(candidate)
            wrappers.append(candidate)
            known.setdefault(candidate.function.name, candidate.param.index)
            progressed = True
    return known, wrappers


def _call_sites(index: _ModuleIndex, known: Mapping[str, int | None],
                modules: Mapping[str, _ModuleIndex]) -> Iterator[CensusSite]:
    for function, call in index.calls:
        argument = _known_call_category(call, known)
        if argument is not None:
            for value in _resolve(argument, index, function, modules):
                yield CensusSite(index.path, call.lineno, value)


def _default_sites(wrappers: Sequence[_Wrapper],
                   modules: Mapping[str, _ModuleIndex]) -> Iterator[CensusSite]:
    for wrapper in wrappers:
        default = wrapper.param.default
        if default is not None:
            for value in _resolve(default, wrapper.owner, None, modules):
                yield CensusSite(wrapper.owner.path, default.lineno, value)


def run_census(roots: Sequence[Path]) -> CensusResult:
    indexes: list[_ModuleIndex] = []
    failures: list[str] = []
    for path, module, is_package in _iter_sources(roots):
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            failures.append(str(path))
            continue
        indexes.append(_index(path, module, is_package, tree))
    modules = {index.module: index for index in indexes if index.module}
    known, wrappers = _discover_wrappers(indexes)
    sites = [site for index in indexes for site in _call_sites(index, known, modules)]
    sites.extend(_default_sites(wrappers, modules))
    return CensusResult(
        sites=tuple(sites),
        wrappers=frozenset(w.function.name for w in wrappers),
        parsed_paths=tuple(index.path for index in indexes),
        parse_failures=tuple(failures),
    )
