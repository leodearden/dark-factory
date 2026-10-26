"""AST scan for tests that assert a CODE DEFAULT off a bare ``OrchestratorConfig()``.

Helper module for :mod:`test_code_default_hatch_guard`.  The ``_`` prefix keeps
it out of pytest collection while remaining importable (``conftest.py`` puts the
tests dir on ``sys.path``), matching ``_orch_helpers.py`` and
``_verify_config_corpus.py``.

WHAT IT LOOKS FOR.  A test is flagged when it (1) builds a *bare*
``OrchestratorConfig()`` — no args, no keywords — either binding a name to it
or asserting directly off the call (``OrchestratorConfig().x.y == literal``),
(2) is not isolated from the ambient operational config, and (3) asserts an
attribute path off that name or call equals a right-hand side that EVALUATES,
under package defaults, to exactly the package default for that path.  A
literal is one such right-hand side; so is numeric arithmetic over evaluable
values, a local bound once to one, another attribute of a bare config, or a
builtin collection constructor over such a value.
Such an assertion passes only because the repo happens NOT to declare that key
in ``dark-factory-orchestrator.yaml``; the autouse ``_isolate_orch_config``
fixture pins that file for every test.  An operator declaring the key turns the
test red without touching any code the test is about (task 3886's incident).

WHAT COUNTS AS ISOLATED.  Three idioms, all honoured:

* ``@pytest.mark.usefixtures("code_default_config")`` on the function or on its
  enclosing class (the sanctioned hatch — detected by substring over the
  unparsed decorator, so any quote spelling works);
* a ``code_default_config`` parameter (requesting the fixture directly);
* the substring ``ORCH_CONFIG_PATH`` anywhere in the unparsed function body,
  which covers the two ad-hoc idioms already widespread in the suite —
  ``monkeypatch.setenv('ORCH_CONFIG_PATH', '')`` (154 uses) and
  ``monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)`` (29 uses).

An ``autouse=True`` fixture defined directly in the test's class or module
counts too, when the fixture itself meets one of those rules.

DELIBERATE ASYMMETRY.  That last rule is a substring test, so it also silences a
test that merely *mentions* the variable — and the ``type()`` match below
silences ``== 1800`` against a ``1800.0`` default.  The evaluator refuses, in
the same direction: a right-hand side reading the asserted path (or a prefix or
extension of it), since both sides track a declaration of that key; a value
reached through a local name unless it is immutable, since a mutation between
binding and assertion is invisible here; a power above
``_MAX_EVALUATED_EXPONENT``; and arithmetic over anything but non-bool numbers.
All of these admit false NEGATIVES and none admits a false POSITIVE.  That is
the trade a *committed* ratchet requires: a guard that occasionally misses a
landmine is a nuisance, while one that occasionally invents one gets disabled.

NOT DETECTED HERE.  Tests that exercise default-only *behaviour* without
asserting any config literal are invisible to a static scan (e.g. a section's
``enabled`` flag defaulting True is what makes a code path run at all).  Those
are found by running the suite against a config that perturbs every undeclared
key; see the task-4481 commit log for that measurement.
"""
from __future__ import annotations

import ast
import operator
import os
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest import mock

from orchestrator.config import OrchestratorConfig

_TestFunc = ast.FunctionDef | ast.AsyncFunctionDef

#: Path guaranteed not to exist, so ``YamlSettingsSource`` skips the layer-2
#: project file and only the package-bundled ``defaults.yaml`` loads.  Same
#: mechanism the ``code_default_config`` fixture uses.
_ABSENT_CONFIG_PATH = '/nonexistent/df-4481-no-such-config.yaml'

#: Marker for "this dotted path does not resolve", distinct from a ``None``
#: default.
_UNRESOLVED = object()


def _load_package_defaults() -> OrchestratorConfig:
    """Build an ``OrchestratorConfig`` from package defaults ONLY.

    Overrides ``ORCH_CONFIG_PATH`` only for the construction, so the scan never
    depends on — and never perturbs — the ambient config the rest of the suite
    relies on.
    """
    with mock.patch.dict(os.environ, {'ORCH_CONFIG_PATH': _ABSENT_CONFIG_PATH}):
        return OrchestratorConfig()


#: Resolved once at import, per the note above.
DEFAULT_CONFIG = _load_package_defaults()


def _resolve_default(dotted: str) -> Any:
    """Resolve *dotted* against :data:`DEFAULT_CONFIG`, or :data:`_UNRESOLVED`."""
    current: Any = DEFAULT_CONFIG
    for part in dotted.split('.'):
        try:
            current = getattr(current, part)
        except AttributeError:
            return _UNRESOLVED
    return current


def _attr_chain(node: ast.expr) -> tuple[ast.expr, str] | None:
    """Unwind ``a.b.c`` into ``(<node a>, 'b.c')``; ``None`` if no attribute."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not parts:
        return None
    return node, '.'.join(reversed(parts))


def _is_bare_config_call(node: ast.AST | None) -> bool:
    """True for exactly ``OrchestratorConfig()`` — no args, no keywords.

    Accepts ``None`` so callers can pass an ``ast.AnnAssign.value`` directly: a
    value-less annotation (``config: OrchestratorConfig``) binds nothing, and
    is therefore not a bare construction.
    """
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == 'OrchestratorConfig'
        and not node.args
        and not node.keywords
    )


def _is_config_root(root: ast.expr, names: frozenset[str]) -> bool:
    """True if *root* is a name bound to a bare config, or a bare call itself."""
    if isinstance(root, ast.Name):
        return root.id in names
    return _is_bare_config_call(root)


def _plain_binding_values(func: _TestFunc) -> dict[ast.Name, ast.expr]:
    """Map each ``name = value`` / ``name: T = value`` target to its value."""
    plain: dict[ast.Name, ast.expr] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    plain[target] = node.value
        elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
              and node.value is not None):
            plain[node.target] = node.value
    return plain


def _opaque_bound_names(node: ast.AST, func: _TestFunc) -> list[str]:
    """Names *node* binds without an ``ast.Name`` target the scan could value."""
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.ExceptHandler) and node.name:
        return [node.name]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(alias.asname or alias.name).split('.')[0] for alias in node.names]
    if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node is not func):
        return [node.name]
    if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
        return [node.name]
    if isinstance(node, ast.MatchMapping) and node.rest:
        return [node.rest]
    return []


def _local_bindings(func: _TestFunc) -> dict[str, list[ast.expr | None]]:
    """Every binding of every name in *func*, nested scopes included.

    One entry per binding: the bound VALUE for a plain ``name = value``, and
    ``None`` for any binding whose value the scan cannot see (parameters,
    unpacking, loop and ``with`` targets, augmented assignment, ``del``...).
    Counting nested scopes only adds bindings, which only ever disqualifies.
    """
    plain = _plain_binding_values(func)
    bindings: dict[str, list[ast.expr | None]] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bindings.setdefault(node.id, []).append(plain.get(node))
        for name in _opaque_bound_names(node, func):
            bindings.setdefault(name, []).append(None)
    return bindings


@dataclass(frozen=True)
class _TestScope:
    """What one test's local bindings let the scan conclude.

    ``bare_configs`` — names whose every binding is a bare
    ``OrchestratorConfig()``.  ``once_bound`` — names bound exactly once, by a
    plain assignment, to that expression.  ``bound`` — every local name.
    """

    bare_configs: frozenset[str]
    once_bound: Mapping[str, ast.expr]
    bound: frozenset[str]


def _test_scope(func: _TestFunc) -> _TestScope:
    bindings = _local_bindings(func)
    return _TestScope(
        bare_configs=frozenset(
            name for name, values in bindings.items()
            if all(_is_bare_config_call(value) for value in values)
        ),
        once_bound=MappingProxyType({
            name: values[0] for name, values in bindings.items()
            if len(values) == 1 and values[0] is not None
        }),
        bound=frozenset(bindings),
    )


@dataclass(frozen=True)
class _Valued:
    """A right-hand side's value under package defaults, and the config paths
    it read to get there."""

    value: Any
    reads: frozenset[str]


_LITERAL_EVAL_ERRORS = (ValueError, TypeError, SyntaxError, MemoryError,
                        RecursionError)

_ARITHMETIC: Mapping[type[ast.operator], Callable[[Any, Any], Any]] = MappingProxyType({
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
})


#: A pathological RHS must not stall the whole-suite ratchet, whose 300s
#: timeout ``os._exit()``s an xdist worker.
_MAX_EVALUATED_EXPONENT = 64

_IMMUTABLE_SCALARS = (bool, int, float, complex, str, bytes, type(None))

_COLLECTION_CONSTRUCTORS: Mapping[str, Callable[..., Any]] = MappingProxyType({
    'frozenset': frozenset,
    'set': set,
    'tuple': tuple,
    'list': list,
})


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_immutable(value: Any) -> bool:
    if isinstance(value, (tuple, frozenset)):
        return all(_is_immutable(member) for member in value)
    return isinstance(value, _IMMUTABLE_SCALARS)


def _evaluate_under_defaults(
    node: ast.expr, scope: _TestScope, resolving: frozenset[str] = frozenset(),
) -> _Valued | None:
    """Value *node* as it would evaluate under package defaults, or ``None``.

    *resolving* holds the local names already being expanded, so a
    self-referential binding (``x = x + 1``) terminates as unevaluable.
    """
    try:
        return _Valued(ast.literal_eval(node), frozenset())
    except _LITERAL_EVAL_ERRORS:
        pass
    if isinstance(node, ast.Attribute):
        return _evaluate_config_attribute(node, scope)
    if isinstance(node, ast.Name):
        return _evaluate_local(node, scope, resolving)
    if isinstance(node, ast.BinOp):
        return _evaluate_arithmetic(node, scope, resolving)
    if isinstance(node, ast.Call):
        return _evaluate_constructor(node, scope, resolving)
    return None


def _evaluate_config_attribute(node: ast.Attribute,
                               scope: _TestScope) -> _Valued | None:
    chain = _attr_chain(node)
    if chain is None:
        return None
    root, path = chain
    if not _is_config_root(root, scope.bare_configs):
        return None
    default = _resolve_default(path)
    if default is _UNRESOLVED:
        return None
    return _Valued(default, frozenset({path}))


def _evaluate_local(node: ast.Name, scope: _TestScope,
                    resolving: frozenset[str]) -> _Valued | None:
    name = node.id
    bound = scope.once_bound.get(name)
    if bound is None or name in scope.bare_configs or name in resolving:
        return None
    valued = _evaluate_under_defaults(bound, scope, resolving | {name})
    if valued is None or not _is_immutable(valued.value):
        return None
    return valued


def _evaluate_arithmetic(node: ast.BinOp, scope: _TestScope,
                         resolving: frozenset[str]) -> _Valued | None:
    apply = _ARITHMETIC.get(type(node.op))
    if apply is None:
        return None
    left = _evaluate_under_defaults(node.left, scope, resolving)
    right = _evaluate_under_defaults(node.right, scope, resolving)
    if left is None or right is None:
        return None
    if not (_is_number(left.value) and _is_number(right.value)):
        return None
    if isinstance(node.op, ast.Pow) and abs(right.value) > _MAX_EVALUATED_EXPONENT:
        return None
    try:
        value = apply(left.value, right.value)
    except (ArithmeticError, ValueError):
        return None
    return _Valued(value, left.reads | right.reads)


def _evaluate_constructor(node: ast.Call, scope: _TestScope,
                          resolving: frozenset[str]) -> _Valued | None:
    """``frozenset(...)`` and kin over an evaluable argument, unless shadowed."""
    if not isinstance(node.func, ast.Name) or node.func.id in scope.bound:
        return None
    construct = _COLLECTION_CONSTRUCTORS.get(node.func.id)
    if construct is None or node.keywords or len(node.args) > 1:
        return None
    if not node.args:
        return _Valued(construct(), frozenset())
    arg = _evaluate_under_defaults(node.args[0], scope, resolving)
    if arg is None:
        return None
    try:
        return _Valued(construct(arg.value), arg.reads)
    except TypeError:
        return None


def _decorated_with_hatch(node: ast.AST) -> bool:
    """True if any decorator on *node* names ``code_default_config``."""
    return any(
        'code_default_config' in ast.unparse(dec)
        for dec in getattr(node, 'decorator_list', [])
    )


def _function_isolates(func: _TestFunc) -> bool:
    """True if *func* — a test or a fixture — insulates from the ambient config."""
    if _decorated_with_hatch(func):
        return True
    args = func.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if any(a.arg == 'code_default_config' for a in params):
        return True
    return any(
        'ORCH_CONFIG_PATH' in ast.unparse(stmt) for stmt in func.body
    )


def _is_autouse_fixture(func: _TestFunc) -> bool:
    for dec in func.decorator_list:
        text = ast.unparse(dec)
        if 'fixture' in text and 'autouse=True' in text.replace(' ', ''):
            return True
    return False


def _scope_isolates(node: ast.Module | ast.ClassDef) -> bool:
    """True if every test directly inside *node* is insulated by *node* itself."""
    if _decorated_with_hatch(node):
        return True
    return any(
        isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _is_autouse_fixture(sub)
        and _function_isolates(sub)
        for sub in node.body
    )


def _is_isolated(func: _TestFunc, enclosing_class: ast.ClassDef | None) -> bool:
    """True if *func* is insulated from the ambient config by itself or its class."""
    if _function_isolates(func):
        return True
    return enclosing_class is not None and _scope_isolates(enclosing_class)


def _paths_overlap(a: str, b: str) -> bool:
    """True if one dotted path equals, or is a segment prefix of, the other."""
    return a == b or a.startswith(b + '.') or b.startswith(a + '.')


def _default_assertion(cmp_node: ast.Compare,
                       scope: _TestScope) -> tuple[int, str, Any] | None:
    """``(lineno, attr_path, value)`` if *cmp_node* asserts a package default."""
    if len(cmp_node.ops) != 1 or len(cmp_node.comparators) != 1:
        return None
    if not isinstance(cmp_node.ops[0], (ast.Eq, ast.Is)):
        return None
    chain = _attr_chain(cmp_node.left)
    if chain is None:
        return None
    root, attr_path = chain
    if not _is_config_root(root, scope.bare_configs):
        return None
    valued = _evaluate_under_defaults(cmp_node.comparators[0], scope)
    if valued is None:
        return None
    if any(_paths_overlap(read, attr_path) for read in valued.reads):
        return None
    default = _resolve_default(attr_path)
    if default is _UNRESOLVED:
        return None
    # The type() match keeps True/1 and False/0 from colliding.
    if type(default) is type(valued.value) and default == valued.value:
        return cmp_node.lineno, attr_path, valued.value
    return None


def _default_assertions(func: _TestFunc,
                        scope: _TestScope) -> Iterator[tuple[int, str, Any]]:
    """Yield ``(lineno, attr_path, value)`` for each default-asserting compare."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Assert):
            continue
        for cmp_node in ast.walk(node.test):
            if isinstance(cmp_node, ast.Compare):
                found = _default_assertion(cmp_node, scope)
                if found is not None:
                    yield found


def iter_default_asserting_tests(
    path: str | Path,
) -> Iterator[tuple[str | None, str, int, str, Any]]:
    """Yield ``(class_name, test_name, lineno, attr_path, value)`` for *path*.

    ``value`` is what the asserted right-hand side evaluates to under package
    defaults.  ``class_name`` is ``None`` for a module-level test function.
    One tuple is yielded per offending ASSERTION, so a test with several
    default assertions appears several times.
    """
    source = Path(path).read_text()
    tree = ast.parse(source, filename=str(path))
    if _scope_isolates(tree):
        return

    def scan(func, enclosing_class):
        if not func.name.startswith('test'):
            return
        if _is_isolated(func, enclosing_class):
            return
        scope = _test_scope(func)
        class_name = enclosing_class.name if enclosing_class is not None else None
        for lineno, attr_path, value in _default_assertions(func, scope):
            yield class_name, func.name, lineno, attr_path, value

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from scan(node, None)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield from scan(sub, node)
