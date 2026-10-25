"""AST scan for tests that assert a CODE DEFAULT off a bare ``OrchestratorConfig()``.

Helper module for :mod:`test_code_default_hatch_guard`.  The ``_`` prefix keeps
it out of pytest collection while remaining importable (``conftest.py`` puts the
tests dir on ``sys.path``), matching ``_orch_helpers.py`` and
``_verify_config_corpus.py``.

WHAT IT LOOKS FOR.  A test is flagged when it (1) builds a *bare*
``OrchestratorConfig()`` — no args, no keywords — either binding a name to it
or asserting directly off the call (``OrchestratorConfig().x.y == literal``),
(2) is not isolated from the ambient operational config, and (3) asserts an
attribute path off that name or call equals a literal that is exactly the
package default for that path.  Such an assertion passes only because the
repo happens NOT to declare that key in ``dark-factory-orchestrator.yaml``; the
autouse ``_isolate_orch_config`` fixture pins that file for every test.  An
operator declaring the key turns the test red without touching any code the
test is about (task 3886's incident).

WHAT COUNTS AS ISOLATED.  Three idioms, all honoured:

* ``@pytest.mark.usefixtures("code_default_config")`` on the function or on its
  enclosing class (the sanctioned hatch — detected by substring over the
  unparsed decorator, so any quote spelling works);
* a ``code_default_config`` parameter (requesting the fixture directly);
* the substring ``ORCH_CONFIG_PATH`` anywhere in the unparsed function body,
  which covers the two ad-hoc idioms already widespread in the suite —
  ``monkeypatch.setenv('ORCH_CONFIG_PATH', '')`` (154 uses) and
  ``monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)`` (29 uses).

DELIBERATE ASYMMETRY.  That last rule is a substring test, so it also silences a
test that merely *mentions* the variable — and the ``type()`` match below
silences ``== 1800`` against a ``1800.0`` default.  Both admit false NEGATIVES
and neither admits a false POSITIVE.  That is the trade a *committed* ratchet
requires: a guard that occasionally misses a landmine is a nuisance, while one
that occasionally invents one gets disabled.

NOT DETECTED HERE.  Tests that exercise default-only *behaviour* without
asserting any config literal are invisible to a static scan (e.g. a section's
``enabled`` flag defaulting True is what makes a code path run at all).  Those
are found by running the suite against a config that perturbs every undeclared
key; see the task-4481 commit log for that measurement.
"""
from __future__ import annotations

import ast
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from orchestrator.config import OrchestratorConfig

#: Path guaranteed not to exist, so ``YamlSettingsSource`` skips the layer-2
#: project file and only the package-bundled ``defaults.yaml`` loads.  Same
#: mechanism the ``code_default_config`` fixture uses.
_ABSENT_CONFIG_PATH = '/nonexistent/df-4481-no-such-config.yaml'

#: Marker for "this dotted path does not resolve", distinct from a ``None``
#: default.
_UNRESOLVED = object()


def _load_package_defaults() -> OrchestratorConfig:
    """Build an ``OrchestratorConfig`` from package defaults ONLY.

    Saves and restores ``ORCH_CONFIG_PATH`` so the scan never depends on — and
    never perturbs — the ambient config the rest of the suite relies on.
    """
    sentinel = object()
    previous: Any = os.environ.get('ORCH_CONFIG_PATH', sentinel)
    try:
        os.environ['ORCH_CONFIG_PATH'] = _ABSENT_CONFIG_PATH
        return OrchestratorConfig()
    finally:
        if previous is sentinel:
            os.environ.pop('ORCH_CONFIG_PATH', None)
        else:
            os.environ['ORCH_CONFIG_PATH'] = previous


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


def _is_config_root(root: ast.expr, names: set[str]) -> bool:
    """True if *root* is a name bound to a bare config, or a bare call itself."""
    if isinstance(root, ast.Name):
        return root.id in names
    return _is_bare_config_call(root)


def _decorated_with_hatch(node: ast.AST) -> bool:
    """True if any decorator on *node* names ``code_default_config``."""
    return any(
        'code_default_config' in ast.unparse(dec)
        for dec in getattr(node, 'decorator_list', [])
    )


def _is_isolated(func: ast.FunctionDef | ast.AsyncFunctionDef,
                 enclosing_class: ast.ClassDef | None) -> bool:
    """True if *func* is insulated from the ambient operational config."""
    if _decorated_with_hatch(func):
        return True
    if enclosing_class is not None and _decorated_with_hatch(enclosing_class):
        return True
    args = func.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if any(a.arg == 'code_default_config' for a in params):
        return True
    return any(
        'ORCH_CONFIG_PATH' in ast.unparse(stmt) for stmt in func.body
    )


def _bare_config_names(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names bound to a bare ``OrchestratorConfig()`` and to nothing else.

    A name reassigned from anything other than a bare construction (a
    ``load_config(...)`` result, a kwargs-built config, a later mutation) is
    dropped: the scan cannot know which binding an assertion sees, and a
    committed ratchet must not guess.
    """
    bare: set[str] = set()
    other: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets, value = [node.target], node.value
        else:
            continue
        bucket = bare if _is_bare_config_call(value) else other
        for target in targets:
            if isinstance(target, ast.Name):
                bucket.add(target.id)
    return bare - other


def _default_assertions(func: ast.FunctionDef | ast.AsyncFunctionDef,
                        names: set[str]) -> Iterator[tuple[int, str, Any]]:
    """Yield ``(lineno, attr_path, literal)`` for each default-asserting compare."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Assert):
            continue
        for cmp_node in ast.walk(node.test):
            if not isinstance(cmp_node, ast.Compare):
                continue
            if len(cmp_node.ops) != 1 or len(cmp_node.comparators) != 1:
                continue
            if not isinstance(cmp_node.ops[0], (ast.Eq, ast.Is)):
                continue
            chain = _attr_chain(cmp_node.left)
            if chain is None:
                continue
            root, attr_path = chain
            if not _is_config_root(root, names):
                continue
            try:
                literal = ast.literal_eval(cmp_node.comparators[0])
            except (ValueError, TypeError, SyntaxError, MemoryError,
                    RecursionError):
                continue
            default = _resolve_default(attr_path)
            if default is _UNRESOLVED:
                continue
            # The type() match keeps True/1 and False/0 from colliding.
            if type(default) is type(literal) and default == literal:
                yield cmp_node.lineno, attr_path, literal


def iter_default_asserting_tests(
    path: str | Path,
) -> Iterator[tuple[str | None, str, int, str, Any]]:
    """Yield ``(class_name, test_name, lineno, attr_path, literal)`` for *path*.

    ``class_name`` is ``None`` for a module-level test function.  One tuple is
    yielded per offending ASSERTION, so a test with several default assertions
    appears several times.
    """
    source = Path(path).read_text()
    tree = ast.parse(source, filename=str(path))

    def scan(func, enclosing_class):
        if not func.name.startswith('test'):
            return
        if _is_isolated(func, enclosing_class):
            return
        names = _bare_config_names(func)
        class_name = enclosing_class.name if enclosing_class is not None else None
        for lineno, attr_path, literal in _default_assertions(func, names):
            yield class_name, func.name, lineno, attr_path, literal

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from scan(node, None)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield from scan(sub, node)
