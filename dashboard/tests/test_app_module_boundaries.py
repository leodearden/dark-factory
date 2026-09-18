"""Structural guard: the handlers and loops extracted out of `app.py` stay out.

Task 5586 moved six route handlers, the two background samplers and their
shared substrate out of `dashboard/src/dashboard/app.py` into focused
modules under `dashboard/src/dashboard/api/` plus `loops.py` and
`project_dbs.py`. That move is only worth doing once: without a guard,
a later change can quietly define a moved name back in `app.py`, leave a
route unregistered, or make a new module reach back into `app.py` for a
symbol — and every behavioural test would still pass, because the
behaviour is what the move deliberately did not change.

So this module asserts the *structure* rather than the behaviour, in four
parts, parsing the real tree with `ast` the way
`test_clock_discipline.py` does:

1. Each extracted module actually defines the names it owns.
2. `app.py` defines none of them. This is the executable form of "no moved
   handler or loop body remains in app.py". `app.py` legitimately
   *imports* several of these names back (`lifespan` constructs the stores
   and spawns the loops; `api_costs` still parses a window), so only
   definition nodes count — an `Import`/`ImportFrom` is never a definition.
3. Every extracted route is registered exactly once on the real app, and
   its `endpoint` is the function object from the module that owns it — so
   a route dropped during registration, or served by a stale copy left
   behind in `app.py`, fails loudly.
4. No extracted module imports `dashboard.app` (code-quality heuristic 13:
   a file makes sense in isolation, and a reach-back here would also be a
   circular import). Matched on `Import`/`ImportFrom` nodes, not on source
   substrings, so a mention of `app.py` inside a docstring is not a false
   positive.

Each data structure is asserted non-empty, so emptying one can never make
its check pass vacuously.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / 'src'
_APP_PY = _SRC / 'dashboard' / 'app.py'

# New module dotted path -> the top-level names it must define.
MOVED_SYMBOLS: dict[str, tuple[str, ...]] = {
    'dashboard.api.window': ('_WINDOW_DAYS', '_parse_window'),
    'dashboard.project_dbs': (
        '_project_scoped_dbs',
        '_project_scoped_dbs_labeled',
        '_cost_dbs',
        '_burndown_dbs',
    ),
}

# URL path -> (module dotted path, handler name) for every extracted route.
MOVED_ROUTES: dict[str, tuple[str, str]] = {}

# Repo-relative path of every file this extraction created.
NEW_MODULES: tuple[str, ...] = (
    'dashboard/src/dashboard/api/window.py',
    'dashboard/src/dashboard/project_dbs.py',
)

_REACH_BACK = 'dashboard.app'


def _module_path(dotted: str) -> Path:
    """Resolve a `dashboard.*` dotted module path to its file in this tree."""
    return _SRC.joinpath(*dotted.split('.')).with_suffix('.py')


def _defined_names(tree: ast.AST) -> set[str]:
    """Top-level names *defined* (never merely imported) by a parsed module."""
    names: set[str] = set()
    for node in getattr(tree, 'body', []):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _parse(path: Path) -> ast.Module:
    """Parse *path*, asserting it exists so a check can't quietly stop checking."""
    assert path.is_file(), f'scan target is missing: {path}'
    return ast.parse(path.read_text())


def test_each_extracted_module_defines_the_names_it_owns():
    """Every name listed in `MOVED_SYMBOLS` is defined by the module that owns it."""
    assert MOVED_SYMBOLS, 'MOVED_SYMBOLS is empty — this guard would pass vacuously'

    missing: list[str] = []
    for dotted, names in sorted(MOVED_SYMBOLS.items()):
        defined = _defined_names(_parse(_module_path(dotted)))
        missing.extend(f'{dotted}.{name}' for name in names if name not in defined)

    assert not missing, (
        'Extracted modules are missing names they are supposed to own:\n'
        + '\n'.join(missing)
    )


def test_app_py_defines_no_extracted_name():
    """`app.py` defines none of the moved names — importing one back is fine.

    The executable form of "no moved handler or loop body remains in
    app.py", and the thing that stops a later change from growing `app.py`
    back one handler at a time.
    """
    assert MOVED_SYMBOLS, 'MOVED_SYMBOLS is empty — this guard would pass vacuously'

    app_defined = _defined_names(_parse(_APP_PY))
    resurrected = sorted(
        f'{name} (belongs to {dotted})'
        for dotted, names in MOVED_SYMBOLS.items()
        for name in names
        if name in app_defined
    )

    assert not resurrected, (
        'app.py defines names that were extracted out of it — move them back '
        'to the module that owns them (an import is fine, a definition is not):\n'
        + '\n'.join(resurrected)
    )


def test_every_extracted_route_is_registered_from_its_own_module():
    """Each moved route resolves to exactly one route whose endpoint is the real handler.

    The anti-vacuity check here is *derived* from the tree rather than a
    hand-written `assert MOVED_ROUTES`: every extracted module that owns an
    `APIRouter` must appear in `MOVED_ROUTES`. That catches a map emptied
    to nothing (what a literal non-empty assertion catches) and also a map
    emptied of just one row (what it does not), and it stays true while the
    extraction is still in progress and no route has moved yet.
    """
    from dashboard.app import app

    router_owners = {
        dotted
        for dotted in MOVED_SYMBOLS
        if 'router' in _defined_names(_parse(_module_path(dotted)))
    }
    uncovered = router_owners - {dotted for dotted, _ in MOVED_ROUTES.values()}
    assert not uncovered, (
        'These extracted modules own an APIRouter but have no row in '
        'MOVED_ROUTES, so their routes are unguarded:\n' + '\n'.join(sorted(uncovered))
    )

    problems: list[str] = []
    for url_path, (dotted, handler_name) in sorted(MOVED_ROUTES.items()):
        # Exact path match, never a prefix: `/api/v2/dashboard/memory` and
        # `/api/v2/dashboard/memory-graphs` are different routes in
        # different modules.
        matches = [r for r in app.routes if getattr(r, 'path', None) == url_path]
        if len(matches) != 1:
            problems.append(f'{url_path}: expected exactly 1 route, found {len(matches)}')
            continue
        expected = getattr(importlib.import_module(dotted), handler_name)
        actual = getattr(matches[0], 'endpoint', None)
        if actual is not expected:
            problems.append(
                f'{url_path}: endpoint is {actual!r}, expected {dotted}.{handler_name}'
            )

    assert not problems, (
        'Extracted routes are not registered from the modules that own them:\n'
        + '\n'.join(problems)
    )


def test_no_extracted_module_reaches_back_into_app():
    """No new module imports `dashboard.app` (heuristic 13; also a cycle)."""
    assert NEW_MODULES, 'NEW_MODULES is empty — this guard would pass vacuously'

    repo_root = _SRC.parent.parent
    reach_backs: list[str] = []
    for rel in NEW_MODULES:
        path = repo_root / rel
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.Import):
                if any(alias.name == _REACH_BACK for alias in node.names):
                    reach_backs.append(f'{rel}:{node.lineno}: import {_REACH_BACK}')
            elif isinstance(node, ast.ImportFrom):
                imported = f'{node.module}.{node.names[0].name}' if node.module else ''
                if node.module == _REACH_BACK or imported == _REACH_BACK:
                    reach_backs.append(f'{rel}:{node.lineno}: from ... import app')

    assert not reach_backs, (
        'Extracted modules reach back into app.py — move the shared symbol to '
        'a module both sides can import instead:\n' + '\n'.join(reach_backs)
    )
