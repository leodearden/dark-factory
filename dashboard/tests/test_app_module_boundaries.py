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
   Module-level control flow is descended into, because a definition
   tucked inside an `if TYPE_CHECKING:` or a `try:`/`except ImportError:`
   fallback still binds the name at module scope; a `def` or `class` body
   is not, because a name bound there does not.
3. Every extracted route is registered exactly once on the real app, and
   its `endpoint` is the function object from the module that owns it — so
   a route dropped during registration, or served by a stale copy left
   behind in `app.py`, fails loudly.
4. No extracted module imports `dashboard.app` (code-quality heuristic 13:
   a file makes sense in isolation, and a reach-back here would also be a
   circular import). Matched on `Import`/`ImportFrom` nodes, not on source
   substrings, so a mention of `app.py` inside a docstring is not a false
   positive. Every alias of a statement is checked and `.`-relative
   spellings are resolved against the importing file's own package, so
   `from dashboard import config, app` and the `from ..app import lifespan`
   that a module inside `dashboard/api/` would most naturally reach back
   with are caught alongside the absolute spellings.

`MOVED_SYMBOLS` pins the moved tuning constants and cache objects as well
as the handlers, loops and stores. That is deliberate: a constant
copy-pasted back into `app.py` beside the one that moved is exactly the
quiet regrowth check 2 exists to catch, and it would leave two copies of
one datum with nothing else complaining.

Each data structure is asserted non-empty, so emptying one can never make
its check pass vacuously. The two matchers below — what counts as a
module-level definition, and what counts as a reach-back — are themselves
covered by table-driven tests over parsed source, so the guard cannot
under-check without a test saying so.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import TypeGuard

import pytest

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
    'dashboard.api.tasks': ('api_tasks',),
    'dashboard.api.orchestrators': ('api_orchestrators',),
    'dashboard.api.memory': ('_MEMORY_ENDPOINT_TIMEOUT_SECONDS', 'api_memory'),
    'dashboard.api.burndown': ('_BURNDOWN_WINDOWS', 'api_burndown'),
    'dashboard.api.merge_queue': ('api_merge_queue',),
    'dashboard.api.escalations': (
        '_TASK_CARDS_TTL_SECONDS',
        '_TASK_CARDS_BUDGET',
        '_task_cards_cache',
        '_task_cards_cache_clear',
        '_load_task_cards',
        'api_escalations',
    ),
    'dashboard.loops': (
        '_SAMPLE_INTERVAL_SECONDS',
        '_DOWNSAMPLE_INTERVAL_SECONDS',
        '_CHECKPOINT_INTERVAL_SECONDS',
        '_BurndownStore',
        '_MetricsStore',
        '_sleep_to_aligned_tick',
        '_burndown_loop',
        '_metrics_loop',
    ),
}

# URL path -> (module dotted path, handler name) for every extracted route.
MOVED_ROUTES: dict[str, tuple[str, str]] = {
    '/api/v2/dashboard/merge-queue': ('dashboard.api.merge_queue', 'api_merge_queue'),
    '/api/v2/dashboard/escalations': ('dashboard.api.escalations', 'api_escalations'),
    '/api/v2/dashboard/burndown': ('dashboard.api.burndown', 'api_burndown'),
    '/api/v2/dashboard/memory': ('dashboard.api.memory', 'api_memory'),
    '/api/v2/dashboard/orchestrators': ('dashboard.api.orchestrators', 'api_orchestrators'),
    '/api/v2/dashboard/tasks': ('dashboard.api.tasks', 'api_tasks'),
}

# Repo-relative path of every file this extraction created.
NEW_MODULES: tuple[str, ...] = (
    'dashboard/src/dashboard/api/merge_queue.py',
    'dashboard/src/dashboard/api/escalations.py',
    'dashboard/src/dashboard/api/__init__.py',
    'dashboard/src/dashboard/api/burndown.py',
    'dashboard/src/dashboard/api/memory.py',
    'dashboard/src/dashboard/api/orchestrators.py',
    'dashboard/src/dashboard/api/window.py',
    'dashboard/src/dashboard/project_dbs.py',
    'dashboard/src/dashboard/loops.py',
    'dashboard/src/dashboard/api/tasks.py',
)

_REACH_BACK = 'dashboard.app'


def _module_path(dotted: str) -> Path:
    """Resolve a `dashboard.*` dotted module path to its file in this tree."""
    return _SRC.joinpath(*dotted.split('.')).with_suffix('.py')


# Statements that can appear at module level and whose bodies still bind
# names there. Async blocks are absent because they cannot occur outside a
# coroutine; `def` and `class` are absent because their bodies bind locally.
_MODULE_LEVEL_BLOCKS = (ast.If, ast.Try, ast.TryStar, ast.With, ast.For, ast.While)


def _block_statements(node: ast.stmt) -> list[ast.stmt]:
    """Every statement a module-level block runs, `except` handlers included."""
    inner: list[ast.stmt] = [
        stmt
        for field in ('body', 'orelse', 'finalbody')
        for stmt in getattr(node, field, [])
    ]
    for handler in getattr(node, 'handlers', []):
        inner.extend(handler.body)
    return inner


def _defined_names(tree: ast.AST) -> set[str]:
    """Module-level names *defined* (never merely imported) by a parsed module.

    Descends through module-level control flow, because a definition inside
    an `if TYPE_CHECKING:` block or a `try:`/`except ImportError:` fallback
    still binds its name at module scope — and check 2, whose whole job is
    to be hard to sneak past, would otherwise miss exactly the spellings
    someone reaching for a quiet resurrection would reach for. It never
    descends into a `def` or `class` body: a name bound there is local.
    """
    names: set[str] = set()
    pending: list[ast.stmt] = list(getattr(tree, 'body', []))
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, _MODULE_LEVEL_BLOCKS):
            pending.extend(_block_statements(node))
    return names


def _parse(path: Path) -> ast.Module:
    """Parse *path*, asserting it exists so a check can't quietly stop checking."""
    assert path.is_file(), f'scan target is missing: {path}'
    return ast.parse(path.read_text())


def _package_parts(path: Path) -> tuple[str, ...]:
    """Dotted parts of the package *containing* `path`, for resolving `.` imports.

    `dashboard/api/window.py` and `dashboard/api/__init__.py` both sit in
    package `dashboard.api`, which is what dropping the file's own stem
    gives for either spelling.
    """
    return path.resolve().relative_to(_SRC).with_suffix('').parts[:-1]


def _absolute_module(node: ast.ImportFrom, pkg_parts: tuple[str, ...]) -> str:
    """The absolute module an `ImportFrom` reads from, `.`-relative levels resolved.

    `node.level` counts the leading dots: 0 is already absolute, 1 is the
    importing file's own package, 2 its parent, and so on. A level that
    climbs above `src/` names nothing importable, so it resolves to `''`.
    """
    if node.level == 0:
        return node.module or ''
    kept = len(pkg_parts) - node.level + 1
    if kept <= 0:
        return ''
    prefix = pkg_parts[:kept]
    return '.'.join((*prefix, node.module) if node.module else prefix)


def _reaches_back(
    node: ast.AST, pkg_parts: tuple[str, ...]
) -> TypeGuard[ast.Import | ast.ImportFrom]:
    """True if *node* is an import of `dashboard.app`, however it is spelled.

    One module has six spellings — `import dashboard.app`,
    `from dashboard import app`, `from dashboard.app import lifespan`, and
    the `.`-relative form of each — so the match resolves each statement to
    an absolute module rather than reading its source text. Every alias is
    checked, not just the first: `from dashboard import config, app` names
    `app` second and is a reach-back all the same.
    """
    if isinstance(node, ast.Import):
        return any(alias.name == _REACH_BACK for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = _absolute_module(node, pkg_parts)
        return module == _REACH_BACK or any(
            f'{module}.{alias.name}' == _REACH_BACK for alias in node.names
        )
    return False


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
        pkg_parts = _package_parts(path)
        reach_backs.extend(
            f'{rel}:{node.lineno}: imports {_REACH_BACK}'
            for node in ast.walk(_parse(path))
            if _reaches_back(node, pkg_parts)
        )

    assert not reach_backs, (
        'Extracted modules reach back into app.py — move the shared symbol to '
        'a module both sides can import instead:\n' + '\n'.join(reach_backs)
    )


# --- coverage for the matchers the four checks above are built on ----------

# A module-level block hiding a definition, and the name it still binds.
# Every one of these would sail past a walk of `tree.body` alone.
_HIDDEN_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ('if TYPE_CHECKING:\n    def api_tasks(): ...\n', 'api_tasks'),
    ('if True:\n    pass\nelse:\n    class _MetricsStore: ...\n', '_MetricsStore'),
    ('try:\n    _SAMPLE_INTERVAL_SECONDS = 600\nexcept NameError:\n    pass\n',
     '_SAMPLE_INTERVAL_SECONDS'),
    ('try:\n    pass\nexcept ImportError:\n    async def _metrics_loop(): ...\n',
     '_metrics_loop'),
    ('try:\n    pass\nfinally:\n    _WINDOW_DAYS: dict[str, int] = {}\n', '_WINDOW_DAYS'),
    ('if TYPE_CHECKING:\n    if True:\n        def _load_task_cards(): ...\n',
     '_load_task_cards'),
)

# Sources that bind `api_tasks` somewhere that is *not* a module-level
# definition — an import, or a name local to a `def`/`class` body.
_NOT_MODULE_DEFINITIONS: tuple[str, ...] = (
    'from dashboard.api.tasks import api_tasks\n',
    'import dashboard.api.tasks as api_tasks\n',
    'def outer():\n    def api_tasks(): ...\n',
    'def outer():\n    api_tasks = 1\n',
    'class Holder:\n    api_tasks = 1\n',
)

# (source, package of the file importing it, is it a reach-back?). The
# relative rows are the spellings a module inside `dashboard/api/` would
# most naturally use, which is why they have to resolve rather than match
# textually.
_REACH_BACK_SPELLINGS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ('import dashboard.app', ('dashboard', 'api'), True),
    ('import dashboard.app as _app', ('dashboard', 'api'), True),
    ('from dashboard.app import lifespan', ('dashboard', 'api'), True),
    ('from dashboard import app', ('dashboard', 'api'), True),
    ('from dashboard import config, app', ('dashboard', 'api'), True),
    ('from . import app', ('dashboard',), True),
    ('from .app import lifespan', ('dashboard',), True),
    ('from .. import app', ('dashboard', 'api'), True),
    ('from ..app import lifespan', ('dashboard', 'api'), True),
    ('from dashboard import config', ('dashboard', 'api'), False),
    ('from dashboard.api.window import _parse_window', ('dashboard', 'api'), False),
    ('import dashboard.loops', ('dashboard', 'api'), False),
    ('from . import window', ('dashboard', 'api'), False),
    ('from ..project_dbs import _cost_dbs', ('dashboard', 'api'), False),
    ('from ... import app', ('dashboard', 'api'), False),
    ('"""Prose naming dashboard.app is not an import."""', ('dashboard', 'api'), False),
)


@pytest.mark.parametrize(('source', 'name'), _HIDDEN_DEFINITIONS)
def test_a_definition_inside_module_level_control_flow_still_counts(
    source: str, name: str
) -> None:
    """Check 2 sees a moved name resurrected under an `if` or a `try`."""
    assert name in _defined_names(ast.parse(source))


@pytest.mark.parametrize('source', _NOT_MODULE_DEFINITIONS)
def test_an_import_or_a_local_binding_is_not_a_definition(source: str) -> None:
    """Importing a moved name back, or reusing it as a local, is not a resurrection."""
    assert 'api_tasks' not in _defined_names(ast.parse(source))


def test_package_parts_locate_the_file_a_relative_import_resolves_against():
    """A module and its package's `__init__` both resolve to the package they sit in."""
    assert _package_parts(_SRC / 'dashboard' / 'api' / 'window.py') == ('dashboard', 'api')
    assert _package_parts(_SRC / 'dashboard' / 'api' / '__init__.py') == ('dashboard', 'api')
    assert _package_parts(_APP_PY) == ('dashboard',)


@pytest.mark.parametrize(('source', 'pkg_parts', 'reaches_back'), _REACH_BACK_SPELLINGS)
def test_the_reach_back_matcher_resolves_every_spelling(
    source: str, pkg_parts: tuple[str, ...], reaches_back: bool
) -> None:
    """Check 4 catches `dashboard.app` however an importer names it."""
    matched = [n for n in ast.walk(ast.parse(source)) if _reaches_back(n, pkg_parts)]
    assert bool(matched) is reaches_back, (
        f'{source!r} imported from package {".".join(pkg_parts)!r}: '
        f'matcher said {bool(matched)}, expected {reaches_back}'
    )
