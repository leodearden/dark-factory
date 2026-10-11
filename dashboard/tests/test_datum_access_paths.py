"""Every task read has one access path: the AST-executed half of PRD decision 17.

``plans/dashboard-one-datum-one-path-prd.md`` decision 17(b) replaces token
greps with one executed check: ``fetch_tasks``, ``fetch_task_page``,
``fetch_statuses`` and ``fetch_task`` are used only where an
:class:`AccessGrant` below says so, and the one named exemption is
``app.py::_fanout_probe_completion``. The apparatus — a finder, fixture tests
of its matcher, then acceptance tests over a scan set that fails loudly — is
``test_clock_discipline.py``'s.

The old-path census (sketch #14) follows in the same shape: :data:`_RETIRED`
is one typed table of the paths PRD decision 16 deleted, each kind checked by
executing something — requesting the asset and parsing index.html, parsing
every served script for window exports, or calling the route.
"""

from __future__ import annotations

import ast
import json
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import AsyncMock, patch
from urllib.parse import urlsplit

from _dashboard_helpers import ScriptTagCollector
from _lock_chip_matrix import node_path

# ---------------------------------------------------------------------------
# The finder
# ---------------------------------------------------------------------------

GUARDED = frozenset({'fetch_tasks', 'fetch_task_page', 'fetch_statuses', 'fetch_task'})


@dataclass(frozen=True, slots=True)
class AccessUse:
    """One Load of a guarded read, and the def it sits in (``'<module>'`` at top level)."""

    line: int
    qualname: str
    name: str


@dataclass(frozen=True, slots=True)
class AccessGrant:
    """Where a guarded read may be used: a module, optionally one def in it.

    *module* is a path relative to ``src/dashboard``. *scope* is ``None`` for
    the whole module, or the exact qualname of the one def the grant covers.
    """

    module: str
    scope: str | None
    names: frozenset[str]
    reason: str

    def excuses(self, module: str, use: AccessUse) -> bool:
        return (
            module == self.module
            and use.name in self.names
            and (self.scope is None or use.qualname == self.scope)
        )


class _UseFinder(ast.NodeVisitor):
    """Collects every Load of a guarded read, under the qualname of its def."""

    def __init__(self, aliases: Mapping[str, str]) -> None:
        self._aliases = aliases
        self._scope: list[str] = []
        self.uses: list[AccessUse] = []

    def _record(self, node: ast.expr, name: str) -> None:
        qualname = '.'.join(self._scope) or '<module>'
        self.uses.append(AccessUse(node.lineno, qualname, name))

    def _visit_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in self._aliases:
            self._record(node, self._aliases[node.id])

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Load) and node.attr in GUARDED:
            self._record(node, node.attr)
        self.generic_visit(node)


def find_access_path_uses(source: str) -> list[AccessUse]:
    """Every Load of a guarded read in *source*: a call, a reference or an alias.

    A def's name and an import are not Loads, so a module that only defines
    or imports a read reports nothing; docstrings and comments never reach
    the AST as names at all.
    """
    tree = ast.parse(source)
    aliases = {name: name for name in GUARDED}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in GUARDED:
                    aliases[alias.asname or alias.name] = alias.name
    finder = _UseFinder(aliases)
    finder.visit(tree)
    return finder.uses


def violations(
    uses_by_module: Mapping[str, Sequence[AccessUse]],
    grants: Iterable[AccessGrant],
) -> list[tuple[str, AccessUse]]:
    """Every ``(module, use)`` no grant excuses."""
    granted = tuple(grants)
    return [
        (module, use)
        for module, uses in uses_by_module.items()
        for use in uses
        if not any(grant.excuses(module, use) for grant in granted)
    ]


# ---------------------------------------------------------------------------
# Matcher unit tests (fixtures, not the real tree)
# ---------------------------------------------------------------------------


def test_a_bare_call_reports_its_enclosing_def():
    source = (
        'async def f(client, config, root):\n'
        '    return await fetch_tasks(client, config, root)\n'
    )
    assert find_access_path_uses(source) == [AccessUse(2, 'f', 'fetch_tasks')]


def test_an_attribute_call_is_reported():
    source = (
        'from dashboard.data import tasks\n'
        'async def g(client, config, root):\n'
        '    return await tasks.fetch_statuses(client, config, root)\n'
    )
    assert find_access_path_uses(source) == [AccessUse(3, 'g', 'fetch_statuses')]


def test_an_aliased_import_is_reported_under_the_guarded_name():
    source = (
        'from dashboard.data.tasks import fetch_tasks as ft\n'
        'async def h(client, config, root):\n'
        '    return await ft(client, config, root)\n'
    )
    assert find_access_path_uses(source) == [AccessUse(3, 'h', 'fetch_tasks')]


def test_a_reference_that_is_not_a_call_is_a_use():
    source = (
        'import functools\n'
        'from dashboard.data.tasks import fetch_task_page\n'
        'def k(client, config, root):\n'
        '    return functools.partial(fetch_task_page, client, config, root)\n'
    )
    assert find_access_path_uses(source) == [AccessUse(4, 'k', 'fetch_task_page')]


def test_the_import_statement_itself_is_not_a_use():
    source = (
        'from dashboard.data.tasks import fetch_tasks, fetch_task\n'
        'from dashboard.data.tasks import fetch_statuses as fs\n'
        'import dashboard.data.tasks\n'
    )
    assert find_access_path_uses(source) == []


def test_prose_naming_a_read_is_not_a_use():
    source = (
        '"""Module docstring naming fetch_tasks."""\n'
        '# a comment naming fetch_task_page(...)\n'
        'def m():\n'
        '    """Calls fetch_statuses, says the docstring."""\n'
        "    return 'fetch_task is only a string here'\n"
    )
    assert find_access_path_uses(source) == []


def test_nested_defs_and_methods_report_dotted_qualnames():
    source = (
        'def outer():\n'
        '    def inner():\n'
        '        return fetch_tasks\n'
        '    return inner\n'
        'class Cls:\n'
        '    async def meth(self):\n'
        '        return await fetch_task(None, None, None, 1)\n'
        'TOP = fetch_statuses\n'
    )
    assert find_access_path_uses(source) == [
        AccessUse(3, 'outer.inner', 'fetch_tasks'),
        AccessUse(7, 'Cls.meth', 'fetch_task'),
        AccessUse(8, '<module>', 'fetch_statuses'),
    ]


def test_a_def_named_like_a_read_is_its_definition_not_a_use():
    source = (
        'async def fetch_tasks(client, config, project_root):\n'
        '    return []\n'
        'async def fetch_task(client, config, project_root, task_id):\n'
        '    return {}\n'
    )
    assert find_access_path_uses(source) == []


def test_lookalike_names_are_not_reported():
    source = (
        'from dashboard.data.tasks import fetch_task_prose, fetch_external_statuses\n'
        'from dashboard.data import tasks\n'
        'async def n(client, config, root):\n'
        '    await fetch_task_prose(client, config, root, 1)\n'
        '    await tasks.fetch_external_statuses(client, config, [])\n'
        '    return tasks.fetch_tasks_later\n'
    )
    assert find_access_path_uses(source) == []


def test_a_use_covered_by_its_grant_is_excused():
    use = AccessUse(10, 'acquire', 'fetch_tasks')
    grant = AccessGrant('data/snap.py', 'acquire', frozenset({'fetch_tasks'}), 'test')

    assert violations({'data/snap.py': [use]}, [grant]) == []


def test_a_function_scoped_grant_does_not_cover_another_def_in_its_module():
    granted = AccessUse(10, 'probe', 'fetch_tasks')
    elsewhere = AccessUse(20, 'handler', 'fetch_tasks')
    grant = AccessGrant('app.py', 'probe', frozenset({'fetch_tasks'}), 'test')

    assert violations({'app.py': [granted, elsewhere]}, [grant]) == [('app.py', elsewhere)]


def test_a_module_scoped_grant_covers_every_def_but_only_its_named_reads():
    named = [
        AccessUse(3, 'a', 'fetch_tasks'),
        AccessUse(9, 'Cls.b', 'fetch_tasks'),
        AccessUse(12, '<module>', 'fetch_tasks'),
    ]
    unnamed = AccessUse(15, 'a', 'fetch_task')
    grant = AccessGrant('data/snap.py', None, frozenset({'fetch_tasks'}), 'test')

    assert violations({'data/snap.py': [*named, unnamed]}, [grant]) == [
        ('data/snap.py', unnamed),
    ]


def test_a_grant_for_one_module_does_not_cover_another():
    use = AccessUse(4, 'f', 'fetch_statuses')
    grant = AccessGrant('data/snap.py', None, frozenset({'fetch_statuses'}), 'test')

    assert violations({'data/other.py': [use]}, [grant]) == [('data/other.py', use)]


# ---------------------------------------------------------------------------
# Acceptance tests: the real tree (the whole dashboard package)
# ---------------------------------------------------------------------------

_PACKAGE_DIR = Path(__file__).resolve().parent.parent / 'src' / 'dashboard'

# 49 modules when this guard was written. A rename or a moved tree must fail
# loudly here rather than silently shrinking the scan.
_MIN_SCANNED_MODULES = 40

_GRANTS: tuple[AccessGrant, ...] = (
    AccessGrant(
        'data/task_snapshot.py', None,
        frozenset({'fetch_tasks', 'fetch_statuses', 'fetch_task_page'}),
        'the snapshot unit and its on-demand terminal window ARE the census '
        'and row access implementation',
    ),
    AccessGrant(
        'data/task_lookup.py', None, frozenset({'fetch_task'}),
        'the per-id miss path for a task the snapshot does not hold (PRD '
        'decision 12)',
    ),
    AccessGrant(
        'app.py', '_fanout_probe_completion', frozenset({'fetch_tasks'}),
        'the /healthz MCP fan-out liveness probe reads raw and uncached by '
        'design: a stored value would mask the wedge it detects (PRD '
        'decision 17). The ONE named exemption.',
    ),
)


def _scanned_modules() -> dict[str, Path]:
    """Every module of the dashboard package, keyed by its path under ``src/dashboard``.

    Raises rather than returning a shorter map when the glob comes up empty or
    short, or a granted module is missing, because a check that quietly stops
    checking is indistinguishable from a passing one.
    """
    modules = {
        path.relative_to(_PACKAGE_DIR).as_posix(): path
        for path in sorted(_PACKAGE_DIR.rglob('*.py'))
    }
    assert len(modules) >= _MIN_SCANNED_MODULES, (
        f'only {len(modules)} modules found under {_PACKAGE_DIR} — fewer than '
        f'the {_MIN_SCANNED_MODULES} this guard was written against'
    )
    missing = sorted({grant.module for grant in _GRANTS} - set(modules))
    assert not missing, f'granted module(s) missing from the scan: {missing}'
    return modules


def _real_uses() -> dict[str, list[AccessUse]]:
    return {
        module: find_access_path_uses(path.read_text())
        for module, path in _scanned_modules().items()
    }


def test_the_scan_covers_the_defining_module_and_every_granted_one():
    scanned = _scanned_modules()

    assert 'data/tasks.py' in scanned, (
        'the module that DEFINES the reads is scanned like any other; it needs '
        'no grant only because it defines them and never uses them'
    )
    assert {grant.module for grant in _GRANTS} <= set(scanned)


def test_every_task_read_has_one_access_path():
    found = violations(_real_uses(), _GRANTS)

    assert not found, (
        'A task read is used outside its access path. Each line below is a new '
        'route to a task datum, and PRD decision 17 exists to keep there being '
        'one. Before granting it, ask whether this caller should read through '
        'task_snapshot (the census and the rows, one cached unit) or '
        'task_lookup (one task by id). Only if neither can serve it is a '
        'grant the answer, and then it is a design decision to record in the '
        'PRD:\n'
        + '\n'.join(
            f'  {module}:{use.line} {use.qualname} {use.name} — would need '
            f'AccessGrant({module!r}, {use.qualname!r}, '
            f'frozenset({{{use.name!r}}}), reason=...)'
            for module, use in found
        )
    )


def test_every_grant_is_exercised():
    uses = _real_uses()
    stale = [
        f'{grant.module}::{grant.scope or "<whole module>"} {name}'
        for grant in _GRANTS
        for name in sorted(grant.names)
        if not any(
            use.name == name and grant.excuses(module, use)
            for module, module_uses in uses.items()
            for use in module_uses
        )
    ]

    assert not stale, (
        'grant(s) that excuse no real use — a stale grant silently widens the '
        'allowlist for whatever arrives next, so narrow or delete it:\n  '
        + '\n  '.join(stale)
    )


# ---------------------------------------------------------------------------
# The old-path census (sketch #14, second half): the census kinds and checkers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RetiredServedAsset:
    """GET *path* 404s, and no script tag of the parsed index.html loads it."""

    path: str
    retired_by: str


@dataclass(frozen=True, slots=True)
class RetiredClientBinding:
    """No served script binds *name* at top level, or exports it as ``window.<name>``."""

    name: str
    retired_by: str


@dataclass(frozen=True, slots=True)
class RetiredWireKey:
    """No entry of *endpoint*'s payload carries *key*."""

    endpoint: str
    key: str
    retired_by: str


def local_script_paths(index_html: str) -> list[str]:
    """The URL path of every ``<script src>`` this app serves, in document order."""
    collector = ScriptTagCollector()
    collector.feed(index_html)
    srcs = [urlsplit(src) for attrs in collector.script_attrs if (src := attrs.get('src'))]
    return [src.path for src in srcs if not src.netloc]


_SERVED_BUNDLE = Path(__file__).resolve().parent / 'js' / '_served_bundle.mjs'

_CLIENT_BINDINGS_DRIVER = (
    "import fs from 'node:fs';\n"
    f'import {{ topLevelBindings }} from {json.dumps(_SERVED_BUNDLE.as_uri())};\n'
    "const sources = JSON.parse(fs.readFileSync(0, 'utf8'));\n"
    'const out = Object.fromEntries(\n'
    '  Object.entries(sources).map(([name, src]) => [name, topLevelBindings(src, name)]),\n'
    ');\n'
    'process.stdout.write(JSON.stringify(out));\n'
)


def client_bindings(sources: Mapping[str, str]) -> dict[str, list[str]]:
    """Each script's top-level bindings and ``window.<name>`` exports, by filename.

    Parsed in node by ``js/_served_bundle.mjs::topLevelBindings`` with the
    Babel build index.html pins, so a ``.jsx`` is read as the browser reads it.
    """
    result = subprocess.run(
        [node_path(), '--input-type=module', '-e', _CLIENT_BINDINGS_DRIVER],
        input=json.dumps(dict(sources)),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, (
        f'the top-level-binding reader exited {result.returncode}:\n{result.stderr}'
    )
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Census checker unit tests (fixtures, not the real tree)
# ---------------------------------------------------------------------------


def test_client_bindings_reports_only_top_level_names():
    source = (
        'function TopFn() {\n'
        '  const inner = 1;\n'
        '  window.DF_INNER = inner;\n'
        '  return <div>{inner}</div>;\n'
        '}\n'
        "const TopConst = 'stringOnly';\n"
        'const { a, b: c } = window.DF_X;\n'
        'class TopClass {}\n'
        'window.DF_Y = { TopFn, TopClass };\n'
    )

    assert client_bindings({'fixture.jsx': source}) == {
        'fixture.jsx': ['TopFn', 'TopConst', 'a', 'c', 'TopClass', 'DF_Y'],
    }


def test_local_script_paths_reads_classic_and_babel_tags_but_not_comments():
    index_html = (
        '<!doctype html><html><head>\n'
        '<script src="https://unpkg.com/react@18.3.1/umd/react.development.js"></script>\n'
        '<script src="/static/redux/data.js?v=7"></script>\n'
        '<!-- <script src="/static/redux/retired.js?v=6"></script> -->\n'
        '<script type="text/babel" src="/static/redux/app.jsx?v=7"></script>\n'
        '</head><body></body></html>\n'
    )

    assert local_script_paths(index_html) == ['/static/redux/data.js', '/static/redux/app.jsx']


# ---------------------------------------------------------------------------
# Acceptance tests: the real census
# ---------------------------------------------------------------------------

_RETIRED: tuple[RetiredServedAsset | RetiredClientBinding | RetiredWireKey, ...] = (
    RetiredServedAsset(
        '/static/redux/task_status_counts.js',
        'PRD decision 16: the client bucketer; the Tasks header reads the served census',
    ),
    RetiredClientBinding(
        'DF_TASK_STATUS_COUNTS',
        "PRD decision 16: task_status_counts.js's export",
    ),
    RetiredWireKey(
        '/api/v2/dashboard/orchestrators', 'summary',
        'PRD decision 16: discovery measures no task count, so it claims none',
    ),
)


_Row = TypeVar('_Row')


def _retired(kind: type[_Row]) -> list[_Row]:
    rows = [row for row in _RETIRED if isinstance(row, kind)]
    assert rows, f'the census lists no {kind.__name__}; its check would pass vacuously'
    return rows


def test_no_retired_asset_is_served_or_loaded(client):
    loaded = local_script_paths(client.get('/static/redux/index.html').text)
    assert loaded, 'index.html loads no local script, so the absence below proves nothing'
    found = [
        f'  {row.path}: GET {status}, loaded by index.html: {row.path in loaded} '
        f'— retired by {row.retired_by}'
        for row in _retired(RetiredServedAsset)
        if (status := client.get(row.path).status_code) != 404 or row.path in loaded
    ]

    assert not found, 'a retired asset is still served or loaded:\n' + '\n'.join(found)


def test_no_served_script_binds_a_retired_client_name(client):
    served = {}
    for path in local_script_paths(client.get('/static/redux/index.html').text):
        resp = client.get(path)
        assert resp.status_code == 200, f'index.html loads {path}, which is not served'
        served[path] = resp.text
    bindings = client_bindings(served)
    unread = sorted(path for path in served if not bindings.get(path))
    assert served and not unread, (
        f'the binding reader saw no top-level name in {unread or "any script"}; '
        'every served script exports at least one, so the reader has gone blind'
    )
    found = [
        f'  {row.name} in {path} — retired by {row.retired_by}'
        for row in _retired(RetiredClientBinding)
        for path, names in bindings.items()
        if row.name in names
    ]

    assert not found, 'a served script binds a retired name again:\n' + '\n'.join(found)


@dataclass(frozen=True, slots=True)
class _WireSubstrate:
    """How to make *endpoint* serve at least one entry, and where its entries sit."""

    serving: Callable[[], AbstractContextManager[Any]]
    entries_at: str


_ONE_DISCOVERED_ORCHESTRATOR = {
    'pids': [4321],
    'prd': '/proj/dark-factory/prd.md',
    'label': '/proj/dark-factory/prd.md',
    'project_root': '/proj/dark-factory',
    'running': True,
    'started': 'Mar18',
}

_WIRE_SUBSTRATES: dict[str, _WireSubstrate] = {
    '/api/v2/dashboard/orchestrators': _WireSubstrate(
        lambda: patch(
            'dashboard.api.orchestrators.discover_orchestrators',
            new=AsyncMock(return_value=[_ONE_DISCOVERED_ORCHESTRATOR]),
        ),
        'ORCHESTRATORS',
    ),
}


def test_no_route_entry_carries_a_retired_wire_key(client):
    found = []
    for row in _retired(RetiredWireKey):
        substrate = _WIRE_SUBSTRATES[row.endpoint]
        with substrate.serving():
            resp = client.get(row.endpoint)
        assert resp.status_code == 200, f'{row.endpoint} answered {resp.status_code}'
        entries = resp.json()[substrate.entries_at]
        assert entries, f'{row.endpoint} served no {substrate.entries_at}, so the check is vacuous'
        found += [
            f'  {row.endpoint} {substrate.entries_at}[{index}] carries {row.key!r} '
            f'— retired by {row.retired_by}'
            for index, entry in enumerate(entries)
            if row.key in entry
        ]

    assert not found, 'a route entry carries a retired key again:\n' + '\n'.join(found)
