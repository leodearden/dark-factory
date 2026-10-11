"""Contract: App announces its tab, and each poll-scope map covers what its surface reads.

data.js polls ``CHROME_ENDPOINTS ∪ TAB_ENDPOINTS[activeTab]`` (task 5825). Three
things outside data.js decide whether that is safe, and none can be seen from
the node suite:

* app.jsx must actually tell data.js which tab is open. Without the
  announcement the loop polls everything forever, which is safe but saves
  nothing.
* Everything rendered on EVERY tab — the rail badges and topbar summary in
  ``App``, and the Toolbar's module-scope captures in shell.jsx — must read
  only endpoints in CHROME_ENDPOINTS. A chrome read of a tab-only endpoint
  would freeze silently on every other tab; an always-on entry nothing reads
  polls for nothing.
* Everything a tab renders must read only endpoints in that tab's
  TAB_ENDPOINTS entry. A missing path freezes that tab's data silently, which
  is how the orch tab's /scheduler and the overview tab's /merge-queue went
  unpolled until they were found by hand.

app.jsx is ``type="text/babel"`` and cannot run under node, so the WIRING is
asserted over its comment-stripped served source (the
test_tab_staleness_indicator.py / test_app_chrome_census.py idiom). The READ
SETS are not hand-listed. Each tab's components are walked from app.jsx's
``renderTab`` through every component and helper they reference, across the
redux .jsx modules. Every reader called over ``DF_DATA`` is then RUN in node over
a Proxy of ``DF_DATA`` that records which keys it touches. So a reader or
component that starts reading a new key changes this test's answer by itself.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from _dashboard_helpers import (
    destructure_bindings,
    extract_function_body,
    strip_js_comments,
    walk_balanced,
)
from _lock_chip_matrix import node_path

REDUX = Path(__file__).parent.parent / 'src' / 'dashboard' / 'static' / 'redux'

_MODULE_DESTRUCTURE = re.compile(r'const\s*\{([^{}]*)\}\s*=\s*window\.(DF_\w+)\s*;')

_CLASSIC_SCRIPT = re.compile(r'<script\s+src="/static/redux/([A-Za-z0-9_.-]+\.js)(?:\?[^"]*)?"\s*></script>')

_CHROME_KEY_FLOOR = {
    'TASKS_SNAPSHOT', 'MERGE_QUEUE', 'MEMORY_STATUS', 'ORCHESTRATORS',
    'RECON_STATE', 'ESCALATIONS', 'COSTS', 'PROJECTS', 'AGENTS',
}
"""What the chrome reads today, as a literal: a non-vacuity floor, not the contract."""


def _app_code(app_jsx_body: str) -> str:
    return strip_js_comments(app_jsx_body)


def _app_body(app_jsx_body: str) -> str:
    body = extract_function_body(_app_code(app_jsx_body), 'App')
    assert body, "could not extract App()'s body from app.jsx"
    return body


def _module_scope(app_jsx_body: str) -> str:
    code = _app_code(app_jsx_body)
    return code.replace(_app_body(app_jsx_body), '')


def _effects(body: str) -> list[tuple[int, str]]:
    """Every ``uE(...)`` call in *body*, with its offset, parens included."""
    effects = []
    for match in re.finditer(r'\buE\(', body):
        call = walk_balanced(body, match.end() - 1, '(', ')')
        assert call, f'an effect at offset {match.start()} is never closed'
        effects.append((match.start(), call))
    return effects


# ---------------------------------------------------------------------------
# (a) App announces its tab
# ---------------------------------------------------------------------------


def test_app_destructures_scope_polling_from_the_loader_without_fallback(
    app_jsx_body: str,
) -> None:
    scope = _module_scope(app_jsx_body)
    loader = [
        bindings for bindings, global_name in _MODULE_DESTRUCTURE.findall(scope)
        if global_name == 'DF_DATA_LOADER'
    ]
    assert loader, 'app.jsx must destructure window.DF_DATA_LOADER at module scope'
    assert any(
        ('scopePollingToTab', 'scopePollingToTab') in destructure_bindings(bindings)
        for bindings in loader
    ), f'app.jsx does not destructure scopePollingToTab from window.DF_DATA_LOADER: {loader}'
    assert not re.search(r'window\.DF_DATA_LOADER\s*(\|\||&&|\?\?)', _app_code(app_jsx_body)), (
        'a fallback on window.DF_DATA_LOADER turns a load-order regression into '
        'a page that silently polls everything forever'
    )


def test_app_announces_the_tab_in_a_tab_effect_before_the_chip_refresh(
    app_jsx_body: str,
) -> None:
    effects = _effects(_app_body(app_jsx_body))
    scoping = [(at, call) for at, call in effects if re.search(r'\bscopePollingToTab\(', call)]
    assert len(scoping) == 1, (
        f'App must call scopePollingToTab in exactly one effect; found {len(scoping)}'
    )
    scope_at, scope_call = scoping[0]
    assert re.search(r',\s*\[\s*tab\s*\]\s*\)$', scope_call), (
        f'the scope effect must depend on exactly [tab]: {scope_call}'
    )
    _assert_the_tab_is_announced_at_its_resolved_window(scope_call)
    refreshing = [at for at, call in effects if 'DF_REFRESH(win)' in call]
    assert len(refreshing) == 1, 'App must keep exactly one DF_REFRESH(win) effect'
    assert scope_at < refreshing[0], (
        'the scope effect must be declared BEFORE the DF_REFRESH(win) effect, so '
        'the mount-time chip refresh already runs against the scoped set'
    )


def _assert_the_tab_is_announced_at_its_resolved_window(scope_call: str) -> None:
    """The scope effect resolves the tab's window, sets it, and announces both.

    A separate setWin effect only SCHEDULES the reset, so an announcement made
    beside it would still see the previous tab's window in data.js, and fetch a
    newly-needed windowed endpoint at a window the new tab does not offer.
    """
    resolved = re.search(r'\bconst\s+(\w+)\s*=\s*windowForTab\(\s*tab\s*,\s*win\s*\)', scope_call)
    assert resolved, (
        'the scope effect must resolve the window the tab will show, '
        f'const <name> = windowForTab(tab, win): {scope_call}'
    )
    tab_win = resolved.group(1)
    assert re.search(rf'\bsetWin\(\s*{tab_win}\s*\)', scope_call), (
        f'the scope effect must set the window it resolved, setWin({tab_win}): {scope_call}'
    )
    assert re.search(rf'\bscopePollingToTab\(\s*tab\s*,\s*{tab_win}\s*\)', scope_call), (
        f'the scope effect must announce the tab AT that window, '
        f'scopePollingToTab(tab, {tab_win}): {scope_call}'
    )


# ---------------------------------------------------------------------------
# The reader driver: what each reader called over DF_DATA actually touches
# ---------------------------------------------------------------------------

_READ_DRIVER = r"""
const vm = require('vm');
const fs = require('fs');
const path = require('path');
const [redux, scriptsJson, readersJson] = process.argv.slice(1);
const context = vm.createContext({ window: {}, console });
for (const src of JSON.parse(scriptsJson)) {
  new vm.Script(fs.readFileSync(path.join(redux, src), 'utf8'), { filename: src }).runInContext(context);
}
const { window } = context;
const loader = window.DF_DATA_LOADER;
const keyToPath = {};
for (const [url, specs] of Object.entries(loader.endpointsFor('24h'))) {
  const pollPath = loader.pollKey(url);
  window.DF_DATA.__receipt[pollPath] = {
    servedAt: '2026-10-03T12:00:00+00:00', receivedAt: Date.now(), window: null,
  };
  for (const key of Object.keys(specs)) keyToPath[key] = pollPath;
}
const SEED_PROJECT = 'df';
const seeded = window.DF_DATUM.unknownDatum('seeded by the read-set driver');
window.DF_DATA.TASKS_SNAPSHOT = {
  [SEED_PROJECT]: { census: seeded, rows: seeded, in_progress_live: 0, in_progress_stranded: 0, skew_seconds: 0 },
};
function recording(touched, rows) {
  const snapshot = new Proxy(window.DF_DATA.TASKS_SNAPSHOT, {
    get(target, project, receiver) {
      const entry = Reflect.get(target, project, receiver);
      if (!entry || typeof entry !== 'object') return entry;
      return new Proxy(entry, {
        get(e, field, r) {
          if (field === 'rows') rows.read = true;
          return Reflect.get(e, field, r);
        },
      });
    },
  });
  return new Proxy(window.DF_DATA, {
    get(target, prop, receiver) {
      if (typeof prop === 'string') touched.add(prop);
      return prop === 'TASKS_SNAPSHOT' ? snapshot : Reflect.get(target, prop, receiver);
    },
  });
}
const reads = {};
const missing = [];
for (const [globalName, exportName] of JSON.parse(readersJson)) {
  const name = `${globalName}.${exportName}`;
  const reader = (window[globalName] || {})[exportName];
  if (typeof reader !== 'function') {
    missing.push(name);
    continue;
  }
  reads[name] = [null, SEED_PROJECT].map(arg => {
    const touched = new Set();
    const rows = { read: false };
    let error = null;
    try {
      reader(recording(touched, rows), arg);
    } catch (err) {
      error = String(err);
    }
    return { arg, touched: [...touched], readsRows: rows.read, error };
  });
}
process.stdout.write(JSON.stringify({
  reads,
  missing,
  keyToPath,
  chrome: [...window.DF_ENDPOINT_STALENESS.CHROME_ENDPOINTS],
  tabs: window.DF_ENDPOINT_STALENESS.TAB_ENDPOINTS,
}));
"""
"""Each reader runs twice, with ``null`` and with a seeded project as its second
argument. That covers both a whole-fleet call and a per-project call. A call
may throw on an argument shape it does not take; the keys it touched before
throwing still count. Reading ``.rows`` off a ``TASKS_SNAPSHOT`` entry is
recorded separately, because only rows need the full /tasks render."""


def _drive_readers(index_html_body: str, readers: list[tuple[str, str]]) -> dict:
    scripts = _CLASSIC_SCRIPT.findall(index_html_body)
    assert scripts, 'extracted no classic /static/redux/*.js scripts from index.html'
    result = subprocess.run(
        [node_path(), '-e', _READ_DRIVER, str(REDUX), json.dumps(scripts), json.dumps(readers)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f'the read driver exited {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
    driven = json.loads(result.stdout)
    assert not driven['missing'], (
        f'{driven["missing"]} are called over DF_DATA but are not functions on their '
        'window.DF_* module, so this test cannot run them'
    )
    return driven


def _data_keys(keys: set[str]) -> set[str]:
    return {key for key in keys if not key.startswith('__')}


# ---------------------------------------------------------------------------
# (b) the chrome's read set IS CHROME_ENDPOINTS
# ---------------------------------------------------------------------------


def _readers_app_calls_over_dd(app_jsx_body: str) -> list[tuple[str, str]]:
    """``(window global, export)`` for every reader App calls with ``DD`` as an argument."""
    locals_to_export: dict[str, tuple[str, str]] = {}
    for bindings, global_name in _MODULE_DESTRUCTURE.findall(_module_scope(app_jsx_body)):
        for canonical, local in destructure_bindings(bindings):
            locals_to_export[local] = (global_name, canonical)
    called = sorted(set(re.findall(r'\b(\w+)\(\s*DD\s*[,)]', _app_body(app_jsx_body))))
    unresolved = [name for name in called if name not in locals_to_export]
    assert not unresolved, (
        f'App calls {unresolved} over DD, but app.jsx destructures none of them '
        'from a window.DF_* module, so this test cannot run them'
    )
    return [locals_to_export[name] for name in called]


def _chrome_reads(app_jsx_body: str, shell_jsx_body: str, index_html_body: str) -> tuple[set[str], dict]:
    readers = _readers_app_calls_over_dd(app_jsx_body)
    assert readers, 'App calls no reader over DD — the extraction has gone stale'
    driven = _drive_readers(index_html_body, readers)

    keys = set(re.findall(r'\bDD\.(\w+)', _app_body(app_jsx_body)))
    for name, variants in driven['reads'].items():
        fleet = next(v for v in variants if v['arg'] is None)
        assert fleet['error'] is None, (
            f'App calls {name}(DD, null), and over the seeded DF_DATA it threw: {fleet["error"]}'
        )
        keys |= set(fleet['touched'])
    keys |= set(re.findall(
        r'^const\s+\w+\s*=\s*window\.DF_DATA\.(\w+)', strip_js_comments(shell_jsx_body), re.M,
    ))
    return _data_keys(keys), driven


def test_the_chrome_reads_exactly_the_chrome_endpoints(
    app_jsx_body: str, shell_jsx_body: str, index_html_body: str,
) -> None:
    keys, driven = _chrome_reads(app_jsx_body, shell_jsx_body, index_html_body)

    assert keys >= _CHROME_KEY_FLOOR, (
        f'the chrome read set lost {sorted(_CHROME_KEY_FLOOR - keys)}; the '
        'extraction has gone stale, or a reader moved'
    )
    unmapped = sorted(key for key in keys if key not in driven['keyToPath'])
    assert not unmapped, f'the chrome reads {unmapped}, which no endpointsFor() row serves'
    chrome = set(driven['chrome'])
    paths = {driven['keyToPath'][key] for key in keys}
    outside = sorted(
        f'{key} ({driven["keyToPath"][key]})' for key in keys
        if driven['keyToPath'][key] not in chrome
    )
    assert not outside, (
        f'the chrome reads {outside}, which CHROME_ENDPOINTS does not poll — '
        'that badge would freeze on every tab that does not list its endpoint'
    )
    assert paths == chrome, (
        f'CHROME_ENDPOINTS polls {sorted(chrome - paths)} on every tab, but no '
        'chrome surface reads it'
    )


# ---------------------------------------------------------------------------
# (c) every tab's read set is inside its TAB_ENDPOINTS entry
# ---------------------------------------------------------------------------

_JSX_DESTRUCTURE = re.compile(r'^const\s*\{([^{}]*)\}\s*=\s*window\.(DF_\w+)\s*;', re.M)
_JSX_NAMESPACE = re.compile(r'^const\s+(\w+)\s*=\s*window\.(DF_\w+)\s*;', re.M)
_JSX_EXPORT = re.compile(r'^window\.(DF_\w+)\s*=\s*\{([^{}]*)\}\s*;', re.M)
_JSX_EXPORT_MEMBER = re.compile(r'^window\.(DF_\w+)\.(\w+)\s*=\s*(\w+)\s*;', re.M)
_JSX_TOP_FUNCTION = re.compile(
    r'^(?:function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*React\.memo\(\s*function\s+(\w+)\s*\()', re.M,
)
_JSX_REFERENCE = re.compile(r'(?<![\w$.])([A-Za-z_$][\w$]*)(?:\.([A-Za-z_$][\w$]*))?')


@dataclass(frozen=True)
class _JsxModule:
    code: str
    functions: dict[str, str]
    """Top-level binding -> the name its ``function`` declaration carries."""
    imports: dict[str, tuple[str, str]]
    """Local name -> ``(window global, export)`` it was destructured from."""
    namespaces: dict[str, str]
    """Local name -> the window global it holds whole (``const C = window.DF_CHARTS``)."""


@dataclass(frozen=True)
class _Function:
    module: str
    name: str


@dataclass
class _Reads:
    keys: set[str]
    readers: set[tuple[str, str]]


class _JsxGraph:
    """The redux .jsx modules, with their top-level functions linked across files.

    Resolves a name the way the page does: a module's own top-level function
    first, then what it destructured or holds from a ``window.DF_*`` global.
    That global is either another .jsx module's export (followed) or a classic
    script (a reader, RUN in node by the driver). The same name can be declared
    in several modules (``fmtAge``, ``useOpenSet``), so nothing resolves by
    bare name across files.
    """

    def __init__(self, redux: Path) -> None:
        self.modules: dict[str, _JsxModule] = {}
        self.exports: dict[tuple[str, str], _Function] = {}
        for path in sorted(redux.glob('*.jsx')):
            code = strip_js_comments(path.read_text())
            functions = {
                m.group(1) or m.group(2): m.group(1) or m.group(3)
                for m in _JSX_TOP_FUNCTION.finditer(code)
            }
            imports = {
                local: (global_name, canonical)
                for bindings, global_name in _JSX_DESTRUCTURE.findall(code)
                for canonical, local in destructure_bindings(bindings)
            }
            self.modules[path.name] = _JsxModule(code, functions, imports, dict(_JSX_NAMESPACE.findall(code)))
            for global_name, body in _JSX_EXPORT.findall(code):
                for exported, local in destructure_bindings(body):
                    self.exports[(global_name, exported)] = _Function(path.name, local)
            for global_name, exported, local in _JSX_EXPORT_MEMBER.findall(code):
                self.exports[(global_name, exported)] = _Function(path.name, local)
        self._jsx_globals = {global_name for global_name, _ in self.exports}

    def rendered_tabs(self) -> dict[str, _Function]:
        """Tab id -> the component ``App``'s ``renderTab`` returns for it."""
        render = extract_function_body(self.modules['app.jsx'].code, 'renderTab')
        tabs = {}
        for tab, component in re.findall(r"case\s+'([\w-]+)'\s*:\s*return\s*<(\w+)", render):
            target = self._resolve('app.jsx', component, '')
            assert isinstance(target, _Function), f'renderTab renders <{component}> for {tab!r}, which resolves to no .jsx function'
            tabs[tab] = target
        assert tabs, 'extracted no tabs from renderTab — the extraction has gone stale'
        return tabs

    def reads_from(self, root: _Function) -> _Reads:
        """Every ``DF_DATA`` key read, and reader called over it, reachable from *root*."""
        reads = _Reads(set(), set())
        aliases_seen: dict[_Function, set[str]] = {}
        work: list[tuple[_Function, frozenset[str]]] = [(root, frozenset())]
        while work:
            function, param_aliases = work.pop()
            seen = aliases_seen.get(function)
            if seen is not None and param_aliases <= seen:
                continue
            aliases_seen[function] = (seen or set()) | param_aliases
            work.extend(self._scan(function, aliases_seen[function], reads))
        return reads

    def _scan(self, function: _Function, param_aliases: set[str], reads: _Reads) -> list[tuple[_Function, frozenset[str]]]:
        module = self.modules[function.module]
        body = extract_function_body(module.code, module.functions[function.name])
        aliases = {name for name, held in module.namespaces.items() if held == 'DF_DATA'} | param_aliases
        data = '(?:' + '|'.join([re.escape(alias) for alias in sorted(aliases)] + [r'window\.DF_DATA']) + ')'
        reads.keys |= set(re.findall(rf'(?<![\w$.]){data}\.([A-Z][A-Z0-9_]*)\b', body))
        over_data = set(re.findall(rf'(?<![\w$.])([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)?)\(\s*{data}\s*[,)]', body))
        reached = []
        for head, member in set(_JSX_REFERENCE.findall(body)):
            target = self._resolve(function.module, head, member)
            spelled = f'{head}.{member}' if member and head in module.namespaces else head
            if isinstance(target, _Function) and target != function:
                param = self._first_param(target) if spelled in over_data else None
                reached.append((target, frozenset([param]) if param else frozenset()))
            elif isinstance(target, tuple) and spelled in over_data:
                reads.readers.add(target)
        return reached

    def _resolve(self, module_name: str, head: str, member: str) -> _Function | tuple[str, str] | None:
        module = self.modules[module_name]
        if member and module.namespaces.get(head, 'DF_DATA') != 'DF_DATA':
            return self._resolve_global(module.namespaces[head], member)
        if head in module.functions:
            return _Function(module_name, head)
        if head in module.imports:
            return self._resolve_global(*module.imports[head])
        return None

    def _resolve_global(self, global_name: str, export: str) -> _Function | tuple[str, str] | None:
        if global_name not in self._jsx_globals:
            return (global_name, export)
        target = self.exports.get((global_name, export))
        if target and target.name in self.modules[target.module].functions:
            return target
        return None

    def _first_param(self, function: _Function) -> str | None:
        declared = self.modules[function.module].functions[function.name]
        match = re.search(
            rf'\bfunction\s+{re.escape(declared)}\s*\(\s*([A-Za-z_$][\w$]*)\s*[,)]',
            self.modules[function.module].code,
        )
        return match.group(1) if match else None


def _tab_read_paths(index_html_body: str) -> tuple[dict[str, set[str]], dict]:
    """Tab id -> the endpoint paths that tab must poll, and the driver's output.

    /tasks is required only by a tab that reads task ROWS. Its census, banner
    lists and project count are served unchanged by the chrome's
    ``?projection=census`` poll, which runs on every tab.
    """
    graph = _JsxGraph(REDUX)
    reads = {tab: graph.reads_from(root) for tab, root in graph.rendered_tabs().items()}
    driven = _drive_readers(index_html_body, sorted(set().union(*(r.readers for r in reads.values()))))
    key_to_path = driven['keyToPath']
    rows_path = key_to_path['TASKS_SNAPSHOT']
    required = {}
    for tab, read in reads.items():
        keys = set(read.keys)
        reads_rows = False
        for reader in read.readers:
            for variant in driven['reads'][f'{reader[0]}.{reader[1]}']:
                keys |= set(variant['touched'])
                reads_rows |= variant['readsRows']
        keys = _data_keys(keys)
        unmapped = sorted(key for key in keys if key not in key_to_path)
        assert not unmapped, f'tab {tab!r} reads {unmapped}, which no endpointsFor() row serves'
        paths = {key_to_path[key] for key in keys}
        if not reads_rows:
            paths.discard(rows_path)
        required[tab] = paths
    return required, driven


_DASH = '/api/v2/dashboard'
_TAB_READ_FLOOR = {
    'orch': {f'{_DASH}/scheduler', f'{_DASH}/tasks'},
    'overview': {f'{_DASH}/merge-queue'},
    'tasks': {f'{_DASH}/tasks'},
}
"""Reads the derivation must see, as literals: a non-vacuity floor, not the contract.

orch's Deps/Locks cells read ``DF.SCHEDULER``. overview's ``LiveFeed`` reads
``MERGE_QUEUE`` through ``buildFeedEntries(window.DF_DATA)``. Both were missing
from TAB_ENDPOINTS until found by hand. orch and tasks render task rows."""


def test_every_tab_polls_every_endpoint_it_reads(index_html_body: str) -> None:
    required, driven = _tab_read_paths(index_html_body)
    for tab, paths in required.items():
        listed = driven['tabs'].get(tab)
        assert listed is not None, f'renderTab renders tab {tab!r}, which has no TAB_ENDPOINTS entry'
        missing = sorted(paths - set(listed))
        assert not missing, (
            f'tab {tab!r} reads {missing}, which TAB_ENDPOINTS.{tab} does not list — '
            'data.js polls only CHROME_ENDPOINTS and that list while the tab is open, '
            'so this data would sit frozen at its last value'
        )


def test_the_tab_read_derivation_sees_what_was_found_by_hand(index_html_body: str) -> None:
    required, _ = _tab_read_paths(index_html_body)
    for tab, floor in _TAB_READ_FLOOR.items():
        assert required[tab] >= floor, (
            f'the derivation for tab {tab!r} lost {sorted(floor - required[tab])}; '
            'the component walk or a reader has gone stale'
        )
    assert f'{_DASH}/tasks' not in required['overview'], (
        'overview reads only the census, which the chrome polls as '
        '?projection=census; the derivation must not demand the full rows render there'
    )
