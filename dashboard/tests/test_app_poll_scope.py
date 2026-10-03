"""Contract: App announces its tab to the poll loop, and the chrome reads exactly CHROME_ENDPOINTS.

data.js polls ``CHROME_ENDPOINTS ∪ TAB_ENDPOINTS[activeTab]`` (task 5825). Two
things outside data.js decide whether that is safe, and neither can be seen
from the node suite:

* app.jsx must actually tell data.js which tab is open. Without the
  announcement the loop polls everything forever, which is safe but saves
  nothing.
* Everything rendered on EVERY tab — the rail badges and topbar summary in
  ``App``, and the Toolbar's module-scope captures in shell.jsx — must read
  only endpoints in CHROME_ENDPOINTS. A chrome read of a tab-only endpoint
  would freeze silently on every other tab; an always-on entry nothing reads
  polls for nothing.

app.jsx is ``type="text/babel"`` and cannot run under node, so the WIRING is
asserted over its comment-stripped served source (the
test_tab_staleness_indicator.py / test_app_chrome_census.py idiom). The READ SET
is not hand-listed: every reader App calls over ``DD`` is RUN in node over a
Proxy of ``DF_DATA`` that records which keys it touches, so a reader that
starts reading a new key changes this test's answer by itself.
"""

from __future__ import annotations

import json
import re
import subprocess
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
    scoping = [
        (at, call) for at, call in effects
        if re.search(r'\bscopePollingToTab\(\s*tab\s*\)', call)
    ]
    assert len(scoping) == 1, (
        f'App must call scopePollingToTab(tab) in exactly one effect; found {len(scoping)}'
    )
    scope_at, scope_call = scoping[0]
    assert re.search(r',\s*\[\s*tab\s*\]\s*\)$', scope_call), (
        f'the scope effect must depend on exactly [tab]: {scope_call}'
    )
    refreshing = [at for at, call in effects if 'DF_REFRESH(win)' in call]
    assert len(refreshing) == 1, 'App must keep exactly one DF_REFRESH(win) effect'
    assert scope_at < refreshing[0], (
        'the scope effect must be declared BEFORE the DF_REFRESH(win) effect, so '
        'the mount-time chip refresh already runs against the scoped set'
    )


# ---------------------------------------------------------------------------
# (b) the chrome's read set IS CHROME_ENDPOINTS
# ---------------------------------------------------------------------------

_CHROME_READ_DRIVER = r"""
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
const reads = {};
for (const [globalName, exportName] of JSON.parse(readersJson)) {
  const touched = new Set();
  const recording = new Proxy(window.DF_DATA, {
    get(target, prop, receiver) {
      if (typeof prop === 'string') touched.add(prop);
      return Reflect.get(target, prop, receiver);
    },
  });
  window[globalName][exportName](recording, null);
  reads[`${globalName}.${exportName}`] = [...touched];
}
process.stdout.write(JSON.stringify({
  reads,
  keyToPath,
  chrome: [...window.DF_ENDPOINT_STALENESS.CHROME_ENDPOINTS],
}));
"""


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
    scripts = _CLASSIC_SCRIPT.findall(index_html_body)
    assert scripts, 'extracted no classic /static/redux/*.js scripts from index.html'
    result = subprocess.run(
        [node_path(), '-e', _CHROME_READ_DRIVER, str(REDUX), json.dumps(scripts), json.dumps(readers)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, (
        f'the chrome read driver exited {result.returncode}\n'
        f'--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}'
    )
    driven = json.loads(result.stdout)

    keys = set(re.findall(r'\bDD\.(\w+)', _app_body(app_jsx_body)))
    for touched in driven['reads'].values():
        keys |= set(touched)
    keys |= set(re.findall(
        r'^const\s+\w+\s*=\s*window\.DF_DATA\.(\w+)', strip_js_comments(shell_jsx_body), re.M,
    ))
    return {key for key in keys if not key.startswith('__')}, driven


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
