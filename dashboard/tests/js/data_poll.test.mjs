// Module-contract tests for data.js — the real-data poll loader for the Dark
// Factory dashboard (window.DF_DATA, the 3s setInterval poll loop, and —
// this task — a per-endpoint in-flight guard, error backoff, and jitter on
// top of it). Run via `node --test` (see dashboard/tests/test_graph_layout_js.py
// for the pytest wrapper that surfaces this suite in CI — it globs
// **/*.test.mjs under dashboard/tests/js/, so this new file needs no new
// wrapper).
//
// data.js has no package.json in the repo, so it resolves as CommonJS
// (`module.exports = <object>`), same as the other redux/*.js modules.
//
// Unlike those siblings, a static ESM `import` of data.js is not an option
// here, not even for the module-contract check: data.js assigns
// `window.DF_DATA = {...}` at module scope, and an ESM `import` statement's
// target module body runs BEFORE the importing file's own body — so
// `globalThis.window` would still be unset when data.js's top level runs,
// throwing `ReferenceError: window is not defined`. Every test in this file
// therefore goes through the `loadDataJs()` helper below, which shims
// `globalThis.window` FIRST and only then loads the module via
// `createRequire(import.meta.url)` (mirrors runtime_format.test.mjs:33-45).
//
// data.js's module scope also unconditionally called `refreshDFData()` and
// `setInterval(...)`. Under node, with no shim at all, that would throw
// before even reaching the interval; but even with window/fetch shimmed it
// would fire real (stubbed) fetches and leave a live timer holding the
// process open — hanging `node --test`. loadDataJs() defends against that
// unconditionally (see its comment below) so this suite can safely run
// against pre-seam data.js too, which is what step-1's RED depends on.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/data.js';
const EXPECTED_FUNCTION_NAMES = [
  'endpointsFor',
  'applyKey',
  'refreshOne',
  'refreshDFData',
  'pollTick',
  'startPolling',
  'createPollState',
];

// Full DF_DATA key set (data.js:41-127) — initialised so the first render
// before fetch completes cannot crash any component reading DF_DATA.*.
const EXPECTED_DF_DATA_KEYS = [
  'PROJECTS', 'AGENTS', 'ORCHESTRATORS', 'ORCHESTRATORS_SPARK',
  'ACTIVE_TASKS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS', 'DONE_COUNTS',
  'PERFORMANCE', 'MEMORY_STATUS', 'MEMORY_TIMESERIES', 'MEMORY_OPS_BREAKDOWN',
  'RECON_STATE', 'MERGE_QUEUE', 'COSTS', 'BURNDOWN', 'BURNDOWN_BY_PROJECT',
  'CURATOR_STATE', 'ESCALATIONS', 'ESCALATION_ANALYTICS', 'SCHEDULER',
];

// Loads data.js fresh against a shimmed browser-ish global. Installs
// `globalThis.window` (a bare object recording dispatched events) and a
// counting `globalThis.fetch` BEFORE requiring, then busts the require
// cache so each call re-executes data.js's module body from scratch — the
// module body only runs once per require otherwise, which would leave later
// callers seeing a stale `window.DF_DATA` / stale flow-control singleton
// from a previous test's shim (mirrors runtime_format.test.mjs:33-49).
//
// Deliberately does NOT set `globalThis.document`: index.html loads data.js
// as a classic script where `document` always exists, but every test in
// this file runs under node, so the auto-start guard must see no `document`
// and stay inert, exactly like the real non-browser (node --test)
// environment.
//
// Also wraps `globalThis.setInterval` for the duration of the require and
// clears any interval it captures before returning. This is defensive
// rather than load-bearing for the seamed implementation (which gates
// auto-start on `document` and never calls setInterval here at all), but it
// means step-1's RED run — against pre-seam data.js, which calls
// setInterval unconditionally — cannot hang node --test: the interval is
// recorded (so the "no live timer" assertion still fails honestly) and then
// cleared immediately, regardless of what the caller asserts.
function loadDataJs({ fetchStub } = {}) {
  const events = [];
  const fetchCalls = [];
  const intervalCalls = [];
  const win = { dispatchEvent: ev => events.push(ev) };
  globalThis.window = win;
  globalThis.fetch = (url, init) => {
    fetchCalls.push({ url, init });
    if (fetchStub) return fetchStub(url, init);
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };

  const require = createRequire(import.meta.url);
  const resolved = require.resolve(MODULE_SPECIFIER);
  delete require.cache[resolved];

  const originalSetInterval = globalThis.setInterval;
  globalThis.setInterval = (...args) => {
    const handle = originalSetInterval(...args);
    intervalCalls.push(handle);
    return handle;
  };

  let api;
  try {
    api = require(MODULE_SPECIFIER);
  } finally {
    globalThis.setInterval = originalSetInterval;
    for (const handle of intervalCalls) clearInterval(handle);
  }

  return { api, window: win, events, fetchCalls, intervalCalls };
}

test('default-imported module exposes the poll-loader functions', () => {
  const { api } = loadDataJs();
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof api[name], 'function', `api.${name} should be a function`);
  }
});

test('module also assigns window.DF_REFRESH / window.__DF_PAUSE / window.DF_DATA (browser dual-export)', () => {
  const { api, window: win } = loadDataJs();

  assert.equal(win.DF_REFRESH, api.refreshDFData, 'window.DF_REFRESH should be the exported refreshDFData');
  assert.equal(win.__DF_PAUSE, false, 'window.__DF_PAUSE should default to false');

  assert.ok(win.DF_DATA, 'window.DF_DATA was not set');
  for (const key of EXPECTED_DF_DATA_KEYS) {
    assert.ok(
      Object.prototype.hasOwnProperty.call(win.DF_DATA, key),
      `DF_DATA.${key} was not initialised`,
    );
  }
});

test('no auto-start outside a browser: loading with no `document` global fires zero fetches and leaves no live timer', () => {
  assert.equal(typeof globalThis.document, 'undefined', 'test environment must not already define document');

  const { fetchCalls, intervalCalls } = loadDataJs();

  assert.equal(fetchCalls.length, 0, 'loading data.js under node must not fire any fetches');
  assert.equal(intervalCalls.length, 0, 'loading data.js under node must not start a live timer');
});
