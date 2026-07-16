// Module-contract tests for runtime_format.js — a plain-JS (no JSX/Babel)
// module holding the pure offline-'—' degradation formatters used to render
// warm-lane runtime fields (loops/attempts/started/lane/phase/lane_state) in
// the Orchestrators tab (OrchTab in tabs.jsx) and the Tasks tab's TaskDetail
// (tab_tasks.jsx). Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper that
// surfaces this suite in CI via its `**/*.test.mjs` glob — no wrapper change
// needed for this new file).
//
// runtime_format.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { rtCell } from '...'` would come back undefined. We therefore
// default-import the module and destructure instead (mirrors
// prd_grouping.test.mjs / graph_layout.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import fmt from '../../src/dashboard/static/redux/runtime_format.js';

const { rtCell, rtAge } = fmt;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/runtime_format.js';
const EXPECTED_FUNCTION_NAMES = ['rtCell', 'rtAge'];

test('default-imported module exposes the runtime-format functions', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof fmt[name], 'function', `fmt.${name} should be a function`);
  }
});

test('module also assigns window.DF_RUNTIME_FMT (browser dual-export)', () => {
  // Shim a bare browser-like global before requiring the module fresh via
  // CommonJS require, so the module body's `if (typeof window !== 'undefined')`
  // branch executes against our shim.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    // Node's ESM loader resolves a CommonJS module (no package.json/type in
    // this repo) by delegating to the CJS loader and populating the shared
    // require.cache — so by the time this test runs, the top-level `import
    // fmt from ...` above has ALREADY cached this exact file. A plain
    // require() here would return that cached module.exports without
    // re-running the module body, meaning the dual-export line would never
    // see our globalThis.window shim. Busting the cache entry forces a
    // fresh execution against the now-shimmed window.
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_RUNTIME_FMT, 'window.DF_RUNTIME_FMT was not set');

    // The fresh require() and the top-level import() produce two distinct
    // API object instances (separate module executions), so we compare
    // structurally — same set of exported names, each a function — rather
    // than asserting reference/deep equality against the ESM-imported `fmt`.
    assert.deepEqual(
      Object.keys(globalThis.window.DF_RUNTIME_FMT).sort(),
      EXPECTED_FUNCTION_NAMES.slice().sort(),
    );
    assert.deepEqual(Object.keys(required).sort(), EXPECTED_FUNCTION_NAMES.slice().sort());
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(typeof globalThis.window.DF_RUNTIME_FMT[name], 'function');
    }
  } finally {
    delete globalThis.window;
  }
});

// ---------------------------------------------------------------------------
// rtCell — v == null (covers both null and undefined, i.e. the
// offline/per-task-read-error shape) -> em-dash; any other value (including
// an honest 0) passes through unchanged.
// ---------------------------------------------------------------------------

test('rtCell: null renders as the em-dash', () => {
  assert.equal(rtCell(null), '—');
});

test('rtCell: undefined renders as the em-dash', () => {
  assert.equal(rtCell(undefined), '—');
});

test('rtCell: an honest zero is preserved, not dashed', () => {
  assert.equal(rtCell(0), 0);
});

test('rtCell: a positive number passes through unchanged', () => {
  assert.equal(rtCell(3), 3);
});

test('rtCell: a lane string passes through unchanged', () => {
  assert.equal(rtCell('_lane-7'), '_lane-7');
});

test('rtCell: a lane_state string passes through unchanged', () => {
  assert.equal(rtCell('assigned'), 'assigned');
});

// ---------------------------------------------------------------------------
// rtAge — null (or undefined) -> em-dash; otherwise `${minutes}m` (an honest
// 0 renders as "0m", not dashed).
// ---------------------------------------------------------------------------

test('rtAge: null renders as the em-dash', () => {
  assert.equal(rtAge(null), '—');
});

test('rtAge: undefined renders as the em-dash', () => {
  assert.equal(rtAge(undefined), '—');
});

test('rtAge: zero renders as an honest "0m", not dashed', () => {
  assert.equal(rtAge(0), '0m');
});

test('rtAge: a positive age renders as "<minutes>m"', () => {
  assert.equal(rtAge(14), '14m');
});
