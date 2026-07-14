// Module-contract tests for prd_grouping.js — a plain-JS (no JSX/Babel)
// module holding the pure PRD-grouping-view logic for the Tasks tab's
// "group by PRD" view. Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper that
// surfaces this suite in CI via its `**/*.test.mjs` glob — no wrapper
// change needed for this new file).
//
// prd_grouping.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { prdTitle } from '...'` would come back undefined. We therefore
// default-import the module and destructure instead (mirrors
// graph_layout.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import grouping from '../../src/dashboard/static/redux/prd_grouping.js';

const { prdTitle } = grouping;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/prd_grouping.js';
const EXPECTED_FUNCTION_NAMES = [
  'prdTitle',
];

test('default-imported module exposes the PRD-grouping functions', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof grouping[name], 'function', `grouping.${name} should be a function`);
  }
});

test('module also assigns window.DF_PRD_GROUPING (browser dual-export)', () => {
  // Shim a bare browser-like global before requiring the module fresh via
  // CommonJS require, so the module body's `if (typeof window !== 'undefined')`
  // branch executes against our shim.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    // Node's ESM loader resolves a CommonJS module (no package.json/type in
    // this repo) by delegating to the CJS loader and populating the shared
    // require.cache — so by the time this test runs, the top-level `import
    // grouping from ...` above has ALREADY cached this exact file. A plain
    // require() here would return that cached module.exports without
    // re-running the module body, meaning the dual-export line would never
    // see our globalThis.window shim. Busting the cache entry forces a
    // fresh execution against the now-shimmed window.
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_PRD_GROUPING, 'window.DF_PRD_GROUPING was not set');

    // The fresh require() and the top-level import() produce two distinct
    // API object instances (separate module executions), so we compare
    // structurally — same set of exported names, each a function — rather
    // than asserting reference/deep equality against the ESM-imported
    // `grouping`.
    assert.deepEqual(
      Object.keys(globalThis.window.DF_PRD_GROUPING).sort(),
      EXPECTED_FUNCTION_NAMES.slice().sort(),
    );
    assert.deepEqual(Object.keys(required).sort(), EXPECTED_FUNCTION_NAMES.slice().sort());
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(typeof globalThis.window.DF_PRD_GROUPING[name], 'function');
    }
  } finally {
    delete globalThis.window;
  }
});

// ---------------------------------------------------------------------------
// prdTitle — basename (after the last '/'), with a trailing '-prd.md'
// stripped, else a trailing '.md' stripped, else returned as-is.
// ---------------------------------------------------------------------------

test('prdTitle: strips a directory prefix and a trailing "-prd.md" suffix', () => {
  assert.equal(
    prdTitle('plans/dashboard-taskgraph-legibility-prd.md'),
    'dashboard-taskgraph-legibility',
  );
});

test('prdTitle: takes the basename after the last "/" when there is no .md suffix', () => {
  assert.equal(prdTitle('reify:docs/prds/foo'), 'foo');
});

test('prdTitle: strips a plain trailing ".md" suffix when there is no directory prefix', () => {
  assert.equal(prdTitle('bar.md'), 'bar');
});

test('prdTitle: returns a bare string with no separators or suffix unchanged', () => {
  assert.equal(prdTitle('plain'), 'plain');
});
