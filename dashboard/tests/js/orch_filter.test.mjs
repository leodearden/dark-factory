// Module-contract tests for orch_filter.js — a plain-JS (no JSX/Babel)
// module holding the pure empty-state label for the Orchestrators tab's
// multi-select task filter (OrchTab in tabs.jsx). Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper that
// surfaces this suite in CI via its `**/*.test.mjs` glob — no wrapper change
// needed for this new file).
//
// orch_filter.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { orchEmptyLabel } from '...'` would come back undefined. We
// therefore default-import the module and destructure instead (mirrors
// runtime_format.test.mjs / prd_grouping.test.mjs / graph_layout.test.mjs).
//
// The regression under test (task 3313) is described once, in orch_filter.js's
// own header comment.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import orchFilter from '../../src/dashboard/static/redux/orch_filter.js';

const { orchEmptyLabel } = orchFilter;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/orch_filter.js';
const EXPECTED_FUNCTION_NAMES = ['orchEmptyLabel'];

const NONE_SELECTED = 'No filters selected — choose Active, Pending or Complete above';

test('default-imported module exposes the orch-filter functions', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(
      typeof orchFilter[name],
      'function',
      `orchFilter.${name} should be a function`,
    );
  }
});

test('module also assigns window.DF_ORCH_FILTER (browser dual-export)', () => {
  // Shim a bare browser-like global before requiring the module fresh via
  // CommonJS require, so the module body's `if (typeof window !== 'undefined')`
  // branch executes against our shim.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    // Node's ESM loader resolves a CommonJS module (no package.json/type in
    // this repo) by delegating to the CJS loader and populating the shared
    // require.cache — so by the time this test runs, the top-level `import
    // orchFilter from ...` above has ALREADY cached this exact file. A plain
    // require() here would return that cached module.exports without
    // re-running the module body, meaning the dual-export line would never
    // see our globalThis.window shim. Busting the cache entry forces a
    // fresh execution against the now-shimmed window.
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_ORCH_FILTER, 'window.DF_ORCH_FILTER was not set');

    // The fresh require() and the top-level import() produce two distinct
    // API object instances (separate module executions), so we compare
    // structurally — same set of exported names, each a function — rather
    // than asserting reference/deep equality against the ESM-imported
    // `orchFilter`.
    assert.deepEqual(
      Object.keys(globalThis.window.DF_ORCH_FILTER).sort(),
      EXPECTED_FUNCTION_NAMES.slice().sort(),
    );
    assert.deepEqual(Object.keys(required).sort(), EXPECTED_FUNCTION_NAMES.slice().sort());
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(typeof globalThis.window.DF_ORCH_FILTER[name], 'function');
    }
  } finally {
    delete globalThis.window;
  }
});

// ---------------------------------------------------------------------------
// orchEmptyLabel — all eight combinations of the three facets.
//
// The facets are named in canonical button order (active, pending, complete),
// matching the segmented control rendered directly above the table in OrchTab
// (tabs.jsx), so the sentence always reads in the order the operator's eye
// scans the buttons.
// ---------------------------------------------------------------------------

const ALL_COMBINATIONS = [
  [{ active: false, pending: false, complete: false }, NONE_SELECTED],
  [{ active: true, pending: false, complete: false }, 'No active tasks'],
  [{ active: false, pending: true, complete: false }, 'No pending tasks'],
  [{ active: false, pending: false, complete: true }, 'No complete tasks'],
  [{ active: true, pending: true, complete: false }, 'No active or pending tasks'],
  [{ active: true, pending: false, complete: true }, 'No active or complete tasks'],
  [{ active: false, pending: true, complete: true }, 'No pending or complete tasks'],
  [{ active: true, pending: true, complete: true }, 'No active, pending or complete tasks'],
];

for (const [filter, expected] of ALL_COMBINATIONS) {
  const on = Object.keys(filter).filter(k => filter[k]);
  const label = on.length === 0 ? 'none' : on.join('+');
  test(`orchEmptyLabel: ${label} selected -> "${expected}"`, () => {
    assert.equal(orchEmptyLabel(filter), expected);
  });
}

// ---------------------------------------------------------------------------
// Canonical ordering — the label order comes from the module's own facet
// table, NOT from Object.keys iteration order. This matters because
// flipFilter rebuilds the per-pid object with spread on every click, so key
// insertion order tracks the operator's click history: without a fixed table
// the same two facets would read "No complete or active tasks" for one
// operator and "No active or complete tasks" for another.
// ---------------------------------------------------------------------------

test('orchEmptyLabel: reversed key insertion order still reads in canonical order', () => {
  assert.equal(orchEmptyLabel({ complete: true, active: true }), 'No active or complete tasks');
});

test('orchEmptyLabel: all three inserted in reverse still reads in canonical order', () => {
  assert.equal(
    orchEmptyLabel({ complete: true, pending: true, active: true }),
    'No active, pending or complete tasks',
  );
});

// ---------------------------------------------------------------------------
// Falsy / omitted keys are treated as off — getFilter normalises to three
// booleans, but the helper must not depend on that normalisation having run.
// ---------------------------------------------------------------------------

test('orchEmptyLabel: omitted keys are treated as off', () => {
  assert.equal(orchEmptyLabel({ active: true }), 'No active tasks');
});

test('orchEmptyLabel: explicit false / 0 / undefined values are treated as off', () => {
  assert.equal(orchEmptyLabel({ active: true, pending: false, complete: 0 }), 'No active tasks');
  assert.equal(
    orchEmptyLabel({ active: true, pending: undefined, complete: '' }),
    'No active tasks',
  );
});

test('orchEmptyLabel: an empty object is the none-selected state', () => {
  assert.equal(orchEmptyLabel({}), NONE_SELECTED);
});

// ---------------------------------------------------------------------------
// Defensive input — a non-object argument must not throw (a throw during
// render takes out all of OrchTab, not just this cell). getFilter normalises
// every stored value, including the legacy strings, to a three-boolean object
// before calling us, so these shapes are unreachable in practice; when the
// branch does run it mirrors getFilter's own DEFAULT_FILTER (active-only)
// rather than inventing a third behaviour, so the sentence agrees with what
// the table would actually be showing.
// ---------------------------------------------------------------------------

const GET_FILTER_DEFAULT_LABEL = 'No active tasks';

test('orchEmptyLabel: undefined does not throw and mirrors getFilter\'s default', () => {
  assert.equal(orchEmptyLabel(undefined), GET_FILTER_DEFAULT_LABEL);
});

test('orchEmptyLabel: null does not throw and mirrors getFilter\'s default', () => {
  assert.equal(orchEmptyLabel(null), GET_FILTER_DEFAULT_LABEL);
});

test('orchEmptyLabel: the legacy \'all\' string shape mirrors getFilter\'s default', () => {
  // getFilter maps this exact stored shape to DEFAULT_FILTER = active-only,
  // so the label must say "active", not "no filters selected".
  assert.equal(orchEmptyLabel('all'), GET_FILTER_DEFAULT_LABEL);
});
