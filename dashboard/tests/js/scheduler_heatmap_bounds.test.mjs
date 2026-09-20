// Module-contract tests for scheduler_heatmap_bounds.js — the pure axis
// SELECTION behind the Scheduler tab's contention heatmap.
//
// WHY THIS SUITE EXISTS, AND WHY THE LOGIC LIVES IN A `.js` AT ALL. The
// heatmap renders a rows x modules cross-product; on the 2026-09-20 live
// snapshot that is 2,991 x 4,302 = 12,867,282 cells, which kills the browser
// renderer. The fix is a bound, and a bound is only believable if it can be
// EXECUTED against a fixture at that scale. `scheduler_heatmap.jsx` is
// Babel-transformed in-browser and cannot be imported by any runner (see
// dashboard/tests/test_chip_label_disambiguation.py:8 — the dashboard has no
// JS/DOM test runner), so a cap written inline in the component could only
// ever be grep-asserted. Extracting the selection into a plain classic script
// puts it on the `node --test` path the sibling `*.test.mjs` files already
// use, which is what makes the bound demonstrable rather than asserted.
//
// This suite is one half of the coverage and is deliberately not sufficient
// alone: it proves the module CAN bound arbitrary input. That the component
// actually CONSUMES it — what makes the cap structural rather than advisory —
// is pinned separately by the JSX source-structure probes in
// dashboard/tests/test_tab_scheduler.py.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py is the pytest
// wrapper; its `js/**/*.test.mjs` glob auto-discovers this file, so no wrapper
// change was needed for it).
//
// scheduler_heatmap_bounds.js resolves as CommonJS (`module.exports =
// <object>`) because this repo has no package.json. Node's cjs-module-lexer
// cannot statically detect named exports assigned from a variable, so
// `import { boundHeatmapAxes } from '...'` would come back undefined and every
// assertion over it would pass vacuously. We therefore default-import and
// destructure (mirrors task_row_cells.test.mjs / prd_grouping.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import bounds from '../../src/dashboard/static/redux/scheduler_heatmap_bounds.js';

const {
  boundHeatmapAxes,
  rowTouchesModule,
  MAX_HEATMAP_ROWS,
  MAX_HEATMAP_COLS,
  MAX_HEATMAP_CELLS,
} = bounds;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/scheduler_heatmap_bounds.js';
const EXPECTED_FUNCTION_NAMES = ['boundHeatmapAxes', 'rowTouchesModule'];
const EXPECTED_CAP_NAMES = ['MAX_HEATMAP_ROWS', 'MAX_HEATMAP_COLS', 'MAX_HEATMAP_CELLS'];
const EXPECTED_EXPORT_NAMES = [...EXPECTED_FUNCTION_NAMES, ...EXPECTED_CAP_NAMES];

test('default-imported module exposes the axis-selection surface', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof bounds[name], 'function', `bounds.${name} should be a function`);
  }
});

test('the caps are positive integers', () => {
  for (const name of EXPECTED_CAP_NAMES) {
    const value = bounds[name];
    assert.equal(typeof value, 'number', `bounds.${name} should be a number`);
    assert.ok(Number.isInteger(value), `bounds.${name} should be an integer, got ${value}`);
    assert.ok(value > 0, `bounds.${name} should be positive, got ${value}`);
  }
});

test('MAX_HEATMAP_CELLS is DERIVED from the two axis caps, not stated independently', () => {
  // SPOT. A hand-written cell cap is a second source for a number the two axis
  // caps already determine, and the two would drift the moment either axis
  // moved — leaving the "showing N of M" affordance and the render bound
  // disagreeing about what the grid is allowed to be.
  assert.equal(MAX_HEATMAP_CELLS, MAX_HEATMAP_ROWS * MAX_HEATMAP_COLS);
});

test('module also assigns window.DF_SCHED_HEATMAP_BOUNDS (browser dual-export)', () => {
  // Shim a bare browser-like global before requiring the module fresh via
  // CommonJS require, so the module body's `if (typeof window !== 'undefined')`
  // branch executes against our shim. The top-level `import` above has already
  // populated the shared require.cache (node's ESM loader delegates CJS
  // resolution to the CJS loader), so the cache entry must be busted to force a
  // fresh execution against the now-shimmed window.
  //
  // The browser branch is the one that actually matters in production:
  // scheduler_heatmap.jsx reads `window.DF_SCHED_HEATMAP_BOUNDS` at module top
  // level, so a module that exported only to CommonJS would test green here and
  // throw a TypeError on the real dashboard.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_SCHED_HEATMAP_BOUNDS, 'window.DF_SCHED_HEATMAP_BOUNDS was not set');
    assert.deepEqual(
      Object.keys(globalThis.window.DF_SCHED_HEATMAP_BOUNDS).sort(),
      EXPECTED_EXPORT_NAMES.slice().sort(),
    );
    assert.deepEqual(Object.keys(required).sort(), EXPECTED_EXPORT_NAMES.slice().sort());
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(typeof globalThis.window.DF_SCHED_HEATMAP_BOUNDS[name], 'function');
    }
  } finally {
    delete globalThis.window;
  }
});

test('boundHeatmapAxes returns the full axis-selection shape', () => {
  // The return shape is a contract with two consumers, not an implementation
  // detail: the component iterates `rows`/`modules`, and the "showing N of M"
  // affordance reads the totals and the truncation flags. A missing key there
  // renders `undefined` into the UI rather than failing.
  const out = boundHeatmapAxes({ rows: [], modules: [] });

  assert.deepEqual(
    Object.keys(out).sort(),
    ['modules', 'modulesTotal', 'modulesTruncated', 'rows', 'rowsTotal', 'rowsTruncated'],
  );
  assert.ok(Array.isArray(out.rows));
  assert.ok(Array.isArray(out.modules));
  assert.equal(typeof out.rowsTotal, 'number');
  assert.equal(typeof out.modulesTotal, 'number');
  assert.equal(typeof out.rowsTruncated, 'boolean');
  assert.equal(typeof out.modulesTruncated, 'boolean');
});
