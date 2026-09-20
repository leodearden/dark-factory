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

// ---------------------------------------------------------------------------
// COLUMN selection — which lock modules earn a heatmap column
// ---------------------------------------------------------------------------

// Mirrors the real wire shape built at dashboard/src/dashboard/data/scheduler.py
// (`result.append({...})`). `contention` is always an int there
// (`counts.get(path, 0)`) and `park_stack` always a list, so the predicate needs
// no None-handling beyond defensive defaults — but the fixture carries the full
// field set anyway, so a future predicate reading `holder` or `has_dead_park`
// has something real to read.
function makeModule(path, contention, extra) {
  return {
    path,
    project: 'dark_factory',
    contention,
    holder: null,
    holder_project: null,
    parked_by: null,
    parked_by_project: null,
    parked_owner_live: false,
    park_stack: [],
    has_dead_park: false,
    ...(extra || {}),
  };
}

// The server returns modules already `sorted(key=lambda m: (-contention, path))`
// (scheduler.py:341), so every column fixture is built in that order.
const COLUMN_FIXTURE = [
  makeModule('src/a/hot.py', 5),
  makeModule('src/b/warm.py', 2),
  makeModule('src/c/solo.py', 1),
  makeModule('src/d/idle.py', 0),
];

test('column selection keeps contended modules and drops uncontended ones', () => {
  // `contention` counts LIVE WAITERS. A module held by exactly one task has no
  // contention to show — it is a column of one coloured cell and 59 blanks,
  // which is exactly the noise that made the grid unreadable at 4,302 columns.
  const out = boundHeatmapAxes({ rows: [], modules: COLUMN_FIXTURE });

  assert.deepEqual(
    out.modules.map(m => m.path),
    ['src/a/hot.py', 'src/b/warm.py'],
  );
});

test('a fully-stranded module survives at contention 0 when it has a park stack', () => {
  // NOT a nicety — scheduler.py:237 deliberately injects an entry for every
  // park-stack key "even if it has no live waiters (contention: 0), so a fully
  // stranded module still gets a module entry". A bare `contention > 1` filter
  // would therefore hide exactly the stranded parks the surrounding UI already
  // shouts about: tab_scheduler.jsx renders a red "N stranded parks" banner
  // keyed on has_dead_park, and ParkStacksSection renders those very stacks.
  const stranded = makeModule('src/e/stranded.py', 0, {
    parked_by: 'T-900',
    parked_owner_live: false,
    park_stack: [{ owner: 'T-900', live: false }],
    has_dead_park: true,
  });
  const out = boundHeatmapAxes({ rows: [], modules: [...COLUMN_FIXTURE, stranded] });

  assert.ok(
    out.modules.map(m => m.path).includes('src/e/stranded.py'),
    'a contention-0 module with a non-empty park_stack must keep its column',
  );
});

test('column selection preserves the input order, inheriting the server sort', () => {
  // `filter` and `slice` are both order-preserving, so filtering the server's
  // already `(-contention, path)`-sorted list yields a PREFIX of that order —
  // the most-contended columns, for free. Re-sorting on the client would
  // duplicate a rule the server owns and let the two drift.
  const shuffledInput = [
    makeModule('src/z/nine.py', 9),
    makeModule('src/y/eight.py', 8),
    makeModule('src/x/seven.py', 7),
  ];
  const out = boundHeatmapAxes({ rows: [], modules: shuffledInput });

  assert.deepEqual(
    out.modules.map(m => m.path),
    ['src/z/nine.py', 'src/y/eight.py', 'src/x/seven.py'],
  );
});

test('a module with a missing contention field is treated as 0, not thrown over', () => {
  // Defensive: the heatmap renders against whatever the last poll returned, and
  // throwing here blanks the whole Scheduler tab rather than one column.
  const out = boundHeatmapAxes({
    rows: [],
    modules: [{ path: 'src/f/nofield.py', project: 'dark_factory' }],
  });

  assert.deepEqual(out.modules, []);
  assert.equal(out.modulesTotal, 1);
});

test('column totals and the truncation flag report the INPUT size', () => {
  const out = boundHeatmapAxes({ rows: [], modules: COLUMN_FIXTURE });

  assert.equal(out.modulesTotal, 4);
  assert.equal(out.modulesTruncated, true);
  assert.equal(
    boundHeatmapAxes({ rows: [], modules: COLUMN_FIXTURE.slice(0, 2) }).modulesTruncated,
    false,
    'nothing was dropped, so the affordance must not claim otherwise',
  );
});

// ---------------------------------------------------------------------------
// rowTouchesModule — the cell-membership rule, single-sourced
// ---------------------------------------------------------------------------
//
// This predicate answers "is this cell non-blank?", which is exactly the
// project-scope + lock_set test cellStateFor performs in its first two
// branches. Row selection needs the same answer, so cellStateFor delegates here
// rather than the two restating it — otherwise the axis filter could drop a row
// whose cells the renderer would have coloured. These are that rule's first
// executable assertions; before this it was covered only by grep.

function makeRow(taskId, lockSet, extra) {
  return {
    task_id: taskId,
    project: 'dark_factory',
    title: `task ${taskId}`,
    lock_set: lockSet,
    park_state: null,
    skip_count: 0,
    ...(extra || {}),
  };
}

test('rowTouchesModule: true when the module path is in the row lock set', () => {
  assert.equal(
    rowTouchesModule(makeRow('T-1', ['src/a/hot.py']), makeModule('src/a/hot.py', 5)),
    true,
  );
});

test('rowTouchesModule: FALSE across projects even when the path matches exactly', () => {
  // The cross-project rule cellStateFor documents. Modules are keyed by
  // `(project, path)` on the server, and two projects can each have
  // `src/utils.py` — a row from project B sharing a path with project A's
  // module entry is NOT contending for project A's lock, so the cell stays
  // blank and the row earns nothing from it. This is the trickiest branch in
  // the classifier and the one a naive `lock_set.includes(path)` gets wrong.
  assert.equal(
    rowTouchesModule(
      makeRow('T-1', ['src/utils.py'], { project: 'other_project' }),
      makeModule('src/utils.py', 5),
    ),
    false,
  );
});

test('rowTouchesModule: an untagged module skips the project check', () => {
  // Falsy `module.project` is legacy/single-project mode; the same allowance
  // cellStateFor makes.
  assert.equal(
    rowTouchesModule(
      makeRow('T-1', ['src/a/hot.py'], { project: 'other_project' }),
      makeModule('src/a/hot.py', 5, { project: null }),
    ),
    true,
  );
});

test('rowTouchesModule: false for a missing or empty lock set', () => {
  const module = makeModule('src/a/hot.py', 5);
  assert.equal(rowTouchesModule(makeRow('T-1', []), module), false);
  assert.equal(rowTouchesModule({ task_id: 'T-1', project: 'dark_factory' }, module), false);
});

// ---------------------------------------------------------------------------
// ROW selection — which task rows earn a heatmap row
// ---------------------------------------------------------------------------

test('row selection keeps rows that touch a SURVIVING module', () => {
  const out = boundHeatmapAxes({
    rows: [makeRow('T-1', ['src/a/hot.py'])],
    modules: COLUMN_FIXTURE,
  });

  assert.deepEqual(out.rows.map(r => r.task_id), ['T-1']);
});

test('row selection drops a row whose lock set only hits dropped columns', () => {
  // The row would render as 60 blank cells — pure noise, and the dominant
  // shape at production scale.
  const out = boundHeatmapAxes({
    rows: [makeRow('T-9', ['src/c/solo.py', 'src/d/idle.py'])],
    modules: COLUMN_FIXTURE,
  });

  assert.deepEqual(out.rows, []);
  assert.equal(out.rowsTotal, 1);
  assert.equal(out.rowsTruncated, true);
});

test('row selection keeps a parked row even when it touches no surviving module', () => {
  // A parked task is the most operationally interesting row on the tab — it is
  // what the stranded-parks banner is pointing at — and the live snapshot has
  // only 7 of them, so keeping them costs nothing against the cap.
  const parked = makeRow('T-7', ['src/c/solo.py'], {
    park_state: { modules: ['src/c/solo.py'] },
  });
  const out = boundHeatmapAxes({ rows: [parked], modules: COLUMN_FIXTURE });

  assert.deepEqual(out.rows.map(r => r.task_id), ['T-7']);
});

test('row selection drops a row that reaches a surviving module only across projects', () => {
  // The path collides but the lock does not. Same rule as the rowTouchesModule
  // case above, asserted through boundHeatmapAxes so the axis filter is pinned
  // to the predicate rather than merely to a compatible-looking one.
  const out = boundHeatmapAxes({
    rows: [makeRow('T-2', ['src/a/hot.py'], { project: 'other_project' })],
    modules: COLUMN_FIXTURE,
  });

  assert.deepEqual(out.rows, []);
});

test('row totals report the INPUT size and the flag is false when nothing dropped', () => {
  const rows = [makeRow('T-1', ['src/a/hot.py']), makeRow('T-2', ['src/b/warm.py'])];
  const out = boundHeatmapAxes({ rows, modules: COLUMN_FIXTURE });

  assert.equal(out.rowsTotal, 2);
  assert.equal(out.rowsTruncated, false);
});
