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
  MAX_RESCUED_COLS,
} = bounds;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/scheduler_heatmap_bounds.js';
const EXPECTED_FUNCTION_NAMES = ['boundHeatmapAxes', 'rowTouchesModule'];
const EXPECTED_CAP_NAMES = ['MAX_HEATMAP_ROWS', 'MAX_HEATMAP_COLS', 'MAX_RESCUED_COLS'];
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
  assert.ok(
    MAX_RESCUED_COLS < MAX_HEATMAP_COLS,
    'the stranded-lock rescue can never take the whole grid',
  );
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
    [
      'modules', 'modulesRescued', 'modulesTotal', 'modulesTruncated',
      'rows', 'rowsTotal', 'rowsTruncated',
    ],
  );
  assert.ok(Array.isArray(out.rows));
  assert.ok(Array.isArray(out.modules));
  assert.equal(typeof out.rowsTotal, 'number');
  assert.equal(typeof out.modulesTotal, 'number');
  assert.equal(typeof out.modulesRescued, 'number');
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

// scheduler.py::_module_contention_counts's sort, reproduced so a fixture
// arrives in the order the client really receives.
function serverSorted(modules) {
  return modules.slice().sort((a, b) =>
    (b.contention - a.contention) || (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
}

// The two stranded-lock shapes a column is rescued for (lockIsStranded).
function strandedPark(path) {
  return makeModule(path, 0, {
    parked_by: 'T-dead',
    parked_owner_live: false,
    park_stack: [{ owner: 'T-dead', live: false }],
    has_dead_park: true,
  });
}

function departedHolder(path, contention) {
  return makeModule(path, contention, {
    holder: 'T-gone',
    holder_project: 'dark_factory',
    holder_live: false,
  });
}

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

test('a fully-stranded module survives at contention 0 when its park has a dead owner', () => {
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

test('a held module keeps its column at contention 1 when its holder has departed (holder_live false)', () => {
  // `contention` counts rows whose lock_set includes the path, so the holder
  // is counted only if the holding task is itself among the composed rows.
  // When it is not — a stale `current_holders` entry whose task has left
  // active_tasks, which the server reports as `holder_live: false` — one
  // genuinely-blocked waiter scores contention 1, and a bare
  // `contention > 1` filter would drop the column along with that waiter's
  // red 'lock-taken' cell. cellStateFor still classifies that state, so the
  // column has something to show.
  const held = makeModule('src/g/held.py', 1, {
    holder: 'T-gone',
    holder_project: 'dark_factory',
    holder_live: false,
  });
  const out = boundHeatmapAxes({ rows: [], modules: [...COLUMN_FIXTURE, held] });

  assert.ok(
    out.modules.map(m => m.path).includes('src/g/held.py'),
    'a held module with a live waiter must keep its column',
  );
});

test('a held module with NO live waiter earns nothing — one cell is not contention', () => {
  // The other side of the holder clause, so it cannot quietly widen into
  // "every held module": a module held by a running task that nobody is
  // waiting on renders one coloured cell and 59 blanks, which is the noise
  // the column filter exists to remove.
  const out = boundHeatmapAxes({
    rows: [],
    modules: [makeModule('src/h/quiet.py', 0, { holder: 'T-1' })],
  });

  assert.deepEqual(out.modules, []);
});

test('a held module whose holder is LIVE earns nothing at contention 1 — the one row counted is the holder itself', () => {
  // Contention cannot tell the holder from a waiter: a live holder is a row
  // whose own lock_set holds the path, so it scores 1 with nobody waiting.
  // On the 2026-10-12 live snapshot 203 of the 204 held contention-1 modules
  // were this shape; rescuing them would bury the one departed holder.
  const out = boundHeatmapAxes({
    rows: [],
    modules: [makeModule('src/i/self.py', 1, {
      holder: 'T-1',
      holder_project: 'dark_factory',
      holder_live: true,
    })],
  });

  assert.deepEqual(out.modules, []);
});

test('a held module of UNKNOWN holder liveness earns nothing at contention 1', () => {
  // Unknown never rescues: an older payload or a mock without holder_live
  // must not flood the grid with every held module.
  const out = boundHeatmapAxes({
    rows: [],
    modules: [makeModule('src/i/unknown.py', 1, { holder: 'T-1', holder_project: 'dark_factory' })],
  });

  assert.deepEqual(out.modules, []);
});

test('a live park earns no column of its own at contention 1 — only a dead owner rescues', () => {
  // 99 modules on the 2026-10-12 live snapshot carried a park stack, every
  // owner live. ParkStacksSection lists each of them; on the grid, one
  // coloured cell is not contention.
  const out = boundHeatmapAxes({
    rows: [],
    modules: [makeModule('src/j/livepark.py', 1, {
      parked_by: 'T-5',
      parked_owner_live: true,
      park_stack: [{ owner: 'T-5', live: true }],
      has_dead_park: false,
    })],
  });

  assert.deepEqual(out.modules, []);
});

test('a live-top park with a dead SHADOWED owner is stranded too', () => {
  // has_dead_park is the server's definition of stranded — any dead entry in
  // the stack, not just the active top — and the stranded-parks banner reads
  // it. Narrowing to parked_owner_live would miss this module.
  const shadowed = makeModule('src/j/shadowed.py', 0, {
    parked_by: 'T-5',
    parked_owner_live: true,
    park_stack: [{ owner: 'T-ghost', live: false }, { owner: 'T-5', live: true }],
    has_dead_park: true,
  });
  const out = boundHeatmapAxes({ rows: [], modules: [shadowed] });

  assert.deepEqual(out.modules.map(m => m.path), ['src/j/shadowed.py']);
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

test('row selection puts parked rows FIRST and keeps the rest in server order', () => {
  // The row axis's only ordering guarantee, stated executably — the column
  // axis has had one since `column selection preserves the input order`, and
  // the "showing N of M" caption is written against this. Note what is NOT
  // claimed: nothing orders rows by contention, which is why the caption's
  // most-contended superlative sits on the COLUMN axis alone. Parked first is
  // what makes the carve-out survive the cap (below); the rest follow the
  // tier of rows waiting on a rescued column, empty in this fixture.
  const rows = [
    makeRow('T-1', ['src/a/hot.py']),
    makeRow('T-2', ['src/b/warm.py']),
    makeRow('T-parked', ['src/c/solo.py'], { park_state: { modules: ['src/c/solo.py'] } }),
    makeRow('T-3', ['src/a/hot.py']),
  ];
  const out = boundHeatmapAxes({ rows, modules: COLUMN_FIXTURE });

  assert.deepEqual(out.rows.map(r => r.task_id), ['T-parked', 'T-1', 'T-2', 'T-3']);
});

test('row selection: parked rows first, then rows waiting on a rescued column, then the rest in server order', () => {
  // A row blocked on a stranded lock is as stuck as a parked one.
  const modules = serverSorted([...COLUMN_FIXTURE, departedHolder('src/k/ghost.py', 1)]);
  const rows = [
    makeRow('T-1', ['src/a/hot.py']),
    makeRow('T-blocked', ['src/k/ghost.py']),
    makeRow('T-parked', ['src/c/solo.py'], { park_state: { modules: ['src/c/solo.py'] } }),
    makeRow('T-2', ['src/b/warm.py']),
  ];
  const out = boundHeatmapAxes({ rows, modules });

  assert.deepEqual(out.rows.map(r => r.task_id), ['T-parked', 'T-blocked', 'T-1', 'T-2']);
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

// ---------------------------------------------------------------------------
// THE BOUND — a structural cap on rendered cell count at production scale
// ---------------------------------------------------------------------------
//
// This is the regression the task exists for. The 2026-09-20 live snapshot is
// 2,991 rows x 4,302 modules = 12,867,282 cells, each rendering a <td> wrapping
// a <div>; the browser renderer dies before it finishes. What must be proven is
// a bound on the RENDERED cell count given a fixture at that scale — not merely
// that the module returns something.
//
// The bound's achievability basis is `Array.prototype.slice(0, N)` returning at
// most N elements (ECMA-262 23.1.3.28), not a tuned threshold. That matters for
// what the PATHOLOGICAL case below is for.

const LIVE_ROWS = 2991;
const LIVE_MODULES = 4302;

// Built once and shared — the dense variant is ~12.9M lock_set entries if
// materialised naively, so both variants share one path list and the dense rows
// share ONE lock_set array reference rather than 2,991 copies of it.
function makeFixture({ nRows, nModules, dense }) {
  const paths = Array.from({ length: nModules }, (_, i) => `src/pkg${i % 97}/mod${i}.py`);

  const modules = paths.map((path, i) =>
    // Realistic: ~43% of modules sit at contention > 1, matching the live
    // snapshot. Pathological: every module is hotly contended, so the column
    // filter removes NOTHING.
    makeModule(path, dense ? 99 : (i % 7 < 3 ? 2 + (i % 5) : i % 2)),
  );

  // Pathological rows all share this one array: every row touches every module,
  // so the row filter removes nothing either.
  const everyPath = dense ? paths : null;

  const rows = Array.from({ length: nRows }, (_, r) => {
    if (dense) return makeRow(`T-${r}`, everyPath);
    // Realistic: mean lock_set of 4.41 paths, spread across the module list.
    const size = 4 + (r % 2);
    const lockSet = Array.from({ length: size }, (_, k) => paths[(r * 7 + k * 131) % nModules]);
    return makeRow(`T-${r}`, lockSet);
  });

  return { rows, modules };
}

let realisticFixture = null;
let pathologicalFixture = null;

test('BOUND: a realistic production-scale snapshot renders at most ROWS x COLS cells', () => {
  realisticFixture = realisticFixture
    || makeFixture({ nRows: LIVE_ROWS, nModules: LIVE_MODULES, dense: false });
  const out = boundHeatmapAxes(realisticFixture);

  assert.ok(
    out.rows.length <= MAX_HEATMAP_ROWS,
    `rendered ${out.rows.length} rows, cap is ${MAX_HEATMAP_ROWS}`,
  );
  assert.ok(
    out.modules.length <= MAX_HEATMAP_COLS,
    `rendered ${out.modules.length} columns, cap is ${MAX_HEATMAP_COLS}`,
  );
  // The cell bound is spelled as the product of the two axis caps, not read
  // from a third exported constant — the module publishes only the two, so
  // there is no second source here to drift away from what it enforces.
  assert.ok(
    out.rows.length * out.modules.length <= MAX_HEATMAP_ROWS * MAX_HEATMAP_COLS,
    `rendered ${out.rows.length * out.modules.length} cells, cap is ${MAX_HEATMAP_ROWS * MAX_HEATMAP_COLS}`,
  );

  assert.equal(out.rowsTotal, LIVE_ROWS);
  assert.equal(out.modulesTotal, LIVE_MODULES);
  assert.equal(out.rowsTruncated, true);
  assert.equal(out.modulesTruncated, true);
});

test('BOUND: a PATHOLOGICAL snapshot where neither filter removes anything is still bounded', () => {
  // The case that proves the bound is STRUCTURAL rather than an artifact of
  // real data being sparse. Every module is at contention 99 and every row's
  // lock_set contains every module path, so the contention filter and the
  // touch filter both keep 100% of their input and ONLY the hard slice is
  // load-bearing. A future regression that re-introduces an uncapped axis —
  // say by trusting the filters to be enough — passes the realistic case above
  // and fails here.
  pathologicalFixture = pathologicalFixture
    || makeFixture({ nRows: LIVE_ROWS, nModules: LIVE_MODULES, dense: true });
  const out = boundHeatmapAxes(pathologicalFixture);

  assert.ok(
    out.rows.length <= MAX_HEATMAP_ROWS,
    `rendered ${out.rows.length} rows, cap is ${MAX_HEATMAP_ROWS}`,
  );
  assert.ok(
    out.modules.length <= MAX_HEATMAP_COLS,
    `rendered ${out.modules.length} columns, cap is ${MAX_HEATMAP_COLS}`,
  );
  assert.ok(
    out.rows.length * out.modules.length <= MAX_HEATMAP_ROWS * MAX_HEATMAP_COLS,
    `rendered ${out.rows.length * out.modules.length} cells, cap is ` +
      `${MAX_HEATMAP_ROWS * MAX_HEATMAP_COLS} — neither axis filter removed ` +
      'anything here, so the hard cap is the only thing standing between the ' +
      'browser and a 12.9M-cell DOM',
  );

  assert.equal(out.rowsTotal, LIVE_ROWS);
  assert.equal(out.modulesTotal, LIVE_MODULES);
  assert.equal(out.rowsTruncated, true);
  assert.equal(out.modulesTruncated, true);
});

test('BOUND: rows are selected against the columns that will ACTUALLY render', () => {
  // Ordering inside boundHeatmapAxes, asserted behaviourally. If rows were
  // chosen before the column slice, a row touching only column 61 — real
  // contention, but past the cap — would survive and then render as 60 blank
  // cells. Here `beyondCap` is the 61st surviving column and the row that only
  // touches it must go with it.
  const modules = Array.from({ length: MAX_HEATMAP_COLS + 1 }, (_, i) =>
    makeModule(`src/wide/mod${i}.py`, 5),
  );
  const beyondCap = modules[MAX_HEATMAP_COLS];
  const out = boundHeatmapAxes({
    rows: [makeRow('T-keep', [modules[0].path]), makeRow('T-orphan', [beyondCap.path])],
    modules,
  });

  assert.equal(out.modules.length, MAX_HEATMAP_COLS);
  assert.deepEqual(out.rows.map(r => r.task_id), ['T-keep']);
});

test('BOUND: a parked row survives the cap even when ordinary rows would fill it', () => {
  // The parked carve-out is only real if it beats the SLICE, not merely the
  // filter — and the single-row fixture above cannot see the difference.
  // `shape_scheduler` appends the synthetic stranded-park rows after every
  // project's ordinary rows, so at production scale the parked rows are
  // exactly the ones a positional prefix always cuts: the one shape where the
  // carve-out is load-bearing is the one shape it used to fail.
  const ordinary = Array.from({ length: MAX_HEATMAP_ROWS + 5 }, (_, i) =>
    makeRow(`T-ord-${i}`, ['src/a/hot.py']),
  );
  const parked = makeRow('T-stranded', ['src/c/solo.py'], {
    park_state: { modules: ['src/c/solo.py'] },
  });
  const out = boundHeatmapAxes({
    rows: [...ordinary, parked],
    modules: COLUMN_FIXTURE,
  });

  assert.equal(out.rows.length, MAX_HEATMAP_ROWS, 'the cap still binds');
  assert.ok(
    out.rows.map(r => r.task_id).includes('T-stranded'),
    'the parked row must survive the cap — it is what the stranded-parks ' +
      'banner points at, and it arrives last in server order',
  );
});

// ---------------------------------------------------------------------------
// COLUMN BUDGET — stranded locks reach the rendered set at cap scale
// ---------------------------------------------------------------------------
//
// The server sorts modules `(-contention, path)`, so a stranded lock (a dead
// park at contention 0, a departed holder with a handful of waiters) sorts to
// the tail, and a positional prefix of 60 never reaches it at live scale. The
// columns are therefore a contended HEAD plus a rescued TAIL of stranded
// modules the head did not reach — the column-axis twin of 'BOUND: a parked
// row survives the cap even when ordinary rows would fill it'.

const STRANDED_PARK_PATH = 'src/stranded/park.py';
const DEPARTED_HOLDER_PATH = 'src/stranded/ghost.py';

// The task's measured shape: makeFixture's realistic distribution (1,845
// contended of 4,302, the head bottoming out at contention 6) plus one of
// each carve-out, server-sorted.
let liveShapedModules = null;
function getLiveShapedModules() {
  liveShapedModules = liveShapedModules || serverSorted([
    ...makeFixture({ nRows: LIVE_ROWS, nModules: LIVE_MODULES, dense: false }).modules,
    strandedPark(STRANDED_PARK_PATH),
    departedHolder(DEPARTED_HOLDER_PATH, 1),
  ]);
  return liveShapedModules;
}

function contendedColumns(count) {
  return Array.from({ length: count }, (_, i) => makeModule(`src/wide/mod${i}.py`, 5));
}

test('BOUND: a stranded-park column survives the cap even when contended columns would fill it', () => {
  const out = boundHeatmapAxes({
    rows: [],
    modules: [...contendedColumns(MAX_HEATMAP_COLS + 5), strandedPark(STRANDED_PARK_PATH)],
  });

  assert.equal(out.modules.length, MAX_HEATMAP_COLS, 'the cap still binds');
  assert.ok(out.modules.map(m => m.path).includes(STRANDED_PARK_PATH));
  assert.equal(out.modulesRescued, 1);
});

test('BOUND: a departed-holder column survives the cap even when contended columns would fill it', () => {
  const out = boundHeatmapAxes({
    rows: [],
    modules: [...contendedColumns(MAX_HEATMAP_COLS + 5), departedHolder(DEPARTED_HOLDER_PATH, 1)],
  });

  assert.equal(out.modules.length, MAX_HEATMAP_COLS, 'the cap still binds');
  assert.ok(out.modules.map(m => m.path).includes(DEPARTED_HOLDER_PATH));
  assert.equal(out.modulesRescued, 1);
});

test('BOUND: both carve-outs reach the rendered set on a live-shaped snapshot', () => {
  const out = boundHeatmapAxes({ rows: [], modules: getLiveShapedModules() });
  const paths = out.modules.map(m => m.path);

  assert.ok(paths.includes(STRANDED_PARK_PATH), 'the dead park must render');
  assert.ok(paths.includes(DEPARTED_HOLDER_PATH), 'the departed holder must render');
  assert.ok(out.modules.length <= MAX_HEATMAP_COLS);
  assert.equal(out.modulesRescued, 2);
});

test('the rendered columns stay in server order: the contended head, then the rescued tail', () => {
  // The property the caption's 'the N most contended ..., then R stranded'
  // rests on.
  const input = getLiveShapedModules();
  const out = boundHeatmapAxes({ rows: [], modules: input });
  const inputIndex = new Map(input.map((m, i) => [m, i]));
  const indices = out.modules.map(m => inputIndex.get(m));

  for (let i = 1; i < indices.length; i++) {
    assert.ok(indices[i] > indices[i - 1], `column ${i} breaks server order`);
  }
  const headLen = out.modules.length - out.modulesRescued;
  assert.deepEqual(
    out.modules.slice(headLen).map(m => m.path).sort(),
    [DEPARTED_HOLDER_PATH, STRANDED_PARK_PATH].sort(),
  );
  assert.deepEqual(
    out.modules.slice(0, headLen),
    input.filter(m => m.contention > 1).slice(0, headLen),
  );
});

test('a stranded module already inside the contended head takes no reserved slot', () => {
  const input = [departedHolder(DEPARTED_HOLDER_PATH, 99), ...contendedColumns(65)];
  const out = boundHeatmapAxes({ rows: [], modules: input });

  assert.deepEqual(out.modules, input.slice(0, MAX_HEATMAP_COLS));
  assert.equal(out.modulesRescued, 0);
});

test('a stranded module at the edge of the head is not squeezed out by a rescue behind it', () => {
  // Reserving the dead park's slot shrinks the head to 59, which pushes the
  // departed holder at index 59 out of it, which needs a slot of its own: the
  // head length is a fixed point (58 here), and a one-pass greedy drops the
  // index-59 module.
  const input = Array.from({ length: 70 }, (_, i) =>
    makeModule(`src/edge/mod${String(i).padStart(2, '0')}.py`, 200 - i));
  input[59] = departedHolder(DEPARTED_HOLDER_PATH, 200 - 59);
  input.push(strandedPark(STRANDED_PARK_PATH));
  const out = boundHeatmapAxes({ rows: [], modules: input });
  const paths = out.modules.map(m => m.path);

  assert.ok(paths.includes(DEPARTED_HOLDER_PATH), 'the edge module must not be squeezed out');
  assert.ok(paths.includes(STRANDED_PARK_PATH));
  assert.equal(out.modules.length, MAX_HEATMAP_COLS);
  assert.equal(out.modulesRescued, 2);
});

test('rescued columns never take more than MAX_RESCUED_COLS slots', () => {
  const contended = contendedColumns(70);
  const parks = Array.from({ length: MAX_RESCUED_COLS + 5 }, (_, i) =>
    strandedPark(`src/stranded/park${String(i).padStart(2, '0')}.py`));
  const out = boundHeatmapAxes({ rows: [], modules: [...contended, ...parks] });
  const headLen = MAX_HEATMAP_COLS - MAX_RESCUED_COLS;

  assert.equal(out.modulesRescued, MAX_RESCUED_COLS);
  assert.equal(out.modules.length, MAX_HEATMAP_COLS);
  assert.deepEqual(out.modules.slice(0, headLen), contended.slice(0, headLen));
});

test('BOUND: a row waiting on a rescued column survives the row cap', () => {
  // Without this the rescued column renders as 60 blank cells: on the live
  // 2026-10-12 payload only 1 of 5 rescued columns had a coloured cell, its
  // waiters sitting ~2,208th of 2,223 touching rows.
  const ordinary = Array.from({ length: MAX_HEATMAP_ROWS + 5 }, (_, i) =>
    makeRow(`T-ord-${i}`, ['src/wide/mod0.py']),
  );
  const out = boundHeatmapAxes({
    rows: [...ordinary, makeRow('T-blocked', ['src/k/ghost.py'])],
    modules: [...contendedColumns(MAX_HEATMAP_COLS + 5), departedHolder('src/k/ghost.py', 1)],
  });

  assert.equal(out.rows.length, MAX_HEATMAP_ROWS, 'the cap still binds');
  assert.ok(out.rows.map(r => r.task_id).includes('T-blocked'));
});

test('BOUND: on a live-shaped snapshot every rescued column has a rendered row that colours it', () => {
  realisticFixture = realisticFixture
    || makeFixture({ nRows: LIVE_ROWS, nModules: LIVE_MODULES, dense: false });
  const rows = [
    ...realisticFixture.rows,
    makeRow('T-dead', [STRANDED_PARK_PATH], {
      park_state: { modules: [STRANDED_PARK_PATH] },
      stranded: true,
    }),
    makeRow('T-wait-1', [DEPARTED_HOLDER_PATH]),
    makeRow('T-wait-2', [DEPARTED_HOLDER_PATH]),
  ];
  const out = boundHeatmapAxes({ rows, modules: getLiveShapedModules() });

  assert.equal(out.modulesRescued, 2);
  for (const module of out.modules.slice(-out.modulesRescued)) {
    assert.ok(
      out.rows.some(row => rowTouchesModule(row, module)),
      `rescued column ${module.path} renders blank`,
    );
  }
});
