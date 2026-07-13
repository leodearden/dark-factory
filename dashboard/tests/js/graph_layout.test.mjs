// Module-contract tests for graph_layout.js — a plain-JS (no JSX/Babel)
// Sugiyama ordering-phase module. Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper that
// surfaces this suite in CI).
//
// graph_layout.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { computeTiers } from '...'` would come back undefined. We
// therefore default-import the module and destructure instead.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import layout from '../../src/dashboard/static/redux/graph_layout.js';

const { computeTiers, partitionComponents, orderRows, countCrossings } = layout;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/graph_layout.js';
const EXPECTED_FUNCTION_NAMES = [
  'computeTiers',
  'partitionComponents',
  'orderRows',
  'countCrossings',
];

// Builds a minimal task fixture matching the dep edge shape the dashboard
// already ships (t.deps = [{id, done, title}, ...]) — see
// tab_tasks.jsx:20,30,268. Only `id` is populated here; computeTiers only
// reads d.id. `status` is omitted entirely (not just `undefined`-valued)
// when not passed, so fixtures used by tests that don't care about status
// keep the exact same object shape as before orderRows needed one.
function mkTask(id, depIds = [], status) {
  return {
    id,
    ...(status !== undefined ? { status } : {}),
    deps: depIds.map(depId => ({ id: depId })),
  };
}

function idsOf(taskArray) {
  return taskArray.map(t => t.id);
}

// Independent oracle for orderRows' initial per-tier permutation, mirroring
// the STATUS_ORDER map documented in the plan (and copied from
// tab_tasks.jsx:137) — deliberately re-implemented here rather than reusing
// any graph_layout.js internal, since STATUS_ORDER itself isn't part of the
// module's four-function export contract.
const TEST_STATUS_ORDER = {
  blocked: 0,
  'in-progress': 1,
  'merge-deferred': 1.5,
  pending: 2,
  deferred: 3,
  done: 4,
  cancelled: 5,
};

function testStatusRank(status) {
  return Object.prototype.hasOwnProperty.call(TEST_STATUS_ORDER, status) ? TEST_STATUS_ORDER[status] : 9;
}

// Buckets componentTasks by tier and stable-sorts each bucket by status rank
// (input order as tiebreak) — the same initial permutation orderRows itself
// is specified to start from.
function statusSortBaseline(componentTasks, tiers) {
  const rows = [];
  componentTasks.forEach((t, index) => {
    const tier = tiers.get(t.id) || 0;
    if (!rows[tier]) rows[tier] = [];
    rows[tier].push({ t, index });
  });
  return rows.map(row =>
    row
      .slice()
      .sort((a, b) => {
        const rankDiff = testStatusRank(a.t.status) - testStatusRank(b.t.status);
        return rankDiff !== 0 ? rankDiff : a.index - b.index;
      })
      .map(entry => entry.t),
  );
}

// Derives the {from, to} internal edge list from componentTasks' in-set deps
// (from = the upstream/parent id, to = the downstream/child id) — the same
// derivation orderRows itself is specified to use.
function edgesFromDeps(componentTasks) {
  const ids = new Set(componentTasks.map(t => t.id));
  const edges = [];
  for (const t of componentTasks) {
    for (const d of t.deps || []) {
      if (ids.has(d.id) && d.id !== t.id) edges.push({ from: d.id, to: t.id });
    }
  }
  return edges;
}

test('default-imported module exposes the four layout functions', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof layout[name], 'function', `layout.${name} should be a function`);
  }
});

test('module also assigns window.DF_GRAPH_LAYOUT (browser dual-export)', () => {
  // Shim a bare browser-like global before requiring the module fresh via
  // CommonJS require, so the module body's `if (typeof window !== 'undefined')`
  // branch executes against our shim.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    // Node's ESM loader resolves a CommonJS module (no package.json/type in
    // this repo) by delegating to the CJS loader and populating the shared
    // require.cache — so by the time this test runs, the top-level `import
    // layout from ...` above has ALREADY cached this exact file. A plain
    // require() here would return that cached module.exports without
    // re-running the module body, meaning the dual-export line would never
    // see our globalThis.window shim. Busting the cache entry forces a
    // fresh execution against the now-shimmed window.
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_GRAPH_LAYOUT, 'window.DF_GRAPH_LAYOUT was not set');

    // The fresh require() and the top-level import() produce two distinct
    // API object instances (separate module executions), so we compare
    // structurally — same set of exported names, each a function — rather
    // than asserting reference/deep equality against the ESM-imported
    // `layout`.
    assert.deepEqual(
      Object.keys(globalThis.window.DF_GRAPH_LAYOUT).sort(),
      EXPECTED_FUNCTION_NAMES.slice().sort(),
    );
    assert.deepEqual(Object.keys(required).sort(), EXPECTED_FUNCTION_NAMES.slice().sort());
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(typeof globalThis.window.DF_GRAPH_LAYOUT[name], 'function');
    }
  } finally {
    delete globalThis.window;
  }
});

// ---------------------------------------------------------------------------
// computeTiers — verbatim copy of tab_tasks.jsx:19-38 (longest-path tiering)
// ---------------------------------------------------------------------------

test('computeTiers: linear chain A->B->C yields tiers 0/1/2', () => {
  const tasks = [mkTask('A'), mkTask('B', ['A']), mkTask('C', ['B'])];
  const tiers = computeTiers(tasks);
  assert.equal(tiers.get('A'), 0);
  assert.equal(tiers.get('B'), 1);
  assert.equal(tiers.get('C'), 2);
});

test('computeTiers: diamond (A; B,C dep A; D dep B,C) yields A=0,B=1,C=1,D=2', () => {
  const tasks = [mkTask('A'), mkTask('B', ['A']), mkTask('C', ['A']), mkTask('D', ['B', 'C'])];
  const tiers = computeTiers(tasks);
  assert.equal(tiers.get('A'), 0);
  assert.equal(tiers.get('B'), 1);
  assert.equal(tiers.get('C'), 1);
  assert.equal(tiers.get('D'), 2);
});

test('computeTiers: a dep outside the input list is ignored (does not raise the tier)', () => {
  const tasks = [mkTask('X', ['OUTSIDE_OF_LIST'])];
  const tiers = computeTiers(tasks);
  assert.equal(tiers.get('X'), 0);
});

test('computeTiers: a 2-cycle does not infinite-loop and returns finite tiers', () => {
  const tasks = [mkTask('X', ['Y']), mkTask('Y', ['X'])];
  const tiers = computeTiers(tasks);
  assert.equal(tiers.size, 2);
  assert.ok(Number.isFinite(tiers.get('X')), 'X tier should be finite (cycle guard, not Infinity/NaN)');
  assert.ok(Number.isFinite(tiers.get('Y')), 'Y tier should be finite (cycle guard, not Infinity/NaN)');
});

test('computeTiers: empty input yields an empty Map', () => {
  const tiers = computeTiers([]);
  assert.equal(tiers.size, 0);
});

// ---------------------------------------------------------------------------
// countCrossings — pairwise inversion count between ADJACENT tiers only.
// rows is an array of per-tier arrays of task objects (the shape orderRows
// produces); countCrossings keys on each row entry's `t.id` to match against
// edges, whose shape is {from: <upper/parentId>, to: <lower/childId>}.
// Multi-tier ("long") edges whose endpoints are not in an adjacent tier pair
// are not counted (documented v1 no-dummy-node limitation).
// ---------------------------------------------------------------------------

test('countCrossings: single inversion between two tiers returns 1', () => {
  const rows = [[mkTask('P1'), mkTask('P2')], [mkTask('C1'), mkTask('C2')]];
  const edges = [
    { from: 'P1', to: 'C2' },
    { from: 'P2', to: 'C1' },
  ];
  assert.equal(countCrossings(rows, edges), 1);
});

test('countCrossings: reordering the lower tier to uncross yields 0', () => {
  const rows = [[mkTask('P1'), mkTask('P2')], [mkTask('C2'), mkTask('C1')]];
  const edges = [
    { from: 'P1', to: 'C2' },
    { from: 'P2', to: 'C1' },
  ];
  assert.equal(countCrossings(rows, edges), 0);
});

test('countCrossings: parallel (non-crossing) edges return 0', () => {
  const rows = [[mkTask('P1'), mkTask('P2')], [mkTask('C1'), mkTask('C2')]];
  const edges = [
    { from: 'P1', to: 'C1' },
    { from: 'P2', to: 'C2' },
  ];
  assert.equal(countCrossings(rows, edges), 0);
});

test('countCrossings: crossings across two adjacent tier pairs sum together', () => {
  const rows = [
    [mkTask('A1'), mkTask('A2')],
    [mkTask('B1'), mkTask('B2')],
    [mkTask('C1'), mkTask('C2')],
  ];
  const edges = [
    // tier0 -> tier1: one inversion
    { from: 'A1', to: 'B2' },
    { from: 'A2', to: 'B1' },
    // tier1 -> tier2: one inversion
    { from: 'B1', to: 'C2' },
    { from: 'B2', to: 'C1' },
  ];
  assert.equal(countCrossings(rows, edges), 2);
});

// ---------------------------------------------------------------------------
// partitionComponents — weakly-connected components over the in-list deps
// edge list, plus singletons (nodes with zero in-list deps either direction).
// Components are ordered by the earliest input index of any member; members
// within a component (and singletons) preserve input order.
// ---------------------------------------------------------------------------

test('partitionComponents: exact partition — multi-component fixture plus isolated singletons', () => {
  // Component X: A -> B -> C (chain). Component Y: D -> E. F, G are isolated.
  const tasks = [
    mkTask('A'),
    mkTask('B', ['A']),
    mkTask('C', ['B']),
    mkTask('D'),
    mkTask('E', ['D']),
    mkTask('F'),
    mkTask('G'),
  ];
  const result = partitionComponents(tasks);

  assert.equal(result.components.length, 2);
  assert.deepEqual(idsOf(result.components[0]).slice().sort(), ['A', 'B', 'C']);
  assert.deepEqual(idsOf(result.components[1]).slice().sort(), ['D', 'E']);
  assert.deepEqual(idsOf(result.singletons), ['F', 'G']);

  // Every input task appears in exactly one bucket (exact partition).
  const allBucketedIds = [...result.components.flatMap(idsOf), ...idsOf(result.singletons)];
  assert.deepEqual(allBucketedIds.slice().sort(), tasks.map(t => t.id).slice().sort());
  assert.equal(new Set(allBucketedIds).size, tasks.length);
});

test('partitionComponents: components ordered by earliest input index; members/singletons preserve input order', () => {
  const tasks = [
    mkTask('A'),
    mkTask('B', ['A']),
    mkTask('C', ['B']),
    mkTask('D'),
    mkTask('E', ['D']),
    mkTask('F'),
    mkTask('G'),
  ];
  const result = partitionComponents(tasks);

  // Component X (earliest member A at index 0) sorts before component Y
  // (earliest member D at index 3), and each component's members — plus the
  // singletons list — preserve their original input order.
  assert.deepEqual(idsOf(result.components[0]), ['A', 'B', 'C']);
  assert.deepEqual(idsOf(result.components[1]), ['D', 'E']);
  assert.deepEqual(idsOf(result.singletons), ['F', 'G']);
});

test('partitionComponents: deterministic across repeated calls on identical input', () => {
  const tasks = [
    mkTask('A'),
    mkTask('B', ['A']),
    mkTask('C', ['B']),
    mkTask('D'),
    mkTask('E', ['D']),
    mkTask('F'),
    mkTask('G'),
  ];
  const first = partitionComponents(tasks);
  const second = partitionComponents(tasks);
  assert.deepEqual(JSON.stringify(first), JSON.stringify(second));
});

test('partitionComponents: singleton-only input — no components, every task a singleton', () => {
  const tasks = [mkTask('H'), mkTask('I')];
  const result = partitionComponents(tasks);
  assert.deepEqual(result.components, []);
  assert.deepEqual(idsOf(result.singletons), ['H', 'I']);
});

test('partitionComponents: single-node input yields exactly one singleton', () => {
  const tasks = [mkTask('Z')];
  const result = partitionComponents(tasks);
  assert.deepEqual(result.components, []);
  assert.deepEqual(idsOf(result.singletons), ['Z']);
});

// ---------------------------------------------------------------------------
// orderRows — initial status-sort permutation, then barycenter sweeps +
// transpose pass to reduce edge crossings. The user-observable signal is
// crossings(orderRows(...)) <= crossings(status-sort baseline), strictly <
// on a tangled fixture, and deterministic across repeated calls.
// ---------------------------------------------------------------------------

// Flattens orderRows' per-tier output back into a single id array — used to
// assert the result is an exact re-permutation of the input (every task
// appears exactly once), independent of crossings.
function flattenedIds(rows) {
  return rows.flat().map(t => t.id);
}

test('orderRows: returns one row per tier, each containing exactly that tier\'s tasks', () => {
  const componentTasks = [
    mkTask('P1', [], 'pending'),
    mkTask('P2', [], 'pending'),
    mkTask('C1', ['P2'], 'pending'),
    mkTask('C2', ['P1'], 'pending'),
  ];
  const tiers = computeTiers(componentTasks);
  const ordered = orderRows(componentTasks, tiers);

  assert.equal(ordered.length, 2, 'expected one row per tier (tier 0 and tier 1)');
  assert.deepEqual(idsOf(ordered[0]).slice().sort(), ['P1', 'P2']);
  assert.deepEqual(idsOf(ordered[1]).slice().sort(), ['C1', 'C2']);
  assert.deepEqual(
    flattenedIds(ordered).sort(),
    idsOf(componentTasks).slice().sort(),
    'every input task should appear exactly once across the returned rows',
  );
});

test('orderRows: barycenter+transpose strictly reduces crossings below the status-sort baseline (tangled fixture)', () => {
  // Parents P1,P2 (tier0), children C1,C2 (tier1); edges P1->C2 & P2->C1;
  // all four share one status, so the status-sort init ties and falls back
  // to input order.
  const componentTasks = [
    mkTask('P1', [], 'pending'),
    mkTask('P2', [], 'pending'),
    mkTask('C1', ['P2'], 'pending'),
    mkTask('C2', ['P1'], 'pending'),
  ];
  const tiers = computeTiers(componentTasks);
  const edges = edgesFromDeps(componentTasks);

  const baseline = statusSortBaseline(componentTasks, tiers);
  assert.deepEqual(idsOf(baseline[0]), ['P1', 'P2']);
  assert.deepEqual(idsOf(baseline[1]), ['C1', 'C2']);
  const baselineCrossings = countCrossings(baseline, edges);
  assert.equal(baselineCrossings, 1, 'status-tied, input-order baseline should have exactly 1 crossing');

  const ordered = orderRows(componentTasks, tiers);
  assert.deepEqual(
    flattenedIds(ordered).sort(),
    idsOf(componentTasks).slice().sort(),
    'orderRows must return an exact re-permutation of its input',
  );

  const orderedCrossings = countCrossings(ordered, edges);
  assert.equal(orderedCrossings, 0);
  assert.ok(orderedCrossings < baselineCrossings, 'orderRows should strictly reduce crossings below the baseline');
});

test('orderRows: crossings never exceed the status-sort baseline (tangled + chain components)', () => {
  const tangled = [
    mkTask('P1', [], 'pending'),
    mkTask('P2', [], 'pending'),
    mkTask('C1', ['P2'], 'pending'),
    mkTask('C2', ['P1'], 'pending'),
  ];
  // Component X (chain, one node per tier) and component Y (chain) from the
  // partitionComponents multi-component fixture — a single node per tier
  // means the baseline is already crossing-free.
  const componentX = [mkTask('A', [], 'pending'), mkTask('B', ['A'], 'pending'), mkTask('C', ['B'], 'pending')];
  const componentY = [mkTask('D', [], 'pending'), mkTask('E', ['D'], 'pending')];

  for (const componentTasks of [tangled, componentX, componentY]) {
    const tiers = computeTiers(componentTasks);
    const edges = edgesFromDeps(componentTasks);
    const ordered = orderRows(componentTasks, tiers);
    assert.deepEqual(
      flattenedIds(ordered).sort(),
      idsOf(componentTasks).slice().sort(),
      'orderRows must return an exact re-permutation of its input',
    );

    const baselineCrossings = countCrossings(statusSortBaseline(componentTasks, tiers), edges);
    const orderedCrossings = countCrossings(ordered, edges);
    assert.ok(
      orderedCrossings <= baselineCrossings,
      `expected orderRows crossings (${orderedCrossings}) <= baseline (${baselineCrossings})`,
    );
  }
});

test('orderRows: deterministic across repeated calls on identical input', () => {
  const componentTasks = [
    mkTask('P1', [], 'pending'),
    mkTask('P2', [], 'pending'),
    mkTask('C1', ['P2'], 'pending'),
    mkTask('C2', ['P1'], 'pending'),
  ];
  const tiers = computeTiers(componentTasks);
  const first = orderRows(componentTasks, tiers);
  const second = orderRows(componentTasks, tiers);

  assert.deepEqual(
    flattenedIds(first).sort(),
    idsOf(componentTasks).slice().sort(),
    'orderRows must return an exact re-permutation of its input',
  );
  assert.deepEqual(first, second);
  assert.equal(JSON.stringify(first), JSON.stringify(second));
});
