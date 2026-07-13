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

const { computeTiers, partitionComponents, countCrossings } = layout;

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
// reads d.id.
function mkTask(id, depIds = []) {
  return { id, deps: depIds.map(depId => ({ id: depId })) };
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

function idsOf(taskArray) {
  return taskArray.map(t => t.id);
}

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
