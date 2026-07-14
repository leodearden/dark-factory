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

const { computeTiers, partitionComponents, orderRows, countCrossings, computeNeighborhood } = layout;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/graph_layout.js';
const EXPECTED_FUNCTION_NAMES = [
  'computeTiers',
  'partitionComponents',
  'orderRows',
  'countCrossings',
  'computeNeighborhood',
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

test('orderRows: initial status-sort permutation ranks by STATUS_ORDER within an edge-free tier', () => {
  // All eight nodes are deliberately dep-free, so they all land in tier 0
  // with zero edges — no barycenter sweep or transpose swap can ever move a
  // node with no cross-tier neighbor (see barycenterSweep's "keeps its
  // current position" fallback and transposePass's strictly-reducing-only
  // acceptance), so the returned order is entirely determined by the
  // initial STATUS_ORDER sort. Statuses are listed in scrambled input order
  // so a passing assertion actually exercises the ranking rather than
  // coinciding with an identity permutation.
  const componentTasks = [
    mkTask('done-1', [], 'done'),
    mkTask('unknown-1', [], 'some-unrecognized-status'),
    mkTask('blocked-1', [], 'blocked'),
    mkTask('cancelled-1', [], 'cancelled'),
    mkTask('pending-1', [], 'pending'),
    mkTask('inprogress-1', [], 'in-progress'),
    mkTask('deferred-1', [], 'deferred'),
    mkTask('mergedeferred-1', [], 'merge-deferred'),
  ];
  const tiers = computeTiers(componentTasks);
  const ordered = orderRows(componentTasks, tiers);

  assert.equal(ordered.length, 1, 'all nodes are dep-free and share tier 0');
  assert.deepEqual(idsOf(ordered[0]), [
    'blocked-1',
    'inprogress-1',
    'mergedeferred-1',
    'pending-1',
    'deferred-1',
    'done-1',
    'cancelled-1',
    'unknown-1',
  ], 'expected blocked < in-progress < merge-deferred < pending < deferred < done < cancelled < unknown');
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

test('orderRows: 3-tier component with multiple nodes per tier and differing row lengths reduces crossings at both adjacent tier pairs', () => {
  // 2/3/2 nodes across three tiers (differing row lengths), all sharing one
  // status so the status-sort baseline ties to input order. The edges are
  // deliberately tangled at BOTH adjacent tier pairs (mirrors the
  // countCrossings "crossings across two adjacent tier pairs sum together"
  // fixture, run through orderRows instead): P1/P2's children are wired
  // crosswise (P2->M1, P1->M2, P1->M3), and M1..M3's children are wired
  // crosswise again (M2,M3->C1, M1->C2) — this exercises the interaction of
  // alternating down/up sweeps across more than two tiers, which the
  // 2-tier tangled fixture above can't.
  const componentTasks = [
    mkTask('P1', [], 'pending'),
    mkTask('P2', [], 'pending'),
    mkTask('M1', ['P2'], 'pending'),
    mkTask('M2', ['P1'], 'pending'),
    mkTask('M3', ['P1'], 'pending'),
    mkTask('C1', ['M2', 'M3'], 'pending'),
    mkTask('C2', ['M1'], 'pending'),
  ];
  const tiers = computeTiers(componentTasks);
  const edges = edgesFromDeps(componentTasks);

  const baseline = statusSortBaseline(componentTasks, tiers);
  assert.deepEqual(idsOf(baseline[0]), ['P1', 'P2']);
  assert.deepEqual(idsOf(baseline[1]), ['M1', 'M2', 'M3']);
  assert.deepEqual(idsOf(baseline[2]), ['C1', 'C2']);
  const baselineCrossings = countCrossings(baseline, edges);
  assert.equal(baselineCrossings, 4, 'input-order baseline should cross at both adjacent tier pairs (2 + 2)');

  const ordered = orderRows(componentTasks, tiers);
  assert.equal(ordered.length, 3, 'expected one row per tier');
  assert.deepEqual(
    flattenedIds(ordered).sort(),
    idsOf(componentTasks).slice().sort(),
    'orderRows must return an exact re-permutation of its input',
  );

  const orderedCrossings = countCrossings(ordered, edges);
  assert.ok(
    orderedCrossings < baselineCrossings,
    `expected orderRows to strictly reduce crossings (got ${orderedCrossings}, baseline ${baselineCrossings})`,
  );
});

test('orderRows: a long (tier0->tier2) edge never contributes to crossings, regardless of arrangement (no-dummy-node limitation)', () => {
  // A (tier0) -> B (tier1) -> C (tier2) is a normal adjacent chain. D also
  // lands in tier2 (1 + max(tier(A)=0, tier(B)=1) = 2) but has a direct dep
  // on A as well as B, so its edge list includes one adjacent edge (B->D)
  // and one long, skip-level edge (A->D) spanning tier0 to tier2. Per the
  // documented v1 no-dummy-node limitation, countCrossings only considers
  // edges whose endpoints lie in an ADJACENT tier pair, so A->D should never
  // affect the crossing count — pinned here by comparing with/without that
  // edge, both on orderRows' own output and after manually swapping C/D.
  const componentTasks = [
    mkTask('A', [], 'pending'),
    mkTask('B', ['A'], 'pending'),
    mkTask('C', ['B'], 'pending'),
    mkTask('D', ['A', 'B'], 'pending'),
  ];
  const tiers = computeTiers(componentTasks);
  assert.equal(tiers.get('D'), 2, 'D should land in tier2 via its longest path (through B), not tier1 via A directly');

  const edgesWithLongEdge = edgesFromDeps(componentTasks);
  const edgesWithoutLongEdge = edgesWithLongEdge.filter(e => !(e.from === 'A' && e.to === 'D'));

  const ordered = orderRows(componentTasks, tiers);
  assert.deepEqual(idsOf(ordered[0]), ['A']);
  assert.deepEqual(idsOf(ordered[1]), ['B']);
  assert.deepEqual(idsOf(ordered[2]).slice().sort(), ['C', 'D']);

  assert.equal(
    countCrossings(ordered, edgesWithLongEdge),
    countCrossings(ordered, edgesWithoutLongEdge),
    'the long A->D edge should not change the crossing count on orderRows\' own arrangement',
  );

  // Manually swap the tier2 row and re-check both edge sets — the long edge
  // must stay inert regardless of C/D's relative position.
  const swapped = ordered.map(row => row.slice());
  const lastTier = swapped[swapped.length - 1];
  [lastTier[0], lastTier[1]] = [lastTier[1], lastTier[0]];
  assert.equal(
    countCrossings(swapped, edgesWithLongEdge),
    countCrossings(swapped, edgesWithoutLongEdge),
    'the long A->D edge should not change the crossing count after swapping C/D either',
  );
});

test('orderRows: falls back to the status-sort baseline above the optimization size cap (large tangled component)', () => {
  // Builds a component with more nodes than graph_layout.js's internal
  // MAX_LAYOUT_OPTIMIZATION_NODES cap so the barycenter+transpose pipeline
  // short-circuits and the status-sort baseline is returned unchanged. The
  // parent/child pairing is a full crosswise reversal (child i depends on
  // parent at the mirrored index) — the same shape as the small tangled
  // fixture, scaled up — so if the optimization pipeline DID run despite the
  // cap, it would visibly reorder tier1 and reduce crossings; this pins
  // that it does not.
  const PAIR_COUNT = 76; // 152 nodes total, above the 150-node cap
  const parents = Array.from({ length: PAIR_COUNT }, (_, i) => mkTask(`P${i}`, [], 'pending'));
  const children = Array.from({ length: PAIR_COUNT }, (_, i) =>
    mkTask(`C${i}`, [`P${PAIR_COUNT - 1 - i}`], 'pending'),
  );
  const componentTasks = [...parents, ...children];
  const tiers = computeTiers(componentTasks);
  const edges = edgesFromDeps(componentTasks);

  const baseline = statusSortBaseline(componentTasks, tiers);
  const ordered = orderRows(componentTasks, tiers);

  assert.deepEqual(
    flattenedIds(ordered),
    flattenedIds(baseline),
    'above the optimization cap, orderRows should return the status-sort baseline unchanged (same order, not just same count)',
  );
  assert.equal(countCrossings(ordered, edges), countCrossings(baseline, edges));
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

// ---------------------------------------------------------------------------
// computeNeighborhood — verbatim port of tab_tasks.jsx's inline `neighborhood`
// memo (TaskGraph, ~lines 122-143): the selected task plus its transitive
// ancestors (walking deps upward) and transitive descendants (scanning
// dependents downward). Single tested source of truth reused by both the
// existing selection highlight and the new focus filter (focusSubset).
// ---------------------------------------------------------------------------

test('computeNeighborhood: returns null when selectedId is null or absent', () => {
  const tasks = [mkTask('A'), mkTask('B', ['A'])];
  assert.equal(computeNeighborhood(tasks, null), null);
  assert.equal(computeNeighborhood(tasks, undefined), null);
  assert.equal(computeNeighborhood(tasks), null);
});

test('computeNeighborhood: isolated selected node yields a Set containing only itself', () => {
  const tasks = [mkTask('A'), mkTask('B'), mkTask('C')];
  const nb = computeNeighborhood(tasks, 'A');
  assert.deepEqual([...nb].sort(), ['A']);
});

test('computeNeighborhood: linear chain focused mid-node includes all transitive ancestors and descendants', () => {
  const tasks = [mkTask('A'), mkTask('B', ['A']), mkTask('C', ['B']), mkTask('D', ['C'])];
  const nb = computeNeighborhood(tasks, 'B');
  assert.deepEqual([...nb].sort(), ['A', 'B', 'C', 'D']);
});

test('computeNeighborhood: diamond focused on a mid-tier node excludes its co-parent', () => {
  // A; B dep A; C dep A; D dep B,C. Focused on B: ancestors={A}, descendants={D}
  // (reached via B->D), but C (a sibling/co-parent of D via a different edge)
  // is NOT an ancestor or descendant of B, so it must be excluded — this pins
  // "strictly ancestors+descendants of the selected node", not the whole
  // weakly-connected component.
  const tasks = [mkTask('A'), mkTask('B', ['A']), mkTask('C', ['A']), mkTask('D', ['B', 'C'])];
  const nb = computeNeighborhood(tasks, 'B');
  assert.deepEqual([...nb].sort(), ['A', 'B', 'D']);
});

test('computeNeighborhood: a dep id outside the input list is ignored (no throw, not added)', () => {
  const tasks = [mkTask('A', ['OUTSIDE_OF_LIST']), mkTask('B', ['A'])];
  assert.doesNotThrow(() => computeNeighborhood(tasks, 'A'));
  const nb = computeNeighborhood(tasks, 'A');
  assert.deepEqual([...nb].sort(), ['A', 'B']);
  assert.ok(!nb.has('OUTSIDE_OF_LIST'));
});
