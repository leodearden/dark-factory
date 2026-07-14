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
// orderPrdGroups takes computeTiers as an injected parameter (see the plan's
// design decisions) — the node suite imports it straight from graph_layout.js,
// same as the browser injects window.DF_GRAPH_LAYOUT.computeTiers.
import layout from '../../src/dashboard/static/redux/graph_layout.js';

const { prdTitle, aggregatePrdStatus, summarizePrdMembers, groupTasksByPrd, orderPrdGroups } = grouping;
const { computeTiers } = layout;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/prd_grouping.js';
const EXPECTED_FUNCTION_NAMES = [
  'prdTitle',
  'aggregatePrdStatus',
  'summarizePrdMembers',
  'groupTasksByPrd',
  'orderPrdGroups',
];

// Builds a minimal task fixture — only the fields prd_grouping.js's
// functions actually read (id, status, prd, deps). Mirrors
// graph_layout.test.mjs's mkTask helper, extended with an optional `prd`.
function mkTask(id, { deps = [], status, prd } = {}) {
  return {
    id,
    ...(status !== undefined ? { status } : {}),
    ...(prd !== undefined ? { prd } : {}),
    deps: deps.map(depId => ({ id: depId })),
  };
}

// Convenience for status-precedence tests that don't care about id/deps/prd.
function tasksWithStatuses(statuses) {
  return statuses.map((status, i) => mkTask(`t${i}`, { status }));
}

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

// ---------------------------------------------------------------------------
// aggregatePrdStatus — precedence ladder: any-blocked > any-(in-progress or
// merge-deferred) > any-(pending or deferred) > all-done > cancelled.
// ---------------------------------------------------------------------------

test('aggregatePrdStatus: any blocked wins over everything else', () => {
  assert.equal(aggregatePrdStatus(tasksWithStatuses(['blocked', 'done'])), 'blocked');
});

test('aggregatePrdStatus: any in-progress (no blocked) wins over pending/done', () => {
  assert.equal(aggregatePrdStatus(tasksWithStatuses(['in-progress', 'pending'])), 'in-progress');
});

test('aggregatePrdStatus: merge-deferred counts as in-progress', () => {
  assert.equal(aggregatePrdStatus(tasksWithStatuses(['merge-deferred', 'done'])), 'in-progress');
});

test('aggregatePrdStatus: any pending (no blocked/in-progress) wins over done', () => {
  assert.equal(aggregatePrdStatus(tasksWithStatuses(['pending', 'done'])), 'pending');
});

test('aggregatePrdStatus: deferred counts as pending', () => {
  assert.equal(aggregatePrdStatus(tasksWithStatuses(['deferred', 'done'])), 'pending');
});

test('aggregatePrdStatus: all done (no blocked/active/pending) yields done', () => {
  assert.equal(aggregatePrdStatus(tasksWithStatuses(['done', 'done'])), 'done');
});

test('aggregatePrdStatus: done+cancelled (not ALL done) falls through to cancelled', () => {
  assert.equal(aggregatePrdStatus(tasksWithStatuses(['done', 'cancelled'])), 'cancelled');
});

test('aggregatePrdStatus: all cancelled yields cancelled', () => {
  assert.equal(aggregatePrdStatus(tasksWithStatuses(['cancelled'])), 'cancelled');
});

// ---------------------------------------------------------------------------
// summarizePrdMembers — per-bucket counts over ALL given members: done,
// inProgress ('in-progress' or 'merge-deferred'), blocked, pending ('pending'
// or 'deferred'), total. Feeds both the "n/m done" count (done/total) and the
// stacked-bar segment sizes.
// ---------------------------------------------------------------------------

test('summarizePrdMembers: mixed fixture yields exact per-bucket counts, total counts every member', () => {
  const tasks = tasksWithStatuses([
    'done', 'done', 'blocked', 'in-progress', 'merge-deferred', 'pending', 'deferred', 'cancelled',
  ]);
  assert.deepEqual(summarizePrdMembers(tasks), {
    done: 2,
    inProgress: 2,
    blocked: 1,
    pending: 2,
    total: 8,
  });
});

test('summarizePrdMembers: total always equals tasks.length, including cancelled/deferred members', () => {
  const tasks = tasksWithStatuses(['cancelled', 'cancelled', 'deferred']);
  assert.equal(summarizePrdMembers(tasks).total, 3);
});

test('summarizePrdMembers: empty input yields all-zero counts', () => {
  assert.deepEqual(summarizePrdMembers([]), { done: 0, inProgress: 0, blocked: 0, pending: 0, total: 0 });
});

// ---------------------------------------------------------------------------
// groupTasksByPrd — buckets by `prd` preserving first-seen-prd input order;
// each group's tasks preserve input order; prd===null tasks collapse into a
// single trailing "no PRD" group (flagged via `noPrd: true`), placed LAST
// regardless of where in the input the null-prd tasks appeared.
// ---------------------------------------------------------------------------

test('groupTasksByPrd: empty input yields an empty array', () => {
  assert.deepEqual(groupTasksByPrd([]), []);
});

test('groupTasksByPrd: two tasks sharing a non-null prd land in one group, preserving input order', () => {
  const tasks = [mkTask('A', { prd: 'p1' }), mkTask('B', { prd: 'p1' })];
  const groups = groupTasksByPrd(tasks);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].prd, 'p1');
  assert.deepEqual(groups[0].tasks.map(t => t.id), ['A', 'B']);
});

test('groupTasksByPrd: groups appear in first-seen-prd input order; each group preserves input order', () => {
  const tasks = [
    mkTask('A', { prd: 'p2' }),
    mkTask('B', { prd: 'p1' }),
    mkTask('C', { prd: 'p2' }),
    mkTask('D', { prd: 'p1' }),
  ];
  const groups = groupTasksByPrd(tasks);
  assert.deepEqual(groups.map(g => g.prd), ['p2', 'p1']);
  assert.deepEqual(groups[0].tasks.map(t => t.id), ['A', 'C']);
  assert.deepEqual(groups[1].tasks.map(t => t.id), ['B', 'D']);
});

test('groupTasksByPrd: null-prd tasks collapse into one trailing "no PRD" group, even when first in input', () => {
  const tasks = [
    mkTask('N1', { prd: null }),
    mkTask('A', { prd: 'p1' }),
    mkTask('N2', { prd: null }),
  ];
  const groups = groupTasksByPrd(tasks);
  assert.equal(groups.length, 2);
  assert.equal(groups[0].prd, 'p1');
  assert.ok(!groups[0].noPrd);
  assert.equal(groups[1].prd, null);
  assert.equal(groups[1].noPrd, true);
  assert.deepEqual(groups[1].tasks.map(t => t.id), ['N1', 'N2']);
});

test('groupTasksByPrd: all-null input yields a single flagged no-PRD group', () => {
  const tasks = [mkTask('A', { prd: null }), mkTask('B', { prd: null })];
  const groups = groupTasksByPrd(tasks);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].noPrd, true);
  assert.deepEqual(groups[0].tasks.map(t => t.id), ['A', 'B']);
});

// ---------------------------------------------------------------------------
// orderPrdGroups — condenses cross-PRD dep edges into a PRD-level mini-DAG,
// tiers it with the INJECTED computeTiers (a PRD consuming another PRD's
// tasks sits below it), tiebreaks within a tier by activity
// (blocked+in-progress count desc, then pending count desc), falls back to
// stable insertion order, and always force-appends the "no PRD" group last.
// ---------------------------------------------------------------------------

test('orderPrdGroups: a cross-PRD dep tiers the upstream PRD before the downstream one, overriding insertion order', () => {
  // B1 (prd B) depends on A1 (prd A) — B "consumes" A's task. B1 is listed
  // FIRST in the input so groupTasksByPrd's first-seen order is [B, A];
  // orderPrdGroups must still reorder to [A, B] via the mini-DAG tiering.
  const tasks = [
    mkTask('B1', { prd: 'B', deps: ['A1'] }),
    mkTask('A1', { prd: 'A' }),
  ];
  const groups = groupTasksByPrd(tasks);
  assert.deepEqual(groups.map(g => g.prd), ['B', 'A'], 'sanity: first-seen insertion order is B before A');

  const ordered = orderPrdGroups(groups, computeTiers);
  assert.deepEqual(ordered.map(g => g.prd), ['A', 'B']);
});

test('orderPrdGroups: within a tier, higher (blocked+in-progress) count sorts first', () => {
  const tasks = [
    mkTask('X1', { prd: 'X', status: 'pending' }),
    mkTask('Y1', { prd: 'Y', status: 'blocked' }),
    mkTask('Y2', { prd: 'Y', status: 'in-progress' }),
  ];
  const groups = groupTasksByPrd(tasks);
  const ordered = orderPrdGroups(groups, computeTiers);
  assert.deepEqual(ordered.map(g => g.prd), ['Y', 'X']);
});

test('orderPrdGroups: equal (blocked+in-progress), higher pending count sorts first as secondary tiebreak', () => {
  const tasks = [
    mkTask('P1', { prd: 'P', status: 'pending' }),
    mkTask('Q1', { prd: 'Q', status: 'pending' }),
    mkTask('Q2', { prd: 'Q', status: 'pending' }),
  ];
  const groups = groupTasksByPrd(tasks);
  const ordered = orderPrdGroups(groups, computeTiers);
  assert.deepEqual(ordered.map(g => g.prd), ['Q', 'P']);
});

test('orderPrdGroups: equal tier/activity/pending falls back to stable insertion order', () => {
  const tasks = [
    mkTask('M1', { prd: 'M', status: 'done' }),
    mkTask('N1', { prd: 'N', status: 'done' }),
  ];
  const groups = groupTasksByPrd(tasks);
  const ordered = orderPrdGroups(groups, computeTiers);
  assert.deepEqual(ordered.map(g => g.prd), ['M', 'N']);
});

test('orderPrdGroups: the "no PRD" group is always last, regardless of tier/activity', () => {
  // The no-PRD group's sole member is 'blocked' (maximal activity) and the
  // other PRD's sole member is 'done' (minimal) — if noPrd were ordered like
  // any other group it would sort FIRST; it must still come last.
  const tasks = [
    mkTask('Z1', { prd: null, status: 'blocked' }),
    mkTask('W1', { prd: 'W', status: 'done' }),
  ];
  const groups = groupTasksByPrd(tasks);
  const ordered = orderPrdGroups(groups, computeTiers);
  assert.deepEqual(ordered.map(g => g.prd), ['W', null]);
  assert.equal(ordered[ordered.length - 1].noPrd, true);
});

test('orderPrdGroups: deterministic across repeated calls on identical input', () => {
  const tasks = [
    mkTask('B1', { prd: 'B', deps: ['A1'] }),
    mkTask('A1', { prd: 'A' }),
    mkTask('N1', { prd: null }),
  ];
  const groups = groupTasksByPrd(tasks);
  const first = orderPrdGroups(groups, computeTiers);
  const second = orderPrdGroups(groups, computeTiers);
  assert.deepEqual(first.map(g => g.prd), second.map(g => g.prd));
  assert.deepEqual(first.map(g => g.prd), ['A', 'B', null]);
});
