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

const { prdTitle, aggregatePrdStatus, summarizePrdMembers } = grouping;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/prd_grouping.js';
const EXPECTED_FUNCTION_NAMES = [
  'prdTitle',
  'aggregatePrdStatus',
  'summarizePrdMembers',
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
