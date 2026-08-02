// Module-contract tests for task_status_counts.js — a plain-JS (no
// JSX/Babel) module holding the pure per-project header counting logic for
// the Tasks tab (tab_tasks.jsx). Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper that
// surfaces this suite in CI via its `**/*.test.mjs` glob — no wrapper
// change needed for this new file).
//
// task_status_counts.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { projectStatusCounts } from '...'` would come back undefined. We
// therefore default-import the module and destructure instead (mirrors
// prd_grouping.test.mjs / graph_layout.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';

import statusCounts from '../../src/dashboard/static/redux/task_status_counts.js';

const { projectStatusCounts } = statusCounts;

// The statuses the legacy single "N active" pip merged into one number.
// Every split-bucket test below is cross-checked against this set so the
// split can never silently drift from the population it replaced.
const LEGACY_ACTIVE_STATUSES = ['in-progress', 'blocked', 'merge-deferred'];

// Builds a minimal task fixture — only the fields task_status_counts.js
// actually reads (id, status). Trimmed from prd_grouping.test.mjs's mkTask.
function mkTask(id, { status } = {}) {
  return {
    id,
    ...(status !== undefined ? { status } : {}),
  };
}

// n tasks all carrying `status`, with ids unique across the whole list they
// are concatenated into (the prefix keeps ids distinct per status bucket).
function tasksWithStatus(status, n) {
  return Array.from({ length: n }, (_, i) => mkTask(`${status}-${i}`, { status }));
}

// ---------------------------------------------------------------------------
// projectStatusCounts — one pass, splitting the legacy merged "active" number
// into its three independent components.
// ---------------------------------------------------------------------------

test('projectStatusCounts: splits the 2026-07-30 false-alarm population into three counts', () => {
  // The shape that triggered the false alarm: the header read "38 active"
  // against a max_concurrent_tasks cap of 24, so an operator saw a cap
  // breach that never happened — only the 24 in-progress tasks are bounded
  // by the cap; blocked and merge-deferred tasks hold no agent slot.
  const tasks = [
    ...tasksWithStatus('in-progress', 24),
    ...tasksWithStatus('blocked', 9),
    ...tasksWithStatus('merge-deferred', 5),
  ];

  const counts = projectStatusCounts(tasks);

  assert.equal(counts.running, 24);
  assert.equal(counts.blocked, 9);
  assert.equal(counts.mergeDeferred, 5);
  // The whole point: `running` is NOT the merged 38.
  assert.notEqual(counts.running, 38);
  assert.equal(counts.total, 38);
});

test('projectStatusCounts: "deferred" is not "merge-deferred", and neither is running', () => {
  // 'deferred' and 'merge-deferred' share a suffix, and a substring/prefix
  // test (or an `.includes("deferred")`) would conflate them. They are
  // wholly different states: merge-deferred work is finished and waiting on
  // the merge lane; deferred work is parked and not in flight at all.
  const tasks = [
    ...tasksWithStatus('deferred', 7),
    ...tasksWithStatus('merge-deferred', 3),
  ];

  const counts = projectStatusCounts(tasks);

  assert.equal(counts.mergeDeferred, 3, 'only literal "merge-deferred" counts as merge-deferred');
  assert.equal(counts.running, 0, 'neither deferred nor merge-deferred is running');
  assert.equal(counts.blocked, 0);
  assert.equal(counts.pending, 0, 'deferred is not pending');
  assert.equal(counts.total, 10, 'deferred still counts toward total');
});

test('projectStatusCounts: counts pending, done and total; parks the rest in total only', () => {
  const tasks = [
    ...tasksWithStatus('in-progress', 2),
    ...tasksWithStatus('blocked', 1),
    ...tasksWithStatus('merge-deferred', 1),
    ...tasksWithStatus('pending', 6),
    ...tasksWithStatus('done', 4),
    ...tasksWithStatus('cancelled', 3),
    ...tasksWithStatus('deferred', 2),
    ...tasksWithStatus('some-future-status', 5),
  ];

  const counts = projectStatusCounts(tasks);

  assert.equal(counts.pending, 6);
  assert.equal(counts.done, 4);
  assert.equal(counts.running, 2);
  assert.equal(counts.blocked, 1);
  assert.equal(counts.mergeDeferred, 1);
  // cancelled / deferred / an unrecognized status reach total and nothing else.
  assert.equal(counts.total, 24);
  const bucketed = counts.running + counts.blocked + counts.mergeDeferred
    + counts.pending + counts.done;
  assert.equal(counts.total - bucketed, 10, 'cancelled + deferred + unknown are total-only');
});

test('projectStatusCounts: a task with no status at all lands in total only', () => {
  const counts = projectStatusCounts([mkTask('t1'), mkTask('t2', { status: 'in-progress' })]);

  assert.equal(counts.total, 2);
  assert.equal(counts.running, 1);
  assert.equal(counts.blocked, 0);
  assert.equal(counts.mergeDeferred, 0);
  assert.equal(counts.pending, 0);
  assert.equal(counts.done, 0);
});

test('projectStatusCounts: empty / undefined / null input yields an all-zero result', () => {
  // The per-project header renders before task data has necessarily
  // arrived; throwing here would blank the whole Tasks tab.
  const allZero = { total: 0, running: 0, blocked: 0, mergeDeferred: 0, pending: 0, done: 0 };
  for (const input of [[], undefined, null]) {
    assert.deepEqual(
      projectStatusCounts(input),
      allZero,
      `projectStatusCounts(${JSON.stringify(input)}) should be all-zero, not a throw`,
    );
  }
});

test('projectStatusCounts: the three components exactly partition the legacy merged set', () => {
  // The decomposition invariant. The legacy pip showed ONE number over
  // {in-progress, blocked, merge-deferred}; the split must reproduce that
  // population exactly — losing a task would under-report in-flight work,
  // and double-counting one would re-create the inflated number this task
  // exists to remove. `statusMatches`'s `active` filter toggle still uses
  // the same three-status disjunction, so this also pins that the header
  // and the filter keep describing the same population.
  const tasks = [
    ...tasksWithStatus('in-progress', 4),
    ...tasksWithStatus('blocked', 3),
    ...tasksWithStatus('merge-deferred', 2),
    ...tasksWithStatus('pending', 5),
    ...tasksWithStatus('done', 6),
    ...tasksWithStatus('cancelled', 1),
    ...tasksWithStatus('deferred', 1),
  ];

  const counts = projectStatusCounts(tasks);
  const legacyActive = tasks.filter(t => LEGACY_ACTIVE_STATUSES.includes(t.status)).length;

  assert.equal(legacyActive, 9);
  assert.equal(
    counts.running + counts.blocked + counts.mergeDeferred,
    legacyActive,
    'the split must neither lose nor double-count a task from the legacy merged set',
  );
});
