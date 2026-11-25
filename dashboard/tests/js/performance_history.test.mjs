// Module-contract tests for performance_history.js, the CLIENT reader of each
// served PERFORMANCE[label] entry's hour-bucketed series, whose server half is
// dashboard/src/dashboard/data/performance.py::PerformanceHistory. PerfTab's
// header sparks and its per-project ttc trend read the history through this
// module, so its decisions are asserted here, where node can execute them.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import { performanceEntry } from './wire_fixtures.mjs';

const SCRIPT = '../../src/dashboard/static/redux/performance_history.js';

function loadPerformanceHistory() {
  const win = {};
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  delete require.cache[require.resolve(SCRIPT)];
  return { api: require(SCRIPT), window: win };
}

const { api: performanceHistory, window: loadedWindow } = loadPerformanceHistory();
const { projectHistory, meanHistorySpark } = performanceHistory;

const NO_HISTORY = { labels: [], p50: [], p95: [], one_pass: [], escalation: [] };
const H10 = '2026-10-04T10:00';
const H11 = '2026-10-04T11:00';
const H12 = '2026-10-04T12:00';
const H13 = '2026-10-04T13:00';

test('the module exposes exactly projectHistory and meanHistorySpark, and is window.DF_PERFORMANCE_HISTORY', () => {
  assert.deepEqual(Object.keys(performanceHistory).sort(), ['meanHistorySpark', 'projectHistory']);
  for (const fn of Object.values(performanceHistory)) assert.equal(typeof fn, 'function');
  assert.equal(loadedWindow.DF_PERFORMANCE_HISTORY, performanceHistory);
});

test('a project with no entry, or a null one, has no history', () => {
  assert.deepEqual(projectHistory({}, 'hive'), NO_HISTORY);
  assert.deepEqual(projectHistory({ hive: null }, 'hive'), NO_HISTORY);
});

test("a well-formed entry's series are read off its one history_labels axis as served", () => {
  const perf = {
    hive: performanceEntry({
      cards: null, labels: [H10, H11], p50: [1000, 2000], p95: [3000, 4000], onePass: [50, 100], escalation: [0, 25],
    }),
  };

  assert.deepEqual(projectHistory(perf, 'hive'), {
    labels: [H10, H11], p50: [1000, 2000], p95: [3000, 4000], one_pass: [50, 100], escalation: [0, 25],
  });
});

test('an entry with no history_labels axis is refused, naming the project and the axis', () => {
  const { history_labels: _axis, ...axisless } = performanceEntry({ cards: null, labels: [H10] });

  assert.throws(() => projectHistory({ hive: axisless }, 'hive'), error => (
    error instanceof Error && /hive/.test(error.message) && /history_labels/.test(error.message)
  ));
});

for (const [name, entry, length] of [
  ['p95', performanceEntry({ cards: null, labels: [H10, H11], p95: [3000] }), 1],
  ['one_pass', performanceEntry({ cards: null, labels: [H10, H11], onePass: [50, 100, 0] }), 3],
]) {
  test(`a ${name} series off the axis is refused, naming the project, the series and both lengths`, () => {
    assert.throws(() => projectHistory({ hive: entry }, 'hive'), error => (
      error instanceof Error
        && /hive/.test(error.message)
        && new RegExp(`\\b${name}\\b`).test(error.message)
        && new RegExp(`\\b${length}\\b`).test(error.message)
        && /\b2\b/.test(error.message)
    ));
  });
}

// Inserted out of label order, so the spark's order is the labels', not the projects'.
const OVERLAPPING = {
  reify: performanceEntry({
    cards: null, labels: [H11, H12, H13],
    p50: [4000, 2000, 8000], p95: [9000, 7000, 3000], onePass: [40, 50, 100], escalation: [25, 10, 0],
  }),
  hive: performanceEntry({
    cards: null, labels: [H10, H11, H12],
    p50: [1000, 2001, 3000], p95: [5000, 6000, 7000], onePass: [50, 100, 0], escalation: [0, 20, 0],
  }),
  solar: performanceEntry({ cards: null, labels: [H12], p50: [1500], p95: [8000], onePass: [50], escalation: [1] }),
};

for (const [series, expected] of [
  // H11 (4000+2001)/2 = 3000.5; H12 (2000+3000+1500)/3 = 2166.7
  ['p50', [1000, 3001, 2167, 8000]],
  // H12 (7000+7000+8000)/3 = 7333.3
  ['p95', [5000, 7500, 7333, 3000]],
  // H12 (50+0+50)/3 = 33.33
  ['one_pass', [50, 70, 33.3, 100]],
  // H11 (25+20)/2 = 22.5; H12 (10+0+1)/3 = 3.67
  ['escalation', [0, 22.5, 3.7, 0]],
]) {
  test(`the ${series} spark is one mean per distinct hour, in label order, over the projects holding it`, () => {
    assert.deepEqual(meanHistorySpark(OVERLAPPING, series), expected);
  });
}

test('the spark over no projects is empty', () => {
  assert.deepEqual(meanHistorySpark({}, 'p50'), []);
});

for (const [name, perf] of [['some projects', OVERLAPPING], ['no projects', {}]]) {
  test(`an unknown series is refused over ${name}, naming it and every valid series`, () => {
    assert.throws(() => meanHistorySpark(perf, 'onePass'), error => (
      error instanceof Error
        && /\bonePass\b/.test(error.message)
        && ['p50', 'p95', 'one_pass', 'escalation'].every(series => new RegExp(`\\b${series}\\b`).test(error.message))
    ));
  });
}
