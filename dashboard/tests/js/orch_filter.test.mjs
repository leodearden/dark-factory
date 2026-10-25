// Module-contract tests for orch_filter.js — the pure empty-state sentence for
// the Orchestrators tab's multi-select VIEW filter (OrchTab in tabs.jsx). Run
// via `node --test` (dashboard/tests/test_graph_layout_js.py surfaces every
// **/*.test.mjs here in CI).
//
// The facets are the census VIEWS, taken from task_snapshot.js::CENSUS_VIEWS
// so the view labels exist once. orch_filter.js destructures
// window.DF_TASK_SNAPSHOT at module scope with no fallback, so it is loaded
// through a window shim in index.html's order: endpoint_staleness → datum →
// task_vocab → task_snapshot → orch_filter (task_snapshot.test.mjs has the same
// shape).
//
// The regression the sentence guards (task 3313) is described once, in
// orch_filter.js's own header comment.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'task_vocab.js', 'task_snapshot.js', 'orch_filter.js'].map(n => REDUX + n);

function loadOrchFilter() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const loaded = LOAD_CHAIN.map(specifier => require(specifier));
  return { api: loaded[loaded.length - 1], window: win };
}

const { api: orchFilter, window: loadedWindow } = loadOrchFilter();
const { orchEmptyLabel } = orchFilter;
const { CENSUS_VIEWS } = loadedWindow.DF_TASK_SNAPSHOT;

const EXPECTED_FUNCTION_NAMES = ['orchEmptyLabel'];

const NONE_SELECTED = 'No filters selected — choose in-flight, backlog or terminal above';

test('the module exposes orchEmptyLabel and assigns window.DF_ORCH_FILTER', () => {
  assert.deepEqual(Object.keys(orchFilter).sort(), EXPECTED_FUNCTION_NAMES);
  assert.equal(typeof orchEmptyLabel, 'function');
  assert.equal(loadedWindow.DF_ORCH_FILTER, orchFilter);
});

// ---------------------------------------------------------------------------
// orchEmptyLabel — all eight combinations of the three views, named in the
// census's view order, which is also the order of the filter buttons above the
// table.
// ---------------------------------------------------------------------------

const ALL_COMBINATIONS = [
  [{ in_flight: false, backlog: false, terminal: false }, NONE_SELECTED],
  [{ in_flight: true, backlog: false, terminal: false }, 'No in-flight tasks'],
  [{ in_flight: false, backlog: true, terminal: false }, 'No backlog tasks'],
  [{ in_flight: false, backlog: false, terminal: true }, 'No terminal tasks'],
  [{ in_flight: true, backlog: true, terminal: false }, 'No in-flight or backlog tasks'],
  [{ in_flight: true, backlog: false, terminal: true }, 'No in-flight or terminal tasks'],
  [{ in_flight: false, backlog: true, terminal: true }, 'No backlog or terminal tasks'],
  [{ in_flight: true, backlog: true, terminal: true }, 'No in-flight, backlog or terminal tasks'],
];

for (const [filter, expected] of ALL_COMBINATIONS) {
  const on = Object.keys(filter).filter(k => filter[k]);
  const label = on.length === 0 ? 'none' : on.join('+');
  test(`orchEmptyLabel: ${label} selected -> "${expected}"`, () => {
    assert.equal(orchEmptyLabel(filter), expected);
  });
}

test('orchEmptyLabel: the facets ARE the census views — labels single-sourced', () => {
  // A hand copy of the view labels here, kept equal by a parity test, is the
  // twin pattern PRD decision 4 rejected.
  const all = Object.fromEntries(CENSUS_VIEWS.map(v => [v.key, true]));
  const labels = CENSUS_VIEWS.map(v => v.label);
  assert.equal(
    orchEmptyLabel(all),
    `No ${labels.slice(0, -1).join(', ')} or ${labels[labels.length - 1]} tasks`,
  );
  for (const v of CENSUS_VIEWS) {
    assert.equal(orchEmptyLabel({ [v.key]: true }), `No ${v.label} tasks`);
  }
});

test('orchEmptyLabel: the none-selected sentence names the three view labels', () => {
  for (const v of CENSUS_VIEWS) assert.ok(NONE_SELECTED.includes(v.label), v.label);
  assert.equal(orchEmptyLabel({}), NONE_SELECTED);
});

// ---------------------------------------------------------------------------
// Canonical ordering — from the view table, NOT from Object.keys order.
// flipFilter rebuilds the per-pid object with spread on every click, so key
// insertion order tracks the operator's click history.
// ---------------------------------------------------------------------------

test('orchEmptyLabel: reversed key insertion order still reads in view order', () => {
  assert.equal(orchEmptyLabel({ terminal: true, in_flight: true }), 'No in-flight or terminal tasks');
  assert.equal(
    orchEmptyLabel({ terminal: true, backlog: true, in_flight: true }),
    'No in-flight, backlog or terminal tasks',
  );
});

test('orchEmptyLabel: omitted and falsy keys are off', () => {
  assert.equal(orchEmptyLabel({ in_flight: true }), 'No in-flight tasks');
  assert.equal(orchEmptyLabel({ in_flight: true, backlog: 0, terminal: '' }), 'No in-flight tasks');
});

test('orchEmptyLabel: the retired active/pending/complete keys no longer name a facet', () => {
  // A browser holding the old object under the old storage key never reaches
  // here (OrchTab persists under a new key), but a stray old-shaped object
  // must read as "nothing selected", not as a phantom facet.
  assert.equal(orchEmptyLabel({ active: true, pending: true, complete: true }), NONE_SELECTED);
});

// ---------------------------------------------------------------------------
// Defensive input — a non-object must not throw (a throw during render takes
// out all of OrchTab). It mirrors OrchTab's own default, in-flight only, so the
// sentence agrees with what the table would be showing.
// ---------------------------------------------------------------------------

test('orchEmptyLabel: a non-object falls back to the in-flight default', () => {
  for (const input of [undefined, null, 'all', 42]) {
    assert.equal(orchEmptyLabel(input), 'No in-flight tasks', String(input));
  }
});
