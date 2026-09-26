// Module-contract tests for task_snapshot.js, the CLIENT reader of the /tasks
// snapshot unit whose server half is dashboard/src/dashboard/data/task_snapshot.py.
// Every census surface — the OrchTab pips, tiles, filter bar and Progress card,
// the Overview tile and pipeline, the topbar pill and the rail badge — reads
// its number through this module, so the decisions it makes are asserted here,
// where node can execute them. The .jsx files are `type="text/babel"` behind
// CDN Babel and cannot run under node; their WIRING to this module is pinned
// structurally in Python (test_tab_orchestrators.py, test_tab_overview.py,
// test_app_chrome_census.py).
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order: task_snapshot.js
// destructures window.DF_DATUM and window.DF_TASK_VOCAB at module scope with no
// fallback, and datum.js in turn destructures window.DF_ENDPOINT_STALENESS. So
// the shim goes in first and the chain is required through it —
// task_done_count.test.mjs::loadGuard has the same shape.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'task_vocab.js', 'task_snapshot.js'].map(name => REDUX + name);

function loadTaskSnapshot() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [, , snapshotApi] = LOAD_CHAIN.map(specifier => require(specifier));
  return { api: snapshotApi, window: win };
}

const { api: snapshot, window: loadedWindow } = loadTaskSnapshot();
const { projectCensus, censusOver, TASKS_ENDPOINT } = snapshot;
const { isDatum, EM_DASH } = loadedWindow.DF_DATUM;

const EXPECTED_FUNCTION_NAMES = ['projectCensus', 'censusOver'];
const EXPECTED_EXPORT_NAMES = [...EXPECTED_FUNCTION_NAMES, 'TASKS_ENDPOINT'];

// ── Fixtures: boundary sketch #1 ────────────────────────────────────────────
//
// One TaskCensus.to_wire() value per project, the shape data/census.py emits:
// `counts` keyed by status VALUE, `views` and `sub_views` keyed by view name.
// dark-factory is the sketch's own numbers; reify is a small second project so
// a fleet total is observably a SUM rather than one project's value.

const DF_CENSUS_VALUE = Object.freeze({
  counts: {
    'in-progress': 25, blocked: 10, review: 3, 'merge-deferred': 3, 'infra-hold': 2,
    pending: 1300, deferred: 10, done: 4000, cancelled: 106,
  },
  total: 5459,
  views: { in_flight: 43, backlog: 1310, terminal: 4106 },
  sub_views: { running: 25 },
});

const REIFY_CENSUS_VALUE = Object.freeze({
  counts: {
    'in-progress': 2, blocked: 1, review: 0, 'merge-deferred': 0, 'infra-hold': 0,
    pending: 5, deferred: 1, done: 20, cancelled: 2,
  },
  total: 31,
  views: { in_flight: 3, backlog: 6, terminal: 22 },
  sub_views: { running: 2 },
});

const AS_OF = '2026-09-26T10:00:00+00:00';
const SERVED_AT = '2026-09-26T10:00:05+00:00';
const RECEIVED_AT = 1_800_000_000_000;
const NOW = RECEIVED_AT + 1_000;
const TASKS_RECEIPT = Object.freeze({ servedAt: SERVED_AT, receivedAt: RECEIVED_AT });

// A wire Datum in *state*, honouring datum.py's unknown triad.
function datumIn(state, value, overrides) {
  const base =
    state === 'unknown'
      ? { value: null, as_of: null, state, reason: 'status map read failed', freshness_bound_seconds: 30 }
      : { value, as_of: AS_OF, state, reason: state === 'fresh' ? null : 'ReadTimeout', freshness_bound_seconds: 30 };
  return { ...base, ...overrides };
}

// One TASKS_SNAPSHOT[p] entry, the five keys TaskSnapshot.to_wire() emits.
function entryWith(census, rows) {
  return { census, rows, in_progress_live: 1, in_progress_stranded: 0, skew_seconds: 0 };
}

const FRESH_ROWS = datumIn('fresh', [{ id: 1, project: 'dark-factory', title: 't', status: 'in-progress' }]);

function sketchData(overrides) {
  return {
    TASKS_SNAPSHOT: {
      'dark-factory': entryWith(datumIn('fresh', DF_CENSUS_VALUE), FRESH_ROWS),
      reify: entryWith(datumIn('fresh', REIFY_CENSUS_VALUE), datumIn('fresh', [])),
    },
    __receipt: { [TASKS_ENDPOINT]: TASKS_RECEIPT },
    ...overrides,
  };
}

// ── The module ──────────────────────────────────────────────────────────────

test('the module exposes its readers and assigns window.DF_TASK_SNAPSHOT', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof snapshot[name], 'function', `${name} should be a function`);
  }
  assert.deepEqual(Object.keys(snapshot).sort(), EXPECTED_EXPORT_NAMES.slice().sort());
  // The browser half of the dual export: every census surface destructures
  // this global at module scope with no fallback.
  assert.equal(loadedWindow.DF_TASK_SNAPSHOT, snapshot);
});

test('TASKS_ENDPOINT is the receipt key data.js publishes the /tasks payload under', () => {
  assert.equal(TASKS_ENDPOINT, '/api/v2/dashboard/tasks');
});

// ── projectCensus: the served census, stamped with the /tasks receipt ───────

test('projectCensus: the wire census, stamped with the /tasks receipt', () => {
  // data.js registers TASKS_SNAPSHOT as PLAIN, so its nested Datums arrive
  // unstamped; without the receipt datumView could not age them.
  const data = sketchData();
  const wire = data.TASKS_SNAPSHOT['dark-factory'].census;
  const pristine = structuredClone(wire);

  const census = projectCensus(data, 'dark-factory');

  assert.equal(isDatum(census), true);
  assert.deepEqual(census.value, DF_CENSUS_VALUE);
  assert.equal(census.state, 'fresh');
  assert.equal(census.as_of, AS_OF);
  assert.equal(census._served_at, SERVED_AT);
  assert.equal(census._received_at, RECEIVED_AT);
  assert.notEqual(census, wire, 'the stamp must land on a copy');
  assert.deepEqual(wire, pristine, 'the wire census was mutated');
});

test('projectCensus: before the first /tasks payload, the census is not yet fetched', () => {
  const census = projectCensus({ TASKS_SNAPSHOT: {}, __receipt: {} }, 'dark-factory');
  assert.equal(census.state, 'unknown');
  assert.equal(census.reason, 'not yet fetched');
});

test('projectCensus: a project the delivered payload does not carry is named in the reason', () => {
  const census = projectCensus(sketchData(), 'hive');
  assert.equal(census.state, 'unknown');
  assert.equal(census.value, null);
  assert.match(census.reason, /hive/);
});

test('projectCensus: an entry without a census Datum is a reasoned hole — never 0, never a throw', () => {
  const malformed = [
    ['a null entry', null],
    ['an entry with no census', { rows: FRESH_ROWS }],
    ['a null census', entryWith(null, FRESH_ROWS)],
    ['a bare census value', entryWith(DF_CENSUS_VALUE, FRESH_ROWS)],
    ['a bare number', entryWith(43, FRESH_ROWS)],
  ];
  for (const [label, entry] of malformed) {
    const data = sketchData({ TASKS_SNAPSHOT: { 'dark-factory': entry } });
    let census;
    assert.doesNotThrow(() => {
      census = projectCensus(data, 'dark-factory');
    }, label);
    assert.equal(isDatum(census), true, label);
    assert.equal(census.state, 'unknown', label);
    assert.equal(census.value, null, label);
    assert.ok(census.reason, `${label}: the hole must say why`);
  }
});

test('projectCensus: an unknown served census stays unknown with the producer\'s reason', () => {
  const data = sketchData({
    TASKS_SNAPSHOT: { 'dark-factory': entryWith(datumIn('unknown'), FRESH_ROWS) },
  });
  const census = projectCensus(data, 'dark-factory');
  assert.equal(census.state, 'unknown');
  assert.equal(census.reason, 'status map read failed');
});

// ── censusOver: one project, or a member-wise total over several ────────────

test('censusOver: a single project in scope is that project\'s census', () => {
  const data = sketchData();
  assert.deepEqual(censusOver(data, ['dark-factory']), projectCensus(data, 'dark-factory'));
});

test('censusOver(data, null) sums EVERY project in the snapshot, member by member', () => {
  const fleet = censusOver(sketchData(), null);

  assert.equal(fleet.state, 'fresh');
  assert.deepEqual(fleet.value, {
    counts: {
      'in-progress': 27, blocked: 11, review: 3, 'merge-deferred': 3, 'infra-hold': 2,
      pending: 1305, deferred: 11, done: 4020, cancelled: 108,
    },
    total: 5490,
    views: { in_flight: 46, backlog: 1316, terminal: 4128 },
    sub_views: { running: 27 },
  });
});

test('censusOver: the fleet total is still a partition (sketch #4, client half)', () => {
  // A member-wise sum of partitions is a partition. Asserted rather than
  // assumed, because a sum that skipped a member (or double-counted a view)
  // would still produce plausible-looking numbers.
  const { value } = censusOver(sketchData(), null);
  const { in_flight, backlog, terminal } = value.views;

  assert.equal(in_flight + backlog + terminal, value.total);
  assert.equal(Object.values(value.counts).reduce((a, b) => a + b, 0), value.total);
  assert.ok(value.sub_views.running <= in_flight, 'running is a subset of in-flight');
});

test('censusOver: one unknown project makes the total unknown, naming that project', () => {
  // A partial sum is an under-count passed off as a total.
  const data = sketchData();
  data.TASKS_SNAPSHOT.reify = entryWith(datumIn('unknown'), datumIn('unknown'));

  const fleet = censusOver(data, null);

  assert.equal(fleet.state, 'unknown');
  assert.equal(fleet.value, null);
  assert.match(fleet.reason, /reify/);
  assert.match(fleet.reason, /status map read failed/);
});

test('censusOver: a named project absent from the snapshot is a hole in the total', () => {
  const fleet = censusOver(sketchData(), ['dark-factory', 'hive']);
  assert.equal(fleet.state, 'unknown');
  assert.match(fleet.reason, /hive/);
});

test('censusOver: an empty snapshot before the first fetch is not yet fetched', () => {
  const fleet = censusOver({ TASKS_SNAPSHOT: {}, __receipt: {} }, null);
  assert.equal(isDatum(fleet), true);
  assert.equal(fleet.state, 'unknown');
  assert.equal(fleet.reason, 'not yet fetched');
});

test('censusOver: a delivered payload with no project in scope says so, not "not yet fetched"', () => {
  const fleet = censusOver(sketchData({ TASKS_SNAPSHOT: {} }), null);
  assert.equal(fleet.state, 'unknown');
  assert.notEqual(fleet.reason, 'not yet fetched');
  assert.ok(fleet.reason);
});

test('censusOver: the total renders the placeholder, never a 0, when nothing is known', () => {
  const { datumView } = loadedWindow.DF_DATUM;
  const view = datumView(censusOver({ TASKS_SNAPSHOT: {}, __receipt: {} }, null), {
    now: NOW,
    format: () => {
      throw new Error('format must never run on a hole');
    },
  });
  assert.equal(view.text, EM_DASH);
});
