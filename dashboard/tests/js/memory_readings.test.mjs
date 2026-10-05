// Module-contract tests for memory_readings.js, the CLIENT reader of the two
// memory payloads: the write-queue Datum served by /memory (server half
// dashboard/src/dashboard/data/memory.py::write_queue_datum) and the MEMORY_OPS
// block served by /memory-graphs (data/write_journal.py::get_memory_ops via
// redux_api.shape_memory_graphs). MemoryTab, the Overview and the topbar read
// both through this module, so the decisions it makes are asserted here, where
// node can execute them. Their WIRING is pinned structurally in Python
// (test_tab_memory.py, test_tab_overview.py, test_app_chrome_census.py).
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order, exactly as
// merge_queue.test.mjs does. The shim is REMOVED once the modules are loaded:
// every fixture carries its own `__receipt` map, so a reader that still reached
// for a browser global would throw here instead of passing by accident.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'memory_readings.js'].map(name => REDUX + name);

function loadMemoryReadings() {
  globalThis.window = { DF_ENDPOINT_STALENESS: staleness };
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [datumApi, readingsApi] = LOAD_CHAIN.map(specifier => require(specifier));
  delete globalThis.window;
  return { datum: datumApi, readings: readingsApi };
}

const { datum, readings } = loadMemoryReadings();
const {
  writeQueue,
  queueCountsText,
  queueHint,
  queueHealth,
  opsTotals,
  opsCaption,
  opsTotalText,
  newestHourOps,
} = readings;
const { isDatum, EM_DASH } = datum;

// ── Fixtures ────────────────────────────────────────────────────────────────

const MEMORY_ENDPOINT = '/api/v2/dashboard/memory';
const MEMORY_GRAPHS_ENDPOINT = '/api/v2/dashboard/memory-graphs';
const SERVED_AT = '2026-10-03T12:00:30+00:00';
const MEASURED_AT = '2026-10-03T12:00:28+00:00';
const RECEIVED_AT = 1_800_000_000_000;
const RECEIPT = Object.freeze({ servedAt: SERVED_AT, receivedAt: RECEIVED_AT });

function counts(overrides) {
  return { pending: 0, retry: 0, dead: 0, oldest_pending_age_seconds: null, ...overrides };
}

// A served queue.stats Datum, honouring datum.py's unknown triad.
function queueStats(state, value) {
  if (state === 'unknown') {
    return {
      value: null,
      as_of: null,
      state,
      reason: 'http://localhost:8002: ConnectError: refused',
      freshness_bound_seconds: 30,
    };
  }
  return {
    value,
    as_of: MEASURED_AT,
    state,
    reason: state === 'fresh' ? null : 'measured 90s before it was served, past the 30s freshness bound',
    freshness_bound_seconds: 30,
  };
}

function memoryData(stats, receipts = { [MEMORY_ENDPOINT]: RECEIPT }) {
  return {
    MEMORY_STATUS: { queue: { stats, spark: { labels: [], values: [] } } },
    __receipt: receipts,
  };
}

const MEMORY_OPS = Object.freeze({
  labels: ['10:00', '11:00', '12:00'],
  reads: [4, 3, 3],
  writes: [1, 2, 2],
  other: [0, 2, 0],
  total: [5, 7, 5],
  totals: { reads: 10, writes: 5, other: 2, total: 17 },
  by_operation: [
    { label: 'search', value: 10 },
    { label: 'add_memory', value: 5 },
    { label: 'compact', value: 2 },
  ],
});

function opsData(ops = MEMORY_OPS, receipts = { [MEMORY_GRAPHS_ENDPOINT]: RECEIPT }) {
  return { MEMORY_OPS: ops, __receipt: receipts };
}

// ── Write queue ─────────────────────────────────────────────────────────────

test('writeQueue: before the first /memory receipt the queue is not yet fetched', () => {
  const q = writeQueue(memoryData(queueStats('fresh', counts({ pending: 4 })), {}));
  assert.equal(q.state, 'unknown');
  assert.equal(q.reason, 'not yet fetched');
});

test('writeQueue: a served unknown queue keeps the SERVER\'s reason verbatim', () => {
  const q = writeQueue(memoryData(queueStats('unknown')));
  assert.equal(q.state, 'unknown');
  assert.equal(q.value, null);
  assert.equal(q.reason, 'http://localhost:8002: ConnectError: refused');
});

test('writeQueue: a fresh queue is the served Datum, stamped with its receipt', () => {
  const served = queueStats('fresh', counts({ pending: 4, retry: 1 }));
  const q = writeQueue(memoryData(served));
  assert.ok(isDatum(q));
  assert.equal(q.state, 'fresh');
  assert.deepEqual(q.value, served.value);
  assert.equal(q.as_of, MEASURED_AT);
  assert.equal(q._served_at, SERVED_AT);
  assert.equal(q._received_at, RECEIVED_AT);
});

test('writeQueue: a delivered payload with no queue.stats is a hole naming the /memory payload', () => {
  const q = writeQueue({ MEMORY_STATUS: { queue: {} }, __receipt: { [MEMORY_ENDPOINT]: RECEIPT } });
  assert.equal(q.state, 'unknown');
  assert.match(q.reason, /\/memory payload/);
});

test('queueCountsText: pending, retry and dead in one line', () => {
  assert.equal(
    queueCountsText(counts({ pending: 4, retry: 1, dead: 0, oldest_pending_age_seconds: 9 })),
    '4 pending · 1 retry · 0 dead',
  );
});

test('queueHint: the oldest pending age when measured, idle when nothing waits, null for a hole', () => {
  assert.equal(queueHint(writeQueue(memoryData(queueStats('fresh', counts({ oldest_pending_age_seconds: 12 }))))), '12s oldest');
  assert.equal(queueHint(writeQueue(memoryData(queueStats('fresh', counts())))), 'idle');
  assert.equal(queueHint(writeQueue(memoryData(queueStats('unknown')))), null);
});

test('queueHealth: an unmeasured queue is amber, never green-ok and never red', () => {
  assert.deepEqual(queueHealth(writeQueue(memoryData(queueStats('unknown')))), { ok: true, warn: true });
});

test('queueHealth: a fresh queue follows the Overview rule', () => {
  const health = c => queueHealth(writeQueue(memoryData(queueStats('fresh', counts(c)))));
  assert.equal(health({ dead: 1 }).ok, false);
  assert.equal(health({ pending: 6 }).warn, true);
  assert.equal(health({ retry: 1 }).warn, true);
  assert.deepEqual(health({}), { ok: true, warn: false });
});

test('queueHealth: a stale queue always warns', () => {
  assert.equal(queueHealth(writeQueue(memoryData(queueStats('stale', counts())))).warn, true);
});

// ── Memory ops (PRD sketch #11, at the client) ──────────────────────────────

test('opsCaption: the three window totals the caption states', () => {
  assert.equal(opsCaption(MEMORY_OPS.totals), '10 reads · 5 writes · 2 other');
});

test('opsTotalText: the donut centre is the served window total', () => {
  assert.equal(opsTotalText(opsData()), '17');
});

test('the caption\'s three numbers sum to the donut\'s total', () => {
  const captioned = opsCaption(opsTotals(opsData()).value).match(/\d+/g).map(Number);
  assert.equal(String(captioned.reduce((sum, n) => sum + n, 0)), opsTotalText(opsData()));
});

test('before the /memory-graphs receipt the ops totals are not yet fetched', () => {
  const data = opsData(MEMORY_OPS, {});
  assert.equal(opsTotalText(data), EM_DASH);
  const totals = opsTotals(data);
  assert.equal(totals.state, 'unknown');
  assert.equal(totals.reason, 'not yet fetched');
});

test('newestHourOps: the LAST hourly total, as a Datum', () => {
  const newest = newestHourOps(opsData({ ...MEMORY_OPS, total: [5, 7, 9] }));
  assert.ok(isDatum(newest));
  assert.equal(newest.state, 'fresh');
  assert.equal(newest.value, 9);
});

test('newestHourOps: an empty series with a receipt is a hole saying so', () => {
  const newest = newestHourOps(opsData({ ...MEMORY_OPS, total: [] }));
  assert.equal(newest.state, 'unknown');
  assert.equal(newest.reason, 'no ops recorded in this window');
});

test('newestHourOps: before the /memory-graphs receipt it is not yet fetched', () => {
  const newest = newestHourOps(opsData(MEMORY_OPS, {}));
  assert.equal(newest.state, 'unknown');
  assert.equal(newest.reason, 'not yet fetched');
});
