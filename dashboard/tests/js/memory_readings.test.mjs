// Module-contract tests for memory_readings.js, the CLIENT reader of the two
// memory payloads: the write-queue and plan-route Datums served by /memory
// (server halves dashboard/src/dashboard/data/memory.py::write_queue_datum and
// ::plan_route_datum) and the MEMORY_OPS
// block served by /memory-graphs, whose totals and newest_hour_total are served
// Datums (data/write_journal.py::get_memory_ops via
// redux_api.shape_memory_graphs; server half test_memory_graphs.py). MemoryTab, the Overview and the topbar read
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
  planRoute,
  planRouteText,
  planRouteHealth,
} = readings;
const { isDatum, datumView, EM_DASH } = datum;

// ── Fixtures ────────────────────────────────────────────────────────────────

const MEMORY_ENDPOINT = '/api/v2/dashboard/memory';
const MEMORY_GRAPHS_ENDPOINT = '/api/v2/dashboard/memory-graphs';
const SERVED_AT = '2026-10-03T12:00:30+00:00';
const MEASURED_AT = '2026-10-03T12:00:28+00:00';
const RECEIVED_AT = 1_800_000_000_000;
const RECEIPT = Object.freeze({ servedAt: SERVED_AT, receivedAt: RECEIVED_AT });

function counts(overrides) {
  return {
    pending: 0, retry: 0, parked: 0, dead: 0, awaiting_operator: 0,
    oldest_pending_age_seconds: null, ...overrides,
  };
}

// The value a dashboard process predating parked rows still serves until it
// restarts: it has neither a parked nor an awaiting_operator count.
function countsPredatingParkedRows(overrides) {
  const { parked, awaiting_operator, ...value } = counts(overrides);
  return value;
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

const OPS_TOTALS = Object.freeze({ reads: 10, writes: 5, other: 2, total: 17 });
const OPS_REASON = 'the write journal query failed: OperationalError: database is locked';

// A served MEMORY_OPS reading, honouring datum.py's unknown triad.
function opsReading(state, value) {
  if (state === 'unknown') {
    return { value: null, as_of: null, state, reason: OPS_REASON, freshness_bound_seconds: 60 };
  }
  return { value, as_of: MEASURED_AT, state, reason: null, freshness_bound_seconds: 60 };
}

// The MEMORY_OPS block as shape_memory_graphs serves it: a hole serves no series.
function memoryOps(state, newestHour = 5) {
  if (state === 'unknown') {
    return {
      labels: [], reads: [], writes: [], other: [], total: [], by_operation: [],
      totals: opsReading('unknown'),
      newest_hour_total: opsReading('unknown'),
    };
  }
  return {
    labels: ['10:00', '11:00', '12:00'],
    reads: [4, 3, 3],
    writes: [1, 2, 2],
    other: [0, 2, 0],
    total: [5, 7, newestHour],
    by_operation: [
      { label: 'search', value: 10 },
      { label: 'add_memory', value: 5 },
      { label: 'compact', value: 2 },
    ],
    totals: opsReading(state, OPS_TOTALS),
    newest_hour_total: opsReading(state, newestHour),
  };
}

const MEMORY_OPS = Object.freeze(memoryOps('fresh'));

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

test('queueCountsText: pending, retry, parked and dead in one line', () => {
  assert.equal(
    queueCountsText(counts({ pending: 4, retry: 1, parked: 2, oldest_pending_age_seconds: 9 })),
    '4 pending · 1 retry · 2 parked · 0 dead',
  );
});

test('queueCountsText: an unmeasured parked count is a hole, never a fabricated 0', () => {
  assert.equal(
    queueCountsText(countsPredatingParkedRows({ pending: 4, retry: 1 })),
    '4 pending · 1 retry · ' + EM_DASH + ' parked · 0 dead',
  );
});

test('queueHint: the oldest pending age when measured, idle when nothing waits, null for a hole', () => {
  assert.equal(queueHint(writeQueue(memoryData(queueStats('fresh', counts({ oldest_pending_age_seconds: 12 }))))), '12s oldest');
  assert.equal(queueHint(writeQueue(memoryData(queueStats('fresh', counts())))), 'idle');
  assert.equal(queueHint(writeQueue(memoryData(queueStats('unknown')))), null);
});

test('queueHint: a parked backlog explains the climbing oldest age', () => {
  const served = counts({ parked: 111, oldest_pending_age_seconds: 86400 });
  assert.equal(queueHint(writeQueue(memoryData(queueStats('fresh', served)))), '111 parked · 86400s oldest');
});

test('queueHealth: an unmeasured queue is amber, never green-ok and never red', () => {
  assert.deepEqual(queueHealth(writeQueue(memoryData(queueStats('unknown')))), { ok: true, warn: true });
});

test('queueHealth: a fresh queue follows the Overview rule', () => {
  const health = c => queueHealth(writeQueue(memoryData(queueStats('fresh', counts(c)))));
  assert.equal(health({ dead: 1, awaiting_operator: 1 }).ok, false);
  assert.equal(health({ parked: 111, awaiting_operator: 111 }).ok, false);
  assert.equal(health({ pending: 6 }).warn, true);
  assert.equal(health({ retry: 1 }).warn, true);
  assert.deepEqual(health({}), { ok: true, warn: false });
});

test('queueHealth: red follows the served awaiting_operator count, not a client-side partition', () => {
  const health = c => queueHealth(writeQueue(memoryData(queueStats('fresh', counts(c)))));
  assert.equal(health({ awaiting_operator: 2 }).ok, false);
  assert.equal(health({ parked: 3, awaiting_operator: 0 }).ok, true);
});

test('queueHealth: a fresh queue with no measured parked count is amber, not green', () => {
  const health = queueHealth(writeQueue(memoryData(queueStats('fresh', countsPredatingParkedRows()))));
  assert.deepEqual(health, { ok: true, warn: true });
});

test('queueHealth: a value predating parked rows still paints its dead letters red', () => {
  const served = countsPredatingParkedRows({ dead: 1 });
  assert.deepEqual(queueHealth(writeQueue(memoryData(queueStats('fresh', served)))), { ok: false, warn: true });
});

test('queueHealth: a stale queue always warns', () => {
  assert.equal(queueHealth(writeQueue(memoryData(queueStats('stale', counts())))).warn, true);
});

// ── Plan route ──────────────────────────────────────────────────────────────

const CAPPED_ROUTE = Object.freeze({
  state: 'capped',
  proxy_host: 'plan-proxy.internal',
  capped_since: '2026-10-03T11:40:00+00:00',
  resets_at: '2026-10-03T16:00:00+00:00',
  next_probe_at: '2026-10-03T12:10:00+00:00',
  parked_items: 3,
  judge_fail_opens_since_latch: 2,
  escalation_id: 'esc-plan-route-capped-1',
});
const OK_ROUTE = Object.freeze({
  ...CAPPED_ROUTE,
  state: 'ok',
  capped_since: null,
  resets_at: null,
  next_probe_at: null,
  parked_items: 0,
  judge_fail_opens_since_latch: 0,
  escalation_id: null,
});
const ROUTE_REASON = 'get_status reports no plan_route: fused-memory predates task 9944';

// A served plan_route Datum, honouring datum.py's unknown triad.
function routeReading(state, value) {
  if (state === 'unknown') {
    return { value: null, as_of: null, state, reason: ROUTE_REASON, freshness_bound_seconds: 30 };
  }
  return {
    value,
    as_of: MEASURED_AT,
    state,
    reason: state === 'fresh' ? null : 'measured 90s before it was served, past the 30s freshness bound',
    freshness_bound_seconds: 30,
  };
}

function routeData(planRouteBlock, receipts = { [MEMORY_ENDPOINT]: RECEIPT }) {
  return { MEMORY_STATUS: { plan_route: planRouteBlock }, __receipt: receipts };
}

test('planRoute: a fresh plan route is the served Datum, stamped with its receipt', () => {
  const route = planRoute(routeData(routeReading('fresh', CAPPED_ROUTE)));
  assert.ok(isDatum(route));
  assert.equal(route.state, 'fresh');
  assert.deepEqual(route.value, CAPPED_ROUTE);
  assert.equal(route.as_of, MEASURED_AT);
  assert.equal(route._served_at, SERVED_AT);
  assert.equal(route._received_at, RECEIVED_AT);
});

test('planRoute: a served unknown keeps the SERVER\'s reason verbatim', () => {
  const route = planRoute(routeData(routeReading('unknown')));
  assert.equal(route.state, 'unknown');
  assert.equal(route.value, null);
  assert.equal(route.reason, ROUTE_REASON);
});

test('planRoute: before the first /memory receipt the route is not yet fetched', () => {
  const route = planRoute(routeData(routeReading('fresh', CAPPED_ROUTE), {}));
  assert.equal(route.state, 'unknown');
  assert.equal(route.reason, 'not yet fetched');
});

test('planRoute: a delivered payload with no plan_route is a hole naming the /memory payload', () => {
  const route = planRoute({ MEMORY_STATUS: {}, __receipt: { [MEMORY_ENDPOINT]: RECEIPT } });
  assert.equal(route.state, 'unknown');
  assert.match(route.reason, /\/memory payload/);
  assert.match(route.reason, /plan_route/);
});

test('planRouteText: a capped route says since when, when it resets and how much is parked', () => {
  assert.equal(
    planRouteText(CAPPED_ROUTE),
    'plan route: CAPPED since 2026-10-03T11:40:00+00:00, resets 2026-10-03T16:00:00+00:00, 3 parked',
  );
});

test('planRouteText: a capped route with no known reset says so rather than inventing one', () => {
  assert.equal(
    planRouteText({ ...CAPPED_ROUTE, resets_at: null }),
    'plan route: CAPPED since 2026-10-03T11:40:00+00:00, resets unknown, 3 parked',
  );
});

test('planRouteText: a capped route whose parked writes could not be counted says so', () => {
  assert.equal(
    planRouteText({ ...CAPPED_ROUTE, parked_items: null }),
    'plan route: CAPPED since 2026-10-03T11:40:00+00:00, resets 2026-10-03T16:00:00+00:00, unknown parked',
  );
});

test('planRouteText: an ok route and an unconfigured one', () => {
  assert.equal(planRouteText(OK_ROUTE), 'plan route: ok');
  assert.equal(planRouteText({ ...OK_ROUTE, state: 'unconfigured' }), 'plan route: not configured');
});

test('planRouteText never runs over a hole', () => {
  const route = planRoute(routeData(routeReading('unknown')));
  assert.equal(datumView(route, { format: planRouteText }).text, EM_DASH);
});

test('planRouteHealth: a capped route is a measured red fault', () => {
  assert.equal(planRouteHealth(planRoute(routeData(routeReading('fresh', CAPPED_ROUTE)))).ok, false);
});

test('planRouteHealth: a fresh ok route is green, and so is an unconfigured one', () => {
  const health = value => planRouteHealth(planRoute(routeData(routeReading('fresh', value))));
  assert.deepEqual(health(OK_ROUTE), { ok: true, warn: false });
  assert.deepEqual(health({ ...OK_ROUTE, state: 'unconfigured' }), { ok: true, warn: false });
});

test('planRouteHealth: an unread latch is amber, never green-ok and never red', () => {
  assert.deepEqual(planRouteHealth(planRoute(routeData(routeReading('unknown')))), { ok: true, warn: true });
  assert.deepEqual(planRouteHealth(planRoute({ MEMORY_STATUS: {}, __receipt: {} })), { ok: true, warn: true });
});

test('planRouteHealth: a stale ok route always warns', () => {
  assert.deepEqual(planRouteHealth(planRoute(routeData(routeReading('stale', OK_ROUTE)))), { ok: true, warn: true });
});

// ── Memory ops (PRD sketch #11, at the client) ──────────────────────────────

test('opsCaption: the three window totals the caption states', () => {
  assert.equal(opsCaption(OPS_TOTALS), '10 reads · 5 writes · 2 other');
});

test('opsTotalText: the donut centre is the served window total', () => {
  assert.equal(opsTotalText(opsData()), '17');
});

test('the caption\'s three numbers sum to the donut\'s total', () => {
  const captioned = opsCaption(opsTotals(opsData()).value).match(/\d+/g).map(Number);
  assert.equal(String(captioned.reduce((sum, n) => sum + n, 0)), opsTotalText(opsData()));
});

test('opsTotals: fresh served totals are that Datum, stamped with the receipt', () => {
  const totals = opsTotals(opsData());
  assert.ok(isDatum(totals));
  assert.equal(totals.state, 'fresh');
  assert.deepEqual(totals.value, OPS_TOTALS);
  assert.equal(totals.as_of, MEASURED_AT);
  assert.equal(totals._served_at, SERVED_AT);
  assert.equal(totals._received_at, RECEIVED_AT);
});

test('opsTotals: served unknown totals keep the SERVER\'s reason, and the caption never runs over the hole', () => {
  const data = opsData(memoryOps('unknown'));
  const totals = opsTotals(data);
  assert.equal(totals.state, 'unknown');
  assert.equal(totals.value, null);
  assert.equal(totals.reason, OPS_REASON);
  assert.equal(opsTotalText(data), EM_DASH);
  assert.equal(datumView(opsTotals(data), { format: opsCaption }).text, EM_DASH);
});

test('opsTotals: a delivered payload whose totals is not a Datum is a hole naming the /memory-graphs payload', () => {
  const totals = opsTotals(opsData({ ...MEMORY_OPS, totals: { ...OPS_TOTALS } }));
  assert.equal(totals.state, 'unknown');
  assert.match(totals.reason, /\/memory-graphs payload/);
});

test('newestHourOps: the served newest_hour_total, as a Datum', () => {
  const newest = newestHourOps(opsData(memoryOps('fresh', 9)));
  assert.ok(isDatum(newest));
  assert.equal(newest.state, 'fresh');
  assert.equal(newest.value, 9);
  assert.equal(newest._served_at, SERVED_AT);
});

test('newestHourOps: a served unknown keeps the SERVER\'s reason, the same one the totals carry', () => {
  const data = opsData(memoryOps('unknown'));
  const newest = newestHourOps(data);
  assert.equal(newest.state, 'unknown');
  assert.equal(newest.reason, OPS_REASON);
  assert.equal(newest.reason, opsTotals(data).reason);
});

test('before the /memory-graphs receipt both ops readings are not yet fetched', () => {
  const data = opsData(MEMORY_OPS, {});
  assert.equal(opsTotalText(data), EM_DASH);
  for (const reading of [opsTotals(data), newestHourOps(data)]) {
    assert.equal(reading.state, 'unknown');
    assert.equal(reading.reason, 'not yet fetched');
  }
});
