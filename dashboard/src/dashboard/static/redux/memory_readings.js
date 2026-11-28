// memory_readings.js — the ONE client reader of the two memory payloads, read
// by MemoryTab, the Overview and the topbar alike so they cannot disagree:
//   - the write queue, served by /memory as one Datum
//     (dashboard/src/dashboard/data/memory.py::write_queue_datum);
//   - the plan route's cap latch, served by /memory as one Datum
//     (data/memory.py::plan_route_datum);
//   - the window's memory operations, served by /memory-graphs as MEMORY_OPS
//     (data/write_journal.py::get_memory_ops via redux_api.shape_memory_graphs),
//     whose totals and newest_hour_total are served Datums the server derives,
//     so the client never re-counts them and a dead journal reads as a hole.
//
// A PLAIN-JS CLASSIC SCRIPT, NOT A .jsx MODULE, so its decisions are
// EXECUTABLE: dashboard/tests/js/memory_readings.test.mjs runs them.
// pins_recovery.js's header holds the CANONICAL statement of why.
//
// LOAD CONTRACT. It destructures window.DF_DATUM at module scope with no
// fallback, so index.html loads it after datum.js and before every JSX consumer
// (test_index_html.py pins the order). Receipts come from `data.__receipt`
// only: no function here reads a browser global.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const {
  servedDatum: servedMemoryDatum,
  datumView: viewOfMemoryDatum,
  EM_DASH: MEMORY_EM_DASH,
} = window.DF_DATUM;

const MEMORY_ENDPOINT = '/api/v2/dashboard/memory';
const MEMORY_GRAPHS_ENDPOINT = '/api/v2/dashboard/memory-graphs';

function memoryReceipts(data) {
  return data.__receipt || {};
}

// ── The write queue ──

function writeQueue(data) {
  const queue = ((data.MEMORY_STATUS || {}).queue) || {};
  return servedMemoryDatum(
    queue.stats,
    MEMORY_ENDPOINT,
    'the /memory payload has no queue.stats Datum',
    memoryReceipts(data),
  );
}

// null when the value was served by a dashboard process predating the count.
function servedQueueCount(counts, key) {
  return typeof counts[key] === 'number' ? counts[key] : null;
}

// The SERVER counts the writes waiting on a human (data/memory.py::write_queue_datum),
// so the client holds no copy of which statuses wait. A value predating that
// count measured only its dead letters.
function awaitingOperatorCount(counts) {
  const served = servedQueueCount(counts, 'awaiting_operator');
  return served === null ? counts.dead : served;
}

function queueCountsText(counts) {
  const parked = servedQueueCount(counts, 'parked');
  return (
    counts.pending + ' pending · ' +
    counts.retry + ' retry · ' +
    (parked === null ? MEMORY_EM_DASH : parked) + ' parked · ' +
    counts.dead + ' dead'
  );
}

function queueHint(queue) {
  if (viewOfMemoryDatum(queue).isHole) return null;
  const parked = servedQueueCount(queue.value, 'parked');
  const oldest = queue.value.oldest_pending_age_seconds;
  const age = oldest === null || oldest === undefined ? 'idle' : oldest + 's oldest';
  return parked > 0 ? parked + ' parked · ' + age : age;
}

// Red is a MEASURED fault: a write waiting on a human. Amber is anything short
// of a fresh, clean reading — an unmeasured queue or parked count included,
// which is never painted green.
function queueHealth(queue) {
  if (viewOfMemoryDatum(queue).isHole) return { ok: true, warn: true };
  const { pending, retry } = queue.value;
  return {
    ok: awaitingOperatorCount(queue.value) === 0,
    warn: (
      queue.state !== 'fresh' || servedQueueCount(queue.value, 'parked') === null ||
      pending > 5 || retry > 0
    ),
  };
}

// ── The plan route's cap latch ──

function planRoute(data) {
  return servedMemoryDatum(
    (data.MEMORY_STATUS || {}).plan_route,
    MEMORY_ENDPOINT,
    'the /memory payload has no plan_route Datum',
    memoryReceipts(data),
  );
}

function orUnknown(served) {
  return served === null || served === undefined ? 'unknown' : served;
}

function planRouteText(route) {
  if (route.state === 'ok') return 'plan route: ok';
  if (route.state === 'unconfigured') return 'plan route: not configured';
  if (route.state !== 'capped') return 'plan route: ' + route.state;
  return (
    'plan route: CAPPED since ' + orUnknown(route.capped_since) +
    ', resets ' + orUnknown(route.resets_at) +
    ', ' + orUnknown(route.parked_items) + ' parked'
  );
}

// queueHealth's rule: red is a MEASURED cap, amber anything short of a fresh
// reading, an unread latch included.
function planRouteHealth(route) {
  if (viewOfMemoryDatum(route).isHole) return { ok: true, warn: true };
  return { ok: route.value.state !== 'capped', warn: route.state !== 'fresh' };
}

// ── The window's memory operations ──

function formatOpsCount(n) {
  return n.toLocaleString();
}

function opsTotals(data) {
  return servedMemoryDatum(
    (data.MEMORY_OPS || {}).totals,
    MEMORY_GRAPHS_ENDPOINT,
    'the /memory-graphs payload has no totals Datum',
    memoryReceipts(data),
  );
}

function opsCaption(totals) {
  return (
    formatOpsCount(totals.reads) + ' reads · ' +
    formatOpsCount(totals.writes) + ' writes · ' +
    formatOpsCount(totals.other) + ' other'
  );
}

function opsTotalText(data) {
  return viewOfMemoryDatum(opsTotals(data), { format: totals => formatOpsCount(totals.total) }).text;
}

function newestHourOps(data) {
  return servedMemoryDatum(
    (data.MEMORY_OPS || {}).newest_hour_total,
    MEMORY_GRAPHS_ENDPOINT,
    'the /memory-graphs payload has no newest_hour_total Datum',
    memoryReceipts(data),
  );
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const MEMORY_READINGS_API = {
  writeQueue,
  queueCountsText,
  queueHint,
  queueHealth,
  planRoute,
  planRouteText,
  planRouteHealth,
  opsTotals,
  opsCaption,
  opsTotalText,
  newestHourOps,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = MEMORY_READINGS_API;
}
if (typeof window !== 'undefined') {
  window.DF_MEMORY_READINGS = MEMORY_READINGS_API;
}
