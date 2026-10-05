// memory_readings.js — the ONE client reader of the two memory payloads, read
// by MemoryTab, the Overview and the topbar alike so they cannot disagree:
//   - the write queue, served by /memory as one Datum
//     (dashboard/src/dashboard/data/memory.py::write_queue_datum);
//   - the window's memory operations, served by /memory-graphs as MEMORY_OPS
//     (data/write_journal.py::get_memory_ops via redux_api.shape_memory_graphs),
//     whose totals the server derives so the client never re-counts them.
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
  plainDatum: plainMemoryDatum,
  derivedDatum: derivedMemoryDatum,
  datumView: viewOfMemoryDatum,
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

function queueCountsText(counts) {
  return counts.pending + ' pending · ' + counts.retry + ' retry · ' + counts.dead + ' dead';
}

function queueHint(queue) {
  if (viewOfMemoryDatum(queue).isHole) return null;
  const oldest = queue.value.oldest_pending_age_seconds;
  return oldest === null || oldest === undefined ? 'idle' : oldest + 's oldest';
}

// Red is a MEASURED fault (dead letters); amber is anything short of a fresh,
// clean reading — an unmeasured queue included, which is never painted green.
function queueHealth(queue) {
  if (viewOfMemoryDatum(queue).isHole) return { ok: true, warn: true };
  const { pending, retry, dead } = queue.value;
  return {
    ok: dead === 0,
    warn: queue.state !== 'fresh' || pending > 5 || retry > 0,
  };
}

// ── The window's memory operations ──

function formatOpsCount(n) {
  return n.toLocaleString();
}

function opsTotals(data) {
  return plainMemoryDatum(
    (data.MEMORY_OPS || {}).totals,
    MEMORY_GRAPHS_ENDPOINT,
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
  const hourly = (data.MEMORY_OPS || {}).total || [];
  return derivedMemoryDatum(
    hourly.length ? hourly[hourly.length - 1] : null,
    MEMORY_GRAPHS_ENDPOINT,
    'no ops recorded in this window',
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
