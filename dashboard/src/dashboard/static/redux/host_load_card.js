// host_load_card.js — the Overview's Host load card reading of /api/load. A
// key it knows reads in its own label and format; any other served key reads
// as its raw key and a plain number, so the card renders every key the server
// serves. The served set's one home is sampler.metrics.METRIC_GROUPS, via
// dashboard/src/dashboard/data/load.py::KNOWN_METRICS.
//
// A PLAIN-JS CLASSIC SCRIPT, NOT A .jsx MODULE, so its decisions are
// EXECUTABLE: dashboard/tests/js/host_load_card.test.mjs runs them.
// pins_recovery.js's header holds the CANONICAL statement of why.
//
// It reads no window global; index.html loads it before tab_overview.jsx.

function hostLoadPercent(v) {
  return `${v.toFixed(2)}%`;
}

function hostLoadGibibytes(v) {
  return `${(v / 1073741824).toFixed(1)} GiB`;
}

function hostLoadWholeNumber(v) {
  return `${Math.round(v)}`;
}

function hostLoadPlainNumber(v) {
  return `${+v.toFixed(2)}`;
}

const HOST_LOAD_PRESENTATION = new Map([
  ['psi_cpu_some_avg10', { label: 'CPU pressure · some', format: hostLoadPercent }],
  ['psi_cpu_full_avg10', { label: 'CPU pressure · full', format: hostLoadPercent }],
  ['psi_mem_some_avg10', { label: 'Mem pressure · some', format: hostLoadPercent }],
  ['psi_mem_full_avg10', { label: 'Mem pressure · full', format: hostLoadPercent }],
  ['psi_io_some_avg10', { label: 'IO pressure · some', format: hostLoadPercent }],
  ['psi_io_full_avg10', { label: 'IO pressure · full', format: hostLoadPercent }],
  ['occt_queue_depth', { label: 'OCCT queue depth', format: hostLoadWholeNumber }],
  ['verify_scopes_live', { label: 'Verify scopes live', format: hostLoadWholeNumber }],
  ['verify_scope_memory_bytes', { label: 'Verify scope memory', format: hostLoadGibibytes }],
]);

function hostLoadRows(body) {
  return Object.entries(body || {}).map(([key, datum]) => {
    const { label, format } = HOST_LOAD_PRESENTATION.get(key) || { label: key, format: hostLoadPlainNumber };
    return {
      key,
      label,
      text: datum.current == null ? '—' : format(datum.current),
      sparkline: datum.sparkline,
    };
  });
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const HOST_LOAD_CARD_API = {
  hostLoadRows,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = HOST_LOAD_CARD_API;
}
if (typeof window !== 'undefined') {
  window.DF_HOST_LOAD_CARD = HOST_LOAD_CARD_API;
}
