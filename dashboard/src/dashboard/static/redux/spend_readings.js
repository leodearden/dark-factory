// spend_readings.js — the ONE client reader of today's spend, read by the
// topbar pill and the Overview's "Spend (today)" tile alike so they cannot
// disagree. /costs serves COSTS.summary.today bare (data.js registers it
// plain), so the reading carries the /costs receipt: before /costs delivers,
// it is a hole, never data.js's seeded $0.00.
//
// A PLAIN-JS CLASSIC SCRIPT, NOT A .jsx MODULE, so its decisions are
// EXECUTABLE: dashboard/tests/js/spend_readings.test.mjs runs them.
// pins_recovery.js's header holds the CANONICAL statement of why.
//
// LOAD CONTRACT. It destructures window.DF_DATUM at module scope with no
// fallback, so index.html loads it after datum.js and before every JSX consumer
// (test_index_html.py pins the order). Receipts come from `data.__receipt`
// only: no function here reads a browser global.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const { plainDatum: plainSpendDatum } = window.DF_DATUM;

const SPEND_COSTS_ENDPOINT = '/api/v2/dashboard/costs';

function todaySpend(data) {
  const summary = (data.COSTS || {}).summary || {};
  return plainSpendDatum(summary.today, SPEND_COSTS_ENDPOINT, data.__receipt || {});
}

function spendText(dollars) {
  return '$' + dollars.toFixed(2);
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const SPEND_READINGS_API = {
  todaySpend,
  spendText,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = SPEND_READINGS_API;
}
if (typeof window !== 'undefined') {
  window.DF_SPEND_READINGS = SPEND_READINGS_API;
}
