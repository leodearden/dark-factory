// orch_summary.js — INTERIM crash guard for the per-orchestrator task counts
// that /api/v2/dashboard/orchestrators no longer puts on the wire.
//
// WHY THIS EXISTS, AND WHEN IT GOES AWAY (task 5589 / leaf γ2, esc-5587-4).
// Task 5587 made `discover_orchestrators` process-discovery only, so
// `redux_api.shape_orchestrators` stopped projecting `summary` — deliberately,
// because a fabricated all-zero summary reads as a measured "this orchestrator
// has no tasks". Twenty-three JSX sites still dereferenced `o.summary.<key>`,
// including one in the ROOT App render body (app.jsx's topbar), so the first
// refresh carrying a real ORCHESTRATORS entry threw a TypeError and took the
// whole SPA down rather than one tab.
//
// The seeded `ORCHESTRATORS: []` in data.js does NOT cover this: `applyKey`
// short-circuits on an ABSENT TOP-LEVEL key, and `summary` sits one level
// BELOW `ORCHESTRATORS`, which is present. The seed is why removing a
// top-level key is safe and why removing a nested one is not.
//
// γ2 moves these surfaces onto TASKS_SNAPSHOT's census, at which point this
// module and its consumers' destructures are deleted outright. It lives in ONE
// place so that deletion is one move with a compiler-visible call-site list,
// rather than twenty-three inline `|| 0`s of which half get missed.
//
// HONEST ABOUT WHAT IT IS: the zero shape below is a CRASH GUARD, not the
// PRD's degradation story. `dashboard-one-datum-one-path-prd.md` decision 2
// says a number that was never measured must render `—` with a reason, never a
// confident `0` — a confident zero is the "0/1 vs Active 33" class of bug the
// PRD exists to remove. So the zero reaches as few sites as possible:
//   · sites whose whole rendered content is the count pair take `—` via
//     `hasOrchSummary`;
//   · the Datum tiles and pips take `orchSummaryTotal`, whose null
//     derivedDatum turns into an unknown Datum carrying
//     ORCH_SUMMARY_ABSENT_REASON, drawn as `—` with that reason as its title;
//   · only the plain-number sites still take the zero: app.jsx's topbar, the
//     Overview pipeline block, and the Progress bar and legend. They cannot
//     show a hole without reworking displays γ2 is about to rewrite.
//
// Plain-JS module, no JSX/Babel — loaded in the browser by a classic
// `<script>` tag in index.html (assigning `window.DF_ORCH_SUMMARY`) and in node
// as CommonJS by the `node --test` suite under dashboard/tests/js/. Both
// export paths are guarded so the file is inert outside the environment it is
// actually running in.

// The count keys the SPA reads off an orchestrator entry. Fixed here rather
// than derived from whatever the wire happens to carry: the guard's whole job
// is to answer for a key that is NOT there, so the roster of keys cannot come
// from the object being guarded.
const ORCH_SUMMARY_KEYS = ['total', 'done', 'in_progress', 'blocked', 'pending'];

// Whether this entry carries a MEASURED summary. The two states — measured and
// not-measured — are kept distinct here so a caller that can afford to render
// `—` is not forced to read a zero back and guess which it was.
function hasOrchSummary(o) {
  return !!o && typeof o.summary === 'object' && o.summary !== null;
}

// Every count key, always present, always a finite number. Per-key rather than
// `o.summary || {}`: the consumers sum and divide these, and an absent key
// would propagate `undefined` into `NaN` — a rendered "NaN" is worse than the
// zero this exists to avoid, and a zero-width progress bar is at least inert.
function orchSummary(o) {
  const raw = hasOrchSummary(o) ? o.summary : {};
  const out = {};
  for (const key of ORCH_SUMMARY_KEYS) {
    const value = raw[key];
    out[key] = (typeof value === 'number' && Number.isFinite(value)) ? value : 0;
  }
  return out;
}

// The total of *key* across *orchs*, or null when ANY entry did not measure
// it. A partial sum would be an under-count passed off as a total. An empty
// list totals 0: nothing in it went unmeasured.
function orchSummaryTotal(orchs, key) {
  let total = 0;
  for (const o of orchs) {
    const value = hasOrchSummary(o) ? o.summary[key] : undefined;
    if (typeof value !== 'number' || !Number.isFinite(value)) return null;
    total += value;
  }
  return total;
}

// Why a Datum tile shows `—` for an orchestrator task count. Declared once,
// here, for every consumer.
const ORCH_SUMMARY_ABSENT_REASON =
  'task counts are not measured by /orchestrators; they now live in the ' +
  '/tasks census (TASKS_SNAPSHOT)';

// Module-unique export const, never a bare `API` — classic scripts share ONE
// top-level lexical scope, so a collision would kill this file before its
// trailing assignment and leave window.DF_ORCH_SUMMARY undefined. Enforced by
// dashboard/tests/js/classic_script_scope.test.mjs; see graph_layout.js's
// header for the full note.
const ORCH_SUMMARY_API = {
  ORCH_SUMMARY_KEYS,
  hasOrchSummary,
  orchSummary,
  orchSummaryTotal,
  ORCH_SUMMARY_ABSENT_REASON,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = ORCH_SUMMARY_API;
}
if (typeof window !== 'undefined') {
  window.DF_ORCH_SUMMARY = ORCH_SUMMARY_API;
}
