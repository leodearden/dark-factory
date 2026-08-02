// task_status_counts.js — pure per-project header counting logic for the
// Tasks tab (tab_tasks.jsx).
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/task_status_counts.js">`
//     tag (like prd_grouping.js), which assigns `window.DF_TASK_STATUS_COUNTS`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// index.html loads this file (classic script, before the Babel JSX tags) so
// `window.DF_TASK_STATUS_COUNTS` is defined before tab_tasks.jsx executes its
// top-level destructure of it.
//
// WHY THIS MODULE EXISTS. The per-project header used to render ONE merged
// "N active" pip over {in-progress, blocked, merge-deferred}. Only the
// `in-progress` component is bounded by max_concurrent_tasks — a blocked or
// merge-deferred task holds no agent slot — so the merged number routinely
// exceeded the configured cap and read as a cap breach. On 2026-07-30 that
// produced a false alarm: dark-factory showed "43 active" against a cap of
// 24, reify "50 active" against 48. Splitting the three counts makes the one
// number an operator compares against the cap actually comparable to it.
//
// Nothing here reads `window` or `document`: the shared-classic-script-scope
// suite loads this file into a bare `vm.createContext({window:{}})`, and a
// module that reached for browser globals would behave differently under test
// than in the browser.

// ── Per-project status tallies for the header ──
// Single pass over `tasks` bucketing by status: running (ONLY 'in-progress'),
// blocked (only 'blocked'), mergeDeferred (only 'merge-deferred'), pending
// (only 'pending'), done (only 'done'), total (every task). Any other status
// — 'cancelled', 'deferred', a status this build doesn't know about, or a
// task carrying no status at all — is counted in total and nothing else.
//
// 'deferred' is deliberately NOT folded into mergeDeferred: the two share a
// suffix but are unrelated states (merge-deferred work is finished and
// waiting on the merge lane; deferred work is parked and not in flight).
//
// A null/undefined `tasks` is tolerated and yields the all-zero result — the
// header renders before task data has necessarily arrived, and throwing here
// would blank the whole Tasks tab.
function projectStatusCounts(tasks) {
  const list = tasks || [];
  const counts = {
    total: list.length,
    running: 0,
    blocked: 0,
    mergeDeferred: 0,
    pending: 0,
    done: 0,
  };
  for (const t of list) {
    if (t.status === 'in-progress') counts.running++;
    else if (t.status === 'blocked') counts.blocked++;
    else if (t.status === 'merge-deferred') counts.mergeDeferred++;
    else if (t.status === 'pending') counts.pending++;
    else if (t.status === 'done') counts.done++;
    // cancelled / deferred / any other or missing status: total only.
  }
  return counts;
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced by
// dashboard/tests/js/classic_script_scope.test.mjs. A collision here would
// leave window.DF_TASK_STATUS_COUNTS undefined and break tab_tasks.jsx's
// top-level destructure of it.
const TASK_STATUS_COUNTS_API = { projectStatusCounts };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TASK_STATUS_COUNTS_API;
}
if (typeof window !== 'undefined') {
  window.DF_TASK_STATUS_COUNTS = TASK_STATUS_COUNTS_API;
}
