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
//
// NOT A CONTRADICTION OF prd_grouping.js. That module deliberately folds
// 'merge-deferred' into its `inProgress` bucket (and 'deferred' into
// `pending`) in aggregatePrdStatus/summarizePrdMembers, which feed the PRD
// box's single status badge and its stacked-bar segments in the same tab.
// The two rules answer different questions and both are correct as written:
//   - Here, the question is "how does this number compare to the
//     max_concurrent_tasks cap?", which only `running` can answer, so the
//     components MUST stay separate.
//   - There, the question is "is this PRD finished?", for which
//     merge-deferred work is legitimately still-in-flight (it is not done
//     until it lands), and a bar with five thin segments would be unreadable.
// Change one and you do NOT have to change the other. If a future task does
// reconcile them, reconcile the QUESTIONS first, not just the buckets.

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

// ── The ordered activity pips to render in a per-project header ──
// Turns a projectStatusCounts result into `[{key, label, count}]` in the
// fixed running → blocked → merge-deferred order, so no caller re-derives
// the ordering and none can drift out of agreement with another.
//
// Zero-valued components are suppressed: most projects have no
// merge-deferred tasks, and a permanent "0 merge-deferred" pip would be
// noise in every one of their headers.
//
// LABELS ARE THE LITERAL TASK STATUS STRINGS, deliberately — including
// 'in-progress' for the key `running`. Three competing words for this one
// population already exist in the dashboard: the task status is
// 'in-progress', the filter toggle directly above this header says 'active'
// (and keeps its merged three-status meaning on purpose), and 'running' is
// already spoken for by ORCHESTRATOR PROCESS liveness in the Orchestrators
// tab (`status-dot running`). Since the entire point of this module is
// making one number unambiguously comparable to max_concurrent_tasks, the
// pip is labelled with the one word that can only mean the task status.
// The object KEY stays `running` — it reads better in code and is what
// tab_tasks.jsx's PIP_DOT_COLOR_T is keyed by.
//
// Pure — no `window`, and no colour/DOM concerns (the caller owns the dot
// colours).
function activityPips(counts) {
  const c = counts || {};
  const all = [
    { key: 'running', label: 'in-progress', count: c.running || 0 },
    { key: 'blocked', label: 'blocked', count: c.blocked || 0 },
    { key: 'merge-deferred', label: 'merge-deferred', count: c.mergeDeferred || 0 },
  ];
  const shown = all.filter(p => p.count > 0);
  // Do NOT "simplify" this into an unconditional filter. When all three are
  // zero the suppression rule would empty the header's activity region
  // entirely, and an empty region is indistinguishable from a failed data
  // load — the same ambiguity this module exists to remove, just inverted.
  // A quiet project shows an unambiguous "0 in-progress" instead.
  return shown.length > 0 ? shown : [all[0]];
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced by
// dashboard/tests/js/classic_script_scope.test.mjs. A collision here would
// leave window.DF_TASK_STATUS_COUNTS undefined and break tab_tasks.jsx's
// top-level destructure of it.
const TASK_STATUS_COUNTS_API = { projectStatusCounts, activityPips };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TASK_STATUS_COUNTS_API;
}
if (typeof window !== 'undefined') {
  window.DF_TASK_STATUS_COUNTS = TASK_STATUS_COUNTS_API;
}
