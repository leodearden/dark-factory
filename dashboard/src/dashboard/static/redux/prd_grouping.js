// prd_grouping.js — pure PRD-grouping-view logic for the Tasks tab's
// per-project "group by PRD" view (tab_tasks.jsx).
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/prd_grouping.js">`
//     tag (like graph_layout.js), which assigns `window.DF_PRD_GROUPING`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// index.html loads this file (classic script, before the Babel JSX tags) so
// `window.DF_PRD_GROUPING` is defined before tab_tasks.jsx executes its
// top-level destructure of it.
//
// orderPrdGroups takes graph_layout.js's computeTiers as an INJECTED
// parameter rather than importing/reading window.DF_GRAPH_LAYOUT internally
// — see the plan's design decisions for why (keeps this module independently
// unit-testable and avoids a hard cross-module load-order/require coupling).

// ── Derive a PRD box title from its path/ref ──
// basename (the substring after the last '/'), with a trailing '-prd.md'
// stripped, else a trailing '.md' stripped, else returned as-is.
function prdTitle(prdPath) {
  if (!prdPath) return prdPath;
  const base = prdPath.includes('/') ? prdPath.slice(prdPath.lastIndexOf('/') + 1) : prdPath;
  if (base.endsWith('-prd.md')) return base.slice(0, base.length - '-prd.md'.length);
  if (base.endsWith('.md')) return base.slice(0, base.length - '.md'.length);
  return base;
}

// ── Aggregate a single status for a PRD box from its member tasks ──
// Precedence: any-blocked > any-(in-progress|merge-deferred) >
// any-(pending|deferred) > all-done > cancelled (the catch-all fallback,
// covering an all-cancelled group or a done+cancelled mix that isn't ALL
// done).
function aggregatePrdStatus(tasks) {
  if (tasks.some(t => t.status === 'blocked')) return 'blocked';
  if (tasks.some(t => t.status === 'in-progress' || t.status === 'merge-deferred')) return 'in-progress';
  if (tasks.some(t => t.status === 'pending' || t.status === 'deferred')) return 'pending';
  if (tasks.every(t => t.status === 'done')) return 'done';
  return 'cancelled';
}

// ── Per-bucket member counts for a PRD box ──
// Single pass over `tasks` bucketing by status: done, inProgress
// ('in-progress' or 'merge-deferred'), blocked, pending ('pending' or
// 'deferred'), total (every member, including cancelled/anything-else,
// which are counted in total but no other bucket). Source of both the
// "n/m done" count (done/total) and the stacked-bar segment sizes.
function summarizePrdMembers(tasks) {
  const counts = { done: 0, inProgress: 0, blocked: 0, pending: 0, total: tasks.length };
  for (const t of tasks) {
    if (t.status === 'done') counts.done++;
    else if (t.status === 'in-progress' || t.status === 'merge-deferred') counts.inProgress++;
    else if (t.status === 'blocked') counts.blocked++;
    else if (t.status === 'pending' || t.status === 'deferred') counts.pending++;
    // cancelled (or any other/unrecognized status) is counted in total only.
  }
  return counts;
}

// ── Bucket tasks by their `prd` field into ordered {prd, tasks, noPrd} groups ──
// Non-null prds are bucketed in first-seen input order, each group's tasks
// preserving input order (a Map preserves insertion order for its keys).
// All prd===null (or missing) tasks are collected separately and, if any
// exist, appended as a single trailing group flagged `noPrd: true` — always
// last, regardless of where in the input those tasks appeared.
function groupTasksByPrd(tasks) {
  const order = [];
  const byPrd = new Map();
  const noPrdTasks = [];

  for (const t of tasks) {
    const prd = t.prd != null ? t.prd : null;
    if (prd === null) {
      noPrdTasks.push(t);
      continue;
    }
    if (!byPrd.has(prd)) {
      byPrd.set(prd, []);
      order.push(prd);
    }
    byPrd.get(prd).push(t);
  }

  const groups = order.map(prd => ({ prd, tasks: byPrd.get(prd), noPrd: false }));
  if (noPrdTasks.length > 0) {
    groups.push({ prd: null, tasks: noPrdTasks, noPrd: true });
  }
  return groups;
}

const API = { prdTitle, aggregatePrdStatus, summarizePrdMembers, groupTasksByPrd };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = API;
}
if (typeof window !== 'undefined') {
  window.DF_PRD_GROUPING = API;
}
