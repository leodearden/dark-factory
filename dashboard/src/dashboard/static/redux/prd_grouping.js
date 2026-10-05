// prd_grouping.js — pure PRD-grouping-view logic for the Tasks tab's
// per-project "group by PRD" view (tab_tasks.jsx).
//
// A PLAIN-JS CLASSIC SCRIPT, NOT A .jsx MODULE, so its decisions are
// EXECUTABLE: dashboard/tests/js/prd_grouping.test.mjs runs them.
// pins_recovery.js's header holds the CANONICAL statement of why.
//
// LOAD CONTRACT. It destructures window.DF_DATUM, window.DF_TASK_VOCAB and
// window.DF_TASK_SNAPSHOT at module scope with no fallback, so index.html loads
// it after datum.js, task_vocab.js and task_snapshot.js and before
// tab_tasks.jsx (test_index_html.py pins the order). A browser classic
// `<script>` assigns `window.DF_PRD_GROUPING`; node requires the same file as
// CommonJS once its test has put those globals on a window shim.
//
// ONE TALLY, OVER THE GENERATED VOCABULARY. summarizePrdMembers counts a PRD's
// members in the served census's shape, and every box decision reads that
// tally, so no status string is bucketed here (PRD decisions 3 and 8,
// plans/dashboard-one-datum-one-path-prd.md).
//
// orderPrdGroups takes graph_layout.js's computeTiers as an INJECTED
// parameter rather than reading window.DF_GRAPH_LAYOUT, which keeps this
// module unit-testable without that module on the shim.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const { combinedDatum: combinePrdParts } = window.DF_DATUM;
const { MEMBERS: PRD_MEMBERS, VIEWS: PRD_VIEWS, SUB_VIEWS: PRD_SUB_VIEWS, TONES: PRD_TONES } = window.DF_TASK_VOCAB;
const { terminalOfTotal: prdTerminalOfTotal } = window.DF_TASK_SNAPSHOT;

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

// ── A PRD's members, tallied like a served census ──
// {counts, total, views, sub_views}, the shape data/census.py::TaskCensus
// emits, so task_snapshot.js's census readings apply to it unchanged. A status
// outside the vocabulary counts in `total` only, so it can never pass as a
// member of some view.
function summarizePrdMembers(tasks) {
  const counts = Object.fromEntries(PRD_MEMBERS.map(member => [member, 0]));
  for (const t of tasks) {
    if (Object.hasOwn(counts, t.status)) counts[t.status] += 1;
  }
  return {
    counts,
    total: tasks.length,
    views: prdViewTally(PRD_VIEWS, counts),
    sub_views: prdViewTally(PRD_SUB_VIEWS, counts),
  };
}

function prdViewTally(views, counts) {
  return Object.fromEntries(
    Object.entries(views).map(([view, members]) => [view, members.reduce((sum, member) => sum + counts[member], 0)]),
  );
}

// ── The box's status class ──
// any blocked > any in-flight > any backlog > all done > cancelled. The
// outputs are the `.prd-box.s-*` classes styles.css defines.
function aggregatePrdStatus(summary) {
  if (summary.counts.blocked > 0) return 'blocked';
  if (summary.views.in_flight > 0) return 'in-progress';
  if (summary.views.backlog > 0) return 'pending';
  if (summary.counts.done === summary.total) return 'done';
  return 'cancelled';
}

// PRD decision 3: a PRD is finished when every member is terminal.
function prdIsFinished(summary) {
  return summary.total > 0 && summary.views.terminal === summary.total;
}

// One segment per member present, in vocabulary order, as a share of the
// total. An unrecognised status has no segment, so its share leaves the track
// showing.
function prdBarSegments(summary) {
  return PRD_MEMBERS
    .filter(member => summary.counts[member] > 0)
    .map(member => ({ member, tone: PRD_TONES[member], share: (summary.counts[member] / summary.total) * 100 }));
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

// ── Order PRD groups for box layout ──
// Splits off the "no PRD" group (if any) before tiering, builds a
// taskId->prd map across the remaining (non-null) groups, and synthesizes
// one mini-DAG node per PRD whose deps are the OTHER prds any of its tasks
// consume (a dep whose task isn't in any group — filtered out, or itself
// null-prd — has no resolvable prd and is simply ignored, mirroring
// graph_layout.js's "dep outside the known set" convention). The injected
// computeTiers tiers that mini-DAG; groups are then sorted by
// (tier asc, in-flight desc, backlog desc, stable insertion index),
// and the "no PRD" group (if present) is force-appended last regardless of
// its own tasks' tier/activity.
function orderPrdGroups(groups, computeTiers) {
  const nonNullGroups = groups.filter(g => !g.noPrd);
  const noPrdGroup = groups.find(g => g.noPrd);

  const prdByTaskId = new Map();
  for (const g of nonNullGroups) {
    for (const t of g.tasks) prdByTaskId.set(t.id, g.prd);
  }

  const syntheticNodes = nonNullGroups.map(g => {
    const upstream = new Set();
    for (const t of g.tasks) {
      for (const d of (t.deps || [])) {
        const depPrd = prdByTaskId.get(d.id);
        if (depPrd != null && depPrd !== g.prd) upstream.add(depPrd);
      }
    }
    return { id: g.prd, deps: Array.from(upstream, id => ({ id })) };
  });
  const tiers = computeTiers(syntheticNodes);

  const ordered = nonNullGroups
    .map((g, index) => ({ g, index, tier: tiers.get(g.prd) || 0, views: summarizePrdMembers(g.tasks).views }))
    .sort((a, b) => {
      if (a.tier !== b.tier) return a.tier - b.tier;
      if (a.views.in_flight !== b.views.in_flight) return b.views.in_flight - a.views.in_flight;
      if (a.views.backlog !== b.views.backlog) return b.views.backlog - a.views.backlog;
      return a.index - b.index; // stable insertion-order final tiebreak
    })
    .map(entry => entry.g);

  return noPrdGroup ? [...ordered, noPrdGroup] : ordered;
}

// ── The PRD box count (PRD decision 8) ──
// A Datum whose value maps each PRD (null for the no-PRD bucket) to the tally
// over its rows from BOTH parts: the snapshot's in-flight and backlog rows and
// the on-demand terminal window. combinedDatum lets the worst part decide, so
// the window's lower_bound renders as '≥', and a hole in either part is a hole
// in the count rather than an under-count passed off as a total.
function prdProgress(rowsDatum, terminalDatum) {
  return combinePrdParts(
    [['in-flight and backlog rows', rowsDatum], ['terminal rows', terminalDatum]],
    ([active, terminal]) => new Map(
      groupTasksByPrd([...active, ...terminal]).map(g => [g.prd, summarizePrdMembers(g.tasks)]),
    ),
    'no rows in scope',
  );
}

// The datumView `format` for one PRD's box: its 'n/m' terminal-of-total.
function prdProgressReading(prd) {
  return byPrd => prdTerminalOfTotal(byPrd.get(prd) || summarizePrdMembers([]));
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced by
// dashboard/tests/js/classic_script_scope.test.mjs. A collision here would
// leave window.DF_PRD_GROUPING undefined and break tab_tasks.jsx's top-level
// destructure of it.
const PRD_GROUPING_API = {
  prdTitle,
  aggregatePrdStatus,
  summarizePrdMembers,
  prdIsFinished,
  prdBarSegments,
  prdProgress,
  prdProgressReading,
  groupTasksByPrd,
  orderPrdGroups,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PRD_GROUPING_API;
}
if (typeof window !== 'undefined') {
  window.DF_PRD_GROUPING = PRD_GROUPING_API;
}
