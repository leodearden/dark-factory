// graph_layout.js — Sugiyama ordering-phase layout helpers for the Tasks
// tab's dependency graph (TaskGraph in tab_tasks.jsx).
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/graph_layout.js">`
//     tag (like data.js), which assigns `window.DF_GRAPH_LAYOUT`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.

// ── Compute dep tiers for a task list (Kahn's algorithm style; tier = max(deps' tier)+1) ──
// Verbatim copy of tab_tasks.jsx:19-38 (TaskGraph's computeTiers). Kept in
// sync intentionally by duplication — see graph_layout.js module header.
function computeTiers(tasks) {
  const byId = new Map(tasks.map(t => [t.id, t]));
  const tiers = new Map(); // id -> tier
  const visiting = new Set();

  function tierOf(id) {
    if (tiers.has(id)) return tiers.get(id);
    const t = byId.get(id);
    if (!t) return 0;             // dep references a task outside the project list — ignore
    if (visiting.has(id)) return 0; // cycle guard
    visiting.add(id);
    const inProject = (t.deps || []).filter(d => byId.has(d.id));
    const tier = inProject.length === 0 ? 0 : 1 + Math.max(...inProject.map(d => tierOf(d.id)));
    tiers.set(id, tier);
    visiting.delete(id);
    return tier;
  }
  for (const t of tasks) tierOf(t.id);
  return tiers;
}

// ── Partition a task list into weakly-connected components + singletons ──
function partitionComponents(tasks) {
  return { components: [], singletons: [] };
}

// ── Order each tier's rows to minimize edge crossings (barycenter + transpose) ──
function orderRows(componentTasks, tiers) {
  return [];
}

// ── Count edge-crossing inversions between adjacent tiers ──
function countCrossings(rows, edges) {
  return 0;
}

const API = { computeTiers, partitionComponents, orderRows, countCrossings };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = API;
}
if (typeof window !== 'undefined') {
  window.DF_GRAPH_LAYOUT = API;
}
