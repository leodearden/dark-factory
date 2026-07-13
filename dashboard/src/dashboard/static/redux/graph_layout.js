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
// Builds an undirected adjacency graph from the in-list deps edge list (a
// task and each dep whose id is present in the input set are connected in
// both directions, so "in-list deps" covers both parent- and child-side
// edges). The weakly-connected components of that graph become
// `components`; a node with zero in-list edges (either direction) is a
// `singleton`. Every step iterates the input array in order (never a
// Set/Map's native iteration order) so the result is fully deterministic:
// components are emitted in the order their earliest-input-index member is
// first reached, and both component members and singletons preserve their
// original input order.
function partitionComponents(tasks) {
  const ids = tasks.map(t => t.id);
  const byId = new Map(tasks.map(t => [t.id, t]));
  const adjacency = new Map(ids.map(id => [id, new Set()]));

  for (const t of tasks) {
    for (const d of t.deps || []) {
      if (byId.has(d.id) && d.id !== t.id) {
        adjacency.get(t.id).add(d.id);
        adjacency.get(d.id).add(t.id);
      }
    }
  }

  const visited = new Set();
  const components = [];
  const singletons = [];

  for (const id of ids) {
    if (visited.has(id)) continue;

    if (adjacency.get(id).size === 0) {
      visited.add(id);
      singletons.push(byId.get(id));
      continue;
    }

    // BFS out from `id` (the earliest-index unvisited member of this
    // component) to find the rest of the component, then re-derive member
    // order from `ids` rather than BFS discovery order.
    const memberIds = new Set([id]);
    visited.add(id);
    const queue = [id];
    while (queue.length > 0) {
      const current = queue.shift();
      for (const neighbor of adjacency.get(current)) {
        if (!memberIds.has(neighbor)) {
          memberIds.add(neighbor);
          visited.add(neighbor);
          queue.push(neighbor);
        }
      }
    }

    components.push(ids.filter(candidate => memberIds.has(candidate)).map(candidate => byId.get(candidate)));
  }

  return { components, singletons };
}

// ── Order each tier's rows to minimize edge crossings (barycenter + transpose) ──
function orderRows(componentTasks, tiers) {
  return [];
}

// ── Count edge-crossing inversions between adjacent tiers ──
// rows: array of per-tier arrays of task-like objects (keyed by t.id).
// edges: [{from: <upper/parentId>, to: <lower/childId>}, ...].
// Only edges whose endpoints lie in an ADJACENT tier pair are considered —
// multi-tier ("long") edges are not decomposed (v1 has no dummy/virtual
// nodes). Two edges in the same adjacent pair cross when their upper-row
// position delta and lower-row position delta have opposite (nonzero) signs;
// edges sharing an endpoint (a zero delta on either side) never cross.
function countCrossings(rows, edges) {
  const posOf = new Map(); // id -> {tier, pos}
  rows.forEach((row, tier) => {
    row.forEach((node, pos) => posOf.set(node.id, { tier, pos }));
  });

  let total = 0;
  for (let tier = 0; tier < rows.length - 1; tier++) {
    const pairEdges = edges.filter(e => {
      const upper = posOf.get(e.from);
      const lower = posOf.get(e.to);
      return upper && lower && upper.tier === tier && lower.tier === tier + 1;
    });
    for (let i = 0; i < pairEdges.length; i++) {
      const upperA = posOf.get(pairEdges[i].from).pos;
      const lowerA = posOf.get(pairEdges[i].to).pos;
      for (let j = i + 1; j < pairEdges.length; j++) {
        const upperB = posOf.get(pairEdges[j].from).pos;
        const lowerB = posOf.get(pairEdges[j].to).pos;
        const upperSign = Math.sign(upperA - upperB);
        const lowerSign = Math.sign(lowerA - lowerB);
        if (upperSign !== 0 && lowerSign !== 0 && upperSign !== lowerSign) total++;
      }
    }
  }
  return total;
}

const API = { computeTiers, partitionComponents, orderRows, countCrossings };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = API;
}
if (typeof window !== 'undefined') {
  window.DF_GRAPH_LAYOUT = API;
}
