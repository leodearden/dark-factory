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

// Status-priority order for orderRows' initial per-tier permutation (module
// constant, copied from tab_tasks.jsx:137's current within-tier sort).
const STATUS_ORDER = {
  blocked: 0,
  'in-progress': 1,
  'merge-deferred': 1.5,
  pending: 2,
  deferred: 3,
  done: 4,
  cancelled: 5,
};

function statusRank(status) {
  return Object.prototype.hasOwnProperty.call(STATUS_ORDER, status) ? STATUS_ORDER[status] : 9;
}

// Normalizes a position within a row to [0,1] so barycenters computed
// against rows of different lengths are comparable. A row of length <= 1
// has no meaningful spread, so its sole occupant normalizes to the midpoint.
function normalizedPosition(pos, rowLength) {
  return rowLength <= 1 ? 0.5 : pos / (rowLength - 1);
}

// Derives the internal {from, to} edge list from componentTasks' in-set deps
// (from = the upstream/parent id, to = the downstream/child id) — the same
// shape countCrossings consumes.
function deriveEdges(componentTasks) {
  const ids = new Set(componentTasks.map(t => t.id));
  const edges = [];
  for (const t of componentTasks) {
    for (const d of t.deps || []) {
      if (ids.has(d.id) && d.id !== t.id) {
        edges.push({ from: d.id, to: t.id });
      }
    }
  }
  return edges;
}

function cloneRows(rows) {
  return rows.map(row => row.slice());
}

// Runs a single barycenter sweep over `rows` in place. direction 'down'
// visits tiers 1..N-1 top-to-bottom, keying each node on the mean normalized
// position of its parents in the (already-updated-this-sweep) row above;
// direction 'up' visits tiers N-2..0 bottom-to-top, keying on children in
// the row below. A node with no neighbor in the reference row keeps its
// current position (its key is its own current normalized position, so a
// row where nothing has a reference-row neighbor sorts back to itself).
function barycenterSweep(rows, edges, direction) {
  const parentsByChild = new Map();
  const childrenByParent = new Map();
  for (const e of edges) {
    if (!parentsByChild.has(e.to)) parentsByChild.set(e.to, []);
    parentsByChild.get(e.to).push(e.from);
    if (!childrenByParent.has(e.from)) childrenByParent.set(e.from, []);
    childrenByParent.get(e.from).push(e.to);
  }
  const neighborsOf = direction === 'down' ? parentsByChild : childrenByParent;

  const tierIndices = [];
  if (direction === 'down') {
    for (let t = 1; t < rows.length; t++) tierIndices.push(t);
  } else {
    for (let t = rows.length - 2; t >= 0; t--) tierIndices.push(t);
  }

  for (const t of tierIndices) {
    const refRow = direction === 'down' ? rows[t - 1] : rows[t + 1];
    const refPos = new Map(refRow.map((node, i) => [node.id, i]));
    const refLength = refRow.length;

    const row = rows[t];
    const keyed = row.map((node, currentPos) => {
      const neighborIds = (neighborsOf.get(node.id) || []).filter(id => refPos.has(id));
      const key = neighborIds.length > 0
        ? neighborIds.reduce((sum, id) => sum + normalizedPosition(refPos.get(id), refLength), 0) / neighborIds.length
        : normalizedPosition(currentPos, row.length);
      return { node, key };
    });
    keyed.sort((a, b) => a.key - b.key); // stable: equal keys preserve current order
    rows[t] = keyed.map(k => k.node);
  }
}

// Greedily swaps adjacent same-tier pairs whenever doing so strictly reduces
// total crossings, repeating full passes until none improve. Mutates `rows`
// in place; never accepts a swap that doesn't strictly help, so it can only
// hold or reduce the crossing count it started with.
//
// Tracks the current crossing count in `currentCrossings` instead of
// recomputing the pre-swap count on every candidate: the "before" value for
// any candidate is always exactly the last-computed count (either the
// `after` of the previous accepted swap, or the unchanged count from the
// previous rejected/reverted swap), so recomputing it via a second
// countCrossings(rows, edges) call was redundant — this halves the number of
// (O(edges^2)-per-tier-pair) countCrossings calls in this pass without
// changing which swaps are accepted.
function transposePass(rows, edges) {
  let currentCrossings = countCrossings(rows, edges);
  let improved = true;
  while (improved) {
    improved = false;
    for (const row of rows) {
      for (let i = 0; i < row.length - 1; i++) {
        [row[i], row[i + 1]] = [row[i + 1], row[i]];
        const after = countCrossings(rows, edges);
        if (after < currentCrossings) {
          currentCrossings = after;
          improved = true;
        } else {
          [row[i], row[i + 1]] = [row[i + 1], row[i]]; // revert — no strict improvement
        }
      }
    }
  }
}

// ── Order each tier's rows to minimize edge crossings (barycenter + transpose) ──
// 1. Bucket componentTasks by tier, stable-sorting each bucket by
//    STATUS_ORDER (input order as tiebreak) — this is candidate 0.
// 2. Run 4 fixed alternating barycenter sweeps (down, up, down, up) over a
//    working copy, keeping the best (fewest-crossings) arrangement seen,
//    candidate 0 included — so the result can never be worse than the
//    status-sort baseline.
// 3. Run a greedy adjacent-transpose pass on the best arrangement, which can
//    only further reduce (never increase) crossings.
// No randomness anywhere, and every sort is stable, so the result is
// deterministic across calls on identical input.
function orderRows(componentTasks, tiers) {
  const maxTier = componentTasks.reduce((max, t) => Math.max(max, tiers.get(t.id) || 0), 0);
  const initial = Array.from({ length: componentTasks.length > 0 ? maxTier + 1 : 0 }, () => []);
  componentTasks.forEach((t, index) => {
    initial[tiers.get(t.id) || 0].push({ t, index });
  });
  const rows = initial.map(row =>
    row
      .slice()
      .sort((a, b) => {
        const rankDiff = statusRank(a.t.status) - statusRank(b.t.status);
        return rankDiff !== 0 ? rankDiff : a.index - b.index; // stable input-order tiebreak
      })
      .map(entry => entry.t),
  );

  const edges = deriveEdges(componentTasks);

  let best = cloneRows(rows);
  let bestCrossings = countCrossings(best, edges);

  const current = cloneRows(rows);
  for (const direction of ['down', 'up', 'down', 'up']) {
    barycenterSweep(current, edges, direction);
    const crossings = countCrossings(current, edges);
    if (crossings < bestCrossings) {
      best = cloneRows(current);
      bestCrossings = crossings;
    }
  }

  transposePass(best, edges);

  return best;
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
