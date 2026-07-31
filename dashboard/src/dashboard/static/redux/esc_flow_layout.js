// esc_flow_layout.js — pure-JS aggregation + geometry for the escalation
// lifecycle flow diagram (mini-Sankey) in the escalation-analytics Workflow
// panel.
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways,
// mirroring graph_layout.js (the barycenter/task-graph layout precedent):
//   - In the browser, via a classic `<script src="/static/redux/esc_flow_layout.js">`
//     tag (index.html loads this before the Babel tags), which assigns
//     `window.DF_ESC_FLOW_LAYOUT`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/ (see dashboard/tests/test_graph_layout_js.py for
//     the pytest wrapper that surfaces it in CI — it globs **/*.test.mjs, so
//     esc_flow_layout.test.mjs is auto-discovered with no new wrapper).
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// aggregateFlow(flowDaily, opts) tallies already-windowed flow rows
// ({date, source, level, tier, class, n} — see the `flow_daily` sparse cube
// produced by dashboard/src/dashboard/data/escalation_analytics.py) into a
// 4-column count model (origin -> level -> tier -> class), folding
// low-volume origin sources into a shared 'other' bucket. layoutFlow(model,
// dims) turns that count model into pixel geometry (node rects +
// cubic-bezier ribbon bands) under a single shared px-per-count scale, so
// ribbon widths conserve flow exactly (Σ ribbon w at a node == that node's
// rect height).
//
// esc_flow_diagram.jsx is the sole consumer: a thin React/SVG render shell
// with no aggregation/geometry logic of its own — see that file.

const OTHER_SOURCE = 'other';

// Canonical resolver-tier display order — duplicated (not imported) from
// tab_escalation_analytics.jsx's `_RESOLVER_TIERS`: this module must stay
// Babel/JSX-free and runnable under plain `node --test`, so it cannot import
// from a .jsx file. A tier outside this list still gets its own node (sorted
// alphabetically after the canonical ones) — only the origin column folds
// low-volume entries into a shared 'other' bucket; tiers are never folded.
const TIER_ORDER = ['human', 'cascade', 'auto-watcher', 'steward', 'reaper-sweep', 'unknown', 'other-auto'];
const CLASS_ORDER = ['benign', 'actionable'];

function sortByCanonicalThenAlpha(ids, canonicalOrder) {
  const rank = new Map(canonicalOrder.map((id, i) => [id, i]));
  return ids.slice().sort((a, b) => {
    const ra = rank.has(a) ? rank.get(a) : canonicalOrder.length + 1;
    const rb = rank.has(b) ? rank.get(b) : canonicalOrder.length + 1;
    if (ra !== rb) return ra - rb;
    return a < b ? -1 : a > b ? 1 : 0;
  });
}

function addToLinkMap(map, from, to, n) {
  // NUL-joined key: from/to values (agent roles, levels, tiers, classes)
  // never contain a NUL byte, so this can't collide the way a printable
  // separator (e.g. a plain space) theoretically could.
  const key = `${from}\0${to}`;
  const existing = map.get(key);
  if (existing) existing.count += n;
  else map.set(key, { from, to, count: n });
}

// Tallies `flowDaily` rows ({date, source, level, tier, class, n}) into a
// 4-column count model: {columns: [origin[], level[], tier[], class[]],
// links: [{col, from, to, count}], total}. `date` is never read here — the
// caller (tab_escalation_analytics.jsx's WorkflowPanel) has already sliced
// rows to the active 7d/28d/all window before calling this, so aggregateFlow
// simply sums whatever rows it's given, grouped by (source, level, tier,
// class) only (see the plan's "window-agnostic" design decision).
//
// Origin sources are folded to the top `topNSources` by total count (desc),
// with any remainder combined into a single 'other' node/link stream. The
// fold is applied to `source` BEFORE tallying links, so links leaving a
// folded source are remapped to 'other' too — keeping Σ origin->level links
// == total even after folding (not just Σ node counts).
function aggregateFlow(flowDaily, { topNSources = 6 } = {}) {
  const rows = flowDaily || [];

  const rawOriginCounts = new Map();
  for (const row of rows) {
    rawOriginCounts.set(row.source, (rawOriginCounts.get(row.source) || 0) + (row.n || 0));
  }
  const sourcesByCountDesc = [...rawOriginCounts.keys()].sort((a, b) => {
    const diff = (rawOriginCounts.get(b) || 0) - (rawOriginCounts.get(a) || 0);
    return diff !== 0 ? diff : (a < b ? -1 : a > b ? 1 : 0); // stable tiebreak
  });
  const keepSources = new Set(sourcesByCountDesc.slice(0, Math.max(topNSources, 0)));

  function originKey(source) {
    return keepSources.has(source) ? source : OTHER_SOURCE;
  }

  const originTotals = new Map();
  const levelTotals = new Map();
  const tierTotals = new Map();
  const classTotals = new Map();
  const link01 = new Map(); // origin -> level
  const link12 = new Map(); // level -> tier
  const link23 = new Map(); // tier -> class

  let total = 0;

  for (const row of rows) {
    const n = row.n || 0;
    if (n === 0) continue;
    total += n;

    const origin = originKey(row.source);
    const level = String(row.level);
    const tier = row.tier;
    const cls = row.class;

    originTotals.set(origin, (originTotals.get(origin) || 0) + n);
    levelTotals.set(level, (levelTotals.get(level) || 0) + n);
    tierTotals.set(tier, (tierTotals.get(tier) || 0) + n);
    classTotals.set(cls, (classTotals.get(cls) || 0) + n);

    addToLinkMap(link01, origin, level, n);
    addToLinkMap(link12, level, tier, n);
    addToLinkMap(link23, tier, cls, n);
  }

  // Origin: kept sources ordered by count desc (stable alpha tiebreak),
  // 'other' always last regardless of its folded count.
  const originIds = [...originTotals.keys()].sort((a, b) => {
    if (a === OTHER_SOURCE) return 1;
    if (b === OTHER_SOURCE) return -1;
    const diff = (originTotals.get(b) || 0) - (originTotals.get(a) || 0);
    return diff !== 0 ? diff : (a < b ? -1 : a > b ? 1 : 0);
  });
  const levelIds = [...levelTotals.keys()].sort((a, b) => Number(a) - Number(b));
  const tierIds = sortByCanonicalThenAlpha([...tierTotals.keys()], TIER_ORDER);
  const classIds = sortByCanonicalThenAlpha([...classTotals.keys()], CLASS_ORDER);

  const columns = [
    originIds.map(id => ({ id, label: id, count: originTotals.get(id) })),
    levelIds.map(id => ({ id, label: id, count: levelTotals.get(id) })),
    tierIds.map(id => ({ id, label: id, count: tierTotals.get(id) })),
    classIds.map(id => ({ id, label: id, count: classTotals.get(id) })),
  ];

  const links = [
    ...[...link01.values()].map(l => ({ col: 0, from: l.from, to: l.to, count: l.count })),
    ...[...link12.values()].map(l => ({ col: 1, from: l.from, to: l.to, count: l.count })),
    ...[...link23.values()].map(l => ({ col: 2, from: l.from, to: l.to, count: l.count })),
  ];

  return { columns, links, total };
}

// Turns an aggregateFlow() count model into pixel geometry: {nodes, ribbons,
// width, height, colX}. `dims` = {width, height, nodeWidth, nodePad}. `colX`
// is the same per-column left-edge x array used to place every node in that
// column (nodes[i].x === colX[nodes[i].col]) — exposed so a caller (e.g.
// esc_flow_diagram.jsx's column headers) can align to the exact same
// coordinates nodes/ribbons use, instead of re-deriving its own approximation
// and silently drifting out of sync.
//
// nodes: one {col, id, label, count, x, y, w, h} per column entry, stacked
// top-to-bottom within each column (vertically centered as a group) with
// `nodePad` gaps, each h = count * scale.
//
// ribbons: one {col, from, to, count, w, d} per link, w = count * scale (the
// SAME scale as node heights — the single shared px-per-count scale required
// for exact flow conservation), `d` a filled cubic-bezier band between the
// right edge of the `from` node and the left edge of the `to` node.
//
// The shared scale is the tightest fit across every column that has a
// nonzero total (height available for that column's nodes, divided by its
// total count) — so every column's stack (nodes + inter-node padding) fits
// within `height` under one common scale, which is what makes Σ ribbon w at
// a node equal that node's own h (both derived from the same count * scale).
function layoutFlow(model, { width = 640, height = 260, nodeWidth = 14, nodePad = 4 } = {}) {
  const columns = (model && model.columns) || [];
  const links = (model && model.links) || [];

  // Hoisted above the anyNodes early-return so `colX` is available on EVERY
  // return path (including the degenerate all-zero-count case) — it depends
  // only on width/nodeWidth/column count, never on the actual node counts.
  const numCols = columns.length;
  const colX = columns.map((_, i) => (numCols <= 1 ? 0 : (i * Math.max(width - nodeWidth, 0)) / (numCols - 1)));

  const anyNodes = columns.some(col => col.length > 0);
  if (!anyNodes) {
    return { nodes: [], ribbons: [], width, height, colX };
  }

  let scale = Infinity;
  for (const col of columns) {
    const colTotal = col.reduce((s, n) => s + n.count, 0);
    if (colTotal > 0) {
      const avail = Math.max(height - nodePad * Math.max(col.length - 1, 0), 1);
      scale = Math.min(scale, avail / colTotal);
    }
  }
  if (!Number.isFinite(scale)) scale = 0; // every present node has count 0 — degenerate but shouldn't NaN

  const nodes = [];
  const nodeIndex = new Map(); // `${col}\0${id}` -> node, for ribbon endpoint lookup
  const colOrderIndex = columns.map(col => new Map(col.map((n, i) => [n.id, i])));

  columns.forEach((col, ci) => {
    const heights = col.map(n => n.count * scale);
    const totalH = heights.reduce((s, h) => s + h, 0) + nodePad * Math.max(col.length - 1, 0);
    let cursor = (height - totalH) / 2; // center the column's stack vertically
    col.forEach((n, i) => {
      const h = heights[i];
      const node = { col: ci, id: n.id, label: n.label, count: n.count, x: colX[ci], y: cursor, w: nodeWidth, h };
      nodes.push(node);
      nodeIndex.set(`${ci}\0${n.id}`, node);
      cursor += h + nodePad;
    });
  });

  // Stable stacking order within each column-pair: group by the `from`
  // node's column position, then the `to` node's column position. Any FIXED
  // deterministic order preserves exact per-node conservation — each link's
  // px-width is carved out of a running per-node cursor (below), so the
  // increments for a given node always sum to exactly its own height
  // regardless of visitation order. This particular order is chosen only to
  // group a node's ribbons together for a tidier (not crossing-minimized)
  // stack.
  const sortedLinks = links.slice().sort((a, b) => {
    const fromOrderA = (colOrderIndex[a.col] || new Map()).get(a.from) ?? 0;
    const fromOrderB = (colOrderIndex[b.col] || new Map()).get(b.from) ?? 0;
    if (fromOrderA !== fromOrderB) return fromOrderA - fromOrderB;
    const toOrderA = (colOrderIndex[a.col + 1] || new Map()).get(a.to) ?? 0;
    const toOrderB = (colOrderIndex[b.col + 1] || new Map()).get(b.to) ?? 0;
    return toOrderA - toOrderB;
  });

  const outCursor = new Map(); // `${col}\0${id}` (source side) -> next free y offset
  const inCursor = new Map(); // `${col+1}\0${id}` (target side) -> next free y offset

  const ribbons = sortedLinks.map(link => {
    const fromKey = `${link.col}\0${link.from}`;
    const toKey = `${link.col + 1}\0${link.to}`;
    const fromNode = nodeIndex.get(fromKey);
    const toNode = nodeIndex.get(toKey);
    const w = link.count * scale;

    const yTop0 = (fromNode ? fromNode.y : 0) + (outCursor.get(fromKey) || 0);
    const yTop1 = (toNode ? toNode.y : 0) + (inCursor.get(toKey) || 0);
    outCursor.set(fromKey, (outCursor.get(fromKey) || 0) + w);
    inCursor.set(toKey, (inCursor.get(toKey) || 0) + w);

    const x0 = fromNode ? fromNode.x + fromNode.w : 0;
    const x1 = toNode ? toNode.x : 0;
    const xm = (x0 + x1) / 2;
    const yBot0 = yTop0 + w;
    const yBot1 = yTop1 + w;

    const d = `M${x0},${yTop0} C${xm},${yTop0} ${xm},${yTop1} ${x1},${yTop1} L${x1},${yBot1} C${xm},${yBot1} ${xm},${yBot0} ${x0},${yBot0} Z`;

    return { col: link.col, from: link.from, to: link.to, count: link.count, w, d };
  });

  return { nodes, ribbons, width, height, colX };
}

// Named ESC_FLOW_LAYOUT_API, never a bare `API`: index.html loads this file
// as one of several classic (non-module) <script> tags on the same page, and
// top-level `const` bindings in classic scripts share ONE global lexical
// scope across ALL of them. A name declared twice kills the second script to
// declare it with "Identifier 'X' has already been declared" — thrown at
// declaration-instantiation time, before a single statement of its body runs,
// so the `window.DF_ESC_FLOW_LAYOUT` assignment below never happens and
// esc_flow_diagram.jsx's top-level destructure of it gets `undefined`. That is
// not hypothetical: this file shipped with a bare `const API` that collided
// with graph_layout.js's, silently breaking the Workflow panel's mini-Sankey.
// Guarded now by dashboard/tests/js/classic_script_scope.test.mjs, which loads
// every classic script into one shared node:vm context (the sibling modules'
// notes predate that harness and could only cite each other).
const ESC_FLOW_LAYOUT_API = { aggregateFlow, layoutFlow };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = ESC_FLOW_LAYOUT_API;
}
if (typeof window !== 'undefined') {
  window.DF_ESC_FLOW_LAYOUT = ESC_FLOW_LAYOUT_API;
}
