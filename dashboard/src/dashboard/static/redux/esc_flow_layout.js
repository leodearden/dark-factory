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

const API = { aggregateFlow };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = API;
}
if (typeof window !== 'undefined') {
  window.DF_ESC_FLOW_LAYOUT = API;
}
