// Module-contract tests for esc_flow_layout.js — a plain-JS (no JSX/Babel)
// aggregation + geometry module for the escalation-analytics Workflow panel's
// lifecycle flow diagram (mini-Sankey). Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper that
// surfaces this suite in CI — it globs **/*.test.mjs under dashboard/tests/js/,
// so this new file needs no new wrapper).
//
// esc_flow_layout.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`), same as graph_layout.js. Node's
// cjs-module-lexer cannot statically detect named exports assigned from a
// variable, so we default-import the module and destructure instead (see
// graph_layout.test.mjs for the same idiom).
import { test } from 'node:test';
import assert from 'node:assert/strict';

import layout from '../../src/dashboard/static/redux/esc_flow_layout.js';

const { aggregateFlow, layoutFlow } = layout;

// Mirrors test_escalation_analytics.py::build_golden_archive's 5 terminal
// escalations' flow_daily cells (dashboard/tests/test_escalation_analytics.py:
// 672-697) — each row's (source, level, tier, class) here matches one of
// that fixture's 5 resolved/dismissed records, n=1 each (that test notes
// "In this fixture every record lands on a distinct date so each cell's
// n == 1"). Marginals verified there and re-derived in the plan:
//   origin  {implementer: 3, architect: 2}
//   level   {0: 2, 1: 1, 2: 2}
//   tier    {reaper-sweep: 1, human: 3, auto-watcher: 1}
//   class   {benign: 2, actionable: 3}
// Dates are arbitrary/inert here — aggregateFlow is window-agnostic and sums
// purely over (source, level, tier, class); the multi-day-sum test below
// exercises that explicitly with two dates for one identical key.
const GOLDEN_FLOW_ROWS = [
  { date: '2026-01-01', source: 'implementer', level: 0, tier: 'reaper-sweep', class: 'benign', n: 1 },
  { date: '2026-01-02', source: 'implementer', level: 0, tier: 'human', class: 'actionable', n: 1 },
  { date: '2026-01-03', source: 'architect', level: 1, tier: 'auto-watcher', class: 'benign', n: 1 },
  { date: '2026-01-04', source: 'architect', level: 2, tier: 'human', class: 'actionable', n: 1 },
  { date: '2026-01-05', source: 'implementer', level: 2, tier: 'human', class: 'actionable', n: 1 },
];

function countsById(nodes) {
  const out = {};
  for (const n of nodes) out[n.id] = n.count;
  return out;
}

function linksByCol(links, col) {
  const out = {};
  for (const l of links.filter(l => l.col === col)) {
    out[`${l.from}|${l.to}`] = l.count;
  }
  return out;
}

// ---------------------------------------------------------------------------
// aggregateFlow — golden-fixture node counts (four columns: origin, level,
// tier, class).
// ---------------------------------------------------------------------------

test('aggregateFlow: golden fixture — returns exactly four columns', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  assert.equal(model.columns.length, 4, 'expected columns = [origin, level, tier, class]');
});

test('aggregateFlow: golden fixture — origin column node counts match verified marginals', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  assert.deepEqual(countsById(model.columns[0]), { implementer: 3, architect: 2 });
});

test('aggregateFlow: golden fixture — level column node counts match verified marginals', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  assert.deepEqual(countsById(model.columns[1]), { '0': 2, '1': 1, '2': 2 });
});

test('aggregateFlow: golden fixture — tier column node counts match verified marginals', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  assert.deepEqual(countsById(model.columns[2]), { 'reaper-sweep': 1, human: 3, 'auto-watcher': 1 });
});

test('aggregateFlow: golden fixture — class column node counts match verified marginals', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  assert.deepEqual(countsById(model.columns[3]), { benign: 2, actionable: 3 });
});

// ---------------------------------------------------------------------------
// aggregateFlow — golden-fixture ribbon (link) counts, one adjacent-column
// pair per `col` (0: origin->level, 1: level->tier, 2: tier->class).
// ---------------------------------------------------------------------------

test('aggregateFlow: golden fixture — origin->level ribbon counts match verified marginals', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  assert.deepEqual(linksByCol(model.links, 0), {
    'implementer|0': 2,
    'architect|1': 1,
    'architect|2': 1,
    'implementer|2': 1,
  });
});

test('aggregateFlow: golden fixture — level->tier ribbon counts match verified marginals', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  assert.deepEqual(linksByCol(model.links, 1), {
    '0|reaper-sweep': 1,
    '0|human': 1,
    '1|auto-watcher': 1,
    '2|human': 2,
  });
});

test('aggregateFlow: golden fixture — tier->class ribbon counts match verified marginals', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  assert.deepEqual(linksByCol(model.links, 2), {
    'reaper-sweep|benign': 1,
    'human|actionable': 3,
    'auto-watcher|benign': 1,
  });
});

test('aggregateFlow: golden fixture — total == 5', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  assert.equal(model.total, 5);
});

// ---------------------------------------------------------------------------
// aggregateFlow — interior-node (level, tier) conservation: incoming ==
// node.count == outgoing. Origin (source-only) and class (sink-only) are the
// two terminal columns and have no "both sides" to check.
// ---------------------------------------------------------------------------

test('aggregateFlow: interior-node conservation — level nodes balance incoming vs outgoing vs own count', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  for (const node of model.columns[1]) {
    const incoming = model.links.filter(l => l.col === 0 && l.to === node.id).reduce((s, l) => s + l.count, 0);
    const outgoing = model.links.filter(l => l.col === 1 && l.from === node.id).reduce((s, l) => s + l.count, 0);
    assert.equal(incoming, node.count, `level ${node.id} incoming should equal its own count`);
    assert.equal(outgoing, node.count, `level ${node.id} outgoing should equal its own count`);
  }
});

test('aggregateFlow: interior-node conservation — tier nodes balance incoming vs outgoing vs own count', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  for (const node of model.columns[2]) {
    const incoming = model.links.filter(l => l.col === 1 && l.to === node.id).reduce((s, l) => s + l.count, 0);
    const outgoing = model.links.filter(l => l.col === 2 && l.from === node.id).reduce((s, l) => s + l.count, 0);
    assert.equal(incoming, node.count, `tier ${node.id} incoming should equal its own count`);
    assert.equal(outgoing, node.count, `tier ${node.id} outgoing should equal its own count`);
  }
});

// ---------------------------------------------------------------------------
// aggregateFlow — multi-day sum: rows sharing (source,level,tier,class) but
// differing only in `date` collapse into a single summed link/node (proves
// aggregateFlow sums whatever window of rows it's handed, per the
// window-agnostic design decision — the caller already sliced the window).
// ---------------------------------------------------------------------------

test('aggregateFlow: rows on different dates with identical (source,level,tier,class) collapse into one summed link', () => {
  const rows = [
    { date: '2026-02-01', source: 'implementer', level: 0, tier: 'human', class: 'actionable', n: 1 },
    { date: '2026-02-02', source: 'implementer', level: 0, tier: 'human', class: 'actionable', n: 1 },
  ];
  const model = aggregateFlow(rows);
  assert.equal(model.total, 2);
  assert.deepEqual(countsById(model.columns[0]), { implementer: 2 });
  assert.deepEqual(countsById(model.columns[1]), { '0': 2 });
  assert.deepEqual(countsById(model.columns[2]), { human: 2 });
  assert.deepEqual(countsById(model.columns[3]), { actionable: 2 });
  assert.deepEqual(linksByCol(model.links, 0), { 'implementer|0': 2 });
  assert.deepEqual(linksByCol(model.links, 1), { '0|human': 2 });
  assert.deepEqual(linksByCol(model.links, 2), { 'human|actionable': 2 });
});

// ---------------------------------------------------------------------------
// aggregateFlow — top-N source fold: excess origin sources fold into a
// shared 'other' node, and links leaving folded sources remap to 'other' too
// (so Σ origin->level links still == total after folding).
// ---------------------------------------------------------------------------

test('aggregateFlow: top-N source fold — excess sources fold into "other", links remap, totals conserved', () => {
  const rows = [
    { date: '2026-03-01', source: 'a', level: 0, tier: 'human', class: 'actionable', n: 5 },
    { date: '2026-03-01', source: 'b', level: 0, tier: 'human', class: 'actionable', n: 4 },
    { date: '2026-03-01', source: 'c', level: 0, tier: 'human', class: 'actionable', n: 3 },
    { date: '2026-03-01', source: 'd', level: 0, tier: 'human', class: 'actionable', n: 2 },
  ];
  const model = aggregateFlow(rows, { topNSources: 2 });

  assert.equal(model.columns[0].length, 3, 'expected topN(2) + 1 "other" node');
  assert.deepEqual(countsById(model.columns[0]), { a: 5, b: 4, other: 5 });
  assert.equal(model.total, 14);

  const originLinks = linksByCol(model.links, 0);
  assert.deepEqual(originLinks, { 'a|0': 5, 'b|0': 4, 'other|0': 5 });
  assert.equal(
    Object.values(originLinks).reduce((s, n) => s + n, 0),
    model.total,
    'origin->level links must still sum to total after folding',
  );
});

test('aggregateFlow: topNSources >= distinct sources — no fold, no "other" node', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS, { topNSources: 6 });
  assert.deepEqual(countsById(model.columns[0]), { implementer: 3, architect: 2 });
  assert.ok(!('other' in countsById(model.columns[0])));
});

// ---------------------------------------------------------------------------
// aggregateFlow — empty input.
// ---------------------------------------------------------------------------

test('aggregateFlow: empty rows yield empty columns/links and total 0, without throwing', () => {
  assert.doesNotThrow(() => aggregateFlow([]));
  const model = aggregateFlow([]);
  assert.equal(model.total, 0);
  assert.equal(model.links.length, 0);
  assert.equal(model.columns.length, 4);
  for (const col of model.columns) assert.equal(col.length, 0);
});

// ---------------------------------------------------------------------------
// layoutFlow — pixel geometry over the golden model (model = aggregateFlow(
// GOLDEN_FLOW_ROWS)). A single shared px-per-count scale must make node
// heights AND ribbon widths both exactly proportional to their counts, so
// that Σ ribbon w incident on any interior node equals that node's own h
// (flow conservation in pixels, not just in the count model).
// ---------------------------------------------------------------------------

const DIMS = { width: 640, height: 260, nodeWidth: 14, nodePad: 4 };

function nodeAt(nodes, col, id) {
  return nodes.find(n => n.col === col && n.id === id);
}

// A node in column `col` is the `to` side of links with `col: col - 1`
// (incoming) and the `from` side of links with `col: col` (outgoing) — col
// numbering matches aggregateFlow: link.col c connects columns[c] ->
// columns[c+1]. These conserve to node.h SEPARATELY (incoming alone == h,
// outgoing alone == h) — summing both together would double-count and is
// NOT the right invariant (an earlier version of this test made exactly
// that mistake).
function incomingRibbons(ribbons, col, id) {
  return ribbons.filter(r => r.col === col - 1 && r.to === id);
}

function outgoingRibbons(ribbons, col, id) {
  return ribbons.filter(r => r.col === col && r.from === id);
}

test('layoutFlow: golden model — node heights are proportional to counts under one shared scale', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  const { nodes } = layoutFlow(model, DIMS);

  // Level column (col 1) has counts {0: 2, 1: 1, 2: 2} — compare any two.
  const level0 = nodeAt(nodes, 1, '0');
  const level1 = nodeAt(nodes, 1, '1');
  assert.ok(level0 && level1, 'expected level nodes 0 and 1 to exist');
  assert.ok(level1.count > 0);
  assert.ok(
    Math.abs(level0.h / level1.h - level0.count / level1.count) < 1e-9,
    `expected h ratio (${level0.h}/${level1.h}) to match count ratio (${level0.count}/${level1.count})`,
  );
});

test('layoutFlow: a count-0 node has h == 0 (hand-built model, independent of aggregateFlow output)', () => {
  // aggregateFlow never emits a 0-count node (rows with n=0 are skipped), so
  // this constructs a small synthetic model directly to pin layoutFlow's
  // handling of a zero-count node sharing a column with a nonzero one.
  const model = {
    columns: [
      [{ id: 'a', label: 'a', count: 0 }, { id: 'b', label: 'b', count: 4 }],
      [{ id: 'x', label: 'x', count: 4 }],
    ],
    links: [{ col: 0, from: 'b', to: 'x', count: 4 }],
    total: 4,
  };
  const { nodes } = layoutFlow(model, DIMS);
  const a = nodeAt(nodes, 0, 'a');
  const b = nodeAt(nodes, 0, 'b');
  assert.ok(a && b);
  assert.equal(a.h, 0);
  assert.ok(b.h > 0);
});

test('layoutFlow: golden model — column x increases strictly with col index; every node stays within [0,width]', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  const { nodes, width } = layoutFlow(model, DIMS);

  const xByCol = new Map();
  for (const n of nodes) {
    if (!xByCol.has(n.col)) xByCol.set(n.col, n.x);
    assert.equal(n.x, xByCol.get(n.col), 'every node in the same column should share the same x');
    assert.ok(n.x >= -1e-9, `node.x (${n.x}) should be >= 0`);
    assert.ok(n.x + n.w <= width + 1e-9, `node.x + node.w (${n.x + n.w}) should be <= width (${width})`);
  }
  const cols = [...xByCol.keys()].sort((a, b) => a - b);
  for (let i = 1; i < cols.length; i++) {
    assert.ok(
      xByCol.get(cols[i]) > xByCol.get(cols[i - 1]),
      `column ${cols[i]}'s x (${xByCol.get(cols[i])}) should be strictly greater than column ${cols[i - 1]}'s x (${xByCol.get(cols[i - 1])})`,
    );
  }
});

test('layoutFlow: golden model — ribbon widths are proportional to counts under the SAME scale as node heights', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  const { nodes, ribbons } = layoutFlow(model, DIMS);

  // Derive the shared scale from a known node (level '0': count 2), then
  // check every ribbon's w matches count * scale.
  const level0 = nodeAt(nodes, 1, '0');
  const scale = level0.h / level0.count;
  assert.ok(scale > 0);
  for (const r of ribbons) {
    assert.ok(
      Math.abs(r.w - r.count * scale) < 1e-9,
      `ribbon (col ${r.col} ${r.from}->${r.to}) w=${r.w} should equal count(${r.count})*scale(${scale})=${r.count * scale}`,
    );
  }
});

test('layoutFlow: golden model — pixel conservation: Σ incoming ribbon w == Σ outgoing ribbon w == node.h, for each interior node', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  const { nodes, ribbons } = layoutFlow(model, DIMS);

  // Interior columns are level (1) and tier (2) — origin (0) is source-only
  // (no incoming side) and class (3) is sink-only (no outgoing side).
  for (const col of [1, 2]) {
    for (const node of nodes.filter(n => n.col === col)) {
      const incomingW = incomingRibbons(ribbons, col, node.id).reduce((s, r) => s + r.w, 0);
      const outgoingW = outgoingRibbons(ribbons, col, node.id).reduce((s, r) => s + r.w, 0);
      assert.ok(
        Math.abs(incomingW - node.h) < 1e-6,
        `node (col ${col} id ${node.id}) incoming ribbon width sum (${incomingW}) should equal its h (${node.h})`,
      );
      assert.ok(
        Math.abs(outgoingW - node.h) < 1e-6,
        `node (col ${col} id ${node.id}) outgoing ribbon width sum (${outgoingW}) should equal its h (${node.h})`,
      );
    }
  }
});

test('layoutFlow: golden model — every ribbon.d is a non-empty cubic-bezier band ("M...C...")', () => {
  const model = aggregateFlow(GOLDEN_FLOW_ROWS);
  const { ribbons } = layoutFlow(model, DIMS);
  assert.ok(ribbons.length > 0);
  for (const r of ribbons) {
    assert.equal(typeof r.d, 'string');
    assert.ok(r.d.length > 0);
    assert.ok(r.d.startsWith('M'), `ribbon.d should start with 'M' (got: ${r.d.slice(0, 20)}...)`);
    assert.ok(r.d.includes('C'), `ribbon.d should contain a 'C' cubic-bezier command (got: ${r.d})`);
  }
});

test('layoutFlow: empty model yields empty nodes/ribbons, without throwing', () => {
  const emptyModel = aggregateFlow([]);
  assert.doesNotThrow(() => layoutFlow(emptyModel, DIMS));
  const { nodes, ribbons } = layoutFlow(emptyModel, DIMS);
  assert.deepEqual(nodes, []);
  assert.deepEqual(ribbons, []);
});
