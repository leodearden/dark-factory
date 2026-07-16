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

const { aggregateFlow } = layout;

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
