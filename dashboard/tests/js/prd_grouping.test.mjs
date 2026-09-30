// Module-contract tests for prd_grouping.js — a plain-JS (no JSX/Babel)
// module holding the pure PRD-grouping-view logic for the Tasks tab's
// "group by PRD" view. Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order: prd_grouping.js
// destructures window.DF_DATUM, window.DF_TASK_VOCAB and window.DF_TASK_SNAPSHOT
// at module scope with no fallback, and datum.js in turn destructures
// window.DF_ENDPOINT_STALENESS. So the shim goes in first and the chain is
// required through it — task_snapshot.test.mjs::loadTaskSnapshot has the same
// shape.
//
// A PRD box's tally is census-shaped and built from the GENERATED vocabulary
// (task_vocab.js, from shared.task_statuses via data/census.py), so every
// member the census counts is counted here too, under the same view.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';
// orderPrdGroups takes computeTiers as an injected parameter — the node suite
// imports it straight from graph_layout.js, same as the browser injects
// window.DF_GRAPH_LAYOUT.computeTiers.
import layout from '../../src/dashboard/static/redux/graph_layout.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'task_vocab.js', 'task_snapshot.js', 'prd_grouping.js'].map(name => REDUX + name);

function loadPrdGrouping() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const loaded = LOAD_CHAIN.map(specifier => require(specifier));
  return { api: loaded[loaded.length - 1], window: win };
}

const { api: grouping, window: loadedWindow } = loadPrdGrouping();
const { prdTitle, aggregatePrdStatus, summarizePrdMembers, prdIsFinished, prdBarSegments } = grouping;
const { prdProgress, prdProgressReading, groupTasksByPrd, orderPrdGroups } = grouping;
const { computeTiers } = layout;
const { MEMBERS, VIEWS, SUB_VIEWS, TONES } = loadedWindow.DF_TASK_VOCAB;
const { isDatum, withReceipt, datumView, EM_DASH } = loadedWindow.DF_DATUM;

const EXPECTED_FUNCTION_NAMES = [
  'prdTitle',
  'aggregatePrdStatus',
  'summarizePrdMembers',
  'prdIsFinished',
  'prdBarSegments',
  'prdProgress',
  'prdProgressReading',
  'groupTasksByPrd',
  'orderPrdGroups',
];

// Builds a minimal task fixture — only the fields prd_grouping.js's
// functions actually read (id, status, prd, deps). Mirrors
// graph_layout.test.mjs's mkTask helper, extended with an optional `prd`.
function mkTask(id, { deps = [], status, prd } = {}) {
  return {
    id,
    ...(status !== undefined ? { status } : {}),
    ...(prd !== undefined ? { prd } : {}),
    deps: deps.map(depId => ({ id: depId })),
  };
}

function tasksWithStatuses(statuses) {
  return statuses.map((status, i) => mkTask(`t${i}`, { status }));
}

function summaryOf(statuses) {
  return summarizePrdMembers(tasksWithStatuses(statuses));
}

// One task per TaskStatus member, stated literally so the fixture is an
// expectation rather than a copy of the vocabulary it is checked against.
const NINE_STATUSES = [
  'pending', 'in-progress', 'blocked', 'deferred', 'review',
  'merge-deferred', 'infra-hold', 'done', 'cancelled',
];

test('the module exposes its functions and assigns window.DF_PRD_GROUPING', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof grouping[name], 'function', `grouping.${name} should be a function`);
  }
  assert.deepEqual(Object.keys(grouping).sort(), EXPECTED_FUNCTION_NAMES.slice().sort());
  // The browser half of the dual export: tab_tasks.jsx destructures this
  // global at module scope with no fallback.
  assert.equal(loadedWindow.DF_PRD_GROUPING, grouping);
});

test('the nine-status fixture is one task per generated member', () => {
  assert.deepEqual(NINE_STATUSES.slice().sort(), MEMBERS.slice().sort());
});

// ---------------------------------------------------------------------------
// prdTitle — basename (after the last '/'), with a trailing '-prd.md'
// stripped, else a trailing '.md' stripped, else returned as-is.
// ---------------------------------------------------------------------------

test('prdTitle: strips a directory prefix and a trailing "-prd.md" suffix', () => {
  assert.equal(
    prdTitle('plans/dashboard-taskgraph-legibility-prd.md'),
    'dashboard-taskgraph-legibility',
  );
});

test('prdTitle: takes the basename after the last "/" when there is no .md suffix', () => {
  assert.equal(prdTitle('reify:docs/prds/foo'), 'foo');
});

test('prdTitle: strips a plain trailing ".md" suffix when there is no directory prefix', () => {
  assert.equal(prdTitle('bar.md'), 'bar');
});

test('prdTitle: returns a bare string with no separators or suffix unchanged', () => {
  assert.equal(prdTitle('plain'), 'plain');
});

// ---------------------------------------------------------------------------
// summarizePrdMembers — the ONE client tally, shaped like a served census
// ---------------------------------------------------------------------------

test('summarizePrdMembers: one task per member counts each member once, under its generated view', () => {
  const summary = summaryOf(NINE_STATUSES);

  assert.deepEqual(Object.keys(summary).sort(), ['counts', 'sub_views', 'total', 'views']);
  assert.deepEqual(Object.keys(summary.counts).sort(), MEMBERS.slice().sort());
  for (const member of MEMBERS) assert.equal(summary.counts[member], 1, member);
  assert.equal(summary.total, 9);
  assert.deepEqual(summary.views, { in_flight: 5, backlog: 2, terminal: 2 });
  assert.deepEqual(summary.sub_views, { running: 1 });
});

test('summarizePrdMembers: the views partition the total, and running sits inside in-flight', () => {
  const summary = summaryOf([
    'done', 'done', 'blocked', 'in-progress', 'in-progress', 'merge-deferred',
    'review', 'pending', 'deferred', 'cancelled', 'infra-hold',
  ]);
  const { in_flight, backlog, terminal } = summary.views;

  assert.equal(in_flight + backlog + terminal, summary.total);
  assert.equal(Object.values(summary.counts).reduce((a, b) => a + b, 0), summary.total);
  assert.equal(summary.sub_views.running, 2);
  assert.ok(summary.sub_views.running <= in_flight, 'running is a subset of in-flight');
});

test('summarizePrdMembers: empty input is all zeros, with every member, view and sub-view present', () => {
  assert.deepEqual(summarizePrdMembers([]), {
    counts: Object.fromEntries(MEMBERS.map(member => [member, 0])),
    total: 0,
    views: Object.fromEntries(Object.keys(VIEWS).map(view => [view, 0])),
    sub_views: Object.fromEntries(Object.keys(SUB_VIEWS).map(view => [view, 0])),
  });
});

test('summarizePrdMembers: a status outside the vocabulary counts in total only', () => {
  const summary = summaryOf(['done', 'mystery', undefined]);

  assert.equal(summary.total, 3);
  assert.equal(summary.counts.done, 1);
  assert.equal(summary.counts.mystery, undefined);
  assert.deepEqual(summary.views, { in_flight: 0, backlog: 0, terminal: 1 });
});

// ---------------------------------------------------------------------------
// aggregatePrdStatus(summary) — the box's CSS status, by view precedence:
// any blocked > any in-flight > any backlog > all done > cancelled.
// ---------------------------------------------------------------------------

const aggregateOf = statuses => aggregatePrdStatus(summaryOf(statuses));

test('aggregatePrdStatus: any blocked wins over everything else', () => {
  assert.equal(aggregateOf(['blocked', 'done']), 'blocked');
  assert.equal(aggregateOf(['review', 'in-progress', 'blocked']), 'blocked');
});

test('aggregatePrdStatus: any in-flight member (no blocked) is in-progress', () => {
  assert.equal(aggregateOf(['in-progress', 'pending']), 'in-progress');
});

test('aggregatePrdStatus: review, infra-hold and merge-deferred alone are each in-progress', () => {
  // review used to fall through every bucket and render the box 'cancelled'.
  for (const status of ['review', 'infra-hold', 'merge-deferred']) {
    assert.equal(aggregateOf([status]), 'in-progress', status);
    assert.equal(aggregateOf([status, 'done']), 'in-progress', `${status} + done`);
  }
});

test('aggregatePrdStatus: any backlog member (nothing in flight) is pending, deferred included', () => {
  assert.equal(aggregateOf(['pending', 'done']), 'pending');
  assert.equal(aggregateOf(['deferred', 'done']), 'pending');
  assert.equal(aggregateOf(['deferred']), 'pending');
});

test('aggregatePrdStatus: all done yields done', () => {
  assert.equal(aggregateOf(['done', 'done']), 'done');
});

test('aggregatePrdStatus: a done+cancelled mix, or all cancelled, is cancelled', () => {
  assert.equal(aggregateOf(['done', 'cancelled']), 'cancelled');
  assert.equal(aggregateOf(['cancelled']), 'cancelled');
});

// ---------------------------------------------------------------------------
// prdIsFinished(summary) — PRD decision 3: terminal vs the rest
// ---------------------------------------------------------------------------

test('prdIsFinished: every member terminal is finished, a done+cancelled mix included', () => {
  assert.equal(prdIsFinished(summaryOf(['done', 'done'])), true);
  assert.equal(prdIsFinished(summaryOf(['done', 'cancelled'])), true);
  assert.equal(prdIsFinished(summaryOf(['cancelled'])), true);
});

test('prdIsFinished: any in-flight or backlog member is not finished', () => {
  for (const status of ['in-progress', 'blocked', 'review', 'merge-deferred', 'infra-hold', 'pending', 'deferred']) {
    assert.equal(prdIsFinished(summaryOf(['done', status])), false, status);
  }
});

test('prdIsFinished: an empty PRD is not finished', () => {
  assert.equal(prdIsFinished(summarizePrdMembers([])), false);
});

// ---------------------------------------------------------------------------
// prdBarSegments(summary) — one segment per present member
// ---------------------------------------------------------------------------

test('prdBarSegments: one segment per present member, in vocabulary order, toned by TONES', () => {
  const segments = prdBarSegments(summaryOf(['done', 'blocked', 'done', 'review']));

  assert.deepEqual(segments.map(s => s.member), MEMBERS.filter(m => ['done', 'blocked', 'review'].includes(m)));
  for (const s of segments) assert.equal(s.tone, TONES[s.member], s.member);
  assert.deepEqual(
    Object.fromEntries(segments.map(s => [s.member, s.share])),
    { blocked: 25, review: 25, done: 50 },
  );
});

test('prdBarSegments: nine members draw nine segments, review and infra-hold included', () => {
  const segments = prdBarSegments(summaryOf(NINE_STATUSES));

  assert.equal(segments.length, 9);
  assert.deepEqual(segments.map(s => s.member), MEMBERS);
  assert.ok(Math.abs(segments.reduce((sum, s) => sum + s.share, 0) - 100) < 1e-9);
});

test('prdBarSegments: an unrecognised status leaves its share of the track undrawn', () => {
  const segments = prdBarSegments(summaryOf(['done', 'mystery']));
  assert.deepEqual(segments.map(s => [s.member, s.share]), [['done', 50]]);
  assert.ok(segments.reduce((sum, s) => sum + s.share, 0) < 100);
});

test('prdBarSegments: an empty PRD draws no segments', () => {
  assert.deepEqual(prdBarSegments(summarizePrdMembers([])), []);
});

// ---------------------------------------------------------------------------
// groupTasksByPrd — buckets by `prd` preserving first-seen-prd input order;
// each group's tasks preserve input order; prd===null tasks collapse into a
// single trailing "no PRD" group (flagged via `noPrd: true`), placed LAST
// regardless of where in the input the null-prd tasks appeared.
// ---------------------------------------------------------------------------

test('groupTasksByPrd: empty input yields an empty array', () => {
  assert.deepEqual(groupTasksByPrd([]), []);
});

test('groupTasksByPrd: two tasks sharing a non-null prd land in one group, preserving input order', () => {
  const tasks = [mkTask('A', { prd: 'p1' }), mkTask('B', { prd: 'p1' })];
  const groups = groupTasksByPrd(tasks);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].prd, 'p1');
  assert.deepEqual(groups[0].tasks.map(t => t.id), ['A', 'B']);
});

test('groupTasksByPrd: groups appear in first-seen-prd input order; each group preserves input order', () => {
  const tasks = [
    mkTask('A', { prd: 'p2' }),
    mkTask('B', { prd: 'p1' }),
    mkTask('C', { prd: 'p2' }),
    mkTask('D', { prd: 'p1' }),
  ];
  const groups = groupTasksByPrd(tasks);
  assert.deepEqual(groups.map(g => g.prd), ['p2', 'p1']);
  assert.deepEqual(groups[0].tasks.map(t => t.id), ['A', 'C']);
  assert.deepEqual(groups[1].tasks.map(t => t.id), ['B', 'D']);
});

test('groupTasksByPrd: null-prd tasks collapse into one trailing "no PRD" group, even when first in input', () => {
  const tasks = [
    mkTask('N1', { prd: null }),
    mkTask('A', { prd: 'p1' }),
    mkTask('N2', { prd: null }),
  ];
  const groups = groupTasksByPrd(tasks);
  assert.equal(groups.length, 2);
  assert.equal(groups[0].prd, 'p1');
  assert.ok(!groups[0].noPrd);
  assert.equal(groups[1].prd, null);
  assert.equal(groups[1].noPrd, true);
  assert.deepEqual(groups[1].tasks.map(t => t.id), ['N1', 'N2']);
});

test('groupTasksByPrd: all-null input yields a single flagged no-PRD group', () => {
  const tasks = [mkTask('A', { prd: null }), mkTask('B', { prd: null })];
  const groups = groupTasksByPrd(tasks);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].noPrd, true);
  assert.deepEqual(groups[0].tasks.map(t => t.id), ['A', 'B']);
});

// ---------------------------------------------------------------------------
// orderPrdGroups — condenses cross-PRD dep edges into a PRD-level mini-DAG,
// tiers it with the INJECTED computeTiers (a PRD consuming another PRD's
// tasks sits below it), tiebreaks within a tier by the tally's views
// (in-flight desc, then backlog desc), falls back to stable insertion order,
// and always force-appends the "no PRD" group last.
// ---------------------------------------------------------------------------

const orderedPrds = tasks => orderPrdGroups(groupTasksByPrd(tasks), computeTiers).map(g => g.prd);

test('orderPrdGroups: a cross-PRD dep tiers the upstream PRD before the downstream one, overriding insertion order', () => {
  // B1 (prd B) depends on A1 (prd A) — B "consumes" A's task. B1 is listed
  // FIRST in the input so groupTasksByPrd's first-seen order is [B, A];
  // orderPrdGroups must still reorder to [A, B] via the mini-DAG tiering.
  const tasks = [
    mkTask('B1', { prd: 'B', deps: ['A1'] }),
    mkTask('A1', { prd: 'A' }),
  ];
  assert.deepEqual(groupTasksByPrd(tasks).map(g => g.prd), ['B', 'A'], 'sanity: first-seen insertion order is B before A');
  assert.deepEqual(orderedPrds(tasks), ['A', 'B']);
});

test('orderPrdGroups: within a tier, a higher in-flight count sorts first', () => {
  const tasks = [
    mkTask('X1', { prd: 'X', status: 'pending' }),
    mkTask('Y1', { prd: 'Y', status: 'blocked' }),
    mkTask('Y2', { prd: 'Y', status: 'in-progress' }),
  ];
  assert.deepEqual(orderedPrds(tasks), ['Y', 'X']);
});

test('orderPrdGroups: equal in-flight, a higher backlog count sorts first as secondary tiebreak', () => {
  const tasks = [
    mkTask('P1', { prd: 'P', status: 'pending' }),
    mkTask('Q1', { prd: 'Q', status: 'pending' }),
    mkTask('Q2', { prd: 'Q', status: 'pending' }),
  ];
  assert.deepEqual(orderedPrds(tasks), ['Q', 'P']);
});

test('orderPrdGroups: a PRD whose only member is review, merge-deferred or infra-hold outranks a pending-only one', () => {
  for (const status of ['review', 'merge-deferred', 'infra-hold']) {
    const tasks = [
      mkTask('P1', { prd: 'P', status: 'pending' }),
      mkTask('P2', { prd: 'P', status: 'pending' }),
      mkTask('R1', { prd: 'R', status }),
    ];
    assert.deepEqual(orderedPrds(tasks), ['R', 'P'], status);
  }
});

test('orderPrdGroups: deferred counts as backlog activity', () => {
  const tasks = [
    mkTask('D1', { prd: 'Finished', status: 'done' }),
    mkTask('F1', { prd: 'Parked', status: 'deferred' }),
  ];
  assert.deepEqual(orderedPrds(tasks), ['Parked', 'Finished']);
});

test('orderPrdGroups: equal tier/activity/backlog falls back to stable insertion order', () => {
  const tasks = [
    mkTask('M1', { prd: 'M', status: 'done' }),
    mkTask('N1', { prd: 'N', status: 'done' }),
  ];
  assert.deepEqual(orderedPrds(tasks), ['M', 'N']);
});

test('orderPrdGroups: the "no PRD" group is always last, regardless of tier/activity', () => {
  // The no-PRD group's sole member is 'blocked' (maximal activity) and the
  // other PRD's sole member is 'done' (minimal) — if noPrd were ordered like
  // any other group it would sort FIRST; it must still come last.
  const tasks = [
    mkTask('Z1', { prd: null, status: 'blocked' }),
    mkTask('W1', { prd: 'W', status: 'done' }),
  ];
  const ordered = orderPrdGroups(groupTasksByPrd(tasks), computeTiers);
  assert.deepEqual(ordered.map(g => g.prd), ['W', null]);
  assert.equal(ordered[ordered.length - 1].noPrd, true);
});

test('orderPrdGroups: deterministic across repeated calls on identical input', () => {
  const tasks = [
    mkTask('B1', { prd: 'B', deps: ['A1'] }),
    mkTask('A1', { prd: 'A' }),
    mkTask('N1', { prd: null }),
  ];
  const groups = groupTasksByPrd(tasks);
  const first = orderPrdGroups(groups, computeTiers);
  const second = orderPrdGroups(groups, computeTiers);
  assert.deepEqual(first.map(g => g.prd), second.map(g => g.prd));
  assert.deepEqual(first.map(g => g.prd), ['A', 'B', null]);
});

// ---------------------------------------------------------------------------
// prdProgress / prdProgressReading — the PRD box count (PRD decision 8)
//
// One combinedDatum over the snapshot rows and the terminal window, so the
// window's lower_bound becomes the '≥' disclosure, and a hole in either part is
// a hole in the count rather than an under-count passed off as a total.
// ---------------------------------------------------------------------------

const PRD = 'plans/x-prd.md';
const OTHER_PRD = 'plans/y-prd.md';
const AS_OF = '2026-09-26T10:00:00+00:00';
const RECEIPT = Object.freeze({ servedAt: '2026-09-26T10:00:05+00:00', receivedAt: 1_800_000_000_000 });
const NOW = RECEIPT.receivedAt + 1_000;
const WINDOW_REASON = 'the newest 400 of 4106 terminal rows; older ones were never read';

const SNAPSHOT_ROW_LIST = [
  { id: 1, status: 'in-progress', prd: PRD },
  { id: 2, status: 'pending', prd: PRD },
  { id: 3, status: 'review', prd: OTHER_PRD },
  { id: 4, status: 'blocked', prd: null },
];
const TERMINAL_ROW_LIST = [
  { id: 5, status: 'done', prd: PRD },
  { id: 6, status: 'cancelled', prd: PRD },
  { id: 7, status: 'done', prd: null },
];

function servedDatum(state, value, reason) {
  return withReceipt({ value, as_of: AS_OF, state, reason, freshness_bound_seconds: 30 }, RECEIPT);
}

function holeDatum(reason) {
  return { value: null, as_of: null, state: 'unknown', reason, freshness_bound_seconds: 30 };
}

const freshRows = () => servedDatum('fresh', SNAPSHOT_ROW_LIST, null);
const terminalWindow = () => servedDatum('lower_bound', TERMINAL_ROW_LIST, WINDOW_REASON);

function spyOn(reading) {
  const spy = value => {
    spy.calls += 1;
    return reading(value);
  };
  spy.calls = 0;
  return spy;
}

test('prdProgress: fresh rows and a lower_bound window combine into a lower_bound tally per PRD', () => {
  const progress = prdProgress(freshRows(), terminalWindow());

  assert.equal(isDatum(progress), true);
  assert.equal(progress.state, 'lower_bound');
  const members = [...SNAPSHOT_ROW_LIST, ...TERMINAL_ROW_LIST];
  for (const prd of [PRD, OTHER_PRD, null]) {
    assert.deepEqual(
      progress.value.get(prd),
      summarizePrdMembers(members.filter(row => row.prd === prd)),
      String(prd),
    );
  }
});

test('prdProgress: the box reads ≥n/m — n its terminal members in the window, m its members across both parts', () => {
  const view = datumView(prdProgress(freshRows(), terminalWindow()), { now: NOW, format: prdProgressReading(PRD) });

  assert.equal(view.text, '≥2/4');
  assert.ok(view.title.includes(WINDOW_REASON), `the tooltip must carry the window's reason: ${view.title}`);
});

test('prdProgressReading(null) reads the no-PRD bucket', () => {
  const view = datumView(prdProgress(freshRows(), terminalWindow()), { now: NOW, format: prdProgressReading(null) });
  assert.equal(view.text, '≥1/2');
});

test('prdProgressReading: a PRD with no rows in scope reads 0/0, not a throw', () => {
  const progress = prdProgress(freshRows(), terminalWindow());
  assert.equal(prdProgressReading('plans/absent-prd.md')(progress.value), '0/0');
});

test('prdProgress: an unknown terminal window is a hole naming the terminal part — never a 0', () => {
  const format = spyOn(prdProgressReading(PRD));
  const view = datumView(prdProgress(freshRows(), holeDatum('requested; waiting for the response')), { now: NOW, format });

  assert.equal(view.text, EM_DASH);
  assert.match(view.title, /terminal rows/);
  assert.match(view.title, /requested; waiting for the response/);
  assert.doesNotMatch(view.text, /0/);
  assert.equal(format.calls, 0, 'format ran on a hole');
});

test('prdProgress: an unknown rows Datum is a hole naming the rows part', () => {
  const format = spyOn(prdProgressReading(PRD));
  const view = datumView(prdProgress(holeDatum('task data unavailable'), terminalWindow()), { now: NOW, format });

  assert.equal(view.text, EM_DASH);
  assert.match(view.title, /in-flight and backlog rows/);
  assert.match(view.title, /task data unavailable/);
  assert.equal(format.calls, 0, 'format ran on a hole');
});

test('prdProgress: neither wire Datum is mutated', () => {
  const rows = freshRows();
  const terminal = terminalWindow();
  const pristine = structuredClone([rows, terminal]);
  prdProgress(rows, terminal);
  assert.deepEqual([rows, terminal], pristine);
});
