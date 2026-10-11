// Module-contract tests for escalation_views.js, the CLIENT reader of the
// escalation corpus' served views, resolution classes and task cards. The
// server half is dashboard/src/dashboard/data/escalation_corpus.py (the views)
// and dashboard/src/dashboard/data/escalation_analytics.py (the class split).
// tab_escalations.jsx and tab_escalation_analytics.jsx read through this
// module, so the decisions it makes are asserted here, where node can execute
// them; their WIRING is pinned structurally in Python
// (test_tab_escalations.py, test_tab_escalation_analytics.py).
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order: escalation_views.js
// destructures window.DF_DATUM and window.DF_ENDPOINT_STALENESS at module
// scope with no fallback — merge_queue.test.mjs has the same shape.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'escalation_views.js'].map(name => REDUX + name);

function loadEscalationViews() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [, viewsApi] = LOAD_CHAIN.map(specifier => require(specifier));
  return { api: viewsApi, window: win };
}

const { api: escalationViews, window: loadedWindow } = loadEscalationViews();
const {
  queuePending,
  subsectionQueuePending,
  openInHistoryOver,
  corpusAgeCaption,
  resolutionSegments,
  windowedClassSplit,
  taskCard,
  levelCount,
  windowedEscPerDone,
} = escalationViews;
const { isDatum, datumView, EM_DASH } = loadedWindow.DF_DATUM;

// ── Fixtures ────────────────────────────────────────────────────────────────

const ESCALATIONS_ENDPOINT = '/api/v2/dashboard/escalations';
const ANALYTICS_ENDPOINT = '/api/v2/dashboard/escalation-analytics';

// One corpus generation: both endpoints serve views measured at its walk.
const CORPUS_AS_OF = '2026-10-01T12:00:00+00:00';
const ESC_SERVED_AT = '2026-10-01T12:00:30+00:00';
const ANALYTICS_SERVED_AT = '2026-10-01T12:00:20+00:00';
const RECEIVED_AT = 1_800_000_000_000;
const ESC_RECEIPT = Object.freeze({ servedAt: ESC_SERVED_AT, receivedAt: RECEIVED_AT });
const ANALYTICS_RECEIPT = Object.freeze({ servedAt: ANALYTICS_SERVED_AT, receivedAt: RECEIVED_AT });

// A served count Datum over the corpus, honouring datum.py's unknown triad.
function viewDatum(state, value, overrides) {
  const base =
    state === 'unknown'
      ? { value: null, as_of: null, state, reason: 'no escalation queue directory was reached', freshness_bound_seconds: 120 }
      : { value, as_of: CORPUS_AS_OF, state, reason: state === 'fresh' ? null : 'p: 1 unreadable file', freshness_bound_seconds: 120 };
  return { ...base, ...overrides };
}

function views(queuePendingN, openInHistoryN) {
  return {
    queue_pending: viewDatum('fresh', queuePendingN),
    open_in_history: viewDatum('fresh', openInHistoryN),
  };
}

function projectEntry(project, served) {
  return { project, terminal: 0, views: served };
}

function receipts({ escalations = true, analytics = true } = {}) {
  const out = {};
  if (escalations) out[ESCALATIONS_ENDPOINT] = ESC_RECEIPT;
  if (analytics) out[ANALYTICS_ENDPOINT] = ANALYTICS_RECEIPT;
  return out;
}

// PRD sketch #10: one project with 2 pending records at the queue root and 3
// more pending records moved under archive/ — 2 pending in the live queue, 5
// open in history, from ONE walk.
function sketch10(receiptOpts) {
  return {
    ESCALATIONS: {
      subsections: [{ id: 'p', label: 'p', kind: 'orchestrator', escalations: [], views: views(2, 5) }],
      summary: {},
      views: views(2, 5),
    },
    ESCALATION_ANALYTICS: {
      generated_at: CORPUS_AS_OF,
      per_project: [projectEntry('p', views(2, 5))],
      views: views(2, 5),
    },
    __receipt: receipts(receiptOpts),
  };
}

function twoProjects(qOpen) {
  return {
    ESCALATIONS: { subsections: [], summary: {}, views: views(0, 0) },
    ESCALATION_ANALYTICS: {
      generated_at: CORPUS_AS_OF,
      per_project: [
        projectEntry('p', views(2, 5)),
        projectEntry('q', { queue_pending: viewDatum('fresh', 0), open_in_history: qOpen }),
      ],
      views: views(2, 8),
    },
    __receipt: receipts(),
  };
}

// ── The module ──────────────────────────────────────────────────────────────

test('the module exposes its readers and assigns window.DF_ESCALATION_VIEWS', () => {
  assert.deepEqual(
    Object.keys(escalationViews).sort(),
    [
      'corpusAgeCaption',
      'levelCount',
      'openInHistoryOver',
      'queuePending',
      'resolutionSegments',
      'subsectionQueuePending',
      'taskCard',
      'windowedClassSplit',
      'windowedEscPerDone',
    ],
  );
  for (const name of Object.keys(escalationViews)) {
    assert.equal(typeof escalationViews[name], 'function', `${name} should be a function`);
  }
  // The browser half of the dual export: the escalation tabs destructure this
  // global at module scope with no fallback.
  assert.equal(loadedWindow.DF_ESCALATION_VIEWS, escalationViews);
});

// ── (a) queuePending: the served view, stamped with the /escalations receipt ─

test('queuePending: the served queue_pending view, stamped with the /escalations receipt', () => {
  const data = sketch10();
  const wire = data.ESCALATIONS.views.queue_pending;
  const pristine = structuredClone(wire);

  const served = queuePending(data);

  assert.equal(isDatum(served), true);
  assert.equal(served.value, 2);
  assert.equal(served.state, 'fresh');
  assert.equal(served.as_of, CORPUS_AS_OF);
  assert.equal(served._served_at, ESC_SERVED_AT);
  assert.equal(served._received_at, RECEIVED_AT);
  assert.equal(datumView(served, { now: RECEIVED_AT }).text, '2');
  assert.notEqual(served, wire, 'the stamp must land on a copy');
  assert.deepEqual(wire, pristine, 'the wire datum was mutated');
});

test('queuePending: before the first /escalations payload, it is not yet fetched', () => {
  const served = queuePending(sketch10({ escalations: false }));
  assert.equal(served.state, 'unknown');
  assert.equal(served.reason, 'not yet fetched');
});

test('queuePending: a payload with no queue_pending view is a reasoned hole, never 0', () => {
  for (const [label, escalations] of [
    ['no views', { subsections: [], summary: {} }],
    ['a bare count', { subsections: [], summary: {}, views: { queue_pending: 2 } }],
    ['no ESCALATIONS', undefined],
  ]) {
    const data = { ...sketch10(), ESCALATIONS: escalations };
    let served;
    assert.doesNotThrow(() => { served = queuePending(data); }, label);
    assert.equal(served.state, 'unknown', label);
    assert.equal(served.value, null, label);
    assert.match(served.reason, /queue_pending/, `${label}: the reason must name the view`);
  }
});

test('subsectionQueuePending: one queue\'s served view, stamped with the /escalations receipt', () => {
  const data = sketch10();
  const [sec] = data.ESCALATIONS.subsections;

  const served = subsectionQueuePending(sec, data.__receipt);

  assert.equal(served.value, 2);
  assert.equal(served._served_at, ESC_SERVED_AT);
  assert.equal(subsectionQueuePending(sec, {}).reason, 'not yet fetched');
});

test('subsectionQueuePending: a subsection with no view is a hole that names it', () => {
  const served = subsectionQueuePending({ id: 'hive', label: 'hive' }, receipts());
  assert.equal(served.state, 'unknown');
  assert.match(served.reason, /hive/);
});

// ── (b) openInHistoryOver: one total over the project filter ────────────────

test('openInHistoryOver: sketch #10 — the strip reads 5 where the pill reads 2', () => {
  const data = sketch10();
  const open = openInHistoryOver(data, null);

  assert.equal(open.state, 'fresh');
  assert.equal(open.value, 5);
  assert.equal(datumView(open, { now: RECEIVED_AT }).text, '5');
  assert.equal(datumView(queuePending(data), { now: RECEIVED_AT }).text, '2');
});

test('openInHistoryOver: null scope sums every project, a filter only its own', () => {
  const data = twoProjects(viewDatum('fresh', 3));
  assert.equal(openInHistoryOver(data, null).value, 8);
  assert.equal(openInHistoryOver(data, ['p']).value, 5);
  assert.equal(openInHistoryOver(data, ['q']).value, 3);
});

test('openInHistoryOver: an empty project filter filters nothing — the tab\'s convention', () => {
  assert.equal(openInHistoryOver(twoProjects(viewDatum('fresh', 3)), []).value, 8);
});

test('openInHistoryOver: a lower-bound part makes the total a floor, and says why', () => {
  const data = twoProjects(viewDatum('lower_bound', 3));

  const total = openInHistoryOver(data, null);
  const view = datumView(total, { now: RECEIVED_AT });

  assert.equal(total.state, 'lower_bound');
  assert.equal(view.text, '≥8');
  assert.match(total.reason, /\bq: /);
});

test('openInHistoryOver: a filtered project the payload does not carry is a hole that names it', () => {
  const data = twoProjects(viewDatum('fresh', 3));

  const total = openInHistoryOver(data, ['p', 'hive']);

  assert.equal(total.state, 'unknown');
  assert.equal(datumView(total, { now: RECEIVED_AT }).text, EM_DASH);
  assert.match(total.reason, /hive/);
});

test('openInHistoryOver: before the first analytics payload, it is not yet fetched', () => {
  const total = openInHistoryOver(sketch10({ analytics: false }), null);
  assert.equal(total.state, 'unknown');
  assert.equal(total.reason, 'not yet fetched');
});

test('openInHistoryOver: no project in the payload has no total, and says why', () => {
  const data = { ...sketch10(), ESCALATION_ANALYTICS: { per_project: [] } };
  const total = openInHistoryOver(data, null);
  assert.equal(total.state, 'unknown');
  assert.match(total.reason, /project/);
});

// ── (c) one corpus generation, one as_of ────────────────────────────────────

test('the pill and the strip, read from one corpus generation, carry the same as_of', () => {
  const data = sketch10();
  assert.equal(queuePending(data).as_of, openInHistoryOver(data, null).as_of);
  assert.equal(queuePending(data).as_of, CORPUS_AS_OF);
});

// ── (d) corpusAgeCaption: the displayed age, always stated ──────────────────

test('corpusAgeCaption: states the displayed age of the corpus walk', () => {
  // 30s between the walk and serving, plus 12s in this browser.
  const served = queuePending(sketch10());
  assert.equal(corpusAgeCaption(served, RECEIVED_AT + 12_000), 'as of 42s ago');
});

test('corpusAgeCaption: grows under a mocked clock, even while the datum is fresh', () => {
  const served = queuePending(sketch10());
  const early = corpusAgeCaption(served, RECEIVED_AT);
  const later = corpusAgeCaption(served, RECEIVED_AT + 90_000);
  assert.equal(early, 'as of 30s ago');
  assert.equal(later, 'as of 2m ago');
});

test('corpusAgeCaption: null-safe before the first payload', () => {
  assert.equal(corpusAgeCaption(null, RECEIVED_AT), '');
  assert.equal(corpusAgeCaption(undefined, RECEIVED_AT), '');
  assert.equal(corpusAgeCaption(queuePending(sketch10({ escalations: false })), RECEIVED_AT), '');
});

// ── (e) resolutionSegments: every served class, parts summing to the whole ──

test('resolutionSegments: one segment per served class key, shares summing to 1', () => {
  const classes = { actionable: 1, benign: 4, 'moot-terminal-subject': 2, 'stale-strand': 1 };

  const segments = resolutionSegments(classes);

  assert.deepEqual(segments.map(s => s.cls), Object.keys(classes));
  assert.deepEqual(segments.map(s => s.n), [1, 4, 2, 1]);
  assert.equal(segments.find(s => s.cls === 'moot-terminal-subject').share, 0.25);
  assert.equal(segments.find(s => s.cls === 'stale-strand').share, 0.125);
  const total = segments.reduce((sum, s) => sum + s.share, 0);
  assert.ok(Math.abs(total - 1) < 1e-12, `shares sum to ${total}`);
});

test('resolutionSegments: a class the client has never heard of still gets its segment', () => {
  const segments = resolutionSegments({ benign: 1, 'next-new-class': 1 });
  assert.deepEqual(segments.map(s => [s.cls, s.share]), [['benign', 0.5], ['next-new-class', 0.5]]);
});

test('resolutionSegments: an empty population has no segments', () => {
  assert.deepEqual(resolutionSegments({ benign: 0, actionable: 0 }), []);
  assert.deepEqual(resolutionSegments({}), []);
  assert.deepEqual(resolutionSegments(undefined), []);
});

// ── (f) windowedClassSplit: the strip's benign rate over EVERY class ───────

test('windowedClassSplit: totals n over every class, so the benign share has the full denominator', () => {
  const rows = [
    { date: '2026-09-30', class: 'benign', n: 2 },
    { date: '2026-09-30', class: 'actionable', n: 1 },
    { date: '2026-10-01', class: 'benign', n: 1 },
    { date: '2026-10-01', class: 'moot-terminal-subject', n: 1 },
    { date: '2026-10-01', class: 'stale-strand', n: 1 },
  ];

  const split = windowedClassSplit(rows);

  assert.equal(split.total, 6);
  assert.deepEqual(split.byClass, { benign: 3, actionable: 1, 'moot-terminal-subject': 1, 'stale-strand': 1 });
  // benign + actionable alone would read 3/4.
  assert.equal(split.benignShare, 0.5);
  assert.deepEqual(split.benignShareDaily, [2 / 3, 1 / 3]);
});

test('windowedClassSplit: rows from several projects on one date fold into one day', () => {
  const split = windowedClassSplit([
    { date: '2026-10-01', class: 'benign', n: 1 },
    { date: '2026-10-01', class: 'stale-strand', n: 3 },
  ]);
  assert.deepEqual(split.benignShareDaily, [0.25]);
});

test('windowedClassSplit: no classified filings in the window has no share — never 0%', () => {
  const split = windowedClassSplit([]);
  assert.equal(split.total, 0);
  assert.equal(split.benignShare, null);
  assert.deepEqual(split.benignShareDaily, []);
  assert.equal(windowedClassSplit(undefined).benignShare, null);
});

test('windowedClassSplit: a window holding no benign filing reads 0, not a hole', () => {
  assert.equal(windowedClassSplit([{ date: '2026-10-01', class: 'actionable', n: 2 }]).benignShare, 0);
});

// ── (g) taskCard: the row's served task Datum ───────────────────────────────

const TASK_ROW = Object.freeze({ id: 5596, title: 'One corpus walk', status: 'in-progress', description: 'eta' });

function cardRow(task) {
  return { id: 'esc-5596-1', task_id: '5596', task };
}

test('taskCard: a served fresh task Datum renders its row fields', () => {
  const row = cardRow({
    value: TASK_ROW, as_of: CORPUS_AS_OF, state: 'fresh', reason: null, freshness_bound_seconds: 60,
  });

  const card = taskCard(row, receipts(), RECEIVED_AT);

  assert.equal(card.isHole, false);
  assert.equal(card.task.title, 'One corpus walk');
  assert.equal(card.task.status, 'in-progress');
  assert.equal(card.reason, null);
});

test('taskCard: an unknown task Datum is a hole carrying the server\'s reason', () => {
  const reason = 'task 9999 is not in the dark-factory store';
  const row = cardRow({ value: null, as_of: null, state: 'unknown', reason, freshness_bound_seconds: 60 });

  const card = taskCard(row, receipts(), RECEIVED_AT);

  assert.equal(card.isHole, true);
  assert.equal(card.task, null);
  assert.equal(card.reason, reason);
});

test('taskCard: a row with no task Datum, or before the first payload, is a reasoned hole', () => {
  const missing = taskCard({ id: 'esc-1-1', task_id: '1' }, receipts(), RECEIVED_AT);
  assert.equal(missing.isHole, true);
  assert.match(missing.reason, /esc-1-1/);

  const early = taskCard(cardRow(null), {}, RECEIVED_AT);
  assert.equal(early.isHole, true);
  assert.equal(early.reason, 'not yet fetched');
});

test('taskCard: a card older than its bound keeps its fields and badges its age', () => {
  const row = cardRow({
    value: TASK_ROW, as_of: CORPUS_AS_OF, state: 'fresh', reason: null, freshness_bound_seconds: 10,
  });

  const card = taskCard(row, receipts(), RECEIVED_AT + 60_000);

  assert.equal(card.isHole, false);
  assert.equal(card.task.title, 'One corpus walk');
  assert.equal(card.age, '1m');
});

// ── levelCount: the header's per-level counts, gated on the /escalations receipt ─

// data.js's seed: the server's healthy empty shape, before any fetch.
function seededEscalations(receiptOpts, byLevel = { 0: 0, 1: 0, 2: 0 }) {
  return {
    ESCALATIONS: { subsections: [], summary: { by_level: byLevel }, views: views(0, 0) },
    __receipt: receiptOpts === null ? {} : receipts(receiptOpts),
  };
}

test('levelCount: before the first /escalations payload, the seed zero never renders', () => {
  for (const level of [0, 1, 2]) {
    const count = levelCount(seededEscalations(null), level);
    assert.equal(count.state, 'unknown', `L${level}`);
    assert.equal(count.reason, 'not yet fetched', `L${level}`);
    assert.equal(datumView(count, { now: RECEIVED_AT }).text, EM_DASH, `L${level}`);
  }
});

test('levelCount: a delivered count is a reading stamped with the receipt; a measured zero is one too', () => {
  const data = seededEscalations({}, { 0: 4, 1: 3, 2: 0 });

  const l1 = levelCount(data, 1);
  const l2 = levelCount(data, 2);

  assert.equal(isDatum(l1), true);
  assert.equal(l1.value, 3);
  assert.equal(l1.state, 'fresh');
  assert.equal(l1._served_at, ESC_SERVED_AT);
  assert.equal(l1._received_at, RECEIVED_AT);
  assert.equal(l2.state, 'fresh');
  assert.equal(l2.value, 0);
});

test('levelCount: a delivered summary that lacks the level is a hole, not a zero', () => {
  const count = levelCount(seededEscalations({}, { 0: 4 }), 2);
  assert.equal(count.state, 'unknown');
  assert.equal(count.reason, 'no value in the payload');
});

test('levelCount: a payload with no ESCALATIONS or summary does not throw', () => {
  for (const data of [{ __receipt: receipts() }, { ESCALATIONS: {}, __receipt: receipts() }, undefined]) {
    let count;
    assert.doesNotThrow(() => {
      count = levelCount(data, 1);
    });
    assert.equal(count.state, 'unknown');
  }
});

// ── windowedEscPerDone: the strip's esc/done reading and the churn tile's filings ─

function epdRow(date, filings, done) {
  return { date, filings, done, ratio: done ? filings / done : null };
}

const NO_COMPLETIONS = 'no tasks completed in this window';

test('windowedEscPerDone: filings and done are summed across projects, date by date', () => {
  const r = windowedEscPerDone([
    { project: 'alpha', doneCountsRead: true, rows: [epdRow('2026-10-02', 2, 4), epdRow('2026-10-01', 1, 0)] },
    {
      project: 'beta',
      doneCountsRead: true,
      rows: [epdRow('2026-10-02', 2, 0), epdRow('2026-10-01', 3, 1), epdRow('2026-10-03', 0, 2)],
    },
  ]);

  assert.equal(r.filings, 8);
  assert.deepEqual(r.filingsByDate, { '2026-10-01': 4, '2026-10-02': 4, '2026-10-03': 0 });
  assert.equal(r.ratio, 8 / 7);
  assert.deepEqual(r.ratioDaily, [4, 1, 0]);
  assert.equal(r.absentReason, NO_COMPLETIONS);
});

test('windowedEscPerDone: read projects that completed nothing have no ratio', () => {
  const r = windowedEscPerDone([
    { project: 'alpha', doneCountsRead: true, rows: [epdRow('2026-10-01', 3, 0)] },
  ]);

  assert.equal(r.filings, 3);
  assert.equal(r.ratio, null);
  assert.deepEqual(r.ratioDaily, []);
  assert.equal(r.absentReason, NO_COMPLETIONS);
});

test('windowedEscPerDone: an unread project is a hole in the ratio that names it; filings still sum', () => {
  const unread = rows => rows.map(row => ({ ...row, done: null, ratio: null }));
  const r = windowedEscPerDone([
    { project: 'alpha', doneCountsRead: true, rows: [epdRow('2026-10-01', 1, 2)] },
    { project: 'beta', doneCountsRead: false, rows: unread([epdRow('2026-10-01', 2, 0)]) },
    { project: 'gamma', doneCountsRead: false, rows: unread([epdRow('2026-10-02', 4, 0)]) },
  ]);

  assert.equal(r.ratio, null);
  assert.deepEqual(r.ratioDaily, []);
  assert.equal(r.absentReason, 'completed-task counts could not be read for beta, gamma');
  assert.equal(r.filings, 7, 'churn needs every filing, read or not');
  assert.deepEqual(r.filingsByDate, { '2026-10-01': 3, '2026-10-02': 4 });
});

test('windowedEscPerDone: an unread project with no rows is still a hole — the flag is the authority', () => {
  const r = windowedEscPerDone([
    { project: 'alpha', doneCountsRead: true, rows: [epdRow('2026-10-01', 1, 2)] },
    { project: 'gamma', doneCountsRead: false, rows: [] },
  ]);

  assert.equal(r.ratio, null);
  assert.match(r.absentReason, /gamma/);
});

test('windowedEscPerDone: a payload that does not say its counts were read has not shown it', () => {
  const r = windowedEscPerDone([{ project: 'alpha', rows: [epdRow('2026-10-01', 1, 2)] }]);

  assert.equal(r.ratio, null);
  assert.match(r.absentReason, /alpha/);
});

test('windowedEscPerDone: no project has no filings and no ratio', () => {
  const r = windowedEscPerDone([]);

  assert.equal(r.filings, 0);
  assert.equal(r.ratio, null);
  assert.deepEqual(r.ratioDaily, []);
});
