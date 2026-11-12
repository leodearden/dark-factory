// Module-contract tests for prd_lanes_view.js, the CLIENT reader of the /prds
// payload whose server half is dashboard/src/dashboard/data/prd_view.py and
// data/prd_contention.py. Every pure decision the PRD lanes page makes is
// asserted here, where node can execute it; prd_lanes.jsx's rendering of them
// is asserted over the real served bodies in boundary_prd_view.test.mjs.
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order: prd_lanes_view.js
// destructures window.DF_DATUM at module scope with no fallback, and datum.js
// in turn destructures window.DF_ENDPOINT_STALENESS.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'prd_lanes_view.js'].map(name => REDUX + name);

function loadPrdLanesView() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [, api] = LOAD_CHAIN.map(specifier => require(specifier));
  return { api, window: win };
}

const { api: lanesView, window: loadedWindow } = loadPrdLanesView();
const { projectPrdView, projectPrdContention, PRDS_ENDPOINT } = lanesView;
const { readyCell, READY_OVERDUE_DAYS, barSegments, doneOfTotal, signalCell } = lanesView;
const { contentionEntryText, heldShareText, neverStartedText, darkMatterSummary } = lanesView;
const { darkMatterPage, DARK_MATTER_PAGE_SIZE, membersDrawn, inputPip, figureTitle } = lanesView;
const { servedDatum } = loadedWindow.DF_DATUM;

const EXPECTED_FUNCTION_NAMES = [
  'projectPrdView',
  'projectPrdContention',
  'readyCell',
  'barSegments',
  'doneOfTotal',
  'signalCell',
  'contentionEntryText',
  'heldShareText',
  'neverStartedText',
  'darkMatterSummary',
  'darkMatterPage',
  'membersDrawn',
  'inputPip',
  'figureTitle',
];

test('prd_lanes_view.js exports its functions and publishes them as window.DF_PRD_LANES_VIEW', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) assert.equal(typeof lanesView[name], 'function', name);
  assert.equal(loadedWindow.DF_PRD_LANES_VIEW, lanesView);
  assert.equal(PRDS_ENDPOINT, '/api/v2/dashboard/prds');
  assert.equal(READY_OVERDUE_DAYS, 30);
  assert.equal(DARK_MATTER_PAGE_SIZE, 100);
});

// ── (a) The two readers: the served Datum, stamped with the /prds receipt ──

const SERVED_AT = '2026-10-04T09:00:05+00:00';
const RECEIVED_AT = Date.parse('2026-10-04T09:00:06Z');
const PRDS_RECEIPT = { servedAt: SERVED_AT, receivedAt: RECEIVED_AT };

function servedNested(value, overrides = {}) {
  return {
    value,
    as_of: '2026-10-04T09:00:00+00:00',
    state: 'fresh',
    reason: null,
    freshness_bound_seconds: 30,
    ...overrides,
  };
}

const VIEW_VALUE = Object.freeze({ prds: [], lanes: [], rules: {} });
const CONTENTION_VALUE = Object.freeze({ lock_depth: 2, top: [], rules: {} });

function prdsData(overrides = {}) {
  return {
    PRD_VIEW: { 'dark-factory': servedNested(VIEW_VALUE, { state: 'stale', reason: 'lineage aged' }) },
    PRD_CONTENTION: { 'dark-factory': servedNested(CONTENTION_VALUE) },
    __receipt: { [PRDS_ENDPOINT]: PRDS_RECEIPT },
    ...overrides,
  };
}

const READERS = [
  { reader: projectPrdView, key: 'PRD_VIEW' },
  { reader: projectPrdContention, key: 'PRD_CONTENTION' },
];

for (const { reader, key } of READERS) {
  test(`${reader.name}: the served ${key} Datum, stamped with the /prds receipt, its value untouched`, () => {
    const data = prdsData();
    const served = data[key]['dark-factory'];
    const read = reader(data, 'dark-factory');

    assert.deepEqual(read, servedDatum(served, PRDS_ENDPOINT, 'unused', data.__receipt));
    assert.equal(read.value, served.value, 'the reader must hand back the served value itself');
    assert.equal(read.state, served.state);
    assert.equal(read.reason, served.reason);
    assert.equal(read.as_of, served.as_of);
    assert.equal(read._received_at, RECEIVED_AT);
  });

  test(`${reader.name}: a project the payload has no ${key} entry for is unknown, naming /prds and the project`, () => {
    const read = reader(prdsData({ [key]: {} }), 'reify');

    assert.equal(read.state, 'unknown');
    assert.match(read.reason, /\/prds/);
    assert.match(read.reason, /reify/);
  });

  test(`${reader.name}: before the /prds receipt lands, it is servedDatum's not-yet-fetched hole`, () => {
    const data = prdsData({ __receipt: {} });
    const read = reader(data, 'dark-factory');

    assert.deepEqual(read, servedDatum(data[key]['dark-factory'], PRDS_ENDPOINT, 'unused', {}));
    assert.equal(read.state, 'unknown');
  });

  test(`${reader.name}: a missing ${key} map reads as a hole, never a throw`, () => {
    const data = prdsData();
    delete data[key];
    assert.equal(reader(data, 'dark-factory').state, 'unknown');
  });
}

// ── (b) The ready cell ──

test('readyCell: a lower-bound age is shown with ≥, and 43 days is overdue, the detail saying why in words', () => {
  assert.deepEqual(
    readyCell({ count: 3, oldest_task_id: 7, oldest_pending_days: 43, oldest_is_lower_bound: true }),
    { text: 'ready 3 · oldest ≥43d', overdue: true, detail: 'overdue: the oldest ready member has been pending 30d or more' },
  );
});

test('readyCell: a real anchor carries no ≥', () => {
  const cell = readyCell({ count: 2, oldest_task_id: 7, oldest_pending_days: 12, oldest_is_lower_bound: false });
  assert.equal(cell.text, 'ready 2 · oldest 12d');
  assert.ok(!cell.text.includes('≥'));
});

test('readyCell: the alarm fires at READY_OVERDUE_DAYS, not a day before, and only an alarm carries a detail', () => {
  const at = days => readyCell({ count: 1, oldest_task_id: 7, oldest_pending_days: days, oldest_is_lower_bound: false });
  assert.equal(at(READY_OVERDUE_DAYS - 1).overdue, false);
  assert.equal(at(READY_OVERDUE_DAYS - 1).detail, null);
  assert.equal(at(READY_OVERDUE_DAYS).overdue, true);
  assert.ok(at(READY_OVERDUE_DAYS).detail.includes(`${READY_OVERDUE_DAYS}d`), at(READY_OVERDUE_DAYS).detail);
});

test('readyCell: with no age the cell is the count alone, and never overdue', () => {
  assert.deepEqual(
    readyCell({ count: 0, oldest_task_id: null, oldest_pending_days: null, oldest_is_lower_bound: false }),
    { text: 'ready 0', overdue: false, detail: null },
  );
  assert.deepEqual(
    readyCell({ count: 1, oldest_task_id: null, oldest_pending_days: null, oldest_is_lower_bound: true }),
    { text: 'ready 1', overdue: false, detail: null },
  );
});

// ── (c) The bar ──

const LEGEND = Object.freeze([
  { segment: 'done', label: 'done', tone: 'ok' },
  { segment: 'in_progress', label: 'in progress', tone: 'accent' },
  { segment: 'blocked', label: 'blocked', tone: 'bad' },
  { segment: 'deferred', label: 'deferred', tone: 'fg3' },
  { segment: 'ready', label: 'ready', tone: 'warn' },
  { segment: 'waiting', label: 'waiting', tone: 'info' },
]);

function prdRow(overrides = {}) {
  return {
    key: 'plans/x-prd.md',
    title: 'X',
    lane: 'stuck',
    segments: { waiting: 2, ready: 1, deferred: 0, blocked: 0, in_progress: 1, done: 4 },
    total: 8,
    direct_open_ids: [11, 12, 13],
    followup_open: [{ task_id: 21, hops: 1 }],
    ...overrides,
  };
}

test('barSegments: one entry per served legend item, in legend order, whatever the row map order', () => {
  const segments = barSegments(prdRow(), LEGEND);

  assert.deepEqual(
    segments.map(s => s.segment),
    LEGEND.map(item => item.segment),
  );
  assert.deepEqual(segments[0], { segment: 'done', label: 'done', tone: 'ok', count: 4, title: 'done: 4' });
  assert.deepEqual(segments[1], {
    segment: 'in_progress',
    label: 'in progress',
    tone: 'accent',
    count: 1,
    title: 'in progress: 1',
  });
});

test('barSegments: a zero count is kept, not dropped', () => {
  const blocked = barSegments(prdRow(), LEGEND).find(s => s.segment === 'blocked');
  assert.deepEqual(blocked, { segment: 'blocked', label: 'blocked', tone: 'bad', count: 0, title: 'blocked: 0' });
});

test('barSegments: a legend segment the row does not serve throws, naming the segment and the row', () => {
  const { waiting, ...partial } = prdRow().segments;
  assert.equal(waiting, 2);
  assert.throws(
    () => barSegments(prdRow({ segments: partial }), LEGEND),
    error => error instanceof TypeError && error.message.includes('waiting') && error.message.includes('plans/x-prd.md'),
  );
});

// ── (d) done/total ──

test('doneOfTotal: the served done count over the served total, with the open members split by attribution', () => {
  assert.deepEqual(doneOfTotal(prdRow()), { text: '4/8', detail: '3 direct and 1 follow-up open' });
});

// ── (e) The signal ──

const SIGNAL_TEXTS = [
  ['delivered', 'delivered'],
  ['partial', 'partial'],
  ['not_yet', 'not yet'],
  ['unknown', 'unknown'],
];

for (const [mark, text] of SIGNAL_TEXTS) {
  test(`signalCell: the ${mark} mark reads '${text}'`, () => {
    assert.equal(signalCell({ mark, cause: null, reason: null, leaves: [] }).text, text);
  });
}

test('signalCell: an unrecognised mark shows the raw mark rather than guess', () => {
  assert.equal(signalCell({ mark: 'withdrawn', cause: null, reason: null, leaves: [] }).text, 'withdrawn');
});

test('signalCell: with no leaves, the detail is the served reason, else null', () => {
  const unknown = { mark: 'unknown', cause: 'no_sidecar', reason: 'no capability-manifest sidecar', leaves: [] };
  assert.equal(signalCell(unknown).detail, 'no capability-manifest sidecar');
  assert.equal(signalCell({ mark: 'delivered', cause: null, reason: null, leaves: [] }).detail, null);
});

test('signalCell: a delivered mark names its done leaf with the done kind', () => {
  const leaves = [{ label: 'epsilon', task_id: 2581, status: 'done', done_kind: 'merged' }];
  assert.equal(signalCell({ mark: 'delivered', cause: null, reason: null, leaves }).detail, 'epsilon #2581 done (merged)');
});

test('signalCell: a partial mark lists its leaves in served order', () => {
  const leaves = [
    { label: 'η', task_id: 8531, status: 'done', done_kind: 'merged' },
    { label: 'ι', task_id: 8532, status: 'pending', done_kind: null },
  ];
  assert.equal(signalCell({ mark: 'partial', cause: null, reason: null, leaves }).detail, 'η #8531 done (merged)\nι #8532 pending');
});

test('signalCell: a not_yet mark names its pending leaf, and the figure title puts it above the rule', () => {
  const leaves = [{ label: 'θ', task_id: 4652, status: 'pending', done_kind: null }];
  const { detail } = signalCell({ mark: 'not_yet', cause: null, reason: null, leaves });
  assert.equal(detail, 'θ #4652 pending');
  assert.equal(figureTitle('RULE', detail), 'θ #4652 pending\n\nRULE');
});

test('signalCell: a done leaf with no served done kind shows no parenthesis', () => {
  const leaves = [{ label: 'a', task_id: 1, status: 'done', done_kind: null }];
  assert.equal(signalCell({ mark: 'delivered', cause: null, reason: null, leaves }).detail, 'a #1 done');
});

test('signalCell: an unbound_label mark puts its reason first, then each leaf, an unbound one as its label alone', () => {
  const reason = "signal leaf bound to no task: label 'γ'";
  const leaves = [
    { label: 'γ', task_id: null, status: null, done_kind: null },
    { label: 'δ', task_id: 1234, status: 'pending', done_kind: null },
  ];
  const { detail } = signalCell({ mark: 'unknown', cause: 'unbound_label', reason, leaves });
  const expected = "signal leaf bound to no task: label 'γ'\nγ\nδ #1234 pending";
  assert.equal(detail, expected);
  assert.equal(figureTitle('RULE', detail), `${expected}\n\nRULE`);
});

test('signalCell: an external_leaf mark shows the external leaf as its label alone, leaving the why to the reason', () => {
  const reason = "signal leaf is an external task: label 'γ' is bound to reify:12";
  const leaves = [{ label: 'γ', task_id: null, status: null, done_kind: null }];
  assert.equal(
    signalCell({ mark: 'unknown', cause: 'external_leaf', reason, leaves }).detail,
    "signal leaf is an external task: label 'γ' is bound to reify:12\nγ",
  );
});

test('signalCell: a served done kind shows in parentheses whatever the leaf status, the server owning when it is sent', () => {
  const leaves = [{ label: 'a', task_id: 1, status: 'in-progress', done_kind: 'merged' }];
  assert.equal(signalCell({ mark: 'not_yet', cause: null, reason: null, leaves }).detail, 'a #1 in-progress (merged)');
});

test('signalCell: a signal serving no leaves list throws rather than hide the leaves', () => {
  assert.throws(() => signalCell({ mark: 'delivered', cause: null, reason: null }), TypeError);
});

test('signalCell: a leaf_not_in_store mark names the bound id of a leaf the store does not hold', () => {
  const reason = "signal leaf task absent from the store: label 'γ' is bound to task 99";
  const leaves = [{ label: 'γ', task_id: 99, status: null, done_kind: null }];
  assert.equal(
    signalCell({ mark: 'unknown', cause: 'leaf_not_in_store', reason, leaves }).detail,
    "signal leaf task absent from the store: label 'γ' is bound to task 99\nγ #99 not in the store",
  );
});

// ── (f) The contention line ──

test('contentionEntryText: a held key names its holder and the holder\'s store status', () => {
  const text = contentionEntryText({ key: 'src/m', wanted_by_ready: 3, holder: { task_id: '1', status: 'pending' } });
  assert.equal(text, 'src/m · 3 ready want it · held by #1 (pending)');
});

test('contentionEntryText: a holder absent from the store says so, never a blank status', () => {
  const text = contentionEntryText({ key: 'src/m', wanted_by_ready: 1, holder: { task_id: '9', status: null } });
  assert.equal(text, 'src/m · 1 ready want it · held by #9 (not in the store)');
});

test('contentionEntryText: an unheld key says no holder', () => {
  assert.equal(contentionEntryText({ key: 'src/n', wanted_by_ready: 2, holder: null }), 'src/n · 2 ready want it · no holder');
});

test('heldShareText: each served band in served order, its held-key share of its ready tasks', () => {
  const share = { medium: { ready: 16, naming_held_key: 0 }, high: { ready: 2, naming_held_key: 1 } };
  assert.equal(heldShareText(share), 'ready naming a held key: medium 0 of 16 · high 1 of 2');
  assert.equal(heldShareText({}), 'ready naming a held key: no ready tasks');
});

test('neverStartedText: the served count of the served total, with the served ids', () => {
  const text = neverStartedText({ count: 2, of: 3, ids: [113, 371], window_start: '2026-09-14T09:00:00+00:00' });
  assert.equal(text, '2 of 3 ready high never started (#113, #371)');
  assert.equal(neverStartedText({ count: 0, of: 0, ids: [], window_start: null }), '0 of 0 ready high never started');
});

// ── (g) The dark-matter summary ──

function darkMatter(overrides = {}) {
  return {
    total: 9,
    by_reason: { no_link: 5, cycle: 0, missing_target: 4 },
    by_status: { pending: 7, 'in-progress': 0, blocked: 2 },
    by_priority: { high: 3, unrecorded: 6 },
    by_source: { unrecorded: 9 },
    ready_high_ids: [],
    ...overrides,
  };
}

test('darkMatterSummary: the total, then each group under its dimension, served-order pairs, zero counts skipped', () => {
  assert.equal(
    darkMatterSummary(darkMatter()),
    '9 open tasks no PRD owns · why: no_link 5, missing_target 4 · status: pending 7, blocked 2 · '
      + 'priority: high 3, unrecorded 6 · source: unrecorded 9',
  );
});

test('darkMatterSummary: an unrecorded priority and an unrecorded source read apart', () => {
  const summary = darkMatterSummary(darkMatter());
  assert.ok(summary.includes('priority: high 3, unrecorded 6'), summary);
  assert.ok(summary.includes('source: unrecorded 9'), summary);
});

test('darkMatterSummary: a group with every count zero, or none, is left out with its label', () => {
  const summary = darkMatterSummary(darkMatter({ by_reason: { no_link: 0 }, by_source: {} }));
  assert.equal(summary, '9 open tasks no PRD owns · status: pending 7, blocked 2 · priority: high 3, unrecorded 6');
});

test('darkMatterSummary: the served ready-high ids close the summary', () => {
  const summary = darkMatterSummary(darkMatter({ ready_high_ids: [113, 371] }));
  assert.ok(summary.endsWith(' · ready high: #113 #371'), summary);
});

// ── (h) The dark-matter page ──

test('darkMatterPage: the first pages*DARK_MATTER_PAGE_SIZE ids, and how many remain', () => {
  const ids = Array.from({ length: 250 }, (_, i) => i + 1);
  const frozen = Object.freeze([...ids]);

  assert.deepEqual(darkMatterPage(frozen, 1), { ids: ids.slice(0, 100), more: 150 });
  assert.deepEqual(darkMatterPage(frozen, 2), { ids: ids.slice(0, 200), more: 50 });
  assert.deepEqual(darkMatterPage(frozen, 3), { ids, more: 0 });
  assert.deepEqual(darkMatterPage([1, 2], 1), { ids: [1, 2], more: 0 });
});

test('darkMatterPage: never hands back its input array', () => {
  const ids = [1, 2, 3];
  const page = darkMatterPage(ids, 1);
  assert.notEqual(page.ids, ids);
  page.ids.push(99);
  assert.deepEqual(ids, [1, 2, 3]);
});

// ── (i) Members drawn from the narrowed pool ──

function poolRow(project, id, title = `task ${id}`) {
  return { id: `${project}/T-${id}`, project, title, status: 'pending', deps: [] };
}

test('membersDrawn: the pool rows whose uid names a wanted id, in pool order, and the ids no row holds', () => {
  const pool = [poolRow('p', 3), poolRow('q', 1), poolRow('p', 1), poolRow('p', 2)];
  const drawn = membersDrawn(pool, 'p', [1, 3, 7]);

  assert.deepEqual(drawn.rows, [pool[0], pool[2]]);
  assert.deepEqual(drawn.notDrawn, [7]);
});

test('membersDrawn: an empty pool draws nothing and lists every id', () => {
  assert.deepEqual(membersDrawn([], 'p', [4, 5]), { rows: [], notDrawn: [4, 5] });
});

// ── (j) Input pips ──

test('inputPip: a fresh input is its name, titled with its as-of instant', () => {
  const pip = inputPip({ name: 'lineage', as_of: '2026-10-04T09:00:00+00:00', state: 'fresh', reason: null });
  assert.deepEqual(pip, { text: 'lineage', tone: 'ok', title: '2026-10-04T09:00:00+00:00' });
});

test('inputPip: a non-fresh input says its state in words, never colour alone, titled with its reason', () => {
  const unknown = inputPip({ name: 'lineage', as_of: null, state: 'unknown', reason: 'the lineage walk failed' });
  assert.deepEqual(unknown, { text: 'lineage unknown', tone: 'bad', title: 'the lineage walk failed' });
  const stale = inputPip({ name: 'activity', as_of: '2026-10-04T08:00:00+00:00', state: 'stale', reason: 'aged' });
  assert.deepEqual(stale, { text: 'activity stale', tone: 'warn', title: 'aged' });
  const bound = inputPip({ name: 'escalations', as_of: '2026-10-04T08:00:00+00:00', state: 'lower_bound', reason: 'partial' });
  assert.equal(bound.text, 'escalations lower_bound');
  assert.equal(bound.tone, 'warn');
});

test('inputPip: a non-fresh input with no reason falls back to its as-of instant', () => {
  const pip = inputPip({ name: 'activity', as_of: '2026-10-04T08:00:00+00:00', state: 'stale', reason: null });
  assert.equal(pip.title, '2026-10-04T08:00:00+00:00');
});

// ── A figure's title: its row detail, then the served rule ──

test('figureTitle: a figure with no row detail is titled with its rule alone', () => {
  assert.equal(figureTitle('R', null), 'R');
  assert.equal(figureTitle('R', ''), 'R');
});

test('figureTitle: a row detail comes first, a blank line before the rule', () => {
  assert.equal(figureTitle('R', 'K'), 'K\n\nR');
});
