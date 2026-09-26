// Module-contract tests for task_snapshot.js, the CLIENT reader of the /tasks
// snapshot unit whose server half is dashboard/src/dashboard/data/task_snapshot.py.
// Every census surface — the OrchTab pips, tiles, filter bar and Progress card,
// the Overview tile and pipeline, the topbar pill and the rail badge — reads
// its number through this module, so the decisions it makes are asserted here,
// where node can execute them. The .jsx files are `type="text/babel"` behind
// CDN Babel and cannot run under node; their WIRING to this module is pinned
// structurally in Python (test_tab_orchestrators.py, test_tab_overview.py,
// test_app_chrome_census.py).
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order: task_snapshot.js
// destructures window.DF_DATUM and window.DF_TASK_VOCAB at module scope with no
// fallback, and datum.js in turn destructures window.DF_ENDPOINT_STALENESS. So
// the shim goes in first and the chain is required through it —
// task_done_count.test.mjs::loadGuard has the same shape.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'task_vocab.js', 'task_snapshot.js'].map(name => REDUX + name);

function loadTaskSnapshot() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [, , snapshotApi] = LOAD_CHAIN.map(specifier => require(specifier));
  return { api: snapshotApi, window: win };
}

const { api: snapshot, window: loadedWindow } = loadTaskSnapshot();
const { projectCensus, censusOver, TASKS_ENDPOINT } = snapshot;
const { CENSUS_VIEWS, CENSUS_TILES, censusSegments, censusHistory } = snapshot;
const { inFlightCount, runningOfInFlight, terminalOfTotal, censusTotal, viewShareText } = snapshot;
const { projectRows, viewRows } = snapshot;
const { isDatum, datumView, displayedAgeMs, EM_DASH } = loadedWindow.DF_DATUM;
const { VIEWS, SUB_VIEWS } = loadedWindow.DF_TASK_VOCAB;

const EXPECTED_FUNCTION_NAMES = [
  'projectCensus',
  'censusOver',
  'inFlightCount',
  'runningOfInFlight',
  'terminalOfTotal',
  'censusTotal',
  'viewShareText',
  'censusSegments',
  'censusHistory',
  'projectRows',
  'viewRows',
];
const EXPECTED_EXPORT_NAMES = [...EXPECTED_FUNCTION_NAMES, 'TASKS_ENDPOINT', 'CENSUS_VIEWS', 'CENSUS_TILES'];

// ── Fixtures: boundary sketch #1 ────────────────────────────────────────────
//
// One TaskCensus.to_wire() value per project, the shape data/census.py emits:
// `counts` keyed by status VALUE, `views` and `sub_views` keyed by view name.
// dark-factory is the sketch's own numbers; reify is a small second project so
// a fleet total is observably a SUM rather than one project's value.

const DF_CENSUS_VALUE = Object.freeze({
  counts: {
    'in-progress': 25, blocked: 10, review: 3, 'merge-deferred': 3, 'infra-hold': 2,
    pending: 1300, deferred: 10, done: 4000, cancelled: 106,
  },
  total: 5459,
  views: { in_flight: 43, backlog: 1310, terminal: 4106 },
  sub_views: { running: 25 },
});

const REIFY_CENSUS_VALUE = Object.freeze({
  counts: {
    'in-progress': 2, blocked: 1, review: 0, 'merge-deferred': 0, 'infra-hold': 0,
    pending: 5, deferred: 1, done: 20, cancelled: 2,
  },
  total: 31,
  views: { in_flight: 3, backlog: 6, terminal: 22 },
  sub_views: { running: 2 },
});

const AS_OF = '2026-09-26T10:00:00+00:00';
const SERVED_AT = '2026-09-26T10:00:05+00:00';
const RECEIVED_AT = 1_800_000_000_000;
const NOW = RECEIVED_AT + 1_000;
const TASKS_RECEIPT = Object.freeze({ servedAt: SERVED_AT, receivedAt: RECEIVED_AT });

// A wire Datum in *state*, honouring datum.py's unknown triad.
function datumIn(state, value, overrides) {
  const base =
    state === 'unknown'
      ? { value: null, as_of: null, state, reason: 'status map read failed', freshness_bound_seconds: 30 }
      : { value, as_of: AS_OF, state, reason: state === 'fresh' ? null : 'ReadTimeout', freshness_bound_seconds: 30 };
  return { ...base, ...overrides };
}

// One TASKS_SNAPSHOT[p] entry, the five keys TaskSnapshot.to_wire() emits.
function entryWith(census, rows) {
  return { census, rows, in_progress_live: 1, in_progress_stranded: 0, skew_seconds: 0 };
}

const FRESH_ROWS = datumIn('fresh', [{ id: 1, project: 'dark-factory', title: 't', status: 'in-progress' }]);

function sketchData(overrides) {
  return {
    TASKS_SNAPSHOT: {
      'dark-factory': entryWith(datumIn('fresh', DF_CENSUS_VALUE), FRESH_ROWS),
      reify: entryWith(datumIn('fresh', REIFY_CENSUS_VALUE), datumIn('fresh', [])),
    },
    __receipt: { [TASKS_ENDPOINT]: TASKS_RECEIPT },
    ...overrides,
  };
}

// ── The module ──────────────────────────────────────────────────────────────

test('the module exposes its readers and assigns window.DF_TASK_SNAPSHOT', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof snapshot[name], 'function', `${name} should be a function`);
  }
  assert.deepEqual(Object.keys(snapshot).sort(), EXPECTED_EXPORT_NAMES.slice().sort());
  // The browser half of the dual export: every census surface destructures
  // this global at module scope with no fallback.
  assert.equal(loadedWindow.DF_TASK_SNAPSHOT, snapshot);
});

test('TASKS_ENDPOINT is the receipt key data.js publishes the /tasks payload under', () => {
  assert.equal(TASKS_ENDPOINT, '/api/v2/dashboard/tasks');
});

// ── projectCensus: the served census, stamped with the /tasks receipt ───────

test('projectCensus: the wire census, stamped with the /tasks receipt', () => {
  // data.js registers TASKS_SNAPSHOT as PLAIN, so its nested Datums arrive
  // unstamped; without the receipt datumView could not age them.
  const data = sketchData();
  const wire = data.TASKS_SNAPSHOT['dark-factory'].census;
  const pristine = structuredClone(wire);

  const census = projectCensus(data, 'dark-factory');

  assert.equal(isDatum(census), true);
  assert.deepEqual(census.value, DF_CENSUS_VALUE);
  assert.equal(census.state, 'fresh');
  assert.equal(census.as_of, AS_OF);
  assert.equal(census._served_at, SERVED_AT);
  assert.equal(census._received_at, RECEIVED_AT);
  assert.notEqual(census, wire, 'the stamp must land on a copy');
  assert.deepEqual(wire, pristine, 'the wire census was mutated');
});

test('projectCensus: before the first /tasks payload, the census is not yet fetched', () => {
  const census = projectCensus({ TASKS_SNAPSHOT: {}, __receipt: {} }, 'dark-factory');
  assert.equal(census.state, 'unknown');
  assert.equal(census.reason, 'not yet fetched');
});

test('projectCensus: a project the delivered payload does not carry is named in the reason', () => {
  const census = projectCensus(sketchData(), 'hive');
  assert.equal(census.state, 'unknown');
  assert.equal(census.value, null);
  assert.match(census.reason, /hive/);
});

test('projectCensus: an entry without a census Datum is a reasoned hole — never 0, never a throw', () => {
  const malformed = [
    ['a null entry', null],
    ['an entry with no census', { rows: FRESH_ROWS }],
    ['a null census', entryWith(null, FRESH_ROWS)],
    ['a bare census value', entryWith(DF_CENSUS_VALUE, FRESH_ROWS)],
    ['a bare number', entryWith(43, FRESH_ROWS)],
  ];
  for (const [label, entry] of malformed) {
    const data = sketchData({ TASKS_SNAPSHOT: { 'dark-factory': entry } });
    let census;
    assert.doesNotThrow(() => {
      census = projectCensus(data, 'dark-factory');
    }, label);
    assert.equal(isDatum(census), true, label);
    assert.equal(census.state, 'unknown', label);
    assert.equal(census.value, null, label);
    assert.ok(census.reason, `${label}: the hole must say why`);
  }
});

test('projectCensus: an unknown served census stays unknown with the producer\'s reason', () => {
  const data = sketchData({
    TASKS_SNAPSHOT: { 'dark-factory': entryWith(datumIn('unknown'), FRESH_ROWS) },
  });
  const census = projectCensus(data, 'dark-factory');
  assert.equal(census.state, 'unknown');
  assert.equal(census.reason, 'status map read failed');
});

// ── censusOver: one project, or a member-wise total over several ────────────

test('censusOver: a single project in scope is that project\'s census', () => {
  const data = sketchData();
  assert.deepEqual(censusOver(data, ['dark-factory']), projectCensus(data, 'dark-factory'));
});

test('censusOver(data, null) sums EVERY project in the snapshot, member by member', () => {
  const fleet = censusOver(sketchData(), null);

  assert.equal(fleet.state, 'fresh');
  assert.deepEqual(fleet.value, {
    counts: {
      'in-progress': 27, blocked: 11, review: 3, 'merge-deferred': 3, 'infra-hold': 2,
      pending: 1305, deferred: 11, done: 4020, cancelled: 108,
    },
    total: 5490,
    views: { in_flight: 46, backlog: 1316, terminal: 4128 },
    sub_views: { running: 27 },
  });
});

test('censusOver: the fleet total is still a partition (sketch #4, client half)', () => {
  // A member-wise sum of partitions is a partition. Asserted rather than
  // assumed, because a sum that skipped a member (or double-counted a view)
  // would still produce plausible-looking numbers.
  const { value } = censusOver(sketchData(), null);
  const { in_flight, backlog, terminal } = value.views;

  assert.equal(in_flight + backlog + terminal, value.total);
  assert.equal(Object.values(value.counts).reduce((a, b) => a + b, 0), value.total);
  assert.ok(value.sub_views.running <= in_flight, 'running is a subset of in-flight');
});

test('censusOver: one unknown project makes the total unknown, naming that project', () => {
  // A partial sum is an under-count passed off as a total.
  const data = sketchData();
  data.TASKS_SNAPSHOT.reify = entryWith(datumIn('unknown'), datumIn('unknown'));

  const fleet = censusOver(data, null);

  assert.equal(fleet.state, 'unknown');
  assert.equal(fleet.value, null);
  assert.match(fleet.reason, /reify/);
  assert.match(fleet.reason, /status map read failed/);
});

test('censusOver: a named project absent from the snapshot is a hole in the total', () => {
  const fleet = censusOver(sketchData(), ['dark-factory', 'hive']);
  assert.equal(fleet.state, 'unknown');
  assert.match(fleet.reason, /hive/);
});

test('censusOver: an empty snapshot before the first fetch is not yet fetched', () => {
  const fleet = censusOver({ TASKS_SNAPSHOT: {}, __receipt: {} }, null);
  assert.equal(isDatum(fleet), true);
  assert.equal(fleet.state, 'unknown');
  assert.equal(fleet.reason, 'not yet fetched');
});

test('censusOver: a delivered payload with no project in scope says so, not "not yet fetched"', () => {
  const fleet = censusOver(sketchData({ TASKS_SNAPSHOT: {} }), null);
  assert.equal(fleet.state, 'unknown');
  assert.notEqual(fleet.reason, 'not yet fetched');
  assert.ok(fleet.reason);
});

test('censusOver: the total renders the placeholder, never a 0, when nothing is known', () => {
  const view = datumView(censusOver({ TASKS_SNAPSHOT: {}, __receipt: {} }, null), {
    now: NOW,
    format: () => {
      throw new Error('format must never run on a hole');
    },
  });
  assert.equal(view.text, EM_DASH);
});

// ── The named readings ──────────────────────────────────────────────────────
//
// A surface's number is a `format` over the WHOLE census Datum — never a
// per-view envelope the client built. datumView never invokes a format on a
// hole, so no reading below ever sees a missing value, and none needs a guard.

const byKey = (entries, key) => entries.find(e => e.key === key);

test('CENSUS_VIEWS: the three generated views, in order, frozen', () => {
  assert.deepEqual(CENSUS_VIEWS.map(v => v.key), ['in_flight', 'backlog', 'terminal']);
  assert.deepEqual(CENSUS_VIEWS.map(v => v.key).sort(), Object.keys(VIEWS).sort());
  assert.deepEqual(CENSUS_VIEWS.map(v => v.label), ['in-flight', 'backlog', 'terminal']);
  assert.deepEqual(CENSUS_VIEWS.map(v => v.tone), ['accent', 'warn', 'ok']);
  assert.ok(Object.isFrozen(CENSUS_VIEWS));
  for (const v of CENSUS_VIEWS) {
    assert.ok(Object.isFrozen(v), `${v.key} entry is frozen`);
    assert.equal(typeof v.count, 'function', `${v.key}.count`);
    assert.equal(typeof v.reading, 'function', `${v.key}.reading`);
  }
});

test('CENSUS_TILES: members, not views — each paired with the burndown series of that member', () => {
  // Burndown persists MEMBERS; a view-level tile beside a member series would
  // be a spark of a different quantity than its headline.
  assert.deepEqual(
    CENSUS_TILES.map(t => [t.key, t.label, t.series, t.tone]),
    [
      ['running', 'Running / in-flight', 'in_progress', 'accent'],
      ['blocked', 'Blocked', 'blocked', 'bad'],
      ['pending', 'Pending', 'pending', 'warn'],
    ],
  );
  assert.ok(Object.isFrozen(CENSUS_TILES));
  const membersShown = { running: SUB_VIEWS.running, blocked: ['blocked'], pending: ['pending'] };
  for (const t of CENSUS_TILES) {
    assert.ok(Object.isFrozen(t), `${t.key} entry is frozen`);
    assert.deepEqual(
      membersShown[t.key].map(member => member.replace('-', '_')),
      [t.series],
      `${t.key}'s spark must be the history of the member its headline shows`,
    );
  }
});

test('the retired "active" word appears in no label and no reading', () => {
  const labels = [...CENSUS_VIEWS, ...CENSUS_TILES].map(e => e.label);
  const readings = [
    ...CENSUS_VIEWS.flatMap(v => [v.count, v.reading]),
    ...CENSUS_TILES.map(t => t.reading),
    inFlightCount,
    runningOfInFlight,
    terminalOfTotal,
    censusTotal,
  ].map(reading => reading(DF_CENSUS_VALUE));
  for (const text of [...labels, ...readings]) {
    assert.doesNotMatch(text, /active/i, `"${text}"`);
  }
});

// ── The surface table: boundary sketches #1–#3 ─────────────────────────────
//
// One row per rendered surface, naming the census it reads and the reading its
// JSX passes as `format`. The JSX wiring to exactly these pairs is pinned in
// the Python structural suites; here the pairs are EXECUTED.

const perProject = data => projectCensus(data, 'dark-factory');
const fleet = data => censusOver(data, null);
const runningTile = byKey(CENSUS_TILES, 'running');

const VIEW_READING_TEXT = { in_flight: '25 running of 43 in-flight', backlog: '1310 backlog', terminal: '4106 terminal' };
const VIEW_COUNT_TEXT = { in_flight: '43', backlog: '1310', terminal: '4106' };
const TILE_READING_TEXT = { running: '25 / 43', blocked: '10', pending: '1300' };
const VIEW_SHARE_TEXT = { in_flight: '1%', backlog: '24%', terminal: '75%' };

const SURFACES = [
  ['OrchTab Progress header', perProject, () => terminalOfTotal, '4106/5459'],
  ...CENSUS_VIEWS.map(v => [`OrchTab pip + legend (${v.key})`, perProject, () => v.reading, VIEW_READING_TEXT[v.key]]),
  ...CENSUS_VIEWS.map(v => [`OrchTab filter button (${v.key})`, perProject, () => v.count, VIEW_COUNT_TEXT[v.key]]),
  ...CENSUS_TILES.map(t => [`OrchTab tile (${t.key})`, fleet, () => t.reading, TILE_READING_TEXT[t.key]]),
  ['Overview running tile', fleet, () => runningTile.reading, '25 / 43'],
  ['Overview pipeline total', fleet, () => censusTotal, '5459 total'],
  ...CENSUS_VIEWS.map(v => [`Overview pipeline row count (${v.key})`, fleet, () => v.count, VIEW_COUNT_TEXT[v.key]]),
  ...CENSUS_VIEWS.map(v => [`Overview pipeline row share (${v.key})`, fleet, () => viewShareText(v.key), VIEW_SHARE_TEXT[v.key]]),
  ['Overview Orchestrators table Terminal cell', perProject, () => terminalOfTotal, '4106/5459'],
  ['topbar pill', fleet, () => runningOfInFlight, '25 running of 43 in-flight'],
  ['rail badge', fleet, () => inFlightCount, '43'],
];

function oneProjectData(census, rows) {
  return sketchData({ TASKS_SNAPSHOT: { 'dark-factory': entryWith(census, rows || FRESH_ROWS) } });
}

test('#1 fresh: every surface renders exactly the named number, no tooltip, no badge', () => {
  const data = oneProjectData(datumIn('fresh', DF_CENSUS_VALUE));
  for (const [surface, censusOf, readingOf, expected] of SURFACES) {
    const view = datumView(censusOf(data), { now: NOW, format: readingOf() });
    assert.equal(view.text, expected, surface);
    assert.equal(view.title, null, `${surface}: title`);
    assert.equal(view.age, null, `${surface}: age`);
  }
});

test('#1 fresh: the topbar and the rail read ONE fleet datum and show one in-flight number', () => {
  const data = oneProjectData(datumIn('fresh', DF_CENSUS_VALUE));
  const tasksCensus = censusOver(data, null);
  const topbar = datumView(tasksCensus, { now: NOW, format: runningOfInFlight }).text;
  const rail = datumView(tasksCensus, { now: NOW, format: inFlightCount }).text;
  assert.equal(topbar.match(/of (\d+) in-flight/)[1], rail);
});

function spyOn(reading) {
  const spy = value => {
    spy.calls += 1;
    return reading(value);
  };
  spy.calls = 0;
  return spy;
}

test('#2 unknown: every surface renders the placeholder with the reason — never 0, never 0/1', () => {
  const served = oneProjectData(datumIn('unknown', null, { reason: 'not yet fetched' }));
  const preFetch = { TASKS_SNAPSHOT: {}, __receipt: {} };
  for (const data of [served, preFetch]) {
    for (const [surface, censusOf, readingOf] of SURFACES) {
      const format = spyOn(readingOf());
      const view = datumView(censusOf(data), { now: NOW, format });
      assert.equal(view.text, EM_DASH, surface);
      assert.match(view.title, /not yet fetched/, `${surface}: title`);
      assert.equal(format.calls, 0, `${surface}: format ran on a hole`);
      assert.doesNotMatch(view.text, /0/, surface);
    }
  }
});

test('#3 stale: the aged value renders, with its reason and a growing age badge', () => {
  const threeHoursBefore = new Date(Date.parse(SERVED_AT) - 3 * 3600_000).toISOString();
  const data = oneProjectData(
    datumIn('stale', DF_CENSUS_VALUE, { as_of: threeHoursBefore, reason: 'ReadTimeout' }),
  );
  for (const [surface, censusOf, readingOf, expected] of SURFACES) {
    const census = censusOf(data);
    const view = datumView(census, { now: NOW, format: readingOf() });
    assert.equal(view.text, expected, surface);
    assert.match(view.title, /ReadTimeout/, `${surface}: title`);
    assert.equal(view.age, staleness.formatAge(displayedAgeMs(census, NOW)), `${surface}: age`);
    assert.equal(view.age, '3h', `${surface}: age`);

    const anHourLater = datumView(census, { now: NOW + 3600_000, format: readingOf() });
    assert.equal(anHourLater.age, '4h', `${surface}: the age must grow with no new payload`);
  }
});

test('the reported shape cannot render: an unknown census beside 33 in-flight rows', () => {
  // The bug this leaf closes: the Progress card said "0/1" while the filter bar
  // counted 33 rows. Both now read the census, so both say the same thing.
  const rows = Array.from({ length: 33 }, (_, i) => ({ id: i, project: 'dark-factory', status: 'in-progress' }));
  const data = oneProjectData(datumIn('unknown'), datumIn('fresh', rows));
  const census = perProject(data);
  const inFlight = byKey(CENSUS_VIEWS, 'in_flight');

  assert.equal(datumView(census, { now: NOW, format: terminalOfTotal }).text, EM_DASH);
  assert.equal(datumView(census, { now: NOW, format: inFlight.count }).text, EM_DASH);
});

// ── censusSegments: the Progress and pipeline bar widths ────────────────────

test('censusSegments: fresh — one segment per view, in view order, summing to 100', () => {
  const segments = censusSegments(perProject(oneProjectData(datumIn('fresh', DF_CENSUS_VALUE))));
  assert.deepEqual(segments.map(s => [s.key, s.tone]), CENSUS_VIEWS.map(v => [v.key, v.tone]));
  assert.ok(Math.abs(segments.reduce((sum, s) => sum + s.share, 0) - 100) < 1e-9);
  assert.equal(byKey(segments, 'terminal').share, (4106 / 5459) * 100);
});

test('censusSegments: stale — the widths come from the aged value (sketch #3)', () => {
  const fresh = censusSegments(perProject(oneProjectData(datumIn('fresh', DF_CENSUS_VALUE))));
  const stale = censusSegments(perProject(oneProjectData(datumIn('stale', DF_CENSUS_VALUE))));
  assert.deepEqual(stale, fresh);
});

test('censusSegments: a measured empty census has zero-width segments, never NaN', () => {
  const empty = {
    counts: Object.fromEntries(Object.keys(DF_CENSUS_VALUE.counts).map(k => [k, 0])),
    total: 0,
    views: { in_flight: 0, backlog: 0, terminal: 0 },
    sub_views: { running: 0 },
  };
  const segments = censusSegments(perProject(oneProjectData(datumIn('fresh', empty))));
  assert.equal(segments.length, CENSUS_VIEWS.length);
  for (const s of segments) assert.equal(s.share, 0, s.key);
});

test('censusSegments: an unknown census draws no bar at all', () => {
  assert.deepEqual(censusSegments(perProject(oneProjectData(datumIn('unknown')))), []);
});

test('viewShareText: each pipeline row reads the share of the bar segment beside it', () => {
  const census = perProject(oneProjectData(datumIn('fresh', DF_CENSUS_VALUE)));
  for (const segment of censusSegments(census)) {
    const view = datumView(census, { now: NOW, format: viewShareText(segment.key) });
    assert.equal(view.text, `${segment.share.toFixed(0)}%`, segment.key);
  }
});

test('viewShareText: a measured empty census reads 0%, never NaN%', () => {
  const empty = {
    counts: Object.fromEntries(Object.keys(DF_CENSUS_VALUE.counts).map(k => [k, 0])),
    total: 0,
    views: { in_flight: 0, backlog: 0, terminal: 0 },
    sub_views: { running: 0 },
  };
  const census = perProject(oneProjectData(datumIn('fresh', empty)));
  for (const { key } of CENSUS_VIEWS) {
    assert.equal(datumView(census, { now: NOW, format: viewShareText(key) }).text, '0%', key);
  }
});

// ── censusHistory: the tile spark, over the tile's own scope ────────────────

const BURNDOWN_DATA = {
  BURNDOWN: { labels: ['a', 'b', 'c'], in_progress: [1, 2, 3], blocked: [4, 5, 6], pending: [7, 8, 9] },
  BURNDOWN_BY_PROJECT: {
    'dark-factory': { labels: ['a', 'b'], in_progress: [10, 11], blocked: [12, 13], pending: [14, 15] },
  },
};

test('censusHistory: no project filter reads the server aggregate', () => {
  for (const t of CENSUS_TILES) {
    assert.deepEqual(censusHistory(BURNDOWN_DATA, null, t), BURNDOWN_DATA.BURNDOWN[t.series], t.key);
  }
});

test('censusHistory: one project reads that project\'s series', () => {
  for (const t of CENSUS_TILES) {
    assert.deepEqual(
      censusHistory(BURNDOWN_DATA, ['dark-factory'], t),
      BURNDOWN_DATA.BURNDOWN_BY_PROJECT['dark-factory'][t.series],
      t.key,
    );
    assert.deepEqual(censusHistory(BURNDOWN_DATA, ['hive'], t), [], `${t.key}: absent project`);
  }
});

test('censusHistory: two or more projects draw no spark — no client re-aggregation', () => {
  // Summing ragged per-project series would be a second copy of
  // redux_api.py::shape_burndown's aggregation. A missing spark is honest.
  for (const t of CENSUS_TILES) {
    assert.equal(censusHistory(BURNDOWN_DATA, ['dark-factory', 'reify'], t), null, t.key);
  }
});

// ── Rows by VIEW ────────────────────────────────────────────────────────────
//
// OrchTab's table lists the rows of the selected views: in-flight and backlog
// from the snapshot's rows Datum, terminal from the on-demand window. Selection
// is by the generated vocabulary, so a member the census counts under a view is
// listed under that same view.

test('projectRows: the served rows Datum, stamped with the /tasks receipt', () => {
  const data = sketchData();
  const wire = data.TASKS_SNAPSHOT['dark-factory'].rows;
  const pristine = structuredClone(wire);

  const rows = projectRows(data, 'dark-factory');

  assert.equal(isDatum(rows), true);
  assert.deepEqual(rows.value, wire.value);
  assert.equal(rows._served_at, SERVED_AT);
  assert.equal(rows._received_at, RECEIVED_AT);
  assert.deepEqual(wire, pristine, 'the wire rows were mutated');
});

test('projectRows: an absent project or rows half is a reasoned hole', () => {
  assert.equal(projectRows({ TASKS_SNAPSHOT: {}, __receipt: {} }, 'dark-factory').reason, 'not yet fetched');

  const missing = projectRows(sketchData(), 'hive');
  assert.equal(missing.state, 'unknown');
  assert.match(missing.reason, /hive/);

  const noRows = projectRows(sketchData({ TASKS_SNAPSHOT: { 'dark-factory': { census: FRESH_ROWS } } }), 'dark-factory');
  assert.equal(noRows.state, 'unknown');
  assert.ok(noRows.reason);
});

// One row per non-terminal member — including the three the old filter bar
// silently dropped (review, merge-deferred, infra-hold).
const SNAPSHOT_ROWS = [
  { id: 1, status: 'in-progress' },
  { id: 2, status: 'blocked' },
  { id: 3, status: 'review' },
  { id: 4, status: 'merge-deferred' },
  { id: 5, status: 'infra-hold' },
  { id: 6, status: 'pending' },
  { id: 7, status: 'deferred' },
];
const TERMINAL_ROWS = [
  { id: 8, status: 'done' },
  { id: 9, status: 'cancelled' },
];

const rowsDatum = rows => datumIn('fresh', rows);
const unknownTerminal = reason => datumIn('unknown', null, { reason });
const ids = rows => rows.map(r => r.id);

test('viewRows: in-flight and backlog select the snapshot rows by generated membership', () => {
  const rows = rowsDatum(SNAPSHOT_ROWS);
  const terminal = rowsDatum(TERMINAL_ROWS);

  assert.deepEqual(ids(viewRows(rows, terminal, { in_flight: true }).rows), [1, 2, 3, 4, 5]);
  assert.deepEqual(ids(viewRows(rows, terminal, { backlog: true }).rows), [6, 7]);
  assert.deepEqual(ids(viewRows(rows, terminal, { terminal: true }).rows), [8, 9]);
});

test('viewRows: every snapshot row lands in exactly one view', () => {
  const rows = rowsDatum(SNAPSHOT_ROWS);
  const terminal = rowsDatum([]);
  for (const row of SNAPSHOT_ROWS) {
    const views = CENSUS_VIEWS.filter(v =>
      ids(viewRows(rows, terminal, { [v.key]: true }).rows).includes(row.id),
    );
    assert.equal(views.length, 1, `${row.status} is listed under ${views.length} views`);
  }
});

test('viewRows: several views concatenate in view order, whatever the filter\'s key order', () => {
  const result = viewRows(rowsDatum(SNAPSHOT_ROWS), rowsDatum(TERMINAL_ROWS), {
    terminal: true,
    backlog: true,
    in_flight: true,
  });
  assert.deepEqual(ids(result.rows), [1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.equal(result.placeholder, null);
});

test('viewRows: a selected view whose Datum is unknown lists nothing and says why', () => {
  // The terminal window is fetched on request only (leaf gamma3 wires that
  // request); before it arrives, data.js::datumFor answers 'not yet fetched'.
  const result = viewRows(rowsDatum(SNAPSHOT_ROWS), unknownTerminal('not yet fetched'), {
    in_flight: true,
    terminal: true,
  });

  assert.deepEqual(ids(result.rows), [1, 2, 3, 4, 5]);
  assert.ok(result.placeholder.text.startsWith(EM_DASH), result.placeholder.text);
  assert.match(result.placeholder.text, /terminal/);
  assert.doesNotMatch(result.placeholder.text, /in-flight/);
  assert.match(result.placeholder.title, /not yet fetched/);
});

test('viewRows: an unknown rows Datum names every selected view it feeds', () => {
  const result = viewRows(datumIn('unknown', null, { reason: 'task data unavailable' }), rowsDatum(TERMINAL_ROWS), {
    in_flight: true,
    backlog: true,
  });
  assert.deepEqual(result.rows, []);
  assert.match(result.placeholder.text, /in-flight/);
  assert.match(result.placeholder.text, /backlog/);
  assert.match(result.placeholder.title, /task data unavailable/);
});

test('viewRows: stale rows still list — an aged value is still a value', () => {
  const stale = datumIn('stale', SNAPSHOT_ROWS);
  const result = viewRows(stale, rowsDatum([]), { in_flight: true });
  assert.deepEqual(ids(result.rows), [1, 2, 3, 4, 5]);
  assert.equal(result.placeholder, null);
});

test('viewRows: nothing selected lists nothing and leaves the sentence to orch_filter', () => {
  const result = viewRows(rowsDatum(SNAPSHOT_ROWS), unknownTerminal('not yet fetched'), {});
  assert.deepEqual(result.rows, []);
  assert.equal(result.placeholder, null);
});
