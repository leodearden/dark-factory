// Module-contract tests for burndown_bands.js — the pure render DECISIONS
// behind the Burndown tab: the nine stacked status-mix bands, their legend,
// the concurrency-parity banner, and the reader that stamps the burndown
// payload's served Datums (tabs.jsx BurnTab, both the aggregate and the
// per-project views).
//
// WHAT THIS SUITE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT. Only the RENDER
// decisions: which bands exist in which order, which colour each gets, whether
// the parity banner draws, and how a served Datum is read. The WIRE shapes —
// that shape_burndown emits every census member's series and the `latest` /
// `forecast` Datums, and that the server computes parity_alarm correctly — are
// covered behaviourally by dashboard/tests/test_redux_api.py and
// test_burndown_parity_alarm.py, and are not restated here.
//
// COLOURS ARE INJECTED, never read off a global (the prd_grouping.js
// convention: orderPrdGroups takes computeTiers as a parameter rather than
// reaching for window.DF_GRAPH_LAYOUT). So these tests pass a SENTINEL palette
// and assert against the sentinels. Pinning literal oklch strings here would
// duplicate charts.jsx's palette into a second place, which is the drift this
// module exists to remove, and would make a pure re-theme fail a test that has
// nothing to say about themes. What matters is that the right palette SLOT
// reaches the right band, and that the slots stay distinct.
//
// THE VOCABULARY IS THE GENERATED ONE. Members, views, tones and series keys
// are read off the loaded window.DF_TASK_VOCAB rather than restated, except
// the stack order, which is spelled out once as a literal because the order
// itself is the decision under test.
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order: burndown_bands.js
// destructures window.DF_DATUM and window.DF_TASK_VOCAB at module scope with no
// fallback, and datum.js in turn destructures window.DF_ENDPOINT_STALENESS —
// task_snapshot.test.mjs::loadTaskSnapshot has the same shape.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file — no wrapper change needed).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'task_vocab.js', 'burndown_bands.js'].map(name => REDUX + name);

function loadBurndownBands() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [, , bandsApi] = LOAD_CHAIN.map(specifier => require(specifier));
  return { api: bandsApi, window: win };
}

const { api: bands, window: loadedWindow } = loadBurndownBands();
const { burndownStacks, burndownLegend, parityBannerState, burndownDatum, forecastText } = bands;
const { BURNDOWN_ENDPOINT } = bands;
const { MEMBERS, TONES, SERIES_KEYS } = loadedWindow.DF_TASK_VOCAB;
const { datumView } = loadedWindow.DF_DATUM;

const EXPECTED_FUNCTION_NAMES = [
  'burndownStacks',
  'burndownLegend',
  'parityBannerState',
  'burndownDatum',
  'forecastText',
];
const EXPECTED_EXPORT_NAMES = [...EXPECTED_FUNCTION_NAMES, 'BURNDOWN_ENDPOINT'];

// Terminal at the bottom, then backlog, then in-flight; members in TaskStatus
// declaration order inside each view. So the three members task 5591 added
// (review, merge-deferred, infra-hold) sit on top, where a pre-migration hole
// in them blanks only their own bands: StackedAreaChart draws a layer only
// where every layer below it is measured.
const STACK_ORDER = [
  'done',
  'cancelled',
  'pending',
  'deferred',
  'in-progress',
  'blocked',
  'review',
  'merge-deferred',
  'infra-hold',
];

// Sentinel palette over the nine tone slots — deliberately not colours. If an
// implementation ever hard-codes a real oklch string instead of reading the
// injected slot, these assertions fail loudly rather than coincidentally
// matching.
const CP = {
  ok: 'C_OK',
  fg2: 'C_FG2',
  warn: 'C_WARN',
  fg3: 'C_FG3',
  accent: 'C_ACCENT',
  bad: 'C_BAD',
  info: 'C_INFO',
  accent2: 'C_ACCENT2',
  stranded: 'C_STRANDED',
};

const SPLIT_KEYS = ['in_progress_live', 'in_progress_stranded', 'in_progress_rows'];

// A burndown block with a distinct array identity per series, so the tests can
// assert each band is sourced from its OWN member's series by reference —
// index-shifting one band onto another's series is the failure mode the
// per-project call site (pb.*) is exposed to.
function mkBlock() {
  const block = { labels: ['d1', 'd2'] };
  [...Object.values(SERIES_KEYS), ...SPLIT_KEYS].forEach((key, i) => {
    block[key] = [i, i + 100];
  });
  return block;
}

test('the module exposes its readers and assigns window.DF_BURNDOWN_BANDS', () => {
  assert.deepEqual(Object.keys(bands).sort(), EXPECTED_EXPORT_NAMES.slice().sort());
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof bands[name], 'function', `bands.${name} should be a function`);
  }
  assert.equal(BURNDOWN_ENDPOINT, '/api/v2/dashboard/burndown');
  // The browser half of the dual export: tabs.jsx destructures this global.
  assert.equal(loadedWindow.DF_BURNDOWN_BANDS, bands);
});

// ---------------------------------------------------------------------------
// burndownStacks — nine census bands, in view order, each on its own series
// ---------------------------------------------------------------------------

test('burndownStacks: exactly nine bands, one per census member', () => {
  const stacks = burndownStacks(mkBlock(), CP);

  assert.equal(stacks.length, 9);
  assert.deepEqual(stacks.map(s => s.member).sort(), MEMBERS.slice().sort());
});

test('burndownStacks: stacks terminal, then backlog, then in-flight, members in declaration order', () => {
  assert.deepEqual(burndownStacks(mkBlock(), CP).map(s => s.member), STACK_ORDER);
});

test("burndownStacks: each band is keyed and sourced by its member's series key, by identity", () => {
  // By identity, not by value: this is what makes the per-project call site
  // (which passes pb.*, a different block from the aggregate b.*) index-safe.
  const block = mkBlock();
  for (const band of burndownStacks(block, CP)) {
    assert.equal(band.key, SERIES_KEYS[band.member]);
    assert.equal(band.values, block[SERIES_KEYS[band.member]], `${band.member} reads another series`);
  }
});

test("burndownStacks: each band draws in its member's census tone", () => {
  for (const band of burndownStacks(mkBlock(), CP)) {
    assert.equal(band.color, CP[TONES[band.member]], `${band.member} is not in its tone`);
  }
});

test('burndownStacks: all nine band colours are pairwise distinct', () => {
  const colors = burndownStacks(mkBlock(), CP).map(s => s.color);

  assert.equal(
    new Set(colors).size,
    colors.length,
    `two bands share a colour and would be unreadable when stacked: ${colors.join(', ')}`,
  );
});

test('burndownStacks: the in-progress split is not stacked among the census members', () => {
  // The split partitions in_progress_rows, the ROWS' count — another instant
  // from the census members. Stacking it beside them would draw a total no
  // census ever produced.
  const keys = burndownStacks(mkBlock(), CP).map(s => s.key);
  for (const split of SPLIT_KEYS) {
    assert.ok(!keys.includes(split), `${split} was stacked among the census bands: ${keys.join(', ')}`);
  }
});

test('burndownStacks: tolerates a null/undefined block without throwing', () => {
  // BurnTab renders before the first burndown payload has necessarily arrived.
  // The band SET is structural and must survive that — an empty chart, not a
  // blanked tab.
  for (const empty of [null, undefined]) {
    const stacks = burndownStacks(empty, CP);

    assert.deepEqual(stacks.map(s => s.member), STACK_ORDER);
    assert.ok(stacks.every(s => s.values === undefined), 'an absent block should yield no series');
    // An absent block must not also cost the colours.
    assert.ok(stacks.every(s => s.color === CP[TONES[s.member]]));
  }
});

test('burndownStacks: tolerates a missing palette without throwing', () => {
  for (const nopalette of [null, undefined]) {
    const block = mkBlock();
    const stacks = burndownStacks(block, nopalette);

    assert.equal(stacks.length, 9);
    assert.ok(stacks.every(s => s.color === undefined), 'an absent palette should yield no colours');
    // The series still arrive — losing the palette must not also lose the data.
    assert.ok(stacks.every(s => s.values === block[s.key]));
  }
});

// ---------------------------------------------------------------------------
// burndownLegend — the key to the bands above, which must agree with them
// ---------------------------------------------------------------------------

test('burndownLegend: nine entries labelled by census member, in stack order', () => {
  assert.deepEqual(burndownLegend(CP).map(e => e.label), STACK_ORDER);
});

test('burndownLegend: legend colours match the stack colours pairwise, in order', () => {
  // A legend that disagrees with its chart is worse than no legend, because it
  // is believed.
  const legend = burndownLegend(CP);
  const stacks = burndownStacks(mkBlock(), CP);

  assert.equal(legend.length, stacks.length);
  for (let i = 0; i < stacks.length; i++) {
    assert.equal(
      legend[i].color,
      stacks[i].color,
      `legend entry ${i} ("${legend[i].label}") does not carry the colour of ` +
        `band "${stacks[i].key}" it explains`,
    );
  }
});

test('burndownLegend: tolerates a missing palette without throwing', () => {
  for (const nopalette of [undefined, null]) {
    const legend = burndownLegend(nopalette);

    assert.deepEqual(legend.map(e => e.label), STACK_ORDER);
    assert.ok(legend.every(e => e.color === undefined), 'an absent palette should yield no colours');
  }
});

// ---------------------------------------------------------------------------
// parityBannerState — draws ONLY on a server-computed alarm
// ---------------------------------------------------------------------------

test('parityBannerState: renders nothing without a block', () => {
  assert.equal(parityBannerState(null, null), null);
  assert.equal(parityBannerState(undefined, null), null);
});

test('parityBannerState: renders nothing when the alarm is false or absent', () => {
  // Both directions matter. A banner that draws when parity_alarm is false is
  // a false alarm about a false alarm — it accuses the operator's fleet of
  // breaching a cap it never breached.
  assert.equal(parityBannerState({ parity_alarm: false }, null), null);
  assert.equal(parityBannerState({}, null), null);
});

test('parityBannerState: exposes the verdict when the alarm fires', () => {
  const state = parityBannerState(
    { parity_alarm: true, parity_peak: 43, parity_cap: 24, parity_breach_count: 7 },
    null,
  );

  assert.notEqual(state, null);
  assert.equal(state.peak, 43);
  assert.equal(state.cap, 24);
  // The breach count is asserted through the rendered text, not through a
  // separate field: `text` is what the banner actually shows, and every field
  // this descriptor returns is one the call site interpolates.
  assert.ok(state.text.includes('7 snapshots over'), `breach count missing from text: ${state.text}`);
  assert.deepEqual(Object.keys(state).sort(), ['cap', 'peak', 'text']);
});

test('parityBannerState: a missing breach count reads as zero rather than undefined', () => {
  // The `?? 0` arm. Pinned on the rendered text, which is where a regression
  // would actually be seen: without the fallback the banner reads "undefined
  // snapshots over" at the operator.
  const state = parityBannerState({ parity_alarm: true, parity_peak: 43, parity_cap: 24 }, null);

  assert.ok(state.text.includes('0 snapshots'), `expected a zero count, got: ${state.text}`);
  assert.ok(!state.text.includes('undefined'), `undefined leaked into the banner: ${state.text}`);
});

test('parityBannerState: pluralises the snapshot count', () => {
  const text = n =>
    parityBannerState({ parity_alarm: true, parity_peak: 43, parity_cap: 24, parity_breach_count: n }, null).text;

  assert.ok(text(1).includes('1 snapshot over'), `singular form wrong: ${text(1)}`);
  assert.ok(text(2).includes('2 snapshots over'), `plural form wrong: ${text(2)}`);
  // Zero takes the plural, English-style — "0 snapshot over" reads as a typo.
  assert.ok(
    parityBannerState({ parity_alarm: true, parity_peak: 43, parity_cap: 24 }, null).text.includes('0 snapshots over'),
    'a zero count should take the plural form',
  );
});

test('parityBannerState: names the offending projects only when there are any', () => {
  // The aggregate view passes b.parity_projects (the breaching subset); the
  // per-project view passes null, because naming a project inside its own
  // panel says nothing. Asserted on the rendered text: the project suffix is
  // folded into it rather than exposed as a field no call site reads.
  const state = projects =>
    parityBannerState(
      { parity_alarm: true, parity_peak: 43, parity_cap: 24, parity_breach_count: 2 },
      projects,
    ).text;

  assert.equal(state(['a', 'b']), ' · 2 snapshots over · a, b');
  // Both empty forms must render the bare count with NO dangling separator —
  // a trailing ' · ' would read as a truncated project list.
  assert.equal(state(null), ' · 2 snapshots over');
  assert.equal(state([]), ' · 2 snapshots over');
});

test('parityBannerState: the trailing text carries the project list when present', () => {
  const state = parityBannerState(
    { parity_alarm: true, parity_peak: 43, parity_cap: 24, parity_breach_count: 2 },
    ['a', 'b'],
  );

  assert.ok(state.text.endsWith(' · a, b'), `expected the project list to close the text: ${state.text}`);
});

// ---------------------------------------------------------------------------
// burndownDatum — a served Datum, stamped with the burndown receipt
// ---------------------------------------------------------------------------

const SERVED_AT = '2026-05-20T00:30:00+00:00';
const RECEIVED_AT = 1_000_000;

function servedLatest(overrides) {
  return {
    value: { counts: { pending: 4 }, completed: 3, velocity: 1.5, window_days: 2 },
    as_of: '2026-05-20T00:10:00+00:00',
    state: 'stale',
    reason: 'bravo: not measured at the newest sample',
    freshness_bound_seconds: 1200,
    ...(overrides || {}),
  };
}

function withBurndownReceipt() {
  return { __receipt: { [BURNDOWN_ENDPOINT]: { servedAt: SERVED_AT, receivedAt: RECEIVED_AT } } };
}

test('burndownDatum: nothing is fetched until the burndown receipt exists', () => {
  const datum = burndownDatum({ __receipt: {} }, { latest: servedLatest() }, 'latest');

  assert.equal(datum.state, 'unknown');
  assert.equal(datum.reason, 'not yet fetched');
});

test('burndownDatum: a missing block or a non-Datum field is a hole naming the field', () => {
  for (const block of [undefined, null, {}, { forecast: 12 }]) {
    const datum = burndownDatum(withBurndownReceipt(), block, 'forecast');

    assert.equal(datum.state, 'unknown');
    assert.ok(datum.reason.includes('forecast'), `the reason does not name the field: ${datum.reason}`);
  }
});

test('burndownDatum: a served Datum comes back stamped, as a copy', () => {
  const latest = servedLatest();
  const block = { latest };
  const before = JSON.parse(JSON.stringify(block));

  const datum = burndownDatum(withBurndownReceipt(), block, 'latest');

  assert.notEqual(datum, latest);
  assert.equal(datum._served_at, SERVED_AT);
  assert.equal(datum._received_at, RECEIVED_AT);
  assert.deepEqual(datum.value, latest.value);
  assert.deepEqual(block, before, 'the polled payload was mutated');
});

test("burndownDatum: a stamped STALE reading badges its age and shows the server's reason", () => {
  const datum = burndownDatum(withBurndownReceipt(), { latest: servedLatest() }, 'latest');
  const view = datumView(datum, { now: RECEIVED_AT, format: r => String(r.completed) });

  assert.equal(view.text, '3');
  assert.equal(view.title, servedLatest().reason);
  assert.notEqual(view.age, null, 'a stale burndown reading drew no age badge');
});

test('burndownDatum: a stamped FRESH reading within its bound draws no badge', () => {
  const fresh = servedLatest({ as_of: '2026-05-20T00:29:00+00:00', state: 'fresh', reason: null });
  const datum = burndownDatum(withBurndownReceipt(), { latest: fresh }, 'latest');
  const view = datumView(datum, { now: RECEIVED_AT, format: r => String(r.completed) });

  assert.equal(view.age, null);
  assert.equal(view.title, null);
});

// ---------------------------------------------------------------------------
// forecastText — the Forecast tile's reading of a served forecast value
// ---------------------------------------------------------------------------

test('forecastText: one number when the two forecasts agree, a range when they do not', () => {
  assert.equal(forecastText({ forecast_low: 12, forecast_high: 12 }), '12d');
  assert.equal(forecastText({ forecast_low: 10, forecast_high: 14 }), '10–14d');
});
