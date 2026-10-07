// Module-contract tests for performance_cards.js, the CLIENT reader of the
// served /performance Datums — each project's cards and the PERFORMANCE_LISTING
// that says whether the listing itself could be read — whose server half is
// dashboard/src/dashboard/data/performance.py. PerfTab's header tiles and its
// per-project blocks read the cards through this module, so the decisions it
// makes are asserted here, where node can execute them. tabs.jsx is
// `type="text/babel"` behind CDN Babel and cannot run under node.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order: performance_cards.js
// destructures window.DF_DATUM at module scope with no fallback, and datum.js
// in turn destructures window.DF_ENDPOINT_STALENESS — merge_queue.test.mjs
// has the same shape.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'performance_cards.js'].map(name => REDUX + name);

function loadPerformanceCards() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [, performanceCardsApi] = LOAD_CHAIN.map(specifier => require(specifier));
  return { api: performanceCardsApi, window: win };
}

const { api: performanceCards, window: loadedWindow } = loadPerformanceCards();
const { projectCards, cardsListing, cardsAbsentReason } = performanceCards;
const { isDatum } = loadedWindow.DF_DATUM;

// ── Fixtures ────────────────────────────────────────────────────────────────

const PERFORMANCE_ENDPOINT = '/api/v2/dashboard/performance';
const SERVED_AT = '2026-10-01T12:00:30+00:00';
const LAST_COMPLETION = '2026-10-01T11:40:00+00:00';
const RECEIVED_AT = 1_800_000_000_000;
const RECEIPT = Object.freeze({ servedAt: SERVED_AT, receivedAt: RECEIVED_AT });
const WINDOW_BOUND_SECONDS = 7 * 86_400;

// A served Datum, honouring datum.py's unknown triad.
function served(state, value, reason) {
  if (state === 'unknown') {
    return { value: null, as_of: null, state, reason, freshness_bound_seconds: WINDOW_BOUND_SECONDS };
  }
  return {
    value, as_of: SERVED_AT, state, reason: state === 'fresh' ? null : reason,
    freshness_bound_seconds: WINDOW_BOUND_SECONDS,
  };
}

const CARDS = Object.freeze({ ttc: { count: 3, p50: 60_000, p95: 120_000 }, paths: [] });

function cardsFor(state = 'fresh') {
  return { ...served(state, CARDS, 'measured before the window'), as_of: LAST_COMPLETION };
}

// One PERFORMANCE[label] entry: the cards Datum beside its histories.
function entryWith(cards) {
  return {
    cards,
    time_centiles_history: { labels: [], p50: [], p95: [] },
    one_pass_history: { labels: [], values: [] },
    escalation_history: { labels: [], values: [] },
  };
}

function perfData(entries, listing, receipt = RECEIPT) {
  const data = {
    PERFORMANCE: entries,
    __receipt: receipt ? { [PERFORMANCE_ENDPOINT]: receipt } : {},
  };
  if (listing !== undefined) data.PERFORMANCE_LISTING = listing;
  return data;
}

const ONE_PROJECT = () => perfData({ hive: entryWith(cardsFor()) }, served('fresh', 1));

// ── The module ──────────────────────────────────────────────────────────────

test('the module exposes its readers and assigns window.DF_PERFORMANCE_CARDS', () => {
  assert.deepEqual(
    Object.keys(performanceCards).sort(),
    ['cardsAbsentReason', 'cardsListing', 'projectCards'],
  );
  for (const name of Object.keys(performanceCards)) {
    assert.equal(typeof performanceCards[name], 'function', `${name} should be a function`);
  }
  // The browser half of the dual export: tabs.jsx destructures this global at
  // module scope with no fallback.
  assert.equal(loadedWindow.DF_PERFORMANCE_CARDS, performanceCards);
});

// ── projectCards: the served cards Datum, stamped with the /performance receipt

test('projectCards: the served Datum, stamped with the /performance receipt', () => {
  const data = ONE_PROJECT();
  const wire = data.PERFORMANCE.hive.cards;
  const pristine = structuredClone(wire);

  const cards = projectCards(data, 'hive');

  assert.equal(isDatum(cards), true);
  assert.equal(cards.state, 'fresh');
  assert.deepEqual(cards.value, CARDS);
  assert.equal(cards.as_of, LAST_COMPLETION);
  assert.equal(cards._served_at, SERVED_AT);
  assert.equal(cards._received_at, RECEIVED_AT);
  assert.notEqual(cards, wire, 'the stamp must land on a copy');
  assert.deepEqual(wire, pristine, 'the wire datum was mutated');
});

test('projectCards: before the first /performance payload, it is not yet fetched', () => {
  const cards = projectCards(perfData({}, undefined, null), 'hive');
  assert.equal(cards.state, 'unknown');
  assert.equal(cards.reason, 'not yet fetched');
});

test('projectCards: a project without a cards Datum is a reasoned hole naming it — never a throw', () => {
  const malformed = [
    ['an absent project', undefined],
    ['a null entry', null],
    ['an entry with no cards', { one_pass_history: { labels: [], values: [] } }],
    ['bare cards', entryWith(CARDS)],
  ];
  for (const [label, entry] of malformed) {
    const entries = entry === undefined ? {} : { hive: entry };
    let cards;
    assert.doesNotThrow(() => {
      cards = projectCards(perfData(entries, served('fresh', 1)), 'hive');
    }, label);
    assert.equal(isDatum(cards), true, label);
    assert.equal(cards.state, 'unknown', label);
    assert.equal(cards.value, null, label);
    assert.match(cards.reason, /hive/, `${label}: the reason must name the project`);
  }
});

// ── cardsListing: the served PERFORMANCE_LISTING Datum, stamped ─────────────

test('cardsListing: the served listing Datum, stamped with the /performance receipt', () => {
  const data = perfData({}, served('lower_bound', 2, '1 of 3 runs.db could not be read; their projects are not listed'));
  const wire = data.PERFORMANCE_LISTING;
  const pristine = structuredClone(wire);

  const listing = cardsListing(data);

  assert.equal(isDatum(listing), true);
  assert.equal(listing.state, 'lower_bound');
  assert.equal(listing.value, 2);
  assert.equal(listing._served_at, SERVED_AT);
  assert.equal(listing._received_at, RECEIVED_AT);
  assert.notEqual(listing, wire, 'the stamp must land on a copy');
  assert.deepEqual(wire, pristine, 'the wire datum was mutated');
});

test('cardsListing: before the first /performance payload, it is not yet fetched', () => {
  const listing = cardsListing(perfData({}, undefined, null));
  assert.equal(listing.state, 'unknown');
  assert.equal(listing.reason, 'not yet fetched');
});

test('cardsListing: a payload without the listing Datum is a reasoned hole', () => {
  for (const [label, listing] of [['no listing', undefined], ['a null listing', null], ['a bare count', 2]]) {
    const read = cardsListing(perfData({}, listing));
    assert.equal(read.state, 'unknown', label);
    assert.equal(read.reason, 'the /performance payload has no PERFORMANCE_LISTING Datum', label);
  }
});

// ── cardsAbsentReason: why a header tile derived over the cards has no value ─

test('cardsAbsentReason: before the first /performance payload, it is not yet fetched', () => {
  assert.equal(cardsAbsentReason(perfData({}, undefined, null)), 'not yet fetched');
});

test('cardsAbsentReason: a fully read listing means the window holds no tasks', () => {
  assert.equal(cardsAbsentReason(perfData({}, served('fresh', 0))), 'no tasks in this window');
});

test('cardsAbsentReason: an unread listing says why, never "no tasks"', () => {
  // THE SIGNAL. An unreadable fleet of runs.db files serves no cards at all;
  // the header tiles must say the reads failed rather than that nothing ran.
  const reason = 'none of the 3 runs.db files could be read';
  assert.equal(cardsAbsentReason(perfData({}, served('unknown', null, reason))), reason);
});

test('cardsAbsentReason: a short listing says which reads are missing', () => {
  const reason = '1 of 3 runs.db could not be read; their projects are not listed';
  assert.equal(cardsAbsentReason(perfData({}, served('lower_bound', 0, reason))), reason);
});

test('cardsAbsentReason: a stale listing gives its reason', () => {
  const reason = 'listed 2h before it was served';
  assert.equal(cardsAbsentReason(perfData({}, served('stale', 0, reason))), reason);
});

test('cardsAbsentReason: a payload missing its listing says so', () => {
  assert.equal(
    cardsAbsentReason(perfData({}, undefined)),
    'the /performance payload has no PERFORMANCE_LISTING Datum',
  );
});
