// Module-contract tests for spend_readings.js, the client reader of today's
// spend (COSTS.summary.today, served bare by /costs). The topbar pill and the
// Overview's "Spend (today)" tile both read it through this module; their
// WIRING is pinned structurally in Python (test_app_chrome_census.py,
// test_tab_overview.py).
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order, exactly as
// system_health.test.mjs does. data.js rides along only so the pre-fetch case
// reads its real seed; with no `document` it never starts polling. The shim is
// REMOVED once the modules are loaded, and every fixture carries its own
// `__receipt` map, so a reader that reached for a browser global would throw
// here instead of passing by accident.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'data.js', 'spend_readings.js'].map(name => REDUX + name);

function loadSpendReadings() {
  globalThis.window = { DF_ENDPOINT_STALENESS: staleness, dispatchEvent() {} };
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [datumApi, , readingsApi] = LOAD_CHAIN.map(specifier => require(specifier));
  const seed = globalThis.window.DF_DATA;
  delete globalThis.window;
  return { datum: datumApi, readings: readingsApi, seed };
}

const { datum, readings, seed } = loadSpendReadings();
const { todaySpend, spendText } = readings;
const { datumView, EM_DASH } = datum;

const COSTS_ENDPOINT = '/api/v2/dashboard/costs';
const RECEIPT = Object.freeze({ servedAt: '2026-10-07T12:00:30+00:00', receivedAt: 1_800_000_000_000 });

function costsData(summary, receipts = { [COSTS_ENDPOINT]: RECEIPT }) {
  return { COSTS: { summary }, __receipt: receipts };
}

function rendered(data) {
  return datumView(todaySpend(data), { format: spendText, now: RECEIPT.receivedAt }).text;
}

test('todaySpend: the pre-fetch seed is a hole, never the seeded $0.00', () => {
  assert.equal(seed.COSTS.summary.today, 0, 'precondition: data.js seeds today = 0');
  assert.deepEqual(seed.__receipt, {}, 'precondition: the seed carries no receipt');
  const spend = todaySpend(seed);
  assert.equal(spend.state, 'unknown');
  assert.equal(spend.reason, 'not yet fetched');
  assert.equal(rendered(seed), EM_DASH);
});

test('todaySpend: once /costs has delivered, the served value renders in dollars', () => {
  const spend = todaySpend(costsData({ today: 12.5 }));
  assert.equal(spend.state, 'fresh');
  assert.equal(spend.value, 12.5);
  assert.equal(rendered(costsData({ today: 12.5 })), '$12.50');
});

test('todaySpend: a delivered, measured zero is $0.00', () => {
  assert.equal(rendered(costsData({ today: 0 })), '$0.00');
});

test('todaySpend: a delivered payload with no today is a hole saying so', () => {
  const spend = todaySpend(costsData({}));
  assert.equal(spend.state, 'unknown');
  assert.equal(spend.reason, 'no value in the payload');
  assert.equal(rendered({ COSTS: {}, __receipt: { [COSTS_ENDPOINT]: RECEIPT } }), EM_DASH);
});

test('todaySpend: only the /costs receipt counts as delivery', () => {
  const spend = todaySpend(costsData({ today: 3 }, { '/api/v2/dashboard/memory': RECEIPT }));
  assert.equal(spend.state, 'unknown');
  assert.equal(spend.reason, 'not yet fetched');
});

test('spendText: dollars to the cent', () => {
  assert.equal(spendText(0.125), '$0.13');
  assert.equal(spendText(1234), '$1234.00');
});
