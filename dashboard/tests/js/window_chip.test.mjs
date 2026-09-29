// Module-contract tests for window_chip.js — the pure decisions behind the
// global window chip: which tabs carry it, which vocabulary each offers, whose
// served-window echo each reports, and how a served window is labelled and
// highlighted (app.jsx's Toolbar wiring and tabs.jsx's panel headers).
//
// THE VOCABULARIES ARE SPELLED ONCE HERE, as literals, because the two sets are
// the decision under test: they mirror dashboard/src/dashboard/api/window.py::_WINDOW_DAYS
// and dashboard/src/dashboard/api/burndown.py::_BURNDOWN_WINDOWS. The server's
// half is pinned by tests/test_app.py's _parse_window tests.
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order: window_chip.js
// destructures window.DF_DATUM at module scope with no fallback, and datum.js
// in turn destructures window.DF_ENDPOINT_STALENESS —
// burndown_bands.test.mjs::loadBurndownBands has the same shape.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file — no wrapper change needed).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'window_chip.js'].map(name => REDUX + name);

function loadWindowChip() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [, chipApi] = LOAD_CHAIN.map(specifier => require(specifier));
  return { api: chipApi, window: win };
}

// data.js is loaded on its own shim, as endpoint_staleness.test.mjs does, so
// the registry check reads the real endpointsFor() table.
function loadDataJs() {
  globalThis.window = { dispatchEvent: () => {}, DF_ENDPOINT_STALENESS: staleness };
  globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({}) });
  const require = createRequire(import.meta.url);
  const datumSpecifier = REDUX + 'datum.js';
  delete require.cache[require.resolve(datumSpecifier)];
  require(datumSpecifier);
  const specifier = REDUX + 'data.js';
  delete require.cache[require.resolve(specifier)];
  return require(specifier);
}

const { api: chip, window: loadedWindow } = loadWindowChip();
const {
  WINDOW_SETS,
  DEFAULT_WINDOW,
  TAB_WINDOWS,
  windowForTab,
  windowEcho,
  windowLabel,
  highlightedWindow,
  recentMergesCaption,
} = chip;
const { EM_DASH } = loadedWindow.DF_DATUM;

const STANDARD = ['24h', '7d', '30d', 'all'];
const BURNDOWN = ['24h', '7d', '30d', '90d'];
const COSTS_PATH = '/api/v2/dashboard/costs';
const PERFORMANCE_PATH = '/api/v2/dashboard/performance';
const MERGE_QUEUE_PATH = '/api/v2/dashboard/merge-queue';
const BURNDOWN_PATH = '/api/v2/dashboard/burndown';

const HONOURED_7D = Object.freeze({ requested: '7d', served: '7d', days: 7 });
const HONOURED_24H = Object.freeze({ requested: '24h', served: '24h', days: 1 });
const DECLINED_90D = Object.freeze({ requested: '90d', served: '30d', days: 30 });

// ---------------------------------------------------------------------------
// (a) the tables
// ---------------------------------------------------------------------------

test('exports: the module publishes exactly its API, on module.exports and window.DF_WINDOW_CHIP', () => {
  assert.deepEqual(Object.keys(chip).sort(), [
    'DEFAULT_WINDOW',
    'TAB_WINDOWS',
    'WINDOW_SETS',
    'highlightedWindow',
    'recentMergesCaption',
    'windowEcho',
    'windowForTab',
    'windowLabel',
  ]);
  assert.equal(loadedWindow.DF_WINDOW_CHIP, chip);
});

test('WINDOW_SETS: the standard and burndown vocabularies, in chip order', () => {
  assert.deepEqual(Object.keys(WINDOW_SETS).sort(), ['burndown', 'standard']);
  assert.deepEqual([...WINDOW_SETS.standard], STANDARD);
  assert.deepEqual([...WINDOW_SETS.burndown], BURNDOWN);
});

test('DEFAULT_WINDOW is a member of every set, so a reset always lands on a servable window', () => {
  for (const [name, windows] of Object.entries(WINDOW_SETS)) {
    assert.ok(windows.includes(DEFAULT_WINDOW), `DEFAULT_WINDOW ${DEFAULT_WINDOW} is not in ${name}`);
  }
});

test('the tables are frozen', () => {
  assert.ok(Object.isFrozen(WINDOW_SETS));
  for (const windows of Object.values(WINDOW_SETS)) assert.ok(Object.isFrozen(windows));
  assert.ok(Object.isFrozen(TAB_WINDOWS));
  for (const entry of Object.values(TAB_WINDOWS)) assert.ok(Object.isFrozen(entry));
});

test('TAB_WINDOWS: exactly the chip tabs, each with its vocabulary and the endpoint whose echo it reports', () => {
  assert.deepEqual(Object.keys(TAB_WINDOWS).sort(), ['burn', 'cost', 'merge', 'overview', 'perf']);
  const expected = {
    overview: { windows: STANDARD, endpoint: COSTS_PATH },
    perf: { windows: STANDARD, endpoint: PERFORMANCE_PATH },
    merge: { windows: STANDARD, endpoint: MERGE_QUEUE_PATH },
    cost: { windows: STANDARD, endpoint: COSTS_PATH },
    burn: { windows: BURNDOWN, endpoint: BURNDOWN_PATH },
  };
  for (const [tab, { windows, endpoint }] of Object.entries(expected)) {
    assert.deepEqual([...TAB_WINDOWS[tab].windows], windows, `${tab} offers the wrong set`);
    assert.equal(TAB_WINDOWS[tab].endpoint, endpoint, `${tab} reports the wrong endpoint's echo`);
  }
});

test('every TAB_WINDOWS endpoint is a real endpointsFor() path', () => {
  // A table that silently stops matching a renamed endpoint would read no
  // echo forever, and every header would sit on the placeholder.
  const { endpointsFor, pollKey } = loadDataJs();
  const real = new Set(Object.keys(endpointsFor('24h')).map(pollKey));
  for (const [tab, { endpoint }] of Object.entries(TAB_WINDOWS)) {
    assert.ok(real.has(endpoint), `TAB_WINDOWS[${tab}] names ${endpoint}, not one of: ${[...real].join(', ')}`);
  }
});

// ---------------------------------------------------------------------------
// (b) windowForTab — re-validation on every tab switch
// ---------------------------------------------------------------------------

test('windowForTab: a chip tab that does not offer the window resets to DEFAULT_WINDOW (sketch #8)', () => {
  assert.equal(windowForTab('cost', '90d'), '24h');
  assert.equal(windowForTab('overview', '90d'), '24h');
  assert.equal(windowForTab('burn', 'all'), '24h');
});

test('windowForTab: a chip tab that offers the window keeps it', () => {
  assert.equal(windowForTab('burn', '90d'), '90d');
  assert.equal(windowForTab('cost', '7d'), '7d');
});

test('windowForTab: a chip-less tab keeps whatever the user last chose', () => {
  assert.equal(windowForTab('orch', '90d'), '90d');
  assert.equal(windowForTab('tasks', 'all'), 'all');
});

test('windowForTab: esc-analytics owns its own 7d/28d toggle, not the global chip', () => {
  assert.equal(Object.hasOwn(TAB_WINDOWS, 'esc-analytics'), false);
  for (const w of ['24h', '90d', 'all']) assert.equal(windowForTab('esc-analytics', w), w);
});

// ---------------------------------------------------------------------------
// (c) windowEcho — the validated echo from a receipt
// ---------------------------------------------------------------------------

test('windowEcho: returns a well-formed echo from the endpoint\'s receipt', () => {
  const receipts = { [COSTS_PATH]: { servedAt: 'S', receivedAt: 1, window: DECLINED_90D } };
  assert.deepEqual(windowEcho(receipts, COSTS_PATH), DECLINED_90D);
});

test('windowEcho: null for a missing receipt, a missing window, or a malformed one', () => {
  const receipt = window => ({ servedAt: 'S', receivedAt: 1, window });
  assert.equal(windowEcho({}, COSTS_PATH), null, 'no receipt yet');
  assert.equal(windowEcho({ [COSTS_PATH]: { servedAt: 'S', receivedAt: 1 } }, COSTS_PATH), null, 'absent window');
  assert.equal(windowEcho({ [COSTS_PATH]: receipt(null) }, COSTS_PATH), null, 'null window');
  for (const bad of [
    { served: '30d', days: 30 },
    { requested: '90d', days: 30 },
    { requested: '90d', served: '30d' },
    { requested: '90d', served: '30d', days: '30' },
    { requested: '90d', served: '30d', days: Number.NaN },
    { requested: '90d', served: '30d', days: Number.POSITIVE_INFINITY },
    { requested: 90, served: '30d', days: 30 },
  ]) {
    assert.equal(windowEcho({ [COSTS_PATH]: receipt(bad) }, COSTS_PATH), null, JSON.stringify(bad));
  }
});

// ---------------------------------------------------------------------------
// (d) windowLabel
// ---------------------------------------------------------------------------

test('windowLabel: an honoured window reads as itself', () => {
  assert.equal(windowLabel(HONOURED_7D), '7d');
});

test('windowLabel: a declined window names what was served and what was not available (sketch #8)', () => {
  assert.equal(windowLabel(DECLINED_90D), '30d (90d not available)');
});

test('windowLabel: nothing served yet reads as the one placeholder', () => {
  assert.equal(windowLabel(null), EM_DASH);
});

// ---------------------------------------------------------------------------
// (e) highlightedWindow — the Toolbar lights the SERVED window
// ---------------------------------------------------------------------------

test('highlightedWindow: an honoured echo in the offered set lights its window', () => {
  assert.equal(highlightedWindow(HONOURED_7D, WINDOW_SETS.standard), '7d');
});

test('highlightedWindow: a declined echo lights no chip (sketch #8)', () => {
  assert.equal(highlightedWindow(DECLINED_90D, WINDOW_SETS.standard), null);
});

test('highlightedWindow: nothing served yet lights no chip', () => {
  assert.equal(highlightedWindow(null, WINDOW_SETS.standard), null);
});

test('highlightedWindow: an honoured window the set does not offer lights no chip', () => {
  const all = { requested: 'all', served: 'all', days: 3650 };
  assert.equal(highlightedWindow(all, WINDOW_SETS.burndown), null);
});

// ---------------------------------------------------------------------------
// (f) recentMergesCaption — "showing N of M in <window>" (sketch #9)
// ---------------------------------------------------------------------------

test('recentMergesCaption: the capped rows against the window total', () => {
  assert.equal(recentMergesCaption(200, 228, HONOURED_7D), 'showing 200 of 228 in 7d');
  assert.equal(recentMergesCaption(19, 19, HONOURED_24H), 'showing 19 of 19 in 24h');
});

test('recentMergesCaption: a declined window is labelled as declined', () => {
  assert.equal(recentMergesCaption(5, 5, DECLINED_90D), 'showing 5 of 5 in 30d (90d not available)');
});

test('recentMergesCaption: an unknown total reads as the placeholder, never undefined or NaN', () => {
  for (const total of [undefined, null, Number.NaN, Number.POSITIVE_INFINITY, '228']) {
    const caption = recentMergesCaption(0, total, HONOURED_24H);
    assert.equal(caption, `showing 0 of ${EM_DASH} in 24h`, `total=${String(total)}`);
  }
  assert.equal(recentMergesCaption(0, 0, null), `showing 0 of 0 in ${EM_DASH}`);
});
