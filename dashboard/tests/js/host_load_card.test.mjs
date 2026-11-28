// Module-contract tests for host_load_card.js, the Host load card's reading of
// /api/load: one row per served key, in served order, in a known key's own
// words or else the raw key and a plain number (task 6987). The JSX call site,
// tab_overview.jsx::HostLoadCard, is render-tested in
// jsx_render_overview.test.mjs.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file). The module reads no window
// global, so it loads as a plain CommonJS default import with no window shim.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import hostLoadCard from '../../src/dashboard/static/redux/host_load_card.js';

const { hostLoadRows } = hostLoadCard;

function served(current, sparkline = []) {
  return { current, sparkline, window_mean: null, window_max: null };
}

function rowFor(body, key) {
  return hostLoadRows(body).find(row => row.key === key);
}

test('hostLoadRows: one row per served key, in the order the body serves them', () => {
  const body = {
    verify_scopes_live: served(2),
    fixture_unlabelled_metric: served(1),
    psi_cpu_some_avg10: served(0.5),
  };
  assert.deepEqual(
    hostLoadRows(body).map(row => row.key),
    ['verify_scopes_live', 'fixture_unlabelled_metric', 'psi_cpu_some_avg10'],
  );
});

test('hostLoadRows: a key with no presentation entry reads as its raw key and a plain number', () => {
  for (const [current, text] of [[2.5, '2.5'], [3, '3'], [0.12345, '0.12']]) {
    const row = rowFor({ fixture_unlabelled_metric: served(current) }, 'fixture_unlabelled_metric');
    assert.equal(row.label, 'fixture_unlabelled_metric');
    assert.equal(row.text, text, `current ${current}`);
  }
});

test('hostLoadRows: a known metric reads in its own format, not under its raw key', () => {
  const cases = [
    ['psi_cpu_some_avg10', 12.5, '12.50%'],
    ['verify_scope_memory_bytes', 1610612736, '1.5 GiB'],
    ['occt_queue_depth', 3.6, '4'],
  ];
  for (const [key, current, text] of cases) {
    const row = rowFor({ [key]: served(current) }, key);
    assert.equal(row.text, text, key);
    assert.notEqual(row.label, key, `${key} has a presentation entry, so it is not labelled by its raw key`);
  }
});

test('hostLoadRows: a metric with nothing to serve is an em dash; a populated one keeps its sparkline', () => {
  const body = { verify_scopes_live: served(null, []), occt_queue_depth: served(4, [1, 2, 4]) };
  const empty = rowFor(body, 'verify_scopes_live');
  assert.equal(empty.text, '—');
  assert.deepEqual(empty.sparkline, []);
  assert.deepEqual(rowFor(body, 'occt_queue_depth').sparkline, [1, 2, 4]);
});

test('hostLoadRows: a key named like an Object.prototype member gets the fallback', () => {
  for (const key of ['constructor', 'toString']) {
    const row = rowFor({ [key]: served(7.25) }, key);
    assert.equal(row.label, key);
    assert.equal(row.text, '7.25', key);
  }
});

test('hostLoadRows: no body yet is no rows', () => {
  assert.deepEqual(hostLoadRows(null), []);
  assert.deepEqual(hostLoadRows(undefined), []);
});

test('host_load_card.js assigns window.DF_HOST_LOAD_CARD to its module surface', () => {
  const win = {};
  globalThis.window = win;
  try {
    const require = createRequire(import.meta.url);
    const specifier = '../../src/dashboard/static/redux/host_load_card.js';
    delete require.cache[require.resolve(specifier)];
    const api = require(specifier);
    assert.equal(win.DF_HOST_LOAD_CARD, api);
    assert.equal(win.DF_HOST_LOAD_CARD.hostLoadRows, api.hostLoadRows);
  } finally {
    delete globalThis.window;
  }
});
