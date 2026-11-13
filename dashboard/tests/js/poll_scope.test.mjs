// Module-contract tests for poll_scope.js — the surface -> endpoint maps that
// decide what the dashboard POLLS (task 5825; moved to their own module by
// task 6304).
//
// Run via `node --test`; auto-discovered by the `**/*.test.mjs` glob in
// dashboard/tests/test_graph_layout_js.py.
//
// poll_scope.js has no package.json in the repo, so it resolves as CommonJS
// (`module.exports = <object>`). Node's cjs-module-lexer cannot statically
// detect named exports assigned from a variable, so we default-import the
// module and destructure (mirrors endpoint_staleness.test.mjs). A plain static
// ESM import is SAFE here: the module touches no browser global at load; its
// `window.DF_POLL_SCOPE` assignment is guarded by `typeof window !==
// 'undefined'`.
//
// data.js polls CHROME_ENDPOINTS ∪ TAB_ENDPOINTS[activeTab], so a path missing
// from these maps is not merely unmonitored: the surface that reads it
// silently stops refreshing.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import pollScope from '../../src/dashboard/static/redux/poll_scope.js';

import { loadDataJs } from './_data_loader.mjs';

const { TAB_ENDPOINTS, CHROME_ENDPOINTS } = pollScope;

const REDUX_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../src/dashboard/static/redux',
);

const DASH = '/api/v2/dashboard';

// ── TAB_ENDPOINTS is machine-checked against the real sources ──────────────

// app.jsx is `type="text/babel"` and cannot be imported here, so the tab ids
// are read out of its source — the same source-text contract idiom the Python
// suite uses for .jsx (see test_tab_escalation_analytics.py).
function appTabIds() {
  const src = fs.readFileSync(path.join(REDUX_DIR, 'app.jsx'), 'utf8');
  const block = src.match(/const tabs = \[([\s\S]*?)\];/);
  assert.ok(block, 'could not locate the `const tabs = [...]` block in app.jsx');
  return [...block[1].matchAll(/id:\s*'([^']+)'/g)].map(m => m[1]);
}

test('every mapped endpoint is a real endpointsFor() path', () => {
  // A map that silently stops matching a renamed endpoint is a check that
  // quietly stops checking: it would report "healthy" forever.
  const { endpointsFor, pollKey } = loadDataJs().api;
  const real = new Set(Object.keys(endpointsFor('24h')).map(pollKey));

  for (const [tab, paths] of Object.entries(TAB_ENDPOINTS)) {
    assert.ok(Array.isArray(paths) && paths.length > 0,
      `tab ${tab} must map to a non-empty array of endpoint paths`);
    for (const p of paths) {
      assert.ok(real.has(p),
        `TAB_ENDPOINTS[${tab}] names ${p}, which is not a pollKey of any ` +
        `endpointsFor('24h') key: ${[...real].join(', ')}`);
    }
  }
});

test('every mapped tab id is one of app.jsx\'s tab ids', () => {
  const ids = appTabIds();
  assert.equal(ids.length, 14, `expected app.jsx's 14 tabs, got ${ids.length}: ${ids.join(', ')}`);
  for (const tab of Object.keys(TAB_ENDPOINTS)) {
    assert.ok(ids.includes(tab),
      `TAB_ENDPOINTS has tab id ${tab}, which app.jsx does not render: ${ids.join(', ')}`);
  }
});

test('every app.jsx tab is mapped — a tab with no entry is silently unmonitored', () => {
  const ids = appTabIds();
  for (const tab of ids) {
    assert.ok(Object.prototype.hasOwnProperty.call(TAB_ENDPOINTS, tab),
      `app.jsx renders tab ${tab} but TAB_ENDPOINTS has no entry for it, so ` +
      'a wedge behind that tab would show no indicator at all');
  }
});

test('the quality tab polls only /orchestrators, for its project selector', () => {
  // Its own read, /quality, is on demand (data.js::ON_DEMAND_KEYS.quality),
  // never polled: one read costs about a second and carries ?project=.
  assert.deepEqual([...TAB_ENDPOINTS.quality], [`${DASH}/orchestrators`]);
});

test('TAB_ENDPOINTS and each of its entries are frozen', () => {
  // The map crosses module boundaries as the shared declaration of what is
  // polled (data.js::pollSetFor, app.jsx's notices); a consumer that mutated
  // it would silently change what every other consumer sees.
  assert.ok(Object.isFrozen(TAB_ENDPOINTS), 'TAB_ENDPOINTS must be frozen');
  for (const [tab, paths] of Object.entries(TAB_ENDPOINTS)) {
    assert.ok(Object.isFrozen(paths), `TAB_ENDPOINTS[${tab}] must be frozen`);
  }
});

// ── CHROME_ENDPOINTS: what the rail, topbar and toolbar read on EVERY tab ───

test('CHROME_ENDPOINTS is an exported, frozen, duplicate-free array', () => {
  assert.ok(Array.isArray(CHROME_ENDPOINTS), `got ${typeof CHROME_ENDPOINTS}`);
  assert.ok(Object.isFrozen(CHROME_ENDPOINTS), 'CHROME_ENDPOINTS must be frozen');
  assert.equal(new Set(CHROME_ENDPOINTS).size, CHROME_ENDPOINTS.length,
    `CHROME_ENDPOINTS has a duplicate: ${CHROME_ENDPOINTS.join(', ')}`);
});

test('every CHROME_ENDPOINTS entry is a real endpointsFor() path', () => {
  const { endpointsFor, pollKey } = loadDataJs().api;
  const real = new Set(Object.keys(endpointsFor('24h')).map(pollKey));
  for (const p of CHROME_ENDPOINTS) {
    assert.ok(real.has(p),
      `CHROME_ENDPOINTS names ${p}, which is not a pollKey of any ` +
      `endpointsFor('24h') key: ${[...real].join(', ')}`);
  }
});

test('CHROME_ENDPOINTS is exactly the chrome\'s seven endpoints', () => {
  // A literal, not a derivation: test_app_poll_scope.py proves the chrome
  // READS exactly this set; this pins the set itself.
  assert.deepEqual(
    [...CHROME_ENDPOINTS].sort(),
    [
      `${DASH}/orchestrators`, `${DASH}/tasks`, `${DASH}/recon`, `${DASH}/merge-queue`,
      `${DASH}/escalations`, `${DASH}/memory`, `${DASH}/costs`,
    ].sort(),
  );
});
