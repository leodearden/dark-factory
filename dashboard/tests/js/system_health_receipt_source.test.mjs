// system_health.js's LOAD CONTRACT, pinned by behaviour: every row reads its
// receipts from the `data.__receipt` it was handed, never from the browser's
// window.DF_DATA. datum.js reads that global whenever a receipts map is
// undefined, and it does so silently, so a row that lost its own default would
// answer for the global's endpoints instead of its payload's.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'tasks_offline_banner.js', 'system_health.js'].map(name => REDUX + name);

function loadSystemHealth() {
  globalThis.window = { DF_ENDPOINT_STALENESS: staleness };
  const require = createRequire(import.meta.url);
  const healthApi = LOAD_CHAIN.map(specifier => require(specifier)).at(-1);
  delete globalThis.window;
  return healthApi;
}

const {
  graphitiHealth, mem0Health, taskStoreHealth, fusedMemoryHealth, reconHealth, walHealth, healthTone,
} = loadSystemHealth();

const RECEIPT = Object.freeze({ servedAt: '2026-10-07T12:00:30+00:00', receivedAt: 1_800_000_000_000 });
const DELIVERED = Object.freeze({
  '/api/v2/dashboard/memory': RECEIPT,
  '/api/v2/dashboard/tasks': RECEIPT,
  '/api/v2/dashboard/recon': RECEIPT,
});

// Every row reads green once its endpoint has delivered.
const GREEN_PAYLOAD = Object.freeze({
  MEMORY_STATUS: {
    graphiti: { connected: true },
    mem0: { connected: true },
    started_at: '2026-10-07T11:00:00+00:00',
    wal: { status: 'ok', reason: null, rows: [{}] },
  },
  TASKS_PROJECT_COUNT: 3,
  RECON_STATE: { verdict: { severity: 'none', action_taken: 'none' } },
});

const ROWS = [
  ['graphiti', graphitiHealth],
  ['mem0', mem0Health],
  ['taskStore', taskStoreHealth],
  ['fusedMemory', fusedMemoryHealth],
  ['recon', reconHealth],
  ['wal', walHealth],
];

function withBrowserReceipts(receipts, body) {
  globalThis.window = { DF_DATA: { __receipt: receipts } };
  try {
    return body();
  } finally {
    delete globalThis.window;
  }
}

for (const [key, rowHealth] of ROWS) {
  test(`${key}: the payload's own receipts make it green`, () => {
    assert.equal(healthTone(rowHealth({ ...GREEN_PAYLOAD, __receipt: DELIVERED })), 'ok');
  });

  test(`${key}: a payload with no receipts is unmeasured, however delivered the browser global says it is`, () => {
    const row = withBrowserReceipts(DELIVERED, () => rowHealth(GREEN_PAYLOAD));
    assert.equal(row.title, 'not yet fetched');
    assert.equal(healthTone(row), 'warn');
  });
}
