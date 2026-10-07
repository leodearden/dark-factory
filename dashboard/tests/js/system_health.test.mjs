// Module-contract tests for system_health.js: the Overview's System health
// panel status decisions. Each row's tone comes from a served field, and the
// header comes from the rows. tab_overview.jsx only spreads these rows; its
// WIRING is pinned structurally in Python (test_tab_overview.py
// TestSystemHealthIsDerived).
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order, exactly as
// memory_readings.test.mjs does. data.js rides along only so the pre-fetch case
// reads its real seed; with no `document` it never starts polling. The shim is
// REMOVED once the modules are loaded, and every fixture carries its own
// `__receipt` map, so a reader that reached for a browser global would throw
// here instead of passing by accident.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'data.js', 'tasks_offline_banner.js', 'system_health.js'].map(
  name => REDUX + name,
);

function loadSystemHealth() {
  globalThis.window = { DF_ENDPOINT_STALENESS: staleness, dispatchEvent() {} };
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [datumApi, , bannerApi, healthApi] = LOAD_CHAIN.map(specifier => require(specifier));
  const seed = globalThis.window.DF_DATA;
  delete globalThis.window;
  return { datum: datumApi, banner: bannerApi, health: healthApi, seed };
}

const { datum, banner, health, seed } = loadSystemHealth();
const {
  graphitiHealth,
  mem0Health,
  taskStoreHealth,
  fusedMemoryHealth,
  healthTone,
  healthSummary,
} = health;
const { EM_DASH } = datum;
const { tasksBannerNoticesFor } = banner;

// ── Fixtures ────────────────────────────────────────────────────────────────

const MEMORY_ENDPOINT = '/api/v2/dashboard/memory';
const TASKS_ENDPOINT = '/api/v2/dashboard/tasks';
const RECEIPT = Object.freeze({ servedAt: '2026-10-07T12:00:30+00:00', receivedAt: 1_800_000_000_000 });
const DELIVERED = Object.freeze({ [MEMORY_ENDPOINT]: RECEIPT, [TASKS_ENDPOINT]: RECEIPT });

const UNMEASURED = Object.freeze({ sub: EM_DASH, ok: true, warn: true, title: 'not yet fetched' });

function memoryData(memoryStatus, receipts = DELIVERED) {
  return { MEMORY_STATUS: memoryStatus, __receipt: receipts };
}

function tasksData(tasksKeys, receipts = DELIVERED) {
  return { ...tasksKeys, __receipt: receipts };
}

const STORE_HEALTH = [
  ['graphiti', graphitiHealth],
  ['mem0', mem0Health],
];

// ── graphitiHealth / mem0Health ─────────────────────────────────────────────

for (const [key, storeHealth] of STORE_HEALTH) {
  test(`${key}: before /memory has delivered the row is unmeasured — amber, never green or red`, () => {
    const data = memoryData({ [key]: { connected: null } }, {});
    assert.deepEqual(storeHealth(data), UNMEASURED);
  });

  test(`${key}: connected === true is green`, () => {
    const data = memoryData({ [key]: { connected: true } });
    assert.deepEqual(storeHealth(data), { sub: 'connected', ok: true, warn: false });
  });

  test(`${key}: connected === false is red, titled with the served error`, () => {
    const data = memoryData({ [key]: { connected: false, error: 'qdrant down' } });
    assert.deepEqual(storeHealth(data), { sub: 'not connected', ok: false, warn: false, title: 'qdrant down' });
  });

  test(`${key}: fused-memory unreachable leaves it unmeasured, and the title carries why`, () => {
    const data = memoryData({ [key]: { connected: null }, offline: true, error: 'ConnectError' });
    const row = storeHealth(data);
    assert.equal(row.sub, EM_DASH);
    assert.equal(row.ok, true);
    assert.equal(row.warn, true);
    assert.ok(row.title.includes('ConnectError'), row.title);
  });

  test(`${key}: an absent connected flag on a live payload is unmeasured, with a reason`, () => {
    const row = storeHealth(memoryData({ [key]: {} }));
    assert.equal(row.sub, EM_DASH);
    assert.equal(row.ok, true);
    assert.equal(row.warn, true);
    assert.equal(typeof row.title, 'string');
    assert.ok(row.title.length > 0);
  });

  test(`${key}: a MEMORY_STATUS lacking the store entry entirely does not throw`, () => {
    const row = storeHealth(memoryData({}));
    assert.equal(healthTone(row), 'warn');
  });
}

// ── taskStoreHealth ─────────────────────────────────────────────────────────

test('taskStore: before /tasks has delivered the row is unmeasured', () => {
  assert.deepEqual(taskStoreHealth(tasksData({ TASKS_PROJECT_COUNT: 3 }, {})), UNMEASURED);
});

test('taskStore: a global outage is red, titled with the banner\'s own global notice', () => {
  const data = tasksData({ TASKS_OFFLINE: true, TASKS_OFFLINE_PROJECTS: ['a'], TASKS_PROJECT_COUNT: 2 });
  const [globalNotice] = tasksBannerNoticesFor(data);
  assert.equal(globalNotice.kind, 'global');
  assert.deepEqual(taskStoreHealth(data), {
    sub: 'unreachable',
    ok: false,
    warn: false,
    title: globalNotice.text,
  });
});

test('taskStore: per-root failures are amber, the sub naming each notice kind and the title each text', () => {
  const data = tasksData({
    TASKS_OFFLINE: false,
    TASKS_OFFLINE_PROJECTS: ['a'],
    TASKS_DEGRADED_PROJECTS: ['b'],
    TASKS_PROJECT_COUNT: 9,
  });
  const notices = tasksBannerNoticesFor(data);
  assert.deepEqual(taskStoreHealth(data), {
    sub: 'partial · degraded',
    ok: true,
    warn: true,
    title: notices.map(n => n.text).join('\n'),
  });
});

test('taskStore: an unmeasured done count alone is amber', () => {
  const row = taskStoreHealth(tasksData({ TASKS_COUNT_UNKNOWN_PROJECTS: ['c'], TASKS_PROJECT_COUNT: 4 }));
  assert.equal(row.sub, 'count-unknown');
  assert.equal(healthTone(row), 'warn');
});

test('taskStore: a clean fan-out is green and counts the projects answering', () => {
  assert.deepEqual(taskStoreHealth(tasksData({ TASKS_PROJECT_COUNT: 3 })), {
    sub: '3 projects answering',
    ok: true,
    warn: false,
  });
  assert.equal(taskStoreHealth(tasksData({ TASKS_PROJECT_COUNT: 1 })).sub, '1 project answering');
});

test('taskStore: a clean fan-out over no projects is amber, not green', () => {
  const row = taskStoreHealth(tasksData({ TASKS_PROJECT_COUNT: 0 }));
  assert.equal(row.sub, 'no task projects');
  assert.equal(row.ok, true);
  assert.equal(row.warn, true);
});

test('taskStore: no sub invents a transport ("mcp")', () => {
  const cases = [
    tasksData({}, {}),
    tasksData({ TASKS_OFFLINE: true, TASKS_PROJECT_COUNT: 2 }),
    tasksData({ TASKS_DEGRADED_PROJECTS: ['b'], TASKS_PROJECT_COUNT: 2 }),
    tasksData({ TASKS_PROJECT_COUNT: 0 }),
    tasksData({ TASKS_PROJECT_COUNT: 2 }),
  ];
  for (const data of cases) {
    const { sub } = taskStoreHealth(data);
    assert.ok(!sub.toLowerCase().includes('mcp'), sub);
  }
});

// ── fusedMemoryHealth: tone and title only ──────────────────────────────────

test('fusedMemory: before /memory has delivered it is unmeasured, and carries no sub', () => {
  assert.deepEqual(fusedMemoryHealth(memoryData({ uptime_seconds: null }, {})), {
    ok: true,
    warn: true,
    title: 'not yet fetched',
  });
});

test('fusedMemory: offline is red, titled with the served error', () => {
  const row = fusedMemoryHealth(memoryData({ offline: true, error: 'ConnectError: refused' }));
  assert.equal(row.ok, false);
  assert.equal(row.title, 'ConnectError: refused');
  assert.equal(healthTone(row), 'bad');
});

test('fusedMemory: answering is green, titled with when it started', () => {
  const row = fusedMemoryHealth(memoryData({ offline: false, started_at: '2026-10-07T08:00:00+00:00' }));
  assert.equal(healthTone(row), 'ok');
  assert.equal(row.title, '2026-10-07T08:00:00+00:00');
});

test('fusedMemory: never returns a sub, so spreading it after the JSX\'s own sub cannot overwrite it', () => {
  const cases = [
    memoryData({}, {}),
    memoryData({ offline: true, error: 'x' }),
    memoryData({ offline: false, started_at: '2026-10-07T08:00:00+00:00' }),
  ];
  for (const data of cases) assert.ok(!('sub' in fusedMemoryHealth(data)));
});

// ── healthTone / healthSummary ──────────────────────────────────────────────

test('healthTone: red outranks amber; amber needs ok; green needs neither fault', () => {
  assert.equal(healthTone({ ok: false, warn: true }), 'bad');
  assert.equal(healthTone({ ok: false, warn: false }), 'bad');
  assert.equal(healthTone({ ok: true, warn: true }), 'warn');
  assert.equal(healthTone({ ok: true, warn: false }), 'ok');
});

const GREEN = { ok: true, warn: false };
const AMBER = { ok: true, warn: true };
const RED = { ok: false, warn: false };

test('healthSummary: all green reads "all ok"', () => {
  assert.equal(healthSummary([GREEN, GREEN, GREEN]), 'all ok');
});

test('healthSummary: counts the red and amber rows, red first', () => {
  assert.equal(healthSummary([AMBER, GREEN, RED, AMBER]), '1 bad · 2 warn');
});

test('healthSummary: amber alone names only the amber count', () => {
  assert.equal(healthSummary([GREEN, AMBER, AMBER]), '2 warn');
});

test('healthSummary: red alone names only the red count', () => {
  assert.equal(healthSummary([GREEN, RED]), '1 bad');
});

test('healthSummary: no rows is a hole, not "all ok"', () => {
  assert.equal(healthSummary([]), EM_DASH);
});

test('healthSummary: the pre-fetch seed is NOT "all ok" — nothing has been measured yet', () => {
  assert.deepEqual(seed.__receipt, {}, 'precondition: the seed carries no receipt');
  const rows = [graphitiHealth(seed), mem0Health(seed), taskStoreHealth(seed), fusedMemoryHealth(seed)];
  for (const row of rows) assert.equal(healthTone(row), 'warn', JSON.stringify(row));
  assert.notEqual(healthSummary(rows), 'all ok');
});
