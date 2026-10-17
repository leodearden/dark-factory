// Module-contract tests for system_health.js: the Overview's System health
// panel status decisions. Each row's tone comes from a served field, and the
// header comes from the rows. tab_overview.jsx only spreads these rows (the
// Write queue row's from memory_readings.js::queueHealth); its WIRING is
// pinned structurally in Python (test_tab_overview.py TestSystemHealthIsDerived).
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
const LOAD_CHAIN = [
  'datum.js', 'data.js', 'tasks_offline_banner.js', 'memory_readings.js', 'system_health.js',
].map(name => REDUX + name);

function loadSystemHealth() {
  globalThis.window = { DF_ENDPOINT_STALENESS: staleness, dispatchEvent() {} };
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [datumApi, , bannerApi, readingsApi, healthApi] = LOAD_CHAIN.map(specifier => require(specifier));
  const seed = globalThis.window.DF_DATA;
  delete globalThis.window;
  return { datum: datumApi, banner: bannerApi, readings: readingsApi, health: healthApi, seed };
}

const { datum, banner, readings, health, seed } = loadSystemHealth();
const {
  graphitiHealth,
  mem0Health,
  taskStoreHealth,
  fusedMemoryHealth,
  reconHealth,
  walHealth,
  healthTone,
  healthSummary,
} = health;
const { EM_DASH } = datum;
const { tasksBannerNoticesFor } = banner;
const { writeQueue, queueHealth } = readings;

// ── Fixtures ────────────────────────────────────────────────────────────────

const MEMORY_ENDPOINT = '/api/v2/dashboard/memory';
const TASKS_ENDPOINT = '/api/v2/dashboard/tasks';
const RECON_ENDPOINT = '/api/v2/dashboard/recon';
const RECEIPT = Object.freeze({ servedAt: '2026-10-07T12:00:30+00:00', receivedAt: 1_800_000_000_000 });
const DELIVERED = Object.freeze({
  [MEMORY_ENDPOINT]: RECEIPT,
  [TASKS_ENDPOINT]: RECEIPT,
  [RECON_ENDPOINT]: RECEIPT,
});

const UNMEASURED = Object.freeze({ sub: EM_DASH, ok: true, warn: true, title: 'not yet fetched' });

function memoryData(memoryStatus, receipts = DELIVERED) {
  return { MEMORY_STATUS: memoryStatus, __receipt: receipts };
}

function tasksData(tasksKeys, receipts = DELIVERED) {
  return { ...tasksKeys, __receipt: receipts };
}

function reconData(verdict, receipts = DELIVERED) {
  return { RECON_STATE: { verdict }, __receipt: receipts };
}

function walData(wal, receipts = DELIVERED) {
  return memoryData({ wal }, receipts);
}

function assertUnmeasuredWithReason(row) {
  assert.equal(row.sub, EM_DASH);
  assert.equal(healthTone(row), 'warn');
  assert.equal(typeof row.title, 'string');
  assert.ok(row.title.length > 0);
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

// ── reconHealth: the newest judge verdict ───────────────────────────────────

test('recon: before /recon has delivered the row is unmeasured', () => {
  assert.deepEqual(reconHealth(reconData(null, {})), UNMEASURED);
});

test('recon: a delivered payload with no verdict is unmeasured, not green', () => {
  assertUnmeasuredWithReason(reconHealth(reconData(null)));
});

test('recon: a clean verdict is green and states the severity and action', () => {
  assert.deepEqual(reconHealth(reconData({ severity: 'none', action_taken: 'none', is_phantom: false })), {
    sub: 'verdict: none · none',
    ok: true,
    warn: false,
  });
});

test('recon: a minor verdict is amber', () => {
  const row = reconHealth(reconData({ severity: 'minor', action_taken: 'logged', is_phantom: false }));
  assert.equal(row.sub, 'verdict: minor · logged');
  assert.equal(healthTone(row), 'warn');
});

test('recon: a serious verdict is red', () => {
  const row = reconHealth(reconData({ severity: 'serious', action_taken: 'halt', is_phantom: false }));
  assert.equal(row.sub, 'verdict: serious · halt');
  assert.equal(healthTone(row), 'bad');
});

test('recon: a phantom verdict is amber and says unreviewed, never "serious"', () => {
  const row = reconHealth(reconData({ severity: 'serious', action_taken: 'halt', is_phantom: true }));
  assert.equal(healthTone(row), 'warn');
  assert.ok(row.sub.includes('unreviewed'), row.sub);
  assert.ok(!row.sub.includes('serious'), row.sub);
  assert.ok(row.sub.endsWith('· halt'), row.sub);
});

// ── walHealth: the SQLite WAL panel status /memory serves ───────────────────

test('wal: before /memory has delivered the row is unmeasured', () => {
  assert.deepEqual(walHealth(walData({ status: 'offline', reason: null, rows: [] }, {})), UNMEASURED);
});

test('wal: ok over measured stores is green and counts them', () => {
  assert.deepEqual(walHealth(walData({ status: 'ok', reason: null, rows: [{}, {}] })), {
    sub: '2 stores · all current',
    ok: true,
    warn: false,
  });
  assert.equal(walHealth(walData({ status: 'ok', reason: null, rows: [{}] })).sub, '1 store · all current');
});

test('wal: ok over no stores is unmeasured, not green', () => {
  assertUnmeasuredWithReason(walHealth(walData({ status: 'ok', reason: null, rows: [] })));
});

test('wal: warn is amber and red is red, each stating the served reason', () => {
  assert.deepEqual(walHealth(walData({ status: 'warn', reason: 'main: log=6000 frames', rows: [{}] })), {
    sub: 'main: log=6000 frames',
    ok: true,
    warn: true,
  });
  assert.deepEqual(walHealth(walData({ status: 'red', reason: 'main: busy=1', rows: [{}] })), {
    sub: 'main: busy=1',
    ok: false,
    warn: false,
  });
});

test('wal: an unreachable WAL probe is unmeasured, never red, and the title carries why', () => {
  const row = walHealth(walData({ status: 'offline', reason: 'ConnectError: refused', rows: [] }));
  assertUnmeasuredWithReason(row);
  assert.equal(row.title, 'ConnectError: refused');
});

test('wal: a MEMORY_STATUS lacking the wal block entirely does not throw', () => {
  assertUnmeasuredWithReason(walHealth(memoryData({})));
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

test('healthSummary: every row of the pre-fetch seed is amber — nothing has been measured yet', () => {
  assert.deepEqual(seed.__receipt, {}, 'precondition: the seed carries no receipt');
  const rows = [
    graphitiHealth(seed),
    mem0Health(seed),
    taskStoreHealth(seed),
    fusedMemoryHealth(seed),
    queueHealth(writeQueue(seed)),
    reconHealth(seed),
    walHealth(seed),
  ];
  for (const row of rows) assert.equal(healthTone(row), 'warn', JSON.stringify(row));
  assert.equal(healthSummary(rows), rows.length + ' warn');
});
