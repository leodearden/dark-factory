// Tab-scoped polling (task 5825): pollSetFor, scopePollingToTab, and the rail
// badges that must keep updating whichever tab is open. Run via `node --test`;
// dashboard/tests/test_graph_layout_js.py globs it.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

import { loadDataJs, drain } from './_data_loader.mjs';
import { EXPECTED_ENDPOINT_COUNT } from './_data_fixtures.mjs';

const { pollKey } = loadDataJs().api;

// ════════════════════════════════════════════════════════════════════════════
// Scoped polling (task 5825): the browser polls CHROME_ENDPOINTS ∪
// TAB_ENDPOINTS[activeTab], with /tasks narrowed to ?projection=census when the
// open tab renders no task rows. Until app.jsx names a tab, every endpoint is
// polled, which is what every other data.js suite relies on.
// ════════════════════════════════════════════════════════════════════════════

const { TAB_ENDPOINTS, CHROME_ENDPOINTS } = staleness;
const DASH = '/api/v2/dashboard';
const FULL_TASKS_URL = `${DASH}/tasks`;
const CENSUS_TASKS_URL = `${DASH}/tasks?projection=census`;
const SCOPE_TICK_MS = 3000;
const TICKS_PER_MINUTE = 20;

// A fresh data.js with an injected clock that advances one poll interval per
// tick, and a fetch that settles within the same instant — so 5823's slowness
// pacing never engages and every tick is free to poll its whole set.
// `respond(url)` answers each request; it may return a response or a promise.
function scopedLoad(respond = () => ({ ok: true, json: async () => ({}) })) {
  const loaded = loadDataJs();
  const requested = [];
  let t = 1_000_000;
  const opts = {
    state: loaded.api.createPollState(),
    jitterMaxMs: 0,
    deps: {
      fetchImpl: url => {
        requested.push(url);
        return Promise.resolve(respond(url));
      },
      now: () => t,
      random: () => 0,
      sleep: () => Promise.resolve(),
      setTimeoutImpl: () => 0,
      clearTimeoutImpl: () => {},
    },
  };
  async function tick() {
    loaded.api.pollTick(opts);
    await drain();
    t += SCOPE_TICK_MS;
  }
  async function scope(tab, win) {
    await loaded.api.scopePollingToTab(tab, win, opts);
  }
  return { ...loaded, opts, requested, tick, scope, clear: () => { requested.length = 0; } };
}

function requestsByPath(urls) {
  const counts = new Map();
  for (const url of urls) counts.set(pollKey(url), (counts.get(pollKey(url)) || 0) + 1);
  return counts;
}

function scopeOf(tab) {
  return new Set([...TAB_ENDPOINTS[tab], ...CHROME_ENDPOINTS]);
}

test('scope: an unscoped or unknown tab polls every endpoint, exactly as before', async () => {
  const { api } = loadDataJs();
  assert.deepEqual(api.pollSetFor(null, '24h'), api.endpointsFor('24h'));
  assert.deepEqual(api.pollSetFor('no-such-tab', '24h'), api.endpointsFor('24h'));

  const load = scopedLoad();
  await load.tick();
  assert.equal(requestsByPath(load.requested).size, EXPECTED_ENDPOINT_COUNT);
  assert.ok(load.requested.includes(FULL_TASKS_URL), load.requested.join(', '));
});

test('scope: over one simulated minute, every tab polls exactly its own endpoints plus the chrome, 20 times each', async () => {
  const everyPath = Object.keys(loadDataJs().api.endpointsFor('24h')).map(pollKey);
  for (const tab of Object.keys(TAB_ENDPOINTS)) {
    const load = scopedLoad();
    await load.scope(tab);
    load.clear();
    for (let i = 0; i < TICKS_PER_MINUTE; i += 1) await load.tick();

    const counts = requestsByPath(load.requested);
    const inScope = scopeOf(tab);
    for (const path of everyPath) {
      assert.equal(counts.get(path) || 0, inScope.has(path) ? TICKS_PER_MINUTE : 0,
        `tab ${tab}: ${path} was requested ${counts.get(path) || 0} times in a minute`);
    }
    const rowsTab = TAB_ENDPOINTS[tab].includes(FULL_TASKS_URL);
    const tasksUrls = new Set(load.requested.filter(url => pollKey(url) === FULL_TASKS_URL));
    assert.deepEqual([...tasksUrls], [rowsTab ? FULL_TASKS_URL : CENSUS_TASKS_URL],
      `tab ${tab} must poll /tasks ${rowsTab ? 'in full' : 'as the census projection'}`);
  }
});

test('scope: on the Scheduler tab the poll set is exactly the scheduler plus the chrome', async () => {
  const load = scopedLoad();
  await load.scope('scheduler');
  load.clear();
  for (let i = 0; i < TICKS_PER_MINUTE; i += 1) await load.tick();

  assert.deepEqual([...new Set(load.requested)].sort(), [
    `${DASH}/scheduler`,
    `${DASH}/orchestrators`,
    `${DASH}/tasks?projection=census`,
    `${DASH}/recon`,
    `${DASH}/merge-queue?window=24h`,
    `${DASH}/escalations`,
    `${DASH}/memory`,
    `${DASH}/costs?window=24h`,
  ].sort());
});

test('scope change: a newly-needed endpoint is fetched at once, and nothing already watched is', async () => {
  const load = scopedLoad();
  await load.scope('scheduler');
  await load.tick();
  load.clear();
  const before = load.events.length;

  await load.scope('curator');

  assert.deepEqual(load.requested, [`${DASH}/curator`]);
  assert.equal(load.events.length - before, 1, 'one df-data-refresh per scope change');
});

test('scope change: opening a rows tab upgrades the census poll to the full /tasks render at once', async () => {
  const load = scopedLoad();
  await load.scope('scheduler');
  await load.tick();
  load.clear();

  await load.scope('tasks');

  assert.deepEqual(load.requested, [FULL_TASKS_URL, `${DASH}/invariants`]);
});

test('scope change: leaving a rows tab fetches nothing — the full payload already holds the census', async () => {
  const load = scopedLoad();
  await load.scope('tasks');
  await load.tick();
  load.clear();
  const before = load.events.length;

  await load.scope('scheduler');
  assert.deepEqual(load.requested, []);
  assert.equal(load.events.length, before, 'nothing fetched, so no df-data-refresh and no re-render');

  await load.tick();
  assert.ok(load.requested.includes(CENSUS_TASKS_URL), load.requested.join(', '));
  assert.ok(!load.requested.includes(FULL_TASKS_URL), load.requested.join(', '));
});

test('scope change: re-announcing the current tab fetches nothing', async () => {
  const load = scopedLoad();
  await load.scope('scheduler');
  await load.tick();
  load.clear();
  const before = load.events.length;

  await load.scope('scheduler');

  assert.deepEqual(load.requested, []);
  assert.equal(load.events.length, before);
});

test('scope change: the first announcement after the whole-set script-load poll dispatches nothing', async () => {
  const load = scopedLoad();
  await load.tick();
  load.clear();
  const before = load.events.length;

  await load.scope('overview');

  assert.deepEqual(load.requested, []);
  assert.equal(load.events.length, before, 'every endpoint is already watched before the first announcement');
});

test('scope change: a tab switch that resets the window fetches the new tab at the window it will show', async () => {
  // Perf offers 'all', Burndown does not: App resolves windowForTab('burn',
  // 'all') to '24h' and announces both together, so the newly-needed burndown
  // is asked for at 24h rather than at the window the previous tab left.
  const burndown = `${DASH}/burndown`;
  const load = scopedLoad();
  await load.scope('perf', 'all');
  await load.api.refreshDFData('all', load.opts);
  await load.tick();
  load.clear();

  await load.scope('burn', '24h');
  assert.deepEqual(load.requested, [`${burndown}?window=24h`]);

  await load.api.refreshDFData('24h', load.opts);
  await load.tick();
  const windowed = load.requested.filter(url => url.includes('?window='));
  assert.ok(windowed.length > 0, load.requested.join(', '));
  assert.ok(windowed.every(url => url.endsWith('?window=24h')),
    `no request may carry the previous tab's window: ${windowed.join(', ')}`);
});

test('scope change: re-announcing the current tab with a window changes nothing', async () => {
  const load = scopedLoad();
  await load.scope('cost', '24h');
  await load.tick();
  load.clear();

  await load.scope('cost', '7d');
  await load.tick();

  assert.ok(load.requested.every(url => !url.includes('?window=7d')),
    `only a chip change (DF_REFRESH) may move the window of an open tab: ${load.requested.join(', ')}`);
});

test('scope change: a newly-needed endpoint inside its backoff window is not fetched, and its failures stand', async () => {
  const curator = `${DASH}/curator`;
  const load = scopedLoad(url => (
    pollKey(url) === curator
      ? { ok: false, json: async () => ({}) }
      : { ok: true, json: async () => ({}) }
  ));
  await load.api.refreshDFData(undefined, load.opts);
  assert.equal(load.opts.state.get(curator).failures, 1);
  await load.scope('scheduler');
  load.clear();

  await load.scope('curator');

  assert.deepEqual(load.requested, [], 'a tab click must not re-hammer a failing endpoint');
  assert.equal(load.opts.state.get(curator).failures, 1);
});

test('scope change: a newly-needed endpoint already in flight is not requested twice', async () => {
  const curator = `${DASH}/curator`;
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const load = scopedLoad(url => (
    pollKey(url) === curator
      ? gate.then(() => ({ ok: true, json: async () => ({}) }))
      : { ok: true, json: async () => ({}) }
  ));
  load.api.pollTick(load.opts);
  await drain();
  await load.scope('scheduler');
  load.clear();

  await load.scope('curator');

  assert.deepEqual(load.requested, []);
  release();
  await drain();
});

test('scope: a scoped-out endpoint keeps its last good value, receipt and staleness, untouched', async () => {
  const curator = `${DASH}/curator`;
  const load = scopedLoad(url => ({
    ok: true,
    json: async () => (pollKey(url) === curator
      ? { CURATOR_STATE: { pending: [{ id: 'c1' }] }, served_at: '2026-10-03T12:00:00+00:00' }
      : {}),
  }));
  await load.scope('curator');
  await load.tick();
  const data = load.window.DF_DATA;
  const held = data.CURATOR_STATE;
  const receipt = data.__receipt[curator];
  const stale = data.__stale[curator];
  assert.deepEqual(held, { pending: [{ id: 'c1' }] });

  await load.scope('scheduler');
  for (let i = 0; i < TICKS_PER_MINUTE; i += 1) await load.tick();

  assert.equal(data.CURATOR_STATE, held);
  assert.equal(data.__loaded.CURATOR_STATE, true);
  assert.equal(data.__receipt[curator], receipt);
  assert.equal(data.__stale[curator], stale);
});

test('scope: a chip change on a scoped tab re-fetches only the windowed endpoints that tab polls', async () => {
  const load = scopedLoad();
  await load.scope('scheduler');
  await load.tick();
  load.clear();

  await load.api.refreshDFData('7d', load.opts);

  assert.deepEqual(
    load.requested.filter(url => url.includes('?window=')).sort(),
    [`${DASH}/costs?window=7d`, `${DASH}/merge-queue?window=7d`],
  );
  for (const path of [`${DASH}/performance`, `${DASH}/burndown`]) {
    assert.ok(!load.requested.some(url => pollKey(url) === path), `${path} is out of scope`);
  }
});

// ── THE REGRESSION the scope must not cause: the rail badges go on updating ──
//
// The rail is mounted on every tab, so its five readings must track the
// latest payload served for their endpoints whichever tab is open. Run with
// the SAME readers app.jsx's railCounts calls, over the DF_DATA the scoped
// loop fills.

const RAIL_READER_CHAIN = ['recon_status.js', 'task_vocab.js', 'task_snapshot.js', 'merge_queue.js']
  .map(name => `../../src/dashboard/static/redux/${name}`);

function installRailReaders() {
  const require = createRequire(import.meta.url);
  for (const specifier of RAIL_READER_CHAIN) {
    delete require.cache[require.resolve(specifier)];
    require(specifier);
  }
  const { DF_TASK_SNAPSHOT, DF_MERGE_QUEUE, DF_RECON_STATUS } = globalThis.window;
  return { ...DF_TASK_SNAPSHOT, ...DF_MERGE_QUEUE, ...DF_RECON_STATUS };
}

function freshWireDatum(value, asOf) {
  return { value, as_of: asOf, state: 'fresh', reason: null, freshness_bound_seconds: 30 };
}

// A stub server whose chrome payloads are a function of how many times their
// endpoint has been served, so a reading that stopped updating is visible.
function railServer() {
  const served = new Map();
  const respond = url => {
    const path = pollKey(url);
    const n = (served.get(path) || 0) + 1;
    served.set(path, n);
    const servedAt = new Date(Date.now() - 1000).toISOString();
    const census = {
      counts: {
        'in-progress': n, blocked: 0, review: 0, 'merge-deferred': 0, 'infra-hold': 0,
        pending: 0, deferred: 0, done: 0, cancelled: 0,
      },
      total: n,
      views: { in_flight: n, backlog: 0, terminal: 0 },
      sub_views: { running: n },
    };
    const rows = url === CENSUS_TASKS_URL
      ? { value: null, as_of: null, state: 'unknown', reason: 'rows not requested: this poll asked /tasks for the census only (?projection=census)', freshness_bound_seconds: 30 }
      : freshWireDatum([{ id: 'df/T-1', project: 'df', title: 't', status: 'in-progress' }], servedAt);
    const bodies = {
      [`${DASH}/orchestrators`]: {
        ORCHESTRATORS: [...Array.from({ length: n }, () => ({ running: true })), { running: false }],
        PROJECTS: [], ORCHESTRATORS_SPARK: { labels: [], values: [] },
      },
      [FULL_TASKS_URL]: {
        TASKS_SNAPSHOT: {
          df: { census: freshWireDatum(census, servedAt), rows, in_progress_live: 0, in_progress_stranded: 0, skew_seconds: 0 },
        },
        TASKS_OFFLINE: false, TASKS_OFFLINE_PROJECTS: [], TASKS_DEGRADED_PROJECTS: [],
        TASKS_COUNT_UNKNOWN_PROJECTS: [], TASKS_PROJECT_COUNT: 1,
      },
      [`${DASH}/recon`]: {
        RECON_STATE: { runs: Array.from({ length: n }, () => ({ status: 'failed' })) },
        AGENTS: [],
      },
      [`${DASH}/merge-queue`]: {
        MERGE_QUEUE: {
          df: {
            in_queue: freshWireDatum(n, servedAt), live_probe_configured: true,
            active: [], active_spark: { labels: [], values: [] },
          },
        },
      },
      [`${DASH}/escalations`]: {
        ESCALATIONS: { subsections: [], summary: { by_status: { pending: n } } },
      },
    };
    const body = { ...(bodies[path] || {}), served_at: servedAt };
    return { ok: true, json: async () => body };
  };
  return { respond, served };
}

test('THE REGRESSION: the rail badges track the latest served payload on EVERY tab', async () => {
  for (const tab of Object.keys(TAB_ENDPOINTS)) {
    const server = railServer();
    const load = scopedLoad(server.respond);
    const readers = installRailReaders();
    // The script-load first poll, unscoped, then the tab App announces.
    await load.api.refreshDFData(undefined, load.opts);
    await load.scope(tab);
    load.clear();
    await load.tick();
    await load.tick();

    const data = load.window.DF_DATA;
    const latest = path => server.served.get(`${DASH}/${path}`);
    const census = readers.censusOver(data, null);
    assert.notEqual(census.state, 'unknown', `tab ${tab}: ${census.reason}`);
    const readings = {
      orch: data.ORCHESTRATORS.filter(o => o.running).length,
      tasks: readers.inFlightCount(census.value),
      recon: readers.reconAttentionCount(readers.reconRunCounts(data.RECON_STATE.runs)),
      merge: readers.inQueueOver(data, null).value,
      esc: data.ESCALATIONS.summary.by_status.pending,
    };
    assert.deepEqual(readings, {
      orch: latest('orchestrators'),
      tasks: String(latest('tasks')),
      recon: latest('recon'),
      merge: latest('merge-queue'),
      esc: latest('escalations'),
    }, `tab ${tab}: a rail badge stopped tracking its endpoint`);
    assert.equal(latest('orchestrators'), 3, `tab ${tab}: the chrome must be polled on every tick`);

    const rowsTab = TAB_ENDPOINTS[tab].includes(FULL_TASKS_URL);
    assert.equal(load.requested.includes(FULL_TASKS_URL), rowsTab,
      `tab ${tab}: the full /tasks render is for rows tabs only`);
  }
});
