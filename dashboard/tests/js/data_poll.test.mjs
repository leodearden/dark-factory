// Module-contract tests for data.js — the real-data poll loader for the Dark
// Factory dashboard (window.DF_DATA, the 3s setInterval poll loop, and —
// this task — a per-endpoint in-flight guard, error backoff, and jitter on
// top of it). Run via `node --test` (see dashboard/tests/test_graph_layout_js.py
// for the pytest wrapper that surfaces this suite in CI — it globs
// **/*.test.mjs under dashboard/tests/js/, so this new file needs no new
// wrapper).
//
// data.js has no package.json in the repo, so it resolves as CommonJS
// (`module.exports = <object>`), same as the other redux/*.js modules.
//
// Unlike those siblings, a static ESM `import` of data.js is not an option
// here, not even for the module-contract check: data.js assigns
// `window.DF_DATA = {...}` at module scope, and an ESM `import` statement's
// target module body runs BEFORE the importing file's own body — so
// `globalThis.window` would still be unset when data.js's top level runs,
// throwing `ReferenceError: window is not defined`. Every test in this file
// therefore goes through the `loadDataJs()` helper below, which shims
// `globalThis.window` FIRST and only then loads the module via
// `createRequire(import.meta.url)` (mirrors runtime_format.test.mjs:33-45).
//
// data.js's module scope also unconditionally called `refreshDFData()` and
// `setInterval(...)`. Under node, with no shim at all, that would throw
// before even reaching the interval; but even with window/fetch shimmed it
// would fire real (stubbed) fetches and leave a live timer holding the
// process open — hanging `node --test`. loadDataJs() defends against that
// unconditionally (see its comment below) so this suite can safely run
// against pre-seam data.js too, which is what step-1's RED depends on.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/data.js';
const EXPECTED_FUNCTION_NAMES = [
  'endpointsFor',
  'applyKey',
  'refreshOne',
  'refreshDFData',
  'pollTick',
  'startPolling',
  'createPollState',
];

// Full DF_DATA key set (data.js:41-127) — initialised so the first render
// before fetch completes cannot crash any component reading DF_DATA.*.
const EXPECTED_DF_DATA_KEYS = [
  'PROJECTS', 'AGENTS', 'ORCHESTRATORS', 'ORCHESTRATORS_SPARK',
  'ACTIVE_TASKS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS', 'DONE_COUNTS',
  'PERFORMANCE', 'MEMORY_STATUS', 'MEMORY_TIMESERIES', 'MEMORY_OPS_BREAKDOWN',
  'RECON_STATE', 'MERGE_QUEUE', 'COSTS', 'BURNDOWN', 'BURNDOWN_BY_PROJECT',
  'CURATOR_STATE', 'ESCALATIONS', 'ESCALATION_ANALYTICS', 'SCHEDULER',
];

// Loads data.js fresh against a shimmed browser-ish global. Installs
// `globalThis.window` (a bare object recording dispatched events) and a
// counting `globalThis.fetch` BEFORE requiring, then busts the require
// cache so each call re-executes data.js's module body from scratch — the
// module body only runs once per require otherwise, which would leave later
// callers seeing a stale `window.DF_DATA` / stale flow-control singleton
// from a previous test's shim (mirrors runtime_format.test.mjs:33-49).
//
// Deliberately does NOT set `globalThis.document`: index.html loads data.js
// as a classic script where `document` always exists, but every test in
// this file runs under node, so the auto-start guard must see no `document`
// and stay inert, exactly like the real non-browser (node --test)
// environment.
//
// Also wraps `globalThis.setInterval` for the duration of the require and
// clears any interval it captures before returning. This is defensive
// rather than load-bearing for the seamed implementation (which gates
// auto-start on `document` and never calls setInterval here at all), but it
// means step-1's RED run — against pre-seam data.js, which calls
// setInterval unconditionally — cannot hang node --test: the interval is
// recorded (so the "no live timer" assertion still fails honestly) and then
// cleared immediately, regardless of what the caller asserts.
function loadDataJs({ fetchStub } = {}) {
  const events = [];
  const fetchCalls = [];
  const intervalCalls = [];
  const win = { dispatchEvent: ev => events.push(ev) };
  globalThis.window = win;
  globalThis.fetch = (url, init) => {
    fetchCalls.push({ url, init });
    if (fetchStub) return fetchStub(url, init);
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };

  const require = createRequire(import.meta.url);
  const resolved = require.resolve(MODULE_SPECIFIER);
  delete require.cache[resolved];

  const originalSetInterval = globalThis.setInterval;
  globalThis.setInterval = (...args) => {
    const handle = originalSetInterval(...args);
    intervalCalls.push(handle);
    return handle;
  };

  let api;
  try {
    api = require(MODULE_SPECIFIER);
  } finally {
    globalThis.setInterval = originalSetInterval;
    for (const handle of intervalCalls) clearInterval(handle);
  }

  return { api, window: win, events, fetchCalls, intervalCalls };
}

// Resolves after the entire pending microtask queue has drained (node
// always fully drains microtasks before running the next macrotask/
// immediate), regardless of how many .then()/await hops a chain needs — so
// awaiting this once after a pollTick() call is enough to let every
// non-held endpoint's fetch -> resp.json() -> applyKey chain run to
// completion before the next tick fires.
function drain() {
  return new Promise(resolve => setImmediate(resolve));
}

// The endpoint the motivating incident hung on (measured 108s response) —
// shared by both in-flight-guard regression tests below.
const SLOW_ENDPOINT_PATH = '/api/v2/dashboard/memory-graphs';

// Builds a fetchImpl that holds `slowPath` open on a manually-settled gate
// while every other endpoint resolves immediately with a valid, empty-bodied
// JSON response. Records, per endpoint PATH (query string stripped, so the
// four ?window= endpoints are tracked the same way production flow-control
// state will key them), a live concurrency counter (incremented on entry,
// decremented when that call's own promise settles) and its running max,
// plus a total call count.
//
// This is installed as BOTH the counting `globalThis.fetch` (via
// loadDataJs({fetchStub})) and `deps.fetchImpl`: today refreshOne only ever
// reaches it through the global-fetch path (deps isn't wired in until
// step-4), while the post-step-4 implementation reaches the identical stub
// through deps.fetchImpl — so this one fixture stays valid across both.
function makeConcurrencyFetch(slowPath) {
  const live = new Map();
  const maxConcurrent = new Map();
  const callCount = new Map();
  let releaseSlow;
  let rejectSlow;
  const slowGate = new Promise((resolve, reject) => {
    releaseSlow = resolve;
    rejectSlow = reject;
  });

  function fetchImpl(url) {
    const path = url.split('?')[0];
    const n = (live.get(path) || 0) + 1;
    live.set(path, n);
    maxConcurrent.set(path, Math.max(maxConcurrent.get(path) || 0, n));
    callCount.set(path, (callCount.get(path) || 0) + 1);
    const settle = () => live.set(path, live.get(path) - 1);

    if (path === slowPath) {
      return slowGate.then(
        () => { settle(); return { ok: true, json: async () => ({}) }; },
        err => { settle(); throw err; },
      );
    }
    return Promise.resolve({ ok: true, json: async () => ({}) }).then(resp => { settle(); return resp; });
  }

  return { fetchImpl, maxConcurrent, callCount, releaseSlow, rejectSlow };
}

// A plain, unwindowed endpoint used as the "flaky" one in the error-backoff
// tests below — distinct from SLOW_ENDPOINT_PATH so those two concerns
// (in-flight concurrency vs. failure backoff) stay independently testable.
const FLAKY_ENDPOINT_PATH = '/api/v2/dashboard/curator';

// Every endpoint PATH from endpointsFor(win) except `excludePath` — used to
// assert that a failure/backoff on one endpoint never affects the other 12.
function otherPaths(api, win, excludePath) {
  return Object.keys(api.endpointsFor(win)).map(url => url.split('?')[0]).filter(p => p !== excludePath);
}

test('default-imported module exposes the poll-loader functions', () => {
  const { api } = loadDataJs();
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof api[name], 'function', `api.${name} should be a function`);
  }
});

test('module also assigns window.DF_REFRESH / window.__DF_PAUSE / window.DF_DATA (browser dual-export)', () => {
  const { api, window: win } = loadDataJs();

  assert.equal(win.DF_REFRESH, api.refreshDFData, 'window.DF_REFRESH should be the exported refreshDFData');
  assert.equal(win.__DF_PAUSE, false, 'window.__DF_PAUSE should default to false');

  assert.ok(win.DF_DATA, 'window.DF_DATA was not set');
  for (const key of EXPECTED_DF_DATA_KEYS) {
    assert.ok(
      Object.prototype.hasOwnProperty.call(win.DF_DATA, key),
      `DF_DATA.${key} was not initialised`,
    );
  }
});

test('no auto-start outside a browser: loading with no `document` global fires zero fetches and leaves no live timer', () => {
  assert.equal(typeof globalThis.document, 'undefined', 'test environment must not already define document');

  const { fetchCalls, intervalCalls } = loadDataJs();

  assert.equal(fetchCalls.length, 0, 'loading data.js under node must not fire any fetches');
  assert.equal(intervalCalls.length, 0, 'loading data.js under node must not start a live timer');
});

// ---------------------------------------------------------------------------
// Per-endpoint in-flight guard — THE REQUIRED REGRESSION TEST.
//
// Motivating incident: memory-graphs measured a 108s response against a 3s
// poll interval, so an unguarded loader stacks ~36 concurrent requests for
// that one endpoint by the time it finally answers. This drives four tick
// fires while memory-graphs is held open and asserts it never exceeds 1
// concurrent fetch, while the other 12 endpoints are completely unaffected
// (proving the guard is per-endpoint, not whole-cycle — a whole-cycle guard
// would starve all 13 endpoints on any single slow one).
// ---------------------------------------------------------------------------

test('per-endpoint in-flight guard: a slow endpoint never exceeds concurrency 1, and does not block the other 12 endpoints (regression)', async () => {
  const { fetchImpl, maxConcurrent, callCount, releaseSlow } = makeConcurrencyFetch(SLOW_ENDPOINT_PATH);
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  const allPaths = Object.keys(api.endpointsFor('24h')).map(url => url.split('?')[0]);
  assert.equal(allPaths.length, 13, 'expected 13 endpoints (sanity check on the endpointsFor fixture)');
  assert.ok(allPaths.includes(SLOW_ENDPOINT_PATH), 'fixture must include the memory-graphs endpoint');
  const fastPaths = allPaths.filter(p => p !== SLOW_ENDPOINT_PATH);

  const state = api.createPollState();
  const deps = {
    fetchImpl,
    now: () => Date.now(),
    random: () => Math.random(),
    sleep: ms => new Promise(resolve => setTimeout(resolve, ms)),
  };
  const opts = { state, deps, jitterMaxMs: 0 };

  // Four 3s interval fires landing inside one still-pending slow response —
  // the slow endpoint is deliberately never released across this loop.
  for (let i = 0; i < 4; i++) {
    api.pollTick(opts);
    await drain();
  }

  assert.equal(
    maxConcurrent.get(SLOW_ENDPOINT_PATH),
    1,
    `a slow endpoint must never have more than 1 concurrent fetch in flight (observed ${maxConcurrent.get(SLOW_ENDPOINT_PATH)})`,
  );
  // The guard should have actually SKIPPED ticks 2-4 for the slow endpoint
  // (not queued them) — only tick 1's fetch ever went out.
  assert.equal(
    callCount.get(SLOW_ENDPOINT_PATH),
    1,
    'ticks 2-4 must skip the slow endpoint outright while it is still in flight, not queue a retry',
  );

  for (const path of fastPaths) {
    assert.equal(
      callCount.get(path),
      4,
      `${path} should have completed all 4 fetches — one slow endpoint must not block the other 12`,
    );
  }

  // The flag clears on a successful settle: releasing the slow endpoint lets
  // the very next tick re-fetch it instead of skipping it forever.
  releaseSlow();
  await drain();
  api.pollTick(opts);
  await drain();
  assert.equal(
    callCount.get(SLOW_ENDPOINT_PATH),
    2,
    'the slow endpoint should be re-fetched once its in-flight flag clears on settle',
  );
});

test('per-endpoint in-flight guard: a rejected fetch also clears the in-flight flag', async () => {
  const { fetchImpl, callCount, rejectSlow } = makeConcurrencyFetch(SLOW_ENDPOINT_PATH);
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  const state = api.createPollState();
  // A fake, manually-advanced clock (not Date.now()): a rejection now also
  // triggers error backoff (steps 5-6), so the retry tick below needs the
  // clock pushed past the resulting 3000ms nextAllowedAt — otherwise this
  // test would depend on real wall-clock time not having advanced 3s
  // between the two pollTick() calls, which happens to hold today only
  // because they run milliseconds apart. This test's concern is the
  // in-flight flag specifically; backoff itself is covered separately below.
  let t = 0;
  const deps = {
    fetchImpl,
    now: () => t,
    random: () => Math.random(),
    sleep: ms => new Promise(resolve => setTimeout(resolve, ms)),
  };
  const opts = { state, deps, jitterMaxMs: 0 };

  api.pollTick(opts);
  await drain();
  assert.equal(callCount.get(SLOW_ENDPOINT_PATH), 1);

  rejectSlow(new Error('simulated network failure'));
  await drain();
  t = state.get(SLOW_ENDPOINT_PATH).nextAllowedAt; // past the backoff window opened by the rejection

  api.pollTick(opts);
  await drain();
  assert.equal(
    callCount.get(SLOW_ENDPOINT_PATH),
    2,
    'a rejected fetch must clear the in-flight flag so the next tick retries rather than skipping forever',
  );
});

// ---------------------------------------------------------------------------
// Error backoff — driven entirely by an injected, manually-advanced `now()`
// so none of this waits on the wall clock.
// ---------------------------------------------------------------------------

test('error backoff: a thrown fetch error backs the endpoint off — skipped on the immediately following tick, other 12 unaffected', async () => {
  const callCount = new Map();
  const fetchImpl = url => {
    const path = url.split('?')[0];
    callCount.set(path, (callCount.get(path) || 0) + 1);
    if (path === FLAKY_ENDPOINT_PATH) return Promise.reject(new Error('simulated failure'));
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };
  const { api } = loadDataJs({ fetchStub: fetchImpl });
  const others = otherPaths(api, '24h', FLAKY_ENDPOINT_PATH);

  const state = api.createPollState();
  const deps = { fetchImpl, now: () => 0, random: () => 0, sleep: () => Promise.resolve() };
  const opts = { state, deps, jitterMaxMs: 0 };

  api.pollTick(opts); // tick 1: flaky endpoint throws, backs off
  await drain();
  assert.equal(callCount.get(FLAKY_ENDPOINT_PATH), 1);

  api.pollTick(opts); // tick 2: immediately following — now() hasn't advanced, still backed off
  await drain();
  assert.equal(
    callCount.get(FLAKY_ENDPOINT_PATH),
    1,
    'a failing endpoint must be skipped (not re-fetched) on the tick immediately after it failed',
  );

  for (const path of others) {
    assert.equal(callCount.get(path), 2, `${path} must not be affected by a different endpoint's failure`);
  }
});

test('error backoff: a non-ok (503) response backs the endpoint off too, not just a thrown error', async () => {
  const callCount = new Map();
  const fetchImpl = url => {
    const path = url.split('?')[0];
    callCount.set(path, (callCount.get(path) || 0) + 1);
    if (path === FLAKY_ENDPOINT_PATH) return Promise.resolve({ ok: false, status: 503, json: async () => ({}) });
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };
  const { api } = loadDataJs({ fetchStub: fetchImpl });
  const others = otherPaths(api, '24h', FLAKY_ENDPOINT_PATH);

  const state = api.createPollState();
  const deps = { fetchImpl, now: () => 0, random: () => 0, sleep: () => Promise.resolve() };
  const opts = { state, deps, jitterMaxMs: 0 };

  api.pollTick(opts); // tick 1: 503, backs off
  await drain();
  assert.equal(callCount.get(FLAKY_ENDPOINT_PATH), 1);

  api.pollTick(opts); // tick 2: still within the backoff window
  await drain();
  assert.equal(
    callCount.get(FLAKY_ENDPOINT_PATH),
    1,
    'a 503 must back the endpoint off exactly like a thrown error, not just be silently retried at full rate',
  );

  for (const path of others) {
    assert.equal(callCount.get(path), 2, `${path} must not be affected by a different endpoint's 503`);
  }
});

test('error backoff: delay schedule is 3000 -> 6000 -> 12000 -> 24000 -> 48000 -> 60000 -> 60000 (3000 * 2^(n-1), capped at 60000)', async () => {
  const fetchImpl = url => {
    const path = url.split('?')[0];
    if (path === FLAKY_ENDPOINT_PATH) return Promise.reject(new Error('always fails'));
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  const state = api.createPollState();
  let t = 0;
  const deps = { fetchImpl, now: () => t, random: () => 0, sleep: () => Promise.resolve() };
  const opts = { state, deps, jitterMaxMs: 0 };

  const expectedDelays = [3000, 6000, 12000, 24000, 48000, 60000, 60000];
  for (let i = 0; i < expectedDelays.length; i++) {
    api.pollTick(opts); // now() is exactly at (or past) nextAllowedAt, so this attempt is not itself skipped
    await drain();
    const st = state.get(FLAKY_ENDPOINT_PATH);
    assert.equal(st.failures, i + 1, `failures should be ${i + 1} after consecutive failure #${i + 1}`);
    assert.equal(
      st.nextAllowedAt,
      t + expectedDelays[i],
      `nextAllowedAt after failure #${i + 1} should be now (${t}) + ${expectedDelays[i]}`,
    );
    t = st.nextAllowedAt; // advance the clock to exactly when the endpoint is allowed again
  }
});

test('error backoff: once retried past the backoff window, a SUCCESSFUL fetch resets failures/nextAllowedAt to 0', async () => {
  let shouldFail = true;
  const fetchImpl = url => {
    const path = url.split('?')[0];
    if (path === FLAKY_ENDPOINT_PATH && shouldFail) return Promise.reject(new Error('boom'));
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  const state = api.createPollState();
  let t = 0;
  const deps = { fetchImpl, now: () => t, random: () => 0, sleep: () => Promise.resolve() };
  const opts = { state, deps, jitterMaxMs: 0 };

  api.pollTick(opts); // failure #1
  await drain();
  let st = state.get(FLAKY_ENDPOINT_PATH);
  assert.equal(st.failures, 1);
  assert.equal(st.nextAllowedAt, 3000);

  shouldFail = false;
  t = 3000; // advance the clock to exactly nextAllowedAt — the retry must not itself be skipped
  api.pollTick(opts); // retried, succeeds this time
  await drain();
  st = state.get(FLAKY_ENDPOINT_PATH);
  assert.equal(st.failures, 0, 'a successful retry should reset failures to 0');
  assert.equal(st.nextAllowedAt, 0, 'a successful retry should reset nextAllowedAt to 0, i.e. full 3s rate resumes');
});

test('error backoff: refreshDFData(win) (chip change) bypasses backoff, but is still refused by the in-flight guard', async () => {
  // Part 1: a chip change fetches a currently-backed-off endpoint immediately.
  {
    const callCount = new Map();
    const fetchImpl = url => {
      const path = url.split('?')[0];
      callCount.set(path, (callCount.get(path) || 0) + 1);
      if (path === FLAKY_ENDPOINT_PATH) return Promise.reject(new Error('boom'));
      return Promise.resolve({ ok: true, json: async () => ({}) });
    };
    const { api } = loadDataJs({ fetchStub: fetchImpl });
    const state = api.createPollState();
    const deps = { fetchImpl, now: () => 0, random: () => 0, sleep: () => Promise.resolve() };

    // Timer path (no win) — fails, backs off. now() stays 0 for the rest of
    // this block, well inside the resulting backoff window.
    await api.refreshDFData(undefined, { state, deps, jitterMaxMs: 0 });
    assert.equal(callCount.get(FLAKY_ENDPOINT_PATH), 1);

    api.pollTick({ state, deps, jitterMaxMs: 0 }); // sanity: timer path stays backed off
    await drain();
    assert.equal(callCount.get(FLAKY_ENDPOINT_PATH), 1, 'sanity check: timer path must still be backed off here');

    // Chip change (explicit non-empty win) — must bypass backoff.
    await api.refreshDFData('7d', { state, deps, jitterMaxMs: 0 });
    assert.equal(
      callCount.get(FLAKY_ENDPOINT_PATH),
      2,
      'refreshDFData(win) must bypass backoff for a currently-backed-off endpoint',
    );
  }

  // Part 2: a chip change does NOT stack a second concurrent request for an
  // endpoint that is already in flight — the in-flight guard still applies.
  {
    const { fetchImpl, maxConcurrent, callCount, releaseSlow } = makeConcurrencyFetch(FLAKY_ENDPOINT_PATH);
    const { api } = loadDataJs({ fetchStub: fetchImpl });
    const state = api.createPollState();
    const deps = { fetchImpl, now: () => 0, random: () => 0, sleep: () => Promise.resolve() };
    const opts = { state, deps, jitterMaxMs: 0 };

    api.pollTick(opts); // starts an in-flight fetch for the flaky endpoint, held open
    await drain();
    assert.equal(callCount.get(FLAKY_ENDPOINT_PATH), 1);

    await api.refreshDFData('7d', opts); // chip change while still in flight
    assert.equal(
      callCount.get(FLAKY_ENDPOINT_PATH),
      1,
      'a chip change must not stack a second concurrent request for an endpoint already in flight',
    );
    assert.equal(maxConcurrent.get(FLAKY_ENDPOINT_PATH), 1);

    releaseSlow();
    await drain();
  }
});
