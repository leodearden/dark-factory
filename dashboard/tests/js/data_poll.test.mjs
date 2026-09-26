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
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

// The UI's own reader of the map data.js publishes. Imported so the recovery
// test below can assert what an OPERATOR sees, not merely what the map holds
// — a static import is safe here (unlike data.js, this module touches no
// browser global at load; its window assignment is typeof-guarded).
import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const { staleNoticesForTab } = staleness;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/data.js';
const DATUM_MODULE_SPECIFIER = '../../src/dashboard/static/redux/datum.js';
const EXPECTED_FUNCTION_NAMES = [
  'endpointsFor',
  'applyKey',
  'refreshOne',
  'refreshDFData',
  'pollTick',
  'startPolling',
  'createPollState',
  'pollKey',
  'datumFor',
  'requestOnDemand',
  'onDemandView',
];

// Full DF_DATA key set (data.js:41-127) — initialised so the first render
// before fetch completes cannot crash any component reading DF_DATA.*.
const EXPECTED_DF_DATA_KEYS = [
  'PROJECTS', 'AGENTS', 'ORCHESTRATORS', 'ORCHESTRATORS_SPARK',
  'ACTIVE_TASKS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS',
  'TASKS_DEGRADED_PROJECTS', 'TASKS_PROJECT_COUNT', 'TASKS_SNAPSHOT',
  'PERFORMANCE', 'MEMORY_STATUS', 'MEMORY_TIMESERIES', 'MEMORY_OPS_BREAKDOWN',
  'RECON_STATE', 'MERGE_QUEUE', 'COSTS', 'BURNDOWN', 'BURNDOWN_BY_PROJECT',
  'CURATOR_STATE', 'ESCALATIONS', 'ESCALATION_ANALYTICS', 'SCHEDULER',
  'MEMORY_EVALS',
  // Per-endpoint staleness, published by refreshOne (task 4884, #4791).
  // Nested under DF_DATA alongside __loaded rather than added as a second
  // global, and seeded here for the same reason every domain key is: the
  // first render happens before any fetch resolves, and a consumer reading
  // DF_DATA.__stale[path] must not have to guard the container itself.
  '__stale',
  // Per-endpoint receipts, published by refreshOne on the SUCCESS path only
  // (task 5588). Where __stale records ATTEMPT history, this records the
  // PROVENANCE of the values now sitting in DF_DATA — which is exactly why it
  // must NOT advance on a failure: that is what makes a wedged endpoint's
  // tiles keep ageing on screen instead of resetting to "just received".
  '__receipt',
];

// Number of rows in endpointsFor() (data.js:16-34). Several tests below assert
// that a cycle touched EVERY endpoint — "all of them" is the actual claim, and
// a literal is the only way to state it without deriving the expectation from
// the same map under test. Bump this whenever endpointsFor gains or loses a
// row; it is deliberately one constant rather than a literal repeated per
// test, because scattered copies is what went stale when the memory-evals
// endpoint was added.
const EXPECTED_ENDPOINT_COUNT = 14;

// The two shared registry specs. Declared here as literals rather than read off
// data.js so the reshape is pinned against a stated expectation instead of
// against itself.
const PLAIN_SPEC = { kind: 'plain' };
const DATUM_SPEC = { kind: 'datum' };

// Every DF_DATA key any endpoint row names, as one sorted list. The registry
// reshape (array of key names -> object of key name to spec) is exactly the
// kind of edit that can silently DROP a key — an object literal with a
// duplicated or mistyped name loses a row with no error anywhere — and a
// dropped key means a tab that simply stops updating. Stated as a literal
// because deriving it from endpointsFor would be deriving the expectation from
// the thing under test.
const EXPECTED_ENDPOINT_KEYS = [
  'ACTIVE_TASKS', 'AGENTS', 'BURNDOWN', 'BURNDOWN_BY_PROJECT', 'COSTS',
  'CURATOR_STATE', 'ESCALATIONS', 'ESCALATION_ANALYTICS',
  'MEMORY_EVALS', 'MEMORY_OPS_BREAKDOWN', 'MEMORY_STATUS', 'MEMORY_TIMESERIES',
  'MERGE_QUEUE', 'ORCHESTRATORS', 'ORCHESTRATORS_SPARK', 'PERFORMANCE',
  'PROJECTS', 'RECON_STATE', 'SCHEDULER', 'TASKS_COUNT_UNKNOWN_PROJECTS',
  'TASKS_DEGRADED_PROJECTS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS',
  'TASKS_PROJECT_COUNT', 'TASKS_SNAPSHOT',
];

// The subject key of the generic datum-kind tests below. SYNTHETIC on purpose:
// those tests are about what applyKey and datumFor do with a key DECLARED
// datum-kinded, and no polled key is one, so borrowing a real key would pin a
// payload that key does not carry (DONE_COUNTS was borrowed, and then retired).
const SYNTHETIC_DATUM_KEY = 'SYNTHETIC_DATUM_KEY';

// A five-key wire envelope, as data/datum.py::Datum.to_wire() emits one.
const SERVED_DATUM = Object.freeze({
  value: { total: 9 },
  as_of: '2026-09-20T09:00:00+00:00',
  state: 'lower_bound',
  reason: 'window truncated at 500 rows',
  freshness_bound_seconds: 60,
});

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
  // DF_ENDPOINT_STALENESS is installed for datum.js, which destructures
  // formatAge off it at module scope; datum.js is then loaded through the same
  // shim so it can publish DF_DATUM, which data.js in turn destructures at
  // module scope. index.html gives them exactly this order —
  // endpoint_staleness.js -> datum.js -> data.js — and test_index_html.py pins
  // both edges.
  const win = { dispatchEvent: ev => events.push(ev), DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  globalThis.fetch = (url, init) => {
    fetchCalls.push({ url, init });
    if (fetchStub) return fetchStub(url, init);
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };

  const require = createRequire(import.meta.url);
  const datumResolved = require.resolve(DATUM_MODULE_SPECIFIER);
  delete require.cache[datumResolved];
  require(DATUM_MODULE_SPECIFIER);
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

// Pure, state-free — reused below so the test fixtures assert against
// data.js's own key derivation rather than a hand-copied duplicate of the
// `url.split('?')[0]` rule (see pollKey's doc comment in data.js). Safe to
// grab once at file scope: loadDataJs()'s window/fetch shim is fully
// re-installed by every test's own loadDataJs() call, so this throwaway
// load leaves nothing behind that a later test could observe.
const { pollKey } = loadDataJs().api;

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
    const path = pollKey(url);
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
  return Object.keys(api.endpointsFor(win)).map(url => pollKey(url)).filter(p => p !== excludePath);
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

test('auto-start: loading WITH a `document` global lets the guard invoke startPolling (exactly one interval registered)', async () => {
  // The mirror image of the previous test: an inverted or mistyped guard
  // (e.g. `typeof document === 'undefined'`) would silently disable all
  // dashboard polling in the browser while the negative test above stayed
  // green — this is the only test in the file that would catch it.
  assert.equal(typeof globalThis.document, 'undefined', 'test environment must not already define document');
  globalThis.document = {};

  // The guard's internal `startPolling()` call takes no opts, so its
  // real jitter (up to 1500ms) and abort-deadline timer cannot be injected
  // away like every other test in this file does. Left alone, that leaves a
  // real, up-to-1500ms-delayed fetch pending when this test returns — which
  // can land on a LATER test's globalThis.fetch once its jitter elapses (a
  // first draft of this test intermittently inflated an unrelated later
  // test's call count this way). Redirecting setTimeout to a microtask for
  // the duration of this test collapses that real delay to "next microtask
  // checkpoint", so a single drain() lets the whole cycle fully settle
  // (successfully or not — this test only cares about the interval count).
  const originalSetTimeout = globalThis.setTimeout;
  globalThis.setTimeout = (fn, _ms, ...args) => {
    queueMicrotask(() => fn(...args));
    return 0;
  };
  try {
    const { intervalCalls } = loadDataJs();
    assert.equal(
      intervalCalls.length,
      1,
      'loading data.js with a `document` global present must invoke startPolling and register exactly one poll interval',
    );
    await drain();
  } finally {
    globalThis.setTimeout = originalSetTimeout;
    delete globalThis.document;
  }
});

test('startPolling: performs an immediate refresh, registers exactly one interval, and stop() clears that exact interval', async () => {
  const { api } = loadDataJs();

  const fetchCalls = [];
  const fetchImpl = url => {
    fetchCalls.push(url);
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };

  // Wrap the real timer globals for the duration of this test only, so the
  // returned stop() handle can be verified to clear the SAME interval
  // startPolling registered (rather than merely trusting clearInterval,
  // a built-in, to have been called at all).
  const originalSetInterval = globalThis.setInterval;
  const originalClearInterval = globalThis.clearInterval;
  const registered = [];
  const cleared = [];
  globalThis.setInterval = (...args) => {
    const handle = originalSetInterval(...args);
    registered.push(handle);
    return handle;
  };
  globalThis.clearInterval = handle => {
    cleared.push(handle);
    return originalClearInterval(handle);
  };

  try {
    const pollHandle = api.startPolling({
      state: api.createPollState(),
      deps: { fetchImpl, now: () => 0, random: () => 0, sleep: () => Promise.resolve() },
      jitterMaxMs: 0,
    });
    await drain();

    assert.ok(
      fetchCalls.length > 0,
      'startPolling must perform an immediate refresh (at least one fetch) without waiting for the first interval tick',
    );
    assert.equal(registered.length, 1, 'startPolling must register exactly one interval');

    pollHandle.stop();
    assert.equal(cleared.length, 1, 'stop() must clear the interval');
    assert.equal(cleared[0], registered[0], 'stop() must clear the SAME interval startPolling registered');
  } finally {
    globalThis.setInterval = originalSetInterval;
    globalThis.clearInterval = originalClearInterval;
  }
});

test('production fallbacks: refreshDFData with no explicit state/deps uses globalThis.fetch and the shared DF_POLL_STATE singleton', async () => {
  // Every other test in this file injects both opts.state and opts.deps, so
  // the production fallbacks — o.state || DF_POLL_STATE, and each entry of
  // DEFAULT_POLL_DEPS (now/random/sleep/fetchImpl/setTimeoutImpl/
  // clearTimeoutImpl/timeoutMs) — are otherwise entirely unexercised. A
  // fetchStub that holds every request open lets one cycle prove both the
  // fetchImpl fallback (globalThis.fetch is reached, with credentials
  // same-origin) and the DF_POLL_STATE fallback (a second concurrent cycle,
  // also with no explicit state, must see the first cycle's in-flight flags).
  let releaseAll;
  const gate = new Promise(resolve => { releaseAll = resolve; });
  const fetchStub = () => gate.then(() => ({ ok: true, json: async () => ({}) }));
  const { api, fetchCalls } = loadDataJs({ fetchStub });

  const firstCycle = api.refreshDFData(undefined, { jitterMaxMs: 0 });
  await drain();

  assert.equal(
    fetchCalls.length,
    EXPECTED_ENDPOINT_COUNT,
    `the default fetchImpl fallback must reach globalThis.fetch for all ${EXPECTED_ENDPOINT_COUNT} endpoints`,
  );
  for (const { init } of fetchCalls) {
    assert.equal(init.credentials, 'same-origin', 'the default fetchImpl fallback must still pass credentials: same-origin');
  }

  // Second cycle, fired while the first is still held open, ALSO with no
  // explicit `state`: if the o.state || DF_POLL_STATE fallback were broken
  // or DF_POLL_STATE were shadowed/undefined, this would see fresh
  // (non-in-flight) state and re-fetch every endpoint again.
  const secondCycle = api.refreshDFData(undefined, { jitterMaxMs: 0 });
  await drain();
  assert.equal(
    fetchCalls.length,
    EXPECTED_ENDPOINT_COUNT,
    'a second cycle sharing the DF_POLL_STATE singleton must skip every still-in-flight endpoint, not re-fetch them',
  );

  releaseAll();
  await Promise.all([firstCycle, secondCycle]);
});

// ---------------------------------------------------------------------------
// Per-endpoint in-flight guard — THE REQUIRED REGRESSION TEST.
//
// Motivating incident: memory-graphs measured a 108s response against a 3s
// poll interval, so an unguarded loader stacks ~36 concurrent requests for
// that one endpoint by the time it finally answers. This drives four tick
// fires while memory-graphs is held open and asserts it never exceeds 1
// concurrent fetch, while every other endpoint is completely unaffected
// (proving the guard is per-endpoint, not whole-cycle — a whole-cycle guard
// would starve all of them on any single slow one).
// ---------------------------------------------------------------------------

test('per-endpoint in-flight guard: a slow endpoint never exceeds concurrency 1, and does not block the other endpoints (regression)', async () => {
  const { fetchImpl, maxConcurrent, callCount, releaseSlow } = makeConcurrencyFetch(SLOW_ENDPOINT_PATH);
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  const allPaths = Object.keys(api.endpointsFor('24h')).map(url => pollKey(url));
  assert.equal(
    allPaths.length,
    EXPECTED_ENDPOINT_COUNT,
    `expected ${EXPECTED_ENDPOINT_COUNT} endpoints (sanity check on the endpointsFor fixture)`,
  );
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

test('in-flight guard: a fetch that never settles is aborted after the timeout deadline, backs off, and is retried rather than wedged forever', async () => {
  // Models the motivating incident with no upper bound at all: without a
  // deadline, a hung fetch wedges st.inFlight permanently and every later
  // tick skips the endpoint for the lifetime of the page. The deadline
  // timer is driven by a fake setTimeoutImpl/clearTimeoutImpl (captured and
  // fired manually below) rather than real wall-clock time or `now()`,
  // since the abort is scheduled independently of the injected clock.
  const callCount = new Map();
  let flakyAttempts = 0;
  const fetchImpl = url => {
    const path = pollKey(url);
    callCount.set(path, (callCount.get(path) || 0) + 1);
    if (path === FLAKY_ENDPOINT_PATH) {
      flakyAttempts += 1;
      if (flakyAttempts === 1) return new Promise(() => {}); // first attempt: never settles
    }
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  const state = api.createPollState();
  let t = 0;
  let nextTimeoutId = 1;
  const pendingTimeouts = new Map(); // id -> {fn, ms} — mirrors a real timer queue, driven manually
  const deps = {
    fetchImpl,
    now: () => t,
    random: () => 0,
    sleep: () => Promise.resolve(),
    setTimeoutImpl: (fn, ms) => {
      const id = nextTimeoutId++;
      pendingTimeouts.set(id, { fn, ms });
      return id;
    },
    clearTimeoutImpl: id => { pendingTimeouts.delete(id); },
    timeoutMs: 30000,
  };
  const opts = { state, deps, jitterMaxMs: 0 };

  api.pollTick(opts); // tick 1: the flaky endpoint hangs; the other 12 succeed and clear their own deadline timers
  await drain();
  assert.equal(callCount.get(FLAKY_ENDPOINT_PATH), 1);
  assert.equal(
    pendingTimeouts.size,
    1,
    'only the hung endpoint should still have a live deadline timer once the other 12 have settled and cleared theirs',
  );
  const [timeoutId] = pendingTimeouts.keys();
  const { fn: fireDeadline, ms } = pendingTimeouts.get(timeoutId);
  assert.equal(ms, 30000, 'the deadline must use the configured timeoutMs');

  api.pollTick(opts); // tick 2: still genuinely in flight (deadline hasn't fired) — must be skipped, not double-fetched
  await drain();
  assert.equal(
    callCount.get(FLAKY_ENDPOINT_PATH),
    1,
    'a still-pending (not yet timed out) fetch must be skipped by the in-flight guard, not retried',
  );

  // Simulate the deadline elapsing (rather than waiting 30 real seconds).
  fireDeadline();
  await drain();
  assert.equal(pendingTimeouts.has(timeoutId), false, 'the deadline timer must be cleared once it fires');

  // A timeout is treated as an ordinary failure, so it also backs off —
  // advance the injected clock past nextAllowedAt before expecting a retry.
  t = state.get(FLAKY_ENDPOINT_PATH).nextAllowedAt;

  api.pollTick(opts); // tick 3: in-flight flag cleared by the aborted attempt's `finally` — retried
  await drain();
  assert.equal(
    callCount.get(FLAKY_ENDPOINT_PATH),
    2,
    'once the deadline elapses the endpoint must be retried instead of staying wedged forever',
  );
});

// ---------------------------------------------------------------------------
// Error backoff — driven entirely by an injected, manually-advanced `now()`
// so none of this waits on the wall clock.
// ---------------------------------------------------------------------------

test('error backoff: a thrown fetch error backs the endpoint off — skipped on the immediately following tick, other 12 unaffected', async () => {
  const callCount = new Map();
  const fetchImpl = url => {
    const path = pollKey(url);
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
    const path = pollKey(url);
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
    const path = pollKey(url);
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
    const path = pollKey(url);
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

// One of the 4 endpoints whose URL actually carries ?window= (data.js's
// endpointsFor) — distinct from FLAKY_ENDPOINT_PATH (curator, unwindowed),
// used below to prove the chip-change bypass is scoped to endpoints the
// chip actually affects, not applied cycle-wide.
const WINDOWED_ENDPOINT_PATH = '/api/v2/dashboard/costs';

test('error backoff: refreshDFData(win) (chip change) bypasses backoff only for the 4 windowed endpoints, without inflating failures, and is still refused by the in-flight guard', async () => {
  // Part 1: a chip change fetches a currently-backed-off WINDOWED endpoint
  // immediately (its URL actually changes on a chip click), but leaves a
  // currently-backed-off UNWINDOWED endpoint (no ?window=, unaffected by
  // the chip) untouched — the bypass must be scoped, not cycle-wide, or a
  // chip click during an outage would re-hammer every endpoint.
  {
    const callCount = new Map();
    const failing = new Set([FLAKY_ENDPOINT_PATH, WINDOWED_ENDPOINT_PATH]);
    const fetchImpl = url => {
      const path = pollKey(url);
      callCount.set(path, (callCount.get(path) || 0) + 1);
      if (failing.has(path)) return Promise.reject(new Error('boom'));
      return Promise.resolve({ ok: true, json: async () => ({}) });
    };
    const { api } = loadDataJs({ fetchStub: fetchImpl });
    const state = api.createPollState();
    const deps = { fetchImpl, now: () => 0, random: () => 0, sleep: () => Promise.resolve() };

    // Timer path (no win) — both fail, both back off. now() stays 0 for the
    // rest of this block, well inside the resulting backoff window.
    await api.refreshDFData(undefined, { state, deps, jitterMaxMs: 0 });
    assert.equal(callCount.get(FLAKY_ENDPOINT_PATH), 1);
    assert.equal(callCount.get(WINDOWED_ENDPOINT_PATH), 1);
    const flakyFailuresBefore = state.get(FLAKY_ENDPOINT_PATH).failures;
    assert.equal(flakyFailuresBefore, 1);

    // Chip change (explicit non-empty win) — must bypass backoff for the
    // windowed endpoint...
    await api.refreshDFData('7d', { state, deps, jitterMaxMs: 0 });
    assert.equal(
      callCount.get(WINDOWED_ENDPOINT_PATH),
      2,
      'refreshDFData(win) must bypass backoff for a currently-backed-off WINDOWED endpoint',
    );
    // ...but must NOT touch an unwindowed endpoint's backoff at all.
    assert.equal(
      callCount.get(FLAKY_ENDPOINT_PATH),
      1,
      'refreshDFData(win) must not bypass backoff for an endpoint the chip change has no bearing on',
    );

    // A forced attempt that fails again must not inflate `failures` beyond
    // what the timer path alone produced — otherwise repeated chip clicks
    // during an outage could escalate the TIMER path's backoff for an
    // endpoint whose URL the user's action changed.
    assert.equal(
      state.get(WINDOWED_ENDPOINT_PATH).failures,
      flakyFailuresBefore,
      'a forced (bypassed) failing attempt must not increment failures',
    );
  }

  // Part 2: a chip change does NOT stack a second concurrent request for an
  // endpoint that is already in flight — the in-flight guard still applies,
  // even to a windowed endpoint that is otherwise bypass-eligible.
  {
    const { fetchImpl, maxConcurrent, callCount, releaseSlow } = makeConcurrencyFetch(WINDOWED_ENDPOINT_PATH);
    const { api } = loadDataJs({ fetchStub: fetchImpl });
    const state = api.createPollState();
    const deps = { fetchImpl, now: () => 0, random: () => 0, sleep: () => Promise.resolve() };
    const opts = { state, deps, jitterMaxMs: 0 };

    api.pollTick(opts); // starts an in-flight fetch for the windowed endpoint, held open
    await drain();
    assert.equal(callCount.get(WINDOWED_ENDPOINT_PATH), 1);

    await api.refreshDFData('7d', opts); // chip change while still in flight
    assert.equal(
      callCount.get(WINDOWED_ENDPOINT_PATH),
      1,
      'a chip change must not stack a second concurrent request for an endpoint already in flight',
    );
    assert.equal(maxConcurrent.get(WINDOWED_ENDPOINT_PATH), 1);

    releaseSlow();
    await drain();
  }
});

// ---------------------------------------------------------------------------
// Jitter — spreads the endpoint fetches across part of the 3s interval
// instead of every tick firing all of them at once (task 185's lesson: 13
// simultaneous requests hammering a single aiosqlite worker thread). Driven
// by a recording `sleep` dep (captures every requested delay, resolves
// immediately unless a test deliberately holds it open) and a deterministic
// per-call `random`, so none of this waits on the wall clock either.
//
// data.js does not export JITTER_MAX_MS/POLL_INTERVAL_MS, so — same as the
// error-backoff tests above hardcoding the 3000/6000/.../60000 schedule
// instead of importing BACKOFF_BASE_MS/BACKOFF_MAX_MS — these mirror the two
// constants' documented values (step-8: JITTER_MAX_MS = 1500; step-2:
// POLL_INTERVAL_MS = 3000) as local expectations.
// ---------------------------------------------------------------------------

const EXPECTED_JITTER_MAX_MS = 1500;
const EXPECTED_POLL_INTERVAL_MS = 3000;

test('jitter: every endpoint awaits a pre-fetch delay in [0, JITTER_MAX_MS), and the delays are not all identical', async () => {
  // Sanity check on the fixture itself: the jitter cap must stay below the
  // poll interval, so a jittered start can never structurally slip past the
  // next tick.
  assert.ok(
    EXPECTED_JITTER_MAX_MS < EXPECTED_POLL_INTERVAL_MS,
    `jitter cap (${EXPECTED_JITTER_MAX_MS}) must stay below the poll interval (${EXPECTED_POLL_INTERVAL_MS})`,
  );

  const fetchImpl = () => Promise.resolve({ ok: true, json: async () => ({}) });
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  const sleepCalls = [];
  let callIndex = 0;
  const deps = {
    fetchImpl,
    now: () => 0,
    // One distinct fraction in [0, 1) per endpoint — deterministic, and
    // spread enough that flooring against JITTER_MAX_MS cannot
    // coincidentally collapse them all to the same integer delay.
    random: () => (callIndex++ % EXPECTED_ENDPOINT_COUNT) / EXPECTED_ENDPOINT_COUNT,
    sleep: ms => { sleepCalls.push(ms); return Promise.resolve(); },
  };

  // jitterMaxMs is intentionally omitted from opts: production callers
  // (pollTick / the setInterval loop) never pass it either, so this
  // exercises data.js's own internal default rather than a test override.
  await api.refreshDFData(undefined, { state: api.createPollState(), deps });

  assert.equal(
    sleepCalls.length,
    EXPECTED_ENDPOINT_COUNT,
    `every one of the ${EXPECTED_ENDPOINT_COUNT} endpoints must await a jitter sleep`,
  );
  for (const ms of sleepCalls) {
    assert.ok(
      ms >= 0 && ms < EXPECTED_JITTER_MAX_MS,
      `jitter delay ${ms} must be in [0, ${EXPECTED_JITTER_MAX_MS})`,
    );
  }
  assert.ok(
    new Set(sleepCalls).size > 1,
    'the jitter delays must not all be identical — the fan-out must be genuinely spread',
  );
});

test('jitter: the sleep happens INSIDE the in-flight window — a second pollTick fired mid-jitter is skipped', async () => {
  const callCount = new Map();
  const fetchImpl = url => {
    const path = pollKey(url);
    callCount.set(path, (callCount.get(path) || 0) + 1);
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  let releaseJitter;
  const jitterGate = new Promise(resolve => { releaseJitter = resolve; });
  const deps = {
    fetchImpl,
    now: () => 0,
    random: () => 0.5,
    // Held open (ignores `ms`) rather than resolving immediately — models
    // "still inside its jitter delay" for every endpoint at once, so a
    // second tick firing in that window has something real to be skipped by.
    sleep: () => jitterGate,
  };
  const opts = { state: api.createPollState(), deps };

  api.pollTick(opts); // tick 1: every endpoint enters its jitter delay, held open
  await drain();
  assert.equal(callCount.get(FLAKY_ENDPOINT_PATH), undefined, 'no fetch should have gone out yet — still jittering');

  api.pollTick(opts); // tick 2: fired while tick 1's endpoints are still jittering
  await drain();
  assert.equal(
    callCount.get(FLAKY_ENDPOINT_PATH),
    undefined,
    'a second pollTick fired mid-jitter must be skipped by the in-flight guard, not start a second fetch',
  );

  releaseJitter();
  await drain();
  assert.equal(
    callCount.get(FLAKY_ENDPOINT_PATH),
    1,
    'once the jitter delay resolves, the fetch proceeds exactly once (tick 2 having been skipped, not queued)',
  );
});

test('jitter: passing jitterMaxMs: 0 issues no sleep at all', async () => {
  const fetchImpl = () => Promise.resolve({ ok: true, json: async () => ({}) });
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  const sleepCalls = [];
  const deps = {
    fetchImpl,
    now: () => 0,
    random: () => 0.5,
    sleep: ms => { sleepCalls.push(ms); return Promise.resolve(); },
  };

  await api.refreshDFData(undefined, { state: api.createPollState(), deps, jitterMaxMs: 0 });

  assert.equal(
    sleepCalls.length,
    0,
    'jitterMaxMs: 0 must skip the sleep entirely — this is what keeps the rest of this file deterministic',
  );
});

// ---------------------------------------------------------------------------
// Preserved-behaviour contract — pins the things the task explicitly says
// NOT to break: __DF_PAUSE, keep-prior-values-on-failure, applyKey
// reference stability, the per-cycle df-data-refresh dispatch (including a
// cycle where every endpoint got skipped), and the ?window= URL shape with
// its path-keyed flow-control state. A later refactor cannot quietly drop
// any of these without one of the tests below going red.
// ---------------------------------------------------------------------------

test('preserved behaviour: window.__DF_PAUSE = true stops pollTick from fetching; false resumes on the next tick', async () => {
  const callCount = new Map();
  const fetchImpl = url => {
    const path = pollKey(url);
    callCount.set(path, (callCount.get(path) || 0) + 1);
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };
  const { api, window: win } = loadDataJs({ fetchStub: fetchImpl });
  const opts = { state: api.createPollState(), deps: { fetchImpl }, jitterMaxMs: 0 };

  win.__DF_PAUSE = true;
  api.pollTick(opts);
  await drain();
  assert.equal(callCount.size, 0, 'pollTick must issue zero fetches while __DF_PAUSE is true');

  win.__DF_PAUSE = false;
  api.pollTick(opts);
  await drain();
  assert.equal(
    callCount.get(FLAKY_ENDPOINT_PATH),
    1,
    'pollTick must resume fetching once __DF_PAUSE is set back to false, on the very next tick',
  );
});

test('preserved behaviour: a thrown fetch error keeps the prior DF_DATA value and still warns', async () => {
  const { api, window: win } = loadDataJs();
  win.DF_DATA.CURATOR_STATE = { marker: 'prior-throw' };

  const originalWarn = console.warn;
  const warnCalls = [];
  console.warn = (...args) => warnCalls.push(args);
  try {
    const state = api.createPollState();
    const deps = { fetchImpl: () => Promise.reject(new Error('boom')), now: () => 0 };
    await api.refreshOne(FLAKY_ENDPOINT_PATH, { CURATOR_STATE: PLAIN_SPEC }, state, deps);
  } finally {
    console.warn = originalWarn;
  }

  assert.deepEqual(
    win.DF_DATA.CURATOR_STATE,
    { marker: 'prior-throw' },
    'a thrown fetch error must leave the prior DF_DATA value untouched rather than blanking it',
  );
  assert.ok(
    warnCalls.some(args => args[0] === 'DF_DATA fetch failed'),
    'a thrown fetch error must still emit the console.warn',
  );
});

test('preserved behaviour: a non-ok (503) response also keeps the prior DF_DATA value intact', async () => {
  const { api, window: win } = loadDataJs();
  win.DF_DATA.CURATOR_STATE = { marker: 'prior-503' };

  const state = api.createPollState();
  const deps = { fetchImpl: () => Promise.resolve({ ok: false, status: 503, json: async () => ({}) }), now: () => 0 };
  await api.refreshOne(FLAKY_ENDPOINT_PATH, { CURATOR_STATE: PLAIN_SPEC }, state, deps);

  assert.deepEqual(
    win.DF_DATA.CURATOR_STATE,
    { marker: 'prior-503' },
    'a non-ok response must leave the prior DF_DATA value untouched rather than blanking it',
  );
});

test('preserved behaviour: applyKey mutates PROJECTS/AGENTS in place, replaces other keys by reference, and ignores undefined/null', () => {
  const { api, window: win } = loadDataJs();

  const projectsRef = win.DF_DATA.PROJECTS;
  const agentsRef = win.DF_DATA.AGENTS;
  api.applyKey('PROJECTS', [{ id: 'p1' }]);
  api.applyKey('AGENTS', [{ id: 'a1' }]);
  assert.equal(
    win.DF_DATA.PROJECTS,
    projectsRef,
    'PROJECTS must stay the same array reference — shell.jsx captures it at module load',
  );
  assert.equal(win.DF_DATA.AGENTS, agentsRef, 'AGENTS must stay the same array reference');
  assert.deepEqual(win.DF_DATA.PROJECTS, [{ id: 'p1' }], 'PROJECTS content must still be updated (in place)');
  assert.deepEqual(win.DF_DATA.AGENTS, [{ id: 'a1' }], 'AGENTS content must still be updated (in place)');

  const newCosts = { summary: { total: 42 } };
  api.applyKey('COSTS', newCosts);
  assert.equal(win.DF_DATA.COSTS, newCosts, 'non-stable keys must be replaced by reference');

  const priorScheduler = win.DF_DATA.SCHEDULER;
  api.applyKey('SCHEDULER', undefined);
  assert.equal(win.DF_DATA.SCHEDULER, priorScheduler, 'undefined values must be ignored');
  api.applyKey('SCHEDULER', null);
  assert.equal(win.DF_DATA.SCHEDULER, priorScheduler, 'null values must be ignored');
});

test('preserved behaviour: df-data-refresh dispatches exactly once per cycle, including a cycle where every endpoint is skipped', async () => {
  const alwaysFail = () => Promise.reject(new Error('boom'));
  const { api, events } = loadDataJs({ fetchStub: alwaysFail });
  const countDfEvents = () => events.filter(e => e.type === 'df-data-refresh').length;

  const state = api.createPollState();
  const t = 0; // fixed clock — cycle 2 lands inside cycle 1's backoff window
  const deps = { fetchImpl: alwaysFail, now: () => t, random: () => 0, sleep: () => Promise.resolve() };
  const opts = { state, deps, jitterMaxMs: 0 };

  // Cycle 1: every endpoint fails and backs off.
  await api.refreshDFData(undefined, opts);
  assert.equal(countDfEvents(), 1, 'cycle 1 (all endpoints failing) must still dispatch exactly one df-data-refresh event');

  // Cycle 2: same clock, so every endpoint is now backed off and skipped
  // outright — zero fetches this cycle, and yet the event must still fire.
  await api.refreshDFData(undefined, opts);
  assert.equal(
    countDfEvents(),
    2,
    'a cycle in which every endpoint was skipped by backoff must still dispatch df-data-refresh exactly once',
  );
});

test("preserved behaviour: refreshDFData(win) updates the ?window= param on the 4 windowed endpoints, and flow-control state stays keyed by path across a chip change", async () => {
  const seenUrls = [];
  const fetchImpl = url => {
    seenUrls.push(url);
    return Promise.resolve({ ok: true, json: async () => ({}) });
  };
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  const state = api.createPollState();
  const deps = { fetchImpl, now: () => 0, random: () => 0, sleep: () => Promise.resolve() };

  await api.refreshDFData('7d', { state, deps, jitterMaxMs: 0 });

  const windowedPaths = [
    '/api/v2/dashboard/merge-queue',
    '/api/v2/dashboard/costs',
    '/api/v2/dashboard/performance',
    '/api/v2/dashboard/burndown',
  ];
  for (const path of windowedPaths) {
    assert.ok(
      seenUrls.includes(`${path}?window=7d`),
      `expected a request to ${path}?window=7d after refreshDFData('7d')`,
    );
    assert.ok(
      state.has(path),
      `flow-control state must be keyed by PATH (${path}), not the full ?window= URL`,
    );
  }

  // Switch chips again — the state entries (keyed by path) must be the SAME
  // objects, not fresh ones a chip change silently reset.
  const priorEntries = windowedPaths.map(p => state.get(p));
  await api.refreshDFData('30d', { state, deps, jitterMaxMs: 0 });
  for (const [i, path] of windowedPaths.entries()) {
    assert.equal(
      state.get(path),
      priorEntries[i],
      `the flow-control state for ${path} must be the SAME object across a chip change (path-keyed, not reset)`,
    );
  }
});

// ---------------------------------------------------------------------------
// Per-endpoint staleness publication (task 4884, #4791)
//
// The 2026-08-27 incident ran 19.8h with the dashboard serving a fully
// rendered UI whose numbers had stopped advancing. Keeping the prior values
// on failure is deliberate (see the keep-last-good tests above) — the defect
// is that nothing RECORDED that they were old, so no consumer could say so.
// These tests pin the producer half: refreshOne publishes, per endpoint PATH,
// how many consecutive attempts have failed and when the last success landed.
// endpoint_staleness.js holds the decision of what to do with that;
// dashboard/tests/js/endpoint_staleness.test.mjs covers it.
// ---------------------------------------------------------------------------

// The contract constants, stated as literals rather than read back out of the
// module under test. data.js prefers window.DF_ENDPOINT_STALENESS's threshold
// when that module has loaded and falls back to its own literal otherwise
// (data.js is the FIRST classic script in index.html, so it must not gain a
// hard load-order dependency on a later one); under this node harness the
// window shim carries no DF_ENDPOINT_STALENESS, so the fallback is what runs.
// endpoint_staleness.test.mjs asserts the two agree.
const STALE_FAILURE_THRESHOLD = 3;
const STALE_TIMEOUT_MS = 5000;

const CURATOR_PATH = '/api/v2/dashboard/curator';
const COSTS_PATH = '/api/v2/dashboard/costs';

const DATA_JS_SOURCE = fs.readFileSync(
  path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    '../../src/dashboard/static/redux/data.js',
  ),
  'utf8',
);

test('staleness: __stale is seeded as an empty object alongside __loaded', () => {
  const { window: win } = loadDataJs();

  assert.deepEqual(win.DF_DATA.__stale, {}, 'DF_DATA.__stale must be seeded as {}');
  assert.ok(win.DF_DATA.__loaded, 'DF_DATA.__loaded must still be seeded');
});

test('staleness: a successful refresh publishes failures 0 and the success instant', async () => {
  const fetchImpl = () => Promise.resolve({ ok: true, json: async () => ({}) });
  const { api, window: win } = loadDataJs({ fetchStub: fetchImpl });

  const state = api.createPollState();
  const T = 1_700_000_000_000;
  const deps = { fetchImpl, now: () => T, random: () => 0, sleep: () => Promise.resolve() };

  await api.refreshDFData(undefined, { state, deps, jitterMaxMs: 0 });

  const entry = win.DF_DATA.__stale[CURATOR_PATH];
  assert.ok(entry, `nothing published for ${CURATOR_PATH}`);
  assert.equal(entry.failures, 0);
  assert.equal(entry.lastSuccessAt, T, 'lastSuccessAt must be the injected clock reading, not Date.now()');
});

test('staleness: entries are keyed by PATH, not URL, and survive a chip change', async () => {
  // Same reason the flow-control state is path-keyed (see pollKey): the four
  // ?window= endpoints change URL on every chip click, so a URL-keyed map
  // would strand the old entry and start a fresh one each time — exactly the
  // shape in which a wedged endpoint's failure history disappears.
  const fetchImpl = url =>
    (pollKey(url) === COSTS_PATH
      ? Promise.reject(new Error('simulated costs failure'))
      : Promise.resolve({ ok: true, json: async () => ({}) }));
  const { api, window: win } = loadDataJs({ fetchStub: fetchImpl });

  const state = api.createPollState();
  let t = 0;
  const deps = { fetchImpl, now: () => t, random: () => 0, sleep: () => Promise.resolve() };
  const opts = { state, deps, jitterMaxMs: 0 };

  await api.refreshDFData(undefined, opts);           // timer path: failure 1
  assert.equal(win.DF_DATA.__stale[COSTS_PATH].failures, 1);

  // A chip change forces the windowed endpoints past their backoff. A forced
  // attempt that fails deliberately does NOT escalate the timer path's
  // backoff (see recordFailure), so the count holds rather than inflating on
  // user clicks — but the entry must still be there, under the same key.
  await api.refreshDFData('7d', opts);
  assert.equal(win.DF_DATA.__stale[COSTS_PATH].failures, 1);

  t = state.get(COSTS_PATH).nextAllowedAt;
  await api.refreshDFData(undefined, opts);           // timer path: failure 2
  assert.equal(win.DF_DATA.__stale[COSTS_PATH].failures, 2);

  for (const key of Object.keys(win.DF_DATA.__stale)) {
    assert.ok(!key.includes('?'), `__stale key ${key} must be query-stripped (pollKey), not a full URL`);
  }
});

test('staleness: the last success instant survives a later failure', async () => {
  // "Loaded once, but failing for N minutes" is precisely the state __loaded
  // alone cannot express — it flips true on the first success and never back,
  // so during the 19.8h wedge every key read as loaded and current.
  let fail = false;
  const fetchImpl = url =>
    (pollKey(url) === CURATOR_PATH && fail
      ? Promise.reject(new Error('simulated curator failure'))
      : Promise.resolve({ ok: true, json: async () => ({}) }));
  const { api, window: win } = loadDataJs({ fetchStub: fetchImpl });

  const state = api.createPollState();
  let t = 1_000;
  const deps = { fetchImpl, now: () => t, random: () => 0, sleep: () => Promise.resolve() };
  const opts = { state, deps, jitterMaxMs: 0 };

  await api.refreshDFData(undefined, opts);
  const successAt = win.DF_DATA.__stale[CURATOR_PATH].lastSuccessAt;
  assert.equal(successAt, 1_000);

  fail = true;
  t = 500_000;
  await api.refreshDFData(undefined, opts);

  const entry = win.DF_DATA.__stale[CURATOR_PATH];
  assert.equal(entry.failures, 1);
  assert.equal(
    entry.lastSuccessAt,
    successAt,
    'a failure must not overwrite or clear the recorded success instant — the age is derived from it',
  );
});

test('staleness: a recovered endpoint clears its failures AND its notice', async () => {
  // THE MOST LIKELY FAILURE MODE of a staleness indicator is a banner that
  // never clears once the endpoint comes back — an operator who has been
  // taught the indicator lies stops reading it, which costs exactly what the
  // 19.8h wedge cost. Failure -> recovery is also the one path
  // publishStaleness + `st.failures = 0` is not covered on at the PUBLISHED
  // map level the UI actually reads.
  let fail = true;
  const fetchImpl = url =>
    (pollKey(url) === CURATOR_PATH && fail
      ? Promise.reject(new Error('simulated curator failure'))
      : Promise.resolve({ ok: true, json: async () => ({}) }));
  const { api, window: win } = loadDataJs({ fetchStub: fetchImpl });

  const state = api.createPollState();
  let t = 1_000;
  const deps = { fetchImpl, now: () => t, random: () => 0, sleep: () => Promise.resolve() };
  const opts = { state, deps, jitterMaxMs: 0 };

  for (let i = 0; i < STALE_FAILURE_THRESHOLD; i += 1) {
    await api.refreshDFData(undefined, opts);
    t = Math.max(t + 1, state.get(CURATOR_PATH).nextAllowedAt);
  }

  const failing = win.DF_DATA.__stale[CURATOR_PATH];
  assert.equal(failing.failures, STALE_FAILURE_THRESHOLD);
  assert.equal(
    staleNoticesForTab({ tab: 'curator', stale: win.DF_DATA.__stale, now: t }).length, 1,
    'the tab must actually be reporting the endpoint stale before recovery is ' +
      'asserted, or the clearing assertions below prove nothing',
  );

  fail = false;
  await api.refreshDFData(undefined, opts);

  const recovered = win.DF_DATA.__stale[CURATOR_PATH];
  assert.equal(
    recovered.failures, 0,
    `the published failure count stayed at ${recovered.failures} after a 200 — ` +
      'the streak must reset on success, or the indicator is permanent',
  );
  assert.equal(
    recovered.lastSuccessAt, t,
    'the recovery instant must be recorded, or the age keeps growing from the ' +
      'pre-outage success and the notice would return with a stale age',
  );
  assert.deepEqual(
    staleNoticesForTab({ tab: 'curator', stale: win.DF_DATA.__stale, now: t }),
    [],
    'the tab still renders a staleness notice for an endpoint that is serving 200s',
  );
});

test('staleness: applyKey cannot clobber __stale (or __loaded)', () => {
  // No server payload may overwrite the map that reports the server is
  // failing. Two independent layers, both asserted:
  //   STRUCTURAL — `__stale` is not an endpoint key, so the production call
  //     site (`keys.forEach(k => applyKey(k, body[k]))`) can never reach it.
  //   ENFORCED — applyKey itself refuses DF_DATA's `__`-prefixed internal
  //     namespace, so the invariant does not depend on nobody ever naming a
  //     server-side key that way.
  const { api, window: win } = loadDataJs();

  for (const keySpecs of Object.values(api.endpointsFor('24h'))) {
    for (const k of Object.keys(keySpecs)) {
      assert.ok(!k.startsWith('__'), `endpointsFor names an internal key: ${k}`);
    }
  }

  win.DF_DATA.__stale[CURATOR_PATH] = { failures: 7, lastSuccessAt: 42 };
  api.applyKey('__stale', {});
  api.applyKey('__loaded', { PROJECTS: false });

  assert.deepEqual(
    win.DF_DATA.__stale[CURATOR_PATH],
    { failures: 7, lastSuccessAt: 42 },
    'applyKey must not replace the published staleness map',
  );
  assert.deepEqual(win.DF_DATA.__loaded, {}, 'applyKey must not replace the __loaded markers either');
});

test('staleness: past the threshold the per-attempt deadline drops to STALE_TIMEOUT_MS', async () => {
  // THE SECOND-ORDER EFFECT. Three wedged endpoints each holding a socket for
  // the full 30s deadline consumed about half the page's ~6-concurrent-
  // connections-per-origin budget continuously, which is why the HEALTHY tabs
  // also felt sluggish during the incident. Once an endpoint has demonstrated
  // it is failing, there is nothing left to wait 30s for.
  const armed = [];
  const fetchImpl = () => Promise.reject(new Error('simulated failure'));
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  const state = api.createPollState();
  let t = 0;
  const deps = {
    fetchImpl,
    now: () => t,
    random: () => 0,
    sleep: () => Promise.resolve(),
    setTimeoutImpl: (fn, ms) => { armed.push(ms); return armed.length; },
    clearTimeoutImpl: () => {},
  };

  for (let i = 0; i < STALE_FAILURE_THRESHOLD; i += 1) {
    await api.refreshOne(CURATOR_PATH, {}, state, deps);
    t = state.get(CURATOR_PATH).nextAllowedAt;
  }

  assert.equal(armed[0], 30000, 'the FIRST attempt must still use the full 30s deadline');
  assert.equal(state.get(CURATOR_PATH).failures, STALE_FAILURE_THRESHOLD);

  const before = armed.length;
  await api.refreshOne(CURATOR_PATH, {}, state, deps);
  assert.equal(
    armed[before],
    STALE_TIMEOUT_MS,
    `an endpoint at ${STALE_FAILURE_THRESHOLD} consecutive failures must arm a ` +
      `${STALE_TIMEOUT_MS}ms deadline, not 30000 — got ${armed[before]}`,
  );
});

test('staleness: an explicit deps.timeoutMs still wins over both defaults', async () => {
  // The reduced deadline is a DEFAULT selection, not an override, so
  // `deps.timeoutMs ?? ...` must stay the outermost choice EVEN past the
  // threshold — where STALE_TIMEOUT_MS would otherwise be chosen.
  //
  // Asserted behaviourally. The earlier form matched /deps\.timeoutMs\s*\?\?/
  // against DATA_JS_SOURCE, which is a raw readFileSync with no comment
  // stripping — and data.js carries that exact text inside a COMMENT, so
  // deleting the production expression left this test green. The very defect
  // this task found (STALE_TIMEOUT_MS dead in every browser) was a
  // source-looks-right/behaviour-wrong gap, which makes a source regex the
  // wrong instrument for the one claim that is directly executable.
  const armed = [];
  const fetchImpl = () => Promise.reject(new Error('simulated failure'));
  const { api } = loadDataJs({ fetchStub: fetchImpl });

  const state = api.createPollState();
  let t = 0;
  const deps = {
    fetchImpl,
    now: () => t,
    random: () => 0,
    sleep: () => Promise.resolve(),
    setTimeoutImpl: (fn, ms) => { armed.push(ms); return armed.length; },
    clearTimeoutImpl: () => {},
  };

  for (let i = 0; i < STALE_FAILURE_THRESHOLD; i += 1) {
    await api.refreshOne(CURATOR_PATH, {}, state, deps);
    t = state.get(CURATOR_PATH).nextAllowedAt;
  }
  assert.equal(state.get(CURATOR_PATH).failures, STALE_FAILURE_THRESHOLD);

  const before = armed.length;
  await api.refreshOne(CURATOR_PATH, {}, state, { ...deps, timeoutMs: 1234 });
  assert.equal(
    armed[before], 1234,
    `an explicitly injected deps.timeoutMs must win even past the threshold; ` +
      `got ${armed[before]} (${STALE_TIMEOUT_MS} means the reduced default ` +
      'overrode the caller, 30000 means the full default did)',
  );
});

test('staleness: DEFAULT_TIMEOUT_MS survives untouched as a parsable literal', () => {
  // TWO Python structural tests parse this constant out of the shipped source
  // with exactly this regex — test_tasks_budget.py and
  // test_fetch_tasks_whole_operation_budget.py — where it is the ONLY ceiling
  // on the server-side budgets. A rename or a computed expression makes both
  // fail loudly, which is why STALE_TIMEOUT_MS had to be a NEW, separately
  // named constant rather than a redefinition of this one.
  const match = DATA_JS_SOURCE.match(/DEFAULT_TIMEOUT_MS\s*=\s*(\d+)/);
  assert.ok(match, 'DEFAULT_TIMEOUT_MS is no longer a literal assignment in data.js');
  assert.equal(Number(match[1]), 30000);

  const stale = DATA_JS_SOURCE.match(/STALE_TIMEOUT_MS\s*=\s*(\d+)/);
  assert.ok(stale, 'STALE_TIMEOUT_MS must be its own named literal constant');
  assert.equal(Number(stale[1]), STALE_TIMEOUT_MS);
  assert.notEqual(
    Number(stale[1]),
    Number(match[1]),
    'the reduced deadline must actually be shorter than the default one',
  );
});

test('staleness: the reduced deadline is reached through the PRODUCTION deps merge, not only a hand-built deps', async () => {
  // REGRESSION FENCE for a defect the sibling test above structurally could
  // NOT catch, found by step-19's real-browser check (task 4884, #4791).
  //
  // WHAT WAS MEASURED. Chrome 151 headless against the worktree's dashboard,
  // /api/v2/dashboard/merge-queue wedged with `await asyncio.Event().wait()`.
  // Correlating Network.requestWillBeSent with Network.loadingFailed gave four
  // consecutive aborts at 30008 / 30164 / 30001 / 29981 ms — including the
  // attempt that STARTED at failures === 3, which had to arm 5000. The banner
  // rendered ("4 consecutive attempts failed"), so the threshold was crossed;
  // only the deadline never dropped.
  //
  // WHY THE OTHER TEST PASSED ANYWAY. It calls refreshOne with a deps object
  // it builds by hand, and that object has no `timeoutMs` key, so
  // `deps.timeoutMs ?? (...)` falls through to the failures-based selection.
  // Production never takes that path: refreshDFData merges DEFAULT_POLL_DEPS
  // FIRST, and pinning `timeoutMs` there made `deps.timeoutMs` permanently
  // 30000 — the ?? could never fall through, and the reduced deadline was
  // dead code in every browser while green in the harness.
  //
  // So this test drives refreshDFData (the merge site) and injects everything
  // EXCEPT timeoutMs, reproducing the production shape exactly. A deps default
  // that re-pins timeoutMs turns it red.
  const armed = [];
  const fetchStub = () => Promise.reject(new Error('simulated failure'));
  const { api } = loadDataJs({ fetchStub });

  const state = api.createPollState();
  let t = 0;
  const deps = {
    now: () => t,
    random: () => 0,
    sleep: () => Promise.resolve(),
    setTimeoutImpl: (fn, ms) => { armed.push(ms); return armed.length; },
    clearTimeoutImpl: () => {},
    // DELIBERATELY NO timeoutMs — that is the whole point of this test.
  };
  const cycle = async () => {
    await api.refreshDFData(undefined, { state, deps, jitterMaxMs: 0 });
    // Clear backoff the way real wall-clock time does, so the next cycle is
    // an ATTEMPT rather than a skip.
    for (const st of state.values()) t = Math.max(t, st.nextAllowedAt);
  };

  for (let i = 0; i < STALE_FAILURE_THRESHOLD; i += 1) await cycle();

  assert.equal(
    armed[0], 30000,
    'the FIRST attempt must still use the full 30s deadline through the production merge',
  );
  for (const st of state.values()) {
    assert.ok(
      st.failures >= STALE_FAILURE_THRESHOLD,
      `every endpoint must be past the threshold before the assertion below; got ${st.failures}`,
    );
  }

  const before = armed.length;
  await cycle();
  const past = armed.slice(before);
  assert.ok(past.length > 0, 'the next cycle must arm at least one deadline');
  for (const ms of past) {
    assert.equal(
      ms, STALE_TIMEOUT_MS,
      'an endpoint past the threshold must arm a ' + STALE_TIMEOUT_MS + 'ms deadline through ' +
        'the PRODUCTION deps merge, not 30000 — measured 4 consecutive ~30000ms aborts in ' +
        'Chrome 151 because DEFAULT_POLL_DEPS pinned timeoutMs',
    );
  }

  // The source-level trap, stated separately so the failure names the cause
  // rather than only the symptom: DEFAULT_POLL_DEPS must not pin `timeoutMs`.
  // Every other DEFAULT_POLL_DEPS entry is a genuine environment capability
  // (a clock, an RNG, fetch, the timer pair); `timeoutMs` is a POLICY value
  // the selection below it is supposed to choose, and pinning a policy in the
  // defaults is what silently disabled it.
  // Matched to the TERMINATING `};` at line start, not to the first `}`. The
  // `[^}]*` form matched the whole literal only because every value in it
  // happens to be brace-free today: one block-bodied arrow (or any object
  // value) would truncate the capture and silently make the fence below
  // vacuous rather than fail it.
  const defaults = DATA_JS_SOURCE.match(/const DEFAULT_POLL_DEPS = \{[\s\S]*?\n\};/);
  assert.ok(defaults, 'DEFAULT_POLL_DEPS must remain a greppable object literal');
  for (const key of ['now', 'random', 'sleep', 'fetchImpl', 'setTimeoutImpl', 'clearTimeoutImpl']) {
    assert.ok(
      defaults[0].includes(key),
      `the captured DEFAULT_POLL_DEPS block is missing ${key} — the match ` +
        'truncated, so the timeoutMs fence below would be checking a fragment',
    );
  }
  assert.ok(
    !/timeoutMs/.test(defaults[0]),
    'DEFAULT_POLL_DEPS must NOT pin timeoutMs — doing so makes `deps.timeoutMs ?? ...` ' +
      'unreachable in the browser and turns STALE_TIMEOUT_MS into dead code (measured: ' +
      '4 consecutive ~30000ms aborts in Chrome 151 with the banner already rendered)',
  );
});

// ---------------------------------------------------------------------------
// The datum registry (task 5588, PRD leaf gamma1)
//
// endpointsFor's rows become endpoint -> {KEY: SPEC}, where a spec declares
// whether the wire delivers that key as a bare value or as a Datum envelope.
// Every polled key is 'plain', which is the HONEST description of the wire:
// the Datums PRD leaf beta serves arrive NESTED inside each TASKS_SNAPSHOT
// entry, and the 'datum' kind validates only a TOP-LEVEL envelope. The
// registry is the single place a later leaf flips a row.
// ---------------------------------------------------------------------------

test('registry: every endpoint row maps key names to declared specs', () => {
  const { api } = loadDataJs();
  const rows = api.endpointsFor('24h');

  assert.equal(Object.keys(rows).length, EXPECTED_ENDPOINT_COUNT);
  for (const [url, keySpecs] of Object.entries(rows)) {
    assert.ok(
      keySpecs && typeof keySpecs === 'object' && !Array.isArray(keySpecs),
      `${url} must map key names to specs, not list them`,
    );
    for (const [key, spec] of Object.entries(keySpecs)) {
      assert.ok(spec && typeof spec === 'object', `${url}/${key} has no spec`);
      assert.ok(['datum', 'plain'].includes(spec.kind), `${url}/${key} kind ${spec.kind}`);
    }
  }
});

test('registry: the reshape drops no key — the union is exactly today\'s set', () => {
  const { api } = loadDataJs();
  const seen = new Set();
  for (const keySpecs of Object.values(api.endpointsFor('24h'))) {
    for (const key of Object.keys(keySpecs)) seen.add(key);
  }
  assert.deepEqual([...seen].sort(), EXPECTED_ENDPOINT_KEYS.slice().sort());
});

test('registry: every polled key is plain, because beta nests its Datums inside TASKS_SNAPSHOT', () => {
  // Not an aspiration — a description. Beta does serve Datums, but each one
  // sits inside a TASKS_SNAPSHOT entry (census, rows), and TASKS_SNAPSHOT
  // itself is a map of project -> entry, never a five-key envelope. Declaring
  // any polled row 'datum' would make applyKey refuse every real payload and
  // freeze that tab at its seed values, which looks exactly like a wedged
  // endpoint.
  const { api } = loadDataJs();
  const rows = api.endpointsFor('24h');
  for (const [url, keySpecs] of Object.entries(rows)) {
    for (const [key, spec] of Object.entries(keySpecs)) {
      assert.equal(spec.kind, 'plain', `${url}/${key} is declared datum-kinded`);
    }
  }
  // Named explicitly as well: this is the row a reader of "beta serves Datums"
  // is most likely to flip, and flipping it freezes both done-count pips.
  assert.equal(rows['/api/v2/dashboard/tasks'].TASKS_SNAPSHOT.kind, 'plain');
});

test('applyKey: a plain-kinded key still applies verbatim, in place for the stable arrays', () => {
  const { api, window: win } = loadDataJs();
  const projectsRef = win.DF_DATA.PROJECTS;

  api.applyKey('PROJECTS', [{ id: 'p1' }], PLAIN_SPEC, { servedAt: null, receivedAt: 5 });
  assert.equal(win.DF_DATA.PROJECTS, projectsRef, 'the captured array reference must survive');
  assert.deepEqual(win.DF_DATA.PROJECTS, [{ id: 'p1' }]);

  const costs = { summary: { total: 42 } };
  api.applyKey('COSTS', costs, PLAIN_SPEC, { servedAt: null, receivedAt: 5 });
  assert.equal(win.DF_DATA.COSTS, costs, 'a plain value is stored verbatim, receipt and all');
});

test('applyKey: a datum-kinded payload is stored as a COPY carrying its receipt', () => {
  const { api, window: win } = loadDataJs();
  const pristine = { ...SERVED_DATUM };

  api.applyKey(SYNTHETIC_DATUM_KEY, SERVED_DATUM, DATUM_SPEC, { servedAt: 'S', receivedAt: 1234 });

  const stored = win.DF_DATA[SYNTHETIC_DATUM_KEY];
  assert.notEqual(stored, SERVED_DATUM, 'the wire payload must not be stored by reference');
  assert.deepEqual(SERVED_DATUM, pristine, 'the wire payload was mutated');
  assert.equal(stored._served_at, 'S');
  assert.equal(stored._received_at, 1234);
  assert.equal(stored.value.total, 9);
  assert.equal(win.DF_DATA.__loaded[SYNTHETIC_DATUM_KEY], true);
});

// Runs *fn* with console.warn captured, and hands back what it said. Every
// refusal below is EXPECTED to warn, so the capture keeps the suite's output
// readable — and makes the warning itself assertable rather than mere noise.
function warningsFrom(fn) {
  const original = console.warn;
  const calls = [];
  console.warn = (...args) => calls.push(args);
  try {
    fn();
  } finally {
    console.warn = original;
  }
  return calls;
}

test('applyKey: a datum-kinded payload that is NOT a Datum is refused, prior value kept', () => {
  // The half that matters. A server regression that starts sending a bare
  // number where a Datum was declared must leave the last good envelope on
  // screen — with its age badge still growing — rather than replacing it with
  // an unprovenanced number that renders as though freshly measured.
  const { api, window: win } = loadDataJs();
  const receipt = { servedAt: null, receivedAt: 1 };
  api.applyKey(SYNTHETIC_DATUM_KEY, SERVED_DATUM, DATUM_SPEC, receipt);
  const good = win.DF_DATA[SYNTHETIC_DATUM_KEY];

  for (const bad of [42, 'nine', [SERVED_DATUM], { value: 1, as_of: null, state: 'fresh', reason: null }]) {
    warningsFrom(() => api.applyKey(SYNTHETIC_DATUM_KEY, bad, DATUM_SPEC, { servedAt: null, receivedAt: 2 }));
    assert.equal(win.DF_DATA[SYNTHETIC_DATUM_KEY], good, `a non-Datum (${JSON.stringify(bad)}) was applied`);
  }
});

test('applyKey: a refusal SAYS SO, naming the key', () => {
  // Refusing in silence would make a schema break pixel-identical to a wedged
  // endpoint: in both cases the key's tiles simply keep ageing, and an operator
  // would diagnose a network outage for a server that is answering perfectly.
  // The value must still not be applied — only the diagnosis was missing.
  const { api, window: win } = loadDataJs();

  const calls = warningsFrom(() =>
    api.applyKey(SYNTHETIC_DATUM_KEY, 42, DATUM_SPEC, { servedAt: null, receivedAt: 1 }),
  );

  assert.equal(calls.length, 1, 'a refused datum payload must warn exactly once');
  assert.ok(/DF_DATA/.test(String(calls[0][0])), `the warning must name the source: ${calls[0][0]}`);
  assert.ok(
    calls[0].includes(SYNTHETIC_DATUM_KEY),
    `the warning must name the refused key, got ${JSON.stringify(calls[0])}`,
  );
  assert.equal(win.DF_DATA[SYNTHETIC_DATUM_KEY], undefined, 'the refused value must still not be applied');
});

test('applyKey: a plain-kinded key is never second-guessed, and never warns', () => {
  // The warning is scoped to a DECLARED datum row receiving a non-Datum. Every
  // polled key is plain, so a warn on the plain path would fire on every
  // healthy poll and train an operator to ignore it.
  const { api } = loadDataJs();
  const calls = warningsFrom(() => api.applyKey('COSTS', 42, PLAIN_SPEC, { servedAt: null, receivedAt: 1 }));
  assert.deepEqual(calls, []);
});

test('applyKey: a refused datum payload does not flip the __loaded marker', () => {
  const { api, window: win } = loadDataJs();
  warningsFrom(() => api.applyKey(SYNTHETIC_DATUM_KEY, 42, DATUM_SPEC, { servedAt: null, receivedAt: 1 }));
  assert.equal(win.DF_DATA.__loaded[SYNTHETIC_DATUM_KEY], undefined, '__loaded must mean a real value LANDED');
});

test('applyKey: __receipt is refused exactly like __loaded and __stale', () => {
  const { api, window: win } = loadDataJs();
  win.DF_DATA.__receipt[CURATOR_PATH] = { servedAt: null, receivedAt: 99 };
  api.applyKey('__receipt', {});
  assert.deepEqual(win.DF_DATA.__receipt[CURATOR_PATH], { servedAt: null, receivedAt: 99 });
});

test('datumFor: unknown before the first apply, the stored Datum after', () => {
  const { api, window: win } = loadDataJs();

  const before = api.datumFor(SYNTHETIC_DATUM_KEY);
  assert.equal(before.state, 'unknown');
  assert.equal(before.reason, 'not yet fetched');

  api.applyKey(SYNTHETIC_DATUM_KEY, SERVED_DATUM, DATUM_SPEC, { servedAt: 'S', receivedAt: 7 });
  assert.equal(api.datumFor(SYNTHETIC_DATUM_KEY), win.DF_DATA[SYNTHETIC_DATUM_KEY]);
});

// ---------------------------------------------------------------------------
// __receipt — published on SUCCESS ONLY
// ---------------------------------------------------------------------------

function okResponse(body) {
  return () => Promise.resolve({ ok: true, json: async () => body });
}

test('receipts: a successful refresh records servedAt from the body and receivedAt from the clock', async () => {
  const { api, window: win } = loadDataJs();
  const deps = { fetchImpl: okResponse({ served_at: '2026-09-20T12:00:00+00:00' }), now: () => 555 };

  await api.refreshOne(CURATOR_PATH, {}, api.createPollState(), deps);

  assert.deepEqual(win.DF_DATA.__receipt[CURATOR_PATH], {
    servedAt: '2026-09-20T12:00:00+00:00',
    receivedAt: 555,
  });
});

test('receipts: a body with no served_at records null, never undefined', async () => {
  // Today's wire for every endpoint. `null` is a stated absence that plainDatum
  // can branch on; `undefined` would read as a malformed receipt.
  const { api, window: win } = loadDataJs();
  const deps = { fetchImpl: okResponse({ CURATOR_STATE: {} }), now: () => 42 };

  await api.refreshOne(CURATOR_PATH, {}, api.createPollState(), deps);

  assert.deepEqual(win.DF_DATA.__receipt[CURATOR_PATH], { servedAt: null, receivedAt: 42 });
});

test('receipts: a FAILED refresh leaves the receipt alone, so the tiles keep ageing', async () => {
  // The single most important property of this map, and the reason it is not
  // merged into __stale: publishStaleness runs in `finally` BY DESIGN, so a
  // 503 counts exactly like a timeout. A receipt advanced on failure would
  // reset every tile's age to zero on each failed poll — the dashboard would
  // look freshest precisely while it was most wedged.
  for (const fetchImpl of [
    () => Promise.reject(new Error('boom')),
    () => Promise.resolve({ ok: false, status: 503, json: async () => ({}) }),
  ]) {
    const { api, window: win } = loadDataJs();
    const state = api.createPollState();
    await api.refreshOne(CURATOR_PATH, {}, state, { fetchImpl: okResponse({}), now: () => 100 });
    const afterSuccess = win.DF_DATA.__receipt[CURATOR_PATH];

    const originalWarn = console.warn;
    console.warn = () => {};
    try {
      await api.refreshOne(CURATOR_PATH, {}, state, { fetchImpl, now: () => 900 });
    } finally {
      console.warn = originalWarn;
    }

    assert.deepEqual(win.DF_DATA.__receipt[CURATOR_PATH], afterSuccess, 'the receipt advanced on a failure');
    assert.equal(win.DF_DATA.__receipt[CURATOR_PATH].receivedAt, 100);
  }
});

// NOT forced with ignoreBackoff: recordFailure deliberately ignores a forced
// attempt, so `failures` would never move and this test would assert nothing.
// A plain second call is allowed anyway — the preceding success reset
// nextAllowedAt to 0.
test('receipts: __stale still advances on failure — the two maps are not one', async () => {
  // Stated alongside the test above so the asymmetry is visible in one place:
  // __stale records ATTEMPT history (and must move), __receipt records the
  // PROVENANCE of the values now in DF_DATA (and must not).
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();
  await api.refreshOne(CURATOR_PATH, {}, state, { fetchImpl: okResponse({}), now: () => 100 });

  const originalWarn = console.warn;
  console.warn = () => {};
  try {
    await api.refreshOne(CURATOR_PATH, {}, state, {
      fetchImpl: () => Promise.reject(new Error('boom')),
      now: () => 900,
    });
  } finally {
    console.warn = originalWarn;
  }

  assert.equal(win.DF_DATA.__stale[CURATOR_PATH].failures, 1);
  assert.equal(win.DF_DATA.__receipt[CURATOR_PATH].receivedAt, 100);
});

// ---------------------------------------------------------------------------
// ON_DEMAND_KEYS / requestOnDemand — per-project parameterised keys
//
// The mechanism PRD leaf gamma3 fetches `?terminal=<project>` through. Two
// properties carry the whole design.
//
// ONE NAME PER KEY. A declared row's key builder names BOTH what the response
// body calls the value and what DF_DATA calls it — exactly the rule every
// polled row already follows, where a row's key name is simultaneously the
// body key and the DF_DATA key. The only difference here is that the name is
// BUILT from a parameter instead of written as a literal, which is why the row
// carries builders rather than strings, and why nothing re-derives either one
// by string surgery at a call site.
//
// ISOLATION FROM THE POLL LOOP. pollKey strips the query string on purpose, so
// an on-demand `/api/v2/dashboard/tasks?terminal=<p>` would otherwise land on
// the POLLED `/api/v2/dashboard/tasks` flow-control entry: it would set that
// endpoint's in-flight flag (so the poll loop skips the real tasks fetch for
// as long as a user's terminal request runs), reset or escalate its backoff,
// and write its __stale entry — corrupting the banner with a failure the
// polled endpoint never had. A user action must not be able to blind a tab.
// ---------------------------------------------------------------------------

const TASKS_PATH = '/api/v2/dashboard/tasks';
const TERMINAL_PROJECT = 'dark-factory';
const TERMINAL_KEY = `TASKS_TERMINAL:${TERMINAL_PROJECT}`;
// The flow-control/staleness key an on-demand request must use instead of the
// polled path. Stated as a literal, not built from the row, so the separation
// is pinned against an expectation rather than against itself.
const TERMINAL_STATE_KEY = `${TASKS_PATH}#terminal:${TERMINAL_PROJECT}`;

// A Datum as the terminal endpoint serves one: a lower_bound, because a
// terminal listing is truncated by construction, whose value is the row LIST
// itself — the PRD's `Datum[list]`, and what api/tasks.py puts on the wire.
const SERVED_TERMINAL_DATUM = Object.freeze({
  value: [{ id: '5588' }],
  as_of: '2026-09-20T09:00:00+00:00',
  state: 'lower_bound',
  reason: 'terminal window truncated at 200 rows',
  freshness_bound_seconds: 30,
});

// Responds with the Datum under whatever key the request asked for, so the
// fixture cannot accidentally hard-code the key the implementation is supposed
// to build. `served_at` is present so the receipt has both halves.
function terminalResponse(datum = SERVED_TERMINAL_DATUM, servedAt = '2026-09-20T09:00:01+00:00') {
  return url => {
    const project = decodeURIComponent((url.split('terminal=')[1] || '').split('&')[0]);
    return Promise.resolve({
      ok: true,
      json: async () => ({ served_at: servedAt, [`TASKS_TERMINAL:${project}`]: datum }),
    });
  };
}

test('on-demand: the terminal row declares its url builder, its key builder and a datum spec', () => {
  const { api } = loadDataJs();
  const row = api.ON_DEMAND_KEYS.terminal;

  assert.ok(row, 'ON_DEMAND_KEYS.terminal must be declared');
  assert.equal(typeof row.url, 'function', 'the row must BUILD its url, not carry a template string');
  assert.equal(typeof row.key, 'function', 'the row must BUILD its key, not carry a template string');
  assert.equal(row.url(TERMINAL_PROJECT), `${TASKS_PATH}?terminal=${TERMINAL_PROJECT}`);
  assert.equal(row.key(TERMINAL_PROJECT), TERMINAL_KEY);
  assert.equal(row.spec.kind, 'datum', 'the terminal endpoint serves a Datum, unlike every polled row');
});

test('on-demand: one request, one fetch, and a validated Datum under the built key', async () => {
  const { api, window: win } = loadDataJs();

  const fetchUrls = [];
  const inner = terminalResponse();
  const deps = { fetchImpl: url => { fetchUrls.push(url); return inner(url); }, now: () => 4242 };

  await api.requestOnDemand('terminal', TERMINAL_PROJECT, { state: api.createPollState(), deps });

  assert.deepEqual(fetchUrls, [`${TASKS_PATH}?terminal=${TERMINAL_PROJECT}`], 'exactly one fetch, to the built url');

  const stored = win.DF_DATA[TERMINAL_KEY];
  assert.ok(stored, `nothing was applied to DF_DATA['${TERMINAL_KEY}']`);
  assert.equal(stored.state, 'lower_bound');
  assert.deepEqual(stored.value, [{ id: '5588' }]);
  assert.equal(stored._served_at, '2026-09-20T09:00:01+00:00', 'the receipt must come from the body');
  assert.equal(stored._received_at, 4242, 'the receipt must come from the injected clock');
  assert.notEqual(stored, SERVED_TERMINAL_DATUM, 'the wire payload must not be stored by reference');
});

test('on-demand: a non-Datum body is refused, so no unprovenanced value reaches a terminal key', async () => {
  // Same guarantee applyKey gives every datum-kinded polled row; asserted here
  // because this is the FIRST row declared datum-kinded, so it is the first
  // path on which the refusal is reachable at all. The payload is the served
  // value stripped of its envelope — the likeliest shape of the regression.
  const { api, window: win } = loadDataJs();
  const deps = {
    fetchImpl: () => Promise.resolve({
      ok: true,
      json: async () => ({ served_at: null, [TERMINAL_KEY]: SERVED_TERMINAL_DATUM.value }),
    }),
    now: () => 1,
  };

  await api.requestOnDemand('terminal', TERMINAL_PROJECT, { state: api.createPollState(), deps });

  assert.equal(win.DF_DATA[TERMINAL_KEY], undefined, 'a bare payload was applied to a datum-kinded key');
});

test('on-demand: a second project gets its own key and leaves the first alone', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();
  const deps = { fetchImpl: terminalResponse(), now: () => 10 };

  await api.requestOnDemand('terminal', TERMINAL_PROJECT, { state, deps });
  const first = win.DF_DATA[TERMINAL_KEY];

  await api.requestOnDemand('terminal', 'other-project', {
    state,
    deps: { fetchImpl: terminalResponse(), now: () => 20 },
  });

  assert.equal(win.DF_DATA[TERMINAL_KEY], first, "the first project's Datum was overwritten");
  assert.equal(win.DF_DATA[TERMINAL_KEY]._received_at, 10);
  assert.equal(win.DF_DATA['TASKS_TERMINAL:other-project']._received_at, 20);
});

test('on-demand: a project name needing escaping is URL-encoded in the url and left literal in the key', () => {
  // The two halves diverge on purpose: the url must survive HTTP parsing, and
  // the DF_DATA key must be the name a caller can look up with the project
  // string it already holds — no call site should have to know which of the
  // two is encoded.
  const { api } = loadDataJs();
  const row = api.ON_DEMAND_KEYS.terminal;
  const messy = 'a b/c&d=e?f';

  assert.equal(row.url(messy), `${TASKS_PATH}?terminal=${encodeURIComponent(messy)}`);
  assert.ok(!/[ &?]/.test(row.url(messy).split('terminal=')[1]), 'the encoded param leaked a delimiter');
  assert.equal(row.key(messy), `TASKS_TERMINAL:${messy}`);
});

test('on-demand: flow-control and staleness are recorded under the request\'s OWN key', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();

  await api.requestOnDemand('terminal', TERMINAL_PROJECT, {
    state,
    deps: { fetchImpl: terminalResponse(), now: () => 77 },
  });

  assert.ok(state.get(TERMINAL_STATE_KEY), `no flow-control entry at ${TERMINAL_STATE_KEY}`);
  assert.equal(state.get(TASKS_PATH), undefined, 'the on-demand request took over the POLLED tasks entry');
  assert.ok(win.DF_DATA.__stale[TERMINAL_STATE_KEY], 'the on-demand request published no staleness of its own');
  assert.equal(win.DF_DATA.__stale[TASKS_PATH], undefined, "the on-demand request wrote the polled endpoint's __stale");
  assert.deepEqual(win.DF_DATA.__receipt[TERMINAL_STATE_KEY], { servedAt: '2026-09-20T09:00:01+00:00', receivedAt: 77 });
  assert.equal(win.DF_DATA.__receipt[TASKS_PATH], undefined, "the on-demand receipt landed on the polled endpoint's path");
});

test('on-demand: a poll of /tasks still runs while a terminal request is in flight', async () => {
  // The isolation property stated as the operator sees it: a user opening a
  // terminal must not stop the Tasks tab from updating. Shared `state` Map, so
  // a collision would be real rather than hypothetical.
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();

  let releaseTerminal;
  const terminalGate = new Promise(resolve => { releaseTerminal = resolve; });
  const polledUrls = [];

  const pending = api.requestOnDemand('terminal', TERMINAL_PROJECT, {
    state,
    deps: {
      now: () => 100,
      fetchImpl: () => terminalGate.then(() => ({
        ok: true,
        json: async () => ({ served_at: null, [TERMINAL_KEY]: SERVED_TERMINAL_DATUM }),
      })),
    },
  });
  await drain();

  assert.equal(state.get(TERMINAL_STATE_KEY).inFlight, true, 'the held request should be in flight');
  assert.notEqual(state.get(TASKS_PATH)?.inFlight, true, 'the POLLED tasks endpoint was marked in flight');

  await api.refreshOne(TASKS_PATH, { ACTIVE_TASKS: PLAIN_SPEC }, state, {
    fetchImpl: url => { polledUrls.push(url); return Promise.resolve({ ok: true, json: async () => ({ ACTIVE_TASKS: [{ id: '1' }] }) }); },
    now: () => 101,
  });

  assert.deepEqual(polledUrls, [TASKS_PATH], 'the polled tasks fetch was skipped while the terminal request ran');
  assert.deepEqual(win.DF_DATA.ACTIVE_TASKS, [{ id: '1' }]);

  releaseTerminal();
  await pending;
  assert.equal(win.DF_DATA[TERMINAL_KEY]._received_at, 100, 'the terminal Datum still landed afterwards');
});

test('on-demand: a failed request never escalates the polled tasks backoff', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();

  const originalWarn = console.warn;
  console.warn = () => {};
  try {
    await api.requestOnDemand('terminal', TERMINAL_PROJECT, {
      state,
      deps: { fetchImpl: () => Promise.reject(new Error('boom')), now: () => 500 },
    });
  } finally {
    console.warn = originalWarn;
  }

  assert.equal(state.get(TERMINAL_STATE_KEY).failures, 1, "the request's own entry should count the failure");
  assert.equal(state.get(TASKS_PATH), undefined, 'a failed user action created backoff for the polled endpoint');
  assert.equal(win.DF_DATA.__stale[TASKS_PATH], undefined, 'a failed user action reported the polled endpoint stale');
});

test('on-demand: a failed request leaves a previously applied terminal Datum untouched', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();

  await api.requestOnDemand('terminal', TERMINAL_PROJECT, {
    state,
    deps: { fetchImpl: terminalResponse(), now: () => 300 },
  });
  const good = win.DF_DATA[TERMINAL_KEY];

  const originalWarn = console.warn;
  console.warn = () => {};
  try {
    for (const fetchImpl of [
      () => Promise.reject(new Error('boom')),
      () => Promise.resolve({ ok: false, status: 503, json: async () => ({}) }),
    ]) {
      await api.requestOnDemand('terminal', TERMINAL_PROJECT, {
        state,
        deps: { fetchImpl, now: () => 900, ignoreBackoff: true },
      });
      assert.equal(win.DF_DATA[TERMINAL_KEY], good, 'a failed retry replaced the last good Datum');
    }
  } finally {
    console.warn = originalWarn;
  }

  assert.equal(win.DF_DATA[TERMINAL_KEY]._received_at, 300, 'the stored Datum must keep ageing from its own receipt');
});

test('on-demand: datumFor a terminal key is unknown/not yet fetched before the request', async () => {
  const { api } = loadDataJs();

  const before = api.datumFor(TERMINAL_KEY);
  assert.equal(before.state, 'unknown');
  assert.equal(before.reason, 'not yet fetched');

  await api.requestOnDemand('terminal', TERMINAL_PROJECT, {
    state: api.createPollState(),
    deps: { fetchImpl: terminalResponse(), now: () => 8 },
  });

  assert.equal(api.datumFor(TERMINAL_KEY).state, 'lower_bound');
});

test('on-demand: an undeclared name is refused loudly rather than fetched', async () => {
  // Nothing re-derives a url or a key by string surgery at a call site, so an
  // unrecognised name has no url to build. Failing loudly here is what keeps
  // that true — a silent no-op would let a typo look like an empty result.
  const { api } = loadDataJs();
  const fetchUrls = [];

  await assert.rejects(
    () => api.requestOnDemand('termnial', TERMINAL_PROJECT, {
      state: api.createPollState(),
      deps: { fetchImpl: url => { fetchUrls.push(url); return terminalResponse()(url); }, now: () => 1 },
    }),
    /termnial/,
  );
  assert.deepEqual(fetchUrls, [], 'an undeclared name must not reach the network');
});

// ---------------------------------------------------------------------------
// REFRESH_OUTCOMES — what one attempt DID
//
// The poll loop ignores this: the next tick retries, so there is nothing for
// it to decide. It exists for a USER ACTION. Without it `await
// requestOnDemand(...)` resolves identically whether the Datum landed, the
// server 503'd, or the key was still inside its backoff window and nothing was
// even asked — and datumFor reports the same pre-request unknown Datum in all
// three cases, so the gamma3 terminal UI could only spin.
// ---------------------------------------------------------------------------

test('outcomes: the vocabulary is a closed, frozen set', () => {
  // Named constants rather than four hand-typed strings at the call sites that
  // compare against them — a misspelled `REFRESH_OUTCOMES.x` is `undefined` at
  // the comparison, not a branch that silently never runs.
  const { api } = loadDataJs();

  assert.deepEqual(api.REFRESH_OUTCOMES, {
    applied: 'applied',
    failed: 'failed',
    skippedInFlight: 'skipped-inflight',
    skippedBackoff: 'skipped-backoff',
  });
  assert.ok(Object.isFrozen(api.REFRESH_OUTCOMES));
});

test('outcomes: a landed response reports `applied`', async () => {
  const { api } = loadDataJs();
  const outcome = await api.refreshOne(CURATOR_PATH, {}, api.createPollState(), {
    fetchImpl: okResponse({ CURATOR_STATE: {} }),
    now: () => 1,
  });
  assert.equal(outcome, api.REFRESH_OUTCOMES.applied);
});

test('outcomes: a non-ok status and a thrown fetch both report `failed`', async () => {
  // Distinct code paths — the `!resp.ok` early return and the catch arm — and
  // one outcome, because a caller can act on neither differently: the rows did
  // not arrive.
  for (const fetchImpl of [
    () => Promise.resolve({ ok: false, status: 503, json: async () => ({}) }),
    () => Promise.reject(new Error('boom')),
  ]) {
    const { api } = loadDataJs();
    let outcome;
    const original = console.warn;
    console.warn = () => {};
    try {
      outcome = await api.refreshOne(CURATOR_PATH, {}, api.createPollState(), { fetchImpl, now: () => 1 });
    } finally {
      console.warn = original;
    }
    assert.equal(outcome, api.REFRESH_OUTCOMES.failed);
  }
});

test('outcomes: the two skips are told apart from each other and from a failure', async () => {
  // The distinction the gamma3 UI needs most: "we did not even ask" is neither
  // an error to report nor rows to draw, and both skips return before any
  // fetch is issued, so nothing else in the response tells them apart.
  const { api } = loadDataJs();
  const state = api.createPollState();
  const fetchUrls = [];
  const deps = {
    fetchImpl: url => { fetchUrls.push(url); return new Promise(() => {}); },
    now: () => 1,
  };

  const pending = api.refreshOne(CURATOR_PATH, {}, state, deps);
  assert.equal(await api.refreshOne(CURATOR_PATH, {}, state, deps), api.REFRESH_OUTCOMES.skippedInFlight);
  assert.equal(fetchUrls.length, 1, 'the in-flight skip must not issue a second request');

  const backedOff = api.createPollState();
  backedOff.set(CURATOR_PATH, { inFlight: false, failures: 3, nextAllowedAt: 9_000, lastSuccessAt: 0 });
  assert.equal(
    await api.refreshOne(CURATOR_PATH, {}, backedOff, deps),
    api.REFRESH_OUTCOMES.skippedBackoff,
  );
  assert.equal(fetchUrls.length, 1, 'the backoff skip must not issue a request either');

  void pending;
});

test('outcomes: requestOnDemand propagates refreshOne\'s answer verbatim', async () => {
  const { api } = loadDataJs();

  const applied = await api.requestOnDemand('terminal', TERMINAL_PROJECT, {
    state: api.createPollState(),
    deps: { fetchImpl: terminalResponse(), now: () => 1 },
  });
  assert.equal(applied, api.REFRESH_OUTCOMES.applied);

  const original = console.warn;
  console.warn = () => {};
  let failed;
  try {
    failed = await api.requestOnDemand('terminal', TERMINAL_PROJECT, {
      state: api.createPollState(),
      deps: { fetchImpl: () => Promise.reject(new Error('boom')), now: () => 1 },
    });
  } finally {
    console.warn = original;
  }
  assert.equal(failed, api.REFRESH_OUTCOMES.failed);
});

test('outcomes: the poll loop ignores them — refreshDFData still resolves to undefined', async () => {
  // Nothing about the loop changes. Stated as a test because "poll-loop callers
  // ignore the return" is the premise that makes this addition safe.
  const { api } = loadDataJs();
  const result = await api.refreshDFData(undefined, {
    state: api.createPollState(),
    jitterMaxMs: 0,
    deps: { fetchImpl: okResponse({}), now: () => 1 },
  });
  assert.equal(result, undefined);
});

// ---------------------------------------------------------------------------
// taskProse — the Task Detail pane's description/details, fetched per selection
//
// The ACTIVE_TASKS rows no longer carry either field (task 5815); the pane asks
// for the ONE selected task through the same on-demand seam the terminal row
// uses. The row is addressed by the row's own uid (`<project>/T-<id>`), whose
// segments are encoded one by one so the '/' between them stays a path
// separator. PLAIN, not DATUM: prose is not a measurement.
// ---------------------------------------------------------------------------

const TASK_PROSE_PREFIX = '/api/v2/dashboard/task/';
const PROSE_UID = 'dark-factory/T-19';
const PROSE_KEY = `TASK_PROSE:${PROSE_UID}`;
// Stated as a literal, not built from the row, for the same reason as
// TERMINAL_STATE_KEY: the isolation is pinned against an expectation.
const PROSE_STATE_KEY = '/api/v2/dashboard/task/dark-factory/T-19#taskProse:dark-factory/T-19';
const SERVED_PROSE = Object.freeze({ description: 'why', details: 'how' });

// Answers under the key rebuilt from the DECODED url, so the fixture cannot
// hard-code the key the implementation is supposed to build.
function taskProseResponse(prose = SERVED_PROSE) {
  return url => {
    const uid = url.slice(TASK_PROSE_PREFIX.length).split('/').map(decodeURIComponent).join('/');
    return Promise.resolve({ ok: true, json: async () => ({ [`TASK_PROSE:${uid}`]: prose }) });
  };
}

test('taskProse: the row builds a per-segment-encoded url, a TASK_PROSE key, and a plain spec', () => {
  const { api } = loadDataJs();
  const row = api.ON_DEMAND_KEYS.taskProse;

  assert.ok(row, 'ON_DEMAND_KEYS.taskProse must be declared');
  assert.equal(typeof row.url, 'function', 'the row must BUILD its url, not carry a template string');
  assert.equal(typeof row.key, 'function', 'the row must BUILD its key, not carry a template string');
  assert.equal(row.url(PROSE_UID), '/api/v2/dashboard/task/dark-factory/T-19');
  const hostile = row.url('my proj#1/T-7');
  assert.equal(hostile, '/api/v2/dashboard/task/my%20proj%231/T-7');
  assert.ok(!/[ #?]/.test(hostile), `the encoded uid leaked a delimiter: ${hostile}`);
  assert.equal(row.key(PROSE_UID), PROSE_KEY);
  assert.equal(row.spec.kind, 'plain', 'prose is not a measurement, so it is not a Datum');
});

test('taskProse: one request, one fetch, the body value stored verbatim, under its own state key only', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();
  const fetchUrls = [];
  const inner = taskProseResponse();
  const deps = { fetchImpl: url => { fetchUrls.push(url); return inner(url); }, now: () => 31 };

  const outcome = await api.requestOnDemand('taskProse', PROSE_UID, { state, deps });

  assert.equal(outcome, api.REFRESH_OUTCOMES.applied);
  assert.deepEqual(fetchUrls, ['/api/v2/dashboard/task/dark-factory/T-19'], 'exactly one fetch, to the built url');
  assert.equal(win.DF_DATA[PROSE_KEY], SERVED_PROSE, 'a plain value is stored as served, with no envelope');
  assert.deepEqual([...state.keys()], [PROSE_STATE_KEY], 'flow control must live under the request\'s OWN key only');
  assert.deepEqual(Object.keys(win.DF_DATA.__stale), [PROSE_STATE_KEY], 'staleness must live under the request\'s OWN key only');
  assert.equal(state.get(TASKS_PATH), undefined, 'a prose request took over the POLLED tasks entry');
  assert.equal(win.DF_DATA.__stale[TASKS_PATH], undefined, "a prose request wrote the polled tasks endpoint's __stale");
});

// ---------------------------------------------------------------------------
// onDemandView — what a waiting caller shows, given the value and its outcome
//
// A pure decision beside REFRESH_OUTCOMES, because what each outcome MEANS to a
// caller waiting on a value is knowledge of the outcome vocabulary: a value in
// hand is shown whatever the latest attempt did; `null` (the caller's own
// request has not settled) and skippedInFlight (someone else's request is still
// coming) are both worth waiting for; anything else means nothing is coming.
// ---------------------------------------------------------------------------

test('onDemandView: the view vocabulary is a closed, frozen set', () => {
  const { api } = loadDataJs();

  assert.deepEqual(api.ON_DEMAND_VIEWS, { ready: 'ready', loading: 'loading', unavailable: 'unavailable' });
  assert.ok(Object.isFrozen(api.ON_DEMAND_VIEWS));
});

test('onDemandView: a value in hand is ready whatever the latest attempt did, a failure included', () => {
  const { api } = loadDataJs();
  const O = api.REFRESH_OUTCOMES;

  for (const outcome of [null, O.applied, O.failed, O.skippedInFlight, O.skippedBackoff]) {
    assert.equal(
      api.onDemandView(SERVED_PROSE, outcome), api.ON_DEMAND_VIEWS.ready,
      `a fetched value must survive outcome ${outcome}`,
    );
  }
  assert.equal(
    api.onDemandView({ description: '', details: '' }, O.failed), api.ON_DEMAND_VIEWS.ready,
    'empty prose is still an answer, not an absence',
  );
});

test('onDemandView: with no value, an unsettled or in-flight request is loading', () => {
  const { api } = loadDataJs();

  for (const absent of [undefined, null]) {
    assert.equal(api.onDemandView(absent, null), api.ON_DEMAND_VIEWS.loading, 'own request not settled yet');
    assert.equal(
      api.onDemandView(absent, api.REFRESH_OUTCOMES.skippedInFlight), api.ON_DEMAND_VIEWS.loading,
      "another caller's request is still coming",
    );
  }
});

test('onDemandView: with no value, a settled request that brought none is unavailable', () => {
  const { api } = loadDataJs();
  const O = api.REFRESH_OUTCOMES;

  for (const absent of [undefined, null]) {
    for (const outcome of [O.failed, O.skippedBackoff, O.applied]) {
      assert.equal(
        api.onDemandView(absent, outcome), api.ON_DEMAND_VIEWS.unavailable,
        `nothing is coming after outcome ${outcome}`,
      );
    }
  }
});

// ---------------------------------------------------------------------------
// On-demand joining and retention
//
// JOINING. A caller that asks for a param whose request is still in flight
// is joined to that request and gets its real outcome. It is never handed
// skippedInFlight, which would tell it something is coming and then never
// say how it ended. The Task Detail pane hits this whenever a user re-selects
// a task whose first request belongs to an effect that was already torn down.
//
// RETENTION. A row that declares `retain` keeps the value and bookkeeping of
// only that many of its most recently requested params. taskProse needs it:
// its params are every task a user clicks in a long-lived tab, and each one
// would otherwise leave its prose in DF_DATA for good.
// ---------------------------------------------------------------------------

function heldProse() {
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const fetchUrls = [];
  const fetchImpl = url => { fetchUrls.push(url); return gate.then(respond => respond(url)); };
  return { fetchImpl, fetchUrls, release };
}

function proseStateKey(uid) {
  return `/api/v2/dashboard/task/${uid}#taskProse:${uid}`;
}

test('on-demand joining: a re-request for a param in flight joins it and learns that it FAILED', async () => {
  const { api } = loadDataJs();
  const state = api.createPollState();
  const held = heldProse();
  const deps = { fetchImpl: held.fetchImpl, now: () => 5 };

  const first = api.requestOnDemand('taskProse', PROSE_UID, { state, deps });
  await drain();
  const second = api.requestOnDemand('taskProse', PROSE_UID, { state, deps });
  held.release(() => ({ ok: false, status: 503 }));

  assert.equal(await first, api.REFRESH_OUTCOMES.failed);
  assert.equal(
    await second, api.REFRESH_OUTCOMES.failed,
    'the joined caller must learn how the request ended, not be told it is still coming',
  );
  assert.equal(held.fetchUrls.length, 1, 'joining must not issue a second request');
});

test('on-demand joining: a joined caller of a request that LANDS gets applied and the value', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();
  const held = heldProse();
  const deps = { fetchImpl: held.fetchImpl, now: () => 5 };

  const first = api.requestOnDemand('taskProse', PROSE_UID, { state, deps });
  await drain();
  const second = api.requestOnDemand('taskProse', PROSE_UID, { state, deps });
  held.release(url => taskProseResponse()(url));

  assert.deepEqual([await first, await second], [api.REFRESH_OUTCOMES.applied, api.REFRESH_OUTCOMES.applied]);
  assert.equal(win.DF_DATA[PROSE_KEY], SERVED_PROSE);
  assert.equal(held.fetchUrls.length, 1);
});

test('on-demand joining: once a request settles, the next one fetches afresh rather than replaying it', async () => {
  const { api } = loadDataJs();
  const state = api.createPollState();
  const fetchUrls = [];
  const inner = taskProseResponse();
  const deps = { fetchImpl: url => { fetchUrls.push(url); return inner(url); }, now: () => 5 };

  await api.requestOnDemand('taskProse', PROSE_UID, { state, deps });
  await api.requestOnDemand('taskProse', PROSE_UID, { state, deps });

  assert.equal(fetchUrls.length, 2);
});

test('on-demand retention: taskProse declares a positive whole-number retain', () => {
  const { api } = loadDataJs();
  const { retain } = api.ON_DEMAND_KEYS.taskProse;

  assert.ok(Number.isInteger(retain) && retain >= 1, `taskProse.retain must be a positive integer, got ${retain}`);
});

test('on-demand retention: one param past the cap forgets the least recently requested one entirely', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();
  const { retain } = api.ON_DEMAND_KEYS.taskProse;
  const deps = { fetchImpl: taskProseResponse(), now: () => 9 };
  const uids = Array.from({ length: retain + 1 }, (_, i) => `dark-factory/T-${i + 1}`);

  for (const uid of uids) await api.requestOnDemand('taskProse', uid, { state, deps });

  const [oldest, ...kept] = uids;
  assert.equal(win.DF_DATA[`TASK_PROSE:${oldest}`], undefined, 'the oldest prose is still held');
  assert.equal(win.DF_DATA.__loaded[`TASK_PROSE:${oldest}`], undefined, 'the oldest __loaded marker is still held');
  assert.equal(state.get(proseStateKey(oldest)), undefined, 'the oldest flow-control entry is still held');
  assert.equal(win.DF_DATA.__stale[proseStateKey(oldest)], undefined, 'the oldest __stale entry is still held');
  assert.equal(win.DF_DATA.__receipt[proseStateKey(oldest)], undefined, 'the oldest __receipt entry is still held');
  for (const uid of kept) {
    assert.equal(win.DF_DATA[`TASK_PROSE:${uid}`], SERVED_PROSE, `${uid} was forgotten inside the cap`);
  }
  assert.equal(state.size, retain);
  assert.equal(Object.keys(win.DF_DATA.__stale).length, retain);
  assert.equal(Object.keys(win.DF_DATA.__receipt).length, retain);
});

test('on-demand retention: re-requesting a param makes it the most recent, so the next one goes instead', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();
  const { retain } = api.ON_DEMAND_KEYS.taskProse;
  const deps = { fetchImpl: taskProseResponse(), now: () => 9 };
  const uids = Array.from({ length: retain }, (_, i) => `dark-factory/T-${i + 1}`);

  for (const uid of uids) await api.requestOnDemand('taskProse', uid, { state, deps });
  await api.requestOnDemand('taskProse', uids[0], { state, deps });
  await api.requestOnDemand('taskProse', 'dark-factory/T-999', { state, deps });

  assert.equal(win.DF_DATA[`TASK_PROSE:${uids[0]}`], SERVED_PROSE, 're-requested, so it is the most recent');
  assert.equal(win.DF_DATA[`TASK_PROSE:${uids[1]}`], undefined, 'now the least recent, so it goes');
});

test('on-demand retention: a param in flight is never forgotten, and goes by a later trim once settled', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();
  const { retain } = api.ON_DEMAND_KEYS.taskProse;
  const held = heldProse();
  const settled = { fetchImpl: taskProseResponse(), now: () => 9 };
  const slowUid = 'dark-factory/T-1';

  const slow = api.requestOnDemand('taskProse', slowUid, { state, deps: { fetchImpl: held.fetchImpl, now: () => 9 } });
  await drain();
  for (let i = 2; i <= retain + 2; i += 1) {
    await api.requestOnDemand('taskProse', `dark-factory/T-${i}`, { state, deps: settled });
  }
  assert.equal(state.get(proseStateKey(slowUid)).inFlight, true, 'the in-flight request lost its flow-control entry');

  held.release(url => taskProseResponse()(url));
  assert.equal(await slow, api.REFRESH_OUTCOMES.applied);
  assert.equal(win.DF_DATA[`TASK_PROSE:${slowUid}`], SERVED_PROSE);

  await api.requestOnDemand('taskProse', 'dark-factory/T-999', { state, deps: settled });
  assert.equal(win.DF_DATA[`TASK_PROSE:${slowUid}`], undefined, 'settled and past the cap, so it goes now');
  assert.equal(state.get(proseStateKey(slowUid)), undefined);
  assert.equal(state.size, retain);
});

test('on-demand retention: a row that declares no retain keeps every param', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();
  const projects = Array.from({ length: 20 }, (_, i) => `project-${i}`);

  assert.equal(api.ON_DEMAND_KEYS.terminal.retain, undefined, 'terminal params are the configured projects, a bounded set');
  for (const project of projects) {
    await api.requestOnDemand('terminal', project, { state, deps: { fetchImpl: terminalResponse(), now: () => 9 } });
  }

  for (const project of projects) {
    assert.ok(win.DF_DATA[`TASKS_TERMINAL:${project}`], `${project} was forgotten`);
  }
});
