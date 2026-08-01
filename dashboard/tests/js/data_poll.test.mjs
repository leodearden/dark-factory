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
  'pollKey',
];

// Full DF_DATA key set (data.js:41-127) — initialised so the first render
// before fetch completes cannot crash any component reading DF_DATA.*.
const EXPECTED_DF_DATA_KEYS = [
  'PROJECTS', 'AGENTS', 'ORCHESTRATORS', 'ORCHESTRATORS_SPARK',
  'ACTIVE_TASKS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS', 'DONE_COUNTS',
  'PERFORMANCE', 'MEMORY_STATUS', 'MEMORY_TIMESERIES', 'MEMORY_OPS_BREAKDOWN',
  'RECON_STATE', 'MERGE_QUEUE', 'COSTS', 'BURNDOWN', 'BURNDOWN_BY_PROJECT',
  'CURATOR_STATE', 'ESCALATIONS', 'ESCALATION_ANALYTICS', 'SCHEDULER',
  'MEMORY_EVALS',
];

// Number of rows in endpointsFor() (data.js:16-34). Several tests below assert
// that a cycle touched EVERY endpoint — "all of them" is the actual claim, and
// a literal is the only way to state it without deriving the expectation from
// the same map under test. Bump this whenever endpointsFor gains or loses a
// row; it is deliberately one constant rather than a literal repeated per
// test, because scattered copies is what went stale when the memory-evals
// endpoint was added.
const EXPECTED_ENDPOINT_COUNT = 14;

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
    await api.refreshOne(FLAKY_ENDPOINT_PATH, ['CURATOR_STATE'], state, deps);
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
  await api.refreshOne(FLAKY_ENDPOINT_PATH, ['CURATOR_STATE'], state, deps);

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
