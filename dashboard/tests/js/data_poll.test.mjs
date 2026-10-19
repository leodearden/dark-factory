// data.js's poll loop: the module contract and auto-start, startPolling, the
// per-endpoint in-flight guard, error backoff, jitter, the per-cycle
// df-data-refresh dispatch, and slowness pacing. The rest of data.js's surface
// has its own data_*.test.mjs suite per feature. Run via `node --test`;
// dashboard/tests/test_graph_layout_js.py globs it.
//
// Every test loads data.js through _data_loader.mjs::loadDataJs, which records
// why a static import is not an option and how auto-start stays inert here.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { loadDataJs, drain, quietly } from './_data_loader.mjs';
import {
  COSTS_PATH,
  CURATOR_PATH,
  EXPECTED_ENDPOINT_COUNT,
  TASKS_PATH,
  TERMINAL_PROJECT,
} from './_data_fixtures.mjs';

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
  'onDemandDatum',
  'pollSetFor',
  'scopePollingToTab',
];

// Full DF_DATA key set (data.js:41-127) — initialised so the first render
// before fetch completes cannot crash any component reading DF_DATA.*.
const EXPECTED_DF_DATA_KEYS = [
  'PROJECTS', 'AGENTS', 'ORCHESTRATORS', 'ORCHESTRATORS_SPARK',
  'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS',
  'TASKS_DEGRADED_PROJECTS', 'TASKS_PROJECT_COUNT', 'TASKS_SNAPSHOT',
  'INVARIANTS',
  'PERFORMANCE', 'PERFORMANCE_LISTING', 'MEMORY_STATUS', 'MEMORY_OPS',
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

// Reused below so the test fixtures assert against data.js's own key
// derivation rather than a hand-copied duplicate of the `url.split('?')[0]`
// rule (see pollKey's doc comment in data.js).
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
// Preserved-behaviour contract — pins the per-cycle df-data-refresh dispatch
// (including a cycle where every endpoint got skipped) and the ?window= URL
// shape with its path-keyed flow-control state. A later refactor cannot
// quietly drop either without one of the tests below going red. The
// keep-prior-on-failure and applyKey reference-stability halves are in
// data_apply.test.mjs; __DF_PAUSE has its own file, data_pause.test.mjs.
// ---------------------------------------------------------------------------

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
// Slowness pacing (task 5823)
//
// A slow-but-successful endpoint is paced by its own measured service time.
// The rule and its threshold live beside data.js::slownessPacingHoldMs.
// ---------------------------------------------------------------------------

const SCHEDULER_PATH = '/api/v2/dashboard/scheduler';
const ESCALATIONS_PATH = '/api/v2/dashboard/escalations';
const SIMULATED_MINUTE_MS = 60_000;
const EXPECTED_MAX_POLL_DUTY_CYCLE = 0.5;
// The failure-backoff ceiling the schedule test above already pins.
const EXPECTED_BACKOFF_MAX_MS = 60000;

// A virtual event loop: every sleep and every response resolves at a virtual
// instant, and each endpoint answers after its configured service time; a path
// in `failingPaths` rejects its fetch after that time instead. The
// no-op timer pair keeps refreshOne's 30s deadline from arming a real timer,
// so no deadline ever fires here: a test that needs one to fire belongs in
// data_attempt_deadline.test.mjs, on its makeVirtualEndpoint.
function makeVirtualServer(serviceMsByPath, { failingPaths = [] } = {}) {
  let clock = 0;
  const timers = [];
  const calls = new Map();
  const urls = [];
  const serviceMs = { ...serviceMsByPath };
  const failing = new Set(failingPaths);

  const after = ms => new Promise(resolve => { timers.push({ at: clock + ms, fire: resolve }); });

  function fetchImpl(url) {
    const path = pollKey(url);
    calls.set(path, (calls.get(path) || 0) + 1);
    urls.push(url);
    return after(serviceMs[path] ?? 0).then(() => {
      if (failing.has(path)) throw new Error(`virtual server: ${path} is down`);
      return { ok: true, json: async () => ({}) };
    });
  }

  async function advanceTo(target) {
    for (;;) {
      timers.sort((a, b) => a.at - b.at);
      if (timers.length === 0 || timers[0].at > target) break;
      const next = timers.shift();
      clock = next.at;
      next.fire();
      await drain();
    }
    clock = target;
  }

  return {
    deps: {
      fetchImpl,
      now: () => clock,
      sleep: after,
      random: () => 0,
      setTimeoutImpl: () => 0,
      clearTimeoutImpl: () => {},
    },
    serviceMs,
    urls,
    count: path => calls.get(path) || 0,
    advanceTo,
  };
}

async function runTicks(api, server, opts, firstTickAt, tickCount) {
  for (let i = 0; i < tickCount; i += 1) {
    await server.advanceTo(firstTickAt + i * EXPECTED_POLL_INTERVAL_MS);
    api.pollTick(opts);
    await drain();
  }
  await server.advanceTo(firstTickAt + tickCount * EXPECTED_POLL_INTERVAL_MS);
}

test('pacing: a slow-but-successful endpoint is not asked again until twice its service time after the request started', async () => {
  const server = makeVirtualServer({ [TASKS_PATH]: 3900 });
  const { api, window: win } = loadDataJs({ fetchStub: server.deps.fetchImpl });
  const opts = { state: api.createPollState(), deps: server.deps, jitterMaxMs: 0 };

  api.pollTick(opts);
  await drain();
  await server.advanceTo(3900);
  assert.equal(win.DF_DATA.__stale[TASKS_PATH].failures, 0, 'a slow answer is a success, not a failure');

  await server.advanceTo(7799);
  api.pollTick(opts);
  await drain();
  assert.equal(server.count(TASKS_PATH), 1, 'asked again before twice its 3900ms service time had passed');

  await server.advanceTo(7800);
  api.pollTick(opts);
  await drain();
  assert.equal(server.count(TASKS_PATH), 2, 'still held once twice its service time had passed');
});

test('pacing: replaying the three endpoints measured 2026-09-23, each is held at or under the duty-cycle cap - fewer requests per minute than the in-flight guard alone admits', async () => {
  const measuredServiceMs = { [TASKS_PATH]: 3886, [SCHEDULER_PATH]: 2297, [ESCALATIONS_PATH]: 2120 };
  // What the in-flight guard alone admits in a minute of 3s ticks: /tasks
  // lands between ticks, so it is re-issued every 2nd tick; the other two are
  // under 3000ms, so every tick.
  const unpacedRequestsPerMinute = { [TASKS_PATH]: 10, [SCHEDULER_PATH]: 20, [ESCALATIONS_PATH]: 20 };
  const tickCount = SIMULATED_MINUTE_MS / EXPECTED_POLL_INTERVAL_MS;
  const server = makeVirtualServer(measuredServiceMs);
  const { api } = loadDataJs({ fetchStub: server.deps.fetchImpl });
  const opts = { state: api.createPollState(), deps: server.deps, jitterMaxMs: 0 };

  await runTicks(api, server, opts, 0, tickCount);

  for (const [path, serviceMs] of Object.entries(measuredServiceMs)) {
    const requests = server.count(path);
    assert.ok(
      requests < unpacedRequestsPerMinute[path],
      `${path}: ${requests} requests in a minute, not fewer than the unpaced ${unpacedRequestsPerMinute[path]}`,
    );
    const dutyCycle = requests * serviceMs / SIMULATED_MINUTE_MS;
    assert.ok(
      dutyCycle <= EXPECTED_MAX_POLL_DUTY_CYCLE,
      `${path}: busy ${dutyCycle.toFixed(3)} of the minute, over the ${EXPECTED_MAX_POLL_DUTY_CYCLE} cap`,
    );
  }
  const fastPaths = Object.keys(api.endpointsFor('24h')).map(pollKey).filter(p => !(p in measuredServiceMs));
  for (const path of fastPaths) {
    assert.equal(server.count(path), tickCount, `${path} is fast and must be fetched on every tick`);
  }
});

test('pacing: a fast endpoint is untouched - even when jitter starts its request late in the tick', async () => {
  const tickCount = SIMULATED_MINUTE_MS / EXPECTED_POLL_INTERVAL_MS;
  const server = makeVirtualServer({ [SCHEDULER_PATH]: 1400 });
  const { api } = loadDataJs({ fetchStub: server.deps.fetchImpl });
  const opts = { state: api.createPollState(), deps: { ...server.deps, random: () => 0.9 } };

  await runTicks(api, server, opts, 0, tickCount);

  const paths = Object.keys(api.endpointsFor('24h')).map(pollKey);
  assert.equal(paths.length, EXPECTED_ENDPOINT_COUNT);
  for (const path of paths) {
    assert.equal(server.count(path), tickCount, `${path} was paced, though its own service time is under the threshold`);
  }
});

test('pacing: relaxes back as the endpoint speeds up', async () => {
  const server = makeVirtualServer({ [TASKS_PATH]: 3900 });
  const { api } = loadDataJs({ fetchStub: server.deps.fetchImpl });
  const opts = { state: api.createPollState(), deps: server.deps, jitterMaxMs: 0 };

  await runTicks(api, server, opts, 0, 9);
  assert.equal(server.count(TASKS_PATH), 3, 'a 3900ms endpoint is asked once per 9s, not every 6s');

  server.serviceMs[TASKS_PATH] = 100;
  await runTicks(api, server, opts, 27_000, 11);
  assert.equal(server.count(TASKS_PATH), 3 + 11, 'a fast answer must restore the full cadence on the very next tick');
});

test('pacing: a chip change still forces a paced WINDOWED endpoint, only the windowed ones, and the forced answer paces the timer path by its own measurement', async () => {
  const server = makeVirtualServer({ [COSTS_PATH]: 4000, [CURATOR_PATH]: 4000 });
  const { api } = loadDataJs({ fetchStub: server.deps.fetchImpl });
  const opts = { state: api.createPollState(), deps: server.deps, jitterMaxMs: 0 };

  api.pollTick(opts);
  await drain();
  await server.advanceTo(5000);
  api.pollTick(opts);
  await drain();
  assert.equal(server.count(COSTS_PATH), 1, 'precondition: costs is paced');
  assert.equal(server.count(CURATOR_PATH), 1, 'precondition: curator is paced');

  const chip = api.refreshDFData('7d', opts);
  await drain();
  assert.equal(server.count(COSTS_PATH), 2, 'the chip change must bypass pacing on a windowed endpoint');
  assert.ok(server.urls.includes(`${COSTS_PATH}?window=7d`), 'the forced request must carry the new window');
  assert.equal(server.count(CURATOR_PATH), 1, 'the chip has no bearing on curator, so its pacing must hold');

  await server.advanceTo(9000);
  await chip;

  await server.advanceTo(12_999);
  api.pollTick(opts);
  await drain();
  assert.equal(server.count(COSTS_PATH), 2, 'the forced 4000ms answer, started at 5000, must pace the timer path');

  await server.advanceTo(13_000);
  api.pollTick(opts);
  await drain();
  assert.equal(server.count(COSTS_PATH), 3);
});

test('pacing: requestOnDemand is never paced - a user action is not a poll loop, even handed a poll cadence in its deps', async () => {
  const server = makeVirtualServer({ [TASKS_PATH]: 4000 });
  const { api, window: win } = loadDataJs({ fetchStub: server.deps.fetchImpl });
  const state = api.createPollState();
  const deps = { ...server.deps, pollIntervalMs: EXPECTED_POLL_INTERVAL_MS };

  const first = api.requestOnDemand('terminal', TERMINAL_PROJECT, { state, deps });
  await drain();
  await server.advanceTo(4000);
  assert.equal(await first, api.REFRESH_OUTCOMES.applied);
  assert.deepEqual(Object.values(win.DF_DATA.__receipt).map(r => r.pacingHoldMs), [0], 'an on-demand success imposed a hold');

  const second = api.requestOnDemand('terminal', TERMINAL_PROJECT, { state, deps });
  await drain();
  await server.advanceTo(8000);
  assert.equal(await second, api.REFRESH_OUTCOMES.applied, 'a user re-request of a slow listing must be fetched, not skipped');
  assert.equal(server.count(TASKS_PATH), 2);
});

test('pacing: a service time inflated by a wall-clock jump can never hold an endpoint beyond BACKOFF_MAX_MS after its request started', async () => {
  // A laptop suspended mid-request: the answer lands ten hours "later" by
  // Date.now(), still a success, so no staleness banner would ever explain
  // an unbounded hold.
  const jumpedServiceMs = 36_000_000;
  const server = makeVirtualServer({ [CURATOR_PATH]: jumpedServiceMs });
  const { api, window: win } = loadDataJs({ fetchStub: server.deps.fetchImpl });
  const opts = { state: api.createPollState(), deps: server.deps, jitterMaxMs: 0 };

  api.pollTick(opts);
  await drain();
  await server.advanceTo(jumpedServiceMs);
  assert.equal(win.DF_DATA.__stale[CURATOR_PATH].failures, 0, 'nothing would warn the operator of this hold');

  // The cap runs from the request's start, so a late answer lands already past
  // it: the very next tick asks again. A cap anchored at arrival would still
  // be holding here.
  await server.advanceTo(jumpedServiceMs + EXPECTED_POLL_INTERVAL_MS);
  api.pollTick(opts);
  await drain();
  assert.equal(server.count(CURATOR_PATH), 2, `held off past ${EXPECTED_BACKOFF_MAX_MS}ms after its request started`);
});

test('pacing: each success publishes the hold it imposed on its receipt — paced, capped and unpaced', async () => {
  const server = makeVirtualServer({ [TASKS_PATH]: 3900 });
  const { api, window: win } = loadDataJs({ fetchStub: server.deps.fetchImpl });
  const opts = { state: api.createPollState(), deps: server.deps, jitterMaxMs: 0 };

  api.pollTick(opts);
  await drain();
  await server.advanceTo(3900);
  assert.equal(win.DF_DATA.__receipt[TASKS_PATH].pacingHoldMs, 7800, 'a 3900ms answer holds twice its service time');
  assert.equal(win.DF_DATA.__receipt[CURATOR_PATH].pacingHoldMs, 0, 'a fast answer in the same cycle published a hold');

  const jumpedServiceMs = 36_000_000;
  const jumped = makeVirtualServer({ [CURATOR_PATH]: jumpedServiceMs });
  const { api: jumpedApi, window: jumpedWin } = loadDataJs({ fetchStub: jumped.deps.fetchImpl });
  jumpedApi.pollTick({ state: jumpedApi.createPollState(), deps: jumped.deps, jitterMaxMs: 0 });
  await drain();
  await jumped.advanceTo(jumpedServiceMs);
  assert.equal(
    jumpedWin.DF_DATA.__receipt[CURATOR_PATH].pacingHoldMs,
    EXPECTED_BACKOFF_MAX_MS,
    'the published hold is not the capped one the poll loop imposed',
  );
});

// Advances the virtual clock to `t`, firing a poll tick when `t` lands on one.
async function advanceAndTick(api, server, opts, t) {
  await server.advanceTo(t);
  if (t % EXPECTED_POLL_INTERVAL_MS !== 0) return;
  api.pollTick(opts);
  await drain();
}

test("pacing: a slow-but-healthy endpoint's plain tile never badges between landings, and badges once it stops answering past its paced bound", async () => {
  // The bound datum.js::plainDatum gives an unpaced receipt, restated rather
  // than imported so this pins the composed data.js -> datum.js behaviour.
  const EXPECTED_PLAIN_BOUND_MS = 12_000;
  const SAMPLE_MS = 250;
  const HEALTHY_FOR_MS = 120_000;
  // /tasks as measured 2026-09-23, then the band that badged before task 5832.
  for (const serviceMs of [3886, 6000, 8500]) {
    const server = makeVirtualServer({ [CURATOR_PATH]: serviceMs });
    const { api, window: win } = loadDataJs({ fetchStub: server.deps.fetchImpl });
    const opts = { state: api.createPollState(), deps: { ...server.deps, random: () => 0.99 } };
    const { plainDatum, datumView } = win.DF_DATUM;
    const tile = now => datumView(plainDatum(1, CURATOR_PATH, win.DF_DATA.__receipt), { now }).age;

    let t = 0;
    for (; t <= HEALTHY_FOR_MS; t += SAMPLE_MS) {
      await advanceAndTick(api, server, opts, t);
      if (!win.DF_DATA.__receipt[CURATOR_PATH]) continue;
      const age = tile(t);
      assert.equal(age, null, `${serviceMs}ms endpoint: its healthy tile badged '${age}' at t=${t}`);
    }

    // A hung request never aborts here (makeVirtualServer arms no deadline), so
    // the in-flight guard skips every later tick. The request already in flight
    // keeps its old service time, so let it land before reading the receipt.
    server.serviceMs[CURATOR_PATH] = Infinity;
    const inFlightLandedBy = t + serviceMs + EXPECTED_JITTER_MAX_MS + EXPECTED_POLL_INTERVAL_MS;
    for (; t <= inFlightLandedBy; t += SAMPLE_MS) await advanceAndTick(api, server, opts, t);

    const last = win.DF_DATA.__receipt[CURATOR_PATH];
    assert.ok(last.pacingHoldMs > 0, `${serviceMs}ms endpoint: its last receipt published no pacing hold (${last.pacingHoldMs})`);
    const overdueAt = last.receivedAt + EXPECTED_PLAIN_BOUND_MS + last.pacingHoldMs + 1000;
    for (; t < overdueAt; t += SAMPLE_MS) await advanceAndTick(api, server, opts, t);
    await server.advanceTo(overdueAt);

    assert.equal(win.DF_DATA.__receipt[CURATOR_PATH].receivedAt, last.receivedAt, `${serviceMs}ms endpoint: delivered after it stopped`);
    assert.ok(tile(overdueAt), `${serviceMs}ms endpoint: silent ${overdueAt - last.receivedAt}ms and its tile did not badge`);
  }
});

// ---------------------------------------------------------------------------
// A chip change that finds a windowed endpoint in flight (task 5833). The
// in-flight guard refuses the forced attempt, and the hold the old-window
// answer then arms must not delay the chosen window past the next tick.
// ---------------------------------------------------------------------------

async function pollAt(api, server, opts, at) {
  await server.advanceTo(at);
  api.pollTick(opts);
  await drain();
}

test('chip change in flight: the pacing hold the superseded answer arms does not delay the chosen window past the next tick, and the deferred force is spent once', async () => {
  const server = makeVirtualServer({ [COSTS_PATH]: 4000 });
  const { api } = loadDataJs({ fetchStub: server.deps.fetchImpl });
  const opts = { state: api.createPollState(), deps: server.deps, jitterMaxMs: 0 };
  const costsUrls = () => server.urls.filter(u => pollKey(u) === COSTS_PATH);

  await pollAt(api, server, opts, 0);
  await server.advanceTo(1000);
  const chip = api.refreshDFData('7d', opts);
  await drain();
  assert.equal(server.count(COSTS_PATH), 1, 'precondition: the chip found costs in flight');

  await pollAt(api, server, opts, 3000);
  await server.advanceTo(4000);
  await chip;

  await pollAt(api, server, opts, 6000);
  assert.equal(
    server.count(COSTS_PATH),
    2,
    'the tick after the superseded answer landed must ask for the chosen window, not wait out the hold that answer armed',
  );
  assert.equal(costsUrls().at(-1), `${COSTS_PATH}?window=7d`);

  await pollAt(api, server, opts, 9000);
  await pollAt(api, server, opts, 12_000);
  assert.equal(
    server.count(COSTS_PATH),
    2,
    'the deferred force is spent by the attempt it forced: the timer path honours that answer\'s own pacing again',
  );

  await pollAt(api, server, opts, 15_000);
  assert.equal(server.count(COSTS_PATH), 3);
});

test('chip change in flight: a chip change back to the window in flight defers no force - that answer is current, so its pacing hold stands', async () => {
  const server = makeVirtualServer({ [COSTS_PATH]: 4000 });
  const { api } = loadDataJs({ fetchStub: server.deps.fetchImpl });
  const opts = { state: api.createPollState(), deps: server.deps, jitterMaxMs: 0 };
  const costsUrls = () => server.urls.filter(u => pollKey(u) === COSTS_PATH);

  await pollAt(api, server, opts, 0);
  await server.advanceTo(1000);
  const away = api.refreshDFData('7d', opts);
  await drain();
  await server.advanceTo(2000);
  const back = api.refreshDFData('24h', opts);
  await drain();
  assert.equal(server.count(COSTS_PATH), 1, 'precondition: both chips found costs in flight');

  await server.advanceTo(4000);
  await Promise.all([away, back]);

  await pollAt(api, server, opts, 6000);
  assert.equal(server.count(COSTS_PATH), 1, 'the answer in flight was for the chosen window, so the pacing hold it armed must stand');

  await pollAt(api, server, opts, 9000);
  assert.deepEqual(costsUrls(), [`${COSTS_PATH}?window=24h`, `${COSTS_PATH}?window=24h`]);
});

test('chip change in flight: the failure backoff the superseded answer arms is waived once for the chosen window, without escalating', async () => {
  const server = makeVirtualServer({ [COSTS_PATH]: 1000 }, { failingPaths: [COSTS_PATH] });
  const { api, window: win } = loadDataJs({ fetchStub: server.deps.fetchImpl });
  const opts = { state: api.createPollState(), deps: server.deps, jitterMaxMs: 0 };
  const costsUrls = () => server.urls.filter(u => pollKey(u) === COSTS_PATH);
  const costsFailures = () => win.DF_DATA.__stale[COSTS_PATH].failures;

  await quietly(async () => {
    await pollAt(api, server, opts, 0);
    const chip = api.refreshDFData('7d', opts);
    await drain();
    await server.advanceTo(1000);
    await chip;
    assert.equal(costsFailures(), 1, 'the superseded answer still counts as a failure');

    await pollAt(api, server, opts, 3000);
    assert.deepEqual(costsUrls(), [`${COSTS_PATH}?window=24h`, `${COSTS_PATH}?window=7d`]);
    await server.advanceTo(4000);
    assert.equal(costsFailures(), 1, 'a deferred chip attempt that fails must not escalate the timer path\'s backoff');

    await pollAt(api, server, opts, 6000);
    await server.advanceTo(7000);
    assert.equal(costsFailures(), 2, 'the deferred force is spent: the next ordinary attempt\'s failure escalates again');
  });
});
