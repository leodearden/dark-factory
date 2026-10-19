// Which abort deadline each data.js::refreshOne attempt arms: DEFAULT_TIMEOUT_MS
// normally, STALE_TIMEOUT_MS once the endpoint is past the staleness
// threshold, and the periodic full-deadline probe that still lets a
// slow-but-working endpoint recover from there. Run via `node --test`;
// dashboard/tests/test_graph_layout_js.py globs it.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// The UI's own reader of DF_DATA.__stale, so the recovery tests assert what an
// operator sees. A static import is safe: unlike data.js, this module touches
// no browser global at load.
import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

import { loadDataJs, drain, quietly } from './_data_loader.mjs';
import { CURATOR_PATH, STALE_FAILURE_THRESHOLD, TASKS_PATH } from './_data_fixtures.mjs';

const { staleNoticesForTab } = staleness;

// ---------------------------------------------------------------------------
// Per-attempt deadline (task 4884, #4791)
//
// Once an endpoint has failed STALE_FAILURE_THRESHOLD consecutive attempts,
// refreshOne arms the shorter STALE_TIMEOUT_MS deadline instead of the full
// one. The __stale publication half of that task is data_freshness.test.mjs.
// ---------------------------------------------------------------------------

// The reduced per-attempt deadline, stated as a literal rather than read back
// out of the module under test.
const STALE_TIMEOUT_MS = 5000;

const DATA_JS_SOURCE = fs.readFileSync(
  path.resolve(
    path.dirname(fileURLToPath(import.meta.url)),
    '../../src/dashboard/static/redux/data.js',
  ),
  'utf8',
);

test('staleness: past the threshold the per-attempt deadline drops to STALE_TIMEOUT_MS', async () => {
  // THE SECOND-ORDER EFFECT. Three wedged endpoints each holding a socket for
  // the full 30s deadline consumed about half the page's ~6-concurrent-
  // connections-per-origin budget continuously, which is why the HEALTHY tabs
  // also felt sluggish during the incident. Once an endpoint has demonstrated
  // it is failing, most attempts gain nothing by waiting 30s; the periodic
  // full-deadline probe below is the exception.
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
// Periodic full-deadline probe (task 5831)
//
// A reduced-deadline abort shows only that the endpoint did not answer within
// STALE_TIMEOUT_MS. If every attempt past the threshold armed it, an endpoint
// now answering in 5-30s could never succeed again, so it stayed stale until
// it got faster than 5s. These tests run the deadlines for real on a virtual
// clock; the no-op timers of data_poll.test.mjs::makeVirtualServer never fire.
// ---------------------------------------------------------------------------

const DEFAULT_TIMEOUT_MS = 30000;
const EXPECTED_STALE_PROBE_EVERY = 4;

// The /tasks service times task 5823's ticket measured under degraded conditions.
const SPIKE_SERVICE_MS = 33000;
const DEGRADED_SERVICE_MS = 16000;

// One endpoint on a virtual clock whose timers fire. `answer` is mutable: set
// serviceMs to Infinity for an endpoint that never answers, and ok to false for
// one that answers with an error status. An aborted request's answer is
// cancelled, as a real fetch cancels its request, so it cannot land during a
// later attempt.
function makeVirtualEndpoint() {
  const answer = { serviceMs: Infinity, ok: true };
  const armed = [];
  const timers = new Map();
  let clock = 0;
  let nextTimerId = 1;

  const schedule = (fn, ms) => {
    const id = nextTimerId;
    nextTimerId += 1;
    timers.set(id, { fn, at: clock + ms });
    return id;
  };
  const earliestTimer = () => {
    let earliest = null;
    for (const [id, timer] of timers) {
      if (earliest === null || timer.at < earliest.at) earliest = { id, ...timer };
    }
    return earliest;
  };

  const deps = {
    now: () => clock,
    random: () => 0,
    sleep: () => Promise.resolve(),
    fetchImpl: (url, init) => new Promise(resolve => {
      if (answer.serviceMs === Infinity) return;
      const { ok } = answer;
      const id = schedule(() => resolve({ ok, json: async () => ({}) }), answer.serviceMs);
      init.signal.addEventListener('abort', () => timers.delete(id));
    }),
    setTimeoutImpl: (fn, ms) => {
      armed.push(ms);
      return schedule(fn, ms);
    },
    clearTimeoutImpl: id => {
      timers.delete(id);
    },
  };

  // One refreshOne attempt, run to its outcome. It first waits out any backoff
  // the way the poll loop's wall clock does, then fires timers in time order
  // until the attempt settles. The clock never moves backwards.
  async function attempt(api, state, path) {
    clock = Math.max(clock, state.get(path)?.nextAllowedAt ?? 0);
    let outcome = null;
    api.refreshOne(path, {}, state, deps).then(result => { outcome = result; });
    await drain();
    while (outcome === null) {
      const timer = earliestTimer();
      assert.ok(timer, `refreshOne(${path}) is still pending with no timer left to fire`);
      timers.delete(timer.id);
      clock = Math.max(clock, timer.at);
      timer.fn();
      await drain();
    }
    return outcome;
  }

  return { answer, armed, attempt, now: () => clock };
}

// Runs `count` attempts on TASKS_PATH and returns their outcomes. Quietly,
// because every failed attempt console.warns by design.
function attemptsQuietly(endpoint, api, state, count) {
  return quietly(async () => {
    const outcomes = [];
    for (let i = 0; i < count; i += 1) outcomes.push(await endpoint.attempt(api, state, TASKS_PATH));
    return outcomes;
  });
}

// Makes TASKS_PATH stale with `staleAnswer`, then has it answer in
// DEGRADED_SERVICE_MS, between the two deadlines. It must recover within one
// probe cycle, both in the poll state and in what the operator sees.
async function assertRecoversAfter(staleAnswer) {
  const endpoint = makeVirtualEndpoint();
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();
  const failures = () => win.DF_DATA.__stale[TASKS_PATH].failures;
  const notices = () => staleNoticesForTab({ tab: 'tasks', stale: win.DF_DATA.__stale, now: endpoint.now() });

  Object.assign(endpoint.answer, staleAnswer);
  await attemptsQuietly(endpoint, api, state, STALE_FAILURE_THRESHOLD);
  assert.equal(failures(), STALE_FAILURE_THRESHOLD, 'precondition: the endpoint must reach the threshold');
  assert.equal(notices().length, 1, 'precondition: the tasks tab must show a stale notice at the threshold');

  Object.assign(endpoint.answer, { ok: true, serviceMs: DEGRADED_SERVICE_MS });
  const outcomes = await attemptsQuietly(endpoint, api, state, EXPECTED_STALE_PROBE_EVERY);
  assert.ok(
    outcomes.includes(api.REFRESH_OUTCOMES.applied),
    `an endpoint answering in ${DEGRADED_SERVICE_MS}ms must recover within ` +
      `${EXPECTED_STALE_PROBE_EVERY} attempts past the threshold; outcomes ` +
      `[${outcomes.join(', ')}], armed deadlines [${endpoint.armed.join(', ')}]`,
  );
  assert.equal(failures(), 0, 'a recovered endpoint must have its failure streak reset');
  assert.deepEqual(notices(), [], 'a recovered endpoint must no longer show a stale notice');
}

test('deadline: an endpoint stale after a timeout spike, now answering between the two deadlines, recovers on a full-deadline probe', () =>
  assertRecoversAfter({ serviceMs: SPIKE_SERVICE_MS }));

test('deadline: an endpoint stale after a run of 503s, now answering between the two deadlines, recovers on a full-deadline probe', () =>
  assertRecoversAfter({ ok: false, serviceMs: 0 }));

test('deadline: a wedged endpoint past the threshold arms the full deadline on one attempt in four, never two in a row', async () => {
  // Pins both sides of the trade-off: the probe lets a slow-but-working
  // endpoint recover, and its rarity keeps the connection-budget rationale
  // above data.js::STALE_TIMEOUT_MS.
  const endpoint = makeVirtualEndpoint();
  const { api } = loadDataJs();
  const state = api.createPollState();

  await attemptsQuietly(endpoint, api, state, STALE_FAILURE_THRESHOLD + 3 * EXPECTED_STALE_PROBE_EVERY);

  const probeCycle = [...Array(EXPECTED_STALE_PROBE_EVERY - 1).fill(STALE_TIMEOUT_MS), DEFAULT_TIMEOUT_MS];
  assert.deepEqual(endpoint.armed, [
    ...Array(STALE_FAILURE_THRESHOLD).fill(DEFAULT_TIMEOUT_MS),
    ...probeCycle,
    ...probeCycle,
    ...probeCycle,
  ]);
});
