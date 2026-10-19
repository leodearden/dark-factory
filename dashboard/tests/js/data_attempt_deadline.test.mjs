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

import { loadDataJs } from './_data_loader.mjs';
import { CURATOR_PATH, STALE_FAILURE_THRESHOLD } from './_data_fixtures.mjs';

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
