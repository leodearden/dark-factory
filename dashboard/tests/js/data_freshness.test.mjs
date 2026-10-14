// The per-endpoint record refreshOne publishes beside the values it applies.
// DF_DATA.__stale is attempt history (task 4884, #4791); DF_DATA.__receipt is
// the provenance of the values now on screen, advanced on success only. The
// consumer's decision is dashboard/tests/js/endpoint_staleness.test.mjs.
// Run via `node --test`; dashboard/tests/test_graph_layout_js.py globs it.
import { test } from 'node:test';
import assert from 'node:assert/strict';

// The UI's own reader of the map data.js publishes. Imported so the recovery
// test below can assert what an OPERATOR sees, not merely what the map holds
// — a static import is safe here (unlike data.js, this module touches no
// browser global at load; its window assignment is typeof-guarded).
import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

import { loadDataJs, quietly } from './_data_loader.mjs';
import {
  COSTS_PATH,
  CURATOR_PATH,
  STALE_FAILURE_THRESHOLD,
  okResponse,
} from './_data_fixtures.mjs';

const { staleNoticesForTab } = staleness;

const { pollKey } = loadDataJs().api;

// ---------------------------------------------------------------------------
// Per-endpoint staleness publication (task 4884, #4791)
//
// The 2026-08-27 incident ran 19.8h with the dashboard serving a fully
// rendered UI whose numbers had stopped advancing. Keeping the prior values
// on failure is deliberate (see data_apply.test.mjs's keep-prior tests) — the
// defect is that nothing RECORDED that they were old, so no consumer could say
// so.
// These tests pin the producer half: refreshOne publishes, per endpoint PATH,
// how many consecutive attempts have failed and when the last success landed.
// endpoint_staleness.js holds the decision of what to do with that;
// dashboard/tests/js/endpoint_staleness.test.mjs covers it.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// __receipt — published on SUCCESS ONLY
// ---------------------------------------------------------------------------

test('receipts: a successful refresh records servedAt from the body and receivedAt from the clock', async () => {
  const { api, window: win } = loadDataJs();
  const deps = { fetchImpl: okResponse({ served_at: '2026-09-20T12:00:00+00:00' }), now: () => 555 };

  await api.refreshOne(CURATOR_PATH, {}, api.createPollState(), deps);

  assert.deepEqual(win.DF_DATA.__receipt[CURATOR_PATH], {
    servedAt: '2026-09-20T12:00:00+00:00',
    receivedAt: 555,
    window: null,
  });
});

test('receipts: a windowed body\'s WINDOW echo is recorded verbatim beside servedAt', async () => {
  // Transported, not validated: window_chip.js::windowEcho validates at read.
  const { api, window: win } = loadDataJs();
  const echo = { requested: '90d', served: '30d', days: 30 };
  const deps = { fetchImpl: okResponse({ served_at: 'S', WINDOW: echo }), now: () => 555 };

  await api.refreshOne(COSTS_PATH, {}, api.createPollState(), deps);

  assert.deepEqual(win.DF_DATA.__receipt[COSTS_PATH], {
    servedAt: 'S',
    receivedAt: 555,
    window: { requested: '90d', served: '30d', days: 30 },
  });
});

test('receipts: a body with no served_at records null, never undefined', async () => {
  // Today's wire for every endpoint. `null` is a stated absence that plainDatum
  // can branch on; `undefined` would read as a malformed receipt.
  const { api, window: win } = loadDataJs();
  const deps = { fetchImpl: okResponse({ CURATOR_STATE: {} }), now: () => 42 };

  await api.refreshOne(CURATOR_PATH, {}, api.createPollState(), deps);

  assert.deepEqual(win.DF_DATA.__receipt[CURATOR_PATH], { servedAt: null, receivedAt: 42, window: null });
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

    await quietly(() => api.refreshOne(CURATOR_PATH, {}, state, { fetchImpl, now: () => 900 }));

    assert.deepEqual(win.DF_DATA.__receipt[CURATOR_PATH], afterSuccess, 'the receipt advanced on a failure');
    assert.equal(win.DF_DATA.__receipt[CURATOR_PATH].receivedAt, 100);
  }
});

test('receipts: a FAILED refresh keeps the WINDOW echo describing the values still on screen', async () => {
  // A failed refresh replaced no values, so it must not relabel them: the echo
  // shares the receipt's success-only lifetime.
  for (const fetchImpl of [
    () => Promise.reject(new Error('boom')),
    () => Promise.resolve({ ok: false, status: 503, json: async () => ({ WINDOW: { requested: '7d', served: '7d', days: 7 } }) }),
  ]) {
    const { api, window: win } = loadDataJs();
    const state = api.createPollState();
    const echo = { requested: '90d', served: '30d', days: 30 };
    await api.refreshOne(COSTS_PATH, {}, state, {
      fetchImpl: okResponse({ served_at: 'S', WINDOW: echo }),
      now: () => 100,
    });
    const before = JSON.stringify(win.DF_DATA.__receipt[COSTS_PATH]);

    await quietly(() => api.refreshOne(COSTS_PATH, {}, state, { fetchImpl, now: () => 900 }));

    assert.equal(JSON.stringify(win.DF_DATA.__receipt[COSTS_PATH]), before, 'a failed refresh relabelled the window');
    assert.deepEqual(win.DF_DATA.__receipt[COSTS_PATH].window, echo);
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

  await quietly(() => api.refreshOne(CURATOR_PATH, {}, state, {
    fetchImpl: () => Promise.reject(new Error('boom')),
    now: () => 900,
  }));

  assert.equal(win.DF_DATA.__stale[CURATOR_PATH].failures, 1);
  assert.equal(win.DF_DATA.__receipt[CURATOR_PATH].receivedAt, 100);
});
