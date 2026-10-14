// data.js's on-demand seam: the ON_DEMAND_KEYS rows (terminal, taskProse);
// requestOnDemand with its joining and retention; REFRESH_OUTCOMES; and what a
// requesting caller reads back (onDemandView, onDemandDatum). Run via
// `node --test`; dashboard/tests/test_graph_layout_js.py globs it.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { loadDataJs, drain, quietly } from './_data_loader.mjs';
import {
  CURATOR_PATH,
  PLAIN_SPEC,
  TASKS_PATH,
  TERMINAL_PROJECT,
  okResponse,
} from './_data_fixtures.mjs';

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
  assert.deepEqual(win.DF_DATA.__receipt[TERMINAL_STATE_KEY], { servedAt: '2026-09-20T09:00:01+00:00', receivedAt: 77, window: null });
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

  const polledSnapshot = { [TERMINAL_PROJECT]: {} };
  await api.refreshOne(TASKS_PATH, { TASKS_SNAPSHOT: PLAIN_SPEC }, state, {
    fetchImpl: url => { polledUrls.push(url); return Promise.resolve({ ok: true, json: async () => ({ TASKS_SNAPSHOT: polledSnapshot }) }); },
    now: () => 101,
  });

  assert.deepEqual(polledUrls, [TASKS_PATH], 'the polled tasks fetch was skipped while the terminal request ran');
  assert.deepEqual(win.DF_DATA.TASKS_SNAPSHOT, polledSnapshot);

  releaseTerminal();
  await pending;
  assert.equal(win.DF_DATA[TERMINAL_KEY]._received_at, 100, 'the terminal Datum still landed afterwards');
});

test('on-demand: a failed request never escalates the polled tasks backoff', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();

  await quietly(() => api.requestOnDemand('terminal', TERMINAL_PROJECT, {
    state,
    deps: { fetchImpl: () => Promise.reject(new Error('boom')), now: () => 500 },
  }));

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

  for (const fetchImpl of [
    () => Promise.reject(new Error('boom')),
    () => Promise.resolve({ ok: false, status: 503, json: async () => ({}) }),
  ]) {
    await quietly(() => api.requestOnDemand('terminal', TERMINAL_PROJECT, {
      state,
      deps: { fetchImpl, now: () => 900, ignoreBackoff: true },
    }));
    assert.equal(win.DF_DATA[TERMINAL_KEY], good, 'a failed retry replaced the last good Datum');
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
    const outcome = await quietly(() =>
      api.refreshOne(CURATOR_PATH, {}, api.createPollState(), { fetchImpl, now: () => 1 }),
    );
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

  const failed = await quietly(() => api.requestOnDemand('terminal', TERMINAL_PROJECT, {
    state: api.createPollState(),
    deps: { fetchImpl: () => Promise.reject(new Error('boom')), now: () => 1 },
  }));
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
// The snapshot rows no longer carry either field (task 5815); the pane asks
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
// onDemandDatum — the Datum a caller that REQUESTED an on-demand row shows
//
// datumFor answers 'not yet fetched' for every absent key, which is a promise
// only a requesting caller can keep. A caller holding its own request's outcome
// knows more: the request is on its way, or it did not succeed. onDemandView
// already decides which; this turns its answer into the Datum a surface renders.
// ---------------------------------------------------------------------------

test('onDemandDatum: nothing stored and a request on its way is an unknown saying so', () => {
  const { api, window: win } = loadDataJs();
  const reasons = [null, api.REFRESH_OUTCOMES.skippedInFlight].map(outcome => {
    const d = api.onDemandDatum('terminal', TERMINAL_PROJECT, outcome);
    assert.equal(win.DF_DATUM.isDatum(d), true, `outcome ${outcome}`);
    assert.equal(d.state, 'unknown', `outcome ${outcome}`);
    assert.equal(d.value, null, `outcome ${outcome}`);
    assert.ok(d.reason, `outcome ${outcome}: the hole must say why`);
    assert.notEqual(d.reason, 'not yet fetched', "datumFor's reason promises nothing about this request");
    return d.reason;
  });
  assert.equal(reasons[0], reasons[1], 'both mean the same thing to the caller: wait');
});

test('onDemandDatum: nothing stored and a request that did not succeed says it failed', () => {
  const { api } = loadDataJs();
  const waiting = api.onDemandDatum('terminal', TERMINAL_PROJECT, null).reason;
  for (const outcome of [api.REFRESH_OUTCOMES.failed, api.REFRESH_OUTCOMES.skippedBackoff]) {
    const d = api.onDemandDatum('terminal', TERMINAL_PROJECT, outcome);
    assert.equal(d.state, 'unknown', `outcome ${outcome}`);
    assert.equal(d.value, null, `outcome ${outcome}`);
    assert.ok(d.reason, `outcome ${outcome}`);
    assert.notEqual(d.reason, waiting, `outcome ${outcome} must not read as still on its way`);
    assert.notEqual(d.reason, 'not yet fetched', `outcome ${outcome}`);
  }
});

test('onDemandDatum: a landed Datum is returned as stored, whatever outcome is passed', async () => {
  const { api, window: win } = loadDataJs();
  const O = api.REFRESH_OUTCOMES;
  await api.requestOnDemand('terminal', TERMINAL_PROJECT, {
    state: api.createPollState(),
    deps: { fetchImpl: terminalResponse(), now: () => 55 },
  });
  const stored = win.DF_DATA[api.ON_DEMAND_KEYS.terminal.key(TERMINAL_PROJECT)];
  assert.ok(stored, 'the request should have landed a Datum');

  for (const outcome of [null, O.applied, O.failed, O.skippedInFlight, O.skippedBackoff]) {
    const d = api.onDemandDatum('terminal', TERMINAL_PROJECT, outcome);
    assert.deepEqual(d, stored, `outcome ${outcome}`);
    assert.equal(d._received_at, 55, `outcome ${outcome}: the receipt stamp must survive`);
  }
});

test('onDemandDatum: a failed refresh after a good one still shows the earlier Datum', async () => {
  const { api, window: win } = loadDataJs();
  const state = api.createPollState();
  await api.requestOnDemand('terminal', TERMINAL_PROJECT, {
    state,
    deps: { fetchImpl: terminalResponse(), now: () => 300 },
  });
  const good = win.DF_DATA[TERMINAL_KEY];

  const outcome = await quietly(() => api.requestOnDemand('terminal', TERMINAL_PROJECT, {
    state,
    deps: { fetchImpl: () => Promise.reject(new Error('boom')), now: () => 900, ignoreBackoff: true },
  }));

  assert.equal(outcome, api.REFRESH_OUTCOMES.failed);
  assert.deepEqual(api.onDemandDatum('terminal', TERMINAL_PROJECT, outcome), good);
});

test('onDemandDatum: an undeclared name is refused loudly, as requestOnDemand refuses it', async () => {
  const { api } = loadDataJs();
  assert.throws(() => api.onDemandDatum('termnial', TERMINAL_PROJECT, null), /termnial/);
  let requestMessage;
  await api.requestOnDemand('termnial', TERMINAL_PROJECT, { state: api.createPollState() }).catch(e => {
    requestMessage = e.message;
  });
  assert.throws(() => api.onDemandDatum('termnial', TERMINAL_PROJECT, null), { message: requestMessage });
});

test('onDemandDatum: a PLAIN row is refused — it has no Datum to answer with', () => {
  const { api } = loadDataJs();
  assert.throws(() => api.onDemandDatum('taskProse', PROSE_UID, null), /taskProse/);
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
