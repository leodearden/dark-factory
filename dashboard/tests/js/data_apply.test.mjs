// How a refresh's answer lands in window.DF_DATA: endpointsFor's key
// registry; applyKey under plain and datum specs, its refusals included;
// datumFor; and the prior value a failed refresh keeps. Run via `node --test`;
// dashboard/tests/test_graph_layout_js.py globs it.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { loadDataJs } from './_data_loader.mjs';
import {
  CURATOR_PATH,
  DATUM_SPEC,
  EXPECTED_ENDPOINT_COUNT,
  PLAIN_SPEC,
} from './_data_fixtures.mjs';

// Every DF_DATA key any endpoint row names, as one sorted list. The registry
// reshape (array of key names -> object of key name to spec) is exactly the
// kind of edit that can silently DROP a key — an object literal with a
// duplicated or mistyped name loses a row with no error anywhere — and a
// dropped key means a tab that simply stops updating. Stated as a literal
// because deriving it from endpointsFor would be deriving the expectation from
// the thing under test.
const EXPECTED_ENDPOINT_KEYS = [
  'AGENTS', 'BURNDOWN', 'BURNDOWN_BY_PROJECT', 'COSTS',
  'CURATOR_STATE', 'ESCALATIONS', 'ESCALATION_ANALYTICS', 'INVARIANTS',
  'MEMORY_EVALS', 'MEMORY_OPS', 'MEMORY_STATUS',
  'MERGE_QUEUE', 'ORCHESTRATORS', 'ORCHESTRATORS_SPARK', 'PERFORMANCE',
  'PERFORMANCE_LISTING', 'PROJECTS', 'RECON_STATE', 'SCHEDULER', 'TASKS_COUNT_UNKNOWN_PROJECTS',
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

// ---------------------------------------------------------------------------
// Keep-prior and applyKey reference stability — a failed refresh leaves the
// prior DF_DATA value in place, and applyKey keeps the array references
// shell.jsx captured at load.
// ---------------------------------------------------------------------------

test('preserved behaviour: a thrown fetch error keeps the prior DF_DATA value and still warns', async () => {
  const { api, window: win } = loadDataJs();
  win.DF_DATA.CURATOR_STATE = { marker: 'prior-throw' };

  const originalWarn = console.warn;
  const warnCalls = [];
  console.warn = (...args) => warnCalls.push(args);
  try {
    const state = api.createPollState();
    const deps = { fetchImpl: () => Promise.reject(new Error('boom')), now: () => 0 };
    await api.refreshOne(CURATOR_PATH, { CURATOR_STATE: PLAIN_SPEC }, state, deps);
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
  await api.refreshOne(CURATOR_PATH, { CURATOR_STATE: PLAIN_SPEC }, state, deps);

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

test('registry: ACTIVE_TASKS is retired from the seed and the /tasks row', () => {
  // data_poll.test.mjs::EXPECTED_DF_DATA_KEYS is a containment check, so it
  // cannot see a leftover seed. The rows travel only as TASKS_SNAPSHOT[p].rows
  // now.
  const { api, window: win } = loadDataJs();
  assert.ok(!Object.prototype.hasOwnProperty.call(win.DF_DATA, 'ACTIVE_TASKS'), 'DF_DATA still seeds ACTIVE_TASKS');
  assert.ok(
    !Object.prototype.hasOwnProperty.call(api.endpointsFor('24h')['/api/v2/dashboard/tasks'], 'ACTIVE_TASKS'),
    'the /tasks registry row still names ACTIVE_TASKS',
  );
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
