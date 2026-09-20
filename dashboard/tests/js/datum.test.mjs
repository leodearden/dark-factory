// Module-contract tests for datum.js — the CLIENT half of the Datum envelope
// whose server half is dashboard/src/dashboard/data/datum.py. One value, one
// provenance record, one render decision: a consumer never has to infer from a
// zero whether a number is measured, stale, or simply unavailable.
//
// WHY THE DECISION LIVES IN A PLAIN-JS MODULE AND NOT IN THE JSX. charts.jsx
// states the constraint in its own header: the .jsx files are
// `type="text/babel"` behind CDN Babel with no node_modules, so nothing in one
// can be EXECUTED by a test. The repo's answer, written twice already
// (spark_path.js for charts.jsx's scale/path math, task_row_cells.js for the
// task row's badge/agent cells), is a dual-exported classic script holding the
// pure decision with a sibling .test.mjs over it — leaving the JSX a thin
// renderer of the returned descriptor. datum.js is built to that template, and
// is NAMED for datum.py so the wire contract's two halves are findable from
// each other.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py is the pytest
// wrapper; its `**/*.test.mjs` glob auto-discovers this file, so no wrapper
// change was needed).
//
// datum.js reads `window.DF_ENDPOINT_STALENESS` at MODULE SCOPE with no
// `|| {}` fallback, so a static ESM `import` of it cannot work here: an import
// target's body runs before the importing file's own body, so `globalThis.window`
// would still be unset when datum.js's top level ran. Every test therefore goes
// through `loadDatumJs()` below, which shims the global FIRST and only then
// loads the module via createRequire (the idiom data_poll.test.mjs::loadDataJs
// documents at length, mirrored in task_row_cells.test.mjs and
// runtime_format.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

// Safe as a static import (unlike datum.js): endpoint_staleness.js touches no
// browser global at load — its window assignment is typeof-guarded.
import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/datum.js';

const EXPECTED_FUNCTION_NAMES = ['isDatum', 'unknownDatum', 'assertDatum'];
const EXPECTED_EXPORT_NAMES = [...EXPECTED_FUNCTION_NAMES, 'DATUM_STATES'];

// Loads datum.js fresh against a shimmed browser-ish global carrying the REAL
// endpoint_staleness API, then busts the require cache so a later call
// re-executes the module body from scratch.
//
// `globalThis.window` is deliberately left installed for the rest of the file
// (the loadDataJs precedent): node's test runner gives each .test.mjs its own
// process, so nothing outside this file can observe it, and the module's own
// functions stay callable afterwards.
function loadDatumJs() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;

  const require = createRequire(import.meta.url);
  const resolved = require.resolve(MODULE_SPECIFIER);
  delete require.cache[resolved];

  return { api: require(MODULE_SPECIFIER), window: win };
}

const { api: datum } = loadDatumJs();
const { isDatum, unknownDatum, assertDatum, DATUM_STATES } = datum;

// The five-key wire envelope datum.py::Datum.to_wire() emits, verbatim: `as_of`
// is an ISO-8601 instant normalised to UTC, `reason` is null only when the
// state is 'fresh', and `freshness_bound_seconds` is the producer's declared
// bound. Every non-Datum fixture below is this object with exactly one thing
// wrong, so a failure names which part of the shape stopped being checked.
const FRESH_WIRE = Object.freeze({
  value: 42,
  as_of: '2026-09-20T12:00:00+00:00',
  state: 'fresh',
  reason: null,
  freshness_bound_seconds: 30,
});

const WIRE_KEYS = ['value', 'as_of', 'state', 'reason', 'freshness_bound_seconds'];

// The things a pre-migration call site would hand a component instead of a
// Datum — a bare number is the shape all 43 StatTile sites pass today — plus
// the container types that are structurally close enough to slip past a lazy
// `typeof x === 'object'` check.
const NON_DATUMS = [
  ['a bare number', 42],
  ['zero', 0],
  ['a string', 'fresh'],
  ['null', null],
  ['undefined', undefined],
  ['an array', [FRESH_WIRE]],
  ['a plain empty object', {}],
  ['a boolean', true],
];

test('default-imported module exposes the Datum envelope readers', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof datum[name], 'function', `datum.${name} should be a function`);
  }
});

test('module also assigns window.DF_DATUM (browser dual-export)', () => {
  const { api: required, window: win } = loadDatumJs();

  assert.ok(win.DF_DATUM, 'window.DF_DATUM was not set');
  assert.deepEqual(Object.keys(win.DF_DATUM).sort(), EXPECTED_EXPORT_NAMES.slice().sort());
  assert.deepEqual(Object.keys(required).sort(), EXPECTED_EXPORT_NAMES.slice().sort());
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof win.DF_DATUM[name], 'function', `window.DF_DATUM.${name}`);
  }
});

test('DATUM_STATES names exactly the four states datum.py declares', () => {
  // The client reads the vocabulary the server already enforces rather than a
  // second, drifting notion of one — DatumState in data/datum.py.
  assert.deepEqual(DATUM_STATES.slice().sort(), ['fresh', 'lower_bound', 'stale', 'unknown']);
});

// ---------------------------------------------------------------------------
// isDatum — the SHAPE check: five keys, one recognised state
// ---------------------------------------------------------------------------

test('isDatum: accepts the wire envelope to_wire() emits', () => {
  assert.equal(isDatum(FRESH_WIRE), true);
});

test('isDatum: accepts every state datum.py declares', () => {
  for (const state of DATUM_STATES) {
    const wire = { ...FRESH_WIRE, state, reason: state === 'fresh' ? null : 'because' };
    assert.equal(isDatum(wire), true, `state ${state} should be a Datum`);
  }
});

test('isDatum: rejects the non-envelope values a pre-migration call site passes', () => {
  for (const [label, candidate] of NON_DATUMS) {
    assert.equal(isDatum(candidate), false, `${label} should not be a Datum`);
  }
});

test('isDatum: rejects an envelope missing any one of the five keys', () => {
  // Per-key rather than one representative: a check that stopped looking at,
  // say, freshness_bound_seconds would still pass a single-fixture test, and
  // that key is exactly the one the age badge decision reads.
  for (const key of WIRE_KEYS) {
    const partial = { ...FRESH_WIRE };
    delete partial[key];
    assert.equal(isDatum(partial), false, `an envelope missing ${key} should not be a Datum`);
  }
});

test('isDatum: rejects an unrecognised state', () => {
  // A server that grew a fifth state without this client learning it must fail
  // the shape check loudly, not render its value as though it were fresh.
  assert.equal(isDatum({ ...FRESH_WIRE, state: 'degraded' }), false);
  assert.equal(isDatum({ ...FRESH_WIRE, state: 'FRESH' }), false);
  assert.equal(isDatum({ ...FRESH_WIRE, state: null }), false);
});

// ---------------------------------------------------------------------------
// unknownDatum — the one client-built "no measurement exists" envelope
// ---------------------------------------------------------------------------

test('unknownDatum: is the unknown triad datum.py validates, carrying the reason', () => {
  assert.deepEqual(unknownDatum('not yet fetched'), {
    value: null,
    as_of: null,
    state: 'unknown',
    reason: 'not yet fetched',
    freshness_bound_seconds: 0,
  });
});

test('unknownDatum: satisfies isDatum, so it can travel anywhere a Datum can', () => {
  assert.equal(isDatum(unknownDatum('endpoint never delivered')), true);
});

test('unknownDatum: returns a fresh object each call, never a shared singleton', () => {
  // Callers stamp receipts onto datums; a shared literal would let one call
  // site's stamp appear on every other site's placeholder.
  const first = unknownDatum('a');
  const second = unknownDatum('a');
  assert.notEqual(first, second);
  assert.deepEqual(first, second);
});

// ---------------------------------------------------------------------------
// assertDatum — the guard that makes a missed migration site fail LOUDLY
// ---------------------------------------------------------------------------

test('assertDatum: returns a Datum unchanged, identity preserved', () => {
  assert.equal(assertDatum(FRESH_WIRE, 'StatTile'), FRESH_WIRE);
});

test('assertDatum: throws a TypeError naming the caller for a bare number', () => {
  // PRD decision 17(c): a component receiving a non-Datum throws. The bare
  // number is the exact shape every un-migrated call site still passes, and
  // `who` is what turns "somewhere a tile is wrong" into a named component.
  assert.throws(
    () => assertDatum(42, 'StatTile'),
    err => err instanceof TypeError && err.message.includes('StatTile'),
  );
});

test('assertDatum: throws for every non-Datum, naming the caller each time', () => {
  for (const [label, candidate] of NON_DATUMS) {
    assert.throws(
      () => assertDatum(candidate, 'LocksCell'),
      err => err instanceof TypeError && err.message.includes('LocksCell'),
      `${label} should make assertDatum throw`,
    );
  }
});

test('assertDatum: throws unconditionally, not only under the test harness', () => {
  // There is no environment sniff to condition the guard on, and a component
  // that quietly renders a bare number in production while throwing in tests is
  // the silent degradation this repo's loud-over-silent norm rejects. The
  // module exposes no way to disable it — assert that by construction.
  assert.equal(Object.keys(datum).some(k => /debug|strict|enable/i.test(k)), false);
});
