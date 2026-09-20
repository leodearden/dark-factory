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

const EXPECTED_FUNCTION_NAMES = [
  'isDatum',
  'unknownDatum',
  'assertDatum',
  'withReceipt',
  'displayedAgeMs',
  'datumView',
  'plainDatum',
];
const EXPECTED_EXPORT_NAMES = [
  ...EXPECTED_FUNCTION_NAMES,
  'DATUM_STATES',
  'EM_DASH',
  'LOWER_BOUND_PREFIX',
  'PLAIN_DATUM_BOUND_SECONDS',
];

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
const { withReceipt, displayedAgeMs } = datum;
const { datumView, EM_DASH, LOWER_BOUND_PREFIX } = datum;
const { plainDatum, PLAIN_DATUM_BOUND_SECONDS } = datum;

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

// ---------------------------------------------------------------------------
// withReceipt / displayedAgeMs — the two-clock age arithmetic
//
// A displayed age spans two clocks and must never mix them. The SERVER-side
// term (served_at − as_of) is how stale the measurement already was when the
// server shaped the payload; the CLIENT-side term (now − received_at) is how
// long that payload has been sitting in this browser. Each term subtracts two
// readings of ONE clock, so the sum is exact even when the two clocks disagree
// — which they routinely do, and which is why the naive `now − as_of` this
// replaces reads a skewed browser's tile as hours stale or negative.
// ---------------------------------------------------------------------------

// A measurement taken three hours before the server served it, received by
// this browser five seconds ago.
const AS_OF = '2026-09-20T09:00:00+00:00';
const SERVED_AT = '2026-09-20T12:00:00+00:00';
const SERVER_GAP_MS = 3 * 60 * 60 * 1000;

const RECEIVED_AT = 1_800_000_000_000;
const CLIENT_GAP_MS = 5_000;
const NOW = RECEIVED_AT + CLIENT_GAP_MS;

const STALE_WIRE = Object.freeze({
  value: 7,
  as_of: AS_OF,
  state: 'stale',
  reason: 'ReadTimeout',
  freshness_bound_seconds: 30,
});

const RECEIPT = Object.freeze({ servedAt: SERVED_AT, receivedAt: RECEIVED_AT });

// Shifts an ISO instant by whole milliseconds, so a test can move ONE clock and
// leave the other alone.
function shiftIso(iso, ms) {
  return new Date(Date.parse(iso) + ms).toISOString();
}

test('withReceipt: stamps provenance onto a COPY, leaving the wire payload untouched', () => {
  // The poll loop holds the object the server sent; stamping through to it
  // would mutate state other readers are already looking at. Heuristic 8 —
  // prefer immutable data.
  const pristine = { ...STALE_WIRE };
  const stamped = withReceipt(STALE_WIRE, RECEIPT);

  assert.notEqual(stamped, STALE_WIRE, 'withReceipt must not return its input');
  assert.deepEqual(STALE_WIRE, pristine, 'the input payload was mutated');
  assert.equal(stamped._served_at, SERVED_AT);
  assert.equal(stamped._received_at, RECEIVED_AT);
});

test('withReceipt: the stamped copy is still a Datum, keys and values intact', () => {
  const stamped = withReceipt(STALE_WIRE, RECEIPT);
  assert.equal(isDatum(stamped), true);
  for (const key of WIRE_KEYS) {
    assert.deepEqual(stamped[key], STALE_WIRE[key], `${key} survived the stamp`);
  }
});

test('displayedAgeMs: is the server-side gap plus the client-side gap', () => {
  const stamped = withReceipt(STALE_WIRE, RECEIPT);
  assert.equal(displayedAgeMs(stamped, NOW), SERVER_GAP_MS + CLIENT_GAP_MS);
});

test('displayedAgeMs: is unchanged when the SERVER clock is shifted wholesale', () => {
  // A server an hour off UTC measures and serves on its own clock; the gap
  // between its two readings is what the operator needs, and it is invariant.
  const skewed = withReceipt(
    { ...STALE_WIRE, as_of: shiftIso(AS_OF, 3600_000) },
    { servedAt: shiftIso(SERVED_AT, 3600_000), receivedAt: RECEIVED_AT },
  );
  assert.equal(displayedAgeMs(skewed, NOW), SERVER_GAP_MS + CLIENT_GAP_MS);
});

test('displayedAgeMs: is unchanged when the CLIENT clock is shifted wholesale', () => {
  const shift = 7 * 24 * 60 * 60 * 1000;
  const stamped = withReceipt(STALE_WIRE, {
    servedAt: SERVED_AT,
    receivedAt: RECEIVED_AT + shift,
  });
  assert.equal(displayedAgeMs(stamped, NOW + shift), SERVER_GAP_MS + CLIENT_GAP_MS);
});

test('displayedAgeMs: GROWS in real time while no new payload arrives', () => {
  // The headline behaviour of the whole leaf: a wedged endpoint's tile keeps
  // ageing on screen instead of sitting at a reassuring constant. Asserted as
  // an exact delta under a mocked clock, not as a "greater than".
  const stamped = withReceipt(STALE_WIRE, RECEIPT);
  const before = displayedAgeMs(stamped, NOW);
  const after = displayedAgeMs(stamped, NOW + 60_000);
  assert.equal(after - before, 60_000);
});

test('displayedAgeMs: null for an unknown datum — there is no measurement to age', () => {
  // Not zero. A fabricated zero age is the `_minutes_since` mistake
  // endpoint_staleness.js::noticeText documents: it manufactures reassurance
  // during exactly the failure the indicator exists to surface.
  const stamped = withReceipt(unknownDatum('scheduler offline'), RECEIPT);
  assert.equal(displayedAgeMs(stamped, NOW), null);
});

test('displayedAgeMs: null for a datum carrying no receipt at all', () => {
  assert.equal(displayedAgeMs(STALE_WIRE, NOW), null);
  assert.equal(displayedAgeMs(withReceipt(STALE_WIRE, {}), NOW), null);
});

test('displayedAgeMs: null when `now` is not a usable clock reading', () => {
  const stamped = withReceipt(STALE_WIRE, RECEIPT);
  assert.equal(displayedAgeMs(stamped, undefined), null);
  assert.equal(displayedAgeMs(stamped, NaN), null);
});

test('displayedAgeMs: with no server served_at, degrades to the client gap — never NaN', () => {
  // Today's wire: no payload carries a top-level `served_at` until PRD leaf
  // beta lands, so `receipt.servedAt` is null for every polled endpoint. The
  // server-side term is then unknown, and an unknown term contributes nothing
  // rather than poisoning the sum — the answer is a LOWER bound on the true
  // age, which is the honest reading, and it still grows.
  const stamped = withReceipt(STALE_WIRE, { servedAt: null, receivedAt: RECEIVED_AT });
  const age = displayedAgeMs(stamped, NOW);
  assert.equal(age, CLIENT_GAP_MS);
  assert.ok(Number.isFinite(age), 'a missing served_at must not produce NaN');
});

test('displayedAgeMs: an unparseable instant yields null, not NaN', () => {
  const stamped = withReceipt({ ...STALE_WIRE, as_of: 'not an instant' }, RECEIPT);
  assert.equal(displayedAgeMs(stamped, NOW), null);
});

// ---------------------------------------------------------------------------
// datumView — THE render decision. Every shared component delegates here and
// holds no arm of its own, which is what makes "one datum, one path" a
// checkable property rather than a slogan: a tile cannot invent a second rule
// for holes, prefixes, tooltips or ages without the probe in
// tests/test_datum_components.py noticing.
// ---------------------------------------------------------------------------

// A formatter that throws if it is ever called. HBarChart's valueText
// hole-guard records why this matters: both its live call sites pass formatters
// that throw on a missing value and would take the whole tab down with them, so
// what is load-bearing is not the placeholder but that `format` is never
// INVOKED on a hole.
function explodingFormat(v) {
  throw new Error(`format must never be invoked on a hole (got ${String(v)})`);
}

// The census's most common formatter shape, so the prefix/format ordering is
// asserted against something a real call site actually passes.
const money = v => `$${v.toFixed(2)}`;

function stampedWire(overrides) {
  return withReceipt({ ...STALE_WIRE, ...overrides }, RECEIPT);
}

test('datumView: an unknown datum renders an em-dash carrying its reason', () => {
  const view = datumView(withReceipt(unknownDatum('scheduler offline'), RECEIPT), {
    now: NOW,
    format: explodingFormat,
  });

  assert.equal(view.text, EM_DASH);
  assert.equal(view.title, 'scheduler offline');
  assert.equal(view.age, null);
});

test('datumView: NEVER invokes format on a hole', () => {
  // Asserted as "did not throw" above and restated here against the raw,
  // unstamped unknown too — the pre-fetch shape every tile renders first.
  assert.doesNotThrow(() =>
    datumView(unknownDatum('not yet fetched'), { now: NOW, format: explodingFormat }),
  );
});

test('datumView: the em-dash is the exported constant, not a per-site literal', () => {
  // Exported for the same reason STRAND_TITLE is in task_row_cells.js: 43 call
  // sites hand-spelling a placeholder is 43 chances to disagree about it.
  assert.equal(EM_DASH, '—');
  assert.equal(LOWER_BOUND_PREFIX, '≥');
});

test('datumView: a stale datum renders its value, its reason, and an age badge', () => {
  const view = datumView(stampedWire({}), { now: NOW, format: money });

  assert.equal(view.text, '$7.00');
  assert.equal(view.title, 'ReadTimeout');
  assert.ok(view.age, 'a stale datum must carry an age badge');
});

test('datumView: a lower_bound datum prefixes the FORMATTED value', () => {
  // '≥$7.00', never '$≥7.00' and never '≥7' — the prefix is a statement about
  // the rendered quantity, so it wraps the formatter's output.
  const view = datumView(
    stampedWire({ state: 'lower_bound', reason: 'window truncated at 500 rows' }),
    { now: NOW, format: money },
  );

  assert.equal(view.text, `${LOWER_BOUND_PREFIX}$7.00`);
  assert.equal(view.prefix, LOWER_BOUND_PREFIX);
  assert.equal(view.title, 'window truncated at 500 rows');
});

test('datumView: no prefix on any other state', () => {
  for (const state of ['fresh', 'stale', 'unknown']) {
    const view = datumView(stampedWire({ state, reason: state === 'fresh' ? null : 'r' }), {
      now: NOW,
      format: String,
    });
    assert.equal(view.prefix, '', `state ${state} must carry no prefix`);
  }
});

test('datumView: a fresh datum inside its bound gets neither badge nor tooltip', () => {
  // Nothing to say: the server measured it, it is within the bound its producer
  // declared, and it has not aged past that in this browser. A badge here would
  // be noise on every healthy tile.
  const view = datumView(
    withReceipt({ ...FRESH_WIRE, as_of: SERVED_AT }, RECEIPT),
    { now: NOW, format: String },
  );

  assert.equal(view.text, '42');
  assert.equal(view.title, null);
  assert.equal(view.age, null);
});

test('datumView: AGE BADGE WINS — a fresh datum aged past its bound badges anyway', () => {
  // The server's verdict was true when it was served; it is the CLIENT-side
  // gap that has since made it false. The operator reads the age rather than
  // the stale verdict, which is the whole reason the age is computed on this
  // side at all.
  const fresh = withReceipt({ ...FRESH_WIRE, as_of: SERVED_AT }, RECEIPT);
  const overBound = FRESH_WIRE.freshness_bound_seconds * 1000 + 1_000;
  const view = datumView(fresh, { now: RECEIVED_AT + overBound, format: String });

  assert.ok(view.age, 'a fresh datum aged past its bound must badge');
  assert.equal(view.text, '42', 'the value still renders — the server said fresh');
  assert.equal(view.title, null, 'the server gave no reason, so none is invented');
});

test('datumView: the badge is formatAge(displayedAgeMs(...)) exactly — one formatter', () => {
  // Pins the single source. If datumView ever grows its own age spelling, the
  // tile badge and the endpoint staleness banner start disagreeing about how
  // long the same wedge has lasted.
  const stamped = stampedWire({});
  const view = datumView(stamped, { now: NOW, format: String });

  assert.equal(view.age, staleness.formatAge(displayedAgeMs(stamped, NOW)));
  assert.equal(view.age, '3h');
});

test('datumView: no badge when the age is unknowable, rather than a fabricated one', () => {
  // An unstamped datum has no receipt, so displayedAgeMs is null. Rendering
  // formatAge(null) would put the literal 'an unknown time' on a tile; showing
  // no badge says the same thing without claiming to have measured anything.
  const view = datumView(STALE_WIRE, { now: NOW, format: String });
  assert.equal(view.age, null);
  assert.equal(view.title, 'ReadTimeout', 'the reason still renders without a receipt');
});

test('datumView: a non-Datum throws via assertDatum, naming the component', () => {
  // The pre-migration prop: a bare number where the envelope belongs.
  assert.throws(
    () => datumView(42, { now: NOW, format: String }),
    err => err instanceof TypeError && err.message.includes('datumView'),
  );
});

test('datumView: format defaults to String, and opts is optional entirely', () => {
  assert.equal(datumView(stampedWire({}), { now: NOW }).text, '7');
  assert.equal(datumView(stampedWire({})).text, '7');
});

// ---------------------------------------------------------------------------
// plainDatum — endpoint-granularity provenance for values not yet SERVED as a
// Datum (PRD decision 7).
//
// No polled payload carries a Datum today: PRD leaf beta has not landed. Rather
// than leave 43 tiles unprovenanced until it does, a plain value is wrapped
// with what IS known about it — which endpoint it came from, and when that
// endpoint last delivered. That is coarser than a server Datum (one instant per
// endpoint, not per value) and the wrapper is confined to exactly this gap.
// ---------------------------------------------------------------------------

const TASKS_PATH = '/api/v2/dashboard/tasks';

function receiptsFor(entry) {
  return { [TASKS_PATH]: entry };
}

test('plainDatum: with a server served_at, as_of is that instant and the state is fresh', () => {
  const wrapped = plainDatum(7, TASKS_PATH, receiptsFor(RECEIPT));

  assert.equal(isDatum(wrapped), true);
  assert.equal(wrapped.value, 7);
  assert.equal(wrapped.as_of, SERVED_AT);
  assert.equal(wrapped.state, 'fresh');
  assert.equal(wrapped._served_at, SERVED_AT);
  assert.equal(wrapped._received_at, RECEIVED_AT);
  assert.equal(wrapped.freshness_bound_seconds, PLAIN_DATUM_BOUND_SECONDS);
});

test('plainDatum: the measurement instant claimed is the SERVING instant, not a guess', () => {
  // A plain value carries no measurement instant of its own — that is the whole
  // difference between it and a served Datum. The strongest true statement
  // available is "the server had this value when it served the payload", so
  // as_of is served_at and the server-side gap is exactly zero. Inventing an
  // earlier as_of would fabricate staleness; inventing a later one would hide
  // it.
  const wrapped = plainDatum(7, TASKS_PATH, receiptsFor(RECEIPT));
  assert.equal(displayedAgeMs(wrapped, NOW), CLIENT_GAP_MS);
});

test('plainDatum: with only a receivedAt — today\'s wire — the age still computes', () => {
  // No payload carries a top-level served_at until beta lands, so every polled
  // endpoint's receipt has servedAt null. The fallback is this browser's own
  // arrival instant, which is a real instant rather than an absent one.
  const wrapped = plainDatum(7, TASKS_PATH, receiptsFor({ servedAt: null, receivedAt: RECEIVED_AT }));

  assert.equal(isDatum(wrapped), true);
  assert.equal(Date.parse(wrapped.as_of), RECEIVED_AT);
  const age = displayedAgeMs(wrapped, NOW);
  assert.equal(age, CLIENT_GAP_MS);
  assert.ok(Number.isFinite(age), 'a missing served_at must not produce NaN');
});

test('plainDatum: with no receipt yet, the tile shows an em-dash — not a seed zero', () => {
  // The PRE-FETCH render, which every tile does before its first payload
  // resolves. DF_DATA seeds most numeric keys to 0, and rendering that 0 is
  // exactly the lie the envelope exists to remove: an operator cannot tell a
  // measured zero from a not-yet-loaded one.
  const wrapped = plainDatum(0, TASKS_PATH, {});

  assert.equal(wrapped.state, 'unknown');
  assert.equal(wrapped.value, null);
  assert.equal(wrapped.reason, 'not yet fetched');
  assert.equal(datumView(wrapped, { now: NOW, format: String }).text, EM_DASH);
});

test('plainDatum: an absent receipts map is the same as an empty one', () => {
  assert.equal(plainDatum(0, TASKS_PATH, undefined).state, 'unknown');
  assert.equal(plainDatum(0, TASKS_PATH, null).state, 'unknown');
});

test('plainDatum: NEVER consults DF_DATA.__stale — one staleness authority, not two', () => {
  // __stale records ATTEMPT history and is endpoint_staleness.js's input; it is
  // republished on FAILURE too, by design. __receipt records the PROVENANCE of
  // the value currently in DF_DATA and must not advance on a failure, which is
  // what makes a wedged endpoint's tiles keep ageing. Reading both here would
  // make the tile a second staleness verdict fired at the same instant as the
  // banner — precisely what PRD decisions 6/B4/B5 forbid.
  const baseline = plainDatum(7, TASKS_PATH, receiptsFor(RECEIPT));

  globalThis.window.DF_DATA = {
    __stale: { [TASKS_PATH]: { failures: 9, lastSuccessAt: RECEIVED_AT } },
    __receipt: receiptsFor(RECEIPT),
  };
  try {
    const withFailures = plainDatum(7, TASKS_PATH, receiptsFor(RECEIPT));
    assert.deepEqual(withFailures, baseline, 'a 9-failure __stale entry changed the datum');
  } finally {
    delete globalThis.window.DF_DATA;
  }
});

test('plainDatum: reads window.DF_DATA.__receipt when no map is passed', () => {
  // Browser call sites pass only (value, endpointKey); the third parameter
  // exists so the node suite can drive it without a DF_DATA shim.
  globalThis.window.DF_DATA = { __receipt: receiptsFor(RECEIPT) };
  try {
    assert.deepEqual(plainDatum(7, TASKS_PATH), plainDatum(7, TASKS_PATH, receiptsFor(RECEIPT)));
  } finally {
    delete globalThis.window.DF_DATA;
  }
});

test('plainDatum: a value aged past the plain bound badges through datumView', () => {
  // Age-badge-wins applies to plain datums too: the wrapper's state is always
  // 'fresh' (a polled value is whatever the endpoint last delivered), so the
  // ONLY signal that the endpoint has stopped delivering is the badge.
  const stale = plainDatum(7, TASKS_PATH, receiptsFor({ servedAt: null, receivedAt: RECEIVED_AT }));
  const overBound = RECEIVED_AT + (PLAIN_DATUM_BOUND_SECONDS + 1) * 1000;

  assert.ok(datumView(stale, { now: overBound, format: String }).age);
  assert.equal(datumView(stale, { now: RECEIVED_AT + 1000, format: String }).age, null);
});

test('plainDatum: the bound is four poll intervals — fine, ahead of the coarse banner', () => {
  // data.js polls every POLL_INTERVAL_MS=3000 with JITTER_MAX_MS=1500, so a
  // healthy receipt is at most ~4.5s old; 12s is ~2.6x headroom and no twitch.
  // endpoint_staleness.js calls an endpoint stale at 3 consecutive failures
  // (~21s of backoff), so the tile badge is the first per-VALUE signal and the
  // banner the later per-ENDPOINT explanation — deliberately ordered
  // fine-then-coarse, one authority each.
  assert.equal(PLAIN_DATUM_BOUND_SECONDS, 12);
});
