// Behavioural tests for memory_evals_fmt.js — the plain-JS (no JSX/Babel)
// module holding the pure display/vocabulary helpers behind the memory-eval
// monitoring section rendered by tab_memory_evals.jsx. Run via `node --test`
// (see dashboard/tests/test_graph_layout_js.py for the pytest wrapper that
// surfaces this suite in CI via its `**/*.test.mjs` glob — no wrapper change
// needed for this new file).
//
// WHY THIS FILE EXISTS (task 3481) — tab_memory_evals.jsx is JSX transformed by
// CDN Babel at runtime and this repo has no node_modules, so node cannot parse
// it and React cannot be rendered in any harness here. Six JSX-FREE helpers
// nonetheless lived inside that .jsx — a verdict×parity badge matrix, an age
// formatter, a null-vs-zero placeholder, a trend-hole counter, a chart-kind
// vocabulary and an unmatched-reason branch — and the only reachable test for
// them was a Python suite that read the .jsx AS TEXT and asserted regexes over
// it. That idiom absorbed review cycles 3-6 of task 3216 and is structurally
// weak in three ways this suite fixes:
//
//   1. It never EXECUTES the code. `verdictBadge` composes a base label with a
//      parity suffix across 5 verdicts × 13 parity states = 65 combinations; a
//      regex can see that a `+ ' · ' +` exists, never that the right string
//      comes out of the right pair.
//   2. A regex that matches NOTHING passes silently. Two regexes in that file
//      were in fact found matching nothing at all (hence the `assert body` /
//      "would pass vacuously" guards littered through it).
//   3. Vocabulary assertions drifted from a hand-pinned COPY of the parity
//      table rather than the real one — tab_memory_evals.jsx records that "a
//      hand-picked three-member copy DID live in that test, and went blind to
//      six states". This suite drives its matrix off the module's EXPORTED
//      tables, so a state the producer adds is covered the moment it lands.
//
// What stays in Python is exactly what node cannot do: cross-language
// completeness against the PYTHON frozenset `memory_evals.PARITY_STATES`, the
// PRD-section-8 G6/INV-5 source guard, index.html load order, and JSX/React
// render wiring.
//
// memory_evals_fmt.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { dash } from '...'` would come back undefined. We therefore
// default-import the module and destructure instead (mirrors
// spark_path.test.mjs / runtime_format.test.mjs / prd_grouping.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import mef from '../../src/dashboard/static/redux/memory_evals_fmt.js';

const { dash, ageText } = mef;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/memory_evals_fmt.js';

// The module's complete public surface, split by kind because the parity
// TABLES are exported as data alongside the functions (so this suite can
// enumerate the real vocabulary instead of pinning a copy). Asserted
// EXHAUSTIVELY below as the union of the two lists: an accidental export would
// otherwise quietly widen the contract tab_memory_evals.jsx depends on.
const EXPECTED_FUNCTION_NAMES = ['ageText', 'dash'];
const EXPECTED_DATA_NAMES = [];

function expectedSurface() {
  return EXPECTED_FUNCTION_NAMES.concat(EXPECTED_DATA_NAMES).slice().sort();
}

// The em-dash placeholder, spelled by codepoint so a copy-paste of a hyphen or
// an en-dash into either file fails here rather than rendering as a near-miss.
const EM_DASH = '—';

// ---------------------------------------------------------------------------
// Module surface and dual export
// ---------------------------------------------------------------------------

test('default-imported module exposes exactly the memory-eval format helpers', () => {
  assert.deepEqual(
    Object.keys(mef).sort(),
    expectedSurface(),
    'the public surface must be exactly these names — an accidental export ' +
      'widens the contract silently',
  );
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof mef[name], 'function', `mef.${name} should be a function`);
  }
});

test('module also assigns window.DF_MEMORY_EVALS_FMT (browser dual-export)', () => {
  // Shim a bare browser-like global before requiring the module fresh via
  // CommonJS require, so the module body's `if (typeof window !== 'undefined')`
  // branch executes against our shim.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    // Node's ESM loader resolves a CommonJS module (no package.json/type in
    // this repo) by delegating to the CJS loader and populating the shared
    // require.cache — so by the time this test runs, the top-level
    // `import mef from ...` above has ALREADY cached this exact file. A plain
    // require() here would return that cached module.exports without
    // re-running the module body, meaning the dual-export line would never see
    // our globalThis.window shim. Busting the cache entry forces a fresh
    // execution against the now-shimmed window.
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_MEMORY_EVALS_FMT, 'window.DF_MEMORY_EVALS_FMT was not set');

    // The fresh require() and the top-level import() produce two distinct API
    // object instances (separate module executions), so compare structurally
    // — same set of exported names — rather than asserting reference/deep
    // equality against the ESM-imported `mef`.
    assert.deepEqual(Object.keys(globalThis.window.DF_MEMORY_EVALS_FMT).sort(), expectedSurface());
    assert.deepEqual(Object.keys(required).sort(), expectedSurface());

    // tab_memory_evals.jsx destructures this global at module top level, so
    // every name it reaches for must be present on the BROWSER export path
    // specifically — not merely on the CommonJS one.
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(
        typeof globalThis.window.DF_MEMORY_EVALS_FMT[name],
        'function',
        `window.DF_MEMORY_EVALS_FMT.${name} should be a function`,
      );
      assert.equal(typeof required[name], 'function', `required.${name} should be a function`);
    }
  } finally {
    delete globalThis.window;
  }
});

// ---------------------------------------------------------------------------
// dash — the null-vs-zero placeholder.
//
// The load-bearing property is NOT "missing renders as a dash"; it is that a
// measured 0 does NOT. `|| 0` (or a `!v` guard) would make an absent scalar
// indistinguishable from a metric that genuinely measured zero, which for a
// proportion metric reads as total failure. That distinction is asserted
// STRICTLY here: dash(0) must be the NUMBER 0, not '0' and not the em-dash.
// ---------------------------------------------------------------------------

test('dash: null renders as the em-dash', () => {
  assert.equal(dash(null), EM_DASH);
});

test('dash: undefined renders as the em-dash', () => {
  assert.equal(dash(undefined), EM_DASH);
});

test('dash: an honest zero is preserved as the number 0, never dashed', () => {
  const out = dash(0);
  assert.equal(out, 0);
  assert.equal(typeof out, 'number', 'a measured zero must stay a number, not become a string');
  assert.notEqual(out, EM_DASH);
});

test('dash: other falsy-but-measured values pass through unchanged', () => {
  // `false` and '' are as distinguishable from "absent" as 0 is; a truthiness
  // guard would collapse all three into the placeholder.
  assert.equal(dash(false), false);
  assert.equal(dash(''), '');
});

test('dash: a negative value passes through unchanged', () => {
  assert.equal(dash(-1), -1);
});

test('dash: a positive value passes through unchanged', () => {
  assert.equal(dash(0.95), 0.95);
});

// ---------------------------------------------------------------------------
// ageText — compact age from `latest_run_age_seconds`. Display only: the
// staleness THRESHOLD lives server-side and is deliberately absent from the
// payload, so nothing here can re-derive `stale`. The `h < 1` / `h < 48`
// comparisons below are DISPLAY unit boundaries, not judgments about a metric
// value against a limit (see the G6/INV-5 guard in
// dashboard/tests/test_tab_memory_evals.py, which matches on threshold-FIELD
// member access rather than banning `<` outright, precisely so these are legal).
//
// The boundaries are read off the source and asserted at the exact hinge
// values, where an off-by-one between `<` and `<=` is visible.
// ---------------------------------------------------------------------------

test('ageText: an absent age renders as the em-dash, never as "0m ago"', () => {
  assert.equal(ageText(null), EM_DASH);
  assert.equal(ageText(undefined), EM_DASH);
});

test('ageText: a zero age is a measured zero, rendered in minutes', () => {
  assert.equal(ageText(0), '0m ago');
});

test('ageText: sub-hour ages round to the nearest minute', () => {
  assert.equal(ageText(59), '1m ago'); // Math.round(59/60) === 1
  assert.equal(ageText(60), '1m ago');
  assert.equal(ageText(90), '2m ago'); // Math.round(1.5) === 2
  assert.equal(ageText(3599), '60m ago');
});

test('ageText: exactly one hour crosses into the hours unit (h < 1 is false at h === 1)', () => {
  assert.equal(ageText(3600), '1h ago');
});

test('ageText: the last hour before the day boundary still reads in hours', () => {
  assert.equal(ageText(47 * 3600), '47h ago');
});

test('ageText: exactly 48h crosses into the days unit (h < 48 is false at h === 48)', () => {
  assert.equal(ageText(172800), '2d ago');
});

test('ageText: beyond 48h reads in days, rounded', () => {
  assert.equal(ageText(49 * 3600), '2d ago'); // Math.round(49/24) === 2
  assert.equal(ageText(72 * 3600), '3d ago');
});
