// Behavioural tests for spark_path.js — the plain-JS (no JSX/Babel) module
// holding the scale/path math behind charts.jsx's `Sparkline` and `StepSpark`
// sparkline primitives. Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper that surfaces
// this suite in CI via its `**/*.test.mjs` glob — no wrapper change needed for
// this new file).
//
// WHY THIS FILE EXISTS (task 3436) — charts.jsx is JSX transformed by CDN Babel
// at runtime and this repo has no node_modules, so node cannot parse it and
// React cannot be rendered in any harness here. Before the extraction, the
// scale/path arithmetic lived inline in the two component bodies, where the
// only reachable test was grepping charts.jsx for the absence of an expression
// — which proves nothing about whether the replacement is correct. Moving that
// arithmetic into a plain-JS sibling puts the entire defect surface somewhere
// this suite can execute it with real assertions. charts.jsx keeps only JSX
// plus a one-line delegation, pinned by dashboard/tests/test_charts_null_samples.py.
//
// THE DEFECT BEING FIXED — the pre-fix expressions were
// `Math.max(...values, 1)` / `Math.min(...values, 0)` and
// `y = height - ((v - min) / range) * height`, with no null handling at all:
//   1. a `null` hole coerces to 0 in the y expression, so a MISSING sample was
//      drawn as a real point at the value-0 baseline, joined by line segments
//      to both neighbours and indistinguishable from a measured regression;
//   2. an `undefined`/`NaN` hole poisoned BOTH extrema to NaN (and
//      `range = NaN - NaN || 1` silently fell back to 1, NaN being falsy), so
//      every y became NaN and the whole chart rendered nothing;
//   3. an all-hole series still produced a full flat line along the chart
//      floor — measurements asserted that were never taken.
//
// spark_path.js has no package.json in the repo, so it resolves as CommonJS
// (`module.exports = <object>`). Node's cjs-module-lexer cannot statically
// detect named exports assigned from a variable, so
// `import { sparkScale } from '...'` would come back undefined. We therefore
// default-import the module and destructure instead (mirrors
// runtime_format.test.mjs / prd_grouping.test.mjs / graph_layout.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import sp from '../../src/dashboard/static/redux/spark_path.js';

const { isPlottable, sparkScale } = sp;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/spark_path.js';

// The surface known at this point in the build-out. Deliberately NOT asserted
// as exhaustive here — sparkPaths/stepPaths land in later steps and the
// exhaustive-surface pin lives with them.
const KNOWN_FUNCTION_NAMES = ['isPlottable', 'sparkScale'];

// Float tolerance for any y that is not exactly representable. The x
// coordinates and the extrema below are all exact at these inputs (width=100
// with 3 or 5 points divides evenly), so exact equality is used for those.
const EPS = 1e-9;

function assertClose(actual, expected, message) {
  assert.ok(
    Math.abs(actual - expected) < EPS,
    `${message} (expected ~${expected}, got ${actual})`,
  );
}

// ---------------------------------------------------------------------------
// Module surface and dual export
// ---------------------------------------------------------------------------

test('default-imported module exposes the spark-path functions', () => {
  for (const name of KNOWN_FUNCTION_NAMES) {
    assert.equal(typeof sp[name], 'function', `sp.${name} should be a function`);
  }
});

test('module also assigns window.DF_SPARK_PATH (browser dual-export)', () => {
  // Shim a bare browser-like global before requiring the module fresh via
  // CommonJS require, so the module body's `if (typeof window !== 'undefined')`
  // branch executes against our shim.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    // Node's ESM loader resolves a CommonJS module (no package.json/type in
    // this repo) by delegating to the CJS loader and populating the shared
    // require.cache — so by the time this test runs, the top-level
    // `import sp from ...` above has ALREADY cached this exact file. A plain
    // require() here would return that cached module.exports without
    // re-running the module body, meaning the dual-export line would never see
    // our globalThis.window shim. Busting the cache entry forces a fresh
    // execution against the now-shimmed window.
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_SPARK_PATH, 'window.DF_SPARK_PATH was not set');

    // charts.jsx destructures this global at module top level, so every name
    // it reaches for must be present on the browser export path specifically —
    // not merely on the CommonJS one.
    for (const name of KNOWN_FUNCTION_NAMES) {
      assert.equal(
        typeof globalThis.window.DF_SPARK_PATH[name],
        'function',
        `window.DF_SPARK_PATH.${name} should be a function`,
      );
      assert.equal(typeof required[name], 'function', `required.${name} should be a function`);
    }
  } finally {
    delete globalThis.window;
  }
});

// ---------------------------------------------------------------------------
// isPlottable — a sample is plottable iff it is a finite number. null,
// undefined, NaN, ±Infinity and non-numbers are all HOLES.
// ---------------------------------------------------------------------------

test('isPlottable: an honest 0 is a real sample, not a hole', () => {
  // This is the whole point of the fix. Distinguishing a MEASURED zero from a
  // MISSING sample is the invariant the pre-fix code destroyed by coercing
  // null to 0; a predicate that also rejected 0 would trade the bug for its
  // mirror image. The repo states the same invariant twice elsewhere —
  // runtime_format.js's rtCell ("an honest not-yet-iterated online task
  // reports an honest 0") and tab_memory_evals.jsx's dash().
  assert.equal(isPlottable(0), true);
  assert.equal(isPlottable(-0), true);
});

test('isPlottable: finite numbers of any sign and magnitude are plottable', () => {
  for (const v of [0, -4, 0.95, 1e-9, -0.5, 1000, Number.MAX_SAFE_INTEGER]) {
    assert.equal(isPlottable(v), true, `${v} should be plottable`);
  }
});

test('isPlottable: null and undefined are holes', () => {
  assert.equal(isPlottable(null), false);
  assert.equal(isPlottable(undefined), false);
});

test('isPlottable: NaN and ±Infinity are holes', () => {
  // These produce the same class of garbage as null/undefined — NaN-poisoned
  // extrema and an unrenderable path — so they are excluded by the same gate
  // rather than left to blank a chart.
  assert.equal(isPlottable(NaN), false);
  assert.equal(isPlottable(Infinity), false);
  assert.equal(isPlottable(-Infinity), false);
});

test('isPlottable: non-numbers are holes even when numeric-looking', () => {
  // A numeric string would coerce silently through the arithmetic and plot as
  // if it were measured. Every consumer's series arrives as JSON numbers, so
  // rejecting these rejects nothing legitimate.
  for (const v of ['3', '', [], {}, true, false, () => 0]) {
    assert.equal(isPlottable(v), false, `${JSON.stringify(v) ?? String(v)} should be a hole`);
  }
});

// ---------------------------------------------------------------------------
// sparkScale — extrema, seeds, and index-preserving points
// ---------------------------------------------------------------------------

test('sparkScale: an undefined hole no longer poisons the extrema to NaN', () => {
  // THE REAL RED SIGNAL for the extrema half of the fix. Pre-fix:
  //   Math.max(0.9, undefined, 0.95, 1) === NaN
  //   Math.min(0.9, undefined, 0.95, 0) === NaN
  //   range = NaN - NaN || 1  ===  1      (NaN is falsy, so the || fires)
  // leaving min === NaN, so EVERY y came out NaN and the entire chart silently
  // rendered nothing. Note the asymmetry: a bare `null` can never move either
  // extremum, because 0 cannot lower a max already seeded with 1 nor raise a
  // min already seeded with 0. undefined/NaN are where exclusion is
  // load-bearing.
  const s = sparkScale([0.9, undefined, 0.95], 100, 28);
  assert.equal(s.max, 1, 'max must fold over plottable samples only');
  assert.equal(s.min, 0, 'min must fold over plottable samples only');
  assert.equal(s.range, 1);
  assert.ok(Number.isFinite(s.max) && Number.isFinite(s.min) && Number.isFinite(s.range));
});

test('sparkScale: a NaN hole no longer poisons the extrema to NaN', () => {
  const s = sparkScale([0.9, NaN, 0.95], 100, 28);
  assert.equal(s.max, 1);
  assert.equal(s.min, 0);
  assert.equal(s.range, 1);
});

test('sparkScale: the 1/0 extrema seeds are preserved (y-axis framing unchanged)', () => {
  // These seeds guarantee the axis always spans at least [0, 1], so a flat
  // all-zero series renders at the floor and a tiny series is not amplified.
  // They are the current framing for every sparkline on six tabs; dropping
  // them while fixing null handling would silently restyle all of them. Only
  // WHICH samples enter the fold changes.
  const small = sparkScale([0.2, 0.4], 100, 28);
  assert.equal(small.max, 1, 'the max seed of 1 must survive a small series');
  assert.equal(small.min, 0, 'the min seed of 0 must survive a positive series');

  const large = sparkScale([2, null, 5], 100, 28);
  assert.equal(large.max, 5, 'a real sample above the seed still wins');
  assert.equal(large.min, 0);

  const negative = sparkScale([-4, null, -2], 100, 28);
  assert.equal(negative.min, -4, 'a real sample below the seed still wins');
  assert.equal(negative.max, 1);
});

test('sparkScale: points are index-preserving — a hole never shifts the x-axis', () => {
  // x is derived from the ORIGINAL index, so the surviving samples stay at the
  // x positions they would have occupied with no hole. Compacting instead
  // would silently redate every remaining point.
  const s = sparkScale([1, null, 3], 100, 28);

  assert.equal(s.points.length, 3, 'points must be the same length as values');
  assert.equal(s.stepX, 50);
  assert.equal(s.points[0][0], 0, 'index 0 sits at x=0');
  assert.equal(s.points[2][0], 100, 'index 2 stays at x=100, not compacted to 50');

  // max === 3 here (a real sample above the seed), min === 0, range === 3, so
  // the top sample plots exactly at the top of the box.
  assert.equal(s.max, 3);
  assert.equal(s.points[2][1], 0, 'the max sample plots at the top edge');
  assertClose(s.points[0][1], 28 - (1 / 3) * 28, 'the first sample plots proportionally');
});

test('sparkScale: a hole carries NO fabricated coordinate', () => {
  // The headline defect. Pre-fix this index produced a real [50, 28] point —
  // pinned to the value-0 baseline and joined to both neighbours, reading as a
  // measured plunge to zero. It must now be strictly null so downstream path
  // building can break the line rather than draw through it.
  const s = sparkScale([1, null, 3], 100, 28);
  assert.equal(s.points[1], null, 'a hole must be null, not [50, 28] or any other point');
});

test('sparkScale: undefined and NaN holes are null points too', () => {
  for (const hole of [undefined, NaN]) {
    const s = sparkScale([1, hole, 3], 100, 28);
    assert.equal(s.points[1], null, `a ${String(hole)} hole must be a null point`);
    assert.ok(Number.isFinite(s.points[0][1]), 'surviving points stay finite');
    assert.ok(Number.isFinite(s.points[2][1]), 'surviving points stay finite');
  }
});

test('sparkScale: an all-hole series measures nothing and plots nothing', () => {
  // Pre-fix this exact input yielded max=1, min=0 from the seeds alone and
  // three REAL points at y=height — a fully synthetic flat line along the
  // chart floor, asserting three measurements that were never taken.
  const s = sparkScale([null, null, null], 100, 28);

  assert.equal(s.min, 0, 'the seeds alone frame an empty series');
  assert.equal(s.max, 1);
  assert.equal(s.range, 1);
  assert.equal(s.points.length, 3, 'points still mirrors the input length');
  for (let i = 0; i < s.points.length; i++) {
    assert.equal(s.points[i], null, `points[${i}] must be null, not a floor-pinned point`);
  }
});

test('sparkScale: an empty or missing series yields the seed-only scale', () => {
  for (const values of [[], null, undefined]) {
    const s = sparkScale(values, 100, 28);
    assert.equal(s.min, 0);
    assert.equal(s.max, 1);
    assert.equal(s.range, 1);
    assert.deepEqual(s.points, [], 'no input samples means no points');
  }
});
