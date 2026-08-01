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

const { isPlottable, sparkScale, sparkPaths, stepPaths } = sp;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/spark_path.js';

// The module's complete public surface. Asserted EXHAUSTIVELY below: the
// run-splitting and area-closing helpers are deliberately module-private, and
// an accidental export would quietly widen the contract charts.jsx depends on.
const EXPECTED_FUNCTION_NAMES = ['isPlottable', 'sparkPaths', 'sparkScale', 'stepPaths'];

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

// ── Path-string readers ────────────────────────────────────────────────────
// Both builders emit space-joined tokens of the form `M<x>,<y>` / `L<x>,<y>` /
// `Z`, so tokenising on whitespace is exact rather than a loose regex scrape.

function pathTokens(d) {
  return d.split(/\s+/).filter(Boolean);
}

function coords(d) {
  return pathTokens(d)
    .filter(t => t[0] === 'M' || t[0] === 'L')
    .map(t => t.slice(1).split(',').map(Number));
}

function countCommand(d, letter) {
  return pathTokens(d).filter(t => t[0] === letter).length;
}

function xs(d) {
  return coords(d).map(([x]) => x);
}

// Distinct plotted positions. A run of one sample is emitted as a zero-length
// segment (`M x,y L x,y`), so counting raw coordinate TOKENS double-counts
// every isolated dot; the meaningful quantity is how many distinct places the
// chart actually marks.
function distinctCoords(d) {
  return [...new Set(coords(d).map(([x, y]) => `${x},${y}`))].map(k => k.split(',').map(Number));
}

function ys(d) {
  return coords(d).map(([, y]) => y);
}

// ── Frozen snapshots of the PRE-EXTRACTION charts.jsx arithmetic ───────────
// Transcribed verbatim from charts.jsx:44-54 (Sparkline) and :68-92
// (StepSpark) as they stood before task 3436. These are deliberately FROZEN
// COPIES, not a live mirror of charts.jsx — charts.jsx no longer contains this
// code at all (test_charts_null_samples.py asserts its absence). Their only
// job is to answer, permanently and by exact string comparison, the question
// reviewers and operators actually care about: "did this refactor quietly move
// every existing chart?" Identical arithmetic produces identical doubles
// produces identical strings, so the comparison needs no float tolerance.
//
// Valid for HOLE-FREE inputs only — fed a hole, these reproduce the very bug
// being fixed.

function legacySparklinePaths(values, width, height) {
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const stepX = width / Math.max(values.length - 1, 1);
  const points = values.map((v, i) => {
    const x = i * stepX;
    const y = height - ((v - min) / range) * height;
    return [x, y];
  });
  const linePath = points.map((p, i) => (i === 0 ? `M${p[0]},${p[1]}` : `L${p[0]},${p[1]}`)).join(' ');
  const areaPath = `${linePath} L${width},${height} L0,${height} Z`;
  return { line: linePath, area: areaPath };
}

function legacyStepPaths(values, width, height) {
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const stepX = width / Math.max(values.length - 1, 1);
  const points = values.map((v, i) => {
    const x = i * stepX;
    const y = height - ((v - min) / range) * height;
    return [x, y];
  });
  const parts = [`M${points[0][0]},${points[0][1]}`];
  if (points.length === 1) {
    parts.push(`L${width},${points[0][1]}`);
  }
  for (let i = 1; i < points.length; i++) {
    parts.push(`L${points[i][0]},${points[i - 1][1]}`);
    parts.push(`L${points[i][0]},${points[i][1]}`);
  }
  const linePath = parts.join(' ');
  const areaPath = `${linePath} L${width},${height} L0,${height} Z`;
  return { line: linePath, area: areaPath };
}

// ---------------------------------------------------------------------------
// Module surface and dual export
// ---------------------------------------------------------------------------

test('default-imported module exposes exactly the spark-path functions', () => {
  assert.deepEqual(
    Object.keys(sp).sort(),
    EXPECTED_FUNCTION_NAMES.slice().sort(),
    'the public surface must be exactly these four — the run-splitting and ' +
      'area-closing helpers stay module-private',
  );
  for (const name of EXPECTED_FUNCTION_NAMES) {
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

    // The fresh require() and the top-level import() produce two distinct API
    // object instances (separate module executions), so compare structurally
    // — same set of exported names, each a function — rather than asserting
    // reference/deep equality against the ESM-imported `sp`.
    assert.deepEqual(
      Object.keys(globalThis.window.DF_SPARK_PATH).sort(),
      EXPECTED_FUNCTION_NAMES.slice().sort(),
    );
    assert.deepEqual(Object.keys(required).sort(), EXPECTED_FUNCTION_NAMES.slice().sort());

    // charts.jsx destructures this global at module top level, so every name
    // it reaches for must be present on the browser export path specifically —
    // not merely on the CommonJS one.
    for (const name of EXPECTED_FUNCTION_NAMES) {
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

// ---------------------------------------------------------------------------
// sparkPaths — the smooth (Sparkline) builder.
//
// Widths/lengths are chosen so stepX divides exactly (100/2 = 50, 100/4 = 25)
// and the last point lands exactly on x=width, which makes the exact-string
// equivalence assertions float-safe.
// ---------------------------------------------------------------------------

test('sparkPaths: the frozen legacy snapshot really is the pre-fix output', () => {
  // Guards the guard. If legacySparklinePaths were mistranscribed, the
  // equivalence tests below would compare the new code against a fiction and
  // pass while every chart silently moved. Pinning one literal proves the
  // snapshot emits genuine pre-fix strings.
  assert.equal(
    legacySparklinePaths([1, 2, 3], 100, 28).line,
    'M0,18.666666666666668 L50,9.333333333333336 L100,0',
  );
});

test('sparkPaths: hole-free output is byte-identical to the pre-fix code', () => {
  // The risk that actually matters: this change touches the render path of six
  // tabs, so "behaviour-preserving for existing data" must be an assertion,
  // not a claim. Note the per-subpath area close collapses to the legacy
  // full-width `L100,28 L0,28 Z` for a single full-width run — x_last is
  // exactly width and x_first is exactly 0 — so the generalisation is provably
  // not a restyle.
  for (const values of [[1, 2, 3], [1, 2, 3, 4, 5], [0, 0, 0], [-4, 2, -1]]) {
    const expected = legacySparklinePaths(values, 100, 28);
    const actual = sparkPaths(values, 100, 28);
    assert.equal(actual.line, expected.line, `line drifted for ${JSON.stringify(values)}`);
    assert.equal(actual.area, expected.area, `area drifted for ${JSON.stringify(values)}`);
  }
});

test('sparkPaths: a hole starts a new subpath and no segment crosses it', () => {
  const { line } = sparkPaths([1, 2, null, 4, 5], 100, 28);

  assert.equal(countCommand(line, 'M'), 2, 'the hole must split the line into two subpaths');
  assert.equal(
    coords(line)[2][0],
    75,
    'the second subpath starts at the index-3 position (75), NOT compacted to 50',
  );
  assert.ok(
    !xs(line).includes(50),
    `nothing may be drawn at the hole's own x — got ${line}`,
  );
  assert.deepEqual(xs(line), [0, 25, 75, 100], 'surviving samples keep their original x');
});

test('sparkPaths: a null is never plotted as a zero-value point', () => {
  // THE HEADLINE DEFECT. Pre-fix, the middle sample of this series landed
  // exactly on the value-0 baseline (y = height) and was joined to both
  // neighbours — a proportion series sitting at ~0.95 rendering as a plunge to
  // the floor and back, indistinguishable from a measured collapse.
  const values = [0.95, null, 0.96];
  const { line } = sparkPaths(values, 100, 28);
  const s = sparkScale(values, 100, 28);
  const baselineY = 28 - ((0 - s.min) / s.range) * 28;

  assert.equal(baselineY, 28, 'sanity: value 0 sits at the chart floor for this scale');
  for (const y of ys(line)) {
    assert.notEqual(y, baselineY, `a hole was drawn at the value-0 baseline: ${line}`);
  }
  assert.equal(distinctCoords(line).length, 2, 'only the two real samples are plotted');

  // Each surviving sample is isolated (flanked by the hole and the series
  // edge), so this series is TWO dots with no connecting line — not one line
  // hopping over the gap. Drawing a segment from x=0 to x=100 here would
  // interpolate straight across a slot that holds no measurement, which is the
  // same fabrication as plotting the hole itself.
  assert.equal(countCommand(line, 'M'), 2, 'two isolated samples, two subpaths');
  assert.deepEqual(distinctCoords(line).map(([x]) => x), [0, 100]);
});

test('sparkPaths: with a negative minimum the fabricated point would land mid-chart', () => {
  // Same defect, different disguise: here min is -4, so the pre-fix null
  // coerced to 0 plotted MID-chart rather than at the floor — a hole rendering
  // as a plausible-looking real measurement.
  const values = [-4, null, -2];
  assert.equal(sparkScale(values, 100, 28).min, -4, 'sanity: the floor is well below zero here');

  const { line } = sparkPaths(values, 100, 28);

  assert.equal(distinctCoords(line).length, 2, 'only the two real samples are plotted');
  assert.ok(!xs(line).includes(50), `nothing may be drawn at the hole's x — got ${line}`);
  assert.deepEqual(distinctCoords(line).map(([x]) => x), [0, 100]);
});

test('sparkPaths: undefined and NaN holes behave exactly like null', () => {
  // Pre-fix these were far WORSE than null: they poisoned the extrema to NaN
  // and produced an all-NaN path (`MNaN,NaN LNaN,NaN ...`) that rendered
  // nothing at all.
  const reference = sparkPaths([1, 2, null, 4, 5], 100, 28);

  for (const hole of [undefined, NaN]) {
    const actual = sparkPaths([1, 2, hole, 4, 5], 100, 28);
    assert.equal(actual.line, reference.line, `${String(hole)} must break like null`);
    assert.equal(actual.area, reference.area, `${String(hole)} must break like null`);
    for (const n of coords(actual.line).flat()) {
      assert.ok(Number.isFinite(n), `every coordinate must be finite — got ${actual.line}`);
    }
  }
});

test('sparkPaths: leading and trailing holes do not shift the x-axis', () => {
  const leading = sparkPaths([null, 2, 3], 100, 28);
  assert.equal(countCommand(leading.line, 'M'), 1, 'one contiguous run');
  assert.equal(coords(leading.line)[0][0], 50, 'the run starts at its own index, not at x=0');
  assert.deepEqual(xs(leading.line), [50, 100]);

  const trailing = sparkPaths([1, 2, null], 100, 28);
  assert.equal(countCommand(trailing.line, 'M'), 1, 'one contiguous run');
  assert.deepEqual(xs(trailing.line), [0, 50], 'the run ends at its own index, not at x=100');
});

test('sparkPaths: an isolated sample is a visible dot, not a vanished one', () => {
  // A subpath containing only a moveto renders NOTHING in SVG — charts.jsx
  // already documented that trap for the single-data-point case. Once holes
  // split the line, every interior island would hit it, trading one
  // invisible-data bug for another. A zero-length segment renders as a dot
  // under Sparkline's existing strokeLinecap="round", so no markup change is
  // needed.
  const { line, area } = sparkPaths([null, 7, null], 100, 28);

  assert.equal(line, 'M50,0 L50,0', 'a lone sample emits a zero-length segment at its own x');
  assert.equal(area, '', 'a single point has no area — a zero-width sliver is meaningless');
});

test('sparkPaths: area closes per subpath, never spanning a hole', () => {
  const { area } = sparkPaths([1, 2, null, 4, 5], 100, 28);

  assert.equal(countCommand(area, 'M'), 2, 'one closed shape per run');
  assert.equal(countCommand(area, 'Z'), 2, 'each run closes on itself');
  // Each run closes at its OWN first/last x (25/0 and 100/75), so no fill is
  // painted across the slot with no measurement.
  assert.equal(area, 'M0,22.4 L25,16.799999999999997 L25,28 L0,28 Z M75,5.599999999999998 L100,0 L100,28 L75,28 Z');
});

test('sparkPaths: an all-hole or empty series draws nothing at all', () => {
  // Pre-fix, [null, null, null] produced max=1/min=0 from the seeds and three
  // real points at y=height — a fully synthetic flat line along the chart
  // floor. Returning empty strings (rather than throwing) leaves the
  // decline-to-render decision at the call site, where the component can skip
  // the <svg> entirely.
  for (const values of [[null, null, null], [undefined, NaN], [], null, undefined]) {
    const { line, area } = sparkPaths(values, 100, 28);
    assert.equal(line, '', `expected no line for ${JSON.stringify(values) ?? String(values)}`);
    assert.equal(area, '', `expected no area for ${JSON.stringify(values) ?? String(values)}`);
  }
});

test('sparkPaths: does not mutate its input array', () => {
  const values = [1, null, 3, null, 5];
  const before = values.slice();
  sparkPaths(values, 100, 28);
  assert.deepEqual(values, before, 'the caller owns the series; it must come back untouched');
});

// ---------------------------------------------------------------------------
// stepPaths — the step (StepSpark) builder: horizontal-then-vertical edges, no
// diagonals, so discrete state transitions read as sharp steps.
// ---------------------------------------------------------------------------

test('stepPaths: the frozen legacy snapshot really is the pre-fix output', () => {
  assert.equal(
    legacyStepPaths([1, 2, 3], 100, 28).line,
    'M0,18.666666666666668 L50,18.666666666666668 L50,9.333333333333336 ' +
      'L100,9.333333333333336 L100,0',
  );
  assert.equal(legacyStepPaths([7], 100, 28).line, 'M0,0 L100,0');
});

test('stepPaths: hole-free output is byte-identical to the pre-fix code', () => {
  for (const values of [[1, 2, 3], [1, 2, 3, 4, 5], [0, 0, 0], [-4, 2, -1]]) {
    const expected = legacyStepPaths(values, 100, 28);
    const actual = stepPaths(values, 100, 28);
    assert.equal(actual.line, expected.line, `line drifted for ${JSON.stringify(values)}`);
    assert.equal(actual.area, expected.area, `area drifted for ${JSON.stringify(values)}`);
  }
});

test('stepPaths: the single-sample full-width tick is preserved verbatim', () => {
  // Pre-existing documented behaviour with no hole involved (charts.jsx:80-84):
  // a lone sample would otherwise be a bare moveto and render nothing, so it
  // is drawn as a full-width horizontal tick. Reproduced exactly, area
  // included, because it is a genuine 2-point run.
  const actual = stepPaths([7], 100, 28);
  assert.equal(actual.line, 'M0,0 L100,0');
  assert.equal(actual.line, legacyStepPaths([7], 100, 28).line);
  assert.equal(actual.area, legacyStepPaths([7], 100, 28).area);
});

test('stepPaths: a hole breaks the step and no horizontal carries across it', () => {
  // The step builder makes this defect worse than the smooth one: a horizontal
  // edge at the pre-hole y would assert the value PERSISTED through the
  // missing slot, then drop vertically to the fabricated zero.
  const { line } = stepPaths([1, 2, null, 4, 5], 100, 28);

  assert.equal(countCommand(line, 'M'), 2, 'the hole must split the step into two subpaths');
  assert.equal(coords(line)[3][0], 75, 'the second subpath starts at the index-3 position');
  assert.ok(!xs(line).includes(50), `nothing may be drawn at the hole's x — got ${line}`);
  assert.equal(
    line,
    'M0,22.4 L25,22.4 L25,16.799999999999997 ' +
      'M75,5.599999999999998 L100,5.599999999999998 L100,0',
  );
});

test('stepPaths: a null is not stepped down to as a zero', () => {
  // Pre-fix this drew a square-cornered plunge to the chart floor and back —
  // a flat series reading as a total, measured collapse.
  const values = [3, null, 3];
  const { line } = stepPaths(values, 100, 28);
  const s = sparkScale(values, 100, 28);
  const baselineY = 28 - ((0 - s.min) / s.range) * 28;

  assert.equal(baselineY, 28, 'sanity: value 0 sits at the chart floor for this scale');
  for (const y of ys(line)) {
    assert.notEqual(y, baselineY, `a hole was stepped to at the value-0 baseline: ${line}`);
  }
  assert.equal(distinctCoords(line).length, 2, 'only the two real samples are plotted');
});

test('stepPaths: an isolated sample is not held across the neighbouring holes', () => {
  // A zero-length segment, rendering as a visible square under StepSpark's
  // existing strokeLinecap="square" — and deliberately NOT a one-step-wide
  // tick. Extending a step into a known-missing slot would assert the value
  // persisted there, which is exactly the synthetic-data class this fix
  // removes.
  const { line, area } = stepPaths([null, 7, null], 100, 28);

  assert.equal(line, 'M50,0 L50,0');
  assert.ok(!xs(line).includes(0), 'the step must not reach back into the leading hole');
  assert.ok(!xs(line).includes(100), 'the step must not reach into the trailing hole');
  assert.equal(area, '', 'a zero-length run has no area');
});

test('stepPaths: undefined and NaN holes behave exactly like null', () => {
  const reference = stepPaths([1, 2, null, 4, 5], 100, 28);

  for (const hole of [undefined, NaN]) {
    const actual = stepPaths([1, 2, hole, 4, 5], 100, 28);
    assert.equal(actual.line, reference.line, `${String(hole)} must break like null`);
    assert.equal(actual.area, reference.area, `${String(hole)} must break like null`);
    for (const n of coords(actual.line).flat()) {
      assert.ok(Number.isFinite(n), `every coordinate must be finite — got ${actual.line}`);
    }
  }
});

test('stepPaths: area closes per subpath, never spanning a hole', () => {
  const { area } = stepPaths([1, 2, null, 4, 5], 100, 28);

  assert.equal(countCommand(area, 'M'), 2, 'one closed shape per run');
  assert.equal(countCommand(area, 'Z'), 2, 'each run closes on itself');
  assert.equal(
    area,
    'M0,22.4 L25,22.4 L25,16.799999999999997 L25,28 L0,28 Z ' +
      'M75,5.599999999999998 L100,5.599999999999998 L100,0 L100,28 L75,28 Z',
  );
});

test('stepPaths: an all-hole or empty series draws nothing at all', () => {
  for (const values of [[null, null, null], [undefined, NaN], [null], [], null, undefined]) {
    const { line, area } = stepPaths(values, 100, 28);
    assert.equal(line, '', `expected no line for ${JSON.stringify(values) ?? String(values)}`);
    assert.equal(area, '', `expected no area for ${JSON.stringify(values) ?? String(values)}`);
  }
});

test('stepPaths: does not mutate its input array', () => {
  const values = [1, null, 3, null, 5];
  const before = values.slice();
  stepPaths(values, 100, 28);
  assert.deepEqual(values, before, 'the caller owns the series; it must come back untouched');
});
