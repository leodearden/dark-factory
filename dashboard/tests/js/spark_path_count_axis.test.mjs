// Behavioural tests for spark_path.js's two COUNT-AXIS rules: formatCountTick
// (the y-tick LABEL rule, task 4232) and niceCountMax (the axis MAXIMUM rule,
// task 5121). They are the only spark_path.js entries that are not scale/path
// math, so they sit apart from spark_path.test.mjs, which covers the rest of the
// module and its exhaustive public surface. Run via `node --test`; the pytest
// shim dashboard/tests/test_graph_layout_js.py picks this file up by glob.
//
// spark_path.js resolves as CommonJS, and node cannot statically detect named
// exports assigned from a variable, so the module is default-imported and
// destructured.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import sp from '../../src/dashboard/static/redux/spark_path.js';

const { formatCountTick, niceCountMax } = sp;

// ---------------------------------------------------------------------------
// formatCountTick — y-tick LABEL formatter for an INTEGER-COUNT axis (task 4232)
//
// charts.jsx's LineChart generates `minV + (range * i) / 4` with minV = 0 and
// range = plottableMax(all, 1), and defaults `formatY` to `(v) => String(v)`.
// On a count axis whose maximum is not a multiple of 4 that draws fractional
// labels: a merge-attempt axis peaking at 7 renders `1.75` / `3.5` / `5.25` —
// counts that cannot exist.
//
// THE FIX LABELS A TICK ONLY WHEN IT IS A WHOLE NUMBER, and returns '' for
// everything else. It deliberately does NOT round, because rounding trades
// fractional labels for FABRICATED ones. Measured under node on the real tick
// shape below:
//   maxV=1   Math.round -> '0','0','1','1','1'  — duplicate labels, the exact
//                                                 shape task 4059 was filed for
//   maxV=2   Math.round -> '0','1','1','2','2'
//   maxV=5   Math.round -> '0','1','3','4','5'  — 1.25 labelled '1', 2.5 '3'
//   maxV=999 Math.round -> a gridline at 749.25 labelled '749'
// and maxV=1 is NOT a corner case: `plottableMax(values, 1)` seeds at 1, so
// every idle or all-zero count series lands exactly there. A rounding helper
// would therefore ship the duplicate-label defect on the most common state a
// dashboard is ever in.
//
// Blanking follows this repo's own stated norm — omit rather than fabricate,
// the same rule as "an absent bar in a labelled slot reads as 'not measured'".
// An unlabelled gridline is an omission; a `2` printed on a 1.75 gridline is a
// measurement that was never taken. And because maxV is always an integer on a
// count axis, the FLOOR and the PEAK are always labelled — the two labels that
// carry the most information.
//
// EXACTNESS, WITH NO TOLERANCE. `range * i` is an exact integer (at most
// 4 * maxV, far inside 2^53) and the tick count 4 is a power of two, so
// `(range * i) / 4` merely decrements the exponent and is exact in IEEE-754.
// `Number.isInteger` is therefore mathematically correct on every tick.
// ---------------------------------------------------------------------------

// LineChart's real y-tick shape at minV = 0: 5 ticks spanning 0..maxV.
const lineChartTicks = maxV => Array.from({ length: 5 }, (_, i) => (maxV * i) / 4);
const countAxisLabels = maxV => lineChartTicks(maxV).map(t => formatCountTick(t));

test('formatCountTick: a whole number renders as its own digits', () => {
  assert.equal(formatCountTick(0), '0', 'a measured zero is a label, never a blank');
  assert.equal(formatCountTick(1), '1');
  assert.equal(formatCountTick(7), '7');
  assert.equal(formatCountTick(999), '999');
  assert.equal(formatCountTick(-3), '-3', 'sign is not the predicate; wholeness is');
});

test('formatCountTick: a fractional tick renders as the empty string, not a rounded one', () => {
  // Every one of these is a real gridline off a real count axis: 1.75/3.5/5.25
  // come off maxV=7, 0.25 off the maxV=1 seed floor, 249.75 off maxV=999.
  assert.equal(formatCountTick(0.25), '');
  assert.equal(formatCountTick(1.75), '');
  assert.equal(formatCountTick(3.5), '');
  assert.equal(formatCountTick(249.75), '');
  assert.equal(formatCountTick(-0.5), '');
});

test('formatCountTick: non-finite and absent input yield a blank, never "NaN" or "undefined"', () => {
  // These must never reach an axis as text. The function is called from JSX
  // inside a render, so it must also never throw: one exception here blanks the
  // entire chart, which is strictly worse than an unlabelled gridline.
  for (const bad of [NaN, Infinity, -Infinity, null, undefined, '4', '', {}, [], true]) {
    assert.equal(
      formatCountTick(bad),
      '',
      `formatCountTick(${JSON.stringify(bad)}) must be blank`,
    );
  }
});

test('formatCountTick: the measured axis table over LineChart real ticks', () => {
  // Pinned from an actual node run over `(maxV * i) / 4`. maxV=1 is the
  // `plottableMax(values, 1)` seed floor — the idle / all-zero dashboard, which
  // every count chart shows before it has data. A ROUNDING helper renders
  // '0','0','1','1','1' there: the task-4059 duplicate-label defect.
  assert.deepEqual(countAxisLabels(1), ['0', '', '', '', '1'], 'the seed floor');
  assert.deepEqual(countAxisLabels(2), ['0', '', '1', '', '2']);
  assert.deepEqual(countAxisLabels(7), ['0', '', '', '', '7'], 'was 1.75/3.5/5.25');
  assert.deepEqual(countAxisLabels(8), ['0', '2', '4', '6', '8'], 'a multiple of 4 loses nothing');
  assert.deepEqual(countAxisLabels(10), ['0', '', '5', '', '10']);

  // The same five axes under `Math.round`, so the table above is a CHOICE on
  // the record rather than an accident. maxV=1 duplicates, maxV=7 fabricates a
  // '2' onto the 1.75 gridline.
  const rounded = maxV => lineChartTicks(maxV).map(t => String(Math.round(t)));
  assert.deepEqual(rounded(1), ['0', '0', '1', '1', '1']);
  assert.deepEqual(rounded(7), ['0', '2', '4', '5', '7']);
});

test('formatCountTick: the anti-4059 invariants hold across every count axis 1..200', () => {
  // The three properties that make this safe to wire at a caller sight-unseen.
  for (let maxV = 1; maxV <= 200; maxV++) {
    const labels = countAxisLabels(maxV);
    assert.equal(labels.length, 5, `maxV=${maxV}`);

    // 1. The floor and the peak are ALWAYS labelled — maxV is an integer on a
    //    count axis, so tick 0 and tick 4 are both whole by construction.
    assert.equal(labels[0], '0', `maxV=${maxV}: the floor must be labelled`);
    assert.equal(labels[4], String(maxV), `maxV=${maxV}: the peak must be labelled`);

    // 2. No non-empty label ever repeats. This is the arm `Math.round` fails —
    //    at maxV=1,2,3,5,6,... it prints the same number on two gridlines,
    //    which is what makes a reader distrust the whole axis.
    const drawn = labels.filter(l => l !== '');
    assert.equal(
      new Set(drawn).size,
      drawn.length,
      `maxV=${maxV}: duplicate label in ${JSON.stringify(labels)}`,
    );

    // 3. No label is ever fractional — the defect this task exists to remove.
    for (const l of labels) {
      assert.ok(!l.includes('.'), `maxV=${maxV}: fractional label ${l}`);
    }
  }
});

test('formatCountTick: Math.round would fail the no-duplicate invariant it replaces', () => {
  // The negative control. Without this, the sweep above could pass vacuously if
  // the helper were quietly reimplemented as a rounder on some future edit.
  const roundedDupes = [];
  for (let maxV = 1; maxV <= 200; maxV++) {
    const drawn = lineChartTicks(maxV).map(t => String(Math.round(t)));
    if (new Set(drawn).size !== drawn.length) roundedDupes.push(maxV);
  }
  assert.ok(
    roundedDupes.length > 0,
    'a rounding formatter must be shown to actually duplicate labels',
  );
  assert.ok(roundedDupes.includes(1), 'maxV=1 (the seed floor) is among them');
  assert.ok(roundedDupes.includes(2), 'and maxV=2');
});

// ---------------------------------------------------------------------------
// niceCountMax — snap a count axis's MAXIMUM up to a multiple of the tick count
// (task 5121), so every tick is a whole number and formatCountTick labels all
// five of them instead of blanking the fractional ones.
// ---------------------------------------------------------------------------

test('niceCountMax: the measured table', () => {
  assert.equal(niceCountMax(1, 4), 4, 'the plottableMax seed floor of an idle count series');
  assert.equal(niceCountMax(4, 4), 4);
  assert.equal(niceCountMax(7, 4), 8);
  assert.equal(niceCountMax(8, 4), 8);
  assert.equal(niceCountMax(999, 4), 1000);
  assert.equal(niceCountMax(1000, 4), 1000);
  assert.equal(niceCountMax(1001, 4), 1004);
  assert.equal(niceCountMax(0, 4), 0);

  // The rule is generic in the tick count, not hardwired to 4.
  assert.equal(niceCountMax(7, 5), 10);
  assert.equal(niceCountMax(11, 3), 12);
});

test('niceCountMax: smallest whole multiple, fixed point, every tick labelled — every count axis 1..200', () => {
  for (let m = 1; m <= 200; m++) {
    const r = niceCountMax(m, 4);
    assert.equal(r % 4, 0, `m=${m}: ${r} is not a multiple of the tick count`);
    assert.ok(r >= m, `m=${m}: ${r} would clip the data`);
    assert.ok(r - 4 < m, `m=${m}: ${r} is not the SMALLEST such multiple`);
    assert.equal(niceCountMax(r, 4), r, `m=${m}: not idempotent at ${r}`);

    const labels = countAxisLabels(r);
    assert.ok(!labels.includes(''), `m=${m}: blank tick in ${JSON.stringify(labels)}`);
    assert.equal(new Set(labels).size, labels.length, `m=${m}: duplicate in ${JSON.stringify(labels)}`);
    assert.equal(labels[4], String(r), `m=${m}: the peak must be labelled`);
  }
});

test('niceCountMax: the cost it removes, measured', () => {
  // The control that keeps the sweep above from being vacuous: WITHOUT the
  // snap, half of all count axes in 1..200 label only their floor and peak.
  const histogram = {};
  for (let m = 1; m <= 200; m++) {
    const labelled = countAxisLabels(m).filter(l => l !== '').length;
    histogram[labelled] = (histogram[labelled] || 0) + 1;
  }
  assert.deepEqual(histogram, { 2: 100, 3: 50, 5: 50 });
});

test('niceCountMax: a non-finite max or an invalid tick count passes the max through, never NaN', () => {
  // This runs inside a render. Unguarded, (7, 0) is Math.ceil(Infinity) * 0 =
  // NaN, which blanks the whole chart — passing the max through is exactly the
  // un-snapped behaviour.
  for (const bad of [NaN, Infinity, -Infinity, null, undefined, '7']) {
    assert.ok(
      Object.is(niceCountMax(bad, 4), bad),
      `niceCountMax(${String(bad)}, 4) must pass the max through`,
    );
  }
  for (const bad of [0, -4, 2.5, NaN, undefined, null]) {
    assert.equal(niceCountMax(7, bad), 7, `niceCountMax(7, ${String(bad)}) must pass 7 through`);
  }
});

test('niceCountMax: a fraction axis must not opt in', () => {
  // The executable reason snapping is opt-in per caller: a 100%-normalized or
  // ratio axis would read 0%..400%.
  assert.equal(niceCountMax(0.8, 4), 4);
  assert.equal(niceCountMax(1, 4), 4);
});
