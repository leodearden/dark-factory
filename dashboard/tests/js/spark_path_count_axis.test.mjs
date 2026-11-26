// Behavioural tests for spark_path.js's one COUNT-AXIS rule, axisMax (task
// 6370): a chart marked `countAxis` snaps its axis maximum via the
// module-private niceCountMax (task 5121); every other chart keeps its exact
// data maximum. It is the one spark_path.js entry that is a policy rather than
// scale/path math, so it sits apart from spark_path.test.mjs, which covers the
// rest of the module and its exhaustive public surface. Run via `node --test`;
// the pytest shim dashboard/tests/test_graph_layout_js.py picks this file up by
// glob.
//
// spark_path.js resolves as CommonJS, and node cannot statically detect named
// exports assigned from a variable, so the module is default-imported and
// destructured.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import sp from '../../src/dashboard/static/redux/spark_path.js';

const { axisMax } = sp;

const wholeTicks = (max, ticks) =>
  Array.from({ length: ticks + 1 }, (_, i) => (max * i) / ticks).filter(t => Number.isInteger(t));

test('axisMax: a count axis snaps to the smallest multiple of the tick count', () => {
  assert.equal(axisMax(1, 4, true), 4, 'the plottableMax seed floor of an idle count series');
  assert.equal(axisMax(4, 4, true), 4);
  assert.equal(axisMax(7, 4, true), 8);
  assert.equal(axisMax(8, 4, true), 8);
  assert.equal(axisMax(999, 4, true), 1000);
  assert.equal(axisMax(1000, 4, true), 1000);
  assert.equal(axisMax(1001, 4, true), 1004);
  assert.equal(axisMax(0, 4, true), 0);

  // The rule is generic in the tick count, not hardwired to 4.
  assert.equal(axisMax(7, 5, true), 10);
  assert.equal(axisMax(11, 3, true), 12);
});

test('axisMax: every other axis keeps its exact data maximum', () => {
  assert.equal(axisMax(7, 4, false), 7);
  assert.equal(axisMax(0.8, 4, false), 0.8);
  assert.equal(axisMax(2.5, 4, false), 2.5);
  assert.equal(axisMax(7, 4, undefined), 7);

  // Snapping a fraction axis is the hazardous direction, so only a literal
  // `true` snaps; anything else keeps the axis exact.
  assert.equal(axisMax(7, 4, 'true'), 7);
  assert.equal(axisMax(7, 4, 1), 7);
});

test('axisMax: smallest whole multiple, fixed point, every tick whole — every count axis 1..200', () => {
  for (let m = 1; m <= 200; m++) {
    const r = axisMax(m, 4, true);
    assert.equal(r % 4, 0, `m=${m}: ${r} is not a multiple of the tick count`);
    assert.ok(r >= m, `m=${m}: ${r} would clip the data`);
    assert.ok(r - 4 < m, `m=${m}: ${r} is not the SMALLEST such multiple`);
    assert.equal(axisMax(r, 4, true), r, `m=${m}: not idempotent at ${r}`);

    const ticks = Array.from({ length: 5 }, (_, i) => (r * i) / 4);
    assert.ok(ticks.every(t => Number.isInteger(t)), `m=${m}: fractional tick in ${JSON.stringify(ticks)}`);
    assert.equal(new Set(ticks).size, ticks.length, `m=${m}: duplicate tick in ${JSON.stringify(ticks)}`);
  }
});

test('axisMax: the cost the snap removes, measured', () => {
  // The control that keeps the sweep above from being vacuous: WITHOUT the
  // snap, half of all count axes in 1..200 have only their floor and peak whole.
  const histogram = {};
  for (let m = 1; m <= 200; m++) {
    const whole = wholeTicks(axisMax(m, 4, false), 4).length;
    histogram[whole] = (histogram[whole] || 0) + 1;
  }
  assert.deepEqual(histogram, { 2: 100, 3: 50, 5: 50 });
});

test('axisMax: on a count axis, a non-finite max or an invalid tick count passes the max through, never NaN', () => {
  // This runs inside a render. Unguarded, (7, 0) is Math.ceil(Infinity) * 0 =
  // NaN, which blanks the whole chart — passing the max through is exactly the
  // un-snapped behaviour.
  for (const bad of [NaN, Infinity, -Infinity, null, undefined, '7']) {
    assert.ok(
      Object.is(axisMax(bad, 4, true), bad),
      `axisMax(${String(bad)}, 4, true) must pass the max through`,
    );
  }
  for (const bad of [0, -4, 2.5, NaN, undefined, null]) {
    assert.equal(axisMax(7, bad, true), 7, `axisMax(7, ${String(bad)}, true) must pass 7 through`);
  }
});

test('axisMax: why snapping is opt-in — a fraction axis would read 0%..400%', () => {
  assert.equal(axisMax(0.8, 4, true), 4);
  assert.equal(axisMax(1, 4, true), 4);
});
