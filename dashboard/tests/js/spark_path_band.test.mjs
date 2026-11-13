// Behavioural tests for spark_path.js's bandPaths: the filled band LineChart
// draws between a lower and an upper bound series (the quality tab's Wilson
// band, task 8461). It sits apart from spark_path.test.mjs, which covers the
// rest of the module and its exhaustive public surface, as
// spark_path_count_axis.test.mjs does for axisMax. Run via `node --test`; the
// pytest shim dashboard/tests/test_graph_layout_js.py picks this file up by
// glob.
//
// spark_path.js resolves as CommonJS, so it is default-imported and
// destructured.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import sp from '../../src/dashboard/static/redux/spark_path.js';

const { axisPaths, axisY, bandPaths } = sp;

const GEOM_3 = { x0: 38, y0: 8, width: 100, height: 190, count: 3, min: 0, range: 4 };
const GEOM_5 = { ...GEOM_3, count: 5 };

function vertices(subpath) {
  return subpath
    .split(' ')
    .filter(t => t[0] === 'M' || t[0] === 'L')
    .map(t => t.slice(1).split(',').map(Number));
}

test('bandPaths: one closed subpath, upper edge left to right then lower edge back', () => {
  assert.deepEqual(bandPaths([0, 1, 1], [2, 3, 4], GEOM_3), [
    'M38,103 L88,55.5 L138,8 L138,150.5 L88,150.5 L38,198 Z',
  ]);
});

test('bandPaths: a hole in the upper bound ends the run', () => {
  assert.deepEqual(bandPaths([0, 0, 0, 0, 0], [1, 1, null, 1, 1], GEOM_5), [
    'M38,150.5 L63,150.5 L63,198 L38,198 Z',
    'M113,150.5 L138,150.5 L138,198 L113,198 Z',
  ]);
});

test('bandPaths: a hole in the lower bound ends the run too', () => {
  assert.deepEqual(
    bandPaths([0, 0, null, 0, 0], [1, 1, 1, 1, 1], GEOM_5),
    bandPaths([0, 0, 0, 0, 0], [1, 1, null, 1, 1], GEOM_5),
  );
});

test('bandPaths: undefined, NaN and a short bound are holes exactly like null', () => {
  const reference = bandPaths([0, 0, 0, 0, 0], [1, 1, null, 1, 1], GEOM_5);
  for (const hole of [undefined, NaN, Infinity, '1']) {
    assert.deepEqual(bandPaths([0, 0, 0, 0, 0], [1, 1, hole, 1, 1], GEOM_5), reference);
  }
  assert.deepEqual(
    bandPaths([0, 0, 0], [1, 1, 1, 1, 1], GEOM_5),
    ['M38,150.5 L63,150.5 L88,150.5 L88,198 L63,198 L38,198 Z'],
    'indices past the shorter bound are holes',
  );
});

test('bandPaths: a single-index run is a degenerate vertical sliver at its own x', () => {
  assert.deepEqual(bandPaths([null, 1, null], [null, 2, null], GEOM_3), ['M88,103 L88,150.5 Z']);
});

test('bandPaths: an all-hole, empty or missing band draws nothing', () => {
  for (const [lo, hi] of [
    [[null, null], [null, null]],
    [[1, null], [null, 1]],
    [[], []],
    [null, null],
    [undefined, [1, 2]],
  ]) {
    assert.deepEqual(bandPaths(lo, hi, GEOM_3), []);
  }
});

test('bandPaths: a degenerate value axis draws nothing', () => {
  assert.deepEqual(bandPaths([0, 1], [1, 2], { ...GEOM_3, range: 0 }), []);
  assert.deepEqual(bandPaths([0, 1], [1, 2], { ...GEOM_3, range: NaN }), []);
});

test('bandPaths: every vertex shares axisPaths x-mapping and axisY, so the band tracks its line', () => {
  const geom = { x0: 38, y0: 8, width: 228, height: 173, count: 12, min: 0, range: 0.7 };
  const lo = [0.1, 0.12, 0.2, 0.05, 0.3, 0.31, 0.2, 0.22, 0.4, 0.1, 0.15, 0.6];
  const hi = lo.map(v => v + 0.07);
  const { stepX } = axisPaths(hi, geom);
  const n = lo.length;

  const [band] = bandPaths(lo, hi, geom);
  const points = vertices(band);
  assert.equal(points.length, 2 * n);
  for (let i = 0; i < n; i++) {
    assert.deepEqual(points[i], [stepX * i + geom.x0, axisY(hi[i], geom)], `upper vertex ${i}`);
    assert.deepEqual(
      points[2 * n - 1 - i],
      [stepX * i + geom.x0, axisY(lo[i], geom)],
      `lower vertex ${i}`,
    );
  }
});

test('bandPaths: does not mutate its inputs', () => {
  const lo = [0, null, 1];
  const hi = [1, 2, null];
  bandPaths(lo, hi, GEOM_3);
  assert.deepEqual(lo, [0, null, 1]);
  assert.deepEqual(hi, [1, 2, null]);
});
