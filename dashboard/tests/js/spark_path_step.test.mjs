// Behavioural tests for spark_path.js's axisStepPaths: LineChart's opt-in step
// series, drawn on the padded chart geometry axisPaths uses (the Quality tab's
// snapshot series, where a value holds until the next snapshot, task 8465). It
// sits apart from spark_path.test.mjs, which covers the rest of the module and
// its exhaustive public surface, as spark_path_band.test.mjs does for
// bandPaths. Run via `node --test`; the pytest shim
// dashboard/tests/test_graph_layout_js.py picks this file up by glob.
//
// spark_path.js resolves as CommonJS, so it is default-imported and
// destructured.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import sp from '../../src/dashboard/static/redux/spark_path.js';

const { axisPaths, axisStepPaths } = sp;

// charts.jsx's LineChart geometry at w=150, height=220: divisible, so every x
// and y below is exact (see spark_path.test.mjs's GEOMETRY NOTE).
const GEOM_5 = { x0: 38, y0: 8, width: 100, height: 190, count: 5, min: 0, range: 4 };
const GEOM_3 = { ...GEOM_5, count: 3 };

function vertexXs(path) {
  return path
    .split(' ')
    .filter(token => token[0] === 'M' || token[0] === 'L')
    .map(token => Number(token.slice(1).split(',')[0]));
}

test('axisStepPaths: horizontal then vertical edges, no diagonal', () => {
  const { line, area } = axisStepPaths([0, 1, 2, 3, 4], GEOM_5);

  assert.equal(line, 'M38,198 L63,198 L63,150.5 L88,150.5 L88,103 L113,103 L113,55.5 L138,55.5 L138,8');
  assert.equal(area, `${line} L138,198 L38,198 Z`);
});

test('axisStepPaths: the same column x’s and stepX as axisPaths', () => {
  const values = [3, 1, 4, 1, 2];
  const step = axisStepPaths(values, GEOM_5);
  const straight = axisPaths(values, GEOM_5);

  assert.equal(step.stepX, straight.stepX);
  assert.deepEqual([...new Set(vertexXs(step.line))], vertexXs(straight.line));
});

test('axisStepPaths: a hole splits the run, never holding a value across it', () => {
  const { line, area } = axisStepPaths([0, 1, null, 3, 4], GEOM_5);

  assert.equal(line, 'M38,198 L63,198 L63,150.5 M113,55.5 L138,55.5 L138,8');
  assert.equal(area, 'M38,198 L63,198 L63,150.5 L63,198 L38,198 Z M113,55.5 L138,55.5 L138,8 L138,198 L113,198 Z');
  assert.ok(!vertexXs(line).includes(88), 'nothing is drawn at the hole’s column');
});

test('axisStepPaths: undefined and NaN holes behave exactly like null', () => {
  const reference = axisStepPaths([0, 1, null, 3, 4], GEOM_5);
  for (const hole of [undefined, NaN]) {
    assert.deepEqual(axisStepPaths([0, 1, hole, 3, 4], GEOM_5), reference);
  }
});

test('axisStepPaths: a run of one is a zero-length segment at its own x, with no area', () => {
  const { line, area } = axisStepPaths([null, 1, null], GEOM_3);

  assert.equal(line, 'M88,150.5 L88,150.5');
  assert.equal(area, '');
});

test('axisStepPaths: an all-hole or empty series yields empty strings', () => {
  for (const values of [[null, null, null], [], null, undefined]) {
    const { line, area, stepX } = axisStepPaths(values, GEOM_3);
    assert.equal(line, '');
    assert.equal(area, '');
    assert.equal(stepX, 50);
  }
});
