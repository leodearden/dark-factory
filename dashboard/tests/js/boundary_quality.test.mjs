// Boundary family: the Quality tab, the client half of
// plans/quality-dashboard-prd.md sketch rows 3, 4, 5, 9, 10 and 12, plus the
// module drill-down.
//
// Each test lands a body the REAL /quality route served (test_boundary_js.py
// builds them and holds each row's server half) through data.js's real
// requestOnDemand under the canonical query qualityQuery builds, then reads it
// the way the Quality tab does: onDemandDatum per field, the receipt's WINDOW
// echo, datumView for a tile, and quality_view.js's decisions.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { applyOnDemand, loadClient, payload } from './_boundary_client.mjs';

const RECEIVED_AT = Date.parse('2026-11-10T12:00:01Z');
const PROJECT = 'dark_factory';
const WINDOW = '100';
const FIELDS = ['QUALITY', 'QUALITY_MARKERS', 'QUALITY_SNAPSHOTS'];
// The module _boundary_payloads.py::QUALITY_MODULE names; the served body does
// not echo it, so the drill-down's query restates it.
const MODULE = 'dashboard';

async function landQuality(name, { module = null } = {}) {
  const client = loadClient();
  const { qualityQuery, onDemandDatum } = client.loader;
  const query = qualityQuery({ project: PROJECT, window: WINDOW, module });
  const body = payload(name);
  const outcome = await applyOnDemand(client, 'quality', query, body, { receivedAt: RECEIVED_AT });
  assert.equal(outcome, 'applied', name);
  return {
    client,
    query,
    body,
    datum: field => onDemandDatum('quality', query, outcome, field),
    view: client.window.DF_QUALITY_VIEW,
    datumView: client.window.DF_DATUM.datumView,
  };
}

function headlineTile(landed, metric) {
  return landed.datumView(metric.headline, {
    now: RECEIVED_AT,
    format: value => landed.view.headlineText(value, metric.unit),
  });
}

function metricArrays(metric) {
  const arrays = [metric.points, ...Object.values(metric.rolling)];
  return metric.ewma === null ? arrays : [...arrays, metric.ewma.line, metric.ewma.ucl];
}

test('quality #3: a fresh series lands all three datums under its query, echoes its landing window, and keeps every landing on the axis', async () => {
  const landed = await landQuality('quality_fresh');
  const { client, query, body, datum } = landed;
  const row = client.loader.ON_DEMAND_KEYS.quality;

  for (const field of FIELDS) {
    assert.deepEqual(client.window.DF_DATA[row.key(query, field)].value, body[field].value, field);
    assert.equal(datum(field).state, body[field].state, field);
  }
  const receipt = client.loader.onDemandReceipt('quality', query);
  assert.deepEqual(client.window.DF_WINDOW_CHIP.receiptWindow(receipt), { requested: '100', served: '100', landings: 100 });
  const { axis } = datum('QUALITY').value;
  assert.equal(axis.length, 100);
  assert.ok(axis.some(point => point.git_first_parent === false), 'a landing off the first parent stays on the axis');
  assert.ok(axis.some(point => point.late_claim === true), 'a late claim stays on the axis');
});

test('quality #4: a source that has not started renders a hole naming its epoch and charts holes, never zero', async () => {
  const landed = await landQuality('quality_before_epoch');
  const blocking = landed.datum('QUALITY').value.metrics.reviewer_blocking;

  const tile = headlineTile(landed, blocking);
  assert.equal(tile.isHole, true);
  assert.match(tile.title, /2026-10-21/);
  const { series, band } = landed.view.chartSeries(blocking);
  for (const entry of series) assert.ok(entry.values.every(value => value === null), entry.key);
  assert.ok([...band.lo, ...band.hi].every(value => value === null), 'the band is unmeasured, not zero');
  assert.ok(landed.view.ruleCaption(blocking.rule).includes('measured since 2026-10-21'));
});

test('quality #5: a small sample renders its headline as k of n with its band', async () => {
  const landed = await landQuality('quality_small_n');
  const rejected = landed.datum('QUALITY').value.metrics.rejected;
  const { value } = rejected.headline;

  assert.equal(value.form, 'k_of_n');
  const tile = headlineTile(landed, rejected);
  assert.equal(tile.isHole, false);
  assert.equal(tile.text, landed.view.headlineText(value, rejected.unit));
  assert.ok(tile.text.startsWith(`${value.k} of ${value.n} [`), tile.text);
});

function alarmReading(landed) {
  const quality = landed.datum('QUALITY').value;
  const rejected = quality.metrics.rejected;
  const { markers } = landed.datum('QUALITY_MARKERS').value;
  const [alarm] = rejected.alarms;
  const verdict = landed.view.alarmVerdict(alarm, landed.view.indexMarkers(markers));
  const marks = landed.view.qualityMarks({
    axis: quality.axis, markers, alarms: rejected.alarms, beforeAfter: quality.before_after,
    metricId: 'rejected', kind: rejected.kind,
  });
  return { alarm, markers, verdict, marks };
}

test('quality #9: an alarm beside a served marker reads as explained by it, and the same shift far from one as unexplained', async () => {
  const near = alarmReading(await landQuality('quality_alarm_explained'));
  const marker = near.markers.find(served => served.id === near.alarm.explained_by);
  assert.ok(marker, `the alarm names a served marker: ${near.alarm.explained_by}`);
  assert.equal(near.verdict, `explained by ${marker.label}`);
  assert.ok(near.marks.some(mark => mark.kind === 'alarm' && mark.label === near.verdict), 'the chart draws the verdict');

  const far = alarmReading(await landQuality('quality_alarm_unexplained'));
  assert.equal(far.alarm.onset_ordinal, near.alarm.onset_ordinal, 'one shift in both scenarios');
  assert.equal(far.verdict, `unexplained change at landing #${far.alarm.onset_ordinal}`);
  assert.ok(far.marks.some(mark => mark.kind === 'alarm' && mark.label === far.verdict), 'the chart draws the verdict');
});

test('quality #10: the same window read twice charts byte-equal series and bands', async () => {
  const charted = [];
  for (const name of ['quality_fresh', 'quality_fresh_again']) {
    const landed = await landQuality(name);
    const { metrics } = landed.datum('QUALITY').value;
    charted.push(JSON.stringify(Object.entries(metrics).map(([id, metric]) => [id, landed.view.chartSeries(metric)])));
  }

  assert.equal(charted[0], charted[1]);
  const [, rejected] = JSON.parse(charted[0]).find(([id]) => id === 'rejected');
  assert.ok(rejected.band.lo.some(value => value !== null), 'a measured band, not an empty one');
});

test('quality #12: a store behind runs.db lands stale naming its unsampled landings, then fresh after the next tick', async () => {
  const behind = await landQuality('quality_main_ahead');
  const stale = behind.datum('QUALITY');
  assert.equal(stale.state, 'stale');
  assert.match(stale.reason, /^\d+ landings? not yet sampled$/);
  const tile = headlineTile(behind, stale.value.metrics.rejected);
  assert.equal(tile.title, stale.reason, 'every headline of a stale series says why');

  const cleared = await landQuality('quality_main_ahead_cleared');
  assert.equal(cleared.datum('QUALITY').state, 'fresh');
});

test('quality module: a drill-down lands under its own query and serves the module subset of the axis', async () => {
  const landed = await landQuality('quality_module', { module: MODULE });
  const whole = landed.client.loader.qualityQuery({ project: PROJECT, window: WINDOW, module: null });
  const { axis, metrics } = landed.datum('QUALITY').value;
  const ordinals = axis.map(point => point.ordinal);

  assert.notEqual(landed.query, whole);
  assert.ok(ordinals.length > 1);
  assert.ok(ordinals.every((ordinal, i) => i === 0 || ordinal > ordinals[i - 1]), 'strictly increasing');
  assert.ok(ordinals.at(-1) - ordinals[0] + 1 > ordinals.length, 'a gapped subset, not a window');
  for (const [id, metric] of Object.entries(metrics)) {
    for (const array of metricArrays(metric)) assert.equal(array.length, ordinals.length, id);
  }
});
