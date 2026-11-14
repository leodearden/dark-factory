// Module-contract tests for quality_view.js — the pure decisions behind the
// Quality tab (task 8461): which served metrics each section draws, the text of
// a headline, a counting rule, an alarm and a before/after comparison, the
// arrays a chart draws (served verbatim, never smoothed), and where each served
// marker and alarm sits on a metric's chart. tab_quality.jsx only wires these.
//
// THE METRIC IDS ARE SPELLED HERE AS LITERALS, because the layout's choice of
// them is the decision under test: they mirror
// dashboard/src/dashboard/data/quality_series.py::CATALOGUE.
//
// quality_view.js destructures window.DF_CHART_MARKS at module scope with no
// fallback, so the loader puts chart_marks.js on the window shim first.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  qualityMarker, qualityMarkers, qualityMetric, qualityPayload, qualitySnapshotPoint, qualitySnapshots,
} from './wire_fixtures.mjs';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['chart_marks.js', 'quality_view.js'].map(name => REDUX + name);

function loadQualityView() {
  const win = {};
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [, viewApi] = LOAD_CHAIN.map(specifier => require(specifier));
  return { api: viewApi, window: win };
}

const { api: view, window: loadedWindow } = loadQualityView();
const {
  QUALITY_LAYOUT,
  metricTitle,
  headlineText,
  ruleCaption,
  chartSeries,
  chartAxis,
  indexMarkers,
  alarmVerdict,
  beforeAfterText,
  costModelReadings,
  qualityMarks,
  snapshotAxis,
  snapshotCharts,
  snapshotMarks,
  snapshotTable,
  instrumentText,
  snapshotLabel,
  snapshotHeading,
  snapshotDiffText,
} = view;
const { markHover } = loadedWindow.DF_CHART_MARKS;

const CATALOGUE_IDS = [
  'rejected', 'reviewer_blocking', 'flakes_confirmed', 'flakes_real_red', 'review_rounds_ge2',
  'turns_capped', 'charter_blocked', 'rejections_per_landing', 'reverifies_per_landing',
  'implementer_invocations', 'turns', 'execute_iterations', 'reverts', 'all_role_usd',
];

test('quality_view.js assigns window.DF_QUALITY_VIEW to its module surface', () => {
  assert.equal(loadedWindow.DF_QUALITY_VIEW, view);
  assert.deepEqual(Object.keys(view).sort(), [
    'QUALITY_LAYOUT', 'alarmVerdict', 'beforeAfterText', 'chartAxis', 'chartSeries', 'costModelReadings',
    'headlineText', 'indexMarkers', 'instrumentText', 'metricTitle', 'qualityMarks', 'ruleCaption',
    'snapshotAxis', 'snapshotCharts', 'snapshotDiffText', 'snapshotHeading', 'snapshotLabel', 'snapshotMarks',
    'snapshotTable',
  ]);
});

// ── The layout ────────────────────────────────────────────────────────────

test('the outcome section draws the five outcome metrics, each with its beside metric', () => {
  assert.deepEqual(QUALITY_LAYOUT.outcomes.map(entry => [entry.metric, entry.beside]), [
    ['rejected', 'rejections_per_landing'],
    ['reviewer_blocking', null],
    ['flakes_confirmed', 'flakes_real_red'],
    ['reverts', null],
    ['reverifies_per_landing', null],
  ]);
});

test('the change-cost section draws the cost, effort and friction metrics', () => {
  assert.deepEqual(QUALITY_LAYOUT.changeCost.map(entry => [entry.metric, entry.beside]), [
    ['all_role_usd', null],
    ['implementer_invocations', null],
    ['turns', 'turns_capped'],
    ['review_rounds_ge2', null],
    ['charter_blocked', null],
  ]);
});

test('every laid-out metric is a served catalogue metric and every entry is titled', () => {
  for (const entry of [...QUALITY_LAYOUT.outcomes, ...QUALITY_LAYOUT.changeCost]) {
    assert.ok(CATALOGUE_IDS.includes(entry.metric), entry.metric);
    assert.ok(entry.beside === null || CATALOGUE_IDS.includes(entry.beside), entry.beside);
    assert.equal(typeof entry.title, 'string');
    assert.notEqual(entry.title, '');
  }
});

test('a laid-out metric is titled by its layout entry', () => {
  const metric = qualityMetric({ points: [0, 1] });

  assert.equal(metricTitle('rejected', metric), 'Verify rejections');
  assert.equal(metricTitle('turns', qualityMetric({ kind: 'count', points: [1] })), 'Turns per landing');
});

test('a metric drawn only beside another is titled by what its served rule counts', () => {
  for (const id of ['rejections_per_landing', 'flakes_real_red', 'turns_capped']) {
    const metric = qualityMetric({ points: [0, 1], source: `${id}_source` });

    assert.equal(metricTitle(id, metric), metric.rule.counts, id);
  }
});

// ── Headlines ─────────────────────────────────────────────────────────────

test('a rate headline is its served share as a percentage with its band', () => {
  const headline = { form: 'rate', n: 100, k: 12, value: 0.12, lo: 0.07, hi: 0.2 };

  assert.equal(headlineText(headline, 'share'), '12.0% [7.0%–20.0%]');
});

test('a headline renders the served fields only, recomputing nothing from k and n', () => {
  const headline = { form: 'rate', n: 3, k: 1, value: 0.12, lo: 0.07, hi: 0.2 };

  assert.equal(headlineText(headline, 'share'), '12.0% [7.0%–20.0%]');
});

test('a k_of_n headline says k of n with its band', () => {
  const headline = { form: 'k_of_n', n: 42, k: 5, value: 5 / 42, lo: 0.05, hi: 0.25 };

  assert.equal(headlineText(headline, 'share'), '5 of 42 [5.0%–25.0%]');
});

test('a mean headline is its served centre and band in its unit', () => {
  const { headline } = qualityMetric({ kind: 'count', points: [1, 2, 3] });

  assert.equal(headlineText(headline.value, 'invocations'), '2.00 [1.90–2.10] invocations');
});

test('a median headline in usd is dollars', () => {
  const headline = { form: 'median', n: 80, k: null, value: 3.5, lo: 2.75, hi: 4.1 };

  assert.equal(headlineText(headline, 'usd'), '$3.50 [$2.75–$4.10]');
});

test('a headline of a form the view does not know is refused, naming it', () => {
  assert.throws(() => headlineText({ form: 'mode', n: 1, k: null, value: 1, lo: 1, hi: 1 }, 'turns'), /mode/);
});

// ── Counting rules ────────────────────────────────────────────────────────

const RULE = {
  counts: 'landings whose verify unit was blocked by post-merge verify',
  excludes: 'landings before the first categorised merge_blocked',
  population: 'landings',
  window_landings: 100,
  measured_since: '2026-10-01T06:30:00+00:00',
  source: 'rejections',
};

test('a rule caption renders each structured field of the counting rule', () => {
  assert.equal(
    ruleCaption(RULE),
    'counts: landings whose verify unit was blocked by post-merge verify · population: landings · '
      + 'headline: last 100 landings · measured since 2026-10-01 · '
      + 'excludes: landings before the first categorised merge_blocked',
  );
});

test('a rule whose source has not started says no producer has written here', () => {
  const caption = ruleCaption({ ...RULE, measured_since: null });

  assert.ok(caption.includes('no producer has written here (rejections)'), caption);
  assert.ok(!caption.includes('measured since'), caption);
});

// ── Chart series: served arrays, verbatim ─────────────────────────────────

function rateMetric() {
  const points = [0, 1, 0, 0, 1];
  return qualityMetric({
    points,
    rolling: {
      center: [null, 0.5, 0.33, 0.25, 0.4],
      lo: [null, 0.1, 0.06, 0.05, 0.12],
      hi: [null, 0.9, 0.79, 0.7, 0.77],
      n: [1, 2, 3, 4, 5],
    },
    ewma: { line: [0.2, 0.24, 0.23, 0.22, 0.26], ucl: [0.3, 0.32, 0.33, 0.34, 0.34] },
  });
}

test('a rate charts its rolling centre, EWMA and dashed UCL with its Wilson band, as served', () => {
  const metric = rateMetric();
  const { series, band } = chartSeries(metric);

  assert.deepEqual(series.map(entry => [entry.key, Boolean(entry.dash)]), [
    ['center', false], ['ewma', false], ['ucl', true],
  ]);
  assert.equal(series[0].values, metric.rolling.center);
  assert.equal(series[1].values, metric.ewma.line);
  assert.equal(series[2].values, metric.ewma.ucl);
  assert.equal(band.lo, metric.rolling.lo);
  assert.equal(band.hi, metric.rolling.hi);
});

test('a count or a cost charts its rolling centre alone, with no band', () => {
  for (const kind of ['count', 'cost']) {
    const metric = qualityMetric({ kind, points: [1, 2, 3] });
    const { series, band } = chartSeries(metric);

    assert.deepEqual(series.map(entry => entry.key), ['center'], kind);
    assert.equal(series[0].values, metric.rolling.center, kind);
    assert.equal(band, null, kind);
  }
});

// ── Chart axes ────────────────────────────────────────────────────────────

test('a rate axis is scaled to its data and ticks in percent, as its headline reads', () => {
  const { formatY, minAxisMax } = chartAxis(rateMetric());

  assert.equal(minAxisMax, 0);
  assert.equal(formatY(0.0183), '1.8%');
  assert.equal(formatY(0), '0.0%');
});

test('a count axis ticks in fixed decimals, so a float-noise tick reads as its value', () => {
  const { formatY, minAxisMax } = chartAxis(qualityMetric({ kind: 'count', points: [1.2, 1.6] }));

  assert.equal(minAxisMax, 0);
  assert.equal(formatY(1.2000000000000002), '1.20');
});

test('a cost axis in usd ticks in dollars', () => {
  const { formatY } = chartAxis(qualityMetric({ kind: 'cost', points: [3.5] }));

  assert.equal(formatY(3.5), '$3.50');
});

test('an axis for a metric kind the view does not know is refused, naming it', () => {
  assert.throws(() => chartAxis(qualityMetric({ kind: 'ratio', points: [1] })), /"ratio"/);
});

// ── Alarms ────────────────────────────────────────────────────────────────

const FLIP = qualityMarker({ id: 'routing:reviewer_comprehensive:1', ordinal: 13, label: 'reviewer effort high → xhigh', ref: 'task 7972' });

test('an alarm near a served marker is explained by that marker’s label', () => {
  const alarm = { ordinal: 20, onset_ordinal: 14, direction: 'up', explained_by: FLIP.id };

  assert.equal(alarmVerdict(alarm, indexMarkers([FLIP])), 'explained by reviewer effort high → xhigh');
});

test('an alarm explained by a marker that is not served names the marker id', () => {
  const alarm = { ordinal: 20, onset_ordinal: 14, direction: 'up', explained_by: 'routing:gone:7' };

  assert.equal(alarmVerdict(alarm, indexMarkers([FLIP])), 'explained by routing:gone:7');
  assert.equal(alarmVerdict(alarm, indexMarkers(null)), 'explained by routing:gone:7');
});

test('an alarm no marker explains is an unexplained change at its onset landing', () => {
  const alarm = { ordinal: 20, onset_ordinal: 14, direction: 'down', explained_by: null };

  assert.equal(alarmVerdict(alarm, indexMarkers([FLIP])), 'unexplained change at landing #14');
});

// ── Before/after a marker ─────────────────────────────────────────────────

function comparison(overrides) {
  return {
    marker_id: FLIP.id, metric: 'rejected', n_before: 210, n_after: 140, effect: 0.05, lo: 0.012,
    hi: 0.088, state: 'measured', eta_days: null, min_detectable: null, ...overrides,
  };
}

test('a measured rate comparison is the difference in points with its band and n', () => {
  assert.equal(beforeAfterText(comparison({}), 'rate'), '+5.0 pp [+1.2 to +8.8 pp] · n 210 → 140');
  assert.equal(
    beforeAfterText(comparison({ effect: -0.031, lo: -0.06, hi: -0.002 }), 'rate'),
    '-3.1 pp [-6.0 to -0.2 pp] · n 210 → 140',
  );
});

test('a measured cost comparison is the ratio with its band and n', () => {
  const entry = comparison({ metric: 'all_role_usd', n_before: 250, n_after: 120, effect: 0.82, lo: 0.7, hi: 0.95 });

  assert.equal(beforeAfterText(entry, 'cost'), '×0.82 [×0.70 to ×0.95] · n 250 → 120');
});

test('a comparison not yet measurable shows n, the ETA and the minimum detectable effect', () => {
  const entry = comparison({ n_after: 40, effect: null, lo: null, hi: null, state: 'not_yet', eta_days: 6.2, min_detectable: 0.043 });

  assert.equal(beforeAfterText(entry, 'rate'), 'n 210 → 40 · ETA 7 d · MDE 4.3 pp');
});

test('a not-yet comparison omits what the server did not measure', () => {
  const entry = comparison({
    metric: 'all_role_usd', n_before: 250, n_after: 40, effect: null, lo: null, hi: null,
    state: 'not_yet', eta_days: null, min_detectable: null,
  });

  assert.equal(beforeAfterText(entry, 'cost'), 'n 250 → 40');
});

test('a measured comparison of a kind the view has no text for is refused, naming the kind, marker and metric', () => {
  const entry = comparison({ metric: 'turns', effect: 1.2, lo: 1.1, hi: 1.3 });

  assert.throws(
    () => beforeAfterText(entry, 'count'),
    error => error instanceof TypeError
      && error.message.includes('"count"')
      && error.message.includes(FLIP.id)
      && error.message.includes('turns'),
  );
});

// ── The cost model in force ───────────────────────────────────────────────

test('a cost model in force reads its reference cost in dollars and its signed median residual', () => {
  const costModel = { a: 0.4, b: 0.7, reference_cost_100: 1.837, residual_median: -0.1204, state: 'frozen' };

  assert.deepEqual(costModelReadings(costModel).map(({ key, label, text, hint }) => [key, label, text, hint]), [
    ['reference_cost_100', 'Reference cost at 100 lines', '$1.84', 'frozen baseline'],
    ['residual_median', 'Median log residual', '-0.120', 'frozen baseline'],
  ]);
  assert.equal(costModelReadings({ ...costModel, residual_median: 0.05, state: 'provisional' })[1].text, '+0.050');
  assert.equal(costModelReadings({ ...costModel, state: 'provisional' })[0].hint, 'provisional baseline');
});

test('with no cost model in force both readings are unmeasured and unhinted', () => {
  const none = qualityPayload({ axis: [], metrics: {} }).cost_model;

  assert.deepEqual(costModelReadings(none).map(({ text, hint }) => [text, hint]), [[null, null], [null, null]]);
});

// ── Marks on a metric's chart ─────────────────────────────────────────────

const GAPPED_AXIS = qualityPayload({
  axis: [{ ordinal: 10 }, { ordinal: 11 }, { ordinal: 13 }, { ordinal: 14 }, { ordinal: 20 }],
  metrics: {},
}).axis;

const BUDGET = qualityMarker({ id: 'budget:implementer:2', ordinal: 11, kind: 'budget', label: 'implementer budget 12 → 16', series: ['rejected'] });
const ELSEWHERE = qualityMarker({ id: 'model:merger:3', ordinal: 14, kind: 'model', series: ['flakes_confirmed'] });
const ARTEFACT = qualityMarker({
  id: 'curated:disk-full', ordinal: 12, untilOrdinal: 14, kind: 'artefact', label: 'disk full',
  series: ['rejected'], source: { type: 'curated', file: 'regime-markers.yaml' },
});
const BEFORE_WINDOW = qualityMarker({ id: 'cli:4', ordinal: 3, kind: 'cli' });
const OFF_AXIS = qualityMarker({ id: 'deploy:5', ordinal: null, kind: 'deploy' });
const ALARM = { ordinal: 20, onset_ordinal: 14, direction: 'up', explained_by: FLIP.id };

function marksFor(metricId) {
  return qualityMarks({
    axis: GAPPED_AXIS,
    markers: qualityMarkers([FLIP, BUDGET, ELSEWHERE, ARTEFACT, BEFORE_WINDOW, OFF_AXIS]).markers,
    alarms: [ALARM],
    beforeAfter: [comparison({}), comparison({ metric: 'flakes_confirmed', effect: 0.2 })],
    metricId,
    kind: 'rate',
  });
}

test('served markers sit at their landing on a gapped axis, alarms after them', () => {
  assert.deepEqual(marksFor('rejected'), [
    { key: FLIP.id, kind: 'effort', label: FLIP.label, fraction: 0.5, ref: 'task 7972', verdict: '+5.0 pp [+1.2 to +8.8 pp] · n 210 → 140' },
    { key: BUDGET.id, kind: 'budget', label: BUDGET.label, fraction: 0.25, ref: null, verdict: null },
    { key: ARTEFACT.id, kind: 'artefact', label: 'disk full', fraction: 0.5, endFraction: 0.75, ref: null, verdict: null },
    { key: 'alarm:20:up', kind: 'alarm', label: 'explained by reviewer effort high → xhigh', fraction: 1, ref: null, verdict: null },
  ]);
});

test('a marker tagged for other series is drawn on those series only', () => {
  const keys = marksFor('flakes_confirmed').map(mark => mark.key);

  assert.ok(keys.includes(ELSEWHERE.id));
  assert.ok(!keys.includes(BUDGET.id));
  assert.ok(!keys.includes(ARTEFACT.id));
});

test('a marker’s hover carries its ref and this metric’s before/after, and nothing else', () => {
  const [flip, budget] = marksFor('rejected');

  assert.equal(markHover(flip), 'reviewer effort high → xhigh\ntask 7972\n+5.0 pp [+1.2 to +8.8 pp] · n 210 → 140');
  assert.equal(markHover(budget), null);
  assert.equal(marksFor('flakes_confirmed')[0].verdict, '+20.0 pp [+1.2 to +8.8 pp] · n 210 → 140');
});

test('an unserved marker stream draws the alarms alone', () => {
  const marks = qualityMarks({ axis: GAPPED_AXIS, markers: null, alarms: [{ ...ALARM, explained_by: null }], beforeAfter: [], metricId: 'rejected', kind: 'rate' });

  assert.deepEqual(marks.map(mark => [mark.kind, mark.label]), [['alarm', 'unexplained change at landing #14']]);
});

test('an instrument marker is not drawn on a landing-axis chart: it annotates the snapshot series', () => {
  const instrument = qualityMarker({
    id: 'instrument:abc', ordinal: 13, kind: 'instrument', ref: 'abc', source: { type: 'instrument', point: 'abc' },
  });
  const marks = qualityMarks({
    axis: GAPPED_AXIS, markers: [FLIP, instrument], alarms: [], beforeAfter: [], metricId: 'rejected', kind: 'rate',
  });

  assert.deepEqual(marks.map(mark => mark.key), [FLIP.id]);
});

// ── The snapshot series ───────────────────────────────────────────────────
// One column per served point, in served order; every figure is the point's
// own served count, and nothing is averaged, totalled, ranked or sorted.

const SHA_1 = '1'.repeat(40);
const SHA_2 = '2'.repeat(40);
const SHA_3 = '3'.repeat(40);

const SCHEMA_2_PARAMS = { complexipy_version: '6.2.0', h14_soft_ceiling_lines: 1500, h14_alarm_lines: 2000 };
const BASELINE = qualitySnapshotPoint({
  id: 'review-all-dark_factory-20261008', baseline: true, asOfSha: SHA_1, committedAt: '2026-10-08T18:30:00+00:00',
  params: SCHEMA_2_PARAMS, unknownParams: ['instrument', 'python_version'],
  h14: { atSoftCeiling: 4, atAlarm: 2 }, cycles: { cycles: 5, hidden: 3, typing: null },
  deferredImports: 30, cycleClosingImports: null, privatePatchTargets: 12,
});
const NIGHTLY_2 = qualitySnapshotPoint({
  id: 'nightly-2', asOfSha: SHA_2, committedAt: '2026-11-01T02:00:00+00:00',
  h14: { atSoftCeiling: 6, atAlarm: 1 }, cycles: { cycles: 2, hidden: 4, typing: 1 },
  deferredImports: 18, cycleClosingImports: 7, privatePatchTargets: 3,
  diff: { previous_id: BASELINE.id, state: 'ready', reason: null },
});
const NIGHTLY_3 = qualitySnapshotPoint({
  id: 'nightly-3', asOfSha: SHA_3, committedAt: '2026-11-02T02:00:00+00:00',
  h14: { atSoftCeiling: 5, atAlarm: 1 }, cycles: { cycles: 3, hidden: 0, typing: 2 },
  deferredImports: 21, cycleClosingImports: 2, privatePatchTargets: 8,
  diff: { previous_id: NIGHTLY_2.id, state: 'pending', reason: null },
});
const POINTS = qualitySnapshots([BASELINE, NIGHTLY_2, NIGHTLY_3]).points;

test('the snapshot axis is one column per served point, labelled by its commit date in served order', () => {
  assert.deepEqual(snapshotAxis(POINTS), ['2026-10-08', '2026-11-01', '2026-11-02']);
  assert.deepEqual(snapshotAxis([NIGHTLY_3, BASELINE]), ['2026-11-02', '2026-10-08'], 'never re-sorted');
  assert.deepEqual(snapshotAxis([]), []);
});

test('the snapshot charts are fixed groups of the served domain counts, each a step series', () => {
  const charts = snapshotCharts(POINTS);

  assert.deepEqual(
    charts.map(chart => [chart.group, chart.series.map(series => series.key)]),
    [
      ['h14', ['at_soft_ceiling', 'at_alarm']],
      ['cycles', ['cycles', 'hidden_cycles', 'typing_cycles']],
      ['imports', ['deferred_imports', 'cycle_closing_imports']],
      ['private_patch_targets', ['private_patch_targets']],
    ],
  );
  for (const chart of charts) {
    assert.ok(typeof chart.title === 'string' && chart.title !== '', `${chart.group} is titled`);
    for (const series of chart.series) {
      assert.equal(series.step, true, `${series.key} is a step series`);
      assert.ok(typeof series.label === 'string' && series.label !== '', `${series.key} is labelled`);
    }
  }
});

test('each snapshot series holds the points’ served values verbatim, unknown as null', () => {
  const values = Object.fromEntries(
    snapshotCharts(POINTS).flatMap(chart => chart.series).map(series => [series.key, series.values]),
  );

  assert.deepEqual(values, {
    at_soft_ceiling: [4, 6, 5],
    at_alarm: [2, 1, 1],
    cycles: [5, 2, 3],
    hidden_cycles: [3, 4, 0],
    typing_cycles: [null, 1, 2],
    deferred_imports: [30, 18, 21],
    cycle_closing_imports: [null, 7, 2],
    private_patch_targets: [12, 3, 8],
  });
});

test('a baseline mark sits at each baseline point’s column, an instrument marker at its point’s', () => {
  const instrument = qualityMarker({
    id: `instrument:${NIGHTLY_2.id}`, ordinal: null, kind: 'instrument', label: 'instrument params changed: instrument (unknown)',
    ref: SHA_2, source: { type: 'instrument', point: NIGHTLY_2.id },
  });

  assert.deepEqual(snapshotMarks(POINTS, [instrument]), [
    { key: `baseline:${BASELINE.id}`, kind: 'baseline', label: '/review-all baseline 2026-10-08', fraction: 0, ref: SHA_1, verdict: null },
    { key: instrument.id, kind: 'instrument', label: instrument.label, fraction: 0.5, ref: SHA_2, verdict: null },
  ]);
});

test('an instrument marker sits at its own point’s column, past a baseline at the same commit, and one naming no served point is dropped', () => {
  const nightlyAtBaseline = qualitySnapshotPoint({ id: 'nightly-1', asOfSha: SHA_1, committedAt: BASELINE.committed_at });
  const points = [BASELINE, nightlyAtBaseline, NIGHTLY_2];
  const atNightly = qualityMarker({
    id: 'instrument:nightly-1', ordinal: null, kind: 'instrument', ref: SHA_1, source: { type: 'instrument', point: 'nightly-1' },
  });
  const unserved = qualityMarker({
    id: 'instrument:gone', ordinal: null, kind: 'instrument', ref: SHA_2, source: { type: 'instrument', point: 'gone' },
  });

  const instrumentMarks = snapshotMarks(points, [atNightly, unserved]).filter(mark => mark.kind === 'instrument');

  assert.deepEqual(instrumentMarks.map(mark => [mark.key, mark.fraction, mark.ref]), [[atNightly.id, 0.5, SHA_1]]);
});

test('only instrument markers are drawn on the snapshot charts', () => {
  const effort = qualityMarker({ id: 'routing:reviewer:1', ordinal: 4, kind: 'effort', ref: SHA_2 });

  assert.deepEqual(snapshotMarks([NIGHTLY_2, NIGHTLY_3], [effort]), []);
  assert.deepEqual(snapshotMarks([NIGHTLY_2, NIGHTLY_3], null), []);
});

test('a series of one point places no mark: there is no column fraction to place it at', () => {
  assert.deepEqual(snapshotMarks([BASELINE], []), []);
});

test('a point’s member table is the instrument’s cell texts, row for row in served order', () => {
  const table = snapshotTable(NIGHTLY_2);

  assert.deepEqual(table.headers, ['member', 'pseudo', ...NIGHTLY_2.members.columns]);
  assert.deepEqual(table.rows, NIGHTLY_2.members.rows.map(row => [row.member, String(row.pseudo), ...row.cells]));
  assert.deepEqual(table.rows.map(row => row[0]), ['dashboard', 'scripts']);
  assert.deepEqual(table.rows.map(row => row[1]), ['false', 'true'], 'pseudo as the instrument’s markdown prints it');
});

test('a point whose schema does not record the instrument reads instrument unknown', () => {
  assert.equal(instrumentText(BASELINE), 'instrument unknown');
});

test('a point that records its instrument names each blob by its short sha', () => {
  assert.equal(instrumentText(NIGHTLY_2), 'instrument quality_metrics_snapshot aaaaaaaaaa · source_measures bbbbbbbbbb');
});

test('a point is labelled by its commit date, its home and its short sha, and headed with its completeness', () => {
  assert.equal(snapshotLabel(BASELINE), '2026-10-08 · baseline · 1111111111');
  assert.equal(snapshotLabel(NIGHTLY_2), '2026-11-01 · nightly · 2222222222');
  assert.equal(snapshotHeading(BASELINE), '2026-10-08 · baseline · 1111111111 · complete · instrument unknown');
  assert.equal(
    snapshotHeading(qualitySnapshotPoint({ asOfSha: SHA_3, committedAt: NIGHTLY_3.committed_at, complete: false })),
    `2026-11-02 · nightly · 3333333333 · incomplete · ${instrumentText(NIGHTLY_3)}`,
  );
});

test('a ready diff reads as its text verbatim, every other state as its reason', () => {
  const text = 'params differ: python_version 3.13.8 -> 3.13.9\n';
  assert.equal(snapshotDiffText({ point_id: 'n', previous_id: 'p', state: 'ready', text, reason: null }), text);
  for (const state of ['first', 'pending', 'failed', 'absent']) {
    const reason = `the ${state} reason`;
    assert.equal(snapshotDiffText({ point_id: 'n', previous_id: null, state, text: null, reason }), reason, state);
  }
});
