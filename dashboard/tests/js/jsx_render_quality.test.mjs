// The quality dashboard's JSX render seams (task 8461): charts.jsx's LineChart
// band and dashed series, the shared marker overlay (Markers, MarkedChart),
// escalation analytics drawn through that overlay, and the Quality tab. The
// real .jsx renders in a jsdom window (dom_harness.mjs); every expected value
// comes from calling the classic-script helpers (spark_path.js,
// chart_marks.js, quality_view.js, datum.js), never from source text.
//
// A separate file from jsx_render.test.mjs, which is past the 1,500-line
// ceiling; each file pays the Babel compile once, in its own process.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { allowAsyncAct, click, openWindow, settle, stubFetch } from './_render_helpers.mjs';
import { buildDashboardWindow, seedDashboardData } from './dom_harness.mjs';
import {
  escalationAnalytics,
  projectRow,
  qualityBody,
  qualityComparison,
  qualityMarker,
  qualityMarkers,
  qualityMetric,
  qualityModuleRow,
  qualityPayload,
  receiptsFor,
} from './wire_fixtures.mjs';

// Arrays built inside the window belong to its realm, so a helper's result is
// spread into a node array before a deepStrictEqual against the DOM.

// charts.jsx's LineChart before its ResizeObserver reports a width.
const CHART_WIDTH = 600;
const CHART_HEIGHT = 220;

function plotGeom(window, count, range, { width = CHART_WIDTH, height = CHART_HEIGHT } = {}) {
  const pad = window.DF_CHARTS.PLOT_PAD;
  return {
    x0: pad.l,
    y0: pad.t,
    width: Math.max(width - pad.l - pad.r, 50),
    height: height - pad.t - pad.b,
    count,
    min: 0,
    range,
  };
}

// LineChart's axis maximum over `values`, its least maximum `minAxisMax`.
function axisRange(window, values, minAxisMax = 1) {
  const { axisMax, plottableMax } = window.DF_SPARK_PATH;
  return axisMax(plottableMax(values, minAxisMax), 4, false);
}

function mountLineChart(harness, props) {
  return harness.mount(harness.window.DF_CHARTS.LineChart, props).container;
}

const bandDs = container => [...container.querySelectorAll('path.chart-band')].map(p => p.getAttribute('d'));

function pathWithD(container, d) {
  return [...container.querySelectorAll('path')].find(p => p.getAttribute('d') === d);
}

function yTickTexts(container) {
  return [...container.querySelectorAll('text[text-anchor="end"]')].map(t => t.textContent);
}

// ── LineChart: band and dash ──────────────────────────────────────────────

const LABELS_4 = ['a', 'b', 'c', 'd'];
const CENTRE = [0.2, 0.3, 0.25, 0.4];
const LO = [0.1, 0.2, 0.15, 0.3];
const HI = [0.3, 0.4, 0.35, 0.5];

test('charts.jsx::LineChart draws bandPaths(lo, hi, geom) as filled paths beneath the series', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const container = mountLineChart(harness, {
    series: [{ key: 'centre', values: CENTRE }],
    band: { lo: LO, hi: HI },
    labels: LABELS_4,
  });

  const geom = plotGeom(window, 4, axisRange(window, [...CENTRE, ...HI]));
  const expected = [...window.DF_SPARK_PATH.bandPaths(LO, HI, geom)];
  assert.equal(expected.length, 1);
  assert.deepEqual(bandDs(container), expected);

  const band = container.querySelector('path.chart-band');
  assert.notEqual(band.getAttribute('fill'), 'none', 'the band is filled');
  const line = pathWithD(container, window.DF_SPARK_PATH.axisPaths(CENTRE, geom).line);
  assert.ok(line, 'the centre line is drawn on the same geometry as its band');
  const paths = [...container.querySelectorAll('path')];
  assert.ok(paths.indexOf(band) < paths.indexOf(line), 'the band is drawn beneath the series');
});

test('charts.jsx::LineChart splits the band at a hole in hi', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const hi = [0.3, null, 0.35, 0.5];
  const container = mountLineChart(harness, {
    series: [{ key: 'centre', values: CENTRE }],
    band: { lo: LO, hi },
    labels: LABELS_4,
  });

  const geom = plotGeom(window, 4, axisRange(window, [...CENTRE, ...hi]));
  const expected = [...window.DF_SPARK_PATH.bandPaths(LO, hi, geom)];
  assert.equal(expected.length, 2);
  assert.deepEqual(bandDs(container), expected);
});

test('charts.jsx::LineChart folds the band hi into the axis maximum', t => {
  const harness = openWindow(t);
  const series = [{ key: 'count', values: [1, 2, 3] }];
  const labels = ['a', 'b', 'c'];

  const without = mountLineChart(harness, { series, labels });
  const withBand = mountLineChart(harness, { series, labels, band: { lo: [0, 1, 2], hi: [1, 9, 4] } });

  assert.equal(yTickTexts(without).at(-1), '3');
  assert.equal(yTickTexts(withBand).at(-1), '9', 'a band above every series value raises the top tick');
});

test('charts.jsx::LineChart scales small fractions to their data when minAxisMax is lowered', t => {
  const harness = openWindow(t);
  const series = [{ key: 'rate', values: [0.02, 0.04] }];
  const labels = ['a', 'b'];

  assert.equal(yTickTexts(mountLineChart(harness, { series, labels })).at(-1), '1', 'the default least maximum is 1');
  assert.equal(yTickTexts(mountLineChart(harness, { series, labels, minAxisMax: 0 })).at(-1), '0.04');
});

test('charts.jsx::LineChart draws no band when none is passed, or when it is all holes', t => {
  const harness = openWindow(t);
  const series = [{ key: 'count', values: [1, 2, 3] }];
  const labels = ['a', 'b', 'c'];
  assert.deepEqual(bandDs(mountLineChart(harness, { series, labels })), []);
  assert.deepEqual(
    bandDs(mountLineChart(harness, { series, labels, band: { lo: [null, null, null], hi: [null, null, null] } })),
    [],
  );
});

test('charts.jsx::LineChart dashes a series that asks for it, and only that one', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const solid = [0.2, 0.3, 0.25];
  const ucl = [0.5, 0.5, 0.5];
  const container = mountLineChart(harness, {
    series: [{ key: 'ewma', values: solid }, { key: 'ucl', values: ucl, dash: true, fill: false }],
    labels: ['a', 'b', 'c'],
  });

  const geom = plotGeom(window, 3, axisRange(window, [...solid, ...ucl]));
  const dashed = pathWithD(container, window.DF_SPARK_PATH.axisPaths(ucl, geom).line);
  const plain = pathWithD(container, window.DF_SPARK_PATH.axisPaths(solid, geom).line);
  assert.ok(dashed.getAttribute('stroke-dasharray'), 'the UCL line is dashed');
  assert.equal(plain.getAttribute('stroke-dasharray'), null, 'the EWMA line is solid');
});

// ── Markers and MarkedChart ───────────────────────────────────────────────

const REGIME_LABELS = ['2026-07-13', '2026-07-14', '2026-07-15', '2026-07-16', '2026-07-17'];
const REGIME_MARKERS = [
  { date: '2026-07-15', label: 'Starvation threshold raised to 72h' },
  { date: '2026-07-15', label: 'Auto-watcher benign closes' },
  { date: '2026-07-16', label: 'Verify-scope flip' },
  { date: '2026-08-01', label: 'outside the window' },
];

// tab_escalation_analytics.jsx's RegimeMarkers as it rendered before the
// overlay was lifted into charts.jsx, frozen here as the DOM a 'regime' mark
// must keep producing.
function legacyRegimeMarkers(React, { labels, markers }) {
  const h = React.createElement;
  const n = labels.length;
  return h(
    'div',
    { style: { position: 'absolute', inset: 0, pointerEvents: 'none' } },
    markers.map((m, i) => {
      const idx = labels.indexOf(m.date);
      if (idx === -1) return null;
      const frac = idx / (n - 1);
      const left = `calc(38px + ${frac} * (100% - 50px))`;
      return h(
        'div',
        { key: i, style: { position: 'absolute', left, top: 0, bottom: 20 } },
        h('div', { style: { height: '100%', borderLeft: '1px dashed var(--fg-3)' } }),
        h(
          'span',
          {
            style: {
              position: 'absolute', top: 0, left: 3, fontSize: 9, color: 'var(--fg-3)',
              whiteSpace: 'nowrap', transform: 'translateY(-2px)',
            },
          },
          m.label,
        ),
      );
    }),
  );
}

function overlayOf(harness, marks, extra = {}) {
  return harness.mount(harness.window.DF_CHARTS.Markers, { marks, ...extra }).container;
}

function plotLeft(window, fraction) {
  const pad = window.DF_CHARTS.PLOT_PAD;
  return `calc(${pad.l}px + ${fraction} * (100% - ${pad.l + pad.r}px))`;
}

test('charts.jsx::Markers renders regime marks as the RegimeMarkers DOM they replace', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const legacy = harness.mount(
    props => legacyRegimeMarkers(window.React, props),
    { labels: REGIME_LABELS, markers: REGIME_MARKERS },
  ).container;
  const marks = window.DF_CHART_MARKS.dateLabelMarks(REGIME_LABELS, REGIME_MARKERS);

  assert.equal(overlayOf(harness, marks).innerHTML, legacy.innerHTML);
});

test('charts.jsx::Markers renders nothing for no marks', t => {
  const harness = openWindow(t);
  assert.equal(overlayOf(harness, []).innerHTML, '');
  assert.equal(overlayOf(harness, undefined).innerHTML, '');
});

test('charts.jsx::Markers draws an artefact as one hatched range spanning its two fractions', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const container = overlayOf(harness, [
    { key: 'art', kind: 'artefact', label: 'mass flake re-run', fraction: 0.25, endFraction: 0.75 },
  ]);

  const ranges = container.querySelectorAll('.chart-mark-range');
  assert.equal(ranges.length, 1);
  const pad = window.DF_CHARTS.PLOT_PAD;
  assert.equal(ranges[0].style.left, plotLeft(window, 0.25));
  assert.equal(ranges[0].style.width, `calc(0.5 * (100% - ${pad.l + pad.r}px))`);
  assert.match(ranges[0].style.background, /repeating-linear-gradient/);
  assert.ok(ranges[0].textContent.includes('mass flake re-run'));
});

test('charts.jsx::Markers draws an alarm as a line carrying its label', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const label = 'unexplained change at landing #212';
  const container = overlayOf(harness, [{ key: 'alarm:212', kind: 'alarm', label, fraction: 0.5 }]);

  const alarms = container.querySelectorAll('.chart-mark-alarm');
  assert.equal(alarms.length, 1);
  assert.equal(alarms[0].style.left, plotLeft(window, 0.5));
  assert.equal(alarms[0].textContent, label);
});

test('charts.jsx::Markers gives a labelled mark a hover title only where markHover says something', t => {
  const harness = openWindow(t);
  const { markHover } = harness.window.DF_CHART_MARKS;
  const bare = { key: 'a', kind: 'model', label: 'opus -> fable', fraction: 0 };
  const referenced = { key: 'b', kind: 'model', label: 'sonnet -> opus', fraction: 1, ref: 'abc123', verdict: '+3.1 pp' };
  const container = overlayOf(harness, [bare, referenced]);

  const titled = [...container.querySelectorAll('[title]')];
  assert.deepEqual(titled.map(el => el.title), [markHover(referenced)]);
  assert.equal(markHover(bare), null);
});

test('charts.jsx::Markers names an unlabelled tick or rug mark in its hover', t => {
  const harness = openWindow(t);
  const { markHover } = harness.window.DF_CHART_MARKS;
  const deploy = { key: 'd', kind: 'deploy', label: 'deploy 0123456789ab', fraction: 0.5 };
  const budget = { key: 'b', kind: 'budget', label: 'implementer budget 20 -> 30', fraction: 0.25, ref: 'cfg' };
  const container = overlayOf(harness, [deploy, budget]);

  const rug = container.querySelector('.chart-mark-rug');
  const tick = container.querySelector('.chart-mark-tick');
  assert.equal(rug.title, deploy.label, 'the label is the only text a rug mark has');
  assert.equal(rug.textContent, '', 'a rug mark draws no label');
  assert.equal(tick.title, markHover(budget));
  assert.equal(tick.textContent, '');
});

test('charts.jsx::Markers draws a legend only when asked, one entry per style present', t => {
  const harness = openWindow(t);
  const { legendEntries } = harness.window.DF_CHART_MARKS;
  const marks = [
    { key: 'm', kind: 'model', label: 'opus -> fable', fraction: 0 },
    { key: 'd', kind: 'deploy', label: 'deploy x', fraction: 0.5 },
    { key: 'a', kind: 'alarm', label: 'unexplained change at landing #4', fraction: 1 },
  ];

  assert.equal(overlayOf(harness, marks).querySelector('.chart-mark-legend'), null);

  const legend = overlayOf(harness, marks, { legend: true }).querySelector('.chart-mark-legend');
  assert.ok(legend, 'legend: true draws the legend');
  assert.deepEqual(
    [...legend.querySelectorAll('.chart-mark-legend-entry')].map(e => e.textContent),
    [...legendEntries(marks)].map(e => e.label),
  );
});

test('charts.jsx::MarkedChart renders its chart and the overlay in one relatively positioned box', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const marks = window.DF_CHART_MARKS.dateLabelMarks(REGIME_LABELS, REGIME_MARKERS);
  const chart = window.React.createElement('div', { className: 'the-chart' });
  const { container } = harness.mount(window.DF_CHARTS.MarkedChart, { marks, children: chart });

  const box = container.firstElementChild;
  assert.equal(box.style.position, 'relative');
  assert.equal(box.children.length, 2);
  assert.ok(box.children[0].classList.contains('the-chart'), 'the chart comes first');
  assert.equal(
    box.children[1].innerHTML,
    overlayOf(harness, marks).firstElementChild.innerHTML,
    'then the Markers overlay for the same marks',
  );
});

// ── Escalation analytics through the shared overlay (sketch row 21) ───────

// The three 2026-07-15 entries of dashboard/regime-markers.yaml, as
// data/escalation_analytics.py::load_regime_markers serves them.
const JULY_15_MARKERS = [
  { date: '2026-07-15', label: 'Starvation threshold raised to 72h', tasks: [2631] },
  { date: '2026-07-15', label: 'Auto-watcher benign closes', tasks: [2630] },
  { date: '2026-07-15', label: 'Verify-scope flip', tasks: [2593] },
];

function mountAnalyticsAcrossJuly15(harness) {
  const { window } = harness;
  const byDay = value => Object.fromEntries(REGIME_LABELS.map((day, i) => [day, value(i)]));
  window.localStorage.setItem('df.escanalytics.window', JSON.stringify('all'));
  seedDashboardData(window, {
    ESCALATION_ANALYTICS: escalationAnalytics(
      [{
        project: 'p',
        origin: { daily_by_source: byDay(i => ({ steward: i + 1 })), sources: [] },
        workflow: {
          churn_daily: byDay(i => i),
          esc_per_done_daily: REGIME_LABELS.map((date, i) => ({ date, ratio: i / 4 })),
          done_counts_read: true,
        },
        samples: [['2026-07-15', 'human', 1, 3600], ['2026-07-16', 'steward', 1, 86400]],
      }],
      { regimeMarkers: JULY_15_MARKERS },
    ),
    __receipt: receiptsFor(window.DF_DATA_LOADER, ['ESCALATION_ANALYTICS']),
  });
  return harness.mount(window.DF_TABS.EscalationAnalyticsTab, { projectFilter: [] }).container;
}

function chartName(C, chart) {
  if (chart.type === C.StackedAreaChart) return 'origin filings';
  const [first] = chart.props.series;
  return { churn: 'workflow churn', ratio: 'escalations per done' }[first.key] ?? 'lifespan ecdf';
}

test('tab_escalation_analytics.jsx hands every time chart to MarkedChart with dateLabelMarks of its own labels', t => {
  const harness = openWindow(t);
  const { window } = harness;
  const C = window.DF_CHARTS;
  const { dateLabelMarks } = window.DF_CHART_MARKS;
  const real = C.MarkedChart;
  const handed = [];
  C.MarkedChart = props => {
    handed.push(props);
    return window.React.createElement(real, props);
  };
  t.after(() => { C.MarkedChart = real; });

  mountAnalyticsAcrossJuly15(harness);

  const byChart = Object.fromEntries(handed.map(props => [chartName(C, props.children), props]));
  assert.deepEqual(
    Object.keys(byChart).sort(),
    ['escalations per done', 'lifespan ecdf', 'origin filings', 'workflow churn'],
  );
  for (const name of ['origin filings', 'workflow churn', 'escalations per done']) {
    const { marks, children } = byChart[name];
    assert.deepEqual(marks, dateLabelMarks(children.props.labels, JULY_15_MARKERS), name);
    assert.equal(marks.length, 3, `${name} marks all three regime changes`);
  }
  const ecdf = byChart['lifespan ecdf'].marks;
  assert.equal(ecdf.length, 1);
  assert.equal(ecdf[0].label, '6h');
  assert.ok(ecdf[0].fraction > 0 && ecdf[0].fraction < 1, 'the 6h mark sits inside the threshold grid');
});

test('tab_escalation_analytics.jsx still draws the three 2026-07-15 regime markers on each date chart', t => {
  const harness = openWindow(t);
  const container = mountAnalyticsAcrossJuly15(harness);
  const july15Left = plotLeft(harness.window, REGIME_LABELS.indexOf('2026-07-15') / (REGIME_LABELS.length - 1));

  for (const { label } of JULY_15_MARKERS) {
    const drawn = [...container.querySelectorAll('span')].filter(span => span.textContent === label);
    assert.equal(drawn.length, 3, `${label}: once on each of the three date charts`);
    for (const span of drawn) assert.equal(span.parentElement.style.left, july15Left, label);
  }
});

// ── The Quality tab ───────────────────────────────────────────────────────
// Mounted alone, its one read answered per url by a stub fetch: the body is
// the whole /quality response, built by wire_fixtures.mjs::qualityBody.

const QUALITY_PATH = '/api/v2/dashboard/quality';
const QUALITY_PROJECT = 'dark_factory';
const OTHER_PROJECT = 'reify';
const NO_PROJECT = 'No project is configured.';
const QUALITY_LOADING = 'loading…';
const QUALITY_UNAVAILABLE = 'The quality series could not be fetched.';

// quality_series.py::CATALOGUE's kinds for the metric ids the layout names.
const RATE_IDS = new Set([
  'rejected', 'reviewer_blocking', 'flakes_confirmed', 'flakes_real_red', 'review_rounds_ge2',
  'turns_capped', 'charter_blocked',
]);
const kindOf = id => (RATE_IDS.has(id) ? 'rate' : id === 'all_role_usd' ? 'cost' : 'count');

const FLIP = qualityMarker({ id: 'routing:reviewer_comprehensive:1', ordinal: 2, label: 'reviewer effort high → max' });
const ARTEFACT = qualityMarker({
  id: 'curated:disk-full', ordinal: 1, untilOrdinal: 3, kind: 'artefact', label: 'disk full',
  source: { type: 'curated', file: 'regime-markers.yaml' },
});
const DEPLOY = qualityMarker({ id: 'deploy:0123456789ab', ordinal: 4, kind: 'deploy', label: 'deploy 0123456789ab' });
const OUTSIDE = qualityMarker({ id: 'model:merger:9', ordinal: 40, kind: 'model', label: 'outside the window' });
const QUALITY_MARKER_ROWS = [FLIP, ARTEFACT, DEPLOY, OUTSIDE];

const EXPLAINED = { ordinal: 4, onset_ordinal: 3, direction: 'up', explained_by: FLIP.id };
const UNEXPLAINED = { ordinal: 5, onset_ordinal: 4, direction: 'up', explained_by: null };
const ALARMS = { rejected: [EXPLAINED], flakes_confirmed: [UNEXPLAINED] };

const NOT_YET = qualityComparison({
  markerId: FLIP.id, metric: 'rejected', state: 'not_yet', nBefore: 290, nAfter: 25, etaDays: 6.25, minDetectable: 0.1255,
});
const COST_RATIO = qualityComparison({ markerId: FLIP.id, metric: 'all_role_usd', effect: 0.82, lo: 0.7, hi: 0.95 });
const BESIDE_RATE = qualityComparison({ markerId: FLIP.id, metric: 'flakes_real_red', effect: -0.02, lo: -0.05, hi: 0.01 });
const COST_MODEL = { a: 0.4, b: 0.7, reference_cost_100: 1.837, residual_median: -0.1204, state: 'frozen' };
const MODULE_ROWS = [
  qualityModuleRow({ module: 'dashboard' }),
  qualityModuleRow({ module: 'orchestrator', usdPer100Lines: 3.5 }),
  qualityModuleRow({ module: null, ranked: false, usdPer100Lines: null }),
];

const LANDINGS = 6;
const unmeasured = () => Array.from({ length: LANDINGS }, () => null);

function qualityFixtureMetric(id) {
  const kind = kindOf(id);
  if (kind === 'rate') {
    const center = [0.1, 0.12, 0.15, 0.2, 0.25, 0.22];
    return qualityMetric({
      kind,
      points: [0, 0, 1, 0, 1, 0],
      rolling: { center, lo: [0.05, 0.06, 0.08, 0.12, 0.16, 0.14], hi: [0.2, 0.22, null, 0.3, 0.35, 0.32], n: center.map(() => 20) },
      ewma: { line: [0.1, 0.11, 0.13, 0.16, 0.2, 0.21], ucl: center.map(() => 0.3) },
      alarms: ALARMS[id] ?? [],
    });
  }
  const center = kind === 'cost' ? [1.2, 1.4, 1.1, 1.6, 1.5, 1.3] : [1, 1.5, 1.25, 1.75, 1.5, 2];
  return qualityMetric({
    kind,
    points: center,
    source: kind === 'cost' ? 'invocations' : 'rejections',
    rolling: { center, lo: unmeasured(), hi: unmeasured(), n: center.map(() => 20) },
  });
}

function layoutEntries(window) {
  const { QUALITY_LAYOUT } = window.DF_QUALITY_VIEW;
  return [...QUALITY_LAYOUT.outcomes, ...QUALITY_LAYOUT.changeCost];
}

function qualityFixture(window, { served = '100', requested = served } = {}) {
  const ids = layoutEntries(window).flatMap(entry => [entry.metric, entry.beside]).filter(id => id !== null);
  const quality = qualityPayload({
    axis: Array.from({ length: LANDINGS }, () => ({})),
    metrics: Object.fromEntries(ids.map(id => [id, qualityFixtureMetric(id)])),
    costModel: COST_MODEL,
    modules: MODULE_ROWS,
    beforeAfter: [NOT_YET, COST_RATIO, BESIDE_RATE],
  });
  const body = qualityBody({ quality, markers: qualityMarkers(QUALITY_MARKER_ROWS), window: served });
  return { ...body, WINDOW: { ...body.WINDOW, requested } };
}

const answered = body => ({ ok: true, status: 200, json: async () => body });

function qualityParam(window, { project = QUALITY_PROJECT, landings = '100', module = null } = {}) {
  return window.DF_DATA_LOADER.qualityQuery({ project, window: landings, module });
}

function qualityUrl(window, query) {
  return window.DF_DATA_LOADER.ON_DEMAND_KEYS.quality.url(qualityParam(window, query));
}

// The tab mounted over PROJECTS (each a projectRow) with `answers` keyed by
// url. Returns its container and a reader of the /quality urls asked so far.
function mountQuality(harness, { projects = [QUALITY_PROJECT, OTHER_PROJECT], answers = {} } = {}) {
  const { window } = harness;
  allowAsyncAct(window);
  if (projects.length > 0) {
    seedDashboardData(window, {
      PROJECTS: projects.map(projectRow),
      __receipt: receiptsFor(window.DF_DATA_LOADER, ['PROJECTS']),
    });
  }
  const asked = stubFetch(window, answers);
  const { container } = harness.mount(window.DF_QUALITY.QualityTab, {});
  return { container, asked: () => asked.filter(url => url.startsWith(QUALITY_PATH)) };
}

async function mountAnswered(harness) {
  const { window } = harness;
  const body = qualityFixture(window);
  const mounted = mountQuality(harness, { answers: { [qualityUrl(window)]: answered(body) } });
  await settle(window);
  return { ...mounted, body, quality: body.QUALITY.value };
}

const section = (container, name) => container.querySelector(`[data-quality-section="${name}"]`);
const metricPanel = (container, id) => container.querySelector(`[data-quality-metric="${id}"]`);

test('tab_quality.jsx::QualityTab says no project is configured and requests nothing without PROJECTS', async t => {
  const harness = openWindow(t);
  const { container, asked } = mountQuality(harness, { projects: [] });
  await settle(harness.window);

  assert.ok(container.textContent.includes(NO_PROJECT), container.textContent);
  assert.deepEqual(asked(), []);
});

test('tab_quality.jsx::QualityTab requests the first project at the default landing window, loading until it answers', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  let answer;
  const { container, asked } = mountQuality(harness, {
    answers: { [qualityUrl(window)]: new Promise(resolve => { answer = resolve; }) },
  });
  await settle(window);

  assert.deepEqual(asked(), [qualityUrl(window, { project: QUALITY_PROJECT, landings: window.DF_WINDOW_CHIP.LANDING_DEFAULT_WINDOW })]);
  assert.ok(container.textContent.includes(QUALITY_LOADING), container.textContent);

  answer(answered(qualityFixture(window)));
  await settle(window);
  assert.equal(container.textContent.includes(QUALITY_LOADING), false, container.textContent);
  assert.ok(section(container, 'headlines'), 'the answered read renders its headlines');
});

test('tab_quality.jsx::QualityTab says the quality series could not be fetched when its read fails', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const { container } = mountQuality(harness, {
    answers: { [qualityUrl(window)]: { ok: false, status: 503, json: async () => ({}) } },
  });
  await settle(window);

  assert.ok(container.textContent.includes(QUALITY_UNAVAILABLE), container.textContent);
  assert.equal(container.textContent.includes(QUALITY_LOADING), false, container.textContent);
});

test('tab_quality.jsx::QualityTab renders a QUALITY hole by its reason', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const reason = 'quality.db unreadable: OperationalError: database is locked';
  const body = { ...qualityFixture(window), QUALITY: { value: null, as_of: null, state: 'unknown', reason, freshness_bound_seconds: 1200 } };
  const { container } = mountQuality(harness, { answers: { [qualityUrl(window)]: answered(body) } });
  await settle(window);

  assert.ok(container.textContent.includes(reason), container.textContent);
  assert.equal(section(container, 'headlines'), null, 'a hole draws no headline tiles');
});

test('tab_quality.jsx::QualityTab renders one StatTile per laid-out metric, its headline through headlineText and its sparkline from rolling.center', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const { container, quality } = await mountAnswered(harness);
  const { headlineText } = window.DF_QUALITY_VIEW;
  const { datumView } = window.DF_DATUM;
  const { sparkPaths } = window.DF_SPARK_PATH;

  const { ruleCaption } = window.DF_QUALITY_VIEW;

  const tiles = [...section(container, 'headlines').querySelectorAll('.kpi')];
  const entries = layoutEntries(window);
  assert.deepEqual(tiles.map(tile => tile.querySelector('.lbl').textContent), entries.map(entry => entry.title));
  entries.forEach((entry, i) => {
    const metric = quality.metrics[entry.metric];
    assert.equal(tiles[i].getAttribute('title'), ruleCaption(metric.rule), `${entry.metric}: its tooltip is its rule`);
    const expected = datumView(metric.headline, { format: value => headlineText(value, metric.unit) }).text;
    assert.equal(tiles[i].querySelector('.val').textContent, expected, entry.metric);
    const spark = tiles[i].querySelector('.spark path[fill="none"]');
    assert.equal(spark.getAttribute('d'), sparkPaths(metric.rolling.center, 100, 28).line, entry.metric);
  });
});

function plotGeomOf(window, panel, count, values, minAxisMax) {
  const [, , width, height] = panel.querySelector('svg').getAttribute('viewBox').split(' ').map(Number);
  return plotGeom(window, count, axisRange(window, values, minAxisMax), { width, height });
}

test('tab_quality.jsx::QualityTab draws each outcome chart from chartSeries: its band, EWMA line and dashed UCL', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const { container, quality } = await mountAnswered(harness);
  const { chartSeries, chartAxis, QUALITY_LAYOUT } = window.DF_QUALITY_VIEW;
  const { axisPaths, bandPaths } = window.DF_SPARK_PATH;

  for (const { metric: id } of QUALITY_LAYOUT.outcomes) {
    const panel = metricPanel(section(container, 'outcomes'), id);
    assert.ok(panel, `the outcomes section draws ${id}`);
    const { series, band } = chartSeries(quality.metrics[id]);
    const { formatY, minAxisMax } = chartAxis(quality.metrics[id]);
    const values = [...series.flatMap(s => s.values), ...(band ? band.hi : [])];
    const geom = plotGeomOf(window, panel, LANDINGS, values, minAxisMax);
    const ticks = Array.from({ length: 5 }, (_, i) => geom.min + (geom.range * i) / 4);

    assert.deepEqual(yTickTexts(panel), ticks.map(formatY), `${id}: the y axis is chartAxis's, in its headline units`);

    assert.deepEqual(bandDs(panel), band ? [...bandPaths(band.lo, band.hi, geom)] : [], `${id}: band`);
    for (const s of series) {
      const line = pathWithD(panel, axisPaths(s.values, geom).line);
      assert.ok(line, `${id}: the ${s.key} line is drawn`);
      assert.equal(Boolean(line.getAttribute('stroke-dasharray')), Boolean(s.dash), `${id}: only the UCL is dashed`);
    }
    assert.equal(series.some(s => s.key === 'ucl'), RATE_IDS.has(id), `${id}: a rate draws its control chart`);
  }
});

test('tab_quality.jsx::QualityTab labels each alarm with its verdict: explained by the served marker, or unexplained at its onset', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const { container } = await mountAnswered(harness);
  const { alarmVerdict, indexMarkers } = window.DF_QUALITY_VIEW;
  const byId = indexMarkers(QUALITY_MARKER_ROWS);
  const alarmTexts = id => [...metricPanel(container, id).querySelectorAll('.chart-mark-alarm')].map(mark => mark.textContent);

  assert.deepEqual(alarmTexts('rejected'), [alarmVerdict(EXPLAINED, byId)]);
  assert.equal(alarmTexts('rejected')[0], `explained by ${FLIP.label}`);
  assert.deepEqual(alarmTexts('flakes_confirmed'), [alarmVerdict(UNEXPLAINED, byId)]);
  assert.equal(alarmTexts('flakes_confirmed')[0], `unexplained change at landing #${UNEXPLAINED.onset_ordinal}`);
});

test('tab_quality.jsx::QualityTab overlays every in-window served marker on the change-cost charts, the artefact hatched', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const { container, quality } = await mountAnswered(harness);
  const { qualityMarks, QUALITY_LAYOUT } = window.DF_QUALITY_VIEW;

  for (const { metric: id } of QUALITY_LAYOUT.changeCost) {
    const panel = metricPanel(section(container, 'change-cost'), id);
    assert.ok(panel, `the change-cost section draws ${id}`);
    const metric = quality.metrics[id];
    const marks = qualityMarks({
      axis: quality.axis, markers: QUALITY_MARKER_ROWS, alarms: metric.alarms,
      beforeAfter: quality.before_after, metricId: id, kind: metric.kind,
    });
    assert.deepEqual(
      [...marks.map(mark => mark.key)].sort(),
      [FLIP.id, ARTEFACT.id, DEPLOY.id].sort(),
      `${id}: the in-window markers, and not the one past the last landing`,
    );
    const overlay = window.document.createElement('div');
    overlay.innerHTML = overlayOf(harness, marks, { legend: true }).innerHTML;
    assert.ok(panel.innerHTML.includes(overlay.innerHTML), `${id}: the overlay is Markers over qualityMarks, with its legend`);
    assert.equal(panel.querySelectorAll('.chart-mark-range').length, 1, `${id}: the artefact is one hatched range`);
    assert.equal(panel.textContent.includes(OUTSIDE.label), false, id);
  }
});

test('tab_quality.jsx::QualityTab reads the cost model in force: reference cost at 100 lines and median residual', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const { container, quality } = await mountAnswered(harness);
  const readings = window.DF_QUALITY_VIEW.costModelReadings(quality.cost_model);
  const tiles = [...section(container, 'cost-model').querySelectorAll('.kpi')];

  assert.deepEqual(
    tiles.map(tile => [tile.querySelector('.lbl').textContent, tile.querySelector('.val').textContent]),
    [...readings.map(reading => [`${reading.label} · ${reading.hint}`, reading.text])],
  );
});

test('tab_quality.jsx::QualityTab shows each before/after comparison under its marker, n, ETA and MDE for one not yet measurable', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const { container } = await mountAnswered(harness);
  const { beforeAfterText } = window.DF_QUALITY_VIEW;
  const panel = section(container, 'before-after');

  assert.ok(panel.textContent.includes(FLIP.label), panel.textContent);
  assert.ok(panel.textContent.includes(beforeAfterText(NOT_YET, 'rate')), panel.textContent);
  assert.ok(panel.textContent.includes(beforeAfterText(COST_RATIO, 'cost')), panel.textContent);
});

test('tab_quality.jsx::QualityTab titles every before/after row through metricTitle, a beside metric by its rule', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const { container, quality } = await mountAnswered(harness);
  const { metricTitle } = window.DF_QUALITY_VIEW;
  const rows = [...section(container, 'before-after').querySelectorAll('.mono')].map(value => value.parentElement.textContent);

  const titles = rows.map(row => row.slice(0, row.indexOf(': ')));
  assert.deepEqual(titles, [NOT_YET, COST_RATIO, BESIDE_RATE].map(entry => metricTitle(entry.metric, quality.metrics[entry.metric])));
  assert.equal(titles.includes(BESIDE_RATE.metric), false, 'no row is titled by its raw metric id');
});

function moduleRow(container, name) {
  const rows = [...section(container, 'modules').querySelectorAll('tbody tr')];
  const row = rows.find(candidate => candidate.firstElementChild.textContent === name);
  assert.ok(row, `the Modules table has no row for ${name}: ${rows.map(r => r.firstElementChild.textContent)}`);
  return row;
}

function buttonNamed(container, label) {
  const buttons = [...container.querySelectorAll('button')].filter(button => button.textContent === label);
  assert.equal(buttons.length, 1, `expected one '${label}' button`);
  return buttons[0];
}

test('tab_quality.jsx::QualityTab drills down server-side when a Modules row is clicked, and clear drops the module', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const whole = qualityUrl(window);
  const drilled = qualityUrl(window, { module: 'dashboard' });
  const { container, asked } = mountQuality(harness, {
    answers: { [whole]: answered(qualityFixture(window)), [drilled]: answered(qualityFixture(window)) },
  });
  await settle(window);

  click(window, moduleRow(container, '(no src)'));
  await settle(window);
  assert.deepEqual(asked(), [whole], 'the no-src bucket names no module to drill into');

  click(window, moduleRow(container, 'dashboard'));
  await settle(window);
  assert.deepEqual(asked(), [whole, drilled]);

  click(window, buttonNamed(container, 'clear'));
  await settle(window);
  assert.deepEqual(asked(), [whole, drilled, whole]);
});

test('tab_quality.jsx::QualityTab re-requests at the chosen landing window and labels the window the server served', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const { onDemandReceipt } = window.DF_DATA_LOADER;
  const { receiptWindow, windowLabel } = window.DF_WINDOW_CHIP;
  const first = qualityUrl(window);
  const wider = qualityUrl(window, { landings: '300' });
  const { container, asked } = mountQuality(harness, {
    answers: {
      [first]: answered(qualityFixture(window)),
      [wider]: answered(qualityFixture(window, { requested: '300', served: '100' })),
    },
  });
  await settle(window);
  const served = () => container.querySelector('[data-quality-window]').textContent;
  assert.equal(served(), windowLabel(receiptWindow(onDemandReceipt('quality', qualityParam(window)))));

  click(window, buttonNamed(container, '300'));
  await settle(window);
  assert.deepEqual(asked(), [first, wider]);
  const label = windowLabel(receiptWindow(onDemandReceipt('quality', qualityParam(window, { landings: '300' }))));
  assert.equal(label, '100 (300 not available)');
  assert.equal(served(), label);
});

test('tab_quality.jsx::QualityTab re-requests for the project chosen in its selector', async t => {
  const harness = openWindow(t);
  const { window } = harness;
  const other = qualityUrl(window, { project: OTHER_PROJECT });
  const { container, asked } = mountQuality(harness, {
    answers: { [qualityUrl(window)]: answered(qualityFixture(window)), [other]: answered(qualityFixture(window)) },
  });
  await settle(window);

  const select = container.querySelector('select');
  assert.deepEqual([...select.options].map(option => option.value), [QUALITY_PROJECT, OTHER_PROJECT]);
  window.React.act(() => {
    select.value = OTHER_PROJECT;
    select.dispatchEvent(new window.Event('change', { bubbles: true }));
  });
  await settle(window);
  assert.deepEqual(asked(), [qualityUrl(window), other]);
});

test("tab_quality.jsx::QualityTab's Snapshots section renders the QUALITY_SNAPSHOTS hole's reason", async t => {
  const harness = openWindow(t);
  const { container, body } = await mountAnswered(harness);

  assert.ok(section(container, 'snapshots').textContent.includes(body.QUALITY_SNAPSHOTS.reason));
});

// The whole page also lands one async App update outside act() after any tab
// switch (measured on the Costs tab too), so this window tolerates exactly
// that warning and nothing else on console.error.
const PAGE_ACT_WARNING = 'Warning: An update to %s inside a test was not wrapped in act(...)';

test('app.jsx renders a Quality rail entry, and selecting it mounts QualityTab', t => {
  const harness = buildDashboardWindow();
  t.after(() => {
    harness.close();
    assert.deepEqual(harness.consoleErrors.filter(entry => !entry.startsWith(PAGE_ACT_WARNING)), []);
  });
  const root = harness.mountPage();
  const button = [...root.querySelectorAll('.rail-nav button')].find(candidate =>
    [...candidate.children].some(part => part.textContent === 'Quality'),
  );
  assert.ok(button, 'the rail has no "Quality" entry');

  click(harness.window, button);
  assert.equal(root.querySelector('.crumbs .here').textContent, 'Quality');
  assert.ok(root.querySelector('[data-quality-tab]'), 'the Quality tab is mounted');
});
