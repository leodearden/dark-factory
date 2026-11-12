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

import { openWindow } from './_render_helpers.mjs';

// charts.jsx's LineChart before its ResizeObserver reports a width.
const CHART_WIDTH = 600;
const CHART_HEIGHT = 220;

function plotGeom(window, count, range) {
  const pad = window.DF_CHARTS.PLOT_PAD;
  return {
    x0: pad.l,
    y0: pad.t,
    width: Math.max(CHART_WIDTH - pad.l - pad.r, 50),
    height: CHART_HEIGHT - pad.t - pad.b,
    count,
    min: 0,
    range,
  };
}

function axisRange(window, values) {
  const { axisMax, plottableMax } = window.DF_SPARK_PATH;
  return axisMax(plottableMax(values, 1), 4, false);
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
  const expected = window.DF_SPARK_PATH.bandPaths(LO, HI, geom);
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
  const expected = window.DF_SPARK_PATH.bandPaths(LO, hi, geom);
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
    legendEntries(marks).map(e => e.label),
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
