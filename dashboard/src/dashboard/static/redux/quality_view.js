// quality_view.js — the pure decisions behind the Quality tab
// (tab_quality.jsx): which served metrics each section draws and their titles,
// the text of a headline, a counting rule, an alarm verdict and a before/after
// comparison, the arrays and y-axis a metric's chart draws, the cost model's
// readings, and where each served marker and alarm sits on that chart.
//
// Every figure is read from the served QUALITY / QUALITY_MARKERS values
// (dashboard/src/dashboard/data/quality_series.py::QualitySeries.to_wire and
// quality_markers.py::MarkerStream.to_wire own their shapes): nothing here
// smooths, re-buckets or recomputes a statistic.
//
// LOAD CONTRACT. It destructures window.DF_CHART_MARKS at module scope with no
// fallback, so index.html loads it after chart_marks.js and before the Babel
// JSX tags (test_index_html.py::_MODULE_EDGES pins both edges). node
// requires the same file as CommonJS once its test has put that global on a
// window shim; coverage is behavioural, in
// dashboard/tests/js/quality_view.test.mjs.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const {
  fractionAt: qualityFractionAt,
  rangeFractions: qualityRangeFractions,
  markStyleFor: qualityMarkStyleFor,
} = window.DF_CHART_MARKS;

const QUALITY_LAYOUT = Object.freeze({
  outcomes: Object.freeze([
    Object.freeze({ metric: 'rejected', beside: 'rejections_per_landing', title: 'Verify rejections' }),
    Object.freeze({ metric: 'reviewer_blocking', beside: null, title: 'Reviewer blocking' }),
    Object.freeze({ metric: 'flakes_confirmed', beside: 'flakes_real_red', title: 'Confirmed flakes' }),
    Object.freeze({ metric: 'reverts', beside: null, title: 'Reverts' }),
    Object.freeze({ metric: 'reverifies_per_landing', beside: null, title: 'Re-verifies per landing' }),
  ]),
  changeCost: Object.freeze([
    Object.freeze({ metric: 'all_role_usd', beside: null, title: 'All-role cost per landing' }),
    Object.freeze({ metric: 'implementer_invocations', beside: null, title: 'Implementer invocations' }),
    Object.freeze({ metric: 'turns', beside: 'turns_capped', title: 'Turns per landing' }),
    Object.freeze({ metric: 'review_rounds_ge2', beside: null, title: '≥2 review rounds' }),
    Object.freeze({ metric: 'charter_blocked', beside: null, title: 'Charter blocked' }),
  ]),
});

const QUALITY_LAID_OUT = Object.freeze([...QUALITY_LAYOUT.outcomes, ...QUALITY_LAYOUT.changeCost]);

// A metric laid out beside another has no layout title, so it is named by what
// its served rule counts, as the panel it sits in names it.
function metricTitle(metricId, metric) {
  const entry = QUALITY_LAID_OUT.find(laidOut => laidOut.metric === metricId);
  return entry ? entry.title : metric.rule.counts;
}

// The one way a renderer is chosen from a table keyed by a served field: a
// value the table has no entry for is refused, named, rather than failing
// later as an anonymous call of undefined inside a render.
function qualityRendererFor(table, key, refusal) {
  if (!Object.prototype.hasOwnProperty.call(table, key)) throw new TypeError(`quality_view.js ${refusal}`);
  return table[key];
}

function qualityPercent(share) {
  return `${(share * 100).toFixed(1)}%`;
}

function qualitySigned(text, x) {
  return x >= 0 ? `+${text}` : text;
}

function qualityPoints(difference) {
  return qualitySigned((difference * 100).toFixed(1), difference);
}

function qualityAmount(x, unit) {
  return unit === 'usd' ? `$${x.toFixed(2)}` : x.toFixed(2);
}

function qualityCentred(headline, unit) {
  const text = `${qualityAmount(headline.value, unit)} [${qualityAmount(headline.lo, unit)}–${qualityAmount(headline.hi, unit)}]`;
  return unit === 'usd' ? text : `${text} ${unit}`;
}

const QUALITY_HEADLINE_TEXT = Object.freeze({
  rate: headline => `${qualityPercent(headline.value)} [${qualityPercent(headline.lo)}–${qualityPercent(headline.hi)}]`,
  k_of_n: headline => `${headline.k} of ${headline.n} [${qualityPercent(headline.lo)}–${qualityPercent(headline.hi)}]`,
  mean: qualityCentred,
  median: qualityCentred,
});

function headlineText(headline, unit) {
  const render = qualityRendererFor(
    QUALITY_HEADLINE_TEXT, headline.form, `renders no headline form ${JSON.stringify(headline.form)}`,
  );
  return render(headline, unit);
}

function ruleCaption(rule) {
  const since = rule.measured_since === null
    ? `no producer has written here (${rule.source})`
    : `measured since ${rule.measured_since.slice(0, 10)}`;
  return [
    `counts: ${rule.counts}`,
    `population: ${rule.population}`,
    `headline: last ${rule.window_landings} landings`,
    since,
    `excludes: ${rule.excludes}`,
  ].join(' · ');
}

function chartSeries(metric) {
  const series = [{ key: 'center', values: metric.rolling.center }];
  if (metric.ewma !== null) {
    series.push({ key: 'ewma', values: metric.ewma.line }, { key: 'ucl', values: metric.ewma.ucl, dash: true });
  }
  const band = metric.kind === 'rate' ? { lo: metric.rolling.lo, hi: metric.rolling.hi } : null;
  return { series, band };
}

const QUALITY_TICK_TEXT = Object.freeze({
  rate: () => qualityPercent,
  count: unit => value => qualityAmount(value, unit),
  cost: unit => value => qualityAmount(value, unit),
});

// A metric chart's y-axis: ticks in its headline's units, scaled to the data
// (minAxisMax 0). LineChart's default least maximum of 1 would draw every rate
// on 0..1, pressing a 3% rate, its band and its UCL onto the floor of the plot.
function chartAxis(metric) {
  const tickText = qualityRendererFor(
    QUALITY_TICK_TEXT, metric.kind, `draws no axis for metric kind ${JSON.stringify(metric.kind)}`,
  );
  return { formatY: tickText(metric.unit), minAxisMax: 0 };
}

function indexMarkers(markers) {
  return new Map((markers || []).map(marker => [marker.id, marker]));
}

function alarmVerdict(alarm, markersById) {
  if (alarm.explained_by === null) return `unexplained change at landing #${alarm.onset_ordinal}`;
  const marker = markersById.get(alarm.explained_by);
  return `explained by ${marker ? marker.label : alarm.explained_by}`;
}

const QUALITY_EFFECT_TEXT = Object.freeze({
  rate: entry => `${qualityPoints(entry.effect)} pp [${qualityPoints(entry.lo)} to ${qualityPoints(entry.hi)} pp]`,
  cost: entry => `×${entry.effect.toFixed(2)} [×${entry.lo.toFixed(2)} to ×${entry.hi.toFixed(2)}]`,
});

function qualityNotYetText(entry, kind) {
  const parts = [`n ${entry.n_before} → ${entry.n_after}`];
  if (entry.eta_days !== null) parts.push(`ETA ${Math.ceil(entry.eta_days)} d`);
  if (entry.min_detectable !== null) {
    parts.push(kind === 'rate' ? `MDE ${(entry.min_detectable * 100).toFixed(1)} pp` : `MDE ${entry.min_detectable}`);
  }
  return parts.join(' · ');
}

function beforeAfterText(entry, kind) {
  if (entry.state === 'not_yet') return qualityNotYetText(entry, kind);
  const effectText = qualityRendererFor(
    QUALITY_EFFECT_TEXT, kind,
    `renders no ${JSON.stringify(kind)} comparison (marker ${entry.marker_id}, metric ${entry.metric})`,
  );
  return `${effectText(entry)} · n ${entry.n_before} → ${entry.n_after}`;
}

const QUALITY_BASELINE_HINT = Object.freeze({
  frozen: 'frozen baseline',
  provisional: 'provisional baseline',
});

// The cost model in force (quality_series.py::CostModel.to_wire) as two
// readings; a figure the server has not fitted yet reads as null text.
function costModelReadings(costModel) {
  const hint = QUALITY_BASELINE_HINT[costModel.state] ?? null;
  const reference = costModel.reference_cost_100;
  const residual = costModel.residual_median;
  return [
    {
      key: 'reference_cost_100',
      label: 'Reference cost at 100 lines',
      text: reference === null ? null : qualityAmount(reference, 'usd'),
      hint,
    },
    {
      key: 'residual_median',
      label: 'Median log residual',
      text: residual === null ? null : qualitySigned(residual.toFixed(3), residual),
      hint,
    },
  ];
}

function qualityAppliesTo(marker, metricId) {
  return marker.series === null || marker.series.includes(metricId);
}

function qualityPlacement(marker, ordinals) {
  if (qualityMarkStyleFor(marker.kind) !== 'range') {
    const fraction = qualityFractionAt(ordinals, marker.ordinal);
    return fraction === null ? null : { fraction };
  }
  const span = qualityRangeFractions(ordinals, marker.ordinal, marker.until_ordinal ?? marker.ordinal);
  return span && { fraction: span.start, endFraction: span.end };
}

function qualityMarkerMark(marker, ordinals, verdict) {
  const placement = qualityPlacement(marker, ordinals);
  return placement && { key: marker.id, kind: marker.kind, label: marker.label, ...placement, ref: marker.ref, verdict };
}

function qualityMarks({ axis, markers, alarms, beforeAfter, metricId, kind }) {
  const ordinals = axis.map(point => point.ordinal);
  const served = markers || [];
  const verdicts = new Map(beforeAfter
    .filter(entry => entry.metric === metricId)
    .map(entry => [entry.marker_id, beforeAfterText(entry, kind)]));
  const placed = served
    .filter(marker => qualityAppliesTo(marker, metricId))
    .map(marker => qualityMarkerMark(marker, ordinals, verdicts.get(marker.id) ?? null))
    .filter(mark => mark !== null);
  const byId = indexMarkers(served);
  const alarmMarks = alarms.map(alarm => ({
    key: `alarm:${alarm.ordinal}:${alarm.direction}`,
    kind: 'alarm',
    label: alarmVerdict(alarm, byId),
    fraction: qualityFractionAt(ordinals, alarm.ordinal),
    ref: null,
    verdict: null,
  })).filter(mark => mark.fraction !== null);
  return [...placed, ...alarmMarks];
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const QUALITY_VIEW_API = {
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
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = QUALITY_VIEW_API;
}
if (typeof window !== 'undefined') {
  window.DF_QUALITY_VIEW = QUALITY_VIEW_API;
}
