// chart_marks.js — the pure decisions behind the marker overlay that
// charts.jsx::Markers draws over a time chart: where a mark sits on a
// date-label axis or a landing-ordinal axis, how each marker kind is drawn,
// what its hover says and which legend entries a set of marks needs.
//
// A mark is { key, kind, label, fraction, endFraction?, ref?, verdict? }:
// `fraction` (and a range's `endFraction`) is the position across the plot
// box, 0 at the first column and 1 at the last.
//
// Classic script, loaded before charts.jsx, which destructures
// window.DF_CHART_MARKS at module scope (test_index_html.py pins the order).
// node requires the same file as CommonJS; coverage is behavioural, in
// dashboard/tests/js/chart_marks.test.mjs.
//
// The exported names are function declarations, not consts: a .jsx consumer's
// destructure becomes a global `var`, which may share the classic scripts'
// scope with a `function` but not with a `const`.

const CHART_MARKS_STYLE_BY_KIND = Object.freeze({
  model: 'line',
  effort: 'line',
  prompt: 'line',
  reviewer_policy: 'line',
  routing_rule: 'line',
  cli: 'line',
  regime: 'line',
  budget: 'tick',
  turns: 'tick',
  deploy: 'rug',
  instrument: 'rug',
  artefact: 'range',
  alarm: 'alarm',
});

const CHART_MARKS_LEGEND = Object.freeze([
  Object.freeze({ style: 'line', label: 'change' }),
  Object.freeze({ style: 'tick', label: 'budget / turns' }),
  Object.freeze({ style: 'rug', label: 'deploy / instrument' }),
  Object.freeze({ style: 'range', label: 'artefact (excluded)' }),
  Object.freeze({ style: 'alarm', label: 'alarm' }),
]);

function markStyleFor(kind) {
  return Object.prototype.hasOwnProperty.call(CHART_MARKS_STYLE_BY_KIND, kind)
    ? CHART_MARKS_STYLE_BY_KIND[kind]
    : 'line';
}

function dateLabelMarks(labels, markers) {
  if (!labels || labels.length < 2 || !markers) return [];
  const last = labels.length - 1;
  const placed = [];
  markers.forEach((marker, key) => {
    const index = labels.indexOf(marker.date);
    if (index !== -1) placed.push({ key, fraction: index / last, label: marker.label, kind: 'regime' });
  });
  return placed;
}

function isOrdinal(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function firstIndexAtOrAfter(ordinals, ordinal) {
  return ordinals.findIndex(served => served >= ordinal);
}

function lastIndexAtOrBefore(ordinals, ordinal) {
  for (let i = ordinals.length - 1; i >= 0; i--) {
    if (ordinals[i] <= ordinal) return i;
  }
  return -1;
}

function fractionAt(ordinals, ordinal) {
  if (!ordinals || ordinals.length < 2 || !isOrdinal(ordinal)) return null;
  if (ordinal < ordinals[0]) return null;
  const index = firstIndexAtOrAfter(ordinals, ordinal);
  return index === -1 ? null : index / (ordinals.length - 1);
}

function rangeFractions(ordinals, from, until) {
  if (!ordinals || ordinals.length < 2 || !isOrdinal(from) || !isOrdinal(until)) return null;
  const start = firstIndexAtOrAfter(ordinals, from);
  const end = lastIndexAtOrBefore(ordinals, until);
  if (start === -1 || end === -1 || start > end) return null;
  const last = ordinals.length - 1;
  return { start: start / last, end: end / last };
}

function isNonEmptyText(value) {
  return typeof value === 'string' && value !== '';
}

function markHover(mark) {
  const extras = [mark.ref, mark.verdict].filter(isNonEmptyText);
  if (extras.length === 0) return null;
  return [mark.label, ...extras].filter(isNonEmptyText).join('\n');
}

function legendEntries(marks) {
  const present = new Set((marks || []).map(mark => markStyleFor(mark.kind)));
  return CHART_MARKS_LEGEND.filter(entry => present.has(entry.style));
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const CHART_MARKS_API = {
  markStyleFor,
  fractionAt,
  rangeFractions,
  dateLabelMarks,
  markHover,
  legendEntries,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = CHART_MARKS_API;
}
if (typeof window !== 'undefined') {
  window.DF_CHART_MARKS = CHART_MARKS_API;
}
