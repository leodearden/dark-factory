// Module-contract tests for chart_marks.js — the pure decisions behind the
// marker overlay charts.jsx draws over a time chart (task 8461): where a mark
// sits on a date-label axis or a (possibly gapped) landing-ordinal axis, how
// each served marker kind is drawn, what a mark's hover says, and which legend
// entries a set of marks needs. charts.jsx::Markers only wires these.
//
// THE KIND VOCABULARY IS SPELLED HERE AS LITERALS, because the mapping is the
// decision under test: it mirrors
// dashboard/src/dashboard/data/quality_markers.py::MarkerKind plus the
// overlay's own 'alarm'.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const SPECIFIER = '../../src/dashboard/static/redux/chart_marks.js';

function loadChartMarks() {
  const win = {};
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  delete require.cache[require.resolve(SPECIFIER)];
  return { api: require(SPECIFIER), window: win };
}

const { api: marks, window: loadedWindow } = loadChartMarks();
const {
  markStyleFor,
  fractionAt,
  rangeFractions,
  dateLabelMarks,
  markHover,
  legendEntries,
} = marks;

test('chart_marks.js assigns window.DF_CHART_MARKS to its module surface', () => {
  assert.equal(loadedWindow.DF_CHART_MARKS, marks);
  assert.deepEqual(Object.keys(marks).sort(), [
    'dateLabelMarks',
    'fractionAt',
    'legendEntries',
    'markHover',
    'markStyleFor',
    'rangeFractions',
  ]);
});

// ── dateLabelMarks: the escalation-analytics adapter ──────────────────────

const LABELS = ['2026-07-13', '2026-07-14', '2026-07-15', '2026-07-16', '2026-07-17'];

test('dateLabelMarks: places a marker at labels.indexOf(date) / (n - 1)', () => {
  assert.deepEqual(dateLabelMarks(LABELS, [{ date: '2026-07-14', label: 'Verify-scope flip' }]), [
    { key: 0, fraction: 0.25, label: 'Verify-scope flip', kind: 'regime' },
  ]);
});

test('dateLabelMarks: a date outside the labels is skipped, keeping the other keys', () => {
  assert.deepEqual(
    dateLabelMarks(LABELS, [
      { date: '2026-06-01', label: 'before the window' },
      { date: '2026-07-17', label: 'last day' },
    ]),
    [{ key: 1, fraction: 1, label: 'last day', kind: 'regime' }],
  );
});

test('dateLabelMarks: three markers on one date are three marks at one fraction', () => {
  const regime = [
    { date: '2026-07-15', label: 'Starvation threshold raised to 72h' },
    { date: '2026-07-15', label: 'Auto-watcher benign closes' },
    { date: '2026-07-15', label: 'Verify-scope flip' },
  ];
  assert.deepEqual(dateLabelMarks(LABELS, regime), [
    { key: 0, fraction: 0.5, label: 'Starvation threshold raised to 72h', kind: 'regime' },
    { key: 1, fraction: 0.5, label: 'Auto-watcher benign closes', kind: 'regime' },
    { key: 2, fraction: 0.5, label: 'Verify-scope flip', kind: 'regime' },
  ]);
});

test('dateLabelMarks: fewer than two labels, or no markers, places nothing', () => {
  const one = [{ date: '2026-07-15', label: 'x' }];
  assert.deepEqual(dateLabelMarks(['2026-07-15'], one), []);
  assert.deepEqual(dateLabelMarks([], one), []);
  assert.deepEqual(dateLabelMarks(null, one), []);
  assert.deepEqual(dateLabelMarks(LABELS, []), []);
  assert.deepEqual(dateLabelMarks(LABELS, null), []);
});

// ── fractionAt: a landing ordinal on a (possibly gapped) ordinal axis ─────

const GAPPED = [3, 4, 9, 12, 20];

test('fractionAt: an ordinal on the axis sits at its own index / (n - 1)', () => {
  assert.equal(fractionAt(GAPPED, 3), 0);
  assert.equal(fractionAt(GAPPED, 9), 0.5);
  assert.equal(fractionAt(GAPPED, 20), 1);
});

test('fractionAt: an ordinal inside a gap sits at the first served ordinal after it', () => {
  assert.equal(fractionAt(GAPPED, 5), 0.5);
  assert.equal(fractionAt(GAPPED, 13), 1);
});

test('fractionAt: off-window, null and too-short axes place nothing', () => {
  assert.equal(fractionAt(GAPPED, 2), null, 'before the first served landing');
  assert.equal(fractionAt(GAPPED, 21), null, 'past the last served landing');
  assert.equal(fractionAt(GAPPED, null), null);
  assert.equal(fractionAt(GAPPED, undefined), null);
  assert.equal(fractionAt([7], 7), null);
  assert.equal(fractionAt([], 7), null);
});

// ── rangeFractions: an artefact's landing range ───────────────────────────

test('rangeFractions: a range inside the window spans its own ordinals', () => {
  assert.deepEqual(rangeFractions(GAPPED, 4, 12), { start: 0.25, end: 0.75 });
});

test('rangeFractions: a range overlapping either window edge is clamped to it', () => {
  assert.deepEqual(rangeFractions(GAPPED, 1, 4), { start: 0, end: 0.25 });
  assert.deepEqual(rangeFractions(GAPPED, 12, 40), { start: 0.75, end: 1 });
  assert.deepEqual(rangeFractions(GAPPED, 1, 40), { start: 0, end: 1 });
});

test('rangeFractions: a range wholly outside the window, or in one gap, is null', () => {
  assert.equal(rangeFractions(GAPPED, 0, 2), null);
  assert.equal(rangeFractions(GAPPED, 21, 30), null);
  assert.equal(rangeFractions(GAPPED, 5, 8), null, 'covers no served landing');
  assert.equal(rangeFractions(GAPPED, null, 12), null);
  assert.equal(rangeFractions(GAPPED, 4, null), null);
  assert.equal(rangeFractions([4], 4, 4), null);
});

// ── markStyleFor: every served kind is drawn ──────────────────────────────

test('markStyleFor: each marker kind maps to its overlay style', () => {
  const expected = {
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
  };
  for (const [kind, style] of Object.entries(expected)) {
    assert.equal(markStyleFor(kind), style, kind);
  }
});

test('markStyleFor: an unknown kind is drawn as a line, never dropped', () => {
  assert.equal(markStyleFor('a_kind_from_the_future'), 'line');
  assert.equal(markStyleFor(undefined), 'line');
});

// ── markHover ─────────────────────────────────────────────────────────────

test('markHover: joins the label, ref and verdict a mark carries', () => {
  assert.equal(
    markHover({ label: 'opus -> fable', ref: 'abc123', verdict: '+3.1 pp [0.4–5.8]' }),
    'opus -> fable\nabc123\n+3.1 pp [0.4–5.8]',
  );
  assert.equal(markHover({ label: 'deploy', ref: 'abc123' }), 'deploy\nabc123');
  assert.equal(markHover({ label: 'deploy', verdict: 'n 40 → 3' }), 'deploy\nn 40 → 3');
});

test('markHover: a mark that says nothing beyond its label has no hover', () => {
  assert.equal(markHover({ label: 'Verify-scope flip' }), null);
  assert.equal(markHover({ label: 'x', ref: null, verdict: '' }), null);
  assert.equal(markHover({ key: 0, fraction: 0.5, label: 'x', kind: 'regime' }), null);
});

// ── legendEntries ─────────────────────────────────────────────────────────

test('legendEntries: each style present appears once, in a fixed order', () => {
  const entries = legendEntries([
    { kind: 'alarm' },
    { kind: 'deploy' },
    { kind: 'model' },
    { kind: 'artefact' },
    { kind: 'prompt' },
    { kind: 'instrument' },
    { kind: 'turns' },
  ]);
  assert.deepEqual(
    entries.map(e => e.style),
    ['line', 'tick', 'rug', 'range', 'alarm'],
  );
  for (const entry of entries) {
    assert.equal(typeof entry.label, 'string');
    assert.ok(entry.label.length > 0, `${entry.style} has a label`);
  }
});

test('legendEntries: only the styles present are listed', () => {
  assert.deepEqual(
    legendEntries([{ kind: 'deploy' }, { kind: 'deploy' }]).map(e => e.style),
    ['rug'],
  );
  assert.deepEqual(legendEntries([]), []);
  assert.deepEqual(legendEntries(null), []);
});
