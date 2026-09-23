// Module-contract tests for burndown_bands.js — the pure render DECISIONS
// behind the Burndown tab's stacked status-mix bands, their legend, and the
// concurrency-parity banner (tabs.jsx BurnTab, both the aggregate and the
// per-project views).
//
// WHAT THIS SUITE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT. Only the RENDER
// decision: which bands exist in which order, which colour each gets, and
// whether the parity banner draws at all. The WIRE shapes — that
// shape_burndown emits in_progress_live / in_progress_stranded /
// concurrency_cap, and that the server computes parity_alarm correctly — are
// already covered behaviourally by dashboard/tests/test_redux_api.py and
// test_burndown_parity_alarm.py, and are not restated here.
//
// COLOURS ARE INJECTED, never read off a global (the prd_grouping.js
// convention: orderPrdGroups takes computeTiers as a parameter rather than
// reaching for window.DF_GRAPH_LAYOUT). So these tests pass a SENTINEL palette
// and assert against the sentinels. Pinning literal oklch strings here would
// duplicate charts.jsx's palette into a second place, which is the drift this
// module exists to remove, and would make a pure re-theme fail a test that has
// nothing to say about themes. What matters is that the right palette SLOT
// reaches the right band, and that the slots stay distinct.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file — no wrapper change needed).
// burndown_bands.js resolves as CommonJS (no package.json in this repo), and
// node's cjs-module-lexer cannot see exports assigned from a variable, so we
// default-import and destructure rather than using named imports.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import bands from '../../src/dashboard/static/redux/burndown_bands.js';

const { burndownStacks, burndownLegend, parityBannerState } = bands;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/burndown_bands.js';
const EXPECTED_FUNCTION_NAMES = ['burndownStacks', 'burndownLegend', 'parityBannerState'];

// Sentinel palette — deliberately not colours. If an implementation ever
// hard-codes a real oklch string instead of reading the injected slot, these
// assertions fail loudly rather than coincidentally matching.
const CP = {
  ok: 'C_OK',
  accent: 'C_ACCENT',
  stranded: 'C_STRANDED',
  bad: 'C_BAD',
  warn: 'C_WARN',
};

// A burndown block with distinct array identities per field, so the tests can
// assert each band is sourced from its OWN same-named field by reference —
// index-shifting one band onto another's series is the failure mode the
// per-project call site (pb.*) is exposed to.
function mkBlock(extra) {
  return {
    labels: ['d1', 'd2'],
    done: [1, 2],
    in_progress_live: [3, 4],
    in_progress_stranded: [5, 6],
    blocked: [7, 8],
    pending: [9, 10],
    ...(extra || {}),
  };
}

test('default-imported module exposes the burndown render decisions', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof bands[name], 'function', `bands.${name} should be a function`);
  }
});

test('module also assigns window.DF_BURNDOWN_BANDS (browser dual-export)', () => {
  // Shim a bare browser-like global, then bust the require cache the top-level
  // `import` above already populated, so the module body's
  // `if (typeof window !== 'undefined')` branch runs against our shim.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_BURNDOWN_BANDS, 'window.DF_BURNDOWN_BANDS was not set');
    assert.deepEqual(
      Object.keys(globalThis.window.DF_BURNDOWN_BANDS).sort(),
      EXPECTED_FUNCTION_NAMES.slice().sort(),
    );
    assert.deepEqual(Object.keys(required).sort(), EXPECTED_FUNCTION_NAMES.slice().sort());
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(typeof globalThis.window.DF_BURNDOWN_BANDS[name], 'function');
    }
  } finally {
    delete globalThis.window;
  }
});

// ---------------------------------------------------------------------------
// burndownStacks — the five bands, in order, each on its own series
// ---------------------------------------------------------------------------

test('burndownStacks: returns exactly the five bands in their fixed order', () => {
  const stacks = burndownStacks(mkBlock(), CP);

  assert.deepEqual(stacks.map(s => s.key), [
    'done',
    'in_progress_live',
    'in_progress_stranded',
    'blocked',
    'pending',
  ]);
});

test('burndownStacks: live and stranded in-progress are SEPARATE stacked bands', () => {
  // The feature itself. Merging them back into one band is the regression this
  // assertion exists to catch — a stranded task would then be invisible,
  // stacked inside the same colour as a healthy one.
  const stacks = burndownStacks(mkBlock(), CP);
  const live = stacks.filter(s => s.key === 'in_progress_live');
  const stranded = stacks.filter(s => s.key === 'in_progress_stranded');

  assert.equal(live.length, 1, 'expected exactly one in_progress_live band');
  assert.equal(stranded.length, 1, 'expected exactly one in_progress_stranded band');
});

test('burndownStacks: the undivided in_progress band NEVER appears alongside its parts', () => {
  // Pins the invariant the call-site comment states: stacking the whole beside
  // its parts would draw a total no census ever produced. The server
  // guarantees live + stranded sum to in_progress, so emitting all three
  // double-counts every in-progress task.
  const stacks = burndownStacks(mkBlock({ in_progress: [99, 99] }), CP);

  assert.ok(
    stacks.every(s => s.key !== 'in_progress'),
    `an undivided 'in_progress' band was stacked beside its parts: ${stacks.map(s => s.key).join(', ')}`,
  );
});

test('burndownStacks: live and stranded get DISTINCT colours, from their own palette slots', () => {
  const stacks = burndownStacks(mkBlock(), CP);
  const live = stacks.find(s => s.key === 'in_progress_live');
  const stranded = stacks.find(s => s.key === 'in_progress_stranded');

  assert.notEqual(
    live.color,
    stranded.color,
    'live and stranded bands share a colour — separating the bands achieves ' +
      'nothing if they render indistinguishably',
  );
  assert.equal(live.color, CP.accent);
  assert.equal(stranded.color, CP.stranded);
});

test('burndownStacks: all five band colours are pairwise distinct', () => {
  const stacks = burndownStacks(mkBlock(), CP);
  const colors = stacks.map(s => s.color);

  assert.equal(
    new Set(colors).size,
    colors.length,
    `two bands share a colour and would be unreadable when stacked: ${colors.join(', ')}`,
  );
});

test('burndownStacks: each band is sourced from its OWN same-named block field', () => {
  // By identity, not by value: this is what makes the per-project call site
  // (which passes pb.*, a different block from the aggregate b.*) index-safe.
  // A band wired to the wrong field would still render a plausible chart.
  const block = mkBlock();
  const stacks = burndownStacks(block, CP);
  const byKey = Object.fromEntries(stacks.map(s => [s.key, s]));

  assert.equal(byKey.done.values, block.done);
  assert.equal(byKey.in_progress_live.values, block.in_progress_live);
  assert.equal(byKey.in_progress_stranded.values, block.in_progress_stranded);
  assert.equal(byKey.blocked.values, block.blocked);
  assert.equal(byKey.pending.values, block.pending);
});

test('burndownStacks: palette slots reach the bands they name', () => {
  const stacks = burndownStacks(mkBlock(), CP);
  const byKey = Object.fromEntries(stacks.map(s => [s.key, s.color]));

  assert.deepEqual(byKey, {
    done: CP.ok,
    in_progress_live: CP.accent,
    in_progress_stranded: CP.stranded,
    blocked: CP.bad,
    pending: CP.warn,
  });
});

test('burndownStacks: tolerates a null/undefined block without throwing', () => {
  // BurnTab renders before the first burndown payload has necessarily arrived.
  // The band SET is structural and must survive that — an empty chart, not a
  // blanked tab. This is what makes the `const b = block || {}` arm real: delete
  // it and this test throws instead of passing.
  for (const empty of [null, undefined]) {
    const stacks = burndownStacks(empty, CP);

    assert.equal(stacks.length, 5, `a ${empty} block should still define five bands`);
    assert.deepEqual(stacks.map(s => s.key), [
      'done',
      'in_progress_live',
      'in_progress_stranded',
      'blocked',
      'pending',
    ]);
    // No series to draw, but the palette slots still resolve: an absent block
    // must not also cost the colours.
    assert.ok(stacks.every(s => s.values === undefined), 'an absent block should yield no series');
    assert.equal(stacks.find(s => s.key === 'in_progress_stranded').color, CP.stranded);
  }
});

test('burndownStacks: tolerates a missing palette without throwing', () => {
  // Symmetric to the block arm above: `const cp = palette || {}`. A caller that
  // has not resolved CP yet gets colourless bands, not an exception.
  for (const nopalette of [null, undefined]) {
    const stacks = burndownStacks(mkBlock(), nopalette);

    assert.equal(stacks.length, 5);
    assert.ok(stacks.every(s => s.color === undefined), 'an absent palette should yield no colours');
    // The series still arrive — losing the palette must not also lose the data.
    assert.deepEqual(stacks.find(s => s.key === 'pending').values, [9, 10]);
  }
});

// ---------------------------------------------------------------------------
// burndownLegend — the key to the bands above, which must agree with them
// ---------------------------------------------------------------------------

test('burndownLegend: five entries, labelled for a reader rather than for the wire', () => {
  const legend = burndownLegend(CP);

  assert.deepEqual(legend.map(e => e.label), ['done', 'live', 'stranded', 'blocked', 'pending']);
});

test('burndownLegend: legend colours match the stack colours pairwise, in order', () => {
  // The real seam. Before the extraction this legend was two verbatim-
  // duplicated array literals (the aggregate view and the per-project view),
  // each free to drift from the bands it explains — a legend that disagrees
  // with its chart is worse than no legend, because it is believed.
  const legend = burndownLegend(CP);
  const stacks = burndownStacks(mkBlock(), CP);

  assert.equal(legend.length, stacks.length);
  for (let i = 0; i < stacks.length; i++) {
    assert.equal(
      legend[i].color,
      stacks[i].color,
      `legend entry ${i} ("${legend[i].label}") does not carry the colour of ` +
        `band "${stacks[i].key}" it explains`,
    );
  }
});

test('burndownLegend: tolerates a missing palette without throwing', () => {
  // The legend is derived from burndownStacks, so it inherits that tolerance —
  // asserted here rather than assumed, because the derivation is the very thing
  // a future edit might unpick.
  for (const nopalette of [undefined, null]) {
    const legend = burndownLegend(nopalette);

    assert.equal(legend.length, 5);
    assert.deepEqual(legend.map(e => e.label), ['done', 'live', 'stranded', 'blocked', 'pending']);
    assert.ok(legend.every(e => e.color === undefined), 'an absent palette should yield no colours');
  }
});

// ---------------------------------------------------------------------------
// parityBannerState — draws ONLY on a server-computed alarm
// ---------------------------------------------------------------------------

test('parityBannerState: renders nothing without a block', () => {
  assert.equal(parityBannerState(null, null), null);
  assert.equal(parityBannerState(undefined, null), null);
});

test('parityBannerState: renders nothing when the alarm is false or absent', () => {
  // Both directions matter. A banner that draws when parity_alarm is false is
  // a false alarm about a false alarm — it accuses the operator's fleet of
  // breaching a cap it never breached.
  assert.equal(parityBannerState({ parity_alarm: false }, null), null);
  assert.equal(parityBannerState({}, null), null);
});

test('parityBannerState: exposes the verdict when the alarm fires', () => {
  const state = parityBannerState(
    { parity_alarm: true, parity_peak: 43, parity_cap: 24, parity_breach_count: 7 },
    null,
  );

  assert.notEqual(state, null);
  assert.equal(state.peak, 43);
  assert.equal(state.cap, 24);
  // The breach count is asserted through the rendered text, not through a
  // separate field: `text` is what the banner actually shows, and every field
  // this descriptor returns is one the call site interpolates.
  assert.ok(state.text.includes('7 snapshots over'), `breach count missing from text: ${state.text}`);
  assert.deepEqual(Object.keys(state).sort(), ['cap', 'peak', 'text']);
});

test('parityBannerState: a missing breach count reads as zero rather than undefined', () => {
  // The `?? 0` arm. Pinned on the rendered text, which is where a regression
  // would actually be seen: without the fallback the banner reads "undefined
  // snapshots over" at the operator.
  const state = parityBannerState({ parity_alarm: true, parity_peak: 43, parity_cap: 24 }, null);

  assert.ok(state.text.includes('0 snapshots'), `expected a zero count, got: ${state.text}`);
  assert.ok(!state.text.includes('undefined'), `undefined leaked into the banner: ${state.text}`);
});

test('parityBannerState: pluralises the snapshot count', () => {
  const text = n =>
    parityBannerState({ parity_alarm: true, parity_peak: 43, parity_cap: 24, parity_breach_count: n }, null).text;

  assert.ok(text(1).includes('1 snapshot over'), `singular form wrong: ${text(1)}`);
  assert.ok(text(2).includes('2 snapshots over'), `plural form wrong: ${text(2)}`);
  // Zero takes the plural, English-style — "0 snapshot over" reads as a typo.
  assert.ok(
    parityBannerState({ parity_alarm: true, parity_peak: 43, parity_cap: 24 }, null).text.includes('0 snapshots over'),
    'a zero count should take the plural form',
  );
});

test('parityBannerState: names the offending projects only when there are any', () => {
  // The aggregate view passes b.parity_projects (the breaching subset); the
  // per-project view passes null, because naming a project inside its own
  // panel says nothing. Asserted on the rendered text: the project suffix is
  // folded into it rather than exposed as a field no call site reads.
  const state = projects =>
    parityBannerState(
      { parity_alarm: true, parity_peak: 43, parity_cap: 24, parity_breach_count: 2 },
      projects,
    ).text;

  assert.equal(state(['a', 'b']), ' · 2 snapshots over · a, b');
  // Both empty forms must render the bare count with NO dangling separator —
  // a trailing ' · ' would read as a truncated project list.
  assert.equal(state(null), ' · 2 snapshots over');
  assert.equal(state([]), ' · 2 snapshots over');
});

test('parityBannerState: the trailing text carries the project list when present', () => {
  const state = parityBannerState(
    { parity_alarm: true, parity_peak: 43, parity_cap: 24, parity_breach_count: 2 },
    ['a', 'b'],
  );

  assert.ok(state.text.endsWith(' · a, b'), `expected the project list to close the text: ${state.text}`);
});
