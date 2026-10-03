// Module-contract tests for pins_recovery.js — the pure render DECISIONS
// behind BOTH pins_recovery surfaces in the dashboard:
//
//   1. the per-row PINNING chip beside the 6h+ breach badge, in
//      tab_escalation_analytics.jsx (pinningBadgeState);
//   2. the "pinning" StatTile fed by the open-items reduction, in
//      tab_escalations.jsx (pinningSummary).
//
// (The task description placed the chip in tab_escalations.jsx. It is not
// there — that file's pins_recovery surface is the StatTile. Both real
// surfaces are covered here rather than the one that was named.)
//
// WHAT THIS SUITE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT. Only the RENDER
// decision: whether a chip draws, what its title says, and which items the
// summary counts. The WIRE contract — that the server emits pins_recovery /
// pins_recovery_task_ids at all, and omits the key when it could not compute
// the annotation — is already covered behaviourally by
// escalation/tests/test_pins_recovery_annotation.py and
// test_escalation_analytics.py, and is not restated here.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file — no wrapper change needed).
// pins_recovery.js resolves as CommonJS (no package.json in this repo), and
// node's cjs-module-lexer cannot see exports assigned from a variable, so we
// default-import and destructure rather than using named imports.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import pins from '../../src/dashboard/static/redux/pins_recovery.js';

const { pinningBadgeState, pinningSummary } = pins;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/pins_recovery.js';
const EXPECTED_FUNCTION_NAMES = ['pinningBadgeState', 'pinningSummary'];

test('default-imported module exposes the pins-recovery render decisions', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof pins[name], 'function', `pins.${name} should be a function`);
  }
});

test('module also assigns window.DF_PINS_RECOVERY (browser dual-export)', () => {
  // Shim a bare browser-like global, then bust the require cache the top-level
  // `import` above already populated, so the module body's
  // `if (typeof window !== 'undefined')` branch runs against our shim.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_PINS_RECOVERY, 'window.DF_PINS_RECOVERY was not set');
    assert.deepEqual(
      Object.keys(globalThis.window.DF_PINS_RECOVERY).sort(),
      EXPECTED_FUNCTION_NAMES.slice().sort(),
    );
    assert.deepEqual(Object.keys(required).sort(), EXPECTED_FUNCTION_NAMES.slice().sort());
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(typeof globalThis.window.DF_PINS_RECOVERY[name], 'function');
    }
  } finally {
    delete globalThis.window;
  }
});

// ---------------------------------------------------------------------------
// pinningBadgeState — THE THREE-STATE UNKNOWN CONTRACT IS THE HEADLINE
//
// The backend emits three distinguishable states, and the render must honour
// all three: `pins_recovery: true` (computed, pins something),
// `pins_recovery: false` (computed, pins nothing), and the key ABSENT
// (could not compute — that project's escalation MCP was unreadable, or the
// record carried no annotation). Absent is NOT false.
// ---------------------------------------------------------------------------

test('pinningBadgeState: an ABSENT annotation renders nothing at all', () => {
  // The headline. "Unknown" must draw nothing — never a negated chip. A "does
  // not pin recovery" badge over an unclassified record is a claim nobody
  // made, and it is worse than silence because an operator would act on it.
  const state = pinningBadgeState({});

  assert.equal(
    state,
    null,
    'an item with no pins_recovery key must render NOTHING — not a negated ' +
      'or "not pinning" badge, which would assert a verdict the backend ' +
      'explicitly declined to compute',
  );
});

test('pinningBadgeState: a computed-false annotation renders nothing', () => {
  assert.equal(pinningBadgeState({ pins_recovery: false }), null);
});

test('pinningBadgeState: tolerates a null/undefined item without throwing', () => {
  assert.equal(pinningBadgeState(null), null);
  assert.equal(pinningBadgeState(undefined), null);
});

test('pinningBadgeState: there is exactly ONE badge shape, and it is affirmative', () => {
  // Guards against a future "negated arm" being added back. Every input that
  // produces a badge at all produces the same affirmative label; every other
  // input produces null. No third rendering exists.
  const inputs = [
    {},
    { pins_recovery: false },
    { pins_recovery: undefined },
    { pins_recovery: true },
    { pins_recovery: true, pins_recovery_task_ids: ['1'] },
    null,
  ];
  const labels = new Set(inputs.map(i => pinningBadgeState(i)).filter(Boolean).map(s => s.label));

  assert.deepEqual([...labels], ['PINNING'], `expected one affirmative label only, got: ${[...labels].join(', ')}`);
});

test('pinningBadgeState: a pinning item gets the chip, naming the task it blocks', () => {
  const state = pinningBadgeState({ pins_recovery: true, pins_recovery_task_ids: ['3543'] });

  assert.notEqual(state, null);
  assert.equal(state.cls, 'badge bad');
  assert.equal(state.label, 'PINNING');
  assert.equal(typeof state.title, 'string');
  assert.ok(state.title.includes('3543'), `title should name the pinned task: ${state.title}`);
  assert.ok(state.title.includes('PINNING'), `title should say PINNING: ${state.title}`);
  assert.ok(state.title.includes('redispatched'), `title should explain the consequence: ${state.title}`);
});

test('pinningBadgeState: multiple pinned tasks are joined into one readable list', () => {
  const state = pinningBadgeState({ pins_recovery: true, pins_recovery_task_ids: ['1', '2'] });

  assert.ok(state.title.includes('1, 2'), `expected a joined id list: ${state.title}`);
});

test('pinningBadgeState: still draws when the ids are missing, WITHOUT a dangling empty slot', () => {
  // Degradation guard. The annotation and the id list are separate fields, so
  // `pins_recovery: true` with no ids is reachable — and the chip must still
  // draw, because the pin itself is the operator-relevant fact. What must NOT
  // happen is the title degrading into "recovery of task  — this escalation",
  // an empty slot that reads as a rendering bug and hides the real message.
  for (const item of [
    { pins_recovery: true },
    { pins_recovery: true, pins_recovery_task_ids: [] },
    { pins_recovery: true, pins_recovery_task_ids: null },
  ]) {
    const state = pinningBadgeState(item);

    assert.notEqual(state, null, `expected a badge for ${JSON.stringify(item)}`);
    assert.equal(state.label, 'PINNING');
    assert.ok(
      !state.title.includes('task  '),
      `title has a dangling empty id slot (double space): ${state.title}`,
    );
    assert.ok(
      !/task\s*$/.test(state.title),
      `title ends on a dangling "task" with nothing after it: ${state.title}`,
    );
    assert.ok(
      !/task\s+—/.test(state.title),
      `title has an empty id slot before the em-dash: ${state.title}`,
    );
    assert.ok(state.title.includes('redispatched'), `title should still explain itself: ${state.title}`);
  }
});

// ---------------------------------------------------------------------------
// pinningSummary — the StatTile reduction over the live open-items array
// ---------------------------------------------------------------------------

test('pinningSummary: an empty queue summarises to zeroes', () => {
  assert.deepEqual(pinningSummary([]), { count: 0, pinnedTaskCount: 0 });
});

test('pinningSummary: tolerates a null/undefined item list', () => {
  // The tile renders before the analytics payload has necessarily arrived.
  assert.deepEqual(pinningSummary(null), { count: 0, pinnedTaskCount: 0 });
  assert.deepEqual(pinningSummary(undefined), { count: 0, pinnedTaskCount: 0 });
});

test('pinningSummary: only affirmatively-pinning items are counted', () => {
  // An item whose annotation is ABSENT falls out of the count entirely — it is
  // counted as neither pinning nor not-pinning. `item.pins_recovery === false`
  // would have counted an unclassified item as "does not pin", which is the
  // three-state violation this guards.
  const summary = pinningSummary([
    {},                                                              // unknown
    { pins_recovery: false },                                        // computed, pins nothing
    { pins_recovery: true, pins_recovery_task_ids: ['7'] },          // pinning
  ]);

  assert.equal(summary.count, 1);
  assert.equal(summary.pinnedTaskCount, 1);
});

test('pinningSummary: TRUTHINESS TRAP — a raw unnormalised [] must not be counted', () => {
  // `[]` is truthy in JS. The server normalises with `bool(pinned)`
  // (escalation_analytics.py), so a well-formed payload never carries a bare
  // list here — but pinning the client-side guard is what stops the two layers
  // each assuming the other did the normalising. An empty list means "pins
  // nothing", and counting it would over-report the tile.
  const summary = pinningSummary([{ pins_recovery: [] }]);

  assert.deepEqual(
    summary,
    { count: 0, pinnedTaskCount: 0 },
    'an unnormalised empty list is truthy in JS and would over-count the tile',
  );
});

test('pinningSummary: pinned task ids are UNIONED and de-duplicated across items', () => {
  // Two escalations pinning the same task means one blocked task, not two.
  const summary = pinningSummary([
    { pins_recovery: true, pins_recovery_task_ids: ['7'] },
    { pins_recovery: true, pins_recovery_task_ids: ['7'] },
  ]);

  assert.equal(summary.count, 2, 'both escalations are still counted');
  assert.equal(summary.pinnedTaskCount, 1, 'but they block one task between them');
});

test('pinningSummary: numeric and string task ids collapse to one task', () => {
  // Task ids arrive as both across the projects' payloads; a Set of raw values
  // would count 7 and '7' as two distinct blocked tasks.
  const summary = pinningSummary([
    { pins_recovery: true, pins_recovery_task_ids: [7] },
    { pins_recovery: true, pins_recovery_task_ids: ['7'] },
  ]);

  assert.equal(summary.pinnedTaskCount, 1);
});

test('pinningSummary: a pinning item with no ids is counted, but blocks no named task', () => {
  // Mirrors the chip's degradation case: the pin is real even when the id list
  // did not survive, so it counts toward the tile's value while contributing
  // nothing to the blocked-task hint.
  const summary = pinningSummary([
    { pins_recovery: true },
    { pins_recovery: true, pins_recovery_task_ids: [] },
  ]);

  assert.equal(summary.count, 2);
  assert.equal(summary.pinnedTaskCount, 0);
});
