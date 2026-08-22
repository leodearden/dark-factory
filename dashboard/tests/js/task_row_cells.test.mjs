// Module-contract tests for task_row_cells.js — the pure render DECISIONS
// behind the two cells of a task row's claim column: the stranded badge at its
// three sites (tab_tasks.jsx renderNode, tab_tasks.jsx TaskDetail, tabs.jsx
// OrchTab) and the agent cell it sits beside at two of them.
//
// WHAT THIS SUITE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT. Only the RENDER
// decision: which descriptor comes back, with what label/title/margin, and when
// nothing renders at all. The `stranded` WIRE field — how the server computes
// the strand verdict from the claim columns — is already covered behaviourally
// by dashboard/tests/test_task_claimant_projection.py and the _build_task_row
// tests, and is not restated here. Two suites asserting one contract is how
// they drift.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py is the pytest
// wrapper; its `**/*.test.mjs` glob auto-discovers this file, so no wrapper
// change was needed for it).
//
// task_row_cells.js resolves as CommonJS (`module.exports = <object>`)
// because this repo has no package.json. Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { strandBadgeState } from '...'` would come back undefined. We
// therefore default-import and destructure (mirrors prd_grouping.test.mjs /
// graph_layout.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import cells from '../../src/dashboard/static/redux/task_row_cells.js';

const { strandBadgeState, agentCellState, STRAND_TITLE } = cells;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/task_row_cells.js';
const EXPECTED_FUNCTION_NAMES = ['strandBadgeState', 'agentCellState'];
const EXPECTED_EXPORT_NAMES = [...EXPECTED_FUNCTION_NAMES, 'STRAND_TITLE'];

test('default-imported module exposes the task-row render decisions', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof cells[name], 'function', `cells.${name} should be a function`);
  }
  assert.equal(typeof cells.STRAND_TITLE, 'string', 'cells.STRAND_TITLE should be a string');
});

test('module also assigns window.DF_TASK_ROW_CELLS (browser dual-export)', () => {
  // Shim a bare browser-like global before requiring the module fresh via
  // CommonJS require, so the module body's `if (typeof window !== 'undefined')`
  // branch executes against our shim. The top-level `import` above has already
  // populated the shared require.cache (node's ESM loader delegates CJS
  // resolution to the CJS loader), so the cache entry must be busted to force a
  // fresh execution against the now-shimmed window.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_TASK_ROW_CELLS, 'window.DF_TASK_ROW_CELLS was not set');
    assert.deepEqual(
      Object.keys(globalThis.window.DF_TASK_ROW_CELLS).sort(),
      EXPECTED_EXPORT_NAMES.slice().sort(),
    );
    assert.deepEqual(Object.keys(required).sort(), EXPECTED_EXPORT_NAMES.slice().sort());
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(typeof globalThis.window.DF_TASK_ROW_CELLS[name], 'function');
    }
  } finally {
    delete globalThis.window;
  }
});

// ---------------------------------------------------------------------------
// strandBadgeState — renders NOTHING unless the server said `stranded`
// ---------------------------------------------------------------------------

test('strandBadgeState: renders nothing for a task that is not stranded', () => {
  assert.equal(strandBadgeState({ stranded: false }), null);
});

test('strandBadgeState: renders nothing when the strand verdict is absent', () => {
  // Unknown is not "not stranded", but neither is it a badge: the affirmative
  // claim is the only one this badge ever makes.
  assert.equal(strandBadgeState({ stranded: undefined }), null);
  assert.equal(strandBadgeState({}), null);
});

test('strandBadgeState: tolerates a null/undefined task without throwing', () => {
  // The task rows render before task data has necessarily arrived; throwing
  // here would blank the whole surrounding table rather than one cell.
  assert.equal(strandBadgeState(null), null);
  assert.equal(strandBadgeState(undefined), null);
});

test('strandBadgeState: a stranded task gets the full badge descriptor', () => {
  assert.deepEqual(strandBadgeState({ stranded: true }), {
    cls: 'badge bad',
    label: 'stranded',
    title: STRAND_TITLE,
    marginLeft: 6,
  });
});

test('strandBadgeState: compact mode swaps the label for the glyph and drops the margin', () => {
  const compact = strandBadgeState({ stranded: true }, { compact: true });

  assert.notEqual(compact, null);
  assert.equal(compact.label, '⚠');
  assert.equal(compact.cls, 'badge bad');
  assert.equal(compact.title, STRAND_TITLE);
  // Absent, not undefined-valued: the compact site (tab_tasks.jsx renderNode)
  // renders no `style` attribute at all, and a caller spreading this descriptor
  // must not emit one.
  assert.ok(
    !('marginLeft' in compact),
    'compact descriptor must omit marginLeft entirely, not carry it as undefined',
  );
});

test('strandBadgeState: marginLeft is overridable, because the two full sites disagree', () => {
  // tab_tasks.jsx TaskDetail renders marginLeft 6; tabs.jsx OrchTab renders 4.
  // Parameterised rather than unified so the extraction stays byte-for-byte
  // behaviour-preserving at both call sites — unifying them is a visual change
  // and would belong to a different task.
  assert.equal(strandBadgeState({ stranded: true }, { marginLeft: 4 }).marginLeft, 4);
  assert.equal(strandBadgeState({ stranded: true }).marginLeft, 6);
});

// ---------------------------------------------------------------------------
// INDEPENDENCE from `agent` — the whole point of the feature
// ---------------------------------------------------------------------------

test('strandBadgeState: the badge is byte-identical whether or not an agent is present', () => {
  // `agent` is worktree presence, not liveness: a warm-lane worktree persists
  // after the agent dies, so a stranded task very often still carries one. If
  // this descriptor varied with `agent`, the stranded task that most needs the
  // badge — the one whose worktree outlived its agent — would render exactly
  // like a live one.
  assert.deepEqual(
    strandBadgeState({ stranded: true, agent: 'claude-task-9' }),
    strandBadgeState({ stranded: true }),
  );
  assert.deepEqual(
    strandBadgeState({ stranded: true, agent: 'claude-task-9' }, { compact: true }),
    strandBadgeState({ stranded: true }, { compact: true }),
  );
});

test('strandBadgeState: a NON-stranded task with no agent still renders nothing', () => {
  // The converse direction. The badge tracks the strand verdict, never agent
  // presence — an unassigned pending task is not stranded.
  assert.equal(strandBadgeState({ stranded: false, agent: null }), null);
  assert.equal(strandBadgeState({ agent: null }), null);
  assert.equal(strandBadgeState({ agent: undefined }), null);
});

// ---------------------------------------------------------------------------
// agentCellState — the cell the badge sits beside, and stays distinct from
// ---------------------------------------------------------------------------

test('agentCellState: an assigned agent renders its own name, unmuted', () => {
  assert.deepEqual(agentCellState({ agent: 'x' }), { text: 'x', muted: false });
});

test('agentCellState: no agent falls back to the default "unassigned" placeholder', () => {
  assert.deepEqual(agentCellState({}), { text: 'unassigned', muted: true });
  assert.deepEqual(agentCellState({ agent: null }), { text: 'unassigned', muted: true });
  assert.deepEqual(agentCellState({ agent: '' }), { text: 'unassigned', muted: true });
  assert.deepEqual(agentCellState(null), { text: 'unassigned', muted: true });
});

test('agentCellState: the placeholder is a parameter, because the two sites disagree', () => {
  // tab_tasks.jsx TaskDetail renders the word 'unassigned'; tabs.jsx OrchTab
  // renders an em-dash. Both are pinned so the extraction cannot silently
  // change either one.
  assert.deepEqual(agentCellState({}, { placeholder: '—' }), { text: '—', muted: true });
  assert.deepEqual(agentCellState({ agent: 'x' }, { placeholder: '—' }), { text: 'x', muted: false });
});

test('agentCellState and strandBadgeState share no field values — they are distinct surfaces', () => {
  const badge = strandBadgeState({ stranded: true, agent: 'x' });
  const cell = agentCellState({ agent: 'x' });

  // Different keys entirely: nothing about the badge is derivable from the
  // cell, or vice versa.
  const badgeKeys = new Set(Object.keys(badge));
  for (const key of Object.keys(cell)) {
    assert.ok(
      !badgeKeys.has(key),
      `agent cell key "${key}" also appears on the strand badge descriptor — the ` +
        'two surfaces must stay structurally distinct',
    );
  }

  assert.notEqual(badge.label, cell.text);
  assert.notEqual(strandBadgeState({ stranded: true }, { compact: true }).label, cell.text);
});

// ---------------------------------------------------------------------------
// STRAND_TITLE — exported once so the three sites cannot drift apart
// ---------------------------------------------------------------------------

test('STRAND_TITLE is one non-empty exported string constant', () => {
  assert.equal(typeof STRAND_TITLE, 'string');
  assert.ok(STRAND_TITLE.length > 0, 'STRAND_TITLE must not be empty');
});

test('STRAND_TITLE is the single source for every site that renders the badge', () => {
  // All three render sites hand-copied this string before the extraction, which
  // is the drift risk the constant removes. Every descriptor shape must return
  // the identical reference, not a re-spelled copy.
  assert.equal(strandBadgeState({ stranded: true }).title, STRAND_TITLE);
  assert.equal(strandBadgeState({ stranded: true }, { compact: true }).title, STRAND_TITLE);
  assert.equal(strandBadgeState({ stranded: true }, { marginLeft: 4 }).title, STRAND_TITLE);
});
