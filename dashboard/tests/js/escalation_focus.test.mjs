// Module-contract tests for escalation_focus.js — resolving a cross-tab focus
// `{queue, id}` (the memory-evals escalation link, handed over by app.jsx) to
// one row of the ESCALATIONS payload. tab_escalations.jsx is
// `type="text/babel"` and cannot run under node; its wiring is pinned
// structurally in test_tab_escalations.py.
//
// THE FIXTURE IS THE MEASURED LIVE COLLISION (2026-10-06): esc-3169-1 sits in
// both the dark-factory orchestrator queue and the reconciliation queue, and
// both rows carry project 'dark-factory'. So neither the id alone nor
// (project, id) addresses one row; (queue, id) does.
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const SPECIFIER = '../../src/dashboard/static/redux/escalation_focus.js';

function loadEscalationFocus() {
  const win = {};
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  delete require.cache[require.resolve(SPECIFIER)];
  return { api: require(SPECIFIER), window: win };
}

const { api: focusApi, window: loadedWindow } = loadEscalationFocus();
const { findEscalationRow } = focusApi;

// ── Fixtures ────────────────────────────────────────────────────────────────

const ORCH_QUEUE = '/home/leo/src/dark-factory';
const RECON_QUEUE = 'reconciliation';
const SHARED_ID = 'esc-3169-1';
const NOTHING = Object.freeze({ row: null, candidates: [] });

function collisionPayload() {
  const orchRow = { id: SHARED_ID, project: 'dark-factory', task_id: '3169', level: 0 };
  const reconRow = { id: SHARED_ID, project: 'dark-factory', task_id: '3169', level: 1 };
  const neighbour = { id: 'esc-12-1', project: 'dark-factory', task_id: '12', level: 0 };
  return {
    orchRow,
    reconRow,
    escalations: {
      subsections: [
        {
          id: ORCH_QUEUE, label: 'dark-factory', kind: 'orchestrator',
          escalations: [neighbour, orchRow],
        },
        {
          id: RECON_QUEUE, label: 'fused-memory', kind: 'reconciliation',
          escalations: [reconRow],
        },
      ],
    },
  };
}

// ── The module ──────────────────────────────────────────────────────────────

test('exports: the module publishes exactly its API, on module.exports and window.DF_ESCALATION_FOCUS', () => {
  assert.deepEqual(Object.keys(focusApi), ['findEscalationRow']);
  assert.equal(typeof findEscalationRow, 'function');
  // The browser half of the dual export: tab_escalations.jsx destructures this
  // global at module scope with no fallback.
  assert.equal(loadedWindow.DF_ESCALATION_FOCUS, focusApi);
});

// ── (a)/(b) the queue picks the row ─────────────────────────────────────────

test('(a) a reconciliation focus returns the reconciliation row, not the first id hit', () => {
  const { escalations, reconRow } = collisionPayload();

  const found = findEscalationRow(escalations, { queue: RECON_QUEUE, id: SHARED_ID });

  assert.equal(found.row, reconRow);
  assert.deepEqual(found.candidates, [reconRow]);
  assert.equal(found.candidates[0], reconRow);
});

test('(b) an orchestrator-queue focus returns the orchestrator row', () => {
  const { escalations, orchRow } = collisionPayload();

  const found = findEscalationRow(escalations, { queue: ORCH_QUEUE, id: SHARED_ID });

  assert.equal(found.row, orchRow);
  assert.equal(found.candidates.length, 1);
  assert.equal(found.candidates[0], orchRow);
});

// ── (c) a miss is nothing ───────────────────────────────────────────────────

test('(c) an unknown id or an unknown queue yields no row and no candidates', () => {
  const { escalations } = collisionPayload();

  assert.deepEqual(findEscalationRow(escalations, { queue: RECON_QUEUE, id: 'esc-9999-1' }), NOTHING);
  assert.deepEqual(findEscalationRow(escalations, { queue: 'no-such-queue', id: SHARED_ID }), NOTHING);
  // The id lives in another queue only: still a miss, never a cross-queue hit.
  assert.deepEqual(findEscalationRow(escalations, { queue: RECON_QUEUE, id: 'esc-12-1' }), NOTHING);
});

// ── (d) a tie elects nothing ────────────────────────────────────────────────

test('(d) two rows sharing an id in ONE queue give every candidate and no row', () => {
  const { escalations, reconRow } = collisionPayload();
  const twin = { ...reconRow, level: 2 };
  escalations.subsections[1].escalations.push(twin);

  const found = findEscalationRow(escalations, { queue: RECON_QUEUE, id: SHARED_ID });

  assert.equal(found.row, null);
  assert.equal(found.candidates.length, 2);
  assert.equal(found.candidates[0], reconRow);
  assert.equal(found.candidates[1], twin);
});

test('(d) the match is strict: a numeric id does not match its string form', () => {
  const escalations = {
    subsections: [{ id: RECON_QUEUE, escalations: [{ id: 3169 }] }],
  };

  assert.deepEqual(findEscalationRow(escalations, { queue: RECON_QUEUE, id: '3169' }), NOTHING);
});

// ── (e) an unusable focus touches nothing ───────────────────────────────────

const UNUSABLE_FOCI = [
  ['null', null],
  ['undefined', undefined],
  ['a bare id string', SHARED_ID],
  ['a number', 3169],
  ['queue absent', { id: SHARED_ID }],
  ['queue null', { queue: null, id: SHARED_ID }],
  ['queue empty', { queue: '', id: SHARED_ID }],
  ['queue not a string', { queue: 7, id: SHARED_ID }],
  ['id absent', { queue: RECON_QUEUE }],
  ['id null', { queue: RECON_QUEUE, id: null }],
  ['id empty', { queue: RECON_QUEUE, id: '' }],
  ['id not a string', { queue: RECON_QUEUE, id: 3169 }],
];

for (const [label, focus] of UNUSABLE_FOCI) {
  test(`(e) an unusable focus (${label}) resolves to nothing, even beside an id-less row`, () => {
    const idless = { id: null, project: 'dark-factory' };
    const undefinedId = { project: 'dark-factory' };
    const escalations = {
      subsections: [
        { id: RECON_QUEUE, escalations: [idless, undefinedId] },
        { id: '', escalations: [idless] },
        { id: null, escalations: [idless] },
      ],
    };

    assert.deepEqual(findEscalationRow(escalations, focus), NOTHING);
  });
}

// ── (f) a payload that has not arrived ──────────────────────────────────────

test('(f) a null payload, or missing or empty subsections, is tolerated', () => {
  const focus = { queue: RECON_QUEUE, id: SHARED_ID };

  assert.deepEqual(findEscalationRow(null, focus), NOTHING);
  assert.deepEqual(findEscalationRow(undefined, focus), NOTHING);
  assert.deepEqual(findEscalationRow({}, focus), NOTHING);
  assert.deepEqual(findEscalationRow({ subsections: [] }, focus), NOTHING);
  assert.deepEqual(findEscalationRow({ subsections: [{ id: RECON_QUEUE }] }, focus), NOTHING);
});
