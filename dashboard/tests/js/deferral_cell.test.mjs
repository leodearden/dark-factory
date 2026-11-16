// Module-contract tests for deferral_cell.js: what the Tasks tab draws for a
// row's `deferral` key, and the per-project needs-human chip counted over it.
//
// The summary's fields are the server's (dashboard/src/dashboard/data/
// deferral_summary.py, covered by dashboard/tests/test_deferral_summary.py);
// this suite asserts only how the helper turns them into a cell. Auto-run by
// dashboard/tests/test_graph_layout_js.py's *.test.mjs glob.
//
// The module destructures window.DF_ENDPOINT_STALENESS at load, so the shim
// goes in first and the module is required after it, as index.html orders the
// real page (see task_row_cells.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/deferral_cell.js';
const EXPORT_NAMES = ['deferralCellState', 'deferralNeedsHumanChip'];

function loadDeferralCell() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  delete require.cache[require.resolve(MODULE_SPECIFIER)];
  return { cell: require(MODULE_SPECIFIER), window: win };
}

const { deferralCellState, deferralNeedsHumanChip } = loadDeferralCell().cell;

const CLOSABLE = {
  label: 'parked: carried by #900 (closable)',
  state: 'needs_human',
  age_seconds: 7200,
  needs_human: 'closable',
  detail: 'stamped by claude-task-1 at 2026-10-21 10:00 UTC; reason: folded into the carrier',
};
const HELD = {
  label: 'planning batch',
  state: 'held',
  age_seconds: 3600,
  needs_human: null,
  detail: 'stamped by planner at 2026-10-21 11:00 UTC',
};

function row(id, fields) {
  return { id, status: 'deferred', ...fields };
}

test('the module is dual-exported as window.DF_DEFERRAL_CELL and CommonJS', () => {
  const { cell, window: win } = loadDeferralCell();

  assert.deepEqual(Object.keys(cell).sort(), EXPORT_NAMES);
  assert.deepEqual(Object.keys(win.DF_DEFERRAL_CELL).sort(), EXPORT_NAMES);
});

test('deferralCellState: a row without its own deferral key draws nothing', () => {
  assert.equal(deferralCellState({ id: 'p/T-1', status: 'pending' }), null);
  assert.equal(deferralCellState(Object.create({ deferral: null })), null);
  assert.equal(deferralCellState(null), null);
  assert.equal(deferralCellState(undefined), null);
});

test('deferralCellState: an explicit null reads no deferral recorded and needs a human', () => {
  const cell = deferralCellState(row('p/T-2', { deferral: null }));

  assert.equal(cell.text, 'no deferral recorded');
  assert.equal(cell.age, null);
  assert.equal(cell.state, 'unrecorded');
  assert.equal(cell.needsHuman, true);
  assert.ok(cell.title.length > 0, 'the unrecorded hold must say why it is drawn');
});

test('deferralCellState: a summary draws its label, its age and its verdict', () => {
  const cell = deferralCellState(row('p/T-3', { deferral: CLOSABLE }));

  assert.equal(cell.text, CLOSABLE.label);
  assert.equal(cell.age, staleness.formatAge(CLOSABLE.age_seconds * 1000));
  assert.equal(cell.state, 'needs_human');
  assert.equal(cell.needsHuman, true);
  assert.ok(cell.title.includes(CLOSABLE.detail), cell.title);
  assert.ok(cell.title.includes('closable'), cell.title);
});

test('deferralCellState: a hold with no verdict needs no human and passes its state through', () => {
  const cell = deferralCellState(row('p/T-4', { deferral: HELD }));

  assert.equal(cell.needsHuman, false);
  assert.equal(cell.state, 'held');
  assert.equal(cell.title, HELD.detail);

  const unknown = deferralCellState(row('p/T-5', { deferral: { ...HELD, state: 'unknown' } }));
  assert.equal(unknown.state, 'unknown');
});

test('deferralCellState: an unknown age reads age unknown, never zero', () => {
  const cell = deferralCellState(row('p/T-6', { deferral: { ...CLOSABLE, age_seconds: null } }));

  assert.equal(cell.age, 'age unknown');
});

test('deferralNeedsHumanChip: counts null summaries and non-null verdicts only', () => {
  const rows = [
    { id: 'p/T-1', status: 'pending' },
    row('p/T-2', { deferral: HELD }),
    row('p/T-3', { deferral: null }),
    row('p/T-4', { deferral: CLOSABLE }),
    row('p/T-5', { deferral: { ...CLOSABLE, needs_human: 'legacy_unknown' } }),
  ];
  const chip = deferralNeedsHumanChip(rows);

  assert.equal(chip.count, 3);
  assert.equal(chip.text, '3 deferrals need a human');
  for (const id of ['p/T-3', 'p/T-4', 'p/T-5']) {
    assert.ok(chip.title.includes(id), `the chip title should name ${id}: ${chip.title}`);
  }
  assert.ok(!chip.title.includes('p/T-2'), chip.title);
});

test('deferralNeedsHumanChip: one hold reads in the singular', () => {
  const chip = deferralNeedsHumanChip([row('p/T-3', { deferral: null })]);

  assert.equal(chip.count, 1);
  assert.equal(chip.text, '1 deferral needs a human');
});

test('deferralNeedsHumanChip: nothing to count draws no chip', () => {
  assert.equal(deferralNeedsHumanChip([{ id: 'p/T-1' }, row('p/T-2', { deferral: HELD })]), null);
  assert.equal(deferralNeedsHumanChip([]), null);
  assert.equal(deferralNeedsHumanChip(null), null);
});
