// Module-contract tests for invariant_chip.js — the plain-JS decision behind
// the Tasks tab's per-project claimant-invariant chip, fed by
// /api/v2/dashboard/invariants (docs/prds/claimant-invariant-detection.md
// E-1/D-2). Surfaced in CI by dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob.
//
// The module resolves as CommonJS (no package.json), so it is default-imported
// and destructured, as in tasks_offline_banner.test.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import chip from '../../src/dashboard/static/redux/invariant_chip.js';

const { invariantChipFor } = chip;

const MEASURED_AT = '2026-10-10T12:00:00+00:00';
const TONES = new Set(['muted', 'warn', 'bad']);

function violation(taskId, ledgered) {
  return {
    tag: 'master',
    task_id: taskId,
    status: 'done',
    claimant_run_id: 'run-a/s/pid=1',
    heartbeat_at: null,
    ledgered,
  };
}

function okRecord(project, violations = [], rowsScanned = 6642) {
  return {
    root: '/srv/' + project,
    project,
    status: 'ok',
    error: null,
    rows_scanned: rowsScanned,
    measured_at: MEASURED_AT,
    terminal_tier: {
      count: violations.length,
      unledgered_count: violations.filter((v) => !v.ledgered).length,
      violations,
    },
    hygiene_tier: { count: 0, stale_count: 0 },
  };
}

function blindRecord(project, error) {
  return {
    root: '/srv/' + project,
    project,
    status: 'blind',
    error,
    rows_scanned: null,
    measured_at: MEASURED_AT,
    terminal_tier: null,
    hygiene_tier: null,
  };
}

test('unknown is not clean: no record renders no chip', () => {
  assert.equal(invariantChipFor(undefined, 'reify'), null);
  assert.equal(invariantChipFor({ project: 'reify' }, 'reify'), null);
  assert.equal(invariantChipFor([], 'reify'), null);
  assert.equal(invariantChipFor([okRecord('dark-factory')], 'reify'), null);
});

test('a blind root is loud and makes no count claim', () => {
  const error = 'tasks.db not found at /srv/reify/.taskmaster/tasks/tasks.db';
  const result = invariantChipFor([blindRecord('reify', error)], 'reify');
  assert.equal(result.tone, 'bad');
  assert.match(result.text, /blind/);
  assert.doesNotMatch(result.text, /\d/);
  assert.ok(result.title.includes(error));
});

test('a measured zero is shown, muted, with its provenance', () => {
  const result = invariantChipFor([okRecord('reify', [], 8394)], 'reify');
  assert.equal(result.tone, 'muted');
  assert.match(result.text, /0/);
  assert.ok(result.title.includes('8394'));
  assert.ok(result.title.includes(MEASURED_AT));
});

test('unledgered violations alarm and are named in the title', () => {
  const record = okRecord('reify', [violation(101, false), violation(102, true), violation(103, false)]);
  const result = invariantChipFor([record], 'reify');
  assert.equal(result.tone, 'bad');
  assert.match(result.text, /3/);
  assert.match(result.text, /2 unledgered/);
  assert.ok(result.title.includes('101'));
  assert.ok(result.title.includes('103'));
});

test('sanctioned (ledgered) mints are reported, not alarmed', () => {
  const record = okRecord('reify', [violation(101, true), violation(102, true)]);
  const result = invariantChipFor([record], 'reify');
  assert.equal(result.tone, 'muted');
  assert.match(result.text, /2/);
  assert.doesNotMatch(result.text, /unledgered/);
});

test('a basename collision never drops a root: blind wins', () => {
  const records = [okRecord('reify'), blindRecord('reify', 'OperationalError: no such table: tasks')];
  const result = invariantChipFor(records, 'reify');
  assert.equal(result.tone, 'bad');
  assert.match(result.text, /blind/);
  assert.ok(result.title.includes('no such table'));
});

test('a basename collision of two measured roots sums their counts', () => {
  const records = [okRecord('reify', [violation(1, true)]), okRecord('reify', [violation(2, true), violation(3, true)])];
  const result = invariantChipFor(records, 'reify');
  assert.equal(result.tone, 'muted');
  assert.match(result.text, /3/);
});

test('every tone is in the runtime_format.js vocabulary', () => {
  const branches = {
    blind: [blindRecord('reify', 'OperationalError: no such table: tasks')],
    'measured zero': [okRecord('reify')],
    ledgered: [okRecord('reify', [violation(1, true)])],
    unledgered: [okRecord('reify', [violation(1, false)])],
  };
  for (const [branch, invariants] of Object.entries(branches)) {
    const { tone } = invariantChipFor(invariants, 'reify');
    assert.ok(TONES.has(tone), `${branch}: ${tone}`);
  }
});
