// Module-contract tests for invariant_chip.js — the plain-JS decisions behind
// the Tasks tab's per-project claimant-invariant and recurrence-chain chips,
// fed by /api/v2/dashboard/invariants (docs/prds/claimant-invariant-detection.md
// E-1/D-2; docs/prds/recurring-deterministic-tasks.md R-D5/C-4). Surfaced in CI
// by dashboard/tests/test_graph_layout_js.py's `**/*.test.mjs` glob.
//
// The module resolves as CommonJS (no package.json), so it is default-imported
// and destructured, as in tasks_offline_banner.test.mjs.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import chip from '../../src/dashboard/static/redux/invariant_chip.js';

const { invariantChipFor, chainChipFor } = chip;

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
    chains: [],
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
    chains: null,
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

const LINK_STATUS_BY_STATE = {
  broken: 'done',
  overdue: 'pending',
  paused: 'blocked',
  scheduled: 'pending',
  ended: 'cancelled',
};
const AT = '2026-10-10T09:00:00+00:00';

function chain(key, state, extra = {}) {
  return {
    tag: 'master',
    key,
    state,
    link_id: 42,
    link_status: LINK_STATUS_BY_STATE[state],
    at: AT,
    pending_escalation_ids: [],
    ...extra,
  };
}

function chainRecord(project, chains) {
  return { ...okRecord(project), chains };
}

function occurrences(text, needle) {
  return text.split(needle).length - 1;
}

test('chain chip: unknown is not clean, so no record renders no chip', () => {
  assert.equal(chainChipFor(undefined, 'reify'), null);
  assert.equal(chainChipFor([], 'reify'), null);
  assert.equal(chainChipFor([chainRecord('dark-factory', [chain('a-job', 'broken')])], 'reify'), null);
});

test('chain chip: a blind root makes no chain claim, even beside a measured twin', () => {
  const blind = blindRecord('reify', 'tasks.db not found at /srv/reify/.taskmaster/tasks/tasks.db');
  assert.equal(chainChipFor([blind], 'reify'), null);
  assert.equal(chainChipFor([chainRecord('reify', [chain('a-job', 'broken')]), blind], 'reify'), null);
});

test('chain chip: a measured root with no chains has nothing to alarm on', () => {
  assert.equal(chainChipFor([okRecord('reify')], 'reify'), null);
});

test('chain chip: a broken chain alarms and is named, never absent', () => {
  const result = chainChipFor([chainRecord('reify', [chain('crashed-mint', 'broken', { link_id: 317 })])], 'reify');
  assert.equal(result.tone, 'bad');
  assert.match(result.text, /1 broken/);
  assert.ok(result.title.includes('crashed-mint'));
  assert.ok(result.title.includes('broken'));
  assert.ok(result.title.includes('#317'));
});

test('chain chip: an overdue chain alarms and carries its fire time', () => {
  const result = chainChipFor([chainRecord('reify', [chain('late-job', 'overdue')])], 'reify');
  assert.equal(result.tone, 'bad');
  assert.match(result.text, /1 overdue/);
  assert.ok(result.title.includes(AT));
});

test('chain chip: a paused chain warns and names its pinning escalation', () => {
  const paused = chain('failing-job', 'paused', { pending_escalation_ids: ['esc-9-1'] });
  const result = chainChipFor([chainRecord('reify', [paused])], 'reify');
  assert.equal(result.tone, 'warn');
  assert.match(result.text, /1 paused/);
  assert.ok(result.title.includes('esc-9-1'));
});

test('chain chip: a paused chain with no pending escalation says so', () => {
  const paused = chain('failing-job', 'paused', { pending_escalation_ids: [] });
  const result = chainChipFor([chainRecord('reify', [paused])], 'reify');
  assert.match(result.title, /no pending escalation/);
});

test('chain chip: an unknown escalation lookup is not reported as none', () => {
  const paused = chain('failing-job', 'paused', { pending_escalation_ids: null });
  const result = chainChipFor([chainRecord('reify', [paused])], 'reify');
  assert.match(result.title, /escalation lookup unavailable/);
  assert.doesNotMatch(result.title, /no pending escalation/);
});

test('chain chip: healthy chains are muted and counted by state', () => {
  const chains = [chain('a-job', 'scheduled'), chain('b-job', 'scheduled'), chain('c-job', 'ended')];
  const result = chainChipFor([chainRecord('reify', chains)], 'reify');
  assert.equal(result.tone, 'muted');
  assert.match(result.text, /2 scheduled/);
  assert.match(result.text, /1 ended/);
  assert.doesNotMatch(result.text, /broken|overdue|paused/);
});

test('chain chip: every non-zero state is named in a fixed order and every chain is listed once', () => {
  const keys = ['ended-job', 'scheduled-job', 'paused-job', 'overdue-job', 'broken-job'];
  const chains = keys.map((key) => chain(key, key.replace('-job', '')));
  const result = chainChipFor([chainRecord('reify', chains)], 'reify');
  const order = ['broken', 'overdue', 'paused', 'scheduled', 'ended'].map((state) => result.text.indexOf('1 ' + state));
  assert.ok(order.every((index) => index >= 0), result.text);
  assert.deepEqual([...order].sort((x, y) => x - y), order);
  for (const key of keys) assert.equal(occurrences(result.title, key), 1, key);
  assert.ok(result.title.includes(MEASURED_AT));
});

test('chain chip: a basename collision of two measured roots drops no chain', () => {
  const records = [
    chainRecord('reify', [chain('first-job', 'broken')]),
    chainRecord('reify', [chain('second-job', 'broken'), chain('third-job', 'scheduled')]),
  ];
  const result = chainChipFor(records, 'reify');
  assert.match(result.text, /2 broken/);
  assert.match(result.text, /1 scheduled/);
  for (const key of ['first-job', 'second-job', 'third-job']) assert.ok(result.title.includes(key), key);
});

test('chain chip: every tone is in the runtime_format.js vocabulary', () => {
  for (const state of ['broken', 'overdue', 'paused', 'scheduled', 'ended']) {
    const { tone } = chainChipFor([chainRecord('reify', [chain('a-job', state)])], 'reify');
    assert.ok(TONES.has(tone), `${state}: ${tone}`);
  }
});
