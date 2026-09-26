// Module-contract tests for recon_status.js — a plain-JS (no JSX/Babel)
// module holding the reconciliation run-status vocabulary and the pure
// counting logic the Recon tab (tabs.jsx::ReconTab) and the rail badge
// (app.jsx) derive from it. Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper, which
// globs `**/*.test.mjs` under this directory — no wrapper change needed).
//
// recon_status.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { reconRunCounts } from '...'` would come back undefined. We
// therefore default-import the module and destructure instead (mirrors
// task_status_counts.test.mjs / prd_grouping.test.mjs).
//
// The JSX WIRING half of this module's coverage — that app.jsx and
// tabs.jsx actually consume these functions rather than restating the
// vocabulary — lives in dashboard/tests/test_tab_recon.py. This file owns
// the executable behaviour; that one owns the call sites.
import { test } from 'node:test';
import assert from 'node:assert/strict';

import reconStatus from '../../src/dashboard/static/redux/recon_status.js';

const {
  RECON_RUN_STATES,
  RECON_RUN_IN_FLIGHT,
  RECON_RUN_SUCCESS,
  RECON_RUN_UNSUCCESSFUL,
  reconRunCounts,
  reconSuccessPct,
  reconAttentionCount,
  reconStatusTone,
} = reconStatus;

const EXPECTED_FUNCTION_NAMES = [
  'reconRunCounts',
  'reconSuccessPct',
  'reconAttentionCount',
  'reconStatusTone',
];

const EXPECTED_ARRAY_NAMES = [
  'RECON_RUN_STATES',
  'RECON_RUN_IN_FLIGHT',
  'RECON_RUN_SUCCESS',
  'RECON_RUN_UNSUCCESSFUL',
];

// The two spellings the dashboard used to test for and that the store has
// never once written — the defect this module exists to close.
const NEVER_WRITTEN = ['success', 'partial'];

// n runs all carrying `status`, ids unique across the concatenated list.
function runsWithStatus(status, n) {
  return Array.from({ length: n }, (_, i) => ({ id: `${status}-${i}`, status }));
}

test('default-imported module exposes the counting functions', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(
      typeof reconStatus[name],
      'function',
      `reconStatus.${name} should be a function`,
    );
  }
});

test('default-imported module exposes the vocabulary arrays', () => {
  for (const name of EXPECTED_ARRAY_NAMES) {
    assert.ok(
      Array.isArray(reconStatus[name]),
      `reconStatus.${name} should be an array`,
    );
    assert.ok(reconStatus[name].length > 0, `reconStatus.${name} should be non-empty`);
  }
});

// ---------------------------------------------------------------------------
// The vocabulary — one closed set, partitioned three ways.
// ---------------------------------------------------------------------------

test('the three partitions are a disjoint cover of RECON_RUN_STATES', () => {
  // Disjoint: no status may be counted into two buckets (which would let
  // the success rate exceed 100%). Covering: no status may be orphaned
  // (which is exactly how 'interrupted' went uncounted in the rail badge).
  const union = [...RECON_RUN_IN_FLIGHT, ...RECON_RUN_SUCCESS, ...RECON_RUN_UNSUCCESSFUL];

  assert.equal(
    union.length,
    new Set(union).size,
    'a status appears in more than one partition',
  );
  assert.deepEqual([...union].sort(), [...RECON_RUN_STATES].sort());
});

test('the partitions place each status where the journal semantics put it', () => {
  // Exact contents, so this test plus the disjoint-cover one above together
  // pin RECON_RUN_STATES to precisely the four statuses
  // fused-memory/src/fused_memory/reconciliation/journal.py writes — the
  // `status` column defaults to 'running' and `complete_run` is called with
  // exactly 'completed' | 'failed' | 'interrupted' across the whole package.
  // Neither retired spelling can re-enter the vocabulary without failing one
  // of the two.
  assert.deepEqual(RECON_RUN_IN_FLIGHT, ['running']);
  assert.deepEqual(RECON_RUN_SUCCESS, ['completed']);
  // 'interrupted' is a run the process died in the middle of — it did not
  // succeed, so it belongs in the rate's denominator-only side (K in
  // M/(M+K)) alongside 'failed'. It is nonetheless NOT an alarm; see the
  // reconAttentionCount tests below.
  assert.deepEqual([...RECON_RUN_UNSUCCESSFUL].sort(), ['failed', 'interrupted']);
});

// ---------------------------------------------------------------------------
// reconRunCounts — one pass, every row landing in exactly one bucket.
// ---------------------------------------------------------------------------

test('reconRunCounts: buckets a mixed window, counting interrupted as unsuccessful', () => {
  const runs = [
    ...runsWithStatus('running', 3),
    ...runsWithStatus('completed', 12),
    ...runsWithStatus('failed', 4),
    ...runsWithStatus('interrupted', 6),
  ];

  const counts = reconRunCounts(runs);

  assert.equal(counts.total, 25);
  assert.equal(counts.inFlight, 3);
  assert.equal(counts.success, 12);
  assert.equal(counts.failed, 4);
  assert.equal(counts.interrupted, 6);
  assert.equal(counts.unsuccessful, 10, 'failed + interrupted');
  assert.equal(counts.terminal, 22);
  assert.equal(counts.unknown, 0);
});

test('reconRunCounts: the buckets partition the window exactly', () => {
  // Every row lands in exactly one of inFlight / success / unsuccessful /
  // unknown, and terminal is derived rather than independently counted —
  // so no arithmetic here can disagree with any other.
  const runs = [
    ...runsWithStatus('running', 7),
    ...runsWithStatus('completed', 11),
    ...runsWithStatus('failed', 2),
    ...runsWithStatus('interrupted', 5),
    ...runsWithStatus('some-future-status', 3),
  ];

  const counts = reconRunCounts(runs);

  assert.equal(
    counts.inFlight + counts.success + counts.unsuccessful + counts.unknown,
    counts.total,
    'a run was lost or double-counted',
  );
  assert.equal(counts.unsuccessful, counts.failed + counts.interrupted);
  assert.equal(counts.terminal, counts.success + counts.unsuccessful);
});

test('reconRunCounts: an out-of-vocabulary status lands in `unknown` and nowhere else', () => {
  // The defect being fixed WAS a silent client-side discard. A status the
  // store grows later must show up as a number an operator can see, not
  // vanish into a filter that matches nothing.
  for (const status of [...NEVER_WRITTEN, '', 'RUNNING', 'compleded']) {
    const counts = reconRunCounts(runsWithStatus(status, 4));

    assert.equal(counts.unknown, 4, `'${status}' should be counted as unknown`);
    assert.equal(counts.total, 4);
    assert.equal(counts.inFlight, 0, `'${status}' must not be counted as in-flight`);
    assert.equal(counts.success, 0, `'${status}' must not be counted as a success`);
    assert.equal(counts.failed, 0, `'${status}' must not be counted as a failure`);
    assert.equal(counts.unsuccessful, 0, `'${status}' must not be counted as unsuccessful`);
    assert.equal(counts.terminal, 0, `'${status}' must not enter the rate denominator`);
  }
});

test('reconRunCounts: a run with a missing status is unknown, not silently dropped', () => {
  const counts = reconRunCounts([{ id: 'a' }, { id: 'b', status: 'completed' }]);

  assert.equal(counts.total, 2);
  assert.equal(counts.success, 1);
  assert.equal(counts.unknown, 1);
});

test('reconRunCounts: empty / undefined / null input yields an all-zero shape', () => {
  // The tiles render before recon data has necessarily arrived; throwing
  // here would blank the whole Recon tab.
  const allZero = {
    total: 0, inFlight: 0, success: 0, failed: 0, interrupted: 0,
    unsuccessful: 0, terminal: 0, unknown: 0,
  };
  for (const input of [[], undefined, null]) {
    assert.deepEqual(
      reconRunCounts(input),
      allZero,
      `reconRunCounts(${JSON.stringify(input)}) should be all-zero, not a throw`,
    );
  }
});

// ---------------------------------------------------------------------------
// reconSuccessPct — M / (M + K) over TERMINAL runs only.
// ---------------------------------------------------------------------------

test('reconSuccessPct: is M/(M+K) over terminal runs, independent of in-flight N', () => {
  // The task's user-observable arithmetic. Dividing by the whole window
  // (what the old code did) would make the rate dip every time
  // reconciliation got busy — a second, subtler version of the same bug.
  const M = 12, K = 8;
  const expected = Math.round(M / (M + K) * 100);

  for (const N of [0, 1, 5, 40]) {
    const counts = reconRunCounts([
      ...runsWithStatus('completed', M),
      ...runsWithStatus('failed', K - 3),
      ...runsWithStatus('interrupted', 3),
      ...runsWithStatus('running', N),
    ]);

    assert.equal(counts.terminal, M + K);
    assert.equal(
      reconSuccessPct(counts),
      expected,
      `${N} in-flight runs must not move the rate off ${expected}%`,
    );
  }
});

test('reconSuccessPct: rounds the quotient', () => {
  const counts = reconRunCounts([
    ...runsWithStatus('completed', 2),
    ...runsWithStatus('failed', 1),
  ]);

  assert.equal(counts.terminal, 3);
  assert.equal(reconSuccessPct(counts), 67, 'Math.round(2/3*100)');
});

test('reconSuccessPct: 100% and 0% are both reachable and distinct', () => {
  assert.equal(reconSuccessPct(reconRunCounts(runsWithStatus('completed', 9))), 100);
  assert.equal(reconSuccessPct(reconRunCounts(runsWithStatus('failed', 9))), 0);
  assert.equal(reconSuccessPct(reconRunCounts(runsWithStatus('interrupted', 9))), 0);
});

test('reconSuccessPct: null when there are no terminal runs', () => {
  // The tile renders '—' for null. Returning 0 instead would make "no run
  // has finished yet" indistinguishable from "every run failed" — the very
  // distinction the 0%-forever tile destroyed.
  assert.equal(reconSuccessPct(reconRunCounts([])), null);
  assert.equal(
    reconSuccessPct(reconRunCounts(runsWithStatus('running', 6))),
    null,
    'a window of nothing but in-flight runs has no rate to report, not 0%',
  );
  assert.equal(
    reconSuccessPct(reconRunCounts(runsWithStatus('partial', 6))),
    null,
    'an all-unknown window has no rate to report either',
  );
});

test('reconSuccessPct: null / undefined counts do not throw', () => {
  assert.equal(reconSuccessPct(null), null);
  assert.equal(reconSuccessPct(undefined), null);
});

// ---------------------------------------------------------------------------
// reconAttentionCount — the rail badge's number. Failures + unrecognised.
// ---------------------------------------------------------------------------

test('reconAttentionCount: counts failures and unrecognised statuses only', () => {
  const counts = reconRunCounts([
    ...runsWithStatus('failed', 5),
    ...runsWithStatus('some-future-status', 2),
    ...runsWithStatus('completed', 30),
    ...runsWithStatus('running', 4),
  ]);

  assert.equal(reconAttentionCount(counts), 7, '5 failed + 2 unrecognised');
});

test('reconAttentionCount: an interrupted run is not an alarm', () => {
  // journal.py records 'interrupted' when a run's process died, and this
  // fleet is restarted routinely — so interrupted rows are a standing
  // background population on a HEALTHY fleet. Counting them here would sit
  // the rail badge permanently nonzero, trading the old under-count for a
  // permanent false alarm. They are still unsuccessful for the RATE, which
  // is a different question; the reconSuccessPct tests above pin that.
  const counts = reconRunCounts(runsWithStatus('interrupted', 12));

  assert.equal(counts.unsuccessful, 12, 'still not a success');
  assert.equal(reconAttentionCount(counts), 0, 'but nothing to investigate');
});

test('reconAttentionCount: a status the store grows later is visible from the rail', () => {
  // The defect this module fixes was a consumer reading a status vocabulary
  // that had moved on without it. `unknown` is the only runtime signal that
  // it has moved again, and the rail badge is the one surface an operator
  // sees without opening the Recon tab — so it must not be dropped here.
  const counts = reconRunCounts(runsWithStatus('aborted', 3));

  assert.equal(counts.unknown, 3);
  assert.equal(reconAttentionCount(counts), 3);
});

test('reconAttentionCount: a quiet healthy window is zero, and null does not throw', () => {
  assert.equal(reconAttentionCount(reconRunCounts(runsWithStatus('completed', 50))), 0);
  assert.equal(reconAttentionCount(reconRunCounts([])), 0);
  assert.equal(reconAttentionCount(null), 0);
  assert.equal(reconAttentionCount(undefined), 0);
});

// ---------------------------------------------------------------------------
// reconStatusTone — the per-row badge class. Classifies; never relabels.
// ---------------------------------------------------------------------------

test('reconStatusTone: maps each store status to its badge tone', () => {
  assert.equal(reconStatusTone('completed'), 'ok');
  assert.equal(reconStatusTone('failed'), 'bad');
  // 'interrupted' is unsuccessful for the RATE but is a recovered restart
  // artefact (journal.get_interrupted_runs feeds the startup sweep), not a
  // failure to investigate — so it is warn, not bad. Two orthogonal
  // questions, answered separately.
  assert.equal(reconStatusTone('interrupted'), 'warn');
  assert.equal(reconStatusTone('running'), 'info');
});

test('reconStatusTone: every vocabulary member has a tone, and failed is alone in bad', () => {
  const tones = RECON_RUN_STATES.map(reconStatusTone);

  assert.ok(!tones.includes('muted'), 'no vocabulary member may render as unknown');
  assert.deepEqual(
    RECON_RUN_STATES.filter(s => reconStatusTone(s) === 'bad'),
    ['failed'],
  );
});

test('reconStatusTone: an unrecognised status is muted, not silently ok', () => {
  for (const status of [...NEVER_WRITTEN, '', 'weird', undefined, null]) {
    assert.equal(
      reconStatusTone(status),
      'muted',
      `an unrecognised status (${JSON.stringify(status)}) must render muted`,
    );
  }
});

test('reconStatusTone returns a tone from its closed set, never anything else', () => {
  // This pins the RETURN SET only. That every one of these tokens has a
  // `.badge.<tone>` rule — the property that keeps a badge from rendering
  // unstyled — is asserted against the real stylesheet by
  // dashboard/tests/test_tab_recon.py::TestBadgeTonesAreStyled, which reads
  // the tones out of recon_status.js rather than out of a second list.
  const KNOWN_TONES = ['ok', 'bad', 'warn', 'info', 'muted'];
  for (const status of [...RECON_RUN_STATES, 'anything-else']) {
    assert.ok(
      KNOWN_TONES.includes(reconStatusTone(status)),
      `reconStatusTone('${status}') returned a tone outside ${KNOWN_TONES.join('/')}`,
    );
  }
});
