// Module-contract tests for runtime_format.js — a plain-JS (no JSX/Babel)
// module holding the pure offline-'—' degradation formatters used to render
// warm-lane runtime fields (loops/attempts/started/lane/phase/lane_state) in
// the Orchestrators tab (OrchTab in tabs.jsx) and the Tasks tab's TaskDetail
// (tab_tasks.jsx). Run via `node --test` (see
// dashboard/tests/test_graph_layout_js.py for the pytest wrapper that
// surfaces this suite in CI via its `**/*.test.mjs` glob — no wrapper change
// needed for this new file).
//
// runtime_format.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { rtCell } from '...'` would come back undefined. We therefore
// default-import the module and destructure instead (mirrors
// prd_grouping.test.mjs / graph_layout.test.mjs).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import fmt from '../../src/dashboard/static/redux/runtime_format.js';

const { rtCell, rtAge, rtProbe, rtProbeSummary } = fmt;

const MODULE_SPECIFIER = '../../src/dashboard/static/redux/runtime_format.js';
const EXPECTED_FUNCTION_NAMES = ['rtCell', 'rtAge', 'rtProbe', 'rtProbeSummary'];

test('default-imported module exposes the runtime-format functions', () => {
  for (const name of EXPECTED_FUNCTION_NAMES) {
    assert.equal(typeof fmt[name], 'function', `fmt.${name} should be a function`);
  }
});

test('module also assigns window.DF_RUNTIME_FMT (browser dual-export)', () => {
  // Shim a bare browser-like global before requiring the module fresh via
  // CommonJS require, so the module body's `if (typeof window !== 'undefined')`
  // branch executes against our shim.
  globalThis.window = {};
  try {
    const require = createRequire(import.meta.url);
    // Node's ESM loader resolves a CommonJS module (no package.json/type in
    // this repo) by delegating to the CJS loader and populating the shared
    // require.cache — so by the time this test runs, the top-level `import
    // fmt from ...` above has ALREADY cached this exact file. A plain
    // require() here would return that cached module.exports without
    // re-running the module body, meaning the dual-export line would never
    // see our globalThis.window shim. Busting the cache entry forces a
    // fresh execution against the now-shimmed window.
    const resolved = require.resolve(MODULE_SPECIFIER);
    delete require.cache[resolved];
    const required = require(MODULE_SPECIFIER);

    assert.ok(globalThis.window.DF_RUNTIME_FMT, 'window.DF_RUNTIME_FMT was not set');

    // The fresh require() and the top-level import() produce two distinct
    // API object instances (separate module executions), so we compare
    // structurally — same set of exported names, each a function — rather
    // than asserting reference/deep equality against the ESM-imported `fmt`.
    assert.deepEqual(
      Object.keys(globalThis.window.DF_RUNTIME_FMT).sort(),
      EXPECTED_FUNCTION_NAMES.slice().sort(),
    );
    assert.deepEqual(Object.keys(required).sort(), EXPECTED_FUNCTION_NAMES.slice().sort());
    for (const name of EXPECTED_FUNCTION_NAMES) {
      assert.equal(typeof globalThis.window.DF_RUNTIME_FMT[name], 'function');
    }
  } finally {
    delete globalThis.window;
  }
});

// ---------------------------------------------------------------------------
// rtCell — v == null (covers both null and undefined, i.e. the
// offline/per-task-read-error shape) -> em-dash; any other value (including
// an honest 0) passes through unchanged.
// ---------------------------------------------------------------------------

test('rtCell: null renders as the em-dash', () => {
  assert.equal(rtCell(null), '—');
});

test('rtCell: undefined renders as the em-dash', () => {
  assert.equal(rtCell(undefined), '—');
});

test('rtCell: an honest zero is preserved, not dashed', () => {
  assert.equal(rtCell(0), 0);
});

test('rtCell: a positive number passes through unchanged', () => {
  assert.equal(rtCell(3), 3);
});

test('rtCell: a lane string passes through unchanged', () => {
  assert.equal(rtCell('_lane-7'), '_lane-7');
});

test('rtCell: a lane_state string passes through unchanged', () => {
  assert.equal(rtCell('assigned'), 'assigned');
});

// ---------------------------------------------------------------------------
// rtAge — null (or undefined) -> em-dash; otherwise `${minutes}m` (an honest
// 0 renders as "0m", not dashed).
// ---------------------------------------------------------------------------

test('rtAge: null renders as the em-dash', () => {
  assert.equal(rtAge(null), '—');
});

test('rtAge: undefined renders as the em-dash', () => {
  assert.equal(rtAge(undefined), '—');
});

test('rtAge: zero renders as an honest "0m", not dashed', () => {
  assert.equal(rtAge(0), '0m');
});

test('rtAge: a positive age renders as "<minutes>m"', () => {
  assert.equal(rtAge(14), '14m');
});

test('rtAge: a fractional-minute value is rounded to the nearest whole minute', () => {
  // Guards against upstream float/precision drift (the documented contract
  // is an integer minute count, but rtAge should not render a raw decimal
  // like "3.5m" if that contract ever slips).
  assert.equal(rtAge(3.5), '4m');
  assert.equal(rtAge(3.4), '3m');
});

// ---------------------------------------------------------------------------
// rtProbe — the per-row probe-status descriptor (task 3517). Turns a row's
// `runtime_status` (emitted by active_tasks._runtime_fields, originating in
// task_runtime._probe_one) into {label, hint, tone}, so an operator sees WHY
// the runtime cells are dashed instead of only seeing the dashes.
// ---------------------------------------------------------------------------

const DEGRADED_STATUSES = ['not_configured', 'unreachable', 'deadline_exceeded', 'unknown'];
const TONES = ['muted', 'warn', 'bad'];

test('rtProbe: the healthy status renders nothing', () => {
  assert.equal(rtProbe('ok'), null);
});

test('rtProbe: every degraded status yields a usable descriptor', () => {
  for (const status of DEGRADED_STATUSES) {
    const d = rtProbe(status);
    assert.ok(d, `rtProbe(${status}) should return a descriptor`);
    assert.equal(typeof d.label, 'string');
    assert.ok(d.label.length > 0, `${status} label should be non-empty`);
    assert.equal(typeof d.hint, 'string');
    assert.ok(d.hint.length > 0, `${status} hint should be non-empty`);
    assert.ok(TONES.includes(d.tone), `${status} tone ${d.tone} not in ${TONES}`);
  }
});

test('rtProbe: the four degraded labels are pairwise distinct', () => {
  // Separability IS the feature — identical labels would reproduce the
  // blank-cell ambiguity this task exists to remove.
  const labels = DEGRADED_STATUSES.map((s) => rtProbe(s).label);
  assert.equal(new Set(labels).size, labels.length, `labels not distinct: ${labels}`);
});

test('rtProbe: deadline_exceeded points at the dashboard, unreachable at the orchestrator', () => {
  const deadline = rtProbe('deadline_exceeded').hint.toLowerCase();
  assert.ok(
    deadline.includes('dashboard') || deadline.includes('probe'),
    `deadline hint should name the dashboard/probe, got: ${deadline}`,
  );
  const unreachable = rtProbe('unreachable').hint.toLowerCase();
  assert.ok(
    unreachable.includes('orchestrator'),
    `unreachable hint should name the orchestrator, got: ${unreachable}`,
  );
});

test('rtProbe: unknown/garbage input falls back to the unknown descriptor', () => {
  // Never throw and never return null for a non-'ok' value: an unrenderable
  // status must still tell the operator we do not know, not silently vanish
  // into a blank cell.
  const expected = rtProbe('unknown');
  for (const bogus of [null, undefined, 'garbage', '', 42, {}]) {
    assert.deepEqual(rtProbe(bogus), expected, `rtProbe(${String(bogus)})`);
  }
});

// ---------------------------------------------------------------------------
// rtProbeSummary — aggregates task rows into the Tasks-tab banner. Derived
// frontend-side from ACTIVE_TASKS rows (deliberately not a new payload key).
// ---------------------------------------------------------------------------

const row = (project, runtime_status) => ({ project, runtime_status });

test('rtProbeSummary: nothing degraded returns null', () => {
  assert.equal(rtProbeSummary([]), null);
  assert.equal(rtProbeSummary([row('a', 'ok'), row('b', 'ok')]), null);
});

test('rtProbeSummary: groups degraded projects by status, deduped', () => {
  const summary = rtProbeSummary([
    row('alpha', 'unreachable'),
    row('alpha', 'unreachable'),
    row('alpha', 'unreachable'),
    row('beta', 'deadline_exceeded'),
    row('gamma', 'ok'),
  ]);
  assert.ok(summary);
  // Each project counted ONCE regardless of how many rows it contributes.
  assert.deepEqual(summary.byStatus.unreachable, ['alpha']);
  assert.deepEqual(summary.byStatus.deadline_exceeded, ['beta']);
  assert.ok(!('ok' in summary.byStatus), 'healthy projects must not be listed');
});

test('rtProbeSummary: a row with no runtime_status is treated as ok', () => {
  // Back-compat with an older cached payload that predates the field.
  assert.equal(rtProbeSummary([{ project: 'a' }, { project: 'b' }]), null);
  const summary = rtProbeSummary([{ project: 'a' }, row('b', 'unreachable')]);
  assert.deepEqual(summary.byStatus.unreachable, ['b']);
});

test('rtProbeSummary: selfInflicted only when >=2 probed projects ALL timed out', () => {
  const summary = rtProbeSummary([
    row('alpha', 'deadline_exceeded'),
    row('beta', 'deadline_exceeded'),
  ]);
  assert.equal(summary.selfInflicted, true);
  assert.equal(summary.probedCount, 2);
  assert.ok(
    summary.text.toLowerCase().includes('dashboard'),
    `self-inflicted text should name the dashboard, got: ${summary.text}`,
  );
});

test('rtProbeSummary: one probed project timing out is NOT self-inflicted', () => {
  // Degenerate: equally consistent with that one orchestrator being down, so
  // blaming the dashboard would be the same unfounded diagnosis in reverse.
  const summary = rtProbeSummary([row('alpha', 'deadline_exceeded')]);
  assert.equal(summary.selfInflicted, false);
});

test('rtProbeSummary: a healthy project alongside a timeout is NOT self-inflicted', () => {
  const summary = rtProbeSummary([
    row('alpha', 'deadline_exceeded'),
    row('beta', 'ok'),
  ]);
  assert.equal(summary.selfInflicted, false);
});

test('rtProbeSummary: mixed degraded reasons are NOT self-inflicted', () => {
  // A real outage is per-project — one of these really is unreachable.
  const summary = rtProbeSummary([
    row('alpha', 'deadline_exceeded'),
    row('beta', 'unreachable'),
  ]);
  assert.equal(summary.selfInflicted, false);
});

test('rtProbeSummary: never-probed projects do not count toward "all probed"', () => {
  // not_configured projects were never probed at all, so counting them would
  // make the heuristic fire on a deployment where only one orchestrator is
  // even configured.
  const summary = rtProbeSummary([
    row('alpha', 'deadline_exceeded'),
    row('beta', 'not_configured'),
    row('gamma', 'not_configured'),
  ]);
  assert.equal(summary.probedCount, 1);
  assert.equal(summary.selfInflicted, false);
});
