// Module-contract tests for endpoint_staleness.js — the pure decision module
// behind the dashboard's PER-ENDPOINT staleness indicator (task 4884, #4791).
//
// Run via `node --test`; auto-discovered by the `**/*.test.mjs` glob in
// dashboard/tests/test_graph_layout_js.py, so this new file needs no pytest
// wrapper change (same as tasks_offline_banner.test.mjs before it).
//
// endpoint_staleness.js has no package.json in the repo, so it resolves as
// CommonJS (`module.exports = <object>`). Node's cjs-module-lexer cannot
// statically detect named exports assigned from a variable, so
// `import { staleNoticesForTab } from '...'` would come back undefined. We
// therefore default-import the module and destructure instead (mirrors
// tasks_offline_banner.test.mjs / task_status_counts.test.mjs). A plain
// static ESM import is SAFE here — unlike data.js, this module touches no
// browser global at load; its `window.DF_ENDPOINT_STALENESS` assignment is
// guarded by `typeof window !== 'undefined'`.
//
// WHY THIS IS A TESTED MODULE AND NOT JSX. The 2026-08-27 incident ran 19.8h
// with the dashboard serving a fully-rendered UI whose numbers had stopped
// advancing: every tab kept its last-good payload (by design — see
// data.js::refreshOne, "keep the prior values so the UI does not blank out")
// and NOTHING on screen said the values were hours old. The decision of when
// an endpoint is stale enough to say so — and of what to say when it has
// never succeeded at all — is exactly the kind of judgement that is only
// greppable in JSX but executable here.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const {
  STALE_FAILURE_THRESHOLD,
  TAB_ENDPOINTS,
  staleEntryFor,
  formatAge,
  staleNoticesForTab,
} = staleness;

const REDUX_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../src/dashboard/static/redux',
);

// A representative endpoint that only ONE tab consumes, so a notice for it
// can never be confused with a different tab's fan-in.
const MERGE_PATH = '/api/v2/dashboard/merge-queue';
const TASKS_PATH = '/api/v2/dashboard/tasks';

const NOW = 1_700_000_000_000;

// Builds a `__stale`-shaped map: path -> {failures, lastSuccessAt}, exactly
// what data.js publishes under window.DF_DATA.__stale.
function staleMap(entries) {
  const out = {};
  for (const [p, entry] of Object.entries(entries)) out[p] = entry;
  return out;
}

test('module exposes its documented surface', () => {
  assert.equal(typeof staleNoticesForTab, 'function');
  assert.equal(typeof staleEntryFor, 'function');
  assert.equal(typeof formatAge, 'function');
  assert.equal(typeof STALE_FAILURE_THRESHOLD, 'number');
  assert.ok(Number.isInteger(STALE_FAILURE_THRESHOLD) && STALE_FAILURE_THRESHOLD >= 2,
    `threshold must be an integer >= 2 so a single blip cannot trip it, got ${STALE_FAILURE_THRESHOLD}`);
  assert.equal(typeof TAB_ENDPOINTS, 'object');
});

// ── (a) healthy tab ────────────────────────────────────────────────────────

test('a healthy tab (failures: 0) produces no notice', () => {
  const notices = staleNoticesForTab({
    tab: 'merge',
    stale: staleMap({ [MERGE_PATH]: { failures: 0, lastSuccessAt: NOW - 3000 } }),
    now: NOW,
  });
  assert.deepEqual(notices, []);
});

test('an absent entry (never polled yet) produces no notice', () => {
  // The very first render happens before any fetch resolves, so __stale is
  // `{}`. Claiming staleness there would fire on every page load.
  assert.deepEqual(staleNoticesForTab({ tab: 'merge', stale: {}, now: NOW }), []);
  assert.deepEqual(staleNoticesForTab({ tab: 'merge' }), []);
  assert.deepEqual(staleNoticesForTab({}), []);
  assert.deepEqual(staleNoticesForTab(), []);
});

// ── (b) a single blip must not trip it ─────────────────────────────────────

test('a single blip does not trip the indicator (#4791 acceptance 2)', () => {
  // One failed poll is the routine case: the journal for the incident window
  // shows isolated 2.0s ReadTimeouts that recovered on the next 3s tick. An
  // indicator that fires on those is an indicator operators learn to ignore.
  for (const failures of [1, STALE_FAILURE_THRESHOLD - 1]) {
    const notices = staleNoticesForTab({
      tab: 'merge',
      stale: staleMap({ [MERGE_PATH]: { failures, lastSuccessAt: NOW - 9000 } }),
      now: NOW,
    });
    assert.deepEqual(notices, [], `failures=${failures} must not produce a notice`);
  }
});

test('reaching the threshold produces exactly one notice', () => {
  const notices = staleNoticesForTab({
    tab: 'merge',
    stale: staleMap({
      [MERGE_PATH]: { failures: STALE_FAILURE_THRESHOLD, lastSuccessAt: NOW - 60_000 },
    }),
    now: NOW,
  });
  assert.equal(notices.length, 1);
  assert.equal(notices[0].kind, 'stale');
  assert.equal(notices[0].path, MERGE_PATH);
  assert.equal(typeof notices[0].text, 'string');
  assert.ok(notices[0].text.length > 0);
});

test('past the threshold it stays exactly one notice, not one per failure', () => {
  const notices = staleNoticesForTab({
    tab: 'merge',
    stale: staleMap({
      [MERGE_PATH]: { failures: STALE_FAILURE_THRESHOLD + 400, lastSuccessAt: NOW - 71_280_000 },
    }),
    now: NOW,
  });
  assert.equal(notices.length, 1);
});

// ── (c) the notice names the endpoint AND the age ──────────────────────────

test('the notice names the endpoint path and a human age', () => {
  const [notice] = staleNoticesForTab({
    tab: 'merge',
    stale: staleMap({
      [MERGE_PATH]: { failures: STALE_FAILURE_THRESHOLD, lastSuccessAt: NOW - 125_000 },
    }),
    now: NOW,
  });

  assert.ok(notice.text.includes(MERGE_PATH),
    `notice must name the endpoint so the operator knows WHICH data is old, got: ${notice.text}`);
  // Assert on the age SUBSTRING, not on exact prose — the wording is free to
  // change, the fact that an age is reported is not.
  assert.ok(notice.text.includes(formatAge(125_000)),
    `notice must carry the age derived from now - lastSuccessAt (${formatAge(125_000)}), got: ${notice.text}`);
});

test('formatAge renders seconds, minutes and hours distinguishably', () => {
  assert.ok(/\d/.test(formatAge(4_000)), formatAge(4_000));
  assert.notEqual(formatAge(4_000), formatAge(400_000));
  assert.notEqual(formatAge(400_000), formatAge(40_000_000));
  // 19.8h — the incident's own duration — must not render as a bare minute
  // count that an operator has to divide in their head.
  assert.ok(/h/.test(formatAge(19.8 * 3600 * 1000)), formatAge(19.8 * 3600 * 1000));
});

// ── (d) never-succeeded must not fabricate a zero age ──────────────────────

test('an endpoint that never succeeded says NEVER, not "0s ago"', () => {
  // Fabricating an age of zero during exactly the failure this indicator
  // exists to surface is the `_minutes_since` mistake active_tasks.py already
  // documents: a missing timestamp is UNKNOWN, never "just now".
  for (const lastSuccessAt of [0, null, undefined]) {
    const [notice] = staleNoticesForTab({
      tab: 'merge',
      stale: staleMap({
        [MERGE_PATH]: { failures: STALE_FAILURE_THRESHOLD, lastSuccessAt },
      }),
      now: NOW,
    });
    assert.ok(notice, `lastSuccessAt=${lastSuccessAt} must still produce a notice`);
    assert.ok(/never/i.test(notice.text),
      `must say the endpoint has NEVER delivered, got: ${notice.text}`);
    assert.ok(!/\b0\s*s\b/.test(notice.text),
      `must not fabricate a zero age, got: ${notice.text}`);
  }
});

// ── (e) the tab->endpoint map is machine-checked against the real sources ──

// Loads data.js against a shimmed browser-ish global. data.js assigns
// `window.DF_DATA = {...}` at module scope, and an ESM `import` statement's
// target module body runs BEFORE the importing file's own body — so
// `globalThis.window` would still be unset when data.js's top level runs,
// throwing `ReferenceError: window is not defined`. We therefore shim
// window/fetch FIRST and only then load via createRequire, exactly as
// data_poll.test.mjs::loadDataJs documents. `document` is deliberately NOT
// shimmed, so data.js's auto-start guard stays inert under node.
function loadDataJs() {
  globalThis.window = { dispatchEvent: () => {} };
  globalThis.fetch = () => Promise.resolve({ ok: true, json: async () => ({}) });
  const require = createRequire(import.meta.url);
  const specifier = '../../src/dashboard/static/redux/data.js';
  delete require.cache[require.resolve(specifier)];
  return require(specifier);
}

// app.jsx is `type="text/babel"` and cannot be imported here, so the tab ids
// are read out of its source — the same source-text contract idiom the Python
// suite uses for .jsx (see test_tab_escalation_analytics.py).
function appTabIds() {
  const src = fs.readFileSync(path.join(REDUX_DIR, 'app.jsx'), 'utf8');
  const block = src.match(/const tabs = \[([\s\S]*?)\];/);
  assert.ok(block, 'could not locate the `const tabs = [...]` block in app.jsx');
  return [...block[1].matchAll(/id:\s*'([^']+)'/g)].map(m => m[1]);
}

test('every mapped endpoint is a real endpointsFor() path', () => {
  // A map that silently stops matching a renamed endpoint is a check that
  // quietly stops checking: it would report "healthy" forever.
  const { endpointsFor, pollKey } = loadDataJs();
  const real = new Set(Object.keys(endpointsFor('24h')).map(pollKey));

  for (const [tab, paths] of Object.entries(TAB_ENDPOINTS)) {
    assert.ok(Array.isArray(paths) && paths.length > 0,
      `tab ${tab} must map to a non-empty array of endpoint paths`);
    for (const p of paths) {
      assert.ok(real.has(p),
        `TAB_ENDPOINTS[${tab}] names ${p}, which is not a pollKey of any ` +
        `endpointsFor('24h') key: ${[...real].join(', ')}`);
    }
  }
});

test('every mapped tab id is one of app.jsx\'s tab ids', () => {
  const ids = appTabIds();
  assert.equal(ids.length, 13, `expected app.jsx's 13 tabs, got ${ids.length}: ${ids.join(', ')}`);
  for (const tab of Object.keys(TAB_ENDPOINTS)) {
    assert.ok(ids.includes(tab),
      `TAB_ENDPOINTS has tab id ${tab}, which app.jsx does not render: ${ids.join(', ')}`);
  }
});

test('every app.jsx tab is mapped — a tab with no entry is silently unmonitored', () => {
  const ids = appTabIds();
  for (const tab of ids) {
    assert.ok(Object.prototype.hasOwnProperty.call(TAB_ENDPOINTS, tab),
      `app.jsx renders tab ${tab} but TAB_ENDPOINTS has no entry for it, so ` +
      'a wedge behind that tab would show no indicator at all');
  }
});

test('an unknown tab id yields no notice rather than throwing', () => {
  assert.deepEqual(
    staleNoticesForTab({
      tab: 'not-a-tab',
      stale: staleMap({ [MERGE_PATH]: { failures: 99, lastSuccessAt: 0 } }),
      now: NOW,
    }),
    [],
  );
});

// ── (f) fan-in and cross-tab isolation ─────────────────────────────────────

test('a multi-endpoint tab yields one notice per FAILING endpoint', () => {
  const paths = TAB_ENDPOINTS.tasks;
  assert.ok(paths.length > 1,
    `the tasks tab fans in from several endpoints; got ${JSON.stringify(paths)}`);

  const all = staleMap(Object.fromEntries(
    paths.map(p => [p, { failures: STALE_FAILURE_THRESHOLD, lastSuccessAt: NOW - 30_000 }]),
  ));
  const notices = staleNoticesForTab({ tab: 'tasks', stale: all, now: NOW });
  assert.equal(notices.length, paths.length);
  assert.deepEqual(notices.map(n => n.path).sort(), [...paths].sort());

  // Only one of them failing -> only that one is named.
  const one = staleMap({
    [TASKS_PATH]: { failures: STALE_FAILURE_THRESHOLD, lastSuccessAt: NOW - 30_000 },
  });
  const single = staleNoticesForTab({ tab: 'tasks', stale: one, now: NOW });
  assert.equal(single.length, 1);
  assert.equal(single[0].path, TASKS_PATH);
});

test('a healthy tab shows nothing while a DIFFERENT tab\'s endpoint is wedged', () => {
  // The whole point of per-endpoint staleness: the 19.8h wedge hit the tasks
  // fan-out while the escalations endpoint stayed current. A global banner
  // would have lied about both.
  const stale = staleMap({
    [TASKS_PATH]: { failures: 400, lastSuccessAt: NOW - 71_280_000 },
  });
  assert.deepEqual(staleNoticesForTab({ tab: 'esc', stale, now: NOW }), []);
  assert.equal(staleNoticesForTab({ tab: 'tasks', stale, now: NOW }).length, 1);
});

// ── staleEntryFor ──────────────────────────────────────────────────────────

test('staleEntryFor reads a path out of the published map and defaults safely', () => {
  const entry = { failures: 2, lastSuccessAt: NOW };
  assert.deepEqual(staleEntryFor(staleMap({ [MERGE_PATH]: entry }), MERGE_PATH), entry);
  assert.equal(staleEntryFor({}, MERGE_PATH), null);
  assert.equal(staleEntryFor(undefined, MERGE_PATH), null);
  assert.equal(staleEntryFor(null, MERGE_PATH), null);
});

test('every notice is a {kind, path, text} triple with kind "stale"', () => {
  const notices = staleNoticesForTab({
    tab: 'tasks',
    stale: staleMap({
      [TASKS_PATH]: { failures: STALE_FAILURE_THRESHOLD, lastSuccessAt: NOW - 45_000 },
    }),
    now: NOW,
  });
  for (const notice of notices) {
    assert.deepEqual(Object.keys(notice).sort(), ['kind', 'path', 'text']);
    assert.equal(notice.kind, 'stale');
    assert.equal(typeof notice.path, 'string');
    assert.equal(typeof notice.text, 'string');
  }
});

test('a missing `now` falls back rather than rendering NaN', () => {
  // App passes Date.now(), but this module also runs in the node harness and
  // must never emit "NaNs ago" into the operator's only staleness signal.
  const notices = staleNoticesForTab({
    tab: 'merge',
    stale: staleMap({
      [MERGE_PATH]: { failures: STALE_FAILURE_THRESHOLD, lastSuccessAt: NOW - 5000 },
    }),
  });
  assert.equal(notices.length, 1);
  assert.ok(!/NaN/.test(notices[0].text), notices[0].text);
});
