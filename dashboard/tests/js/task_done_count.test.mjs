// Module-contract tests for task_done_count.js, the INTERIM client guard for a
// project's done count now that /api/v2/dashboard/tasks serves it inside
// TASKS_SNAPSHOT[p].census instead of the retired DONE_COUNTS map. Run via
// `node --test` (dashboard/tests/test_graph_layout_js.py surfaces every
// **/*.test.mjs here in CI, so this file needs no wrapper change).
//
// WHY THIS SUITE EXISTS. Removing DONE_COUNTS did not blank the pip. It made it
// LIE: data.js seeded `DONE_COUNTS: {}`, so `DONE_COUNTS[p]` read undefined and
// both consumers fell through to a count of the done rows in ACTIVE_TASKS. The
// default render fetches none of those, so every healthy project rendered a
// confident "0 done". The same two halves as orch_summary_guard.test.mjs, for
// the same reason (nothing executes a .jsx file):
//
//   1. BEHAVIOUR: the guard, run over real wire entries, answers a measured
//      count or datum.js's placeholder, and never a zero it did not read.
//   2. WIRING: neither consumer still reads DONE_COUNTS, and both take the
//      guard off window.DF_TASK_DONE_COUNT.
//
// LOADED THROUGH A WINDOW SHIM, unlike orch_summary.js. The guard takes its
// placeholder from window.DF_DATUM at module scope, and datum.js in turn
// destructures window.DF_ENDPOINT_STALENESS at module scope. So the shim is
// installed first, datum.js is required through it, and only then the guard:
// data_poll.test.mjs::loadDataJs has the same shape, and index.html gives the
// three files the same order.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX_DIR = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '../../src/dashboard/static/redux',
);
const DATUM_MODULE_SPECIFIER = '../../src/dashboard/static/redux/datum.js';
const MODULE_SPECIFIER = '../../src/dashboard/static/redux/task_done_count.js';

function loadGuard() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of [DATUM_MODULE_SPECIFIER, MODULE_SPECIFIER]) {
    delete require.cache[require.resolve(specifier)];
  }
  require(DATUM_MODULE_SPECIFIER);
  return { guard: require(MODULE_SPECIFIER), window: win };
}

const { guard, window: loadedWindow } = loadGuard();
const { hasDoneCount, doneCount } = guard;
const { EM_DASH } = loadedWindow.DF_DATUM;

// One TaskCensus.to_wire() value, printed from the real census.py for a
// five-task map. Transcribed rather than hand-shaped so the guard reads the
// path the server actually emits: `counts` keyed by status VALUE.
const CENSUS_VALUE = Object.freeze({
  counts: {
    pending: 1, 'in-progress': 1, blocked: 0, deferred: 0, review: 0,
    'merge-deferred': 0, 'infra-hold': 0, done: 2, cancelled: 1,
  },
  total: 5,
  views: { in_flight: 1, backlog: 1, terminal: 3 },
  sub_views: { running: 1 },
});

const AS_OF = '2026-09-22T12:00:00+00:00';

// A Datum in *state*, honouring datum.py's triad: unknown has no value and no
// instant, every other state has both.
function datumIn(state, value) {
  if (state === 'unknown') {
    return { value: null, as_of: null, state, reason: 'status map read failed', freshness_bound_seconds: 30 };
  }
  return {
    value,
    as_of: AS_OF,
    state,
    reason: state === 'fresh' ? null : 'status map read failed',
    freshness_bound_seconds: 30,
  };
}

// One TASKS_SNAPSHOT[p] entry, the five keys TaskSnapshot.to_wire() emits.
function entryWith(census, rows) {
  return {
    census,
    rows,
    in_progress_live: rows.state === 'unknown' ? null : 1,
    in_progress_stranded: rows.state === 'unknown' ? null : 0,
    skew_seconds: census.as_of && rows.as_of ? 0 : null,
  };
}

const FRESH_ROWS = datumIn('fresh', []);
const FRESH_ENTRY = entryWith(datumIn('fresh', CENSUS_VALUE), FRESH_ROWS);

test('the module exposes the guard and assigns window.DF_TASK_DONE_COUNT', () => {
  for (const name of ['hasDoneCount', 'doneCount']) {
    assert.equal(typeof guard[name], 'function', `${name} should be a function`);
  }
  // The browser half of the dual export: both consumers destructure this
  // global UNGUARDED at module top level, so an undefined one blanks the tab.
  assert.equal(loadedWindow.DF_TASK_DONE_COUNT, guard);
});

// ── (a) A fresh census is a measurement, and the guard hands it over ──────

test('(a) a fresh census answers its measured done count, as a number', () => {
  assert.equal(hasDoneCount(FRESH_ENTRY), true);
  assert.equal(doneCount(FRESH_ENTRY), 2);
});

test('(a) a measured zero is a count, not a hole', () => {
  // The `!value` trap: a guard testing truthiness would turn a project that
  // really has no done tasks into the placeholder.
  const none = entryWith(
    datumIn('fresh', { ...CENSUS_VALUE, counts: { ...CENSUS_VALUE.counts, done: 0 } }),
    FRESH_ROWS,
  );
  assert.equal(hasDoneCount(none), true);
  assert.equal(doneCount(none), 0);
});

// ── (b) Everything else is datum.js's placeholder, never a zero ───────────

const NO_COUNT = [
  ['no entry at all', undefined],
  ['a null entry', null],
  ['an entry with no census', { rows: FRESH_ROWS }],
  ['a null census', { census: null, rows: FRESH_ROWS }],
  // Deliberate for a stale census, whose aged count is real: the guard's
  // verdict mirrors task_snapshot.classify, which names a project whose census
  // is not fresh in TASKS_COUNT_UNKNOWN_PROJECTS. Showing the aged number beside
  // that banner would be the pip contradicting it. Leaf γ3 owns aged rendering.
  ['a stale census', entryWith(datumIn('stale', CENSUS_VALUE), FRESH_ROWS)],
  ['an unknown census', entryWith(datumIn('unknown'), FRESH_ROWS)],
  ['a lower_bound census', entryWith(datumIn('lower_bound', CENSUS_VALUE), FRESH_ROWS)],
  ['a fresh census with no counts', entryWith(datumIn('fresh', { total: 5 }), FRESH_ROWS)],
  ['a fresh census with null counts', entryWith(datumIn('fresh', { counts: null }), FRESH_ROWS)],
  ['a fresh census with no done key', entryWith(datumIn('fresh', { counts: { pending: 1 } }), FRESH_ROWS)],
  ['a string done count', entryWith(datumIn('fresh', { counts: { done: '2' } }), FRESH_ROWS)],
  ['a NaN done count', entryWith(datumIn('fresh', { counts: { done: NaN } }), FRESH_ROWS)],
  ['a null done count', entryWith(datumIn('fresh', { counts: { done: null } }), FRESH_ROWS)],
];

for (const [label, entry] of NO_COUNT) {
  test(`(b) ${label} answers the placeholder, never 0`, () => {
    assert.equal(hasDoneCount(entry), false);
    // Identity with datum.js's own constant rather than a re-typed dash: the
    // placeholder has exactly one authority (datum.js::EM_DASH).
    assert.equal(doneCount(entry), EM_DASH);
  });
}

// ── (c) The pip and the banners cannot disagree ────────────────────────────
//
// What the server emits per task_snapshot.classify branch: an entry of that
// shape, and the banner list that branch names the project in. classify keys
// on the ROWS half's failure kind first and only then on the census's state,
// which is why an OFFLINE entry can still carry a census that was measured.

const CLASSIFIED = [
  {
    branch: 'OK: both halves measured',
    entry: FRESH_ENTRY,
    list: null,
  },
  {
    branch: 'COUNT_UNKNOWN: rows current, map failed with no last good',
    entry: entryWith(datumIn('unknown'), FRESH_ROWS),
    list: 'TASKS_COUNT_UNKNOWN_PROJECTS',
  },
  {
    branch: 'COUNT_UNKNOWN: rows current, map failed, last good aged',
    entry: entryWith(datumIn('stale', CENSUS_VALUE), FRESH_ROWS),
    list: 'TASKS_COUNT_UNKNOWN_PROJECTS',
  },
  {
    branch: 'DEGRADED: the budget cut the share off, census aged',
    entry: entryWith(datumIn('stale', CENSUS_VALUE), datumIn('unknown')),
    list: 'TASKS_DEGRADED_PROJECTS',
  },
  {
    branch: 'OFFLINE: the share raised',
    entry: entryWith(datumIn('unknown'), datumIn('unknown')),
    list: 'TASKS_OFFLINE_PROJECTS',
  },
  {
    branch: 'OFFLINE: the row read failed, the map read did not',
    entry: entryWith(datumIn('fresh', CENSUS_VALUE), datumIn('unknown')),
    list: 'TASKS_OFFLINE_PROJECTS',
  },
];

test('(c) a project named in no banner always shows a measured count', () => {
  for (const { branch, entry, list } of CLASSIFIED) {
    if (list !== null) continue;
    assert.equal(hasDoneCount(entry), true, `${branch}: a healthy project showed the placeholder`);
  }
});

test('(c) a project named count-unknown never shows a number', () => {
  for (const { branch, entry, list } of CLASSIFIED) {
    if (list !== 'TASKS_COUNT_UNKNOWN_PROJECTS') continue;
    assert.equal(doneCount(entry), EM_DASH, `${branch}: the pip contradicts the banner`);
  }
});

test('(c) where the rows were measured, fresh <=> named in no banner', () => {
  // classify's own rule for a unit whose rows half did not fail: OK iff the
  // census is fresh, COUNT_UNKNOWN otherwise.
  for (const { branch, entry, list } of CLASSIFIED) {
    if (list === 'TASKS_OFFLINE_PROJECTS' || list === 'TASKS_DEGRADED_PROJECTS') continue;
    assert.equal(hasDoneCount(entry), list === null, branch);
  }
});

test('(c) the verdict is read off the census alone, whatever banner the rows earned', () => {
  // An OFFLINE or DEGRADED banner reports the ROW read ("task data
  // unavailable", "the task fetch timed out"). It is not a claim about the
  // count. A census measured beside a failed row read is still a measurement,
  // and the guard, which sees one entry and no banner list, answers from it.
  for (const { branch, entry } of CLASSIFIED) {
    assert.equal(hasDoneCount(entry), entry.census.state === 'fresh', branch);
  }
});

// ── Wiring: no consumer reads DONE_COUNTS, both take the guard ─────────────

const CONSUMERS = ['tab_tasks.jsx', 'tabs.jsx'];

test('no consumer still reads DONE_COUNTS', () => {
  for (const name of CONSUMERS) {
    const source = fs.readFileSync(path.join(REDUX_DIR, name), 'utf8');
    const reads = source.match(/\.DONE_COUNTS\b/g) || [];
    assert.deepEqual(
      reads,
      [],
      `${name} still reads DONE_COUNTS (${reads.length} site(s)). /tasks no longer ` +
        'serves it, and data.js no longer seeds it, so the read is undefined and ' +
        "the site's fallback renders a confident \"0 done\". Read the census " +
        'through doneCount() from window.DF_TASK_DONE_COUNT instead.',
    );
  }
});

test('every consumer destructures the guard it depends on', () => {
  for (const name of CONSUMERS) {
    const source = fs.readFileSync(path.join(REDUX_DIR, name), 'utf8');
    assert.ok(
      /=\s*window\.DF_TASK_DONE_COUNT\s*;/.test(source),
      `${name} never destructures window.DF_TASK_DONE_COUNT, so its done count ` +
        'has no guard to go through',
    );
  }
});
