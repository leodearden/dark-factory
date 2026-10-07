// Module-contract tests for merge_queue.js, the CLIENT reader of the served
// /merge-queue Datums (in_queue, speculative, recent_total) whose server half is
// dashboard/src/dashboard/data/merge_queue.py, and of each queued row's enqueue
// instant. The MergeTab tiles, its per-project pips, the live feed and the rail
// badge all read the queue through this module, so the decisions it makes are asserted here,
// where node can execute them. tabs.jsx and app.jsx are `type="text/babel"`
// behind CDN Babel and cannot run under node; their WIRING is pinned
// structurally in Python (test_tab_merge_queue.py).
//
// Run via `node --test` (dashboard/tests/test_graph_layout_js.py's
// `**/*.test.mjs` glob auto-discovers this file).
//
// LOADED THROUGH A WINDOW SHIM, in index.html's order: merge_queue.js
// destructures window.DF_DATUM at module scope with no fallback, and datum.js
// in turn destructures window.DF_ENDPOINT_STALENESS — task_snapshot.test.mjs
// has the same shape.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'merge_queue.js'].map(name => REDUX + name);

function loadMergeQueue() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [, mergeQueueApi] = LOAD_CHAIN.map(specifier => require(specifier));
  return { api: mergeQueueApi, window: win };
}

const { api: mergeQueue, window: loadedWindow } = loadMergeQueue();
const {
  projectInQueue, inQueueOver, inQueueHistory, latencyCaption,
  queuedSince, projectSpeculative, speculativeOver, hitRateText, recentTotal,
} = mergeQueue;
const { isDatum, datumView, EM_DASH } = loadedWindow.DF_DATUM;

// ── Fixtures ────────────────────────────────────────────────────────────────

const MERGE_QUEUE_ENDPOINT = '/api/v2/dashboard/merge-queue';
const SERVED_AT = '2026-10-01T12:00:30+00:00';
const PROBED_AT = '2026-10-01T12:00:28+00:00';
const THREE_HOURS_EARLIER = '2026-10-01T09:00:30+00:00';
const RECEIVED_AT = 1_800_000_000_000;
const NOW = RECEIVED_AT + 1_000;
const RECEIPT = Object.freeze({ servedAt: SERVED_AT, receivedAt: RECEIVED_AT });

const T1 = '2026-10-01T11:58:00+00:00';
const T2 = '2026-10-01T11:59:00+00:00';
const T3 = '2026-10-01T12:00:00+00:00';

// A served in_queue Datum, honouring datum.py's unknown triad.
function inQueue(state, value, overrides) {
  const base =
    state === 'unknown'
      ? { value: null, as_of: null, state, reason: 'connect refused; no sample in the history window', freshness_bound_seconds: 30 }
      : { value, as_of: PROBED_AT, state, reason: state === 'fresh' ? null : 'connect refused', freshness_bound_seconds: 30 };
  return { ...base, ...overrides };
}

// One MERGE_QUEUE[label] entry: the keys these readers touch.
function entryWith(served, spark) {
  return {
    in_queue: served,
    live_probe_configured: true,
    active: [],
    active_spark: spark || { labels: [], values: [] },
  };
}

// A project the server has no live probe for: data/merge_queue.py::resolve_active
// serves it unknown, with live_probe_configured false.
function unprobedEntry(label, spark) {
  const reason = 'no live get_merge_queue probe is configured for ' + label;
  return { ...entryWith(inQueue('unknown', null, { reason }), spark), live_probe_configured: false };
}

function mqData(entries, receipt = RECEIPT) {
  return {
    MERGE_QUEUE: entries,
    __receipt: receipt ? { [MERGE_QUEUE_ENDPOINT]: receipt } : {},
  };
}

const TWO_FRESH = () => mqData({
  a: entryWith(inQueue('fresh', 2)),
  b: entryWith(inQueue('fresh', 1)),
});

// A served runs.db reading (speculative, recent_total), honouring the unknown triad.
function runsDbRead(state, value, overrides) {
  const base =
    state === 'unknown'
      ? { value: null, as_of: null, state, reason: 'the merge_attempt events could not be read from runs.db', freshness_bound_seconds: 30 }
      : { value, as_of: PROBED_AT, state, reason: state === 'fresh' ? null : 'measured 45s before it was served', freshness_bound_seconds: 30 };
  return { ...base, ...overrides };
}

function counts(hits, discards) {
  const total = hits + discards;
  return { hit_count: hits, discard_count: discards, total, hit_rate: total > 0 ? hits / total : null };
}

function entryReading(spec, recent, served = inQueue('fresh', 0)) {
  return { ...entryWith(served), speculative: spec, recent_total: recent };
}

// ── The module ──────────────────────────────────────────────────────────────

test('the module exposes its readers and assigns window.DF_MERGE_QUEUE', () => {
  assert.deepEqual(
    Object.keys(mergeQueue).sort(),
    [
      'hitRateText', 'inQueueHistory', 'inQueueOver', 'latencyCaption', 'projectInQueue',
      'projectSpeculative', 'queuedSince', 'recentTotal', 'speculativeOver',
    ],
  );
  for (const name of Object.keys(mergeQueue)) {
    assert.equal(typeof mergeQueue[name], 'function', `${name} should be a function`);
  }
  // The browser half of the dual export: tabs.jsx and app.jsx destructure this
  // global at module scope with no fallback.
  assert.equal(loadedWindow.DF_MERGE_QUEUE, mergeQueue);
});

// ── projectInQueue: the served datum, stamped with the /merge-queue receipt ─

test('projectInQueue: the served Datum, stamped with the /merge-queue receipt', () => {
  const data = TWO_FRESH();
  const wire = data.MERGE_QUEUE.a.in_queue;
  const pristine = structuredClone(wire);

  const served = projectInQueue(data, 'a');

  assert.equal(isDatum(served), true);
  assert.equal(served.value, 2);
  assert.equal(served.state, 'fresh');
  assert.equal(served.as_of, PROBED_AT);
  assert.equal(served._served_at, SERVED_AT);
  assert.equal(served._received_at, RECEIVED_AT);
  assert.notEqual(served, wire, 'the stamp must land on a copy');
  assert.deepEqual(wire, pristine, 'the wire datum was mutated');
});

test('projectInQueue: before the first /merge-queue payload, it is not yet fetched', () => {
  const served = projectInQueue(mqData({}, null), 'a');
  assert.equal(served.state, 'unknown');
  assert.equal(served.reason, 'not yet fetched');
});

test('projectInQueue: an entry without an in_queue Datum is a reasoned hole — never 0, never a throw', () => {
  const malformed = [
    ['an absent project', undefined],
    ['a null entry', null],
    ['an entry with no in_queue', { active: [] }],
    ['a bare count', entryWith(3)],
  ];
  for (const [label, entry] of malformed) {
    const entries = entry === undefined ? {} : { hive: entry };
    let served;
    assert.doesNotThrow(() => {
      served = projectInQueue(mqData(entries), 'hive');
    }, label);
    assert.equal(isDatum(served), true, label);
    assert.equal(served.state, 'unknown', label);
    assert.equal(served.value, null, label);
    assert.match(served.reason, /hive/, `${label}: the reason must name the project`);
  }
});

// ── inQueueOver: one total, a hole anywhere a hole in it ────────────────────

test('inQueueOver: null scope sums every project the payload carries', () => {
  const total = inQueueOver(TWO_FRESH(), null);
  assert.equal(total.state, 'fresh');
  assert.equal(total.value, 3);
});

test('inQueueOver: a scope sums only its own projects', () => {
  assert.equal(inQueueOver(TWO_FRESH(), ['a']).value, 2);
});

test('inQueueOver: a project whose probe failed renders its last sample with its age and why', () => {
  // THE SIGNAL. The rail badge and the tile read this one total: a failed
  // probe's last sample is still a number, but it says how old it is and why
  // it is not live, instead of passing for a confident present-tense count.
  const data = mqData({
    a: entryWith(inQueue('fresh', 2, { as_of: SERVED_AT })),
    b: entryWith(inQueue('stale', 1, { as_of: THREE_HOURS_EARLIER, reason: 'connect refused' })),
  });

  const view = datumView(inQueueOver(data, null), { now: NOW });

  assert.equal(view.text, '3');
  assert.equal(view.age, '3h');
  assert.match(view.title, /connect refused/);
});

test('inQueueOver: one unknown project makes the total unknown, and names it', () => {
  const data = mqData({
    a: entryWith(inQueue('fresh', 2)),
    b: entryWith(inQueue('unknown')),
  });

  const total = inQueueOver(data, null);
  const view = datumView(total, { now: NOW });

  assert.equal(total.state, 'unknown');
  assert.equal(view.text, EM_DASH);
  assert.match(total.reason, /\bb: /);
});

test('inQueueOver: a project with no live probe configured is outside the total, not a hole in it', () => {
  // A configured root may run no orchestrator at all. It has no queue this
  // dashboard can read, now or ever, so it must not blank every other
  // project's count for good. Contrast the case above: a PROBED project that
  // cannot be read is still a hole.
  const data = mqData({ ...TWO_FRESH().MERGE_QUEUE, c: unprobedEntry('c') });

  const total = inQueueOver(data, null);

  assert.equal(total.state, 'fresh');
  assert.equal(total.value, 3);
  assert.equal(inQueueOver(data, ['a', 'c']).value, 2);
});

test('inQueueOver: a scope of only unprobed projects has no total, and says why', () => {
  const total = inQueueOver(mqData({ c: unprobedEntry('c') }), null);

  assert.equal(total.state, 'unknown');
  assert.match(total.reason, /probe/);
});

test('projectInQueue: an unprobed project still reads its own served datum', () => {
  const served = projectInQueue(mqData({ c: unprobedEntry('c') }), 'c');

  assert.equal(served.state, 'unknown');
  assert.match(served.reason, /no live get_merge_queue probe is configured for c/);
});

test('inQueueOver: before the first payload, the total is not yet fetched', () => {
  const total = inQueueOver(mqData({}, null), null);
  assert.equal(total.state, 'unknown');
  assert.equal(total.reason, 'not yet fetched');
});

// ── inQueueHistory: the tile's spark, over labels every project sampled ────

test('inQueueHistory: one project is its own sampled values', () => {
  const data = mqData({ a: entryWith(inQueue('fresh', 2), { labels: [T1, T2, T3], values: [4, 3, 2] }) });
  assert.deepEqual(inQueueHistory(data, ['a']), [4, 3, 2]);
});

test('inQueueHistory: several projects sum label-wise, only where every one sampled', () => {
  // b missed the T2 sample: a sum there would be a's count passed off as the
  // fleet's, a dip that never happened.
  const data = mqData({
    a: entryWith(inQueue('fresh', 2), { labels: [T1, T2, T3], values: [4, 3, 2] }),
    b: entryWith(inQueue('fresh', 1), { labels: [T1, T3], values: [1, 1] }),
  });
  assert.deepEqual(inQueueHistory(data, ['a', 'b']), [5, 3]);
  assert.deepEqual(inQueueHistory(data, null), [5, 3]);
});

test('inQueueHistory: an unprobed project is outside the spark too', () => {
  const data = mqData({
    a: entryWith(inQueue('fresh', 2), { labels: [T1, T2], values: [4, 3] }),
    c: unprobedEntry('c'),
  });
  assert.deepEqual(inQueueHistory(data, null), [4, 3]);
  assert.deepEqual(inQueueHistory(data, ['a', 'c']), [4, 3]);
});

test('inQueueHistory: a project in scope with no samples leaves no shared label', () => {
  const data = mqData({ a: entryWith(inQueue('fresh', 2), { labels: [T1], values: [4] }) });
  assert.deepEqual(inQueueHistory(data, ['a', 'missing']), []);
});

// ── latencyCaption ──────────────────────────────────────────────────────────

test('latencyCaption: the centiles say how many attempts they were computed over', () => {
  assert.equal(
    latencyCaption({ with_duration: 167, without_duration: 79 }),
    'of 167 with recorded duration · 79 without',
  );
});

test('latencyCaption: a latency block without the split states nothing rather than a zero', () => {
  assert.equal(latencyCaption({}), '');
  assert.equal(latencyCaption(undefined), '');
});

// ── queuedSince: when a queued row joined, from the probe instant and its age ─

test('queuedSince: the probe instant minus the row\'s age', () => {
  const entry = entryWith(inQueue('fresh', 1));
  assert.equal(queuedSince(entry, { task_id: '7', age_secs: 90 }), '2026-10-01T11:58:58.000Z');
  assert.equal(queuedSince(entry, { task_id: '7', age_secs: 0 }), '2026-10-01T12:00:28.000Z');
});

test('queuedSince: a row with no usable age has no enqueue instant — never "just now"', () => {
  const entry = entryWith(inQueue('fresh', 1));
  for (const age of [null, undefined, Number.NaN, -1]) {
    assert.equal(queuedSince(entry, { task_id: '7', age_secs: age }), null, String(age));
  }
});

test('queuedSince: with no probe instant there is nothing to subtract from', () => {
  const row = { task_id: '7', age_secs: 90 };
  assert.equal(queuedSince(entryWith(inQueue('unknown')), row), null);
  assert.equal(queuedSince({ active: [] }, row), null);
  assert.equal(queuedSince(null, row), null);
});

test('queuedSince: reads its inputs and alters neither', () => {
  const entry = entryWith(inQueue('fresh', 1));
  const row = { task_id: '7', age_secs: 90 };
  const pristine = structuredClone([entry, row]);

  queuedSince(entry, row);

  assert.deepEqual([entry, row], pristine);
});

// ── projectSpeculative / recentTotal: one project's runs.db readings, stamped ─

for (const [field, reader, value] of [
  ['speculative', projectSpeculative, counts(5, 3)],
  ['recent_total', recentTotal, 228],
]) {
  test(`${field}: the served Datum, stamped with the /merge-queue receipt as a copy`, () => {
    const data = mqData({ a: entryReading(runsDbRead('fresh', counts(5, 3)), runsDbRead('fresh', 228)) });
    const wire = data.MERGE_QUEUE.a[field];
    const pristine = structuredClone(wire);

    const served = reader(data, 'a');

    assert.equal(isDatum(served), true);
    assert.deepEqual(served.value, value);
    assert.equal(served.state, 'fresh');
    assert.equal(served._served_at, SERVED_AT);
    assert.equal(served._received_at, RECEIVED_AT);
    assert.notEqual(served, wire, 'the stamp must land on a copy');
    assert.deepEqual(wire, pristine, 'the wire datum was mutated');
  });

  test(`${field}: before the first /merge-queue payload, it is not yet fetched`, () => {
    const served = reader(mqData({}, null), 'a');
    assert.equal(served.state, 'unknown');
    assert.equal(served.reason, 'not yet fetched');
  });

  test(`${field}: a missing project or a missing Datum is a hole naming the project`, () => {
    const malformed = [
      ['an absent project', {}],
      ['an entry with no Datum', { hive: entryWith(inQueue('fresh', 0)) }],
      ['a bare value', { hive: { ...entryWith(inQueue('fresh', 0)), [field]: 0 } }],
    ];
    for (const [label, entries] of malformed) {
      const served = reader(mqData(entries), 'hive');
      assert.equal(served.state, 'unknown', label);
      assert.equal(served.value, null, label);
      assert.match(served.reason, /hive/, `${label}: the reason must name the project`);
    }
  });

  test(`${field}: an unread runs.db keeps the server's reason`, () => {
    const unread = runsDbRead('unknown');
    const served = reader(mqData({ a: entryReading(unread, unread) }), 'a');
    assert.equal(served.state, 'unknown');
    assert.equal(served.reason, unread.reason);
  });
}

// ── speculativeOver: the cross-project speculative tile ────────────────────

test('speculativeOver: sums the counts across every payload project, unprobed ones included', () => {
  // live_probe_configured describes the live queue probe; speculative counts
  // come from each project's runs.db whatever that probe can reach.
  const data = mqData({
    a: entryReading(runsDbRead('fresh', counts(3, 1)), runsDbRead('fresh', 4)),
    c: { ...unprobedEntry('c'), speculative: runsDbRead('fresh', counts(2, 2)), recent_total: runsDbRead('fresh', 4) },
  });

  const total = speculativeOver(data, null);

  assert.equal(total.state, 'fresh');
  assert.deepEqual(total.value, { hit_count: 5, discard_count: 3, total: 8 });
  assert.deepEqual(speculativeOver(data, ['a']).value, { hit_count: 3, discard_count: 1, total: 4 });
});

test('speculativeOver: a hole in any project is a hole in the total, naming that project', () => {
  const data = mqData({
    a: entryReading(runsDbRead('fresh', counts(3, 1)), runsDbRead('fresh', 4)),
    b: entryReading(runsDbRead('unknown'), runsDbRead('unknown')),
  });

  const total = speculativeOver(data, null);

  assert.equal(total.state, 'unknown');
  assert.match(total.reason, /\bb: /);
  assert.equal(speculativeOver(data, ['a', 'missing']).state, 'unknown');
  assert.match(speculativeOver(data, ['a', 'missing']).reason, /missing/);
});

test('speculativeOver: a measured window with no attempts has no hit rate to show', () => {
  const data = mqData({ a: entryReading(runsDbRead('fresh', counts(0, 0)), runsDbRead('fresh', 0)) });

  const total = speculativeOver(data, null);

  assert.equal(total.state, 'unknown');
  assert.equal(total.reason, 'no speculative attempts in this window');
});

test('speculativeOver: before the first payload, the total is not yet fetched', () => {
  const total = speculativeOver(mqData({}, null), null);
  assert.equal(total.state, 'unknown');
  assert.equal(total.reason, 'not yet fetched');
});

test('speculativeOver: a stale part makes the total stale', () => {
  const data = mqData({
    a: entryReading(runsDbRead('fresh', counts(1, 0)), runsDbRead('fresh', 1)),
    b: entryReading(runsDbRead('stale', counts(1, 1)), runsDbRead('stale', 2)),
  });

  const total = speculativeOver(data, null);

  assert.equal(total.state, 'stale');
  assert.match(total.reason, /\bb: /);
});

// ── hitRateText ────────────────────────────────────────────────────────────

test('hitRateText: hits over attempts, as a whole percentage', () => {
  assert.equal(hitRateText({ hit_count: 3, total: 4 }), '75%');
});

test('hitRateText: zero attempts have no rate', () => {
  assert.equal(hitRateText({ hit_count: 0, total: 0 }), EM_DASH);
});
