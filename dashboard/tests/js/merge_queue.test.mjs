// Module-contract tests for merge_queue.js, the CLIENT reader of the served
// "In queue now" datum whose server half is dashboard/src/dashboard/data/merge_queue.py.
// The MergeTab tile, its per-project "queued" pip and the rail badge all read
// the queue through this module, so the decisions it makes are asserted here,
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
const { projectInQueue, inQueueOver, inQueueHistory, latencyCaption } = mergeQueue;
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

// ── The module ──────────────────────────────────────────────────────────────

test('the module exposes its four readers and assigns window.DF_MERGE_QUEUE', () => {
  assert.deepEqual(
    Object.keys(mergeQueue).sort(),
    ['inQueueHistory', 'inQueueOver', 'latencyCaption', 'projectInQueue'],
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
