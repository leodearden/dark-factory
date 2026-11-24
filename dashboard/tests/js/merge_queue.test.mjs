// Module-contract tests for merge_queue.js, the CLIENT reader of the served
// /merge-queue Datums (in_queue, speculative, recent_total, depth, attempts) whose server half is
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
// has the same shape. window_chip.js rides along because MergeTab hands
// recentTotal's value to its recentMergesCaption.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import staleness from '../../src/dashboard/static/redux/endpoint_staleness.js';

const REDUX = '../../src/dashboard/static/redux/';
const LOAD_CHAIN = ['datum.js', 'window_chip.js', 'merge_queue.js'].map(name => REDUX + name);

function loadMergeQueue() {
  const win = { DF_ENDPOINT_STALENESS: staleness };
  globalThis.window = win;
  const require = createRequire(import.meta.url);
  for (const specifier of LOAD_CHAIN) delete require.cache[require.resolve(specifier)];
  const [, , mergeQueueApi] = LOAD_CHAIN.map(specifier => require(specifier));
  return { api: mergeQueueApi, window: win };
}

const { api: mergeQueue, window: loadedWindow } = loadMergeQueue();
const {
  projectInQueue, inQueueOver, inQueueHistory, latencyCaption,
  queuedSince, projectSpeculative, speculativeOver, hitRateText, recentTotal,
  projectDepth, projectAttempts, attemptCount, attemptsOver, attemptsHistory, worstP95Over,
  totalHint, projectLandingRate, landingRateCaption, landingRateReading,
} = mergeQueue;
const { isDatum, datumView, EM_DASH, LOWER_BOUND_PREFIX, missingParts } = loadedWindow.DF_DATUM;
const { recentMergesCaption } = loadedWindow.DF_WINDOW_CHIP;

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

// A served depth series value: merge_attempt counts per bucket, its leading
// zero buckets trimmed by the shaper.
function depthSeries(labels, values) {
  return { labels, values };
}

// A served landing-rate value: landings per day at each point over the trailing
// window the server names, its leading zero points trimmed by the shaper, and
// the window mean the server counted over the whole window.
function landingRate(labels, values, trailingMinutes = 1440, windowMean = 0) {
  return { labels, values, trailing_minutes: trailingMinutes, window_mean_per_day: windowMean };
}

const UNTIMED = Object.freeze({ p50: null, p95: null, p99: null, mean_ms: null });

// A served attempts value, as redux_api.py::_shape_attempts projects one window's
// MergeAttempts: the colored outcome chart and the latency block. With no
// *latency* given, the window is untimed — no centile over zero samples.
function attemptsValue(outcomeCounts, latency) {
  const labels = Object.keys(outcomeCounts);
  const values = labels.map(label => outcomeCounts[label]);
  const attempts = values.reduce((sum, n) => sum + n, 0);
  return {
    outcomes: { labels, values, colors: labels.map(() => '#888888') },
    latency: latency || { ...UNTIMED, with_duration: 0, without_duration: attempts },
  };
}

// A timed latency block whose worst centile is *p95*.
function timedLatency(p95, withDuration, withoutDuration = 0) {
  return { p50: p95 / 2, p95, p99: p95 + 1, mean_ms: p95 / 2, with_duration: withDuration, without_duration: withoutDuration };
}

// One MERGE_QUEUE[label] entry carrying its depth and attempts readings.
function entryWindow(depth, attempts, served = inQueue('fresh', 0)) {
  return { ...entryWith(served), depth, attempts };
}

// A project with no runs.db file at all: data/merge_queue.py serves it
// runs_db_present false, every runs.db reading unknown saying so.
function noRunsDbEntry() {
  const absent = runsDbRead('unknown', null, { reason: 'no runs.db file exists for this project' });
  return {
    ...entryWith(inQueue('fresh', 0)),
    runs_db_present: false,
    depth: absent,
    attempts: absent,
    speculative: absent,
    recent_total: absent,
  };
}

// ── The module ──────────────────────────────────────────────────────────────

test('the module exposes its readers and assigns window.DF_MERGE_QUEUE', () => {
  assert.deepEqual(
    Object.keys(mergeQueue).sort(),
    [
      'attemptCount', 'attemptsHistory', 'attemptsOver',
      'hitRateText', 'inQueueHistory', 'inQueueOver',
      'landingRateCaption', 'landingRateReading', 'latencyCaption',
      'projectAttempts', 'projectDepth',
      'projectInQueue', 'projectLandingRate', 'projectSpeculative', 'queuedSince', 'recentTotal',
      'speculativeOver',
      'totalHint', 'worstP95Over',
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

test('inQueueOver: a probed project with no reading makes the total a lower bound naming it', () => {
  // A probe that fails with no sample in its history (an orchestrator that is
  // stopped) is a hole, not a zero: the others still count, as a floor.
  const data = mqData({
    a: entryWith(inQueue('fresh', 2)),
    b: entryWith(inQueue('unknown')),
  });

  const total = inQueueOver(data, null);

  assert.equal(total.state, 'lower_bound');
  assert.equal(datumView(total, { now: NOW }).text, `${LOWER_BOUND_PREFIX}2`);
  assert.match(total.reason, /\bb: /);
  assert.equal(totalHint(total), 'b unreadable');
});

test('inQueueOver: every probed project unreadable has no total', () => {
  const data = mqData({ a: entryWith(inQueue('unknown')), b: entryWith(inQueue('unknown')) });

  const total = inQueueOver(data, null);

  assert.equal(total.state, 'unknown');
  assert.equal(datumView(total, { now: NOW }).text, EM_DASH);
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

// ── landingRateCaption / landingRateReading: the throughput panel's words ──

test('landingRateCaption: names the trailing window the server counted each point over', () => {
  assert.equal(landingRateCaption({ trailing_minutes: 1440 }), 'landings/day · 1-day trailing');
  assert.equal(landingRateCaption({ trailing_minutes: 10080 }), 'landings/day · 7-day trailing');
  assert.equal(landingRateCaption({ trailing_minutes: 60 }), 'landings/day · 1-h trailing');
  assert.equal(landingRateCaption({ trailing_minutes: 90 }), 'landings/day · 90-min trailing');
});

test('landingRateCaption: a rate naming no usable trailing window states none', () => {
  const holes = [
    undefined, null, {}, { trailing_minutes: null }, { trailing_minutes: 0 },
    { trailing_minutes: -15 }, { trailing_minutes: 7.5 }, { trailing_minutes: '15' },
  ];
  for (const series of holes) {
    assert.equal(landingRateCaption(series), EM_DASH, JSON.stringify(series));
  }
});

test('landingRateReading: a window with no landings reads a measured zero, saying so', () => {
  assert.equal(landingRateReading(landingRate([], [])), '0/day · no landings in this window');
});

test('landingRateReading: the mean is the served window mean, not an average of the points', () => {
  assert.equal(landingRateReading(landingRate([T1, T2], [2, 3], 1440, 2 / 7)), 'now 3/day · mean 0.29/day');
});

test('landingRateReading: rates of 1 or more show at most one decimal, a trailing .0 dropped', () => {
  assert.equal(
    landingRateReading(landingRate([T1, T2], [27.43, 31], 1440, 29.24)), 'now 31/day · mean 29.2/day',
  );
  assert.equal(landingRateReading(landingRate([T1, T2], [2, 3], 1440, 4)), 'now 3/day · mean 4/day');
});

test('landingRateReading: rates below 1 keep two significant figures, never a false 0/day', () => {
  for (const [mean, text] of [[0.14, '0.14'], [1 / 30, '0.033'], [1 / 3650, '0.00027']]) {
    const reading = landingRateReading(landingRate([T1, T2], [2, 3], 1440, mean));
    assert.equal(reading, `now 3/day · mean ${text}/day`);
    assert.doesNotMatch(reading, /mean 0\/day/);
  }
  assert.equal(landingRateReading(landingRate([T1, T2], [0.5, 0.14], 1440, 1)), 'now 0.14/day · mean 1/day');
});

test('landingRateReading: a rate serving no usable window mean states none', () => {
  for (const mean of [null, NaN, '0.5']) {
    assert.equal(
      landingRateReading(landingRate([T1, T2], [2, 3], 1440, mean)), `now 3/day · mean ${EM_DASH}`,
      String(mean),
    );
  }
  const { window_mean_per_day: _dropped, ...unmeaned } = landingRate([T1, T2], [2, 3]);
  assert.equal(landingRateReading(unmeaned), `now 3/day · mean ${EM_DASH}`);
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

// ── projectSpeculative / recentTotal / projectDepth / projectLandingRate /
// projectAttempts: one project's runs.db readings, stamped ────────────────

for (const [field, reader, value] of [
  ['speculative', projectSpeculative, counts(5, 3)],
  ['recent_total', recentTotal, 228],
  ['depth', projectDepth, depthSeries([T1, T2], [2, 0])],
  ['landing_rate', projectLandingRate, landingRate([T1, T2], [2, 1.5])],
  ['attempts', projectAttempts, attemptsValue({ done: 3, conflict: 1 }, timedLatency(900, 4))],
]) {
  test(`${field}: the served Datum, stamped with the /merge-queue receipt as a copy`, () => {
    const data = mqData({ a: { ...entryWith(inQueue('fresh', 0)), [field]: runsDbRead('fresh', value) } });
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

  test(`${field}: an entry without the Datum names both the project and the field`, () => {
    const served = reader(mqData({ hive: entryWith(inQueue('fresh', 0)) }), 'hive');
    assert.equal(served.state, 'unknown');
    assert.equal(served.reason, `the /merge-queue entry for hive has no ${field} Datum`);
  });

  test(`${field}: an unread runs.db keeps the server's reason`, () => {
    const unread = runsDbRead('unknown');
    const served = reader(mqData({ a: { ...entryWith(inQueue('fresh', 0)), [field]: unread } }), 'a');
    assert.equal(served.state, 'unknown');
    assert.equal(served.reason, unread.reason);
  });
}

// ── recentTotal into the Recent-merges caption, as MergeTab wires it ────────

test('recentTotal: a window total that is a hole captions as unknown, never "of 0"', () => {
  const echo = Object.freeze({ requested: '24h', served: '24h', days: 1 });
  const holes = [
    ['an unread runs.db', { a: entryReading(runsDbRead('fresh', counts(0, 0)), runsDbRead('unknown')) }],
    ['an entry with no recent_total', { a: entryWith(inQueue('fresh', 0)) }],
  ];
  for (const [label, entries] of holes) {
    const total = recentTotal(mqData(entries), 'a');
    assert.equal(total.state, 'unknown', label);
    assert.equal(recentMergesCaption(0, total.value, echo), `showing 0 of ${EM_DASH} in 24h`, label);
  }
});

test('recentTotal: a measured window total captions as its count', () => {
  const echo = Object.freeze({ requested: '24h', served: '24h', days: 1 });
  const total = recentTotal(mqData({ a: entryReading(runsDbRead('fresh', counts(0, 0)), runsDbRead('fresh', 0)) }), 'a');
  assert.equal(recentMergesCaption(0, total.value, echo), 'showing 0 of 0 in 24h');
});

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

test('speculativeOver: a hole in any project is a hole in the total — a rate over a subset is no bound', () => {
  const data = mqData({
    a: entryReading(runsDbRead('fresh', counts(3, 1)), runsDbRead('fresh', 4)),
    b: { ...entryReading(runsDbRead('unknown'), runsDbRead('unknown')), runs_db_present: true },
  });

  const total = speculativeOver(data, null);

  assert.equal(total.state, 'unknown');
  assert.match(total.reason, /\bb: /);
  assert.deepEqual(missingParts(total), []);
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

// ── attemptCount: how many attempts one window's outcome chart holds ──────

test('attemptCount: sums the outcome chart, every attempt counted once', () => {
  assert.equal(attemptCount(attemptsValue({ done: 3, conflict: 2, blocked: 1 })), 6);
});

test('attemptCount: a measured window with no attempts counts 0', () => {
  assert.equal(attemptCount(attemptsValue({})), 0);
});

// ── attemptsOver: the 'Merges (window)' tile ───────────────────────────────

const freshAttempts = outcomeCounts => runsDbRead('fresh', attemptsValue(outcomeCounts));

test('attemptsOver: sums every payload project\'s attempts, or only its scope\'s', () => {
  const data = mqData({
    a: entryWindow(runsDbRead('fresh', depthSeries([], [])), freshAttempts({ done: 3, conflict: 1 })),
    b: entryWindow(runsDbRead('fresh', depthSeries([], [])), freshAttempts({ done: 2 })),
  });

  const total = attemptsOver(data, null);

  assert.equal(total.state, 'fresh');
  assert.equal(total.value, 6);
  assert.equal(attemptsOver(data, ['b']).value, 2);
});

test('attemptsOver: a present-but-unread project makes the total a lower bound that names it', () => {
  const data = mqData({
    a: entryWindow(runsDbRead('fresh', depthSeries([], [])), freshAttempts({ done: 3 })),
    b: entryWindow(runsDbRead('unknown'), runsDbRead('unknown')),
  });

  const total = attemptsOver(data, null);

  assert.equal(total.state, 'lower_bound');
  assert.equal(total.value, 3);
  assert.equal(datumView(total, { now: NOW }).text, `${LOWER_BOUND_PREFIX}3`);
  assert.match(total.reason, /\bb: /);
  assert.equal(totalHint(total), 'b unreadable');
});

test('attemptsOver: every in-scope project unread has no total', () => {
  const data = mqData({
    a: entryWindow(runsDbRead('unknown'), runsDbRead('unknown')),
    b: entryWindow(runsDbRead('unknown'), runsDbRead('unknown')),
  });

  assert.equal(attemptsOver(data, null).state, 'unknown');
});

test('attemptsOver: a stale part makes the total stale', () => {
  const data = mqData({
    a: entryWindow(runsDbRead('fresh', depthSeries([], [])), freshAttempts({ done: 3 })),
    b: entryWindow(runsDbRead('stale', depthSeries([], [])), runsDbRead('stale', attemptsValue({ done: 1 }))),
  });

  const total = attemptsOver(data, null);

  assert.equal(total.state, 'stale');
  assert.equal(total.value, 4);
  assert.match(total.reason, /\bb: /);
});

test('attemptsOver: before the first payload, the total is not yet fetched', () => {
  const total = attemptsOver(mqData({}, null), null);
  assert.equal(total.state, 'unknown');
  assert.equal(total.reason, 'not yet fetched');
});

test('attemptsOver: an unprobed project is still counted — its attempts come from its runs.db', () => {
  const data = mqData({
    a: entryWindow(runsDbRead('fresh', depthSeries([], [])), freshAttempts({ done: 3 })),
    c: { ...unprobedEntry('c'), depth: runsDbRead('fresh', depthSeries([], [])), attempts: freshAttempts({ done: 2 }) },
  });

  assert.equal(attemptsOver(data, null).value, 5);
});

// ── attemptsHistory: the 'Merges (window)' spark ───────────────────────────

test('attemptsHistory: one project is its own depth values', () => {
  const data = mqData({ a: entryWindow(runsDbRead('fresh', depthSeries([T1, T2, T3], [4, 0, 2])), freshAttempts({})) });
  assert.deepEqual(attemptsHistory(data, ['a']), [4, 0, 2]);
});

test('attemptsHistory: several projects sum per label over the union, a trimmed label counting 0', () => {
  // b's trim dropped its leading empty T1 bucket: every project's buckets are
  // epoch-aligned and end at the same bucket, so T1 is a 0 for b, not a miss.
  const data = mqData({
    b: entryWindow(runsDbRead('fresh', depthSeries([T2, T3], [1, 5])), freshAttempts({})),
    a: entryWindow(runsDbRead('fresh', depthSeries([T1, T2, T3], [4, 3, 2])), freshAttempts({})),
  });

  assert.deepEqual(attemptsHistory(data, ['a', 'b']), [4, 4, 7]);
  assert.deepEqual(attemptsHistory(data, null), [4, 4, 7]);
});

test('attemptsHistory: a project in scope whose depth is unread or absent draws no spark at all', () => {
  // Summing the others alone would draw a dip that never happened.
  const readable = entryWindow(runsDbRead('fresh', depthSeries([T1, T2], [4, 3])), freshAttempts({}));
  const holes = [
    ['an unread depth', { a: readable, b: entryWindow(runsDbRead('unknown'), runsDbRead('unknown')) }],
    ['an entry with no depth', { a: readable, b: entryWith(inQueue('fresh', 0)) }],
    ['an absent project', { a: readable }],
  ];
  for (const [label, entries] of holes) {
    assert.deepEqual(attemptsHistory(mqData(entries), ['a', 'b']), [], label);
  }
});

// ── worstP95Over: the 'p95 latency · worst project' tile ───────────────────

const timedWindow = (p95, withDuration) => entryWindow(
  runsDbRead('fresh', depthSeries([], [])),
  runsDbRead('fresh', attemptsValue({ done: withDuration }, timedLatency(p95, withDuration))),
);

test('worstP95Over: the worst p95 over the timed projects in scope, and how many were timed', () => {
  const data = mqData({
    a: timedWindow(900, 3),
    b: timedWindow(4200, 2),
    c: entryWindow(runsDbRead('fresh', depthSeries([], [])), freshAttempts({ done: 2 })),
  });

  const worst = worstP95Over(data, null);

  assert.equal(worst.state, 'fresh');
  assert.deepEqual(worst.value, { p95_ms: 4200, timed_projects: 2 });
  assert.deepEqual(worstP95Over(data, ['a', 'c']).value, { p95_ms: 900, timed_projects: 1 });
});

test('worstP95Over: a hole in any project makes the worst readable p95 a lower bound naming it', () => {
  const data = mqData({ a: timedWindow(900, 3), b: entryWindow(runsDbRead('unknown'), runsDbRead('unknown')) });

  const worst = worstP95Over(data, null);

  assert.equal(worst.state, 'lower_bound');
  assert.deepEqual(worst.value, { p95_ms: 900, timed_projects: 1 });
  assert.equal(datumView(worst, { now: NOW, format: v => String(v.p95_ms) }).text, `${LOWER_BOUND_PREFIX}900`);
  assert.match(worst.reason, /\bb: /);
  assert.deepEqual(missingParts(worst), ['b']);
});

test('worstP95Over: a lower bound whose readable projects have no timed merge is unknown, still naming the hole', () => {
  const data = mqData({
    a: entryWindow(runsDbRead('fresh', depthSeries([], [])), freshAttempts({ done: 2 })),
    b: entryWindow(runsDbRead('unknown'), runsDbRead('unknown')),
  });

  const worst = worstP95Over(data, null);

  assert.equal(worst.state, 'unknown');
  assert.match(worst.reason, /no timed merges in this window/);
  assert.match(worst.reason, /\bb: /);
});

test('worstP95Over: no timed project in scope has no worst centile to show', () => {
  const data = mqData({ a: entryWindow(runsDbRead('fresh', depthSeries([], [])), freshAttempts({ done: 2 })) });

  const worst = worstP95Over(data, null);

  assert.equal(worst.state, 'unknown');
  assert.equal(worst.reason, 'no timed merges in this window');
});

test('worstP95Over: before the first payload, it is not yet fetched', () => {
  const worst = worstP95Over(mqData({}, null), null);
  assert.equal(worst.state, 'unknown');
  assert.equal(worst.reason, 'not yet fetched');
});

// ── Rule 1: a project with no runs.db file is outside every runs.db total ──

test('the runs.db totals leave out a project with no runs.db file — never a lower bound over it', () => {
  const measured = {
    a: { ...timedWindow(900, 3), speculative: runsDbRead('fresh', counts(3, 1)) },
    b: { ...timedWindow(4200, 2), speculative: runsDbRead('fresh', counts(1, 1)) },
  };
  const without = mqData(measured);
  const withAbsent = mqData({ ...measured, gone: noRunsDbEntry() });

  for (const total of [attemptsOver, worstP95Over, speculativeOver]) {
    const expected = total(without, null);
    const actual = total(withAbsent, null);
    assert.equal(actual.state, expected.state, total.name);
    assert.deepEqual(actual.value, expected.value, total.name);
    assert.notEqual(actual.state, 'lower_bound', total.name);
    assert.deepEqual(missingParts(actual), [], total.name);
    assert.deepEqual(total(withAbsent, ['a', 'gone']).value, total(without, ['a']).value, total.name);
  }
});

test('attemptsHistory: a project with no runs.db file is outside the spark', () => {
  const data = mqData({
    a: entryWindow(runsDbRead('fresh', depthSeries([T1, T2], [4, 3])), freshAttempts({})),
    gone: noRunsDbEntry(),
  });

  assert.deepEqual(attemptsHistory(data, null), [4, 3]);
  assert.deepEqual(attemptsHistory(data, ['a', 'gone']), [4, 3]);
});

test('attemptsOver: a scope of only projects with no runs.db file has no total, and says why', () => {
  const total = attemptsOver(mqData({ gone: noRunsDbEntry(), lost: noRunsDbEntry() }), null);

  assert.equal(total.state, 'unknown');
  assert.match(total.reason, /runs\.db/);
});

// ── totalHint: the tile hint naming what a lower-bound total could not read ─

test('totalHint: a lower-bound total appends the unreadable projects to what it covered', () => {
  const data = mqData({
    a: entryWindow(runsDbRead('fresh', depthSeries([], [])), freshAttempts({ done: 3 })),
    b: entryWindow(runsDbRead('unknown'), runsDbRead('unknown')),
  });

  assert.equal(totalHint(attemptsOver(data, null), '2 projects'), '2 projects · b unreadable');
});

test('totalHint: several unreadable projects are joined in scope order', () => {
  const data = mqData({
    a: entryWindow(runsDbRead('fresh', depthSeries([], [])), freshAttempts({ done: 3 })),
    b: entryWindow(runsDbRead('unknown'), runsDbRead('unknown')),
    c: entryWindow(runsDbRead('unknown'), runsDbRead('unknown')),
  });

  assert.equal(totalHint(attemptsOver(data, null)), 'b, c unreadable');
});

test('totalHint: a total missing nothing hints only what it covered, or nothing', () => {
  const fresh = attemptsOver(
    mqData({ a: entryWindow(runsDbRead('fresh', depthSeries([], [])), freshAttempts({ done: 3 })) }),
    null,
  );
  const unknown = attemptsOver(mqData({ a: entryWindow(runsDbRead('unknown'), runsDbRead('unknown')) }), null);

  assert.equal(totalHint(fresh, 'x'), 'x');
  assert.equal(totalHint(fresh), undefined);
  assert.equal(totalHint(unknown), undefined);
});
