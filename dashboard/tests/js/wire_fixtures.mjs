// Builders for DF_DATA wire payloads: the shapes the dashboard's routes serve,
// for the render tests to seed through dom_harness.mjs::seedDashboardData.
// Pure: no window, no assertions. Where a shape depends on the client's own
// vocabulary or endpoint table, that is passed in rather than restated here.
//
// A builder may leave out a key the route serves, but never emit one it does
// not: boundary_wire_fixtures.test.mjs checks each builder against the bodies
// the real routes serve, so a server-side rename fails there instead of
// leaving the render tests green on a shape nothing serves.
//
// A helper module, not a test: the `*.test.mjs` glob skips it.

// The instant every built Datum was measured, and every receipt was served.
const WIRE_AS_OF = '2026-10-14T10:00:00+00:00';

// The five-key envelope data/datum.py::Datum.to_wire() emits; fresh unless
// overridden. datum.js::isDatum refuses anything missing a key.
export function wireDatum(value, overrides = {}) {
  return { value, as_of: WIRE_AS_OF, state: 'fresh', reason: null, freshness_bound_seconds: 60, ...overrides };
}

// data/census.py::TaskCensus.to_wire(): a count per status (absent ones 0),
// plus each view's and sub-view's sum, grouped by the vocabulary task_vocab.js
// publishes (`window.DF_TASK_VOCAB`).
export function taskCensus(counts, vocab) {
  const all = Object.fromEntries(vocab.MEMBERS.map(status => [status, counts[status] ?? 0]));
  const sum = statuses => statuses.reduce((n, status) => n + all[status], 0);
  const sums = groups => Object.fromEntries(Object.entries(groups).map(([key, statuses]) => [key, sum(statuses)]));
  return { counts: all, total: sum(vocab.MEMBERS), views: sums(vocab.VIEWS), sub_views: sums(vocab.SUB_VIEWS) };
}

// One TASKS_SNAPSHOT[project] entry: the five keys
// data/task_snapshot.py::TaskSnapshot.to_wire() emits, census and rows each a
// Datum measured at the same instant.
export function tasksSnapshotEntry(rows, census) {
  const inProgress = rows.filter(row => row.status === 'in-progress');
  const stranded = inProgress.filter(row => row.stranded).length;
  return {
    census: wireDatum(census),
    rows: wireDatum(rows),
    in_progress_live: inProgress.length - stranded,
    in_progress_stranded: stranded,
    skew_seconds: 0,
  };
}

// The endpoint data.js polls for `key`, as the key its receipt is filed under,
// and the spec that endpoint's row declares for `key`. `loader` is data.js's
// `window.DF_DATA_LOADER`, whose endpoint table is the one source of both.
export function servingEndpoint(loader, key, win = '24h') {
  const served = Object.entries(loader.endpointsFor(win)).find(([, keySpecs]) => key in keySpecs);
  if (!served) throw new Error(`data.js polls no endpoint that serves ${key}`);
  const [url, keySpecs] = served;
  return { stateKey: loader.pollKey(url), spec: keySpecs[key] };
}

// The __receipt entries a poll records for the endpoints serving `keys`,
// received now so nothing reads as aged.
export function receiptsFor(loader, keys, win = '24h') {
  const receipt = { servedAt: WIRE_AS_OF, receivedAt: Date.now() };
  return Object.fromEntries(keys.map(key => [servingEndpoint(loader, key, win).stateKey, receipt]));
}

// `count` sample labels `stepMinutes` apart ending at WIRE_AS_OF, oldest first.
function sampleLabels(count, stepMinutes = 60) {
  const newest = Date.parse(WIRE_AS_OF);
  return Array.from({ length: count }, (_, i) => new Date(newest - (count - 1 - i) * stepMinutes * 60_000).toISOString());
}

// One burndown block, as data/redux_api.py::shape_burndown serves BURNDOWN and
// each BURNDOWN_BY_PROJECT entry. It holds:
//   - `sampleCount` labels;
//   - a non-empty series per census member, so every band draws;
//   - the rows' in-progress split;
//   - the `latest` and `forecast` Datums;
//   - the parity verdict's fields. Only the aggregate block names the
//     breaching `projects`; leave them out for a per-project block.
export function burndownBlock(vocab, sampleCount, { alarm, cap, peak, breachCount, projects }) {
  const labels = sampleLabels(sampleCount);
  const series = Object.fromEntries(
    Object.values(vocab.SERIES_KEYS).map((key, i) => [key, labels.map(() => i + 1)]),
  );
  const inProgress = series[vocab.SERIES_KEYS['in-progress']];
  const split = {
    in_progress_live: inProgress.map(n => n - 1),
    in_progress_stranded: inProgress.map(() => 1),
    in_progress_rows: inProgress,
  };
  const columns = { ...series, ...split };
  const counts = Object.fromEntries(Object.entries(columns).map(([key, values]) => [key, values.at(-1)]));
  return {
    labels,
    ...columns,
    latest: wireDatum({ counts, completed: 0, velocity: 0, window_days: 1 }),
    forecast: wireDatum({ forecast_low: 1, forecast_high: 2 }),
    parity_alarm: alarm,
    parity_cap: cap,
    parity_peak: peak,
    parity_breach_count: breachCount,
    ...(projects === undefined ? {} : { parity_projects: projects }),
  };
}

// One open escalation, as data/escalation_analytics.py::_lifespan_block lists
// it. `pinnedTaskIds` is the escalation server's pins annotation: omit it for a
// record whose annotation is unknown, and the item carries neither pins key;
// otherwise pins_recovery is whether it pins any task, derived as the server
// derives it.
export function openEscalationItem({ id, taskId, level = 1, ageSecs = 3600, breach6h = false, pinnedTaskIds }) {
  const item = { id, task_id: taskId, level, age_secs: ageSecs, breach_6h: breach6h };
  if (pinnedTaskIds === undefined) return item;
  return { ...item, pins_recovery: pinnedTaskIds.length > 0, pins_recovery_task_ids: pinnedTaskIds };
}

// The ESCALATION_ANALYTICS payload, one per_project entry per
// `{project, openItems}`. Each lifespan block lists only its open items; the
// origin and workflow blocks are empty, so their panels render their empty
// states, and no escalation is terminal.
export function escalationAnalytics(projects) {
  return {
    generated_at: WIRE_AS_OF,
    parse_failures: 0,
    regime_markers: [],
    per_project: projects.map(({ project, openItems }) => ({
      project,
      origin: {},
      lifespan: {
        percentiles_by_level: {},
        l1_to_l2_promotion: { count: 0, p50_secs: null, p90_secs: null },
        samples: [],
        open_items: openItems,
      },
      workflow: {},
      terminal: 0,
    })),
  };
}

// One MERGE_QUEUE[project] block, as data/redux_api.py::shape_merge_queue
// serves it: a measured, empty queue whose depth series holds `buckets` labels
// ending at WIRE_AS_OF, `bucketMinutes` apart, every count zero but the newest.
// A null `bucketMinutes` is a series that names no width; its labels are then
// hourly.
export function mergeQueueBlock({ bucketMinutes = null, buckets = 3 } = {}) {
  const labels = sampleLabels(buckets, bucketMinutes ?? 60);
  return {
    depth: { labels, values: labels.map((_, i) => (i === buckets - 1 ? 1 : 0)), bucket_minutes: bucketMinutes },
    outcomes: { labels: [], values: [], colors: [] },
    latency: { p50: 0, p95: 0, p99: 0, mean_ms: 0, with_duration: 0, without_duration: 0 },
    recent: [],
    active: [],
    recent_total: wireDatum(0),
    speculative: wireDatum({ hit_count: 0, discard_count: 0, total: 0, hit_rate: null }),
    in_queue: wireDatum(0),
    live_probe_configured: true,
    active_spark: { labels: [], values: [] },
    halt: { offline: true },
    train_events: [],
    metrics: {},
  };
}

// One ORCHESTRATORS row, as /orchestrators serves it.
export function orchestratorRow({ pid, project, running }) {
  return { pid, project, running };
}

// One PROJECTS row, as /orchestrators serves it.
export function projectRow(id) {
  return { id, name: id };
}
