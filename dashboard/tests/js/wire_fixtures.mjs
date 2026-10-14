// Builders for DF_DATA wire payloads: the shapes the dashboard's routes serve,
// for the render tests to seed through dom_harness.mjs::seedDashboardData.
// Pure: no window, no assertions. Where a shape depends on the client's own
// vocabulary or endpoint table, that is passed in rather than restated here.
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

// The __receipt entries a poll records for the endpoints serving `keys`, keyed
// as data.js's refreshOne keys them, and received now so nothing reads as aged.
// `loader` is data.js's `window.DF_DATA_LOADER`, whose endpoint table says which
// endpoint serves each key.
export function receiptsFor(loader, keys, win = '24h') {
  const endpoints = Object.entries(loader.endpointsFor(win));
  const receipt = { servedAt: WIRE_AS_OF, receivedAt: Date.now() };
  return Object.fromEntries(
    keys.map(key => {
      const served = endpoints.find(([, keySpecs]) => key in keySpecs);
      if (!served) throw new Error(`data.js polls no endpoint that serves ${key}`);
      return [loader.pollKey(served[0]), receipt];
    }),
  );
}

// `count` hourly sample labels ending at WIRE_AS_OF, oldest first.
function sampleLabels(count) {
  const newest = Date.parse(WIRE_AS_OF);
  return Array.from({ length: count }, (_, i) => new Date(newest - (count - 1 - i) * 3_600_000).toISOString());
}

// One burndown block, as data/redux_api.py::shape_burndown serves BURNDOWN and
// each BURNDOWN_BY_PROJECT entry. It holds:
//   - `sampleCount` labels;
//   - a non-empty series per census member, so every band draws;
//   - the rows' in-progress split;
//   - the `latest` and `forecast` Datums;
//   - `parity`, the verdict's parity_* fields, passed through as served.
export function burndownBlock(vocab, sampleCount, parity) {
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
    ...parity,
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
// origin, workflow and terminal blocks are empty, so their panels render their
// empty states.
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
      terminal: {},
    })),
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
