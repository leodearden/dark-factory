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

// One TASKS_SNAPSHOT[project].rows entry as active_tasks.py::_build_task_row
// emits it, holding only the keys the render tests read.
export function taskRow(overrides = {}) {
  return {
    id: '42', project: 'p', title: 'TT', status: 'in-progress', agent: null, stranded: false, deps: [],
    created_at: '2026-10-14T09:00:00.000Z', created_at_source: 'stamped',
    ...overrides,
  };
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
// `{project, openItems, origin?, workflow?, samples?}`, carrying
// `regimeMarkers` ({date, label, tasks}, as
// data/escalation_analytics.py::load_regime_markers serves them). Origin and
// workflow blocks default to empty and lifespan samples to none, so those
// panels render their empty states; no escalation is terminal.
export function escalationAnalytics(projects, { regimeMarkers = [] } = {}) {
  return {
    generated_at: WIRE_AS_OF,
    parse_failures: 0,
    regime_markers: regimeMarkers,
    per_project: projects.map(({ project, openItems = [], origin = {}, workflow = {}, samples = [] }) => ({
      project,
      origin,
      lifespan: {
        percentiles_by_level: {},
        l1_to_l2_promotion: { count: 0, p50_secs: null, p90_secs: null },
        samples,
        open_items: openItems,
      },
      workflow,
      terminal: 0,
    })),
  };
}

// One ESCALATIONS row, as data/redux_api.py::shape_escalations serves it: the
// escalation record's fields plus `project` and its task card `task`, never
// `project_root`. `deliveredCheckOwner` is emitted only when given, so leaving
// it out is real absence (a record written before task 7514), not an
// undefined-valued key; pass null for a record that has the key but no owner.
export function escalationRow({
  id, taskId, level = 2, status = 'pending', severity = 'critical', category = 'dependency_capability',
  summary, detail = '', project, deliveredCheckOwner, task,
}) {
  const row = {
    id,
    task_id: taskId,
    level,
    status,
    severity,
    category,
    summary,
    detail,
    project,
    task: task ?? wireDatum({ id: Number(taskId), title: `task ${taskId}`, status: 'blocked' }),
  };
  return deliveredCheckOwner === undefined ? row : { ...row, delivered_check_owner: deliveredCheckOwner };
}

const ESCALATION_LEVELS = [0, 1, 2];
const ESCALATION_STATUSES = ['pending', 'resolved', 'dismissed'];

// data/escalations.py::_bucket's counts over `rows`.
function escalationSummary(rows) {
  const tally = (keys, field) => Object.fromEntries(keys.map(key => [key, rows.filter(row => row[field] === key).length]));
  return { by_level: tally(ESCALATION_LEVELS, 'level'), by_status: tally(ESCALATION_STATUSES, 'status'), skipped_count: 0 };
}

// The corpus views over `rows` with nothing archived: every pending row is
// both in the live queue and open in history.
function escalationViews(rows) {
  const pending = rows.filter(row => row.status === 'pending').length;
  return { queue_pending: wireDatum(pending), open_in_history: wireDatum(pending) };
}

// The ESCALATIONS payload, as data/redux_api.py::shape_escalations serves it:
// one orchestrator subsection per `{project, rows}`, none with skipped files,
// and every count and view computed from the rows. Those counts re-derive the
// server's; boundary_wire_fixtures.test.mjs checks them against a served body
// only for one pending row, so rows the server counts differently (archived,
// non-pending) are not covered.
export function escalationsPayload(subsections) {
  const allRows = subsections.flatMap(({ rows }) => rows);
  return {
    subsections: subsections.map(({ project, rows }) => ({
      id: project,
      label: project,
      kind: 'orchestrator',
      escalations: rows,
      skipped: [],
      summary: escalationSummary(rows),
      views: escalationViews(rows),
    })),
    summary: escalationSummary(allRows),
    views: escalationViews(allRows),
  };
}

// The unknown triad data/datum.py::unknown_datum emits, for a runs.db that
// could not be read.
function unreadWireDatum(reason) {
  return wireDatum(null, { as_of: null, state: 'unknown', reason });
}

// One MERGE_QUEUE[project] block, as data/redux_api.py::shape_merge_queue
// serves it: a measured, empty queue whose depth series holds `buckets` labels
// ending at WIRE_AS_OF, `bucketMinutes` apart, every count zero but the newest.
// A null `bucketMinutes` is a series that names no width; its labels are then
// hourly. `outcomes` counts the window's untimed attempts by outcome. With
// `unread`, the project's runs.db could not be read, so its depth and attempts
// are unknown.
export function mergeQueueBlock({ bucketMinutes = null, buckets = 3, outcomes = {}, unread = false } = {}) {
  const labels = sampleLabels(buckets, bucketMinutes ?? 60);
  const outcomeLabels = Object.keys(outcomes);
  const outcomeValues = outcomeLabels.map(label => outcomes[label]);
  const attempts = outcomeValues.reduce((sum, n) => sum + n, 0);
  const unreadReason = 'the merge_attempt events could not be read from runs.db';
  return {
    depth: unread
      ? unreadWireDatum(unreadReason)
      : wireDatum({ labels, values: labels.map((_, i) => (i === buckets - 1 ? 1 : 0)), bucket_minutes: bucketMinutes }),
    attempts: unread
      ? unreadWireDatum(unreadReason)
      : wireDatum({
        outcomes: { labels: outcomeLabels, values: outcomeValues, colors: outcomeLabels.map(() => '#888888') },
        latency: { p50: null, p95: null, p99: null, mean_ms: null, with_duration: 0, without_duration: attempts },
      }),
    recent: [],
    active: [],
    recent_total: wireDatum(0),
    speculative: wireDatum({ hit_count: 0, discard_count: 0, total: 0, hit_rate: null }),
    in_queue: wireDatum(0),
    live_probe_configured: true,
    active_spark: { labels: [], values: [] },
    halt: { offline: true },
    train_events: [],
    train_throughput: wireDatum({}),
    metrics: {},
  };
}

// The SCHEDULER payload, as /scheduler serves it: the keys data.js seeds it
// with, every project in `offlineProjects` listed offline, nothing paused.
export function schedulerSnapshot({ offlineProjects = [], rows = [], modules = [] } = {}) {
  return {
    rows,
    modules,
    pin_queue: [],
    events_by_task: {},
    snapshot_at: WIRE_AS_OF,
    offline: offlineProjects.length > 0,
    offline_projects: offlineProjects,
    paused: false,
    paused_projects: [],
  };
}

// One SCHEDULER row: the modules task `taskId` of `project` declares it locks.
export function schedulerLockRow({ project, taskId, lockSet }) {
  return { project, task_id: taskId, lock_set: lockSet };
}

// One SCHEDULER.modules entry: who holds `path` in `project`, and who has it
// parked. Every lock field is null unless given, which reads as a free module.
export function schedulerModule({ project, path, holder = null, holderProject = null, parkedBy = null, parkedOwnerLive = null }) {
  return { project, path, holder, holder_project: holderProject, parked_by: parkedBy, parked_owner_live: parkedOwnerLive };
}

// One ORCHESTRATORS row, as /orchestrators serves it.
export function orchestratorRow({ pid, project, running }) {
  return { pid, project, running };
}

// One PROJECTS row, as /orchestrators serves it.
export function projectRow(id) {
  return { id, name: id };
}

// The MEMORY_EVALS builders below are not yet checked by
// boundary_wire_fixtures.test.mjs: no boundary payload serves MEMORY_EVALS.

// One open escalation, as data/memory_evals.py::_escalation_projection projects
// it onto a metric row, the storm banner or unmatched_escalations.
export function memoryEvalEscalation(id, overrides = {}) {
  return {
    id,
    summary: 'memory-eval regression',
    severity: 'blocking',
    level: 1,
    created_at: WIRE_AS_OF,
    dedupe_fingerprint: `fp-${id}`,
    ...overrides,
  };
}

// One metric row, as data/memory_evals.py::_build_eval emits it: unjudged and
// unlinked unless overridden.
export function memoryEvalRow(overrides = {}) {
  return {
    metric_id: 'canonical-in-top-5',
    kind: 'proportion',
    rule_kind: 'proportion',
    current_value: 0.9,
    n: 30,
    denominator: 30,
    direction: 'lower_is_worse',
    trend: { labels: ['20260704T031500Z', '20260705T031500Z'], values: [0.9, 0.9] },
    verdict: null,
    verdict_detail: null,
    value: null,
    item: null,
    limit_ref: null,
    fingerprint: null,
    run_stamp: null,
    escalation: null,
    parity: 'unjudged',
    ...overrides,
  };
}

// One eval card, as data/memory_evals.py::_build_eval emits it, over the run
// axis of its first metric's trend: fresh, untruncated, with no limits
// artifact. memoryEvalsPayload copies the run's storm_escape onto it.
export function memoryEvalCard(evalId, metrics) {
  const runStamps = metrics[0]?.trend.labels ?? [];
  return {
    eval_id: evalId,
    latest_run_stamp: runStamps.at(-1) ?? null,
    latest_run_age_seconds: 3600,
    stale: false,
    limits: null,
    storm_escape: null,
    run_stamps: runStamps,
    run_count: runStamps.length,
    runs_on_disk: runStamps.length,
    truncated: false,
    corpus: null,
    metrics,
  };
}

// The MEMORY_EVALS payload, as data/redux_api.py::shape_memory_evals serves it,
// with no artifact issues. Each card carries the run-scoped storm_escape, as
// data/memory_evals.py::_build_payload hands every card the top-level block.
export function memoryEvalsPayload({
  evals, storm_escape = null, unmatched_escalations = [], escalation_queue = 'reconciliation',
}) {
  return {
    generated_at: WIRE_AS_OF,
    root_present: true,
    storm_escape,
    evals: evals.map(card => ({ ...card, storm_escape })),
    issues: [],
    issue_count: 0,
    unmatched_escalations,
    escalation_queue,
  };
}

// The QUALITY value and its metrics, as
// dashboard/src/dashboard/data/quality_series.py::QualitySeries.to_wire serves
// them; that method owns the shape.

const NO_GIT_DISAGREEMENT = { off_first_parent: 0, merges_without_landing: 0, late_claims: 0, disagreement: 0 };
const NO_COST_MODEL = { a: null, b: null, reference_cost_100: null, residual_median: null, state: null };
const NO_BASELINE = { p0: null, state: null, since_ordinal: null };

// One axis point per entry of `axis`, ordinals 0, 1, ... landing an hour apart
// and ending at WIRE_AS_OF; each entry overrides its point's fields, so `{}` is
// a first-parent, first-claim landing. A unit is its merge sha, or for a late
// claim its own unit of one, unless the entry names it.
function qualityAxis(axis) {
  const landedAt = sampleLabels(axis.length);
  return axis.map((overrides, ordinal) => {
    const point = {
      ordinal,
      merge_sha: `sha-${ordinal}`,
      task_id: String(1000 + ordinal),
      landed_at: landedAt[ordinal],
      git_first_parent: true,
      late_claim: false,
      ...overrides,
    };
    const ownUnit = point.late_claim ? `${point.merge_sha}:${point.task_id}` : point.merge_sha;
    return { ...point, unit_id: overrides.unit_id ?? ownUnit };
  });
}

// A measured headline over the non-null `points`: a rate's k events of n, or a
// count's or a cost's centre. The band is a placeholder around the figure.
function qualityHeadline(kind, points, form) {
  const measured = points.filter(x => x !== null);
  const total = measured.reduce((sum, x) => sum + x, 0);
  const value = measured.length === 0 ? 0 : total / measured.length;
  return wireDatum({
    form,
    n: measured.length,
    k: kind === 'rate' ? total : null,
    value,
    lo: Math.max(0, value - 0.1),
    hi: kind === 'rate' ? Math.min(1, value + 0.1) : value + 0.1,
  });
}

// One QUALITY.metrics[id] entry with one point per landing. A null
// `measuredSince` is the no-producer state (INV-13): its rule names no start
// and its headline is unknown, saying so. The rolling band is unmeasured
// (null) at every point unless `rolling` is given, and there is no control
// chart unless `ewma`, `alarms` or `baseline` are given.
export function qualityMetric({
  kind = 'rate',
  points,
  headlineForm = kind === 'rate' ? 'rate' : kind === 'cost' ? 'median' : 'mean',
  measuredSince = WIRE_AS_OF,
  source = 'rejections',
  unit = kind === 'rate' ? 'share' : kind === 'cost' ? 'usd' : 'per landing',
  rolling = null,
  ewma = null,
  alarms = [],
  baseline = NO_BASELINE,
}) {
  const unmeasured = points.map(() => null);
  return {
    kind,
    rule: {
      counts: `${source} events`,
      excludes: `landings before the first ${source}`,
      population: 'landings',
      window_landings: 100,
      measured_since: measuredSince,
      source,
    },
    unit,
    points,
    rolling: rolling ?? { center: unmeasured, lo: unmeasured, hi: unmeasured, n: points.map(() => 0) },
    ewma,
    alarms,
    baseline,
    headline: measuredSince === null
      ? unreadWireDatum(`no producer has written here: runs.db ${source} has recorded nothing for this project`)
      : qualityHeadline(kind, points, headlineForm),
  };
}

// The QUALITY Datum's value: `axis` as qualityAxis builds it, `metrics` keyed
// by metric id (each from qualityMetric, one point per axis entry), and a git
// cross-check that disagrees nowhere unless `gitCheck` is given. No cost model
// is in force unless `costModel` names some of its fields, and there are no
// module rows (qualityModuleRow) or before/after comparisons
// (qualityComparison) unless they are given.
export function qualityPayload({
  axis, metrics, gitCheck = NO_GIT_DISAGREEMENT, costModel = {}, modules = [], beforeAfter = [],
}) {
  return {
    axis: qualityAxis(axis),
    metrics,
    cost_model: { ...NO_COST_MODEL, ...costModel },
    modules,
    before_after: beforeAfter,
    git_check: gitCheck,
  };
}

// One QUALITY.value.modules row, as quality_series.py::ModuleRow.to_wire
// serves it (that method owns the shape); a null `module` is the no-src
// bucket.
export function qualityModuleRow({
  module, landings = 10, tasks = 10, allocatedLines = 1200, allocatedUsd = 24, usdPer100Lines = 2, ranked = true,
}) {
  return {
    module, landings, tasks, allocated_lines: allocatedLines, allocated_usd: allocatedUsd,
    usd_per_100_lines: usdPer100Lines, ranked,
  };
}

// One QUALITY.value.before_after entry, as quality_series.py::BeforeAfter.to_wire
// serves it (that method owns the shape): measured unless `state` is 'not_yet',
// whose effect and band are then unmeasured.
export function qualityComparison({
  markerId, metric, nBefore = 300, nAfter = 120, effect = 0.05, lo = 0.01, hi = 0.09,
  state = 'measured', etaDays = null, minDetectable = null,
}) {
  const measured = state !== 'not_yet';
  return {
    marker_id: markerId, metric, n_before: nBefore, n_after: nAfter,
    effect: measured ? effect : null, lo: measured ? lo : null, hi: measured ? hi : null,
    state, eta_days: etaDays, min_detectable: minDetectable,
  };
}

const NO_SNAPSHOT_PRODUCER = 'no producer has written here: the nightly quality-metrics snapshot series has no writer yet';

// The whole /api/v2/dashboard/quality body, as api/quality.py::api_quality
// serves it: `quality` (a qualityPayload value) and `markers` (a
// qualityMarkers value) as fresh Datums, the snapshot Datum unknown because
// its producer has not landed, and the landing window echo `window` asked
// for and served.
export function qualityBody({ quality, markers = qualityMarkers([]), window = '100' }) {
  const landings = window === 'all' ? null : Number(window);
  return {
    QUALITY: wireDatum(quality),
    QUALITY_MARKERS: wireDatum(markers),
    QUALITY_SNAPSHOTS: unreadWireDatum(NO_SNAPSHOT_PRODUCER),
    WINDOW: { requested: window, served: window, landings },
    served_at: WIRE_AS_OF,
  };
}

// One QUALITY_MARKERS.value.markers entry, as
// dashboard/src/dashboard/data/quality_markers.py::Marker.to_wire serves it
// (that method owns the shape): a derived reviewer effort change in
// dark_factory placed at landing `ordinal`, applying to every series unless
// `series` names some. A range (an artefact) also names its `untilOrdinal`.
export function qualityMarker({
  id,
  ordinal,
  kind = 'effort',
  label = `marker ${id}`,
  series = null,
  untilOrdinal = null,
  until = null,
  ref = null,
  at = WIRE_AS_OF,
  scope = 'dark_factory',
  source = { type: 'routing', role: 'reviewer_comprehensive' },
  tasks = [],
}) {
  return { id, at, ordinal, until_ordinal: untilOrdinal, kind, scope, series, until, label, source, ref, tasks };
}

// The QUALITY_MARKERS Datum's value, as quality_markers.py::MarkerStream.to_wire
// serves it: `markers` as given, no availability flag, prompt coverage not yet
// written, and nothing skipped or unparsed.
export function qualityMarkers(markers) {
  return {
    markers,
    availability: [],
    prompt_coverage: {
      state: 'not_yet_written', since: null, roles_py_landings: null, counted_since: null,
      reason: 'no routing decision is folded yet',
    },
    curated_parse_failures: 0,
    skipped_records: 0,
  };
}
