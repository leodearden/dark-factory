// merge_queue.js — the CLIENT twin of dashboard/src/dashboard/data/merge_queue.py,
// and the ONE reader of the served "In queue now" datum: MergeTab's tile and
// its spark, the per-project "queued" pip, and the rail badge all read the
// queue here, so the rail and the tile cannot disagree. Named for its server
// twin (the datum.js precedent).
//
// A PLAIN-JS CLASSIC SCRIPT, NOT A .jsx MODULE, so its decisions are
// EXECUTABLE: dashboard/tests/js/merge_queue.test.mjs runs what no harness here
// can run inside a .jsx body. pins_recovery.js's header holds the CANONICAL
// statement of why.
//
// LOAD CONTRACT. It destructures window.DF_DATUM at module scope with no
// fallback, so index.html loads it after datum.js and before every JSX consumer
// (test_index_html.py pins the order). A browser classic `<script>` assigns
// `window.DF_MERGE_QUEUE`; node requires the same file as CommonJS once its test
// has put DF_ENDPOINT_STALENESS on a window shim.
//
// THE CLIENT NEVER ALTERS A SERVED COUNT. Each project's `in_queue` is the
// server's Datum — the live probe, or its own sampled history's last value,
// stale and saying why. The only envelopes built here are datum.js's own: a
// receipt stamp, an unknown placeholder, and a combined total in which a hole
// anywhere is a hole in the sum.
//
// A TOTAL'S SCOPE IS THE PROBED PROJECTS. A project the server serves with
// `live_probe_configured: false` has no queue this dashboard can read (a
// configured root may run no orchestrator), so it is outside every total and
// spark rather than a hole that blanks them for good. A probed project that
// cannot be read IS a hole: its own datum says why.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const {
  servedDatum: servedQueueDatum,
  unknownDatum: unknownQueueDatum,
  combinedDatum: combineInQueueDatums,
} = window.DF_DATUM;

// data.js registers MERGE_QUEUE as PLAIN, so the in_queue Datums inside it
// arrive carrying no receipt of their own; this endpoint's receipt stamps them.
const MERGE_QUEUE_ENDPOINT = '/api/v2/dashboard/merge-queue';

const QUEUE_NOT_YET_FETCHED = 'not yet fetched';

function queueReceipts(data) {
  return data.__receipt || {};
}

function queueEntries(data) {
  return data.MERGE_QUEUE || {};
}

// `projects` null means every project the payload carries. A project the
// payload does not carry stays in scope, so a total over it is a hole.
function queueScope(data, projects) {
  const entries = queueEntries(data);
  return (projects === null ? Object.keys(entries) : projects).filter(project => {
    const entry = entries[project];
    return !entry || entry.live_probe_configured !== false;
  });
}

function projectInQueue(data, project) {
  const entry = queueEntries(data)[project];
  const why = entry
    ? 'the /merge-queue entry for ' + project + ' has no in_queue Datum'
    : 'the /merge-queue payload has no entry for ' + project;
  return servedQueueDatum(entry && entry.in_queue, MERGE_QUEUE_ENDPOINT, why, queueReceipts(data));
}

function inQueueOver(data, projects) {
  if (!queueReceipts(data)[MERGE_QUEUE_ENDPOINT]) return unknownQueueDatum(QUEUE_NOT_YET_FETCHED);
  return combineInQueueDatums(
    queueScope(data, projects).map(project => [project, projectInQueue(data, project)]),
    counts => counts.reduce((sum, n) => sum + n, 0),
    'no project in scope has a live get_merge_queue probe configured',
  );
}

// The tile's spark: the in-scope projects' sampled counts summed label-wise,
// over only the labels EVERY one of them sampled. A label one project missed
// would sum the others alone, drawing a dip that never happened.
function inQueueHistory(data, projects) {
  const series = queueScope(data, projects).map(project => sampledCounts(queueEntries(data)[project]));
  if (series.length === 0) return [];
  const [first, ...rest] = series;
  return [...first.keys()]
    .filter(label => rest.every(counts => counts.has(label)))
    .sort()
    .map(label => series.reduce((sum, counts) => sum + counts.get(label), 0));
}

function sampledCounts(entry) {
  const spark = (entry && entry.active_spark) || {};
  const labels = spark.labels || [];
  const values = spark.values || [];
  return new Map(labels.map((label, i) => [label, Number(values[i]) || 0]));
}

// The latency centiles are computed over attempts with a recorded duration
// only; this says over how many, and how many the outcomes total holds beside
// them. A block without the split states nothing rather than inventing a zero.
function latencyCaption(latency) {
  const l = latency || {};
  if (!Number.isFinite(l.with_duration) || !Number.isFinite(l.without_duration)) return '';
  return 'of ' + l.with_duration + ' with recorded duration · ' + l.without_duration + ' without';
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const MERGE_QUEUE_API = {
  projectInQueue,
  inQueueOver,
  inQueueHistory,
  latencyCaption,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = MERGE_QUEUE_API;
}
if (typeof window !== 'undefined') {
  window.DF_MERGE_QUEUE = MERGE_QUEUE_API;
}
