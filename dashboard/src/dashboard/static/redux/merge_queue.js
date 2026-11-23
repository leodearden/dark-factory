// merge_queue.js — the CLIENT twin of dashboard/src/dashboard/data/merge_queue.py,
// and the ONE reader of the /merge-queue Datums — in_queue, speculative,
// recent_total, depth (the attempts tile's spark), landing_rate and attempts —
// and of each queued row's enqueue instant. MergeTab's tiles,
// pips and captions, the Overview live feed and the rail badge all read them
// here, so no two of them can disagree. Named for its server twin (the
// datum.js precedent).
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
// stale and saying why — and its `speculative`, `recent_total`, `depth`,
// `landing_rate` and `attempts` are its runs.db window reads, unknown when that read failed. The
// only envelopes built here are datum.js's own: a receipt stamp, an unknown
// placeholder, and a combined total. A lower-bound total over the readable
// projects that names the unread ones is a combination, not an alteration.
//
// A RUNS.DB TOTAL'S SCOPE IS THE PROJECTS WITH A RUNS.DB FILE. A project served
// `runs_db_present: false` has no runs.db to read, now or ever, so it is
// outside every speculative, attempts and latency total and spark rather than
// a hole that blanks them for good. A runs.db that exists but did not read IS
// a hole: its own datum says why.
//
// AN IN-QUEUE TOTAL'S SCOPE IS THE PROBED PROJECTS. A project served
// `live_probe_configured: false` has no queue this dashboard can read (a
// configured root may run no orchestrator), so it is outside every in-queue
// total and spark. A probed project whose probe fails stays in scope: with a
// sample in its history it counts stale, and without one (a stopped
// orchestrator) it is a hole. The two scopes are independent: the probe has no
// bearing on runs.db, nor runs.db on the probe.
//
// A HOLE IN SCOPE MAKES A TOTAL A LOWER BOUND. The in-queue, attempts and
// worst-p95 totals combine monotonically (a sum, a max), so the readable
// projects' total is a true '≥' of the whole: it is served lower_bound and
// totalHint names the unreadable projects. The hit rate is the exception: a
// ratio over a subset bounds the whole neither way, so a hole there is still a
// hole in the rate.
//
// SPARKS KEEP THE BLANK-ON-HOLE RULE. A spark over only the readable projects
// draws a dip that never happened, and a shape has no '≥' to carry.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const {
  servedDatum: servedQueueDatum,
  unknownDatum: unknownQueueDatum,
  combinedDatum: combineQueueDatums,
  lowerBoundCombinedDatum: lowerBoundQueueDatums,
  missingParts: missingQueueParts,
  EM_DASH: QUEUE_EM_DASH,
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

// `projects` null means every project the payload carries; a project whose
// entry serves *flag* false is outside the scope. A project the payload does
// not carry stays in scope, so a total over it is a hole.
function scopeFlagged(data, projects, flag) {
  const entries = queueEntries(data);
  return (projects === null ? Object.keys(entries) : projects).filter(project => {
    const entry = entries[project];
    return !entry || entry[flag] !== false;
  });
}

function queueScope(data, projects) {
  return scopeFlagged(data, projects, 'live_probe_configured');
}

function runsDbScope(data, projects) {
  return scopeFlagged(data, projects, 'runs_db_present');
}

// *project*'s served *field* Datum, stamped with this endpoint's receipt.
function projectServed(data, project, field) {
  const entry = queueEntries(data)[project];
  const why = entry
    ? 'the /merge-queue entry for ' + project + ' has no ' + field + ' Datum'
    : 'the /merge-queue payload has no entry for ' + project;
  return servedQueueDatum(entry && entry[field], MERGE_QUEUE_ENDPOINT, why, queueReceipts(data));
}

function projectInQueue(data, project) {
  return projectServed(data, project, 'in_queue');
}

function inQueueOver(data, projects) {
  if (!queueReceipts(data)[MERGE_QUEUE_ENDPOINT]) return unknownQueueDatum(QUEUE_NOT_YET_FETCHED);
  return lowerBoundQueueDatums(
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

// When a queued row joined the queue: the probe instant its entry's in_queue
// was measured at, less the row's age at that instant. The server serves a
// queued row only beside a reachable probe's fresh in_queue, so the anchor
// exists whenever the row does; a row with no usable age has no instant, which
// is not "just now".
function queuedSince(entry, row) {
  const probedAt = Date.parse(entry && entry.in_queue ? entry.in_queue.as_of : undefined);
  const age = row ? row.age_secs : undefined;
  if (!Number.isFinite(probedAt) || !Number.isFinite(age) || age < 0) return null;
  return new Date(probedAt - age * 1000).toISOString();
}

// *project*'s speculative-merge counts and rate over the window.
function projectSpeculative(data, project) {
  return projectServed(data, project, 'speculative');
}

// How many merges *project*'s window holds; its `recent` rows are capped.
function recentTotal(data, project) {
  return projectServed(data, project, 'recent_total');
}

// The speculative counts summed over `projects` (null: every payload project).
// Strict: a hole anywhere is a hole in the rate. A window with no attempts has
// no rate, so the total is a hole rather than 0%.
function speculativeOver(data, projects) {
  const total = runsDbOver(data, projects, projectSpeculative, sumSpeculativeCounts, combineQueueDatums);
  if (total.state !== 'unknown' && total.value.total === 0) {
    return unknownQueueDatum(withTotalReason('no speculative attempts in this window', total));
  }
  return total;
}

// Each in-scope project's `reader` Datum, folded by `combine` under the
// `combineParts` hole policy; a hole before the endpoint's first receipt.
function runsDbOver(data, projects, reader, combine, combineParts) {
  if (!queueReceipts(data)[MERGE_QUEUE_ENDPOINT]) return unknownQueueDatum(QUEUE_NOT_YET_FETCHED);
  return combineParts(
    runsDbScope(data, projects).map(project => [project, reader(data, project)]),
    combine,
    'no project in scope has a runs.db',
  );
}

// *why*, then *total*'s own reason, so a hole the total read past stays named.
function withTotalReason(why, total) {
  return total.reason ? why + '; ' + total.reason : why;
}

// A tile's hint: what its total covered, then the projects it could not read;
// undefined when there is neither.
function totalHint(total, covered) {
  const missing = missingQueueParts(total);
  const parts = [covered, missing.length > 0 ? missing.join(', ') + ' unreadable' : null].filter(Boolean);
  return parts.length > 0 ? parts.join(' · ') : undefined;
}

function sumSpeculativeCounts(parts) {
  return parts.reduce(
    (sum, part) => ({
      hit_count: sum.hit_count + part.hit_count,
      discard_count: sum.discard_count + part.discard_count,
      total: sum.total + part.total,
    }),
    { hit_count: 0, discard_count: 0, total: 0 },
  );
}

// *project*'s merge_attempt counts per bucket over the window, leading empty
// buckets trimmed by the server. It feeds only the attempts tile's spark.
function projectDepth(data, project) {
  return projectServed(data, project, 'depth');
}

// *project*'s landings per day at each point, each over the trailing window the
// server names; leading zero points trimmed by the server.
function projectLandingRate(data, project) {
  return projectServed(data, project, 'landing_rate');
}

// *project*'s window of attempts: ONE read behind the donut and the latency.
function projectAttempts(data, project) {
  return projectServed(data, project, 'attempts');
}

// How many attempts one known attempts value holds: its donut's total.
function attemptCount(attempts) {
  return attempts.outcomes.values.reduce((sum, n) => sum + n, 0);
}

// The attempts summed over `projects` (null: every payload project). A
// measured window with no attempts is a measured 0.
function attemptsOver(data, projects) {
  return runsDbOver(
    data, projects, projectAttempts, parts => parts.reduce((sum, part) => sum + attemptCount(part), 0),
    lowerBoundQueueDatums,
  );
}

// The attempts tile's spark: the in-scope depth series summed per label over
// the union of their labels. Every series is epoch-aligned at one width and
// ends at the same bucket, so a label one series lacks is a leading zero the
// server trimmed, and counts 0 for it. An unread series is no such zero: one
// anywhere in scope blanks the spark rather than draw a dip that never happened.
function attemptsHistory(data, projects) {
  const depths = runsDbScope(data, projects).map(project => projectDepth(data, project));
  if (depths.some(depth => depth.state === 'unknown')) return [];
  const byLabel = new Map();
  for (const { value } of depths) {
    value.labels.forEach((label, i) => {
      byLabel.set(label, (byLabel.get(label) || 0) + (Number(value.values[i]) || 0));
    });
  }
  return [...byLabel.keys()].sort().map(label => byLabel.get(label));
}

// The worst p95 over the in-scope projects whose window holds a timed attempt,
// and how many did. No timed project has no worst centile: a hole, not 0.
function worstP95Over(data, projects) {
  const worst = runsDbOver(data, projects, projectAttempts, worstTimedP95, lowerBoundQueueDatums);
  if (worst.state !== 'unknown' && worst.value.timed_projects === 0) {
    return unknownQueueDatum(withTotalReason('no timed merges in this window', worst));
  }
  return worst;
}

function worstTimedP95(parts) {
  const timed = parts.map(part => part.latency.p95).filter(p95 => p95 !== null && p95 !== undefined);
  return { p95_ms: timed.length > 0 ? Math.max(...timed) : null, timed_projects: timed.length };
}

function hitRateText(counts) {
  if (!(counts.total > 0)) return QUEUE_EM_DASH;
  return Math.round((counts.hit_count / counts.total) * 100) + '%';
}

// The latency centiles are computed over attempts with a recorded duration
// only; this says over how many, and how many the outcomes total holds beside
// them. A block without the split states nothing rather than inventing a zero.
function latencyCaption(latency) {
  const l = latency || {};
  if (!Number.isFinite(l.with_duration) || !Number.isFinite(l.without_duration)) return '';
  return 'of ' + l.with_duration + ' with recorded duration · ' + l.without_duration + ' without';
}

// A span of whole minutes in its largest whole unit ('1-day', '6-h',
// '15-min'), or null for a value that is no positive whole number of minutes.
function spanText(minutes) {
  if (!Number.isInteger(minutes) || minutes <= 0) return null;
  if (minutes % 1440 === 0) return `${minutes / 1440}-day`;
  if (minutes % 60 === 0) return `${minutes / 60}-h`;
  return `${minutes}-min`;
}

// The trailing window the server counted each landing-rate point over
// (data/merge_queue.py::TRAILING_WINDOW_LADDER); a rate naming none states none.
function landingRateCaption(series) {
  const span = spanText((series || {}).trailing_minutes);
  return span === null ? QUEUE_EM_DASH : `landings/day · ${span} trailing`;
}

function rateText(perDay) {
  return String(Math.round(perDay * 10) / 10);
}

// The newest point and the mean over the served points: a reading of the served
// series, never a re-bucketing of it (the depth panel's peak was the precedent).
function landingRateReading(series) {
  const values = series.values;
  if (values.length === 0) return '0/day · no landings in this window';
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  return `now ${rateText(values[values.length - 1])}/day · mean ${rateText(mean)}/day`;
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const MERGE_QUEUE_API = {
  projectInQueue,
  inQueueOver,
  inQueueHistory,
  queuedSince,
  projectSpeculative,
  speculativeOver,
  hitRateText,
  recentTotal,
  latencyCaption,
  projectDepth,
  projectLandingRate,
  landingRateCaption,
  landingRateReading,
  projectAttempts,
  attemptCount,
  attemptsOver,
  attemptsHistory,
  worstP95Over,
  totalHint,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = MERGE_QUEUE_API;
}
if (typeof window !== 'undefined') {
  window.DF_MERGE_QUEUE = MERGE_QUEUE_API;
}
