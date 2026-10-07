// escalation_views.js — the CLIENT reader of the escalation corpus' served
// views, its resolution-class split, each row's task card and the header's
// per-level counts. The views are
// counted once, server-side, over one walk of every queue's root and archive
// (dashboard/src/dashboard/data/escalation_corpus.py); both escalation tabs read
// them here, so the pill ("queue pending") and the strip ("open in history")
// are two named populations of one measurement rather than two counts at two
// freshnesses.
//
// A PLAIN-JS CLASSIC SCRIPT, NOT A .jsx MODULE, so its decisions are
// EXECUTABLE: dashboard/tests/js/escalation_views.test.mjs runs what no harness
// here can run inside a .jsx body. pins_recovery.js's header holds the
// CANONICAL statement of why.
//
// LOAD CONTRACT. It destructures window.DF_DATUM and window.DF_ENDPOINT_STALENESS
// at module scope with no fallback, so index.html loads it after both and
// before tab_escalations.jsx and tab_escalation_analytics.jsx
// (test_index_html.py pins the order). A browser classic `<script>` assigns
// `window.DF_ESCALATION_VIEWS`; node requires the same file as CommonJS once its
// test has put DF_ENDPOINT_STALENESS on a window shim.
//
// THE CLIENT NEVER RE-COUNTS A SERVED VIEW. Each view is the server's Datum;
// the only envelopes built here are datum.js's own: a receipt stamp, a
// plain-wrapped payload count, an unknown placeholder, and a combined total
// over the project filter in which a hole anywhere is a hole in the sum.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header. endpoint_staleness.js declares a top-level `function formatAge`.
const {
  servedDatum: servedEscalationDatum,
  unknownDatum: unknownEscalationDatum,
  combinedDatum: combineEscalationDatums,
  displayedAgeMs: displayedCorpusAgeMs,
  datumView: viewOfEscalationDatum,
  plainDatum: plainEscalationDatum,
} = window.DF_DATUM;
const { formatAge: formatCorpusAge } = window.DF_ENDPOINT_STALENESS;

// data.js registers both endpoints as PLAIN, so the Datums inside them arrive
// carrying no receipt of their own; each endpoint's receipt stamps its own.
const ESCALATIONS_VIEWS_ENDPOINT = '/api/v2/dashboard/escalations';
const ANALYTICS_VIEWS_ENDPOINT = '/api/v2/dashboard/escalation-analytics';

const VIEWS_NOT_YET_FETCHED = 'not yet fetched';

function escalationReceipts(data) {
  return (data && data.__receipt) || {};
}

// ── "Pending in the live queue": pending records at a queue's root ──

function queuePending(data) {
  const escalations = (data && data.ESCALATIONS) || {};
  return servedEscalationDatum(
    (escalations.views || {}).queue_pending,
    ESCALATIONS_VIEWS_ENDPOINT,
    'the /escalations payload has no queue_pending view',
    escalationReceipts(data),
  );
}

function subsectionQueuePending(sec, receipts) {
  const s = sec || {};
  return servedEscalationDatum(
    (s.views || {}).queue_pending,
    ESCALATIONS_VIEWS_ENDPOINT,
    'the /escalations subsection ' + (s.label || s.id) + ' has no queue_pending view',
    receipts,
  );
}

// ── The header's count of level-N records ──
// A plain count in the /escalations summary, wrapped with that endpoint's
// receipt. data.js seeds by_level with zeros, so before the first payload the
// count is not yet fetched rather than a seed zero passed off as measured.
function levelCount(data, level) {
  const summary = (((data || {}).ESCALATIONS || {}).summary) || {};
  return plainEscalationDatum(
    (summary.by_level || {})[level], ESCALATIONS_VIEWS_ENDPOINT, escalationReceipts(data),
  );
}

// ── "Open in history": pending records anywhere, root or archive ──
// Served per project by /escalation-analytics. A filter with nothing selected
// filters nothing — the escalation tabs' own convention for `projectFilter`.
// A filtered project the payload does not carry stays in scope, so a total
// over it is a hole that names it.
function openInHistoryOver(data, projects) {
  const receipts = escalationReceipts(data);
  if (!receipts[ANALYTICS_VIEWS_ENDPOINT]) return unknownEscalationDatum(VIEWS_NOT_YET_FETCHED);
  const entries = ((data.ESCALATION_ANALYTICS || {}).per_project) || [];
  const byProject = new Map(entries.map(entry => [entry.project, entry]));
  const scope = projects && projects.length > 0 ? projects : [...byProject.keys()];
  return combineEscalationDatums(
    scope.map(project => [project, projectOpenInHistory(byProject.get(project), project, receipts)]),
    counts => counts.reduce((sum, n) => sum + n, 0),
    'the /escalation-analytics payload carries no project',
  );
}

function projectOpenInHistory(entry, project, receipts) {
  const why = entry
    ? 'the /escalation-analytics entry for ' + project + ' has no open_in_history view'
    : 'the /escalation-analytics payload has no entry for ' + project;
  return servedEscalationDatum(
    entry && (entry.views || {}).open_in_history, ANALYTICS_VIEWS_ENDPOINT, why, receipts,
  );
}

// ── How old the corpus walk looks right now ──
// Stated even while the datum is fresh: a count of a 60s-cached walk should
// say when it was walked. Empty when there is no walk to date.
function corpusAgeCaption(datum, now) {
  const ageMs = datum ? displayedCorpusAgeMs(datum, now) : null;
  return ageMs === null ? '' : 'as of ' + formatCorpusAge(ageMs) + ' ago';
}

// ── One segment per served resolution class ──
// The class keys are the payload's, never a list held here, so a class the
// server adds renders without an edit to this file. An empty population has no
// shares to draw.
function resolutionSegments(classes) {
  const entries = Object.entries(classes || {});
  const total = entries.reduce((sum, [, n]) => sum + n, 0);
  if (total <= 0) return [];
  return entries.map(([cls, n]) => ({ cls, n, share: n / total }));
}

// ── The strip's benign rate, over EVERY class ──
// `rows` are workflow.flow_daily rows, already windowed, from any number of
// projects. The denominator is every classified filing in them, whatever its
// class, so the rate is a share of the same whole origin's split adds up to.
// No filing in the window has no rate; one with no benign filing reads 0.
function windowedClassSplit(rows) {
  const byClass = {};
  const byDate = {};
  for (const row of rows || []) {
    byClass[row.class] = (byClass[row.class] || 0) + row.n;
    const day = byDate[row.date] || (byDate[row.date] = { benign: 0, total: 0 });
    day.total += row.n;
    if (row.class === 'benign') day.benign += row.n;
  }
  const total = Object.values(byClass).reduce((sum, n) => sum + n, 0);
  return {
    total,
    byClass,
    benignShare: total > 0 ? (byClass.benign || 0) / total : null,
    benignShareDaily: Object.keys(byDate).sort().map(date => byDate[date].benign / byDate[date].total),
  };
}

// ── A row's task card ──
// The server's task Datum, stamped with its endpoint's receipt and drawn by
// datumView, so a hole and its reason come from the one hole decision.
// `reason` is the producer's, shown for anything but a fresh card; `age` is
// the badge, present once the card has outlived its bound.
function taskCard(row, receipts, now) {
  const r = row || {};
  const datum = servedEscalationDatum(
    r.task,
    ESCALATIONS_VIEWS_ENDPOINT,
    'the /escalations row ' + r.id + ' has no task Datum',
    receipts,
  );
  const view = viewOfEscalationDatum(datum, { now, format: task => String(task.title || '') });
  return {
    task: view.isHole ? null : datum.value,
    isHole: view.isHole,
    reason: view.title,
    age: view.age,
  };
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const ESCALATION_VIEWS_API = {
  queuePending,
  subsectionQueuePending,
  openInHistoryOver,
  corpusAgeCaption,
  resolutionSegments,
  windowedClassSplit,
  taskCard,
  levelCount,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = ESCALATION_VIEWS_API;
}
if (typeof window !== 'undefined') {
  window.DF_ESCALATION_VIEWS = ESCALATION_VIEWS_API;
}
