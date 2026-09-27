// task_snapshot.js — the CLIENT twin of dashboard/src/dashboard/data/task_snapshot.py,
// and the ONE reader of DF_DATA.TASKS_SNAPSHOT for every census surface: the
// OrchTab pips, tiles, filter bar and Progress card, the Overview tile and
// pipeline, the topbar pill, the rail badge, and the Tasks tab's header pips
// and rows. Named for its server twin so
// the two halves of the snapshot unit can find each other (the datum.js
// precedent).
//
// A PLAIN-JS CLASSIC SCRIPT, NOT A .jsx MODULE, so its decisions are
// EXECUTABLE: dashboard/tests/js/task_snapshot.test.mjs runs what no harness
// here can run inside a .jsx body. pins_recovery.js's header holds the
// CANONICAL statement of why.
//
// LOAD CONTRACT. It destructures window.DF_DATUM and window.DF_TASK_VOCAB at
// module scope with no fallback, so index.html loads it after datum.js and
// task_vocab.js and before every JSX consumer (test_index_html.py pins the
// order). A browser classic `<script>` assigns `window.DF_TASK_SNAPSHOT`; node
// requires the same file as CommonJS once its test has put those two globals
// on a window shim.
//
// THE CLIENT NEVER ALTERS A SERVED CENSUS. A surface's number is a `format`
// over the WHOLE census Datum, so its value, as_of, state and reason reach the
// screen as served. The only envelopes built here are datum.js's own: a
// receipt stamp, an unknown placeholder, and a combined total.
//
// TWO DELIBERATE REFUSALS, both from the PRD's rule that consumers "never
// re-count, re-bucket or re-fetch" (plans/dashboard-one-datum-one-path-prd.md,
// "Sketch of approach" items 1 and 5): a hole in any project is a hole in a
// multi-project total (censusOver), and a tile scoped to two or more projects
// draws no spark (censusHistory) rather than summing per-project series here.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const {
  withReceipt: stampSnapshotReceipt,
  unknownDatum: unknownSnapshotDatum,
  combinedDatum: combineSnapshotDatums,
  isDatum: isSnapshotDatum,
  assertDatum: assertSnapshotDatum,
  EM_DASH: SNAPSHOT_PLACEHOLDER,
} = window.DF_DATUM;
const { VIEWS: TASK_VIEW_MEMBERS } = window.DF_TASK_VOCAB;

// The receipt every nested snapshot Datum is stamped with. data.js registers
// TASKS_SNAPSHOT as PLAIN (its header says why), so the census and rows Datums
// inside it arrive carrying no receipt of their own.
const TASKS_ENDPOINT = '/api/v2/dashboard/tasks';

const SNAPSHOT_NOT_YET_FETCHED = 'not yet fetched';

function snapshotReceipt(data) {
  return (data.__receipt || {})[TASKS_ENDPOINT] || null;
}

function snapshotEntries(data) {
  return data.TASKS_SNAPSHOT || {};
}

// The Datum under one wire key of one project's entry — `census` or `rows`,
// as TaskSnapshot.to_wire() names them — stamped with the /tasks receipt, or a
// hole saying which part is missing. No receipt outranks everything else, as
// in datum.js::plainDatum: before the first payload, nothing that is absent is
// yet evidence of anything.
function snapshotDatum(data, project, wireKey) {
  const receipt = snapshotReceipt(data);
  if (!receipt) return unknownSnapshotDatum(SNAPSHOT_NOT_YET_FETCHED);
  const entry = snapshotEntries(data)[project];
  if (!entry) return unknownSnapshotDatum('the /tasks payload has no entry for ' + project);
  if (!isSnapshotDatum(entry[wireKey])) {
    return unknownSnapshotDatum('the /tasks entry for ' + project + ' has no ' + wireKey + ' Datum');
  }
  return stampSnapshotReceipt(entry[wireKey], receipt);
}

function projectCensus(data, project) {
  return snapshotDatum(data, project, 'census');
}

function projectRows(data, project) {
  return snapshotDatum(data, project, 'rows');
}

// Every row the snapshot holds, across projects, in entry order: what the Tasks
// tab probes and looks a selected task up in. An aged rows Datum still
// contributes; a hole contributes nothing.
function snapshotRowsOver(data) {
  return Object.values(snapshotEntries(data)).flatMap(entry =>
    entry && isSnapshotDatum(entry.rows) ? entry.rows.value || [] : [],
  );
}

// ── A census over several projects ──
// `projects` null means every project the snapshot carries: task ROOTS, not
// ORCHESTRATORS entries, so a tile's headline and its burndown history count
// one population. A hole in any project is a hole in the total
// (datum.js::combinedDatum).
function censusOver(data, projects) {
  if (!snapshotReceipt(data)) return unknownSnapshotDatum(SNAPSHOT_NOT_YET_FETCHED);
  const scope = projects === null ? Object.keys(snapshotEntries(data)) : projects;
  return combineSnapshotDatums(
    scope.map(project => [project, projectCensus(data, project)]),
    sumCensusValues,
    'no project census in scope',
  );
}

// Member-wise, over every keyed part of data/census.py::TaskCensus.to_wire().
// A sum of partitions is a partition, so the total keeps the census's own
// invariants — the views partition `total`, `running` sits inside in_flight.
function sumCensusValues(values) {
  return {
    counts: sumCensusMembers(values.map(v => v.counts)),
    total: values.reduce((sum, v) => sum + v.total, 0),
    views: sumCensusMembers(values.map(v => v.views)),
    sub_views: sumCensusMembers(values.map(v => v.sub_views)),
  };
}

function sumCensusMembers(maps) {
  const summed = {};
  for (const map of maps) {
    for (const [key, n] of Object.entries(map)) summed[key] = (summed[key] || 0) + n;
  }
  return summed;
}

// ── The named readings ──
// Each is a datumView `format` over a census VALUE. datumView never invokes a
// format on a hole, so no reading ever sees a missing value and none carries a
// guard. Numbers are plain String(n), no locale: one number, one spelling on
// every surface.
function runningOfInFlight(census) {
  return `${census.sub_views.running} running of ${census.views.in_flight} in-flight`;
}

function terminalOfTotal(census) {
  return `${census.views.terminal}/${census.total}`;
}

function censusTotal(census) {
  return `${census.total} total`;
}

function censusViewCount(key) {
  return census => String(census.views[key]);
}

function censusViewReading(key, label) {
  return census => `${census.views[key]} ${label}`;
}

function censusMemberCount(member) {
  return census => String(census.counts[member]);
}

const inFlightCount = censusViewCount('in_flight');

// The generated views (DF_TASK_VOCAB.VIEWS, from data/census.py), each with
// its label, its palette tone, the bare count a filter button shows, and the
// reading a pip or legend entry shows. in_flight's reading shows its running
// sub-view WITH its superset, never alone (PRD decision 3).
function censusView(key, label, tone, reading = censusViewReading(key, label)) {
  return Object.freeze({ key, label, tone, count: censusViewCount(key), reading });
}

const CENSUS_VIEWS = Object.freeze([
  censusView('in_flight', 'in-flight', 'accent', runningOfInFlight),
  censusView('backlog', 'backlog', 'warn'),
  censusView('terminal', 'terminal', 'ok'),
]);

const EVERY_VIEW = Object.freeze(Object.fromEntries(CENSUS_VIEWS.map(view => [view.key, true])));

// The OrchTab census tiles show MEMBERS, not views. Burndown persists members,
// so each tile's `series` is the history of the very member its headline
// shows; a view tile would sit beside the spark of a different quantity.
const CENSUS_TILES = Object.freeze([
  Object.freeze({
    key: 'running',
    label: 'Running / in-flight',
    tone: 'accent',
    series: 'in_progress',
    reading: census => `${census.sub_views.running} / ${census.views.in_flight}`,
  }),
  Object.freeze({ key: 'blocked', label: 'Blocked', tone: 'bad', series: 'blocked', reading: censusMemberCount('blocked') }),
  Object.freeze({ key: 'pending', label: 'Pending', tone: 'warn', series: 'pending', reading: censusMemberCount('pending') }),
]);

// ── The Progress and pipeline bar ──
// One segment per view, as a share of the total. Keyed on the presence of a
// value, datumView's own hole rule: an aged census still draws its bar, a hole
// draws none, and a measured empty census draws zero-width segments rather
// than dividing by zero. A pipeline row's share is a reading of the same
// number, so the text beside a segment always matches its width.
function censusSegments(census) {
  const { value } = assertSnapshotDatum(census, 'censusSegments');
  if (value === null) return [];
  return CENSUS_VIEWS.map(({ key, tone }) => ({ key, tone, share: viewShare(value, key) }));
}

function viewShare(census, key) {
  return census.total > 0 ? (census.views[key] / census.total) * 100 : 0;
}

function viewShareText(key) {
  return census => `${viewShare(census, key).toFixed(0)}%`;
}

// ── A tile's spark, over the tile's own scope ──
// The server aggregate with no project filter, that project's series with
// one, and NO spark over two or more: summing ragged per-project series here
// would be a second copy of redux_api.py::shape_burndown's aggregation.
function censusHistory(data, projects, tile) {
  if (projects === null) return data.BURNDOWN[tile.series];
  if (projects.length !== 1) return null;
  const projectSeries = (data.BURNDOWN_BY_PROJECT || {})[projects[0]] || {};
  return projectSeries[tile.series] || [];
}

// ── The rows a view filter lists ──
// `filter` is OrchTab's {in_flight, backlog, terminal} booleans. In-flight and
// backlog rows come from the snapshot's rows Datum, terminal rows from the
// on-demand window, and every view selects by the generated membership, so a
// member the census counts under a view is listed under that same view. Rows
// concatenate in view order, whatever order the filter's keys were set in.
//
// A selected view whose Datum is a hole lists nothing and is named in ONE
// placeholder, titled with its reason: an offline project reads why, not a
// confident "No in-flight tasks". Nothing selected answers no placeholder —
// that sentence is orch_filter.js's.
//
// A listed view whose Datum is not fresh lists its rows AND gets a `notes`
// line saying why: the terminal window is always lower_bound (the newest N
// only), and an aged snapshot is still a value, so neither passes for the
// whole, current population.
function viewRows(rowsDatum, terminalDatum, filter) {
  const sourced = CENSUS_VIEWS
    .filter(view => filter[view.key])
    .map(view => [view, view.key === 'terminal' ? terminalDatum : rowsDatum]);
  const holes = sourced.filter(([, datum]) => assertSnapshotDatum(datum, 'viewRows').value === null);
  const partial = sourced.filter(([, datum]) => datum.value !== null && datum.state !== 'fresh');
  return {
    rows: sourced.flatMap(([view, datum]) => rowsInView(view.key, datum.value || [])),
    placeholder: holes.length === 0 ? null : viewRowsPlaceholder(holes),
    notes: partial.map(viewReason),
  };
}

function rowsInView(key, rows) {
  return rows.filter(row => TASK_VIEW_MEMBERS[key].includes(row.status));
}

function viewRowsPlaceholder(holes) {
  return {
    text: SNAPSHOT_PLACEHOLDER + ' ' + holes.map(([view]) => view.label).join(', ') + ' rows',
    title: holes.map(viewReason).join('; '),
  };
}

function viewReason([view, datum]) {
  return view.label + ': ' + datum.reason;
}

// ── The terminal window ──
// Terminal rows come only from the on-demand ?terminal=<project> window.
// OrchTab lists that window when another surface's request has landed one, but
// never requests it itself. data.js::datumFor would call an absent window
// 'not yet fetched', which promises a fetch that is not coming, so the hole
// states what is actually true instead.
const TERMINAL_NOT_REQUESTED = 'the terminal window is fetched on request only, and this view does not request it';

function unrequestedTerminalRows(served) {
  return isSnapshotDatum(served) ? served : unknownSnapshotDatum(TERMINAL_NOT_REQUESTED);
}

// PRD decision 5: the projects whose window a surface requests — every one
// while the terminal view is selected, else each one grouped by PRD.
function terminalWindowProjects(projectIds, filter, groupedIds) {
  return projectIds.filter(id => filter.terminal || groupedIds.includes(id));
}

// The projects in `wanted` that were not in `wantedBefore`: the only ones a
// change to the wanted set owes a request, since the rest already had theirs.
function terminalWindowEntrants(wantedBefore, wanted) {
  return wanted.filter(id => !wantedBefore.includes(id));
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced by classic_script_scope.test.mjs.
const TASK_SNAPSHOT_API = {
  TASKS_ENDPOINT,
  projectCensus,
  censusOver,
  CENSUS_VIEWS,
  EVERY_VIEW,
  CENSUS_TILES,
  inFlightCount,
  runningOfInFlight,
  terminalOfTotal,
  censusTotal,
  viewShareText,
  censusSegments,
  censusHistory,
  projectRows,
  snapshotRowsOver,
  viewRows,
  unrequestedTerminalRows,
  terminalWindowProjects,
  terminalWindowEntrants,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TASK_SNAPSHOT_API;
}
if (typeof window !== 'undefined') {
  window.DF_TASK_SNAPSHOT = TASK_SNAPSHOT_API;
}
