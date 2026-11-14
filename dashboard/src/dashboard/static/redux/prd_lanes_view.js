// prd_lanes_view.js — the CLIENT reader of /api/v2/dashboard/prds, and every
// pure decision the Tasks tab's PRD lanes page (prd_lanes.jsx) makes. The
// shapes it reads are data/prd_view.py::PrdView.to_wire (DF_DATA.PRD_VIEW) and
// data/prd_contention.py::Contention.to_wire (DF_DATA.PRD_CONTENTION), each a
// map of project to one nested Datum.
//
// A PLAIN-JS CLASSIC SCRIPT, NOT A .jsx MODULE, so its decisions are
// EXECUTABLE under node: dashboard/tests/js/prd_lanes_view.test.mjs.
//
// LOAD CONTRACT. It destructures window.DF_DATUM at module scope with no
// fallback, so index.html loads it after datum.js and before every JSX
// consumer (test_index_html.py pins the order).
//
// THE CLIENT NEVER RE-COUNTS OR CLASSIFIES A SERVED FIGURE. Segment order,
// labels and tones come from the served segment_legend; lane order, labels and
// rules from the served lanes. The one vocabulary held here is display text
// for the signal marks, which the server serves no label for; an unrecognised
// mark shows as served. The one threshold held here is READY_OVERDUE_DAYS, the
// ready cell's alarm, which its title states in words.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const { servedDatum: servedPrdsDatum } = window.DF_DATUM;

// data.js registers /prds as PLAIN, so each nested Datum arrives with no
// receipt of its own; the readers stamp it, as task_snapshot.js does /tasks.
const PRDS_ENDPOINT = '/api/v2/dashboard/prds';

const READY_OVERDUE_DAYS = 30;
const DARK_MATTER_PAGE_SIZE = 100;

const SIGNAL_MARK_TEXT = Object.freeze({
  delivered: 'delivered',
  partial: 'partial',
  not_yet: 'not yet',
  unknown: 'unknown',
});

// The dimension each served dark-matter count group is labelled with, in the
// order the summary reads them.
const DARK_MATTER_GROUPS = Object.freeze([
  ['by_reason', 'why'],
  ['by_status', 'status'],
  ['by_priority', 'priority'],
  ['by_source', 'source'],
]);

const INPUT_STATE_TONES = Object.freeze({
  fresh: 'ok',
  stale: 'warn',
  lower_bound: 'warn',
  unknown: 'bad',
});

// ── The two readers ──

function prdsDatum(data, wireKey, project) {
  const entries = data[wireKey] || {};
  const why = 'the /prds payload has no ' + wireKey + ' entry for ' + project;
  return servedPrdsDatum(entries[project], PRDS_ENDPOINT, why, data.__receipt || {});
}

function projectPrdView(data, project) {
  return prdsDatum(data, 'PRD_VIEW', project);
}

function projectPrdContention(data, project) {
  return prdsDatum(data, 'PRD_CONTENTION', project);
}

// ── A figure's title: its row detail, if any, then the served rule ──

function figureTitle(rule, detail) {
  return detail ? `${detail}\n\n${rule}` : rule;
}

// ── A row's cells ──

function readyCell(ready) {
  const days = ready.oldest_pending_days;
  if (days === null || days === undefined) return { text: `ready ${ready.count}`, overdue: false, detail: null };
  const bound = ready.oldest_is_lower_bound ? '≥' : '';
  const overdue = days >= READY_OVERDUE_DAYS;
  return {
    text: `ready ${ready.count} · oldest ${bound}${days}d`,
    overdue,
    detail: overdue ? `overdue: the oldest ready member has been pending ${READY_OVERDUE_DAYS}d or more` : null,
  };
}

function barSegments(row, legend) {
  return legend.map(({ segment, label, tone }) => {
    if (!Object.prototype.hasOwnProperty.call(row.segments, segment)) {
      throw new TypeError(`PRD row ${row.key} serves no '${segment}' segment, which the legend names`);
    }
    const count = row.segments[segment];
    return { segment, label, tone, count, title: `${label}: ${count}` };
  });
}

function doneOfTotal(row) {
  return {
    text: `${row.segments.done}/${row.total}`,
    detail: `${row.direct_open_ids.length} direct and ${row.followup_open.length} follow-up open`,
  };
}

function storeStatusText(status) {
  return status === null ? 'not in the store' : status;
}

// A leaf bound to no local task has no store status to show; the served
// reason line above it says why.
function signalLeafText(leaf) {
  if (leaf.task_id === null) return leaf.label;
  const kind = leaf.done_kind === null ? '' : ` (${leaf.done_kind})`;
  return `${leaf.label} #${leaf.task_id} ${storeStatusText(leaf.status)}${kind}`;
}

function signalCell(signal) {
  const text = Object.prototype.hasOwnProperty.call(SIGNAL_MARK_TEXT, signal.mark)
    ? SIGNAL_MARK_TEXT[signal.mark]
    : signal.mark;
  const lines = [signal.reason, ...signal.leaves.map(signalLeafText)].filter(Boolean);
  return { text, detail: lines.length === 0 ? null : lines.join('\n') };
}

// ── The contention line ──

function contentionEntryText(entry) {
  const holder = entry.holder;
  const held = holder === null
    ? 'no holder'
    : `held by #${holder.task_id} (${storeStatusText(holder.status)})`;
  return `${entry.key} · ${entry.wanted_by_ready} ready want it · ${held}`;
}

function heldShareText(heldShare) {
  const bands = Object.entries(heldShare).map(
    ([band, share]) => `${band} ${share.naming_held_key} of ${share.ready}`,
  );
  return 'ready naming a held key: ' + (bands.length === 0 ? 'no ready tasks' : bands.join(' · '));
}

function neverStartedText(neverStarted) {
  const ids = neverStarted.ids.length === 0 ? '' : ` (${neverStarted.ids.map(id => `#${id}`).join(', ')})`;
  return `${neverStarted.count} of ${neverStarted.of} ready high never started${ids}`;
}

// ── Dark matter ──

function countsText(counts) {
  return Object.entries(counts)
    .filter(([, n]) => n !== 0)
    .map(([key, n]) => `${key} ${n}`)
    .join(', ');
}

function darkMatterSummary(darkMatter) {
  const groups = DARK_MATTER_GROUPS
    .map(([by, label]) => [label, countsText(darkMatter[by])])
    .filter(([, counts]) => counts !== '')
    .map(([label, counts]) => `${label}: ${counts}`);
  const readyHigh = darkMatter.ready_high_ids.map(id => `#${id}`).join(' ');
  return [
    `${darkMatter.total} open tasks no PRD owns`,
    ...groups,
    ...(readyHigh === '' ? [] : [`ready high: ${readyHigh}`]),
  ].join(' · ');
}

function darkMatterPage(ids, pages) {
  const shown = ids.slice(0, pages * DARK_MATTER_PAGE_SIZE);
  return { ids: shown, more: ids.length - shown.length };
}

// The rows of `pool` — the Tasks tab's narrowed rows — that are the tasks
// `ids` name, in pool order, matched on the row id active_tasks.py::task_uid
// builds; and the ids no pool row holds, which the page lists as not drawn.
function membersDrawn(pool, project, ids) {
  const wanted = new Map(ids.map(id => [`${project}/T-${id}`, id]));
  const rows = pool.filter(row => wanted.has(row.id));
  const drawn = new Set(rows.map(row => row.id));
  return { rows, notDrawn: ids.filter(id => !drawn.has(`${project}/T-${id}`)) };
}

// ── The footer's input pips ──

function inputPip(input) {
  const fresh = input.state === 'fresh';
  return {
    text: fresh ? input.name : `${input.name} ${input.state}`,
    tone: INPUT_STATE_TONES[input.state] || 'fg3',
    title: input.reason || input.as_of,
  };
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced by classic_script_scope.test.mjs.
const PRD_LANES_VIEW_API = {
  PRDS_ENDPOINT,
  READY_OVERDUE_DAYS,
  DARK_MATTER_PAGE_SIZE,
  projectPrdView,
  projectPrdContention,
  readyCell,
  barSegments,
  doneOfTotal,
  signalCell,
  contentionEntryText,
  heldShareText,
  neverStartedText,
  darkMatterSummary,
  darkMatterPage,
  membersDrawn,
  inputPip,
  figureTitle,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PRD_LANES_VIEW_API;
}
if (typeof window !== 'undefined') {
  window.DF_PRD_LANES_VIEW = PRD_LANES_VIEW_API;
}
