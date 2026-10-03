// burndown_bands.js — the pure render DECISIONS behind the Burndown tab
// (tabs.jsx BurnTab, aggregate and per-project views alike): the nine bands of
// the stacked status-mix chart and the legend that explains them, whether the
// concurrency-parity banner draws, and the reader that stamps the burndown
// payload's served Datums for the tiles, pips and cells.
//
// LOAD CONTRACT. It destructures window.DF_DATUM and window.DF_TASK_VOCAB at
// module scope with no fallback, so index.html loads it after datum.js and
// task_vocab.js and before the Babel JSX tags, so the global exists before
// tabs.jsx runs its top-level destructure of it (test_index_html.py pins both
// orders). A browser classic `<script>` assigns `window.DF_BURNDOWN_BANDS`;
// node requires the same file as CommonJS once its test has put those two
// globals on a window shim.
//
// ── THE SHARED SUBSTRATE DECISION IS NOT RESTATED HERE ────────────────────
// Why these helpers exist at all, why a DOM harness was considered and
// REJECTED, and why no module here reads a browser global for its data:
// written out ONCE, in pins_recovery.js's header (the block marked CANONICAL).
// Coverage is behavioural, in dashboard/tests/js/burndown_bands.test.mjs.
//
// ── THE PALETTE IS INJECTED ───────────────────────────────────────────────
// The colours arrive as a PARAMETER, never off a global `CP` or
// `window.DF_CHARTS`, the same way prd_grouping.js takes computeTiers. That is
// what lets the tests assert against a sentinel palette, and it keeps the real
// palette owned by exactly one file (charts.jsx). Which palette SLOT each
// member draws in is census.py's TONES, reached through the generated
// vocabulary — never a burndown-local colour map.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const {
  isDatum: isBurndownDatum,
  withReceipt: stampBurndownReceipt,
  unknownDatum: unknownBurndownDatum,
} = window.DF_DATUM;
const {
  MEMBERS: BAND_MEMBERS,
  VIEWS: BAND_VIEWS,
  TONES: BAND_TONES,
  SERIES_KEYS: BAND_SERIES_KEYS,
} = window.DF_TASK_VOCAB;

// ── The nine stacked bands of the status-mix chart ──
// One band per census member, keyed and sourced by the member's series key.
// The in-progress live/stranded split is NOT stacked: it partitions
// in_progress_rows, the rows' count, which is a different instant from the
// census members, so stacking it among them would draw a total no census ever
// produced. It stays on the wire for the parity alarm.
//
// Stacked bottom-up terminal → backlog → in_flight, members in TaskStatus
// declaration order inside each view. The three members task 5591 added
// (review, merge-deferred, infra-hold) come last in that order, so they sit on
// top: StackedAreaChart draws a layer only where every layer below it is
// measured, and a pre-migration row's hole in them then blanks only their own
// bands instead of the whole history above.
const BAND_VIEW_ORDER = ['terminal', 'backlog', 'in_flight'];
const BAND_STACK_MEMBERS = BAND_VIEW_ORDER.flatMap(view =>
  BAND_MEMBERS.filter(member => BAND_VIEWS[view].includes(member)),
);

// Takes a burndown block (the aggregate `b` or a project's `pb`) and the
// injected palette; returns `[{key, member, color, values}]` in drawing order.
//
// `values` is passed through by reference from the block handed in. That is
// what makes the per-project call site safe: `labels` there is the project's
// own snapshot row, not the cross-project union, so a band wired to a field
// from a different block would both overrun and index-shift its series while
// still drawing a plausible-looking chart.
//
// A null/undefined block or palette is tolerated and yields the nine bands
// with undefined `values` / `color` rather than throwing: BurnTab renders
// before the first burndown payload has necessarily arrived, and throwing here
// would blank the whole tab rather than draw an empty chart.
function burndownStacks(block, palette) {
  const b = block || {};
  const cp = palette || {};
  return BAND_STACK_MEMBERS.map(member => ({
    key: BAND_SERIES_KEYS[member],
    member,
    color: cp[BAND_TONES[member]],
    values: b[BAND_SERIES_KEYS[member]],
  }));
}

// ── The legend for those bands ──
// Returns `[{label, color}]` positionally aligned with burndownStacks, labelled
// by census member. Derived FROM the stack definition rather than re-listed,
// so the legend cannot drift from the chart it explains — a legend that
// disagrees with its chart is worse than no legend, because it is believed.
function burndownLegend(palette) {
  return burndownStacks({}, palette).map(s => ({ label: s.member, color: s.color }));
}

// ── Should the concurrency-parity banner draw, and what does it say? ──
// Returns `{peak, cap, text}` or null for "draw nothing".
//
// THE VERDICT IS COMPUTED SERVER-SIDE and this only renders it. Each snapshot
// is judged against the cap stored ON that snapshot: max_concurrent_tasks is
// restart-only, but a burndown window spans restarts and the cap also varies
// between projects, so it is TIME-VARYING across the window regardless.
// Re-deriving one cap here from the rendered series would forgive a real past
// breach after a raise, and invent one after a cut.
//
// Both null arms matter and are pinned separately. Returning the object
// unconditionally would accuse the operator's fleet of breaching a cap it
// never breached; returning null unconditionally would silently drop a real
// breach.
//
// EVERY FIELD RETURNED IS RENDERED: the sole call site (BurnTab's
// parityBanner closure) interpolates peak, cap and text and nothing else, so
// the breach count and the offending-project suffix are folded INTO `text`.
// The count falls back to 0 rather than undefined, so the text never reads
// "undefined snapshots over". The project suffix is the aggregate view's
// breaching subset; the per-project view passes null, because naming a project
// inside its own panel says nothing.
function parityBannerState(block, projects) {
  if (!block || !block.parity_alarm) return null;
  const n = block.parity_breach_count ?? 0;
  const who = projects && projects.length ? ` · ${projects.join(', ')}` : '';
  return {
    peak: block.parity_peak,
    cap: block.parity_cap,
    text: ` · ${n} snapshot${n !== 1 ? 's' : ''} over${who}`,
  };
}

// ── A served burndown Datum, stamped with the burndown receipt ──
// Every burndown block carries two served Datums, `latest` and `forecast`.
// data.js registers BURNDOWN / BURNDOWN_BY_PROJECT as PLAIN, so the Datums
// nested in them arrive carrying no receipt of their own; this stamps a COPY
// with the /burndown receipt so datum.js can age it (task_snapshot.js's
// snapshotDatum is the same reader for the /tasks unit). No receipt outranks
// everything else: before the first payload, nothing absent is yet evidence of
// anything.
const BURNDOWN_ENDPOINT = '/api/v2/dashboard/burndown';

function burndownDatum(data, block, field) {
  const receipt = (data.__receipt || {})[BURNDOWN_ENDPOINT];
  if (!receipt) return unknownBurndownDatum('not yet fetched');
  const served = (block || {})[field];
  if (!isBurndownDatum(served)) {
    return unknownBurndownDatum('the burndown payload has no ' + field + ' Datum');
  }
  return stampBurndownReceipt(served, receipt);
}

// ── How the Forecast tile reads a served forecast value ──
// One number when the recent and lifetime forecasts agree, a range otherwise.
function forecastText(forecast) {
  const { forecast_low: low, forecast_high: high } = forecast;
  return low === high ? `${low}d` : `${low}–${high}d`;
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced at
// runtime by dashboard/tests/js/classic_script_scope.test.mjs. A collision
// here would leave window.DF_BURNDOWN_BANDS undefined and break tabs.jsx's
// top-level destructure of it.
const BURNDOWN_BANDS_API = {
  burndownStacks,
  burndownLegend,
  parityBannerState,
  burndownDatum,
  forecastText,
  BURNDOWN_ENDPOINT,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = BURNDOWN_BANDS_API;
}
if (typeof window !== 'undefined') {
  window.DF_BURNDOWN_BANDS = BURNDOWN_BANDS_API;
}
