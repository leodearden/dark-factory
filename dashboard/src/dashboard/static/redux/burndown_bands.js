// burndown_bands.js — the pure render DECISIONS behind the Burndown tab's
// stacked status-mix chart: which bands are drawn in which order and colour,
// the legend that explains them, and whether the concurrency-parity banner
// draws at all (tabs.jsx BurnTab, aggregate and per-project views alike).
//
// Dual-loaded: a browser classic `<script>` assigns `window.DF_BURNDOWN_BANDS`,
// node resolves the same file as CommonJS for `dashboard/tests/js/`. index.html
// loads it before the Babel JSX tags, so the global exists before tabs.jsx runs
// its top-level destructure of it.
//
// ── THE SHARED SUBSTRATE DECISION IS NOT RESTATED HERE ────────────────────
// Why these helpers exist at all, why a DOM harness was considered and
// REJECTED, why re-hardening the deleted greps is out of scope, and why no
// module here reads a browser global: written out ONCE, in pins_recovery.js's
// header (the block marked CANONICAL). Read it there before re-litigating any
// of it. It is deliberately not copied into this file — three hand-copies of
// one rationale drift, which is the hazard these modules exist to remove.
//
// ── WHY THIS MODULE EXISTS (the burndown specifics) ───────────────────────
// The block that covered THIS surface, test_tab_burndown.py:133-270, carried
// the shared defect in its own dialect: it pinned JSX object-literal SPELLING
// and captured a colour identifier as source text, so it tracked how the bands
// were WRITTEN rather than what they DREW. Renaming a local would have failed
// it; merging the two in-progress bands would not have.
//
// That is now the other way round. The band, legend and banner decisions live
// in the pure functions below with behavioural coverage in
// dashboard/tests/js/burndown_bands.test.mjs, so merging the two in-progress
// bands, colliding their colours, or drawing the parity banner on a false
// alarm each fails a named test instead of nothing.
//
// ── THE PALETTE IS INJECTED ───────────────────────────────────────────────
// Per (4) of the canonical block: the colours arrive as a PARAMETER, never off
// a global `CP` or `window.DF_CHARTS`, the same way prd_grouping.js takes
// computeTiers. That is what lets the tests assert against a sentinel palette,
// and it keeps the real palette owned by exactly one file (charts.jsx) instead
// of duplicated into this one.

// ── The five stacked bands of the status-mix chart ──
// Takes a burndown block (the aggregate `b` or a project's `pb`) and the
// injected palette; returns `[{key, color, values}]` in fixed drawing order.
//
// in_progress IS BANDED AS live + stranded, NEVER ALONGSIDE THEM. The server
// guarantees the two parts sum to the whole, so emitting all three would
// stack the whole beside its parts and draw a total no census ever produced —
// every in-progress task counted twice. If a future block grows an
// `in_progress` key, it is still not drawn here; that is deliberate, and
// dashboard/tests/js/burndown_bands.test.mjs pins it.
//
// The two parts must also stay VISUALLY distinct (accent vs stranded).
// Splitting the band achieves nothing if both halves render in one colour —
// the split exists so a stranded task is visible as such.
//
// `values` is passed through by reference from the SAME-NAMED field of the
// block handed in. That is what makes the per-project call site safe: `labels`
// there is the project's own snapshot row, not the cross-project union, so a
// band wired to a field from a different block would both overrun and
// index-shift its series while still drawing a plausible-looking chart.
//
// A null/undefined block or palette is tolerated and yields the five bands
// with undefined `values` / `color` rather than throwing — the same tolerance
// the sibling extractions state for strandBadgeState and pinningSummary.
// BurnTab renders before the first burndown payload has necessarily arrived,
// and throwing here would blank the whole tab rather than draw an empty chart.
// Both `|| {}` arms below are pinned by burndown_bands.test.mjs, so neither is
// deletable in silence.
function burndownStacks(block, palette) {
  const b = block || {};
  const cp = palette || {};
  return [
    { key: 'done', color: cp.ok, values: b.done },
    { key: 'in_progress_live', color: cp.accent, values: b.in_progress_live },
    { key: 'in_progress_stranded', color: cp.stranded, values: b.in_progress_stranded },
    { key: 'blocked', color: cp.bad, values: b.blocked },
    { key: 'pending', color: cp.warn, values: b.pending },
  ];
}

// ── The legend for those bands ──
// Returns `[{label, color}]` positionally aligned with burndownStacks above.
//
// Derived FROM the stack definition rather than re-listed, so the legend
// cannot drift from the chart it explains. Before this extraction it was two
// verbatim-duplicated array literals — one per view — each free to drift from
// the bands AND from each other. A legend that disagrees with its chart is
// worse than no legend, because it is believed.
//
// The labels are the reader's words, not the wire's: the bands are keyed
// `in_progress_live` / `in_progress_stranded` because that is what the payload
// calls them, but a legend reading "in_progress_live" beside a chart titled
// "Status mix" is noise. Only the two in-progress halves are renamed; the
// other three already read as English.
const LEGEND_LABELS = {
  done: 'done',
  in_progress_live: 'live',
  in_progress_stranded: 'stranded',
  blocked: 'blocked',
  pending: 'pending',
};

function burndownLegend(palette) {
  // An empty block is enough: only the keys and colours are wanted here, and
  // deriving them from the same function the chart uses is the whole point.
  return burndownStacks({}, palette).map(s => ({
    label: LEGEND_LABELS[s.key],
    color: s.color,
  }));
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
// breach. A one-sided test would catch only the second.
//
// EVERY FIELD RETURNED IS RENDERED. The sole call site (tabs.jsx BurnTab's
// parityBanner closure) interpolates peak, cap and text and nothing else, so
// the breach count and the offending-project suffix are folded INTO `text`
// rather than ALSO exposed as fields nobody reads. An unread public field
// attracts coverage aimed at an output no operator ever sees, which reads as
// tested behaviour and is not; both are asserted through `text` instead.
//
// The count falls back to 0 rather than undefined, so the text never reads
// "undefined snapshots over". The project suffix is the aggregate view's
// breaching subset; the per-project view passes null, because naming a project
// inside its own panel says nothing.
//
// Returns the DECISION, not markup — the `<div className="badge bad">` that
// wraps it stays in tabs.jsx.
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

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced at
// runtime by dashboard/tests/js/classic_script_scope.test.mjs. A collision
// here would leave window.DF_BURNDOWN_BANDS undefined and break tabs.jsx's
// top-level destructure of it.
const BURNDOWN_BANDS_API = { burndownStacks, burndownLegend, parityBannerState };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = BURNDOWN_BANDS_API;
}
if (typeof window !== 'undefined') {
  window.DF_BURNDOWN_BANDS = BURNDOWN_BANDS_API;
}
