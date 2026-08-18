// burndown_bands.js — the pure render DECISIONS behind the Burndown tab's
// stacked status-mix chart: which bands are drawn in which order and colour,
// the legend that explains them, and whether the concurrency-parity banner
// draws at all (tabs.jsx BurnTab, aggregate and per-project views alike).
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/burndown_bands.js">`
//     tag (like task_status_counts.js), which assigns `window.DF_BURNDOWN_BANDS`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// index.html loads this file (classic script, before the Babel JSX tags) so
// `window.DF_BURNDOWN_BANDS` is defined before tabs.jsx executes its top-level
// destructure of it.
//
// ── WHY THIS MODULE EXISTS ────────────────────────────────────────────────
// Commit 039e55c7ef deleted four JSX source-text meta-test blocks (task 3543)
// because they asserted regexes and substrings over raw .jsx source fetched
// over HTTP rather than over behaviour, and could not discriminate. The
// decisive demonstration is on the sibling pins_recovery surface: a
// whole-file substring grep is satisfied by a MENTION, so the explanatory
// COMMENT at tab_escalation_analytics.jsx:414-419 alone satisfied
// `'pins_recovery' in body` even with the render arm at :420-428 deleted —
// the test could not fail for the one reason it existed. The burndown block
// (test_tab_burndown.py:133-270) had the same defect in its own dialect: it
// pinned JSX object-literal SPELLING and captured a colour identifier as
// source text, so it tracked how the bands were written rather than what they
// drew.
//
// Deleting them was correct and left a real hole. This module closes the part
// of it that can be closed: the band/legend/banner decisions now live in pure
// functions with genuine behavioural coverage
// (dashboard/tests/js/burndown_bands.test.mjs), so merging the two
// in-progress bands, or colliding their colours, fails a test instead of
// nothing.
//
// ── WHY NOT A DOM HARNESS (considered and rejected) ───────────────────────
// jsdom or a headless browser would cover the JSX seam this extraction leaves
// open, and was the first thing considered. It does not fit this repo: there
// is no package.json, lockfile or tracked node_modules anywhere in git; React
// 18.3.1 and @babel/standalone 7.29.0 are unpkg CDN tags with SRI hashes,
// transpiled in-browser with no build step. The ABSENCE of a package.json is
// load-bearing — it is exactly what makes these static/redux/*.js files
// resolve as CommonJS for the existing node --test suite, so adding one to
// host jsdom would break that resolution model. The gate is
// `cd dashboard && uv run pytest tests/`, so a DOM harness would need an
// install at gate time or a skip-when-deps-missing guard — and a test that
// silently skips is the same "passes but does not discriminate" hole this
// module exists to close, merely relocated. The settled precedent for this
// trade is commit bea3edc34f, "GREEN — extract lockChipState helper, rewire
// chips, drop meta-test" (scheduler_utils.jsx + test_lock_chip_state.py).
//
// ── WHAT IS OUT OF SCOPE ──────────────────────────────────────────────────
// Re-hardening the deleted tests with tightened regexes or comment-stripped
// source fixtures is explicitly out of scope, by the reviewer guidance that
// motivated 039e55c7ef: hardening the greps deepens the same hole rather than
// closing it. The residual gap — deleting the thin surviving JSX call site
// still fails no test, because nothing in this repo renders JSX — is measured
// rather than assumed (see the mutation-verification step of task 4361) and
// is filed as a browser-harness follow-up. Do not "restore" the greps.
//
// Nothing here reads `window` or `document`: the shared-classic-script-scope
// suite loads this file into a bare `vm.createContext({window:{}})`, and a
// module that reached for browser globals would behave differently under test
// than in the browser. THE COLOUR PALETTE IS THEREFORE INJECTED as a
// parameter, never read off a global `CP` or `window.DF_CHARTS` — the same
// convention prd_grouping.js:18-21 uses for computeTiers. That also keeps the
// palette owned by exactly one file (charts.jsx) instead of being duplicated
// into this one.

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
