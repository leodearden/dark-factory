// memory_evals_fmt.js — pure display/vocabulary helpers for the memory-eval
// monitoring section rendered by tab_memory_evals.jsx.
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/memory_evals_fmt.js">`
//     tag (like spark_path.js/runtime_format.js), which assigns
//     `window.DF_MEMORY_EVALS_FMT`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// index.html loads this file (classic script, before the Babel JSX tags) so
// `window.DF_MEMORY_EVALS_FMT` is defined before tab_memory_evals.jsx executes
// its top-level destructure of it.
//
// ── WHY THIS FILE EXISTS (task 3481) ───────────────────────────────────────
//
// Everything here is JSX-FREE branching logic that happened to be declared
// inside a `type="text/babel"` .jsx. That placement was the whole problem:
// tab_memory_evals.jsx is transformed by CDN Babel at runtime and this repo has
// no node_modules, so node cannot parse it and React cannot be rendered in any
// harness here. The only reachable test was therefore a Python suite that read
// the .jsx AS TEXT and asserted regexes over the source — an idiom that
//
//   1. never EXECUTES the logic (a regex can see that `verdictBadge` contains a
//      `+ ' · ' +` composition; it cannot see which of the 5 verdicts × 13
//      parity states actually produce which label);
//   2. passes SILENTLY when it matches nothing (two regexes in that suite were
//      found matching nothing at all, which is why it is full of `assert body`
//      anti-vacuity guards); and
//   3. drifted from a hand-pinned COPY of the parity vocabulary rather than the
//      real table — a three-member copy once went blind to six states.
//
// Moving these helpers into a plain-JS sibling puts the branching somewhere
// `node --test` can execute it with real assertions
// (dashboard/tests/js/memory_evals_fmt.test.mjs). tab_memory_evals.jsx keeps
// the JSX and destructures this module's API at its top level.
//
// What deliberately stays in Python: cross-language vocabulary completeness
// against the producer's frozenset `memory_evals.PARITY_STATES`, the PRD
// section 8 (G6/INV-5) source guard — which scans BOTH this file and the .jsx,
// since the logic it guards now lives here — index.html load order, and
// JSX/React render wiring.

// Missing scalars render an em-dash, never `|| 0`: a synthetic zero reads as a
// measured zero.  Same placeholder the Memory tab already uses (tabs.jsx:589).
function dash(v) {
  if (v === null || v === undefined) return '—';
  return v;
}

// Compact age from `latest_run_age_seconds`.  Display only — the staleness
// THRESHOLD lives server-side and is deliberately absent from the payload, so
// nothing here can re-derive `stale`.
function ageText(seconds) {
  if (seconds === null || seconds === undefined) return '—';
  const h = seconds / 3600;
  if (h < 1) return `${Math.round(seconds / 60)}m ago`;
  if (h < 48) return `${Math.round(h)}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

// ── Unmatched escalations ──
//
// Branched on `reason`, with distinct wording per value.  Collapsing the three
// into one undifferentiated "unexplained" list would fire on escalations that
// are in fact fully explained and train operators to ignore the one signal
// that catches a real parity orphan (memory_evals._unmatched_projection()).
function unmatchedReasonText(reason) {
  if (reason === 'no_matching_verdict') return 'no metric row explains this';
  if (reason === 'storm_suppressed') return "explained, but this run's links are collapsed into the aggregate";
  if (reason === 'no_fingerprint') return 'producer emitted no dedupe_fingerprint';
  return `unrecognised reason: ${String(reason)}`;
}

// ── Chart primitive per metric kind (PRD open question 1) ──
//
// RETURNS A TAG, NOT A COMPONENT.  The name is unchanged from the .jsx version
// but the return type is not, so a reader who assumes "component" is silently
// wrong.  This function used to return the `MEStep` / `MESpark` React component
// references destructured from `window.DF_CHARTS` — a browser global node
// cannot resolve, which is precisely what pinned this vocabulary inside a .jsx
// and out of reach of any runner (task 3481, DD-1).  It now returns a
// closed-vocabulary tag string and the tag→component mapping is the CALLER's
// job: tab_memory_evals.jsx holds a two-entry `ME_CHART_BY_TAG` lookup at its
// single call site.  That split is what lets node execute the vocabulary at all.
//
// The payload's kind vocabulary is exactly {tripwire, proportion, count,
// scalar}.  A kind outside that set is a RENDERING gap, not a data error: the
// builder passes the value through verbatim and files an `unknown_kind` issue
// for it.  So the fallback is `null` — value only, NO chart.  Guessing a
// primitive would render an unvalidated shape as though it were understood.
// The `null` tag also preserves the caller's `plottable` truthiness gate: an
// unknown kind resolves to a null Chart there exactly as it did before.
function chartForKind(kind) {
  if (kind === 'tripwire') return 'step';   // step-shaped item counts
  if (kind === 'proportion') return 'spark';
  if (kind === 'count') return 'spark';
  if (kind === 'scalar') return 'spark';
  return null;
}

const MEMORY_EVALS_FMT_API = { dash, ageText, unmatchedReasonText, chartForKind };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = MEMORY_EVALS_FMT_API;
}
if (typeof window !== 'undefined') {
  window.DF_MEMORY_EVALS_FMT = MEMORY_EVALS_FMT_API;
}
