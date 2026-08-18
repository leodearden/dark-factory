// task_strand_badge.js — the pure render DECISION behind the stranded badge
// and the agent cell it sits beside, for the Tasks tab (tab_tasks.jsx) and the
// Orchestrators tab (tabs.jsx).
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/task_strand_badge.js">`
//     tag (like task_status_counts.js), which assigns `window.DF_TASK_STRAND_BADGE`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// index.html loads this file (classic script, before the Babel JSX tags) so
// `window.DF_TASK_STRAND_BADGE` is defined before tab_tasks.jsx and tabs.jsx
// execute their top-level destructures of it.
//
// ── WHY THIS MODULE EXISTS ────────────────────────────────────────────────
// Commit 039e55c7ef deleted four JSX source-text meta-test blocks (task 3543)
// because they asserted regexes and substrings over raw .jsx source fetched
// over HTTP rather than over behaviour, and could not discriminate. The
// decisive demonstration is on the sibling pins_recovery surface: a
// whole-file substring grep is satisfied by a MENTION, so the explanatory
// COMMENT at tab_escalation_analytics.jsx:414-419 alone satisfied
// `'pins_recovery' in body` even with the render arm at :420-428 deleted. The
// test could not fail for the one reason it existed. The stranded-badge file
// (test_tab_tasks_stranded_badge.py, 187 lines) had the same defect plus
// negated regexes that passed vacuously when the feature was absent
// altogether.
//
// Deleting them was correct and left a real hole. This module closes the part
// of it that can be closed: the render DECISION now lives in a pure function
// with genuine behavioural coverage
// (dashboard/tests/js/task_strand_badge.test.mjs), so deleting a decision arm
// fails a test instead of nothing.
//
// ── WHY NOT A DOM HARNESS (considered and rejected) ───────────────────────
// jsdom or a headless browser would cover the JSX seam this extraction leaves
// open, and was the first thing considered. It does not fit this repo:
// there is no package.json, lockfile or tracked node_modules anywhere in git;
// React 18.3.1 and @babel/standalone 7.29.0 are unpkg CDN tags with SRI
// hashes, transpiled in-browser with no build step. The ABSENCE of a
// package.json is load-bearing — it is exactly what makes these
// static/redux/*.js files resolve as CommonJS for the existing node --test
// suite, so adding one to host jsdom would break that resolution model. The
// gate is `cd dashboard && uv run pytest tests/`, so a DOM harness would need
// an install at gate time or a skip-when-deps-missing guard — and a test that
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
// than in the browser. For the same reason these functions return plain
// render DESCRIPTORS rather than React elements — a module calling a global
// `React` would violate that suite by construction.

// ── The one true strand tooltip ──
// Exported once because all three render sites (tab_tasks.jsx renderNode,
// tab_tasks.jsx TaskDetail, tabs.jsx OrchTab) hand-copied this string
// verbatim before the extraction — three copies of one sentence, each free to
// drift from the others. It names BOTH triggers of the server-side verdict on
// purpose: an operator who reads only "stranded" cannot tell whether the
// claimant vanished or its heartbeat went stale.
const STRAND_TITLE = 'stranded: in-progress with no live claimant / stale heartbeat';

// ── Should the stranded badge render, and how? ──
// Returns a render descriptor `{cls, label, title[, marginLeft]}`, or null for
// "render nothing" — null rather than a disabled/negated descriptor so the
// call site keeps React's existing falsy-gate semantics unchanged.
//
// GATED ON `task.stranded` AND NOTHING ELSE. This is the entire point of the
// feature (task 3543) and the one thing that must never be "simplified": the
// strand verdict is computed server-side from the claim columns, whereas
// `task.agent` is only WORKTREE PRESENCE and stays truthy after the agent
// dies. Keying this off `agent` would make the stranded task that most needs
// the badge — the one whose warm-lane worktree outlived its agent — render
// identically to a live one. The badge and the agent cell below are
// deliberately independent surfaces.
//
// Truthiness, never an equality test against false: a task row can arrive
// before the strand verdict has been computed, and `stranded === false` would
// conflate "not stranded" with "not yet known". Both correctly render nothing
// here; only the affirmative claim ever draws.
//
// A null/undefined `task` is tolerated and renders nothing — rows render
// before task data has necessarily arrived, and throwing here would blank the
// surrounding table rather than one cell.
//
// `compact` is the dense graph-node site, which renders a glyph and no
// `style` attribute at all — so marginLeft is OMITTED from the descriptor
// rather than set to undefined. `marginLeft` is overridable because the two
// full-text sites genuinely disagree (TaskDetail 6, OrchTab 4); parameterised
// rather than unified so the extraction stays byte-for-byte
// behaviour-preserving. Unifying them is a visual change and belongs to a
// different task.
function strandBadgeState(task, opts) {
  if (!task || !task.stranded) return null;
  const o = opts || {};
  if (o.compact) {
    return { cls: 'badge bad', label: '⚠', title: STRAND_TITLE };
  }
  return {
    cls: 'badge bad',
    label: 'stranded',
    title: STRAND_TITLE,
    marginLeft: o.marginLeft !== undefined ? o.marginLeft : 6,
  };
}

// ── What does the `agent` cell say, and is it muted? ──
// Returns `{text, muted}`. `muted` means "this is a placeholder, not a real
// agent name", which is what the call sites render in the dim tertiary colour.
//
// The placeholder is a parameter because the two sites disagree: TaskDetail
// spells it 'unassigned', OrchTab renders an em-dash. Both are preserved.
//
// An empty-string agent is treated as absent — a blank cell would be
// indistinguishable from a failed load, and the placeholder is the
// unambiguous rendering.
//
// Kept in this module BECAUSE it must stay distinct from the badge above, not
// because they share logic: they read different fields and share no output
// keys, and that separation is the contract. Housing them together is what
// makes a future merge of the two visible in one diff.
function agentCellState(task, opts) {
  const t = task || {};
  const o = opts || {};
  if (t.agent) return { text: t.agent, muted: false };
  return { text: o.placeholder !== undefined ? o.placeholder : 'unassigned', muted: true };
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced at
// runtime by dashboard/tests/js/classic_script_scope.test.mjs. A collision
// here would leave window.DF_TASK_STRAND_BADGE undefined and break the
// top-level destructures in tab_tasks.jsx and tabs.jsx.
const TASK_STRAND_BADGE_API = { strandBadgeState, agentCellState, STRAND_TITLE };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TASK_STRAND_BADGE_API;
}
if (typeof window !== 'undefined') {
  window.DF_TASK_STRAND_BADGE = TASK_STRAND_BADGE_API;
}
