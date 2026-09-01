// task_row_cells.js — the pure render DECISIONS behind the two cells that make
// up a task row's claim column, for the Tasks tab (tab_tasks.jsx) and the
// Orchestrators tab (tabs.jsx):
//
//   1. the stranded badge (strandBadgeState), at three sites;
//   2. the agent cell it sits beside (agentCellState), at two of them.
//
// NAMED FOR THE ROW, NOT FOR THE BADGE. The first spelling of this module was
// task_strand_badge.js, which was a lie by omission: agentCellState decides
// nothing about the badge — it reads `task.agent`, the field the badge is
// deliberately independent OF — so a reader looking for the agent-cell rule had
// no reason to open a file named for the strand badge. The two belong in one
// module (see agentCellState's own note below: co-located so that a future
// merge of the two is visible in ONE diff), but the module has to be named for
// what it actually owns. Renamed in the task-4361 amendment pass; the browser
// global moved with it, DF_TASK_STRAND_BADGE -> DF_TASK_ROW_CELLS.
//
// Dual-loaded: a browser classic `<script>` assigns `window.DF_TASK_ROW_CELLS`,
// node resolves the same file as CommonJS for `dashboard/tests/js/`. index.html
// loads it before the Babel JSX tags, so the global exists before tab_tasks.jsx
// and tabs.jsx run their top-level destructures of it.
//
// ── THE SHARED SUBSTRATE DECISION IS NOT RESTATED HERE ────────────────────
// Why these helpers exist at all, why a DOM harness was considered and
// REJECTED, why re-hardening the deleted greps is out of scope, and why no
// module here reads a browser global: written out ONCE, in pins_recovery.js's
// header (the block marked CANONICAL). Read it there before re-litigating any
// of it. It is deliberately not copied into this file — three hand-copies of
// one rationale drift, which is the hazard these modules exist to remove, and
// which STRAND_TITLE below was extracted to fix in miniature.
//
// ── WHY THIS MODULE EXISTS (the stranded-badge specifics) ─────────────────
// The block that covered THIS surface, test_tab_tasks_stranded_badge.py (187
// lines), carried the shared defect plus one of its own: NEGATED regexes, which
// passed vacuously when the feature was absent altogether. Delete the badge and
// the assertions asking that it not be rendered wrongly all went green.
//
// That is now the other way round. The render DECISION lives in the pure
// function below with behavioural coverage in
// dashboard/tests/js/task_row_cells.test.mjs, so deleting a decision arm — or
// keying the badge off `agent` instead of `stranded` — fails a named test
// instead of nothing.

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
// makes a future merge of the two visible in one diff. That co-location is
// also why the module is named for the ROW rather than for the badge — see the
// header.
//
// The two call sites consume this differently ON PURPOSE: TaskDetail honours
// `muted` by wrapping the placeholder in the dim tertiary colour, OrchTab takes
// only `.text` and renders its em-dash undimmed. Both are what those sites drew
// before the extraction, and unifying them is a visual change belonging to a
// different task — the same reason marginLeft above stayed a parameter.
function agentCellState(task, opts) {
  const t = task || {};
  const o = opts || {};
  if (t.agent) return { text: t.agent, muted: false };
  return { text: o.placeholder !== undefined ? o.placeholder : 'unassigned', muted: true };
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced at
// runtime by dashboard/tests/js/classic_script_scope.test.mjs. A collision
// here would leave window.DF_TASK_ROW_CELLS undefined and break the
// top-level destructures in tab_tasks.jsx and tabs.jsx.
const TASK_ROW_CELLS_API = { strandBadgeState, agentCellState, STRAND_TITLE };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TASK_ROW_CELLS_API;
}
if (typeof window !== 'undefined') {
  window.DF_TASK_ROW_CELLS = TASK_ROW_CELLS_API;
}
