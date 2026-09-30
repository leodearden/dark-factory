// task_row_cells.js — the pure render DECISIONS behind the two cells that make
// up a task row's claim column, for the Tasks tab (tab_tasks.jsx) and the
// Orchestrators tab (tabs.jsx):
//
//   1. the stranded badge (strandBadgeState), at three sites;
//   2. the agent cell it sits beside (agentCellState), at two of them;
//   3. the Locks cell of that same row (locksCellState), at one.
//
// THE THIRD ONE ARRIVED LAST AND FOR THE SAME REASON THE FIRST TWO DID. The
// Locks column is a cell of the identical task row, and its decision was a
// branch inside tabs.jsx — un-executable JSX, which is how its offline-
// scheduler arm came to be un-asserted at all. Housing the row's three render
// decisions together is what makes a change to the row visible in ONE diff,
// the same argument agentCellState's note below makes for the first two. It
// also keeps tabs.jsx (at its soft size ceiling) from growing a branch.
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

// The Datum envelope's readers. Module scope, no fallback, RENAMED — this is a
// classic script, so a bare `datumView` would collide with datum.js's top-level
// declaration of the same name. See the CANONICAL note in datum.js's header.
const {
  assertDatum: requireDatum,
  datumView: viewOfDatum,
  EM_DASH: DATUM_PLACEHOLDER,
} = window.DF_DATUM;

// ── The one true strand tooltip ──
// Exported once because all three render sites (tab_tasks.jsx renderNode,
// tab_tasks.jsx TaskDetail, tabs.jsx OrchTab) hand-copied this string
// verbatim before the extraction — three copies of one sentence, each free to
// drift from the others. It names BOTH triggers of the server-side verdict on
// purpose: an operator who reads only "stranded" cannot tell whether the
// claimant vanished or its heartbeat went stale.
const STRAND_TITLE = 'stranded: in-progress with no live claimant / stale heartbeat';

// ── The one true mute colour ──
// The dim tertiary custom property a PLACEHOLDER renders in, exported for
// exactly the reason STRAND_TITLE above is. Before task 4408 this literal was
// hand-written in tab_tasks.jsx's JSX, and unifying the two agent-cell sites by
// hand-copying it into tabs.jsx as well would have recreated, in miniature, the
// same drift hazard this module exists to remove. Exported once, so the two
// sites cannot come to disagree about how dim a placeholder is — and so the
// colour is assertable in the node suite, which a JSX literal never is.
const MUTED_COLOR = 'var(--fg-3)';

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

// ── What does the `agent` cell say, and how is it coloured? ──
// Returns `{text, color}`. `color` is MUTED_COLOR when the text is a
// PLACEHOLDER rather than a real agent name, and null when it is a real name
// and no colour override applies. Both call sites render it identically.
//
// The placeholder STRING is a parameter, because the two sites genuinely
// disagree: TaskDetail spells it 'unassigned', OrchTab renders an em-dash.
// Both are preserved. The placeholder STYLING is NOT a parameter — how dim a
// placeholder renders is one decision for the whole surface, and it is made
// here.
//
// WHY A COLOUR AND NOT A BOOLEAN (task 4408). This returned `{text, muted}`
// until that task, and `muted` was honoured at ONE of its two call sites:
// TaskDetail wrapped its placeholder in the dim tertiary colour, while OrchTab
// took only `.text` and discarded the mute — so its em-dash rendered at exactly
// the colour a real agent name would. A placeholder that reads as a value is
// worst precisely THERE, because the OrchTab agent cell abuts the stranded
// badge, which is the one place an operator scans for "is anything actually
// claiming this task". Returning the colour fixes both halves at once: the mute
// is honoured at both sites, and the field carrying the placeholder-ness is now
// the field a site must read in order to render at all, so a site that drops it
// is visibly dropping the descriptor's only other key. `muted` was REMOVED
// rather than kept alongside `color` — one bit living in two discardable fields
// would re-create this very defect one layer down.
//
// `color` is present-and-null for a real agent, never omitted — deliberately
// the opposite of how compact omits marginLeft above. The two resolutions have
// DIFFERENT reasons, and neither is about spreading: no call site spreads
// either descriptor, they all read explicit fields. compact omits marginLeft
// because that site renders no `style` attribute at all (see the badge note
// above), so there is nothing for the key to feed; these two agent-cell sites
// BRANCH on `color`, so it must always be there to branch on. A key that is
// always present makes `ac.color ? … : ac.text` a total function over a stable
// shape, and keeps "no colour override" an affirmative decision rather than an
// absent key.
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
// Note the contrast with the badge above, which is NOT a leftover: the agent
// cell's mute styling is unified across both sites as of task 4408, while
// strandBadgeState's marginLeft (TaskDetail 6, OrchTab 4) deliberately is not.
// That is a separate open visual question about a different field on a
// different descriptor, and unifying it here would have smuggled a second
// unrequested pixel change into this one.
function agentCellState(task, opts) {
  const t = task || {};
  const o = opts || {};
  if (t.agent) return { text: t.agent, color: null };
  return {
    text: o.placeholder !== undefined ? o.placeholder : 'unassigned',
    color: MUTED_COLOR,
  };
}

// ── Does the Locks cell draw its chips, or a placeholder? ──
// Returns `{placeholder, title}`: `placeholder` is the shared em-dash when
// nothing is known about this task's locks and null when the caller should
// render its chip list; `title` is the producer's reason as a tooltip, or null.
//
// WHAT IT FIXES. This cell's chips come from DF.SCHEDULER, so a project whose
// scheduler is offline renders an EMPTY chip list — indistinguishable from a
// task that genuinely holds no locks. "We don't know" and "nothing is held"
// are opposite answers for an operator deciding whether a task is blocked, and
// they rendered identically. With the datum, a blank cell means what it says
// again and the em-dash carries its own reason (PRD decision 15).
//
// `lockInfo` IS A PARAMETER AND IS DELIBERATELY NEVER READ. It is here because
// the caller holds it and a later reader will look for it, and the thing worth
// knowing is that it is NOT an input: an unknown datum draws the placeholder
// even when lock paths are sitting right there. Chips from a snapshot the
// datum cannot vouch for are exactly the unprovenanced render this envelope
// exists to remove, so "we don't know" has to win over "here is what we last
// saw". task_row_cells.test.mjs pins that absence of a branch directly.
//
// NEITHER THE HOLE RULE NOR THE TOOLTIP RULE IS RE-DERIVED HERE. datumView
// decides both — when a reason is worth surfacing (a fresh value has nothing to
// explain, so a stray reason on one is not shown), and whether there is a
// measurement at all — and asking it for both is what keeps this cell following
// those rules if either ever changes. A `datum.state === 'unknown'` test here
// would make the Locks column a second authority on the question 43 tiles and
// 17 pips already ask datumView, and its own node suite would stay green while
// the two answers diverged. Its `text` is discarded and `format` returns
// nothing on purpose: this cell has no text of its own — its value is a chip
// LIST, which the caller renders — so what is read back is the hole flag and
// the tooltip.
//
// `placeholder` is present-and-null on the known arms rather than an absent
// key, for the reason agentCellState's `color` is: the call site BRANCHES on
// it, so it must always be there to branch on.
//
// Unlike strandBadgeState, a non-Datum THROWS rather than rendering nothing.
// The two differ because their inputs do: a task row legitimately arrives
// before its strand verdict exists, whereas a caller that reaches this
// function at all has been migrated to pass a Datum, so a bare value here is
// a missed migration site and must fail by name in dev.
function locksCellState(datum, lockInfo) {
  requireDatum(datum, 'locksCellState');
  const view = viewOfDatum(datum, { format: () => '' });
  return {
    placeholder: view.isHole ? DATUM_PLACEHOLDER : null,
    title: view.title,
  };
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced at
// runtime by dashboard/tests/js/classic_script_scope.test.mjs. A collision
// here would leave window.DF_TASK_ROW_CELLS undefined and break the
// top-level destructures in tab_tasks.jsx and tabs.jsx.
const TASK_ROW_CELLS_API = { strandBadgeState, agentCellState, locksCellState, STRAND_TITLE, MUTED_COLOR };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TASK_ROW_CELLS_API;
}
if (typeof window !== 'undefined') {
  window.DF_TASK_ROW_CELLS = TASK_ROW_CELLS_API;
}
