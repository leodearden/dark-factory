// pins_recovery.js — the pure render DECISIONS behind both pins_recovery
// surfaces in the dashboard:
//
//   1. the per-row PINNING chip beside the 6h+ breach badge, in
//      tab_escalation_analytics.jsx (pinningBadgeState);
//   2. the "pinning" StatTile fed by the open-items reduction, in
//      tab_escalations.jsx (pinningSummary).
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/pins_recovery.js">`
//     tag (like task_status_counts.js), which assigns `window.DF_PINS_RECOVERY`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// index.html loads this file (classic script, before the Babel JSX tags) so
// `window.DF_PINS_RECOVERY` is defined before tab_escalations.jsx and
// tab_escalation_analytics.jsx execute their top-level destructures of it.
//
// ── WHY THIS MODULE EXISTS ────────────────────────────────────────────────
// Commit 039e55c7ef deleted four JSX source-text meta-test blocks (task 3543)
// because they asserted regexes and substrings over raw .jsx source fetched
// over HTTP rather than over behaviour, and could not discriminate. THIS
// SURFACE IS THE CLEANEST DEMONSTRATION OF WHY, and the reason the deletion
// was right: a whole-file substring grep is satisfied by a MENTION, so the
// explanatory COMMENT at tab_escalation_analytics.jsx:414-419 alone satisfied
// `'pins_recovery' in body` — the assertion passed with the render arm at
// :420-428 deleted. It could not fail for the one reason it existed. The
// tab_escalations.jsx block had the same defect: `label="pinning"` was
// satisfied by the raw body with the tile removed.
//
// Deleting them was correct and left a real hole. This module closes the part
// of it that can be closed: both decisions now live in pure functions with
// genuine behavioural coverage (dashboard/tests/js/pins_recovery.test.mjs), so
// deleting the pins_recovery arm — or adding a negated one — fails a test
// instead of nothing.
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
// than in the browser. For the same reason these functions return plain
// render DESCRIPTORS rather than React elements.
//
// ── THE THREE-STATE CONTRACT THAT GOVERNS BOTH FUNCTIONS ──────────────────
// The backend emits THREE distinguishable states, and both surfaces below
// honour all three:
//   - `pins_recovery: true`  — computed; this escalation pins a recovery;
//   - `pins_recovery: false` — computed; it pins nothing;
//   - key ABSENT             — could NOT be computed (that project's
//     escalation MCP was unreadable, or the record carried no annotation).
// ABSENT IS NOT FALSE. Hence truthiness throughout and never an equality test
// against `false`: `item.pins_recovery === false` would treat an
// unclassified item as a computed "does not pin", inventing a verdict the
// backend explicitly declined to reach. Unknown simply falls out — it draws
// nothing and counts as nothing. The server side of this contract is
// escalation_analytics.py's `item['pins_recovery'] = bool(pinned)`, which
// omits the key entirely when the pin map is None.

// ── Does this item pin a recovery? ──
// ONE definition, shared by both surfaces below so they cannot disagree about
// what counts as a pin.
//
// THE `.length === 0` GUARD IS NOT REDUNDANT WITH TRUTHINESS. An empty array
// is truthy in JavaScript, so an unnormalised `pins_recovery: []` would
// otherwise count as a pin. The server normalises with `bool(pinned)` and a
// well-formed payload never carries a bare list here — but pinning the
// client-side guard is what stops the two layers each assuming the other did
// the normalising.
//
// Truthiness, never `=== false`: see the three-state contract above.
function pinsRecovery(item) {
  return !!(item && item.pins_recovery && item.pins_recovery.length !== 0);
}

// ── Should the per-row PINNING chip draw, and what does its tooltip say? ──
// Returns `{cls, label, title}` or null for "render nothing" — null rather
// than a disabled descriptor so the call site keeps React's falsy-gate
// semantics unchanged.
//
// THERE IS NO NEGATED ARM, deliberately. "Does not pin recovery" over an
// unclassified record would be a claim nobody made, and an operator would act
// on it. Only the affirmative case ever draws; both other states draw nothing
// and are indistinguishable at the chip, which is correct — the chip's job is
// to flag a pin, not to certify its absence.
//
// THE ID LIST IS BUILT FIRST, AND THE "of task <ids>" CLAUSE IS ONLY
// INTERPOLATED WHEN IT IS NON-EMPTY. The annotation and the id list are
// separate fields, so `pins_recovery: true` with no ids is reachable — and the
// chip must still draw, because the pin itself is the operator-relevant fact.
// What must not happen is the title degrading into "recovery of task  — this
// escalation ...", a dangling empty slot that reads as a rendering bug and
// buries the real message. (The inline template literal this replaces did
// exactly that.)
function pinningBadgeState(item) {
  if (!pinsRecovery(item)) return null;
  const ids = (item.pins_recovery_task_ids || []).join(', ');
  const what = ids ? ` of task ${ids}` : '';
  return {
    cls: 'badge bad',
    label: 'PINNING',
    title: `PINNING recovery${what} — this escalation is what stops it being redispatched`,
  };
}

// ── How many open escalations pin a recovery, and how many tasks do they block? ──
// Reduces the live open-items array (lifespan.open_items, concatenated across
// projects) to `{count, pinnedTaskCount}` for the "pinning" StatTile.
//
// `count` is escalations; `pinnedTaskCount` is DISTINCT tasks. Two escalations
// pinning the same task block one task between them, so the ids are unioned
// rather than summed, and they are Stringified before entering the set because
// task ids arrive as both numbers and strings across projects' payloads — a
// set of raw values would count 7 and '7' as two blocked tasks.
//
// A pinning item whose id list did not survive still counts toward `count` —
// the pin is real — while contributing nothing to `pinnedTaskCount`, which is
// the same asymmetry the chip's degradation case makes.
//
// What counts as a pin is `pinsRecovery` above, shared with the chip.
function pinningSummary(openItems) {
  const items = (openItems || []).filter(pinsRecovery);
  const pinnedTasks = new Set();
  for (const item of items) {
    for (const tid of item.pins_recovery_task_ids || []) pinnedTasks.add(String(tid));
  }
  return { count: items.length, pinnedTaskCount: pinnedTasks.size };
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced at
// runtime by dashboard/tests/js/classic_script_scope.test.mjs. A collision
// here would leave window.DF_PINS_RECOVERY undefined and break the top-level
// destructures in tab_escalations.jsx and tab_escalation_analytics.jsx.
//
// `pinsRecovery` is deliberately NOT exported: it is the shared predicate the
// two public surfaces agree through, not a third surface of its own.
const PINS_RECOVERY_API = { pinningBadgeState, pinningSummary };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PINS_RECOVERY_API;
}
if (typeof window !== 'undefined') {
  window.DF_PINS_RECOVERY = PINS_RECOVERY_API;
}
