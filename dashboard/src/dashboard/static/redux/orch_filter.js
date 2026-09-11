// orch_filter.js — the pure empty-state label for the Orchestrators tab's
// multi-select task filter (OrchTab in tabs.jsx).
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/orch_filter.js">`
//     tag (like graph_layout.js/prd_grouping.js/runtime_format.js), which
//     assigns `window.DF_ORCH_FILTER`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// index.html loads this file (classic script, before the Babel JSX tags) so
// `window.DF_ORCH_FILTER` is defined before tabs.jsx executes its top-level
// destructure of it. That destructure carries a `|| { orchEmptyLabel }`
// fallback, so a 404'd or mis-ordered load costs the operator one cosmetic
// label rather than blanking every tab defined in tabs.jsx.
//
// Load-bearing contract (task 3313): the cell this feeds used to read
// `No {filter === 'all' ? '' : filter + ' '}tasks`, but OrchTab's `filter` is
// an object ({active, pending, complete}) — the equality was permanently
// false and the concatenation stringified, rendering the literal
// "No [object Object] tasks" to an operator. Nothing here may interpolate the
// filter object itself; the label is assembled only from the facet names
// below.

// ── Canonical facet order: [state key, operator-facing name] ──
// Fixed here rather than derived from Object.keys(filter): flipFilter rebuilds
// the per-pid object with spread on every click, so key insertion order tracks
// the operator's click history and would make the same two facets read
// "No complete or active tasks" for one operator and "No active or complete
// tasks" for another. This order matches the Active/Pending/Complete segmented
// control rendered directly above the table, so the sentence names the facets
// in the order the eye scans the buttons.
const ORCH_FILTER_FACETS = [
  ['active', 'active'],
  ['pending', 'pending'],
  ['complete', 'complete'],
];

// The none-selected state is deliberately a different sentence shape, not a
// degenerate "No tasks": "your filters exclude everything" and "this
// orchestrator has no matching tasks" call for different operator responses,
// and all-off is reachable by clicking (flipFilter has no at-least-one
// invariant) and persists to localStorage under df.orch.filter — so an
// operator can land on it cold with no memory of having toggled anything. The
// suffix names the remedy.
const ORCH_FILTER_NONE_SELECTED =
  'No filters selected — choose Active, Pending or Complete above';

// ── Empty-state label for a given filter state ──
// The only producer is OrchTab's getFilter (tabs.jsx), which always hands us a
// normalised {active, pending, complete} object — the legacy string values it
// still tolerates in localStorage are mapped to DEFAULT_FILTER before they ever
// reach this function. The non-object branch below is therefore unreachable in
// practice and exists only so a future caller cannot make this throw (a throw
// during render takes out all of OrchTab, not just this one cell).
//
// It falls back to getFilter's own default — active-only — rather than
// inventing a third behaviour: were the branch ever reached, the table would in
// fact be showing the active facet, and a "no filters selected" sentence would
// point the operator at a remedy that isn't the problem.
function orchEmptyLabel(filter) {
  const f = (filter && typeof filter === 'object') ? filter : { active: true };
  const names = ORCH_FILTER_FACETS.filter(([key]) => f[key]).map(([, name]) => name);

  if (names.length === 0) return ORCH_FILTER_NONE_SELECTED;
  if (names.length === 1) return `No ${names[0]} tasks`;

  // Two or more: comma-separate all but the last, then " or " the last —
  // "No active or pending tasks" / "No active, pending or complete tasks".
  const head = names.slice(0, -1).join(', ');
  const last = names[names.length - 1];
  return `No ${head} or ${last} tasks`;
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced by
// dashboard/tests/js/classic_script_scope.test.mjs. A collision here would
// leave window.DF_ORCH_FILTER undefined and downgrade every empty cell to
// tabs.jsx's fallback label. The same rule is why this file's other top-level
// names — ORCH_FILTER_FACETS, ORCH_FILTER_NONE_SELECTED, orchEmptyLabel — are
// prefixed rather than generic.
const ORCH_FILTER_API = { orchEmptyLabel };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = ORCH_FILTER_API;
}
if (typeof window !== 'undefined') {
  window.DF_ORCH_FILTER = ORCH_FILTER_API;
}
