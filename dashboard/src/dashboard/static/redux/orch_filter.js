// orch_filter.js — the pure empty-state sentence for the Orchestrators tab's
// multi-select VIEW filter (OrchTab in tabs.jsx).
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/orch_filter.js">`
//     tag, which assigns `window.DF_ORCH_FILTER`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require` for the `node --test` suite under
//     dashboard/tests/js/.
//
// index.html loads this file after task_snapshot.js, whose CENSUS_VIEWS it
// destructures at module scope, and before the Babel JSX tags, so
// `window.DF_ORCH_FILTER` is defined before tabs.jsx executes its top-level
// destructure of it. That destructure carries a `|| { orchEmptyLabel }`
// fallback, so a 404'd or mis-ordered load costs the operator one cosmetic
// label rather than blanking every tab defined in tabs.jsx.
//
// Load-bearing contract (task 3313): the cell this feeds used to read
// `No {filter === 'all' ? '' : filter + ' '}tasks`, but OrchTab's `filter` is
// an object — the equality was permanently false and the concatenation
// stringified, rendering the literal "No [object Object] tasks" to an
// operator. Nothing here may interpolate the filter object itself; the
// sentence is assembled only from the view labels.

// ── The facets ARE the census views ──
// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header. Taken from task_snapshot.js rather than hand-copied, so the view
// labels exist once. Their order is also fixed there rather than derived from
// Object.keys(filter): flipFilter rebuilds the per-pid object with spread on
// every click, so key insertion order tracks the operator's click history and
// would make the same two views read in either order. View order is the order
// of the filter buttons directly above the table.
const { CENSUS_VIEWS: ORCH_FILTER_VIEWS } = window.DF_TASK_SNAPSHOT;

// "a", "a or b", "a, b or c".
function orchFilterList(names) {
  const head = names.slice(0, -1);
  const last = names[names.length - 1];
  return head.length === 0 ? last : `${head.join(', ')} or ${last}`;
}

// The none-selected state is deliberately a different sentence shape, not a
// degenerate "No tasks": "your filters exclude everything" and "this
// orchestrator has no matching tasks" call for different operator responses,
// and all-off is reachable by clicking (flipFilter has no at-least-one
// invariant) and persists to localStorage — so an operator can land on it cold
// with no memory of having toggled anything. The suffix names the remedy.
const ORCH_FILTER_NONE_SELECTED =
  `No filters selected — choose ${orchFilterList(ORCH_FILTER_VIEWS.map(view => view.label))} above`;

// OrchTab's own default — in-flight only.
const ORCH_FILTER_DEFAULT = Object.freeze({ in_flight: true });

// ── Empty-state sentence for a given filter state ──
// The only producer is OrchTab's getFilter (tabs.jsx), which always hands us a
// normalised object. The non-object branch is therefore unreachable in
// practice and exists only so a future caller cannot make this throw (a throw
// during render takes out all of OrchTab, not just this one cell). It falls
// back to OrchTab's default rather than inventing a third behaviour: were it
// ever reached, the table would in fact be showing in-flight rows.
function orchEmptyLabel(filter) {
  const f = filter && typeof filter === 'object' ? filter : ORCH_FILTER_DEFAULT;
  const names = ORCH_FILTER_VIEWS.filter(view => f[view.key]).map(view => view.label);
  return names.length === 0 ? ORCH_FILTER_NONE_SELECTED : `No ${orchFilterList(names)} tasks`;
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced by dashboard/tests/js/classic_script_scope.test.mjs.
// A collision here would leave window.DF_ORCH_FILTER undefined and downgrade
// every empty cell to tabs.jsx's fallback label. The same rule is why this
// file's other top-level names are prefixed rather than generic.
const ORCH_FILTER_API = { orchEmptyLabel };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = ORCH_FILTER_API;
}
if (typeof window !== 'undefined') {
  window.DF_ORCH_FILTER = ORCH_FILTER_API;
}
