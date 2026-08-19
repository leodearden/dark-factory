// runtime_format.js — pure offline-'—' degradation formatters for warm-lane
// runtime fields (loops/attempts/started/lane/phase/lane_state) rendered by
// the Orchestrators tab (OrchTab in tabs.jsx) and the Tasks tab's TaskDetail
// (tab_tasks.jsx).
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/runtime_format.js">`
//     tag (like graph_layout.js/prd_grouping.js), which assigns
//     `window.DF_RUNTIME_FMT`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in.
//
// index.html loads this file (classic script, before the Babel JSX tags) so
// `window.DF_RUNTIME_FMT` is defined before tabs.jsx / tab_tasks.jsx execute
// their top-level `const { rtCell, rtAge } = window.DF_RUNTIME_FMT;` /
// `const { rtCell } = window.DF_RUNTIME_FMT;` destructures.
//
// Load-bearing contract (shared/src/shared/task_runtime_state.py +
// active_tasks.py::_runtime_fields, task 2636): when a project's orchestrator
// snapshot is offline, EVERY runtime field on a task row is null; an honest
// not-yet-iterated online task reports an honest 0. A loose `== null` check
// (covers both null and undefined) therefore renders '—' only for the
// offline/per-task-read-error case, never for a genuine zero.

// ── Per-cell offline formatter: null/undefined -> em-dash, else passthrough ──
function rtCell(v) {
  return v == null ? '—' : v;
}

// ── Age formatter: null/undefined -> em-dash, else "<minutes>m" ──
// `started` is contractually an integer minute count (task_runtime_state.py),
// but Math.round guards the render against float/precision drift if that
// ever changes upstream (e.g. a fractional-minute value would otherwise
// render as "3.5m" verbatim) — a no-op for the documented integer contract.
function rtAge(m) {
  return m == null ? '—' : `${Math.round(m)}m`;
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced by
// dashboard/tests/js/classic_script_scope.test.mjs. A collision here would
// leave window.DF_RUNTIME_FMT undefined and break tabs.jsx's / tab_tasks.jsx's
// top-level destructure of it.
const RUNTIME_FORMAT_API = { rtCell, rtAge };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = RUNTIME_FORMAT_API;
}
if (typeof window !== 'undefined') {
  window.DF_RUNTIME_FMT = RUNTIME_FORMAT_API;
}
