// task_done_count.js — INTERIM guard for a project's done count, the CLIENT
// half of PRD leaf beta (task 5587). /api/v2/dashboard/tasks no longer serves
// DONE_COUNTS; the count now lives in TASKS_SNAPSHOT[p].census, a Datum that
// says whether it was measured.
//
// WHY THIS EXISTS. Removing DONE_COUNTS did not blank the done pips, it made
// them lie. data.js seeded `DONE_COUNTS: {}`, so `DONE_COUNTS[p]` read
// undefined, and both consumers fell through to counting the done rows in
// ACTIVE_TASKS. The default render fetches none of those rows, so every
// healthy project showed a confident "0 done".
//
// WHEN IT GOES AWAY. Leaves γ2 (the Orchestrators tab, task 5589) and γ3 (the
// Tasks tab) render the census as a full Datum, with its age and its reason,
// and delete this module and both destructures. It lives in ONE place so that
// deletion is one move.
//
// WHAT IT ANSWERS. The count when the census is `fresh`, and datum.js's
// placeholder in every other case, never a zero it did not read. Keyed on
// `state === 'fresh'` rather than on "a value is present", so that for a
// project whose rows were measured the pip agrees with task_snapshot.classify:
// a census that is not fresh is exactly what names a project in
// TASKS_COUNT_UNKNOWN_PROJECTS. The price, for this one leaf, is that a
// `stale` census's real, aged count shows as the placeholder too. γ3 adds the
// aged rendering.
//
// Plain-JS classic script, no JSX/Babel: loaded in the browser by a
// `<script>` tag in index.html (assigning `window.DF_TASK_DONE_COUNT`) and in
// node as CommonJS by dashboard/tests/js/task_done_count.test.mjs.

// The placeholder is datum.js's, destructured at module scope with no
// fallback and RENAMED, because classic scripts share one lexical scope — the
// CANONICAL note in datum.js's header.
const { EM_DASH: DONE_COUNT_UNMEASURED } = window.DF_DATUM;

// Whether *entry* (one TASKS_SNAPSHOT[p] value) carries a MEASURED done count.
// Kept apart from doneCount so a caller that needs the verdict does not have
// to compare the rendered text against the placeholder.
function hasDoneCount(entry) {
  const census = entry && entry.census;
  if (!census || census.state !== 'fresh') return false;
  const counts = census.value && census.value.counts;
  const done = counts && counts.done;
  return typeof done === 'number' && Number.isFinite(done);
}

// The measured done count, or the placeholder. A measured 0 is a count.
function doneCount(entry) {
  return hasDoneCount(entry) ? entry.census.value.counts.done : DONE_COUNT_UNMEASURED;
}

// Module-unique export const, never a bare `API`: see orch_summary.js, and
// dashboard/tests/js/classic_script_scope.test.mjs, which enforces it.
const TASK_DONE_COUNT_API = { hasDoneCount, doneCount };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TASK_DONE_COUNT_API;
}
if (typeof window !== 'undefined') {
  window.DF_TASK_DONE_COUNT = TASK_DONE_COUNT_API;
}
