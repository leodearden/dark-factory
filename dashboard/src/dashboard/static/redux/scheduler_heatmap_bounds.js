/* scheduler_heatmap_bounds.js — axis selection for the Scheduler contention
   heatmap: which task rows and which lock modules are worth rendering.

   Pure and React-free by design, so it runs under `node --test` against a
   fixture at production scale (dashboard/tests/js/scheduler_heatmap_bounds.test.mjs).
   scheduler_heatmap.jsx is Babel-transformed in-browser and cannot be imported
   by any runner, so a bound written inline there could only be grep-asserted —
   which cannot demonstrate a bound at all.

   WHY 60. The cap is read off the rendered geometry, not picked. `.sched-cell`
   is `width: 22px` and `.sched-heatmap` sets `border-spacing: 2px`
   (styles.css), so a column costs ~26px; 60 columns is ~1,560px, which fits
   beside the 360px sticky `.sched-row-label` inside a 1,920px viewport. A cap
   wider than the screen buys nothing a human can read. 60 x 60 = 3,600 cells,
   against the 12,867,282 measured on the 2026-09-20 live snapshot.

   Exports: window.DF_SCHED_HEATMAP_BOUNDS =
     { boundHeatmapAxes, rowTouchesModule,
       MAX_HEATMAP_ROWS, MAX_HEATMAP_COLS, MAX_HEATMAP_CELLS }
*/

const MAX_HEATMAP_ROWS = 60
const MAX_HEATMAP_COLS = 60

// Derived, never stated independently: the two axis caps already determine it,
// and a hand-written third number would drift the moment either axis moved.
const MAX_HEATMAP_CELLS = MAX_HEATMAP_ROWS * MAX_HEATMAP_COLS

// Does this task row contend for this lock module?
function rowTouchesModule(row, module) {
  if (!row || !module) return false
  return (row.lock_set || []).includes(module.path)
}

// Does this module earn a column?
//
// Two disjuncts, and the second is not a nicety. `contention` counts LIVE
// WAITERS, so a module held by exactly one task has no contention to show. But
// scheduler.py injects an entry for every park-stack key even with no live
// waiters (contention: 0) so a fully-stranded module still gets one — and a
// bare `contention > 1` would hide exactly the stranded parks the Scheduler
// tab's red banner and ParkStacksSection already single out.
function moduleEarnsColumn(module) {
  return (module.contention || 0) > 1 || (module.park_stack || []).length > 0
}

// Choose the rows and columns the heatmap will actually render.
//
// Returns the selected axes alongside the INPUT totals and a per-axis
// truncation flag, so the caller can tell the user how much it is not showing.
function boundHeatmapAxes({ rows, modules }) {
  const allRows = rows || []
  const allModules = modules || []

  // `filter` preserves relative order, so the server's `(-contention, path)`
  // sort is inherited rather than re-derived on the client.
  const keptModules = allModules.filter(moduleEarnsColumn)
  const keptRows = allRows

  return {
    rows: keptRows,
    modules: keptModules,
    rowsTotal: allRows.length,
    modulesTotal: allModules.length,
    rowsTruncated: keptRows.length < allRows.length,
    modulesTruncated: keptModules.length < allModules.length,
  }
}

const SCHEDULER_HEATMAP_BOUNDS_API = {
  boundHeatmapAxes,
  rowTouchesModule,
  MAX_HEATMAP_ROWS,
  MAX_HEATMAP_COLS,
  MAX_HEATMAP_CELLS,
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = SCHEDULER_HEATMAP_BOUNDS_API
}
if (typeof window !== 'undefined') {
  window.DF_SCHED_HEATMAP_BOUNDS = SCHEDULER_HEATMAP_BOUNDS_API
}
