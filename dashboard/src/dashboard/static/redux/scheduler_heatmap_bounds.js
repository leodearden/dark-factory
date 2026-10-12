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
       MAX_HEATMAP_ROWS, MAX_HEATMAP_COLS, MAX_RESCUED_COLS }
*/

// The cell bound is the PRODUCT of these two and is deliberately not given a
// third name: a `MAX_HEATMAP_CELLS` constant would be a second source for a
// number these already determine, with no caller of its own to serve.
const MAX_HEATMAP_ROWS = 60
const MAX_HEATMAP_COLS = 60
// At most this many of MAX_HEATMAP_COLS go to stranded locks the contention
// cut would hide, so the head always keeps >= 50 columns; the 2026-10-12 live
// snapshot had 5.
const MAX_RESCUED_COLS = 10

// Does this task row contend for this lock module — i.e. would its cell be
// anything but blank?
//
// THE SINGLE SOURCE of that rule. scheduler_heatmap.jsx's cellStateFor
// delegates here for its first two branches, and row selection below reaches
// it through modulesByPath. Were the axis filter to restate the rule instead,
// it could drop a row whose cells the renderer would have coloured.
//
// Modules are keyed by `(project, path)` on the server, so a path match is not
// a lock match: two projects can each have `src/utils.py`, and a row from
// project B is not contending for project A's lock. A falsy project on either
// side is legacy/single-project mode and skips the check.
function rowTouchesModule(row, module) {
  if (!row || !module) return false
  if (module.project && row.project && module.project !== row.project) return false
  return (row.lock_set || []).includes(module.path)
}

// A stranded lock is one no live task can release: a park with a dead owner
// anywhere in its stack, or a holder that has left the active task set while
// a waiter still queues behind it. Both facts come from the server
// (has_dead_park, holder_live); `=== false` keeps unknown liveness inert.
function lockIsStranded(module) {
  return !!module.has_dead_park
    || (!!module.holder && module.holder_live === false && (module.contention || 0) > 0)
}

// The columns: a contended HEAD (a prefix of the contention > 1 modules in
// server order) plus a rescued TAIL (stranded modules the head did not reach).
// A slot is reserved only when a stranded module falls outside the head.
function selectHeatmapColumns(allModules) {
  const contended = allModules.filter(m => (m.contention || 0) > 1)
  const stranded = allModules.filter(lockIsStranded)
  const rankOf = new Map(contended.map((m, i) => [m, i]))
  const rescuedPast = headLen => stranded
    .filter(m => !(rankOf.has(m) && rankOf.get(m) < headLen))
    .slice(0, MAX_RESCUED_COLS)
  let headLen = Math.min(contended.length, MAX_HEATMAP_COLS)
  // Shrinking the head can push a stranded module out of it, so the length
  // is a fixed point; it settles within MAX_RESCUED_COLS steps, because at
  // MAX_HEATMAP_COLS - MAX_RESCUED_COLS the sum is always within budget.
  while (headLen + rescuedPast(headLen).length > MAX_HEATMAP_COLS) headLen--
  return { mostContended: contended.slice(0, headLen), rescued: rescuedPast(headLen) }
}

// Index columns by path so a row is scanned against its own lock_set rather
// than against every column: O(rows x lock_set) probes (~13k on the live
// snapshot) instead of O(rows x modules) (~12.9M). A path can carry more than
// one module when two projects share it, which is why the value is a list and
// why the project rule still has to run per hit.
function heatmapModulesByPath(modules) {
  const modulesByPath = new Map()
  for (const module of modules) {
    const atPath = modulesByPath.get(module.path)
    if (atPath) atPath.push(module)
    else modulesByPath.set(module.path, [module])
  }
  return modulesByPath
}

// Deliberately reaches the exported predicate rather than reimplementing it
// against the index — the index accelerates the lookup and carries no copy
// of the membership rule.
function rowTouchesAnyIndexed(row, modulesByPath) {
  for (const path of (row.lock_set || [])) {
    for (const module of modulesByPath.get(path) || []) {
      if (rowTouchesModule(row, module)) return true
    }
  }
  return false
}

// A parked row is kept whatever its columns do — it is what the Scheduler
// tab's stranded-parks banner is pointing at — and it is taken FIRST,
// because a carve-out that only survives the filter does not survive this
// function.  `shape_scheduler` appends the synthetic stranded-park rows
// after every project's ordinary rows, so on a list the size of the live
// snapshot a positional prefix fills up long before it reaches them and
// drops precisely the rows the carve-out was written to keep.  A row waiting
// on a rescued column comes next: it is blocked on a stranded lock, as stuck
// as a parked one, and without it the rescued column renders blank.  That
// tier is bounded by MAX_RESCUED_COLS columns, each no more contended than
// the head's minimum (15 rows on the live snapshot).
//
// Ordinary rows therefore yield to parked ones when the cap binds.  That
// trade is one-sided in practice (the live snapshot has 7 parked rows
// against 2,991) and is the right way round when it is not: a row is on
// this tab to be diagnosed, and a parked one is already stuck.
function selectHeatmapRows(allRows, keptModules, rescued) {
  const keptByPath = heatmapModulesByPath(keptModules)
  const rescuedByPath = heatmapModulesByPath(rescued)
  const parkedRows = []
  const waitingRows = []
  const touchingRows = []
  for (const row of allRows) {
    if (((row.park_state || {}).modules || []).length > 0) parkedRows.push(row)
    else if (rowTouchesAnyIndexed(row, rescuedByPath)) waitingRows.push(row)
    else if (rowTouchesAnyIndexed(row, keptByPath)) touchingRows.push(row)
  }
  return parkedRows.concat(waitingRows, touchingRows).slice(0, MAX_HEATMAP_ROWS)
}

// Choose the rows and columns the heatmap will actually render.
//
// Returns the selected axes alongside the INPUT totals and a per-axis
// truncation flag, so the caller can tell the user how much it is not showing.
//
// The server's `(-contention, path)` sort is inherited rather than re-derived:
// the head is a prefix of the contended modules in that order, i.e. the most
// contended ones, and every rescued module lies after the head in it, so
// head-then-tail is still server order.  Columns are chosen BEFORE rows so
// rows are chosen against the columns that will actually render — a row
// touching only a column past the cap would otherwise survive it and render
// as 60 blank cells.
function boundHeatmapAxes({ rows, modules }) {
  const allRows = rows || []
  const allModules = modules || []
  const { mostContended, rescued } = selectHeatmapColumns(allModules)
  const keptModules = mostContended.concat(rescued)
  const keptRows = selectHeatmapRows(allRows, keptModules, rescued)

  return {
    rows: keptRows,
    modules: keptModules,
    rowsTotal: allRows.length,
    modulesTotal: allModules.length,
    modulesRescued: rescued.length,
    rowsTruncated: keptRows.length < allRows.length,
    modulesTruncated: keptModules.length < allModules.length,
  }
}

const SCHEDULER_HEATMAP_BOUNDS_API = {
  boundHeatmapAxes,
  rowTouchesModule,
  MAX_HEATMAP_ROWS,
  MAX_HEATMAP_COLS,
  MAX_RESCUED_COLS,
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = SCHEDULER_HEATMAP_BOUNDS_API
}
if (typeof window !== 'undefined') {
  window.DF_SCHED_HEATMAP_BOUNDS = SCHEDULER_HEATMAP_BOUNDS_API
}
