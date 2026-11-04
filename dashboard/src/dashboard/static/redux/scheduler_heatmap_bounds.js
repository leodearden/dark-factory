/* scheduler_heatmap_bounds.js — axis selection for the Scheduler contention
   heatmap: which task rows and which lock modules are worth rendering, and the
   caption that tells the user what that selection left out.

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
     { boundHeatmapAxes, rowTouchesModule, describeHeatmapCap,
       MAX_HEATMAP_ROWS, MAX_HEATMAP_COLS, MAX_RESCUED_COLS }
*/

// The cell bound is the PRODUCT of these two and is deliberately not given a
// third name: a `MAX_HEATMAP_CELLS` constant would be a second source for a
// number these already determine, with no caller of its own to serve.
const MAX_HEATMAP_ROWS = 60
const MAX_HEATMAP_COLS = 60
// At most this many of MAX_HEATMAP_COLS go to abandoned locks the contention
// cut would hide, so the head always keeps at least
// MAX_HEATMAP_COLS - MAX_RESCUED_COLS columns.
const MAX_RESCUED_COLS = 10

// Does this task row contend for this lock module — i.e. would its cell be
// anything but blank?
//
// THE SINGLE SOURCE of that rule. scheduler_heatmap.jsx's cellStateFor
// delegates here for its first two branches, and row selection below reaches
// it through rowTouchesAnyIndexed. Were the axis filter to restate the rule instead,
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

// An abandoned lock is one no live task can release: a park with a dead owner
// anywhere in its stack, or a holder that has left the active task set while
// a waiter still queues behind it. Both facts come from the server
// (has_dead_park, holder_live); `=== false` keeps unknown liveness inert.
// Deliberately not called "stranded": the Scheduler tab's stranded-parks
// banner counts dead parks only, and this set adds departed holders.
//
// THE SINGLE SOURCE of "is this lock stuck?" for both axes: it rescues a
// column, and it is what lifts a parked row ahead of the rest.
function lockIsAbandoned(module) {
  return !!module.has_dead_park
    || (!!module.holder && module.holder_live === false && (module.contention || 0) > 0)
}

// The columns: a contended HEAD (a prefix of the contention > 1 modules in
// server order) plus a rescued TAIL (abandoned modules the head did not reach).
// A slot is reserved only when an abandoned module falls outside the head.
function selectHeatmapColumns(allModules) {
  const contended = allModules.filter(m => (m.contention || 0) > 1)
  const abandoned = allModules.filter(lockIsAbandoned)
  const rankOf = new Map(contended.map((m, i) => [m, i]))
  const rescuedPast = headLen => abandoned
    .filter(m => !(rankOf.has(m) && rankOf.get(m) < headLen))
    .slice(0, MAX_RESCUED_COLS)
  let headLen = Math.min(contended.length, MAX_HEATMAP_COLS)
  // Shrinking the head can push an abandoned module out of it, so the length
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

// Rows come in three tiers, then the cap. Every tier is a row that colours a
// kept column, so none of them can bring a blank row:
//
//   1. a parked row blocked on an abandoned lock, such as the dead owner the
//      stranded-parks banner points at. `shape_scheduler` appends the dead
//      owners' synthetic rows after every project's ordinary rows, so a
//      positional prefix would cut exactly these;
//   2. a row waiting on a rescued column, without which that column renders
//      blank. There are at most MAX_RESCUED_COLS such columns, each no more
//      contended than the head's least contended column;
//   3. every other row touching a kept column.
//
// Each tier keeps server order. A live park earns no priority, just as it
// earns no column.
function selectHeatmapRows(allRows, keptModules, rescued) {
  const keptByPath = heatmapModulesByPath(keptModules)
  const abandonedByPath = heatmapModulesByPath(keptModules.filter(lockIsAbandoned))
  const rescuedByPath = heatmapModulesByPath(rescued)
  const parkedOnAbandoned = []
  const waitingOnRescued = []
  const touching = []
  for (const row of allRows) {
    const parked = ((row.park_state || {}).modules || []).length > 0
    if (parked && rowTouchesAnyIndexed(row, abandonedByPath)) parkedOnAbandoned.push(row)
    else if (rowTouchesAnyIndexed(row, rescuedByPath)) waitingOnRescued.push(row)
    else if (rowTouchesAnyIndexed(row, keptByPath)) touching.push(row)
  }
  return parkedOnAbandoned.concat(waitingOnRescued, touching).slice(0, MAX_HEATMAP_ROWS)
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

// The truncation caption for a boundHeatmapAxes result, or null when nothing
// was cut. The copy lives beside the selection because each claim in it —
// 'most contended' for the head, 'then' for the tail's position — is a
// property of selectHeatmapColumns, and the two must change together.
function describeHeatmapCap(bounded) {
  if (!bounded.rowsTruncated && !bounded.modulesTruncated) return null
  const tasks = `Showing ${bounded.rows.length} of ${bounded.rowsTotal} tasks`
  const contended = bounded.modules.length - bounded.modulesRescued
  const rescued = bounded.modulesRescued
  const of = `of ${bounded.modulesTotal} modules`
  const abandoned = 'abandoned (dead park owner or departed holder)'
  if (rescued === 0) return `${tasks} and the ${contended} most contended ${of}.`
  if (contended === 0) return `${tasks} and ${rescued} ${of}, all ${abandoned}.`
  return `${tasks} and the ${contended} most contended ${of}, then ${rescued} ${abandoned}.`
}

const SCHEDULER_HEATMAP_BOUNDS_API = {
  boundHeatmapAxes,
  rowTouchesModule,
  describeHeatmapCap,
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
