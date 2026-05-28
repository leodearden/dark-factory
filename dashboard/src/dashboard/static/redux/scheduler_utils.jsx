/* scheduler_utils.jsx — shared helpers for the scheduler heatmap/drawer/tab,
   and the lock-display join used by tabs.jsx and tab_tasks.jsx.
   Loaded early in index.html (before tabs.jsx and tab_tasks.jsx) so all
   consumers can destructure from the global.

   Exports: window.DF_SCHED_UTILS = { fmtAge, totalEvents, avgWaitSeconds, buildSchedLockInfo }
*/

// Format seconds into a human-readable elapsed string.
function fmtAge(seconds) {
  if (seconds == null || isNaN(seconds)) return '—';
  if (seconds < 60)   return `${seconds}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h`;
}

// Given a sparkline {labels, values}, return the total event count.
function totalEvents(spark) {
  if (!spark || !spark.values) return 0;
  return spark.values.reduce((a, b) => a + b, 0);
}

// Approximate average wait as the mean skip interval (seconds).
// Only meaningful when totalEvents(spark) >= 10 (enough signal to estimate).
// `spark.labels` are ISO timestamps of each bucket start; bucket width is a
// hardcoded 300s — must stay in sync with scheduler.py's _skip_event_sparkline
// default. This is the mean inter-skip interval, not a true percentile.
function avgWaitSeconds(spark) {
  const total = totalEvents(spark);
  if (total < 10) return null;
  const n = (spark.labels || []).length;
  if (n === 0) return null;
  // Window is n buckets of 300s each; mean interval between skips.
  return Math.round((n * 300) / total);
}

// Build the scheduler-lock lookup info for a given task.
// Returns { rawTaskId, lockSet, moduleByPath } where moduleByPath is a Map<path, module|null>.
// Pass the scheduler data object (D.SCHEDULER or DF.SCHEDULER or DF_T.SCHEDULER) as schedulerData.
// O(L) per call (L = lock_set size) — use the returned Map for O(1) per-chip lookups.
function buildSchedLockInfo(task, schedulerData) {
  const rawTaskId = String(task.id).split('/T-').pop();
  const sched = schedulerData || {};
  const schedRows = sched.rows || [];
  const schedModules = sched.modules || [];
  const myRow = schedRows.find(r => r.project === task.project && r.task_id === rawTaskId);
  const lockSet = (myRow && myRow.lock_set) || [];
  const moduleByPath = new Map(
    lockSet.map(path => {
      const m = schedModules.find(mm => mm.project === task.project && mm.path === path);
      return [path, m || null];
    })
  );
  return { rawTaskId, lockSet, moduleByPath };
}

window.DF_SCHED_UTILS = { fmtAge, totalEvents, avgWaitSeconds, buildSchedLockInfo };
