/* scheduler_utils.jsx — shared helpers for the scheduler heatmap/drawer/tab.
   Loaded before scheduler_heatmap/drawer/tab in index.html so those consumers
   can destructure from the global (removes the verbatim duplication of these
   three helpers across the drawer and the tab).

   Exports: window.DF_SCHED_UTILS = { fmtAge, totalEvents, avgWaitSeconds }
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

window.DF_SCHED_UTILS = { fmtAge, totalEvents, avgWaitSeconds };
