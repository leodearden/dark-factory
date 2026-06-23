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

// Build a Map<fullPath, label> where each label is the shortest trailing
// path suffix that uniquely identifies the path within the given set.
//
// Algorithm:
//   1. Dedupe the input (same path appearing N times counts once).
//   2. For each unique path p, compute k = 1 + max over every OTHER unique
//      path q of sharedTrailingSegments(p, q).  (k=1 when the set has 0 or
//      1 members, or when no trailing segments are shared with any peer.)
//   3. label(p) = last min(k, segCount(p)) segments joined with '/'.
//   4. Return a Map keyed by the original full path string (including dupes).
//
// Deduping is essential: a path repeated across projects is NOT forced to its
// full length (the only "other" copy is itself, contributing 0 shared segments
// rather than a full-length collision).
function disambiguateLabels(paths) {
  const unique = [...new Set(paths)];
  const segs = {};
  for (const p of unique) segs[p] = p.split('/');

  function sharedTrailing(aSegs, bSegs) {
    let count = 0;
    const la = aSegs.length, lb = bSegs.length;
    while (count < la && count < lb && aSegs[la - 1 - count] === bSegs[lb - 1 - count]) {
      count++;
    }
    return count;
  }

  const labelMap = new Map();
  for (const p of unique) {
    const ps = segs[p];
    let maxShared = 0;
    for (const q of unique) {
      if (q === p) continue;
      const shared = sharedTrailing(ps, segs[q]);
      if (shared > maxShared) maxShared = shared;
    }
    const k = 1 + maxShared;
    const label = ps.slice(-Math.min(k, ps.length)).join('/');
    labelMap.set(p, label);
  }
  return labelMap;
}

// labelFor(paths) is the stable-contract companion to disambiguateLabels.
// Unlike calling disambiguateLabels directly, labelFor is safe to destructure
// and invoke unconditionally at call sites that already hard-depend on
// window.DF_SCHED_UTILS (scheduler_drawer.jsx, tab_scheduler.jsx): because
// scheduler_utils.jsx loads first, labelFor is always present.
// The function is intentionally a thin wrapper so node-vm tests for
// disambiguateLabels continue to cover the core logic without change.
function labelFor(paths) {
  return disambiguateLabels(paths || []);
}

// Compute the visual state for a module-lock chip.
// Encodes the holder > parked > free precedence so both LockChip (tabs.jsx)
// and the inline Task-detail chip (tab_tasks.jsx) can call this pure helper
// instead of duplicating the if/else-if chain.
// Returns { cls, hint, ownerLabel } where ownerLabel is null when the chip has
// no owner to display (lock-free or lock-mine without a counterpart label).
function lockChipState({ holder, isMine, parkedBy, parkedOwnerLive }) {
  if (holder) {
    if (isMine) return { cls: 'lock-mine', hint: 'held by this task', ownerLabel: null };
    return { cls: 'lock-taken', hint: `held by T-${holder}`, ownerLabel: `T-${holder}` };
  }
  if (parkedBy) {
    const label = `T-${parkedBy}${parkedOwnerLive === false ? ' ⚠' : ''}`;
    return { cls: 'lock-parked', hint: `parked by T-${parkedBy}`, ownerLabel: label };
  }
  return { cls: 'lock-free', hint: 'available', ownerLabel: null };
}

window.DF_SCHED_UTILS = { fmtAge, totalEvents, avgWaitSeconds, buildSchedLockInfo, disambiguateLabels, labelFor, lockChipState };
