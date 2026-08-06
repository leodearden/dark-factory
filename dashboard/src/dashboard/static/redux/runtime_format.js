// runtime_format.js — pure formatters for warm-lane runtime state, rendered by
// the Orchestrators tab (OrchTab in tabs.jsx) and the Tasks tab's TaskDetail /
// TasksTab (tab_tasks.jsx). Two related concerns live here:
//
//   1. The per-field offline-'—' degradation (rtCell/rtAge) — WHAT to render
//      when a runtime field is null.
//   2. The probe-status discriminator (rtProbe/rtProbeSummary, task 3517) —
//      WHY it is null. A dash alone cannot tell an operator whether the
//      orchestrator is down, the dashboard was too starved to ask within its
//      own probe budget, or no orchestrator is configured for that root at
//      all. Those three demand opposite responses, and collapsing them into
//      identical blank cells is what made the 2026-07-30 event get
//      misdiagnosed as an orchestrator outage.
//
// Load-bearing producer chain for (2): dashboard/data/task_runtime.py's
// _probe_one synthesizes TaskRuntimeSnapshot.offline_reason, which
// active_tasks.py's _probe_status maps into the row's `runtime_status` via
// _runtime_fields. The vocabulary below MUST match active_tasks.RuntimeStatus.
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

// ── Probe-status descriptors (task 3517) ──
// Keyed by the four DEGRADED members of active_tasks.RuntimeStatus ('ok' is
// deliberately absent — it renders nothing). `tone` names an existing badge
// class from styles.css, so this adds no CSS:
//   muted = expected, not a fault · warn = likely OUR fault · bad = their fault.
const PROBE_STATUS = {
  not_configured: {
    label: 'no runtime endpoint',
    hint: 'No orchestrator is configured for this project root, so its runtime '
      + 'state was never probed. Expected and permanent — not a fault.',
    tone: 'muted',
  },
  unreachable: {
    label: 'orchestrator unreachable',
    hint: 'The probe reached the network but the orchestrator refused, errored, '
      + 'or answered with a malformed payload. Go look at that orchestrator.',
    tone: 'bad',
  },
  deadline_exceeded: {
    label: 'probe timed out',
    hint: 'The dashboard\'s own probe deadline fired before an answer arrived. '
      + 'The orchestrator may be perfectly healthy — a starved dashboard event '
      + 'loop produces this too, so check the dashboard before restarting anything.',
    tone: 'warn',
  },
  unknown: {
    label: 'runtime state unknown',
    hint: 'The snapshot reported offline without saying why, so we genuinely do '
      + 'not know which side failed. Not a diagnosis — just an honest gap.',
    tone: 'warn',
  },
};

// ── Per-row probe descriptor: 'ok' -> null, anything else -> a descriptor ──
// Only the exact string 'ok' renders nothing. Every other value — including
// null/undefined/garbage from an older or drifting payload — falls back to the
// 'unknown' descriptor rather than throwing or silently vanishing into a blank
// cell, which is the very failure mode this function exists to remove.
// hasOwnProperty, not a bare lookup: `PROBE_STATUS['constructor']` would
// otherwise resolve up the prototype chain to a truthy non-descriptor and be
// returned as if it were one.
function rtProbe(status) {
  if (status === 'ok') return null;
  return Object.prototype.hasOwnProperty.call(PROBE_STATUS, status)
    ? PROBE_STATUS[status]
    : PROBE_STATUS.unknown;
}

// ── Tasks-tab banner aggregation over ACTIVE_TASKS rows ──
// Pure over an array of task rows ({project, runtime_status}). Derived
// frontend-side rather than as a new top-level payload key: the per-project
// fact is already fully recoverable from the rows, so a new key would carry
// zero extra information while churning every exact-set payload assertion.
// Returns null when nothing is degraded, else
// {byStatus, probedCount, selfInflicted, text}.
function rtProbeSummary(rows) {
  // Dedupe to one status per project — a project with 40 rows must not
  // outvote a project with 1.
  const byProject = new Map();
  for (const r of rows || []) {
    // A row predating `runtime_status` (older cached payload) is treated as
    // healthy: absence of evidence is not evidence of a failed probe.
    const status = (r && r.runtime_status) || 'ok';
    if (status === 'ok') continue;
    if (!byProject.has(r.project)) byProject.set(r.project, status);
  }
  if (byProject.size === 0) return null;

  // Null-prototype so an unexpected status string cannot collide with an
  // inherited Object member (same reason as rtProbe's hasOwnProperty guard).
  const byStatus = Object.create(null);
  for (const [project, status] of byProject) {
    (byStatus[status] = byStatus[status] || []).push(project);
  }

  // A project is "probed" iff something was actually attempted against it —
  // not_configured projects were never probed, so counting them would make the
  // heuristic fire on a deployment where only one orchestrator is configured.
  const probed = [...byProject.values()].filter((s) => s !== 'not_configured');
  const probedCount = probed.length;
  // >= 2 matches task_runtime.fetch_task_runtime's aggregate WARNING exactly,
  // so the log and this banner can never disagree. With ONE probed project the
  // all-at-once pattern is degenerate — it is equally consistent with that one
  // orchestrator being down, so claiming the dashboard is at fault would be an
  // unfounded diagnosis in the other direction.
  const selfInflicted = probedCount >= 2 && probed.every((s) => s === 'deadline_exceeded');

  const describe = (status) => `${rtProbe(status).label}: ${byStatus[status].join(', ')}`;
  const parts = Object.keys(byStatus).map(describe).join(' · ');
  const text = selfInflicted
    ? `The dashboard could not complete its own runtime probes for all ${probedCount} `
      + 'probed projects. The orchestrators may be healthy — a real outage is '
      + `per-project, so check the dashboard first. (${parts})`
    : `Runtime state unavailable for ${byProject.size} project(s) — ${parts}`;

  return { byStatus, probedCount, selfInflicted, text };
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced by
// dashboard/tests/js/classic_script_scope.test.mjs. A collision here would
// leave window.DF_RUNTIME_FMT undefined and break tabs.jsx's / tab_tasks.jsx's
// top-level destructure of it.
const RUNTIME_FORMAT_API = { rtCell, rtAge, rtProbe, rtProbeSummary };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = RUNTIME_FORMAT_API;
}
if (typeof window !== 'undefined') {
  window.DF_RUNTIME_FMT = RUNTIME_FORMAT_API;
}
