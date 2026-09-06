// runtime_format.js — pure formatters for warm-lane runtime state. Two
// related concerns live here, with DIFFERENT consumer coverage:
//
//   1. The per-field offline-'—' degradation (rtCell/rtAge) — WHAT to render
//      when a runtime field is null. Consumed by BOTH the Orchestrators tab
//      (OrchTab in tabs.jsx) and the Tasks tab's TaskDetail / TasksTab
//      (tab_tasks.jsx).
//   2. The probe-status discriminator (rtProbe/rtProbeSummary, task 3517) —
//      WHY it is null. A dash alone cannot tell an operator whether the
//      orchestrator is down, the dashboard was too starved to ask within its
//      own probe budget, or no orchestrator is configured for that root at
//      all. Those three demand opposite responses, and collapsing them into
//      identical blank cells is what made the 2026-07-30 event get
//      misdiagnosed as an orchestrator outage.
//
//      KNOWN GAP: (2) is wired into the Tasks tab ONLY. OrchTab still
//      destructures just {rtCell, rtAge} and renders loops/attempts/lane/
//      phase/lane_state as bare em-dashes with no probe explanation, so the
//      2026-07-30 ambiguity survives one tab over — on arguably the tab an
//      operator reaches for FIRST during a suspected orchestrator outage.
//      Deliberately not fixed here: tabs.jsx's test surface
//      (dashboard/tests/test_tab_orchestrators.py) is outside task 3517's
//      locks, so the change could not be landed with tests. Do not read the
//      list in (1) as claiming OrchTab coverage for (2).
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

// Tone severity ordering, for picking the WORST tone in a set. Kept next to
// PROBE_STATUS so a new member cannot be added without a rank.
const TONE_RANK = { muted: 0, warn: 1, bad: 2 };

// ── The single missing-value policy, shared by BOTH helpers below ──
// They must agree, or the UI contradicts itself: every TaskDetail showing a
// 'runtime state unknown' warn badge while the aggregate banner reports
// nothing degraded at all. ABSENT (null/undefined) means the payload predates
// `runtime_status` — an older cached response — and is read as 'ok', matching
// the producing side's own default (active_tasks._build_task_row does
// `rt.get('runtime_status', 'ok')`). The 'unknown' descriptor stays reserved
// for the two cases that are genuinely informative: a value that IS present
// but is not a vocabulary member (real drift), and the server reporting
// offline without saying why.
function normStatus(status) {
  return status == null ? 'ok' : status;
}

// ── Per-row probe descriptor: 'ok' -> null, anything else -> a descriptor ──
// A present-but-unrecognized value falls back to the 'unknown' descriptor
// rather than throwing or silently vanishing into a blank cell, which is the
// very failure mode this function exists to remove.
// hasOwnProperty, not a bare lookup: `PROBE_STATUS['constructor']` would
// otherwise resolve up the prototype chain to a truthy non-descriptor and be
// returned as if it were one.
function rtProbe(status) {
  const s = normStatus(status);
  if (s === 'ok') return null;
  return Object.prototype.hasOwnProperty.call(PROBE_STATUS, s)
    ? PROBE_STATUS[s]
    : PROBE_STATUS.unknown;
}

// ── Tasks-tab banner aggregation over ACTIVE_TASKS rows ──
// Pure over an array of task rows ({project, runtime_status}). Derived
// frontend-side rather than as a new top-level payload key: the per-project
// fact is already fully recoverable from the rows, so a new key would carry
// zero extra information while churning every exact-set payload assertion.
// Returns null when there is nothing to ALARM about, else
// {byStatus, probedCount, degradedCount, selfInflicted, tone, text}.
//
// 'not_configured' is deliberately NOT an input to this banner. It is both
// expected and PERMANENT — dashboard/config.py's _discover_escalation_urls
// explicitly anticipates "a legitimately non-orchestrator root" — so any
// deployment tracking such a root would carry a never-clearing alert on the
// Tasks tab, which is exactly how operators learn to ignore the banner that
// carries the real diagnosis. Those rows are not silently dropped: rtProbe
// still renders their (muted) per-row badge in TaskDetail, which is the right
// altitude for a permanent, per-project, non-fault fact.
function rtProbeSummary(rows) {
  // ONE map, not two. Dedupe to one status per project — a project with 40
  // rows must not outvote a project with 1 — and keep the HEALTHY probed
  // projects in it, because they are the denominator (see below). Two
  // independently-filtered maps could disagree if a project ever contributed
  // both an 'ok' row and a degraded row, since first-seen would resolve
  // differently in each; one map makes that state unrepresentable.
  const probedProjects = new Map();
  for (const r of rows || []) {
    const status = normStatus(r && r.runtime_status);
    if (status === 'not_configured') continue;
    if (!probedProjects.has(r.project)) probedProjects.set(r.project, status);
  }

  // Null-prototype so an unexpected status string cannot collide with an
  // inherited Object member (same reason as rtProbe's hasOwnProperty guard).
  // byStatus stays degraded-only: it is the banner's LISTING, and healthy
  // projects are not something to alarm about.
  const byStatus = Object.create(null);
  const degraded = [];
  for (const [project, status] of probedProjects) {
    if (status === 'ok') continue;
    degraded.push(status);
    (byStatus[status] = byStatus[status] || []).push(project);
  }
  if (degraded.length === 0) return null;

  // probedCount MEANS what its name says: every project the dashboard
  // actually probed, HEALTHY ONES INCLUDED. Omitting them is what let a
  // PARTIAL timeout read as dashboard-wide starvation — with only the
  // degraded projects in the denominator, "all probed projects timed out" is
  // vacuously true of any pure-timeout subset, so two slow orchestrators
  // alongside three healthy ones reported the dashboard as the fault. A
  // project that answered is positive proof the dashboard was not too starved
  // to ask, which is precisely the claim `selfInflicted` makes.
  //
  // never-probed (not_configured) projects are still excluded, so the
  // heuristic cannot fire on a deployment where only one orchestrator is even
  // configured.
  const probedCount = probedProjects.size;
  const degradedCount = degraded.length;
  // The threshold (>= 2) and the all-probed-must-be-deadline_exceeded rule are
  // now identical to task_runtime.fetch_task_runtime's aggregate WARNING. The
  // two can still differ, but only in ONE way: this denominator is derived
  // from ACTIVE_TASKS ROWS, so a probed project with zero task rows is
  // invisible here while Python counts it in `labels`. (Do not restate this as
  // "can never disagree" — that unqualified claim was asserted here once and
  // was false.) With ONE probed project the all-at-once pattern is degenerate
  // — equally consistent with that one orchestrator being down — so claiming
  // the dashboard is at fault would be an unfounded diagnosis in reverse.
  const selfInflicted = probedCount >= 2
    && degradedCount === probedCount
    && degraded.every((s) => s === 'deadline_exceeded');

  // Worst tone present, so the caller can colour the banner by what is
  // actually wrong. Deriving the accent from `selfInflicted` instead would
  // paint a lone timed-out project — a 'warn', quite possibly our own starved
  // loop — in the same alarm colour as a confirmed orchestrator outage.
  const tone = Object.keys(byStatus).reduce((worst, s) => {
    const t = rtProbe(s).tone;
    return TONE_RANK[t] > TONE_RANK[worst] ? t : worst;
  }, 'muted');

  const describe = (status) => `${rtProbe(status).label}: ${byStatus[status].join(', ')}`;
  const parts = Object.keys(byStatus).map(describe).join(' · ');
  // The self-inflicted branch reports the DENOMINATOR ("all N probed"), which
  // is the whole point of that claim. The other branch reports the DEGRADED
  // count, so its number and its listing describe the same set — using
  // probedCount there would claim 5 projects while naming 2.
  const text = selfInflicted
    ? `The dashboard could not complete its own runtime probes for all ${probedCount} `
      + 'probed projects. The orchestrators may be healthy — a real outage is '
      + `per-project, so check the dashboard first. (${parts})`
    : `Runtime state unavailable for ${degradedCount} project(s) — ${parts}`;

  return { byStatus, probedCount, degradedCount, selfInflicted, tone, text };
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
