/* What the dashboard polls: the surface -> endpoint maps (task 5825; given
 * their own module by task 6304).
 *
 * The browser polls CHROME_ENDPOINTS ∪ TAB_ENDPOINTS[activeTab]
 * (data.js::pollSetFor), so a path missing from these maps is not merely
 * unmonitored: the surface that reads it silently stops refreshing. app.jsx
 * also hands TAB_ENDPOINTS[tab] to endpoint_staleness.js's notice functions,
 * so the same entry decides which endpoints the open tab reports as stale or
 * still loading.
 *
 * Pure by construction: no window/document access at load. It is loaded as a
 * CLASSIC script (see index.html) whose top-level bindings share one global
 * lexical scope with the other /static/redux/*.js files, so every top-level
 * name here must stay unique across them —
 * dashboard/tests/js/classic_script_scope.test.mjs enforces that by loading
 * them all into one node:vm context. Consumers therefore destructure these
 * maps under an alias.
 */

// Tab id -> the endpoint PATHS whose payload that tab renders, and therefore
// the endpoints data.js polls while that tab is open (see the header).
//
// Derived mechanically from what each tab actually reads off window.DF_DATA,
// mapped through data.js::endpointsFor's key lists: e.g. tab_overview.jsx
// reads ORCHESTRATORS/ORCHESTRATORS_SPARK (-> /orchestrators), MEMORY_STATUS
// (-> /memory), MEMORY_OPS (-> /memory-graphs), and so on. Paths are
// QUERY-STRIPPED, matching data.js::pollKey — the four ?window= endpoints key
// their flow-control state the same way, and a URL-keyed map would miss every
// one of them after the first chip change.
//
// ONE MAP, ONE RENDER SITE. app.jsx renders the notices for TAB_ENDPOINTS[tab]
// once, for the active tab, rather than one banner per tab, so adding a
// tab is a one-line edit here. poll_scope.test.mjs machine-checks BOTH
// directions — every path is a real endpointsFor() key, and every app.jsx tab
// id has an entry — because a map that silently stops matching a renamed
// endpoint is a check that has quietly stopped checking. test_app_poll_scope.py
// walks each tab's components and fails when a path the tab reads is missing
// from its entry. `toolbarConfig` in app.jsx is the sibling per-tab map; keep
// the two tab-id lists in step.
//
// LIST /tasks FOR A TAB ONLY IF IT RENDERS TASK ROWS. Listing it polls the full
// multi-MB render on that tab; the census every tab's chrome reads is already
// polled through CHROME_ENDPOINTS, as /tasks?projection=census.
//
// Frozen, map and entries alike: data.js, app.jsx and the notice functions all
// read this one declaration, so no consumer may edit it under the others.
const TAB_ENDPOINTS = Object.freeze({
  overview: Object.freeze([
    '/api/v2/dashboard/orchestrators',
    '/api/v2/dashboard/memory',
    '/api/v2/dashboard/memory-graphs',
    '/api/v2/dashboard/recon',
    '/api/v2/dashboard/scheduler',
    '/api/v2/dashboard/costs',
    '/api/v2/dashboard/burndown',
    '/api/v2/dashboard/merge-queue',
  ]),
  orch: Object.freeze([
    '/api/v2/dashboard/orchestrators',
    '/api/v2/dashboard/tasks',
    '/api/v2/dashboard/burndown',
    '/api/v2/dashboard/scheduler',
  ]),
  tasks: Object.freeze([
    '/api/v2/dashboard/tasks',
    '/api/v2/dashboard/orchestrators',
    '/api/v2/dashboard/scheduler',
    '/api/v2/dashboard/invariants',
    '/api/v2/dashboard/prds',
  ]),
  scheduler: Object.freeze(['/api/v2/dashboard/scheduler']),
  curator: Object.freeze(['/api/v2/dashboard/curator']),
  perf: Object.freeze(['/api/v2/dashboard/performance']),
  memory: Object.freeze([
    '/api/v2/dashboard/memory',
    '/api/v2/dashboard/memory-graphs',
    '/api/v2/dashboard/memory-evals',
  ]),
  recon: Object.freeze(['/api/v2/dashboard/recon']),
  merge: Object.freeze(['/api/v2/dashboard/merge-queue']),
  cost: Object.freeze(['/api/v2/dashboard/costs']),
  burn: Object.freeze([
    '/api/v2/dashboard/burndown',
    '/api/v2/dashboard/orchestrators',
  ]),
  esc: Object.freeze([
    '/api/v2/dashboard/escalations',
    '/api/v2/dashboard/escalation-analytics',
  ]),
  'esc-analytics': Object.freeze(['/api/v2/dashboard/escalation-analytics']),
  // Its own /quality read is on demand (data.js::ON_DEMAND_KEYS.quality);
  // this is its project selector's PROJECTS.
  quality: Object.freeze(['/api/v2/dashboard/orchestrators']),
})

// The endpoint PATHS the always-mounted chrome reads, whatever tab is open.
// Polled on every tab, beside the open tab's own TAB_ENDPOINTS entry.
//
// EXACTLY the chrome's read set, no more: test_app_poll_scope.py derives what
// app.jsx's rail/topbar and shell.jsx's toolbar actually read and fails if it
// differs from this list in either direction — a missing path freezes a badge
// on every other tab, and a dead one polls an endpoint for nothing.
const CHROME_ENDPOINTS = Object.freeze([
  '/api/v2/dashboard/orchestrators', // rail orch badge, topbar orch counts, Toolbar PROJECTS
  '/api/v2/dashboard/tasks',         // rail and topbar census (census only)
  '/api/v2/dashboard/recon',         // rail recon badge, Toolbar AGENTS
  '/api/v2/dashboard/merge-queue',   // rail merge badge
  '/api/v2/dashboard/escalations',   // rail esc badge
  '/api/v2/dashboard/memory',        // topbar queue
  '/api/v2/dashboard/costs',         // topbar spend
])

const POLL_SCOPE_API = {
  TAB_ENDPOINTS,
  CHROME_ENDPOINTS,
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = POLL_SCOPE_API
}
if (typeof window !== 'undefined') {
  window.DF_POLL_SCOPE = POLL_SCOPE_API
}
