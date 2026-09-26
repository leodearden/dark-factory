/* Per-endpoint staleness decision for the dashboard's tab bodies (task 4884,
 * #4791).
 *
 * THE FAILURE THIS CLOSES. On 2026-08-27 the dashboard served a
 * fully-rendered UI for 19.8h whose numbers had stopped advancing. That is
 * by design at the fetch layer — data.js::refreshOne deliberately keeps the
 * prior values "so the UI does not blank out" — but nothing on screen said
 * the values were hours old, so the operator had no way to tell a quiet
 * factory from a wedged fetch. This module holds the decision of WHEN to say
 * so and WHAT to say; app.jsx renders whatever it returns.
 *
 * Pure by construction: no window/document access at load, every input
 * optional. It is loaded as a CLASSIC script (see index.html) whose top-level
 * bindings share one global lexical scope with the other /static/redux/*.js
 * files, so every top-level name here must stay unique across them —
 * dashboard/tests/js/classic_script_scope.test.mjs enforces that by loading
 * them all into one node:vm context.
 */

// How many consecutive failed polls before an endpoint is called stale.
//
// data.js backs off at BACKOFF_BASE_MS (3000) * 2^(failures-1), so three
// failures is ~3s + 6s + 12s ≈ 21s of sustained failure: comfortably past the
// isolated 2.0s ReadTimeouts the incident journal shows recovering on the next
// tick (#4791 acceptance 2), and far short of the BACKOFF_MAX_MS (60s) plateau
// a genuinely wedged endpoint then sits at indefinitely.
const STALE_FAILURE_THRESHOLD = 3

// Tab id -> the endpoint PATHS whose payload that tab renders.
//
// Derived mechanically from what each tab actually reads off window.DF_DATA,
// mapped through data.js::endpointsFor's key lists: e.g. tab_overview.jsx
// reads ORCHESTRATORS/ORCHESTRATORS_SPARK (-> /orchestrators), MEMORY_STATUS
// (-> /memory), MEMORY_TIMESERIES (-> /memory-graphs), and so on. Paths are
// QUERY-STRIPPED, matching data.js::pollKey — the four ?window= endpoints key
// their flow-control state the same way, and a URL-keyed map would miss every
// one of them after the first chip change.
//
// ONE MAP, ONE RENDER SITE. app.jsx renders staleNoticesForTab(...) once for
// the active tab rather than thirteen per-tab banners, so adding a tab is a
// one-line edit here. endpoint_staleness.test.mjs machine-checks BOTH
// directions — every path is a real endpointsFor() key, and every app.jsx tab
// id has an entry — because a map that silently stops matching a renamed
// endpoint is a check that has quietly stopped checking. `toolbarConfig` in
// app.jsx is the sibling per-tab map; keep the two tab-id lists in step.
const TAB_ENDPOINTS = {
  overview: [
    '/api/v2/dashboard/orchestrators',
    '/api/v2/dashboard/memory',
    '/api/v2/dashboard/memory-graphs',
    '/api/v2/dashboard/recon',
    '/api/v2/dashboard/scheduler',
    '/api/v2/dashboard/costs',
    '/api/v2/dashboard/burndown',
  ],
  orch: [
    '/api/v2/dashboard/orchestrators',
    '/api/v2/dashboard/tasks',
    '/api/v2/dashboard/burndown',
  ],
  tasks: [
    '/api/v2/dashboard/tasks',
    '/api/v2/dashboard/orchestrators',
    '/api/v2/dashboard/scheduler',
  ],
  scheduler: ['/api/v2/dashboard/scheduler'],
  curator: ['/api/v2/dashboard/curator'],
  perf: ['/api/v2/dashboard/performance'],
  memory: [
    '/api/v2/dashboard/memory',
    '/api/v2/dashboard/memory-graphs',
    '/api/v2/dashboard/memory-evals',
  ],
  recon: ['/api/v2/dashboard/recon'],
  merge: ['/api/v2/dashboard/merge-queue'],
  cost: ['/api/v2/dashboard/costs'],
  burn: [
    '/api/v2/dashboard/burndown',
    '/api/v2/dashboard/orchestrators',
  ],
  esc: [
    '/api/v2/dashboard/escalations',
    '/api/v2/dashboard/escalation-analytics',
  ],
  'esc-analytics': ['/api/v2/dashboard/escalation-analytics'],
}

/**
 * The `{failures, lastSuccessAt}` record data.js published for `path`, or null.
 *
 * Returns null rather than a zero-filled default for a missing entry: before
 * the first poll resolves, `DF_DATA.__stale` is `{}` for every endpoint, and a
 * fabricated `{failures: 0}` there is indistinguishable from a real healthy
 * reading. Absent means UNKNOWN, and unknown produces no claim.
 */
function staleEntryFor(stale, path) {
  if (!stale || typeof stale !== 'object') return null
  const entry = stale[path]
  if (!entry || typeof entry !== 'object') return null
  return entry
}

/**
 * A human age for `ms`, e.g. '45s', '6m', '19h 48m'.
 *
 * Hours are rendered as hours rather than as a four-digit minute count: the
 * incident this indicator exists for ran 19.8h, and an operator should not
 * have to divide 1188 by 60 to notice that.
 */
function formatAge(ms) {
  const n = Number(ms)
  if (!Number.isFinite(n) || n < 0) return 'an unknown time'
  const secs = Math.floor(n / 1000)
  if (secs < 60) return secs + 's'
  const mins = Math.floor(secs / 60)
  if (mins < 60) return mins + 'm'
  const hours = Math.floor(mins / 60)
  const rest = mins % 60
  return rest === 0 ? hours + 'h' : hours + 'h ' + rest + 'm'
}

function attemptPhrase(failures) {
  return failures + ' consecutive attempt' + (failures === 1 ? '' : 's')
}

function noticeText(path, entry, now, failures) {
  const last = Number(entry.lastSuccessAt)
  // A missing or zero lastSuccessAt means this endpoint has never delivered —
  // NOT that it delivered a moment ago. Rendering an age of zero from a
  // missing timestamp is the `_minutes_since` mistake active_tasks.py already
  // documents, and it would fabricate reassurance during exactly the failure
  // this indicator exists to surface.
  if (!Number.isFinite(last) || last <= 0) {
    return path + ' has never delivered data — ' + attemptPhrase(failures) +
      ' failed since this page loaded'
  }
  const age = formatAge(Math.max(0, Number(now) - last))
  return path + ' is stale — last updated ' + age + ' ago (' +
    attemptPhrase(failures) + ' failed since)'
}

/**
 * Which staleness notices the given tab should render.
 *
 * Returns `[{kind: 'stale', path, text}]`, one per endpoint of that tab that
 * has failed at least STALE_FAILURE_THRESHOLD consecutive polls — empty when
 * everything the tab renders is current, when the tab is unknown, or when
 * nothing has been polled yet.
 *
 * PER-ENDPOINT, NOT GLOBAL. The 19.8h wedge hit the tasks fan-out while other
 * endpoints stayed current; a single page-level banner would have been wrong
 * about both halves. Notices are ADDITIVE — they render above `renderTab()`,
 * never in place of it, so the last-good payload stays on screen, marked.
 *
 * Every input is optional: this runs on DF_DATA's pre-fetch defaults during
 * the very first render, so a missing key must produce no notice rather than
 * a TypeError that takes the whole page down.
 */
function staleNoticesForTab(input) {
  const s = input || {}
  const paths = TAB_ENDPOINTS[s.tab]
  if (!Array.isArray(paths)) return []

  // `now` is Date.now() from app.jsx. Falling back rather than trusting a
  // non-finite value keeps 'NaNs ago' out of the operator's only staleness
  // signal.
  const nowMs = Number.isFinite(Number(s.now)) ? Number(s.now) : Date.now()

  const notices = []
  for (const path of paths) {
    const entry = staleEntryFor(s.stale, path)
    if (!entry) continue
    const failures = Number(entry.failures)
    if (!Number.isFinite(failures) || failures < STALE_FAILURE_THRESHOLD) continue
    notices.push({
      kind: 'stale',
      path,
      text: noticeText(path, entry, nowMs, failures),
    })
  }
  return notices
}

const ENDPOINT_STALENESS_API = {
  STALE_FAILURE_THRESHOLD,
  TAB_ENDPOINTS,
  staleEntryFor,
  formatAge,
  staleNoticesForTab,
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = ENDPOINT_STALENESS_API
}
if (typeof window !== 'undefined') {
  window.DF_ENDPOINT_STALENESS = ENDPOINT_STALENESS_API
}
