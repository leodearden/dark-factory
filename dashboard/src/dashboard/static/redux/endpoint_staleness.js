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
 * ITS SECOND ROLE (task 5825): the two surface -> endpoint maps declared here,
 * TAB_ENDPOINTS and CHROME_ENDPOINTS, also decide what data.js FETCHES. The
 * browser polls CHROME_ENDPOINTS ∪ TAB_ENDPOINTS[activeTab] (data.js::
 * pollSetFor), so a path missing from these maps is not merely unmonitored:
 * the surface that reads it silently stops refreshing.
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
// ONE MAP, ONE RENDER SITE. app.jsx renders staleNoticesForTab(...) once for
// the active tab rather than thirteen per-tab banners, so adding a tab is a
// one-line edit here. endpoint_staleness.test.mjs machine-checks BOTH
// directions — every path is a real endpointsFor() key, and every app.jsx tab
// id has an entry — because a map that silently stops matching a renamed
// endpoint is a check that has quietly stopped checking. `toolbarConfig` in
// app.jsx is the sibling per-tab map; keep the two tab-id lists in step.
//
// LIST /tasks FOR A TAB ONLY IF IT RENDERS TASK ROWS. Listing it polls the full
// multi-MB render on that tab; the census every tab's chrome reads is already
// polled through CHROME_ENDPOINTS, as /tasks?projection=census.
const TAB_ENDPOINTS = {
  overview: [
    '/api/v2/dashboard/orchestrators',
    '/api/v2/dashboard/memory',
    '/api/v2/dashboard/memory-graphs',
    '/api/v2/dashboard/recon',
    '/api/v2/dashboard/scheduler',
    '/api/v2/dashboard/costs',
    '/api/v2/dashboard/burndown',
    '/api/v2/dashboard/merge-queue',
  ],
  orch: [
    '/api/v2/dashboard/orchestrators',
    '/api/v2/dashboard/tasks',
    '/api/v2/dashboard/burndown',
    '/api/v2/dashboard/scheduler',
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
 * A human age for `ms`, e.g. '45s', '6m', '19h 48m', '1d 3h'.
 *
 * Hours are rendered as hours rather than as a four-digit minute count: the
 * incident this indicator exists for ran 19.8h, and an operator should not
 * have to divide 1188 by 60 to notice that. Days are rendered as days for the
 * same reason: an idle project's cards are weeks old, and '480h' is 20 days.
 */
function formatAge(ms) {
  const n = Number(ms)
  if (!Number.isFinite(n) || n < 0) return 'an unknown time'
  const secs = Math.floor(n / 1000)
  if (secs < 60) return secs + 's'
  const mins = Math.floor(secs / 60)
  if (mins < 60) return mins + 'm'
  const hours = Math.floor(mins / 60)
  if (hours < 24) {
    const restMins = mins % 60
    return restMins === 0 ? hours + 'h' : hours + 'h ' + restMins + 'm'
  }
  const days = Math.floor(hours / 24)
  const restHours = hours % 24
  return restHours === 0 ? days + 'd' : days + 'd ' + restHours + 'h'
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

/**
 * The one "still loading" notice the given tab should render, or null.
 *
 * Returns `{kind: 'loading', paths, text}` naming every endpoint of that tab
 * that has not delivered a response since this page loaded — read off
 * `receipt`, data.js's success-only `DF_DATA.__receipt` map. A tab opened for
 * the first time asks for its endpoints at once, and until they answer its
 * body is the pre-fetch seed, which looks exactly like a measured empty
 * payload.
 *
 * ONE NOTICE PER TAB, NOT PER ENDPOINT. Every path is pending at page load, so
 * a per-endpoint banner would stack up to one per path (eight on Overview)
 * for a single transient fact. A stale notice stays per-endpoint: there each
 * endpoint's age and failure count is its own fact.
 *
 * A path already failing STALE_FAILURE_THRESHOLD times is left out:
 * staleNoticesForTab names that one, and two banners for one fact is noise.
 *
 * ADDITIVE, like the stale notices, and every input optional: a missing
 * `receipt` produces no notice rather than a claim about every path.
 */
function loadingNoticeForTab(input) {
  const s = input || {}
  const paths = TAB_ENDPOINTS[s.tab]
  if (!Array.isArray(paths)) return null
  if (!s.receipt || typeof s.receipt !== 'object') return null

  const pending = paths.filter(path => {
    if (s.receipt[path]) return false
    const entry = staleEntryFor(s.stale, path)
    return !(entry && Number(entry.failures) >= STALE_FAILURE_THRESHOLD)
  })
  if (pending.length === 0) return null
  return {
    kind: 'loading',
    paths: pending,
    text: pending.join(', ') + (pending.length === 1 ? ' has' : ' have') +
      ' not delivered data yet — loading',
  }
}

const ENDPOINT_STALENESS_API = {
  STALE_FAILURE_THRESHOLD,
  TAB_ENDPOINTS,
  CHROME_ENDPOINTS,
  staleEntryFor,
  formatAge,
  staleNoticesForTab,
  loadingNoticeForTab,
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = ENDPOINT_STALENESS_API
}
if (typeof window !== 'undefined') {
  window.DF_ENDPOINT_STALENESS = ENDPOINT_STALENESS_API
}
