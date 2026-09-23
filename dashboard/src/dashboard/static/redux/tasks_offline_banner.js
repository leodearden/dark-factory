// tasks_offline_banner.js — pure banner-copy decision for the Tasks tab
// (tab_tasks.jsx).
//
// This is a plain-JS module: no JSX, no Babel. It is loaded two ways:
//   - In the browser, via a classic `<script src="/static/redux/tasks_offline_banner.js">`
//     tag (like task_status_counts.js / prd_grouping.js), which assigns
//     `window.DF_TASKS_OFFLINE_BANNER`.
//   - In node (no package.json in this repo, so this file resolves as
//     CommonJS), via `require`/`import` for the `node --test` suite under
//     dashboard/tests/js/.
//
// Both export paths are guarded so this file has no effect outside the
// environment it's actually running in. index.html loads it as a classic
// script BEFORE the Babel JSX tags, so the global is defined by the time
// tab_tasks.jsx runs.
//
// WHY THIS MODULE EXISTS. The Tasks tab rendered ONE banner, gated on a single
// boolean, reading:
//
//     fused-memory offline — task data unavailable
//
// The boolean was `bool(offline_projects)` server-side, so ANY one project's
// fetch failing raised it. On a nine-root config that printed a total-outage
// claim directly above eight healthy projects' rows — carried in the very same
// payload. Both halves of the sentence were false: fused-memory was
// demonstrably reachable (those rows came back through it), and task data was
// emphatically available.
//
// The honest distinction is now made SERVER-SIDE, in `app.api_tasks`, which
// emits four separate facts (TASKS_OFFLINE = no root produced rows AND at
// least one demonstrably failed; TASKS_OFFLINE_PROJECTS = roots whose fetch
// demonstrably failed; TASKS_DEGRADED_PROJECTS = roots the handler ran out of
// budget for, whose state is UNKNOWN; TASKS_COUNT_UNKNOWN_PROJECTS = roots
// whose rows are current but whose done count was never measured), plus
// TASKS_PROJECT_COUNT as the denominator the "k of N" copy below needs. Each
// of the four gets its own notice kind here. This module only RENDERS that
// distinction — it must not re-derive it, or the two would drift.
//
// Nothing here reads `window` or `document`: a module that reached for browser
// globals would behave differently under `node --test` than in the browser.

// Preserved VERBATIM from the pre-change JSX. The global case was the one
// state this wording actually described, so it survives unedited; only its
// TRIGGER narrowed.
const GLOBAL_TEXT = 'fused-memory offline — task data unavailable'

function asList(value) {
  return Array.isArray(value) ? value.filter(Boolean) : []
}

// Renders a project list for banner copy. Bounded so a pathological config
// (every root degraded) cannot produce a banner longer than the grid it sits
// above; the count in the surrounding copy is always the FULL count, so the
// elision never hides the scale of the problem.
const MAX_NAMED = 6

function nameList(labels) {
  if (labels.length <= MAX_NAMED) return labels.join(', ')
  return labels.slice(0, MAX_NAMED).join(', ') + ', +' + (labels.length - MAX_NAMED) + ' more'
}

// A k-of-N phrase that stays truthful when N is unknown or stale.
// `totalProjects` and `offlineProjects` arrive from the same payload but
// `totalProjects` is derived from the config, so a root added mid-render can
// make k > N momentarily. Falling back to a bare count beats printing "1 of 0".
function countPhrase(k, total) {
  const n = Number(total)
  if (Number.isFinite(n) && n >= k && n > 0) return k + ' of ' + n
  return k === 1 ? '1 project' : k + ' projects'
}

/**
 * Decide which banner notices the Tasks tab should render.
 *
 * Returns an array of `{kind, text}` — empty when nothing failed. Kinds:
 *
 *   'global'   — no root produced rows and at least one demonstrably failed:
 *                fused-memory itself is unreachable. Carries the verbatim
 *                outage copy, and SUBSUMES the other three (its "task data
 *                unavailable" already covers everything they could add).
 *                The parenthetical names the roots that demonstrably failed,
 *                which in a hang is a SUBSET of the roots without rows (the
 *                rest timed out) — a partial attribution, never a false one,
 *                since the sentence's claim is about the fetch as a whole.
 *   'partial'  — some roots failed, not all. Deliberately avoids the words
 *                "fused-memory offline": fused-memory is demonstrably
 *                reachable in this state.
 *   'degraded' — the handler ran out of budget before reaching these roots.
 *                Nothing was proven unreachable, so this never claims an
 *                outage; it says the view is partial. Composes with 'partial'
 *                (a project can fail while a different one times out).
 *   'count-unknown'
 *              — the rows for these roots are current, but their compact
 *                status map failed, so the done count was never measured and
 *                the terminal window was skipped. Neither offline nor
 *                degraded, so it never claims an outage either. Without this
 *                notice such a project renders as healthy with a confident
 *                "0 done".
 *
 * Every input is optional and defaulted: this runs against window.DF_DATA's
 * pre-fetch defaults on the very first render, so a missing key must produce
 * no banner rather than a TypeError that takes the whole tab down.
 */
function tasksBannerNotices(state) {
  const s = state || {}
  const offlineProjects = asList(s.offlineProjects)
  const degradedProjects = asList(s.degradedProjects)
  const countUnknownProjects = asList(s.countUnknownProjects)

  // The server already decided whether this is a total outage; do not
  // re-derive it from the lists here (see the header note on drift).
  if (s.offline) {
    const named = offlineProjects.length > 0 ? ' (' + nameList(offlineProjects) + ')' : ''
    return [{ kind: 'global', text: GLOBAL_TEXT + named }]
  }

  const notices = []

  if (offlineProjects.length > 0) {
    notices.push({
      kind: 'partial',
      text:
        'task data unavailable for ' +
        countPhrase(offlineProjects.length, s.totalProjects) +
        ' (' + nameList(offlineProjects) + ') — other projects below are current',
    })
  }

  if (degradedProjects.length > 0) {
    notices.push({
      kind: 'degraded',
      text:
        'the task fetch timed out for ' +
        countPhrase(degradedProjects.length, s.totalProjects) +
        ' (' + nameList(degradedProjects) + ') — this view is partial, not empty',
    })
  }

  // Neither offline nor degraded: the rows are current, but the done count
  // was never measured and the terminal window was skipped. Without this
  // notice the project renders as healthy with a confident "0 done".
  if (countUnknownProjects.length > 0) {
    notices.push({
      kind: 'count-unknown',
      text:
        'the completed-task count is unavailable for ' +
        countPhrase(countUnknownProjects.length, s.totalProjects) +
        ' (' + nameList(countUnknownProjects) + ') — their rows below are ' +
        'current, but done counts show as unknown',
    })
  }

  return notices
}

const TASKS_OFFLINE_BANNER_API = { tasksBannerNotices }

if (typeof module !== 'undefined' && module.exports) {
  module.exports = TASKS_OFFLINE_BANNER_API
}
if (typeof window !== 'undefined') {
  window.DF_TASKS_OFFLINE_BANNER = TASKS_OFFLINE_BANNER_API
}
