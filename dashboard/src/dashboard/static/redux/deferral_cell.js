// deferral_cell.js — what the Tasks tab draws for a task row's `deferral` key,
// and the per-project chip counting the holds that need a human.
//
// Plain JS, no JSX: loaded in the browser as a classic <script> before the
// Babel tags (assigning window.DF_DEFERRAL_CELL), and in node as CommonJS for
// dashboard/tests/js/deferral_cell.test.mjs.
//
// The summary is the server's (dashboard/src/dashboard/data/deferral_summary.py),
// and its label, state and needs_human verdict are drawn as given. The one
// verdict this module decides itself: an explicit null summary, a deferred row
// with no record, needs a human. test_tab_tasks_deferral.py pins that rule
// against shared.task_deferral.needs_human through the rendered header count.

// Renamed per datum.js's CANONICAL note: datum.js already binds formatAgeMs.
const { formatAge: deferralAgeText } = window.DF_ENDPOINT_STALENESS

const DEFERRAL_UNRECORDED_TEXT = 'no deferral recorded'
const DEFERRAL_UNRECORDED_TITLE =
  'deferred with no deferral record, so nothing says what ends this hold\nneeds a human: missing_record'

function deferralUnrecordedCell() {
  return {
    text: DEFERRAL_UNRECORDED_TEXT,
    age: null,
    state: 'unrecorded',
    needsHuman: true,
    title: DEFERRAL_UNRECORDED_TITLE,
  }
}

function deferralSummaryCell(summary) {
  const needsHuman = summary.needs_human != null
  return {
    text: summary.label,
    age: summary.age_seconds == null ? 'age unknown' : deferralAgeText(summary.age_seconds * 1000),
    state: summary.state,
    needsHuman,
    title: needsHuman ? summary.detail + '\nneeds a human: ' + summary.needs_human : summary.detail,
  }
}

/**
 * The hold drawn for *task*, as `{text, age, state, needsHuman, title}`, or
 * null when the row has no own `deferral` key: it is not deferred, or an older
 * backend served it, and either way it renders exactly as before.
 */
function deferralCellState(task) {
  if (task == null || !Object.prototype.hasOwnProperty.call(task, 'deferral')) return null
  return task.deferral == null ? deferralUnrecordedCell() : deferralSummaryCell(task.deferral)
}

/**
 * The per-project chip, as `{count, text, title}`, over the rows whose hold
 * needs a human; null when none does.
 */
function deferralNeedsHumanChip(rows) {
  if (!Array.isArray(rows)) return null
  const lines = []
  for (const row of rows) {
    const cell = deferralCellState(row)
    if (cell !== null && cell.needsHuman) lines.push(row.id + ': ' + cell.text)
  }
  if (lines.length === 0) return null
  return {
    count: lines.length,
    text: lines.length === 1 ? '1 deferral needs a human' : lines.length + ' deferrals need a human',
    title: lines.join('\n'),
  }
}

const DEFERRAL_CELL_API = { deferralCellState, deferralNeedsHumanChip }

if (typeof module !== 'undefined' && module.exports) {
  module.exports = DEFERRAL_CELL_API
}
if (typeof window !== 'undefined') {
  window.DF_DEFERRAL_CELL = DEFERRAL_CELL_API
}
