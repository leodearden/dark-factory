// invariant_chip.js — the claimant-invariant chip decision for the Tasks tab's
// per-project groups (tab_tasks.jsx).
//
// Plain JS, no JSX: loaded in the browser as a classic <script> tag before the
// Babel tags (assigning window.DF_INVARIANT_CHIP), and in node as CommonJS for
// dashboard/tests/js/invariant_chip.test.mjs. Nothing here reads window or
// document at load.
//
// Input is DF_DATA.INVARIANTS: the per-root records served by
// /api/v2/dashboard/invariants (docs/prds/claimant-invariant-detection.md
// E-1). Two rules from D-2 shape every branch. Blind is not clean: a root the
// server could not read renders a loud chip with no count in it. Unknown is
// not clean either: a project with no record renders no chip at all, never a
// "0". The counts are the server's; this module only adds them up when two
// roots share one basename, so a collision can never silently drop a root.

function invariantChipRecords(invariants, project) {
  if (!Array.isArray(invariants)) return []
  return invariants.filter((record) => record && record.project === project)
}

function invariantChipSum(records, read) {
  return records.reduce((total, record) => total + read(record), 0)
}

function invariantChipBlind(blindRecords) {
  return {
    text: 'claimant gauge blind',
    tone: 'bad',
    title:
      'claimant-invariant gauge could not read this project: ' +
      blindRecords.map((record) => record.error).join('; '),
  }
}

function invariantChipMeasured(records) {
  const count = invariantChipSum(records, (record) => record.terminal_tier.count)
  const unledgered = invariantChipSum(records, (record) => record.terminal_tier.unledgered_count)
  const unledgeredIds = records
    .flatMap((record) => record.terminal_tier.violations)
    .filter((violation) => !violation.ledgered)
    .map((violation) => violation.tag + '#' + violation.task_id)
  const titleLines = [
    'terminal tasks still carrying a claimant: ' + count,
    invariantChipSum(records, (record) => record.rows_scanned) +
      ' rows scanned at ' + records[0].measured_at,
    'stale hygiene-tier claimants: ' +
      invariantChipSum(records, (record) => record.hygiene_tier.stale_count) + ' of ' +
      invariantChipSum(records, (record) => record.hygiene_tier.count),
  ]
  if (unledgeredIds.length > 0) titleLines.push('unledgered: ' + unledgeredIds.join(', '))
  return {
    text: 'terminal claimants ' + count + (unledgered > 0 ? ' · ' + unledgered + ' unledgered' : ''),
    tone: unledgered > 0 ? 'bad' : 'muted',
    title: titleLines.join('\n'),
  }
}

/**
 * The chip for *project*, as `{text, tone, title}`, or null when no record
 * names it. `tone` is one of runtime_format.js's 'muted' | 'warn' | 'bad'.
 * Ledgered violations (sanctioned C4-E2 mints) are counted but stay muted;
 * only unledgered ones alarm.
 */
function invariantChipFor(invariants, project) {
  const records = invariantChipRecords(invariants, project)
  if (records.length === 0) return null
  const blindRecords = records.filter((record) => record.status !== 'ok')
  if (blindRecords.length > 0) return invariantChipBlind(blindRecords)
  return invariantChipMeasured(records)
}

const INVARIANT_CHIP_API = { invariantChipFor }

if (typeof module !== 'undefined' && module.exports) {
  module.exports = INVARIANT_CHIP_API
}
if (typeof window !== 'undefined') {
  window.DF_INVARIANT_CHIP = INVARIANT_CHIP_API
}
