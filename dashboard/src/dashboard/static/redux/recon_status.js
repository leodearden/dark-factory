// Reconciliation run-status vocabulary and the pure counting logic derived
// from it — the single home for "what statuses does a reconciliation run
// have, and which of them count as a success?".
//
// PROVENANCE. The store's sole writer is
// fused-memory/src/fused_memory/reconciliation/journal.py::ReconciliationJournal:
// the `runs.status` column DEFAULTS to 'running', and `complete_run` is
// called with exactly 'completed', 'failed' or 'interrupted' across the
// whole reconciliation package. 'success' and 'partial' were never in that
// vocabulary — the dashboard tested for both anyway, which pinned the run
// success rate at 0% forever and left every 'interrupted' run uncounted.
//
// The dashboard cannot import the journal to check this (fused-memory is
// not a dashboard dependency), so drift is caught at RUNTIME instead: a
// status outside the vocabulary lands in the `unknown` bucket and is
// excluded from the success-rate denominator. Both consumers then surface
// it rather than discarding it — the Recon tab prints the count on the rate
// tile, and reconAttentionCount folds it into the rail badge, so a status
// the journal grows later is visible without opening the tab.

const RECON_RUN_IN_FLIGHT = ['running'];
const RECON_RUN_SUCCESS = ['completed'];
// Two leaves under one union, because the rate and the rail badge ask
// different questions of the same rows. For the RATE both are unsuccessful:
// neither run produced a reconciliation. For an ATTENTION signal only
// 'failed' qualifies — 'interrupted' is what journal.py records for a run
// whose process died, and this fleet is restarted routinely (watchdog
// liveness probes, the staleness backstop, and the merge-landed
// coordinator, which passes no --drain and so kills mid-flight units), so
// interrupted rows are a standing population on a perfectly healthy fleet.
// Counting them in the badge would trade the old under-count for a
// permanent false alarm. reconStatusTone draws the same line when it paints
// 'interrupted' warn and 'failed' bad.
const RECON_RUN_FAILED = ['failed'];
const RECON_RUN_INTERRUPTED = ['interrupted'];
const RECON_RUN_UNSUCCESSFUL = [...RECON_RUN_FAILED, ...RECON_RUN_INTERRUPTED];

// Derived, never re-listed: the union cannot drift from its parts.
const RECON_RUN_STATES = [
  ...RECON_RUN_IN_FLIGHT,
  ...RECON_RUN_SUCCESS,
  ...RECON_RUN_UNSUCCESSFUL,
];

const RECON_STATUS_TONES = {
  completed: 'ok',
  failed: 'bad',
  interrupted: 'warn',
  running: 'info',
};

// One pass over a run window. Every row lands in exactly one of inFlight /
// success / failed / interrupted / unknown; `unsuccessful` and `terminal`
// are derived from those rather than counted separately, so no two numbers
// here can disagree.
function reconRunCounts(runs) {
  const list = runs || [];
  const counts = {
    total: list.length,
    inFlight: 0,
    success: 0,
    failed: 0,
    interrupted: 0,
    unsuccessful: 0,
    terminal: 0,
    unknown: 0,
  };
  for (const run of list) {
    const status = run && run.status;
    if (RECON_RUN_IN_FLIGHT.includes(status)) counts.inFlight++;
    else if (RECON_RUN_SUCCESS.includes(status)) counts.success++;
    else if (RECON_RUN_FAILED.includes(status)) counts.failed++;
    else if (RECON_RUN_INTERRUPTED.includes(status)) counts.interrupted++;
    else counts.unknown++;
  }
  counts.unsuccessful = counts.failed + counts.interrupted;
  counts.terminal = counts.success + counts.unsuccessful;
  return counts;
}

// M / (M + K) over TERMINAL runs only. In-flight and unknown rows are not
// in the denominator: a run that has not finished has not failed, and
// including it would make the rate dip whenever reconciliation got busy.
// null (not 0) when nothing has finished — "no terminal runs yet" and
// "everything failed" must not render the same.
function reconSuccessPct(counts) {
  const terminal = (counts && counts.terminal) || 0;
  if (terminal === 0) return null;
  return Math.round(counts.success / terminal * 100);
}

// How many rows in this window an operator should go and look at — the rail
// badge's number. Failures, plus any row carrying a status this module does
// not recognise: unknown rows are the only runtime signal that the journal's
// vocabulary has moved, and a consumer that read a known bucket alone would
// under-count them exactly as silently as the defect this module fixes.
// 'interrupted' is deliberately NOT here (see RECON_RUN_FAILED above).
function reconAttentionCount(counts) {
  if (!counts) return 0;
  return counts.failed + counts.unknown;
}

// The badge class for a run row. Classifies only — the caller still renders
// the raw store status as the badge TEXT, so an unrecognised status shows
// its own name under a muted tone rather than being hidden or mislabelled.
function reconStatusTone(status) {
  // Membership in the vocabulary gates the lookup, so the "unrecognised is
  // muted" rule reads off the same closed set the counts partition, and no
  // inherited Object property ('constructor', 'toString') can leak out as a
  // className.
  return RECON_RUN_STATES.includes(status) ? RECON_STATUS_TONES[status] : 'muted';
}

const RECON_STATUS_API = {
  RECON_RUN_STATES,
  RECON_RUN_IN_FLIGHT,
  RECON_RUN_SUCCESS,
  RECON_RUN_UNSUCCESSFUL,
  reconRunCounts,
  reconSuccessPct,
  reconAttentionCount,
  reconStatusTone,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = RECON_STATUS_API
}
if (typeof window !== 'undefined') {
  window.DF_RECON_STATUS = RECON_STATUS_API
}
