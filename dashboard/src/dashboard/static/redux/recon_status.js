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
// status outside the vocabulary lands in the `unknown` bucket, is excluded
// from the success-rate denominator, and is surfaced on the tile — visible
// to an operator rather than silently discarded.

const RECON_RUN_IN_FLIGHT = ['running'];
const RECON_RUN_SUCCESS = ['completed'];
// 'interrupted' did not succeed, so it belongs here for the rate — but it
// is a recovered restart artefact rather than a failure to investigate,
// which is why reconStatusTone paints it 'warn' and not 'bad'.
const RECON_RUN_UNSUCCESSFUL = ['failed', 'interrupted'];

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
// success / unsuccessful / unknown; `terminal` is derived from two of them
// rather than counted separately, so no two numbers here can disagree.
function reconRunCounts(runs) {
  const list = runs || [];
  const counts = {
    total: list.length,
    inFlight: 0,
    success: 0,
    unsuccessful: 0,
    terminal: 0,
    unknown: 0,
  };
  for (const run of list) {
    const status = run && run.status;
    if (RECON_RUN_IN_FLIGHT.includes(status)) counts.inFlight++;
    else if (RECON_RUN_SUCCESS.includes(status)) counts.success++;
    else if (RECON_RUN_UNSUCCESSFUL.includes(status)) counts.unsuccessful++;
    else counts.unknown++;
  }
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
  reconStatusTone,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = RECON_STATUS_API
}
if (typeof window !== 'undefined') {
  window.DF_RECON_STATUS = RECON_STATUS_API
}
