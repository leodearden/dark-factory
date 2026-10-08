// performance_cards.js — the CLIENT twin of the cards half of
// dashboard/src/dashboard/data/performance.py, and the ONE reader of the
// /performance Datums: each project's cards and the PERFORMANCE_LISTING that
// says whether the listing itself could be read. PerfTab's header tiles and its
// per-project blocks read them here. Named for its server twin's cards (the
// datum.js precedent).
//
// A PLAIN-JS CLASSIC SCRIPT, NOT A .jsx MODULE, so its decisions are
// EXECUTABLE: dashboard/tests/js/performance_cards.test.mjs runs what no
// harness here can run inside a .jsx body. pins_recovery.js's header holds the
// CANONICAL statement of why.
//
// LOAD CONTRACT. It destructures window.DF_DATUM at module scope with no
// fallback, so index.html loads it after datum.js and before tabs.jsx
// (test_index_html.py pins the order). A browser classic `<script>` assigns
// `window.DF_PERFORMANCE_CARDS`; node requires the same file as CommonJS once
// its test has put DF_ENDPOINT_STALENESS on a window shim.
//
// AN EMPTY PERFORMANCE IS NOT "NO TASKS" BY ITSELF. Projects are discovered
// from each runs.db, so a runs.db that could not be read lists nothing; only
// the listing knows whether every one was read. A header tile derived over the
// cards with no value therefore asks the listing why.

// Module scope, no fallback, RENAMED — see the CANONICAL note in datum.js's
// header.
const { servedDatum: servedCardsDatum, datumView: viewCardsDatum } = window.DF_DATUM;

// data.js registers PERFORMANCE and PERFORMANCE_LISTING as PLAIN, so the
// Datums they carry arrive with no receipt of their own; this endpoint's
// receipt stamps them.
const PERF_CARDS_ENDPOINT = '/api/v2/dashboard/performance';

function cardsReceipts(data) {
  return data.__receipt || {};
}

// *project*'s served cards Datum, stamped with this endpoint's receipt.
function projectCards(data, project) {
  const entry = (data.PERFORMANCE || {})[project];
  const why = entry
    ? 'the /performance entry for ' + project + ' has no cards Datum'
    : 'the /performance payload has no entry for ' + project;
  return servedCardsDatum(entry && entry.cards, PERF_CARDS_ENDPOINT, why, cardsReceipts(data));
}

// How many projects the payload lists, and whether every runs.db was read to
// list them.
function cardsListing(data) {
  return servedCardsDatum(
    data.PERFORMANCE_LISTING,
    PERF_CARDS_ENDPOINT,
    'the /performance payload has no PERFORMANCE_LISTING Datum',
    cardsReceipts(data),
  );
}

// Why a value derived over the cards is absent: the listing's own reason when
// it is anything but fresh — its title is null exactly then — else the window
// genuinely holds no tasks.
function cardsAbsentReason(data) {
  return viewCardsDatum(cardsListing(data)).title || 'no tasks in this window';
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const PERFORMANCE_CARDS_API = {
  projectCards,
  cardsListing,
  cardsAbsentReason,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PERFORMANCE_CARDS_API;
}
if (typeof window !== 'undefined') {
  window.DF_PERFORMANCE_CARDS = PERFORMANCE_CARDS_API;
}
