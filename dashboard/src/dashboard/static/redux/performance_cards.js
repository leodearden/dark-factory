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
// cards with no value therefore asks the listing why, and a tile with a value
// asks whether it covers only part of its scope.
//
// THE LISTING IS FLEET-WIDE; PerfTab's tiles are scoped by its projectFilter
// (empty is the whole fleet). The projects an unread runs.db would have listed
// are exactly the ones missing from PERFORMANCE, so a filter naming only listed
// projects lost none of them to the shortfall, and the listing's reason is not
// its cause.

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

function scopeListedInFull(data, projectFilter) {
  const listed = data.PERFORMANCE || {};
  return projectFilter.length > 0 && projectFilter.every(project => Object.hasOwn(listed, project));
}

// Why a value derived over *projectFilter*'s cards is absent: the listing's own
// reason when it is anything but fresh — its title is null exactly then — and
// the scope may hold a project it could not list; else the window genuinely
// holds no tasks.
function cardsAbsentReason(data, projectFilter) {
  const why = viewCardsDatum(cardsListing(data)).title;
  return why && !scopeListedInFull(data, projectFilter) ? why : 'no tasks in this window';
}

// Why the values derived over *projectFilter*'s cards may cover only part of
// it: a short listing's reason when the scope may hold a project it could not
// list, else null. A tile with a value has no hole to carry the reason, so it
// rides as the tile's caveat.
function cardsShortfall(data, projectFilter) {
  const listing = cardsListing(data);
  const short = listing.state === 'lower_bound' && !scopeListedInFull(data, projectFilter);
  return short ? listing.reason : null;
}

// Module-unique export const, never a bare `API` — the CANONICAL note in
// datum.js's header, enforced at runtime by classic_script_scope.test.mjs.
const PERFORMANCE_CARDS_API = {
  projectCards,
  cardsListing,
  cardsAbsentReason,
  cardsShortfall,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PERFORMANCE_CARDS_API;
}
if (typeof window !== 'undefined') {
  window.DF_PERFORMANCE_CARDS = PERFORMANCE_CARDS_API;
}
