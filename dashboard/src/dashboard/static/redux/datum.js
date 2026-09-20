// datum.js — the CLIENT half of the Datum envelope. Every number the dashboard
// renders arrives with the instant it was measured, how fresh that makes it,
// and — for anything but `fresh` — the producer's verbatim reason. This module
// holds the one decision that turns that envelope into something to draw.
//
// NAMED FOR ITS SERVER TWIN. dashboard/src/dashboard/data/datum.py declares the
// wire contract (the five keys Datum.to_wire() emits, the four DatumState
// names, and the invariants validate_datum enforces). isDatum below is written
// as the reader of exactly that shape, so the SPA checks the contract the
// server already enforces rather than a second, drifting notion of "looks like
// a datum". Matching filenames is what makes the two halves findable from each
// other.
//
// THE CLIENT NEVER CONSTRUCTS OR MUTATES A SERVER DATUM. datum.py's dataclass
// is frozen for the same reason: a Datum is produced server-side only. Two
// client-built envelopes are sanctioned and no more — unknownDatum, for "no
// measurement exists yet", and plainDatum, confined to values not yet SERVED as
// a Datum (see its own note). withReceipt returns a COPY; nothing here writes
// through to a payload the poll loop is holding.
//
// WHY A PLAIN-JS CLASSIC SCRIPT AND NOT A .jsx MODULE. charts.jsx's header
// states the constraint: the .jsx files are `type="text/babel"` behind CDN
// Babel with no node_modules, so nothing in one can be EXECUTED by a test. The
// render DECISION therefore lives here, behaviourally covered by
// dashboard/tests/js/datum.test.mjs, and each shared component becomes a thin
// renderer of the descriptor datumView returns. spark_path.js and
// task_row_cells.js are the same arrangement; the CANONICAL statement of why it
// exists at all is in pins_recovery.js's header and is deliberately not
// restated here.
//
// Dual-loaded: a browser classic `<script>` assigns `window.DF_DATUM`, node
// resolves the same file as CommonJS for `dashboard/tests/js/`. index.html
// loads it after endpoint_staleness.js and before data.js, task_row_cells.js
// and the Babel JSX tags, so the global exists before any consumer's top-level
// destructure runs.

// ── One age formatter for the whole dashboard ──
// Destructured at MODULE SCOPE with no `|| {}` fallback, the DF_SPARK_PATH
// convention: a missing dependency throws at load with a clear message rather
// than deferring to a TypeError inside a render or silently degrading. The
// load-order edge it creates is pinned in tests/test_index_html.py.
//
// formatAge — not window.DF_SHELL.timeago, which PRD open question 4 floated.
// formatAge already produces the coarse humanised shape that question asks for
// ('45s', '6m', '19h 48m'), and it is plain JS with node coverage, so the age
// badge is behaviourally assertable; timeago lives in un-executable JSX, which
// would make this leaf's headline signal — a badge that grows under a mocked
// clock — untestable by construction. Reusing it also means the tile badge and
// the endpoint banner state an age in exactly one format.
//
// BOUND UNDER A MODULE-UNIQUE NAME, not as a bare `formatAge`. Classic scripts
// share ONE global lexical scope, and endpoint_staleness.js already declares a
// top-level `function formatAge` — a same-named const here dies with
// "Identifier 'formatAge' has already been declared" BEFORE reaching the
// window.DF_DATUM assignment at the foot of this file, taking every consumer's
// destructure with it. classic_script_scope.test.mjs measures exactly this, and
// did catch this line in its first spelling.
const { formatAge: formatAgeMs } = window.DF_ENDPOINT_STALENESS;

// ── The vocabulary, read from the server's declaration ──
// The four names datum.py::DatumState declares, in its order. `lower_bound` is
// a measured value known to UNDER-report — a windowed count whose rows outside
// the window were never read — which is why it renders with a '≥' prefix
// rather than as a plain number.
const DATUM_STATES = ['fresh', 'stale', 'unknown', 'lower_bound'];

// The five keys Datum.to_wire() emits. Listed once, here, because both the
// shape check below and any future reader of the envelope must agree on it.
const DATUM_WIRE_KEYS = ['value', 'as_of', 'state', 'reason', 'freshness_bound_seconds'];

// ── Is this the envelope, or a bare value a call site forgot to wrap? ──
// A SHAPE check — the five keys present, the state recognised — and
// deliberately not a re-implementation of validate_datum's invariants. The
// server enforces the invariants; the client's job is to notice when what
// arrived is not an envelope at all. Duplicating the invariant rules here would
// put the contract in two places free to drift, and would make the SPA the
// second authority on a question the producer has already answered.
//
// `null` has typeof 'object' and an array has all its indices, so both are
// excluded explicitly rather than by a lazy typeof test.
function isDatum(x) {
  if (x === null || typeof x !== 'object' || Array.isArray(x)) return false;
  for (const key of DATUM_WIRE_KEYS) {
    if (!(key in x)) return false;
  }
  return DATUM_STATES.indexOf(x.state) !== -1;
}

// ── "No measurement exists" — the one client-built placeholder envelope ──
// The unknown TRIAD datum.py::validate_datum enforces: state 'unknown' iff
// value is null iff as_of is null. Returning a real Datum rather than null or
// undefined is what lets a hole travel through the same code path as a value —
// datumView renders it as an em-dash carrying `reason` as its tooltip, so an
// operator reads WHY a number is absent instead of seeing a seed zero.
//
// freshness_bound_seconds is 0 because there is no measurement for any bound to
// apply to; the field is present because the contract has five keys and an
// envelope missing one is not a Datum.
//
// A fresh object per call, never a shared frozen singleton: callers stamp
// receipts onto datums, and a shared literal would let one call site's stamp
// show up on every other site's placeholder.
function unknownDatum(reason) {
  return {
    value: null,
    as_of: null,
    state: 'unknown',
    reason,
    freshness_bound_seconds: 0,
  };
}

// ── The guard that makes a missed migration site fail loudly ──
// Returns *x* unchanged so it composes inline at the head of a render.
//
// THROWS UNCONDITIONALLY — in the browser exactly as under node. There is no
// environment sniff to condition it on, and a component that quietly renders a
// bare number in production while throwing in tests is precisely the silent
// degradation this repo's loud-over-silent norm rejects. During the 43-site
// StatTile migration a missed site fails immediately and by name in dev, rather
// than rendering an unprovenanced number that looks exactly like a measured
// one — which is the failure the envelope exists to remove.
function assertDatum(x, who) {
  if (isDatum(x)) return x;
  throw new TypeError(
    who + ' was given ' + describeNonDatum(x) + ' where a Datum was required — ' +
      'every value it renders must arrive wrapped (see datum.js / data/datum.py). ' +
      'Wrap a not-yet-served value with plainDatum(value, endpointKey).',
  );
}

// A short, safe rendering of whatever arrived instead of a Datum. Its own
// function so assertDatum's message stays one sentence, and so the description
// can never itself throw on an exotic value (a getter, a revoked proxy) while
// building an error about one.
function describeNonDatum(x) {
  if (x === null) return 'null';
  if (Array.isArray(x)) return 'an array';
  if (typeof x !== 'object') return typeof x + ' ' + String(x);
  return 'an object with keys [' + Object.keys(x).join(', ') + ']';
}

// ── Provenance: when did THIS browser receive THIS payload? ──
// Returns a COPY of *datum* carrying the two receipt fields. A copy, never a
// write-through: the poll loop holds the object the server sent, and stamping
// into it would mutate state other readers are already looking at.
//
// `_served_at` is the server's own serving instant (ISO-8601, or null until PRD
// leaf beta puts a top-level `served_at` on the wire); `_received_at` is this
// browser's clock at the moment the response resolved. They are underscored to
// mark them as client-side annotations on a server payload rather than part of
// the five-key wire contract — isDatum ignores them, as it must, because a
// datum is equally valid before and after it is stamped.
function withReceipt(datum, receipt) {
  const r = receipt || {};
  return { ...datum, _served_at: r.servedAt, _received_at: r.receivedAt };
}

// ── How old does this number look RIGHT NOW? ──
//   (served_at − as_of)  +  (now − received_at)
//     server-side gap         client-side gap
//
// TWO CLOCKS, NEVER MIXED. Each term subtracts two readings of ONE clock, so
// the sum is exact even when the browser and the server disagree — the naive
// `now − as_of` reads a skewed browser's perfectly fresh tile as hours stale,
// or as negative. It is also the only formulation that GROWS: the client term
// advances every time this is called, which is what makes a wedged endpoint's
// tile visibly age instead of resting at a reassuring constant.
//
// RETURNS NULL, NOT ZERO AND NOT NaN, whenever a term is unavailable: an
// unknown datum has no as_of to age, an unstamped one has no receipt. Rendering
// an age of zero from a missing timestamp is the `_minutes_since` mistake
// endpoint_staleness.js::noticeText already documents — it fabricates
// reassurance during exactly the failure this signal exists to surface — and
// NaN would reach the operator as 'an unknown time' via formatAgeMs, which says
// the same thing far less clearly than showing no badge at all.
//
// A MISSING `_served_at` is the one absence that degrades rather than nulls,
// because it is today's normal case: no polled payload carries a top-level
// `served_at` yet, so the server-side gap is unknown rather than wrong.
// Contributing zero for it makes the result a LOWER bound on the true age —
// honest, and still growing — where nulling would leave every tile un-aged
// until beta lands.
function displayedAgeMs(datum, now) {
  const d = datum || {};
  const clientGap = Number(now) - Number(d._received_at);
  if (!Number.isFinite(clientGap)) return null;

  const measuredAt = Date.parse(d.as_of);
  if (!Number.isFinite(measuredAt)) return null;

  const servedAt = Date.parse(d._served_at);
  const serverGap = Number.isFinite(servedAt) ? servedAt - measuredAt : 0;

  return serverGap + clientGap;
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced at
// runtime by dashboard/tests/js/classic_script_scope.test.mjs. A collision here
// would leave window.DF_DATUM undefined and break the top-level destructures in
// data.js, task_row_cells.js, charts.jsx, shell.jsx and tabs.jsx.
const DATUM_API = {
  DATUM_STATES,
  isDatum,
  unknownDatum,
  assertDatum,
  withReceipt,
  displayedAgeMs,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = DATUM_API;
}
if (typeof window !== 'undefined') {
  window.DF_DATUM = DATUM_API;
}
