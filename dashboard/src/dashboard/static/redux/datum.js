// datum.js — the CLIENT half of the Datum envelope. Every number the dashboard
// renders arrives with the instant it was measured, how fresh that makes it,
// and — for anything but `fresh` — the producer's verbatim reason. This module
// holds the one decision that turns that envelope into something to draw.
//
// NAMED FOR ITS SERVER TWIN, dashboard/src/dashboard/data/datum.py, which
// declares the wire contract this file reads. Matching filenames are what make
// the two halves findable from each other.
//
// THE CLIENT NEVER CONSTRUCTS OR MUTATES A SERVER DATUM — datum.py's dataclass
// is frozen for the same reason. Three client-built envelopes are sanctioned
// and no more: unknownDatum, plainDatum and derivedDatum, each below with the
// gap it covers.
//
// A PLAIN-JS CLASSIC SCRIPT, NOT A .jsx MODULE, so the render decision is
// EXECUTABLE: the .jsx files are `type="text/babel"` behind CDN Babel with no
// node_modules. spark_path.js and task_row_cells.js are the same arrangement,
// and pins_recovery.js's header holds the CANONICAL statement of why.
//
// Dual-loaded: a browser classic `<script>` assigns `window.DF_DATUM`, node
// resolves the same file as CommonJS for `dashboard/tests/js/`. index.html
// loads it after endpoint_staleness.js and before data.js, task_row_cells.js
// and the Babel JSX tags, so the global exists before any consumer's top-level
// destructure runs.
//
// ── CANONICAL: HOW A CONSUMER TAKES THIS MODULE, AND WHY ──────────────────
// Stated once, here, because nine files now destructure DF_DATUM and nine
// hand-copies of one rationale drift — the hazard these modules exist to
// remove. Each consumer carries a one-line pointer to this block instead.
// (task_row_cells.js's header points at pins_recovery.js's CANONICAL block for
// the shared-substrate rationale by the same rule.)
//
// AT MODULE SCOPE, WITH NO `|| {}` FALLBACK — the DF_SPARK_PATH convention. A
// missing or mis-ordered dependency throws at load with a clear message rather
// than deferring to a TypeError inside a render or degrading silently.
// index.html's load order is the enforced contract, pinned per-module by
// tests/test_index_html.py.
//
// WHICH NAME TO BIND IT UNDER DEPENDS ON THE KIND OF FILE, and getting it wrong
// fails at LOAD rather than at first use:
//   · a classic `<script>` — data.js, task_row_cells.js, and this file — shares
//     ONE global lexical scope with every other classic script, so a `const`
//     matching a top-level declaration elsewhere dies with "Identifier 'x' has
//     already been declared" before the file reaches its own `window.DF_*`
//     assignment, taking every downstream destructure with it. Rename in the
//     destructure: `{ datumView: viewOfDatum }`.
//   · a `type="text/babel"` tag — charts.jsx, shell.jsx, tabs.jsx and the
//     tab_*.jsx files — is downlevelled by Babel-standalone, whose top-level
//     bindings never join that scope. Bind under datum.js's own names.
// Measured, not assumed: classic_script_scope.test.mjs's SCOPE note records
// three independent witnesses, and it caught the destructure below in its
// first spelling.

// ── One age formatter for the whole dashboard ──
// Renamed per the CANONICAL note above: endpoint_staleness.js already declares
// a top-level `function formatAge`.
//
// formatAge — not window.DF_SHELL.timeago, which PRD open question 4 floated.
// formatAge already produces the coarse humanised shape that question asks for
// ('45s', '6m', '19h 48m'), and it is plain JS with node coverage, so the age
// badge is behaviourally assertable; timeago lives in un-executable JSX, which
// would make this leaf's headline signal — a badge that grows under a mocked
// clock — untestable by construction. Reusing it also means the tile badge and
// the endpoint banner state an age in exactly one format.
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
// A SHAPE check, deliberately NOT a re-implementation of validate_datum's
// invariants: the server enforces those, and a second copy here would make the
// SPA a drifting authority on a question the producer has already answered.
function isDatum(x) {
  if (x === null || typeof x !== 'object' || Array.isArray(x)) return false;
  for (const key of DATUM_WIRE_KEYS) {
    if (!(key in x)) return false;
  }
  return DATUM_STATES.indexOf(x.state) !== -1;
}

// ── "No measurement exists" — the client-built placeholder envelope ──
// The unknown TRIAD datum.py::validate_datum enforces: state 'unknown' iff
// value is null iff as_of is null. Returning a real Datum rather than null lets
// a hole travel the same path as a value, so an operator reads WHY a number is
// absent instead of seeing a seed zero.
//
// A fresh object per call, never a shared frozen singleton: callers stamp
// receipts onto datums, and a shared literal would show one call site's stamp
// on every other site's placeholder.
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
// THROWS UNCONDITIONALLY, in the browser exactly as under node. A component
// that quietly renders a bare number in production while throwing in tests is
// the silent degradation this repo's loud-over-silent norm rejects — and an
// unprovenanced number that looks exactly like a measured one is the failure
// the envelope exists to remove.
function assertDatum(x, who) {
  if (isDatum(x)) return x;
  throw new TypeError(
    who + ' was given ' + describeNonDatum(x) + ' where a Datum was required — ' +
      'every value it renders must arrive wrapped (see datum.js / data/datum.py). ' +
      'Wrap a not-yet-served value with plainDatum(value, endpointKey).',
  );
}

// A short, safe rendering of whatever arrived instead of a Datum — its own
// function so it can never itself throw on an exotic value (a getter, a revoked
// proxy) while building an error about one.
function describeNonDatum(x) {
  if (x === null) return 'null';
  if (Array.isArray(x)) return 'an array';
  if (typeof x !== 'object') return typeof x + ' ' + String(x);
  return 'an object with keys [' + Object.keys(x).join(', ') + ']';
}

// ── Provenance: when did THIS browser receive THIS payload? ──
// A COPY, never a write-through: the poll loop holds the object the server
// sent, and stamping into it would mutate state other readers are looking at.
//
// `_served_at` is the server's serving instant, null for a payload with no
// top-level `served_at` (since PRD leaf beta, /tasks and /orchestrators carry
// one); `_received_at` is this browser's clock when the response resolved. Underscored as client-side annotations rather
// than wire keys — isDatum ignores them, as it must, because a datum is equally
// valid before and after it is stamped.
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
// RETURNS NULL, NOT ZERO AND NOT NaN, whenever a term is unavailable. An age of
// zero from a missing timestamp is the `_minutes_since` mistake
// endpoint_staleness.js::noticeText documents — it fabricates reassurance
// during exactly the failure this signal exists to surface — and NaN would
// reach the operator as 'an unknown time' via formatAgeMs.
//
// A MISSING `_served_at` DEGRADES RATHER THAN NULLS, because it is still the
// normal case: only /tasks and /orchestrators carry a top-level `served_at`
// (since PRD leaf beta), so for every other endpoint the server gap is unknown
// rather than wrong. Contributing zero makes the result a LOWER bound: honest
// and still growing, where nulling would leave those tiles un-aged.
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

// ── The one true placeholder, and the one true under-report prefix ──
// Exported for task_row_cells.js::STRAND_TITLE's reason: 43 call sites
// hand-spelling a placeholder is 43 chances to disagree about it. '≥' marks a
// `lower_bound` value — measured, but known to under-report, so it reads as a
// floor rather than a count.
const EM_DASH = '—';
const LOWER_BOUND_PREFIX = '≥';

// ── How should this datum be drawn? ──
// Returns `{text, title, age, prefix, isHole}`:
//   text   — what the value cell says, formatter and prefix already applied;
//   title  — the producer's reason, as a tooltip, or null when there is none;
//   age    — the humanised displayed age to badge, or null for no badge;
//   prefix — '≥' or '', the same decision surfaced separately for a call site
//            that wants to style the marker; `text` already carries it, so a
//            caller rendering only `text` is complete;
//   isHole — is there a measurement here at all? Surfaced BECAUSE a caller
//            whose value is not text (task_row_cells.js::locksCellState renders
//            a chip LIST) still has to know, and re-deriving it from
//            `datum.state` there would make that cell a second authority on the
//            hole rule — free to keep answering the old way if this arm ever
//            widens. Reading it here is what keeps the answer in one place.
//
// ONE DECISION, NO PER-CALLER BRANCHING — the entire point of the leaf. A tile
// that grew its own arm for holes or its own age spelling would be a second
// authority on a question answered here, which is how 43 tiles came to
// hand-roll 14 different null guards in the first place.
//
// FORMAT IS NEVER INVOKED ON A HOLE — the unknown arm returns first. That, and
// not the placeholder, is the load-bearing half: charts.jsx::HBarChart records
// that its live call sites pass formatters which throw on a missing value, so
// invoking one on a hole takes down the whole tab rather than blanking a cell.
// It is also what let the migration DELETE each site's `x == null ? '—' : f(x)`
// sentinel instead of keeping two hole decisions per tile.
//
// A NEGATIVE DISPLAYED AGE IS UNBADGEABLE, for displayedAgeMs's own reason:
// formatAgeMs answers any n < 0 with 'an unknown time'. Reachable on both
// clocks — an NTP step backwards makes the client gap negative, and a producer
// whose `as_of` is later than the response's `served_at` makes the server gap
// negative. The age stays computed and honest; only the badge is withheld.
//
// THE AGE BADGE WINS OVER THE SERVER'S STATE. A datum served `fresh` was fresh
// when it was served; if it has since aged past its producer's bound sitting in
// this browser, the badge appears anyway. No reason is invented — the server
// gave none — but the operator reads the age rather than a verdict that has
// quietly expired.
function datumView(datum, opts) {
  assertDatum(datum, 'datumView');
  const o = opts || {};
  const format = o.format || String;
  const now = o.now === undefined ? Date.now() : o.now;

  if (datum.state === 'unknown') {
    return { text: EM_DASH, title: datum.reason, age: null, prefix: '', isHole: true };
  }

  const ageMs = displayedAgeMs(datum, now);
  const badgeable = ageMs !== null && ageMs >= 0;
  const overBound = badgeable && ageMs > datum.freshness_bound_seconds * 1000;
  const prefix = datum.state === 'lower_bound' ? LOWER_BOUND_PREFIX : '';

  return {
    text: prefix + format(datum.value),
    title: datum.state === 'fresh' ? null : datum.reason,
    age: badgeable && (datum.state !== 'fresh' || overBound) ? formatAgeMs(ageMs) : null,
    prefix,
    isHole: false,
  };
}

// ── How long a plain-wrapped value may sit before its tile badges ──
// TWELVE SECONDS = FOUR POLL INTERVALS. data.js polls every 3000ms with up to
// 1500ms of jitter, so a healthy endpoint's receipt is at most ~4.5s old; 12s
// is ~2.6x headroom, enough that a jittered late poll never makes a working
// tile twitch.
//
// ORDERED FINE-THEN-COARSE AGAINST THE BANNER, deliberately not aligned with
// it: endpoint_staleness.js declares an endpoint stale after ~21s of backoff,
// so badging at 12s makes the tile the FIRST per-value signal and the banner
// the later per-endpoint explanation — one authority each. Matching 21s would
// read as a duplicate of the banner; the PRD's suggested 6s would badge a
// healthy tile whenever a jittered poll ran late.
const PLAIN_DATUM_BOUND_SECONDS = 12;

// ── Provenance for a value the server does not yet serve as a Datum ──
// Every polled payload is still a bare number: PRD leaf beta has not landed.
// Rather than leave 43 tiles unprovenanced until it does, this wraps a plain
// value in what IS known — which endpoint delivered it, and when. That is
// ENDPOINT granularity, coarser than a served Datum's per-value instant, and
// the wrapper exists only to cover that gap (PRD decision 7). A row whose
// payload starts carrying a real Datum stops coming through here; data.js's
// registry is the one place that flips.
//
// `as_of` IS THE SERVING INSTANT, not a guess at a measurement one. A plain
// value carries no measurement instant of its own, so the strongest true
// statement available is "the server had this when it served the payload", and
// the server-side gap is exactly zero: an earlier as_of would fabricate
// staleness, a later one would hide it.
//
// NO RECEIPT MEANS UNKNOWN, NOT ZERO. Before the first payload resolves,
// DF_DATA still holds its seeds, and a seed 0 rendered as a number is
// indistinguishable from a measured one.
//
// READS `__receipt` AND NEVER `__stale`. __stale records ATTEMPT history and is
// republished on failure by design; __receipt records the provenance of the
// value now in DF_DATA and advances on success only, which is what makes a
// wedged endpoint's tiles keep ageing. A tile consulting both would be a second
// staleness verdict fired at the same instant as the banner. Taking the
// __receipt map as the parameter makes that separation structural.
//
// AN ABSENT VALUE IS A HOLE even once the endpoint HAS delivered, which is what
// keeps this wrapper inside validate_datum's UNKNOWN_TRIAD rule. Stamping
// `value: null, state: 'fresh'` would build client-side the one envelope the
// server is forbidden to emit, and datumView would hand that null to a call
// site's `format` and badge the result with a real age.
//
// `== null` and never `!value`: a measured 0 (or '' or false) IS a measurement.
// `undefined` is in, because an optional-chained read (`x?.y`) is the commonest
// way a missing payload field reaches a tile.
function plainDatum(value, endpointKey, receipts) {
  const map = receipts === undefined || receipts === null ? plainDatumReceipts() : receipts;
  const receipt = map ? map[endpointKey] : undefined;
  if (!receipt || !Number.isFinite(Number(receipt.receivedAt))) {
    return unknownDatum('not yet fetched');
  }
  if (value === null || value === undefined) {
    return unknownDatum('no value in the payload');
  }

  return withReceipt(
    {
      value,
      as_of: receipt.servedAt || new Date(receipt.receivedAt).toISOString(),
      state: 'fresh',
      reason: null,
      freshness_bound_seconds: PLAIN_DATUM_BOUND_SECONDS,
    },
    receipt,
  );
}

// ── A value the PAYLOAD delivered, that a DOMAIN condition left absent ──
// plainDatum's absent arm says 'no value in the payload', which is true only
// when the payload is the reason. A large share of the migrated tiles pass a
// value derived AT THE SITE — a ratio whose denominator is zero, a forecast
// that needs seven days of history, an ISO instant that exists only once a run
// has completed — and there that string is an accusation against a server which
// answered perfectly. The site is the only place that knows the real reason,
// and in several cases it is already sitting in the tile's `hint`.
//
// COMPOSES THE TWO CONSTRUCTORS AND DECIDES NOTHING ITSELF. Exported rather
// than hand-written per file because three copies of one two-line composition
// is how the 14 null guards this leaf deleted began.
//
// THE ENDPOINT'S OWN ABSENCE OUTRANKS THE SITE'S REASON. Before the first
// payload resolves, a value derived from DF_DATA's seeds is null for a reason
// that is not the domain's — an empty seed array has no last bucket — so
// 'no ops in the last 24h' there would be a confident lie about data this
// browser has never seen. That question stays plainDatum's and is ASKED rather
// than re-implemented: a probe value it cannot call absent comes back unknown
// only when there is no receipt. The probe is never rendered.
//
// `absentReason` is REQUIRED. A caller with nothing better to say than 'no
// value in the payload' is describing the payload and should call plainDatum.
function derivedDatum(value, endpointKey, absentReason, receipts) {
  if (value !== null && value !== undefined) return plainDatum(value, endpointKey, receipts);
  const probe = plainDatum(0, endpointKey, receipts);
  return probe.state === 'unknown' ? probe : unknownDatum(absentReason);
}

// The browser default for plainDatum's third parameter, read LAZILY: a node
// caller passing its own map never touches a browser global, and a render
// before data.js has published degrades to "no receipt" rather than throwing.
function plainDatumReceipts() {
  return typeof window !== 'undefined' && window.DF_DATA ? window.DF_DATA.__receipt : null;
}

// Module-unique export const, never a bare `API` — the CANONICAL note at the
// head of this file, enforced at runtime by classic_script_scope.test.mjs.
const DATUM_API = {
  DATUM_STATES,
  isDatum,
  unknownDatum,
  assertDatum,
  withReceipt,
  displayedAgeMs,
  datumView,
  EM_DASH,
  LOWER_BOUND_PREFIX,
  plainDatum,
  derivedDatum,
  PLAIN_DATUM_BOUND_SECONDS,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = DATUM_API;
}
if (typeof window !== 'undefined') {
  window.DF_DATUM = DATUM_API;
}
