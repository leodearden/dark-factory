/* Real-data loader for Dark Factory dashboard.
 *
 * Replaces the mockup's synthetic fixtures.  Polls the server-side JSON API
 * exposed by dashboard.app under /api/v2/dashboard/* and merges results into
 * window.DF_DATA.
 *
 * Reference-stability matters: shell.jsx captures DF_DATA.PROJECTS and
 * DF_DATA.AGENTS at module-load (`const SHELL_PROJECTS = window.DF_DATA.PROJECTS`),
 * so we MUTATE those arrays in place rather than replacing the references.
 * Other DF_DATA.* values are read through the DF_DATA object reference each
 * render and can be replaced freely.
 */

// The Datum envelope's readers, destructured at module scope with no fallback
// (the DF_SPARK_PATH convention: throw loudly at load rather than defer to a
// TypeError inside a poll). index.html loads datum.js immediately before this
// file and test_index_html.py pins that edge.
//
// RENAMED IN THE DESTRUCTURE, not bound under datum.js's own names. Classic
// scripts share ONE global lexical scope, and each of these is already a
// top-level `function` declaration in datum.js — a same-named `const` here
// dies with "Identifier 'x' has already been declared" before this file
// reaches its own `window.DF_DATA = {...}`, which would take the entire
// dashboard down. classic_script_scope.test.mjs measures exactly that.
const {
  isDatum: isDatumPayload,
  withReceipt: stampWithReceipt,
  unknownDatum: unknownDatumPlaceholder,
} = window.DF_DATUM;

// How the wire delivers a key: 'plain' is a bare value, 'datum' is the
// five-key envelope data/datum.py::Datum.to_wire() emits. Two shared frozen
// objects rather than a fresh literal per row — a spec is a declaration, not
// per-row state, and freezing says so.
//
// EVERY POLLED KEY IS PLAIN TODAY, and that is a description rather than a
// placeholder: PRD leaf beta is what puts Datums on the wire, and until it
// lands, declaring a row datum-kinded would make applyKey refuse every real
// payload and freeze that tab at its seed values. This registry is the ONE
// place a later leaf flips a row.
const PLAIN = Object.freeze({ kind: 'plain' });
const DATUM = Object.freeze({ kind: 'datum' });

// Endpoint → {DF_DATA key: spec} map, parameterised on the active window chip.
// Only the four windowed endpoints append ?window=; the rest stay static.
//
// THE KEY NAMES ARE QUOTED, and must stay quoted. Four Python structural tests
// — test_tab_curator.py, test_tab_escalations.py,
// test_tab_escalation_analytics.py and test_tab_memory_evals.py — assert their
// endpoint is registered here by searching this shipped source for `'KEY'` or
// `"KEY"`. Unquoting them is silent: the registry still works and those four
// tests fail with "add it as the mapped key", naming a key that is in fact
// already present. Same reasoning as DEFAULT_TIMEOUT_MS further down, which
// two other Python tests parse straight out of this file.
function endpointsFor(win) {
  const w = encodeURIComponent(win);
  return {
    '/api/v2/dashboard/orchestrators':                { 'ORCHESTRATORS': PLAIN, 'PROJECTS': PLAIN, 'ORCHESTRATORS_SPARK': PLAIN },
    '/api/v2/dashboard/tasks':                        { 'ACTIVE_TASKS': PLAIN, 'TASKS_OFFLINE': PLAIN, 'TASKS_OFFLINE_PROJECTS': PLAIN,
                                                        'TASKS_DEGRADED_PROJECTS': PLAIN, 'TASKS_COUNT_UNKNOWN_PROJECTS': PLAIN, 'TASKS_PROJECT_COUNT': PLAIN,
                                                        'DONE_COUNTS': PLAIN },
    '/api/v2/dashboard/memory':                       { 'MEMORY_STATUS': PLAIN },
    '/api/v2/dashboard/memory-graphs':                { 'MEMORY_TIMESERIES': PLAIN, 'MEMORY_OPS_BREAKDOWN': PLAIN },
    '/api/v2/dashboard/recon':                        { 'RECON_STATE': PLAIN, 'AGENTS': PLAIN },
    [`/api/v2/dashboard/merge-queue?window=${w}`]:    { 'MERGE_QUEUE': PLAIN },
    [`/api/v2/dashboard/costs?window=${w}`]:          { 'COSTS': PLAIN },
    [`/api/v2/dashboard/performance?window=${w}`]:    { 'PERFORMANCE': PLAIN },
    [`/api/v2/dashboard/burndown?window=${w}`]:       { 'BURNDOWN': PLAIN, 'BURNDOWN_BY_PROJECT': PLAIN },
    '/api/v2/dashboard/curator':                      { 'CURATOR_STATE': PLAIN },
    '/api/v2/dashboard/scheduler':                    { 'SCHEDULER': PLAIN },
    '/api/v2/dashboard/escalations':                  { 'ESCALATIONS': PLAIN },
    '/api/v2/dashboard/escalation-analytics':         { 'ESCALATION_ANALYTICS': PLAIN },
    '/api/v2/dashboard/memory-evals':                 { 'MEMORY_EVALS': PLAIN },
  };
}

// Keys fetched on a USER ACTION rather than by the poll loop, parameterised
// per project. One declared row today: `terminal`, the mechanism PRD leaf
// gamma3 fetches `?terminal=<project>` through.
//
// A ROW CARRIES BUILDERS, NOT TEMPLATE STRINGS, and nothing re-derives either
// one at a call site — a caller holds a project name and asks for the row, so
// the url/key pair is constructed in exactly one place and cannot drift.
//
// `key(param)` names BOTH what the response body calls the value and what
// DF_DATA calls it. That is the same rule endpointsFor's rows already follow,
// where a row's key name is simultaneously the body key and the DF_DATA key;
// the only difference here is that the name is BUILT from a parameter instead
// of written as a literal, which is why these keys cannot be seeded in the
// DF_DATA block above and why datumFor exists.
//
// Note which half is encoded: the url must survive HTTP parsing, and the key
// must be the name a caller can look up with the project string it already
// holds. No call site should have to know which is which.
const ON_DEMAND_KEYS = {
  terminal: {
    url: project => `/api/v2/dashboard/tasks?terminal=${encodeURIComponent(project)}`,
    key: project => `TASKS_TERMINAL:${project}`,
    spec: DATUM,
  },
};

// Keys whose array reference is captured at module-load by shell.jsx — mutate
// in place rather than reassigning, so cached references stay valid.
const STABLE_ARRAY_KEYS = new Set(['PROJECTS', 'AGENTS']);

// ── Empty defaults — all keys initialised so the first render before fetch
//    completes does not crash any component reading DF_DATA.* ──
window.DF_DATA = {
  PROJECTS: [],
  AGENTS: [],
  ORCHESTRATORS: [],
  ORCHESTRATORS_SPARK: { labels: [], values: [] },
  // ACTIVE_TASKS row shape: {id, project, title, status, agent, started, loops,
  //   attempts, lane, phase, lane_state, runtime_offline, deps, meta_files,
  //   train, external_deps, prd, claimant_run_id, heartbeat_at, stranded}.
  //   `agent` is worktree PRESENCE (it stays truthy after the agent dies);
  //   `stranded` (task 3543) is the independent liveness verdict, computed
  //   server-side from the claim columns via shared.task_claimant.is_stranded.
  ACTIVE_TASKS: [],
  TASKS_OFFLINE: false,
  TASKS_OFFLINE_PROJECTS: [],
  // Projects the tasks handler ran out of budget for — state UNKNOWN, not
  // offline. Defaulted here (not just on the wire) because the first render
  // happens before any fetch completes.
  TASKS_DEGRADED_PROJECTS: [],
  // Projects whose rows are current but whose compact status map failed, so
  // the done count was never measured — neither offline nor degraded. Without
  // this list they would render as healthy with a confident "0 done".
  TASKS_COUNT_UNKNOWN_PROJECTS: [],
  // N for the banner's "k of N": how many task project roots the tasks handler
  // fanned out over. Comes from the server so it denominates the same
  // population TASKS_OFFLINE_PROJECTS is drawn from — PROJECTS (orchestrator-
  // derived) is a different one. 0 pre-fetch, which the banner reads as "no
  // count yet" rather than dividing by it.
  TASKS_PROJECT_COUNT: 0,
  DONE_COUNTS: {},
  PERFORMANCE: {},
  MEMORY_STATUS: {
    graphiti: { connected: false, node_count: 0, edge_count: 0, episode_count: 0 },
    mem0: { connected: false, memory_count: 0 },
    taskmaster: { connected: false },
    queue: { counts: { pending: 0, retry: 0, dead: 0 }, oldest_pending_age_seconds: null },
    projects: {},
    wal: { status: 'offline', reason: null, rows: [] },
  },
  MEMORY_TIMESERIES: { labels: [], reads: [], writes: [] },
  MEMORY_OPS_BREAKDOWN: [],
  RECON_STATE: {
    buffer: { buffered_count: 0, oldest_event_age_seconds: null },
    burst_state: [],
    watermarks: {},
    verdict: null,
    runs: [],
  },
  // MERGE_QUEUE: {project_label: {depth, outcomes, latency, recent, speculative,
  //   active, active_spark, halt, train_events: [{event_type, task_id, run_id,
  //   timestamp, data: {train_id, member_task_ids, ...event-specific keys}}]}}
  MERGE_QUEUE: {},
  COSTS: {
    summary: { total: 0, runs: 0, today: 0, tokens: null, p95_run_cost: null, delta_pct: null, delta_hint: null },
    by_project: [],
    by_account: [],
    by_role: [],
    trend: { labels: [], values: [] },
    events: [],
    // Per-(model×role) outcome rollup (task 2534 δ): invocation/done/blocked
    // counts+rates, cap-hit rate, $/done, plus per-role turn-cap saturation.
    by_model_role: { rows: [], turn_cap_saturation: {} },
  },
  // in_progress_live + in_progress_stranded band the in_progress census; the
  // parity_* fields are the server's cap-breach verdict (task 3543). Seeded so
  // a cold client renders empty bands and no banner, never `undefined` ones.
  BURNDOWN: { labels: [], done: [], in_progress: [], in_progress_live: [], in_progress_stranded: [], blocked: [], pending: [], forecast_low: null, forecast_high: null, parity_alarm: false, parity_cap: null, parity_peak: null, parity_breach_count: 0, parity_projects: [] },
  BURNDOWN_BY_PROJECT: {},
  // CURATOR_STATE is an object (not a captured top-level array), so it is NOT
  // added to STABLE_ARRAY_KEYS. applyKey replaces the reference on each poll;
  // tab_curator.jsx reads through DF_DATA.CURATOR_STATE per render.
  CURATOR_STATE: {
    pending: [],
    latency_spark: { labels: [], p50: [], p90: [], p99: [] },
    pending_spark: { labels: [], values: [] },
    capped_spark: { labels: [], values: [] },
    state: { capped_now: 0, paused_reason: null, pending_total: 0, accounts_summary: { total: 0, capped: 0, available: 0, capped_accounts: [] } },
  },
  // ESCALATIONS is an object (not a captured top-level array), so it is NOT
  // added to STABLE_ARRAY_KEYS. applyKey replaces the reference on each poll;
  // tab_escalations.jsx reads through DF_DATA.ESCALATIONS per render.
  ESCALATIONS: {
    subsections: [],
    summary: {
      by_level: { 0: 0, 1: 0, 2: 0 },
      by_status: { pending: 0, resolved: 0, dismissed: 0 },
      skipped_count: 0,
    },
  },
  // ESCALATION_ANALYTICS is an object (not a captured top-level array), so it is
  // NOT added to STABLE_ARRAY_KEYS. applyKey replaces the reference on each poll;
  // a future analytics tab reads through DF_DATA.ESCALATION_ANALYTICS per render.
  ESCALATION_ANALYTICS: {
    generated_at: null,
    parse_failures: 0,
    regime_markers: [],
    per_project: [],
  },
  // SCHEDULER is read through DF_DATA.SCHEDULER per render (not captured at
  // module-load), so reference replacement on each poll is safe.
  SCHEDULER: {
    rows: [],
    modules: [],
    pin_queue: [],
    events_by_task: {},
    snapshot_at: null,
    offline: false,
    offline_projects: [],
    paused: false,
    paused_projects: [],
  },
  // MEMORY_EVALS is an object (not a captured top-level array), so it is NOT
  // added to STABLE_ARRAY_KEYS. applyKey replaces the reference on each poll;
  // tab_memory_evals.jsx reads through DF_DATA.MEMORY_EVALS per render.
  //
  // This seed mirrors redux_api.shape_memory_evals' default body exactly:
  // root_present false with empty lists is the server's OWN healthy
  // no-artifacts shape, so the pre-fetch render and a real empty response are
  // indistinguishable and no component has to branch on which it got.  No
  // illustrative rows — an invented eval would be synthetic data.
  //
  // Deliberately no per-endpoint poll interval: the shared 3s tick plus the
  // route's 60s server-side TTL single-flight cache (PRD DD4) already bounds
  // the daily-cadence artifact file scan.
  MEMORY_EVALS: {
    generated_at: null,
    root_present: false,
    storm_escape: null,
    evals: [],
    issues: [],
    issue_count: 0,
    unmatched_escalations: [],
  },
  // Per-key FIRST-SUCCESS markers: `__loaded[KEY]` flips true the first time a
  // real server value for that key is applied, and never back.
  //
  // Consumers need this to tell a PRE-FETCH SEED from a loaded-but-genuinely-
  // EMPTY payload.  By design the two are structurally identical — the seeds
  // above deliberately mirror the server's own healthy empty shape so no
  // component has to branch on which it got (see the MEMORY_EVALS note) — so
  // nothing about a payload's CONTENTS can distinguish them.
  //
  // Nor can arrival be inferred client-side from object identity.  Capturing a
  // seed reference at module-eval time and testing `payload !== SEED` races
  // this file: a `type="text/babel"` module is transpiled and evaluated after
  // DOMContentLoaded, while startPolling()'s immediate first fetch can resolve
  // BEFORE that — freezing a REAL payload as the "seed", and never unfreezing
  // it while polling is paused (`__DF_PAUSE` skips every later pollTick) or the
  // endpoint sits in backoff.  A consumer keyed on that comparison then claims
  // "still loading" over a fully-populated table, indefinitely.
  //
  // Nested under DF_DATA rather than added as a second global so a consumer
  // holding `const DF = window.DF_DATA` reads it with no new capture.  It is
  // not an endpoint key, so applyKey never overwrites it.
  __loaded: {},
  // Per-endpoint staleness, keyed by endpoint PATH (pollKey): each entry is
  // `{failures, lastSuccessAt}`, republished by refreshOne on BOTH the
  // success and the failure path. endpoint_staleness.js turns it into the
  // notices app.jsx renders above every tab body.
  //
  // Nested under DF_DATA for the same reason __loaded is — see that block
  // above: a consumer holding `const DF = window.DF_DATA` reads it with no
  // new capture, and it is not an endpoint key, so applyKey never overwrites
  // it.
  //
  // Staleness is derived from RECORDED TIMESTAMPS, never from an identity
  // comparison against a captured seed. The __loaded block documents why
  // that pattern is unsafe here (a text/babel module is evaluated after
  // DOMContentLoaded, while the immediate first fetch can resolve BEFORE
  // that, freezing a REAL payload as the "seed"), and the same race would
  // make an identity-derived staleness verdict wrong in the same direction:
  // it would report "current" for a payload that has not moved in hours.
  // Do not reintroduce it.
  __stale: {},
  // Per-endpoint RECEIPTS, keyed by the same flow-control key __stale uses:
  // each entry is `{servedAt, receivedAt}`, published by refreshOne on the
  // SUCCESS path only. datum.js::plainDatum reads this to give a value that
  // is not yet served as a Datum the provenance it does have.
  //
  // DELIBERATELY NOT MERGED WITH __stale, which sits three lines above it.
  // The two answer different questions and have different lifetimes. __stale
  // records ATTEMPT history and is republished in refreshOne's `finally` by
  // design, so a 503 counts exactly like a timeout; it is
  // endpoint_staleness.js's input and the sole endpoint-freshness authority.
  // __receipt records the PROVENANCE of the values currently sitting in
  // DF_DATA, and must NOT advance on a failure — that is precisely what makes
  // a wedged endpoint's tiles keep ageing on screen instead of looking
  // freshest while the server is least reachable. `lastSuccessAt` and
  // `receivedAt` coincide today BY CONSTRUCTION (refreshOne reads the clock
  // once and uses it for both), which is worth this comment rather than a
  // merge: merging would put a staleness verdict and a provenance stamp in one
  // entity, and make "no second staleness decision" unenforceable.
  //
  // `__`-prefixed, so applyKey's existing refusal already protects it from a
  // server payload key of the same name.
  __receipt: {},
};

// Apply one key of a response body to DF_DATA.
//
// `spec` declares how the wire delivers this key and defaults to PLAIN, so
// every existing two-argument caller is unaffected. On a datum-kinded row the
// payload is VALIDATED before it is applied: one that fails isDatum is
// refused outright and the previous value is left exactly where it is, with
// its own receipt intact and its age still growing. Refusing rather than
// storing is the point — a server regression that starts sending a bare
// number where a Datum was declared would otherwise replace a provenanced
// value with an unprovenanced one that renders as though freshly measured,
// which is the single failure this envelope exists to remove.
function applyKey(key, value, spec, receipt) {
  if (value === undefined || value === null) return;
  // `__`-prefixed names are DF_DATA's internal namespace (__loaded, __stale)
  // — never endpoint keys, and never anything a server payload may write.
  // The invariant was previously structural only (no endpointsFor() key list
  // names one), which left both maps one server-side key rename away from
  // being silently overwritten: a payload key literally named `__loaded`
  // would flip every marker, and one named `__stale` would erase the very
  // record that reports the server is failing. Enforce it here instead.
  if (typeof key === 'string' && key.startsWith('__')) return;
  let applied = value;
  if ((spec || PLAIN).kind === 'datum') {
    if (!isDatumPayload(value)) return;
    applied = stampWithReceipt(value, receipt);
  }
  if (STABLE_ARRAY_KEYS.has(key) && Array.isArray(window.DF_DATA[key]) && Array.isArray(applied)) {
    window.DF_DATA[key].length = 0;
    window.DF_DATA[key].push(...applied);
  } else {
    window.DF_DATA[key] = applied;
  }
  // Marked AFTER the apply, and only past the null/undefined guard above, so
  // the marker means "a real server value for this key has LANDED" — never
  // "a fetch was attempted" and never "the response omitted this key".
  window.DF_DATA.__loaded[key] = true;
}

// The Datum now sitting under *key*, or an unknown one saying nothing has
// arrived yet.
//
// AN ACCESSOR RATHER THAN UNKNOWN-DATUM LITERALS IN THE SEED BLOCK ABOVE. A
// seed literal per datum-kinded key would be a second copy of the unknown
// Datum, free to drift from unknownDatum() (SPOT), and it could not express
// the case that actually matters: the parameterised per-project keys
// (TASKS_TERMINAL:<project>) are not statically enumerable, so there is no
// place to seed them. One accessor answers both.
function datumFor(key) {
  const value = window.DF_DATA[key];
  return isDatumPayload(value) ? value : unknownDatumPlaceholder('not yet fetched');
}

// Flow-control state is keyed by endpoint PATH (query string stripped): four
// of the 13 endpoints carry ?window=<chip>, whose URL changes on every chip
// click (app.jsx:71 -> DF_REFRESH(win)). URL-keyed state would create a
// fresh entry on every chip change, silently resetting the in-flight flag
// (and, once backoff lands, its deadline) for those four endpoints.
function pollKey(url) {
  return url.split('?')[0];
}

// Takes the resolved stateKey rather than the url, so the ONE decision of
// "which flow-control entry does this request belong to" is made by refreshOne
// and made once. Every polled caller still gets pollKey(url); an on-demand
// request gets its own key, which is the whole point (see ON_DEMAND_KEYS).
function stateFor(state, stateKey) {
  let st = state.get(stateKey);
  if (!st) {
    st = { inFlight: false, failures: 0, nextAllowedAt: 0, lastSuccessAt: 0 };
    state.set(stateKey, st);
  }
  return st;
}

// Error backoff: 3000ms * 2^(failures-1), capped at 60s. A non-ok HTTP
// status counts as a failure alongside a thrown error — the motivating
// incident is an overloaded server (503), not just network blips, and a
// !resp.ok today is silently retried at the full 3s rate forever.
const BACKOFF_BASE_MS = 3000;
const BACKOFF_MAX_MS = 60000;
function backoffDelay(failures) {
  return Math.min(BACKOFF_BASE_MS * Math.pow(2, failures - 1), BACKOFF_MAX_MS);
}

// A forced (backoff-bypassing) attempt that still fails must not escalate
// the TIMER path's backoff — otherwise repeated chip clicks during an
// outage would inflate failures/nextAllowedAt derived from user action
// alone, and the dashboard could stay dark longer than the real failure
// history warrants once the server recovers.
function recordFailure(st, deps) {
  if (deps.ignoreBackoff) return;
  st.failures += 1;
  st.nextAllowedAt = deps.now() + backoffDelay(st.failures);
}

// Jitter: spreads the 13 endpoint fetches across part of the 3s interval
// instead of every tick firing all 13 at once (task 185's lesson — 13
// simultaneous requests hammering a single aiosqlite worker thread). Capped
// at half the poll interval so a jittered start can never structurally slip
// past the next tick.
const JITTER_MAX_MS = 1500;

// A hung request (the motivating incident measured a 108s memory-graphs
// response, with no upper bound) must not wedge an endpoint's in-flight
// flag forever — every later tick would then skip it for the lifetime of
// the page, with no console warning and no UI signal. Bound each attempt
// with an abort deadline; timing out is treated exactly like a thrown
// fetch error (counts toward backoff, clears in-flight in `finally`).
//
// LEAVE THIS EXACTLY AS IT IS — same name, same literal, same assignment
// shape. Two Python structural tests parse it straight out of this shipped
// source with /DEFAULT_TIMEOUT_MS\s*=\s*(\d+)/ — test_tasks_budget.py and
// test_fetch_tasks_whole_operation_budget.py — where it is the ONLY ceiling
// on the server-side budgets. Renaming it, or tidying it into a computed
// expression, fails both loudly.
const DEFAULT_TIMEOUT_MS = 30000; // 10x the poll interval

// A shorter deadline for an endpoint that has already demonstrated it is
// failing. Deliberately a SECOND, separately named constant rather than a
// redefinition of the one above.
//
// THE CONNECTION-BUDGET ARITHMETIC. Browsers allow ~6 concurrent HTTP/1.1
// connections per origin. Three wedged endpoints each holding a socket for
// the full 30s deadline is about half that budget held continuously, which
// is why the HEALTHY tabs also felt sluggish and slow to answer chip changes
// during the 2026-08-27 incident. Once an endpoint has failed
// STALE_FAILURE_THRESHOLD times in a row there is nothing left to wait 30s
// for: it is already reported stale in the UI, and a shorter deadline gets
// the socket back for the tabs that are still working.
const STALE_TIMEOUT_MS = 5000;

// Fallback for endpoint_staleness.js's threshold, read LAZILY with this
// literal as a fallback — deliberately unlike the DF_DATUM destructure at the
// head of this file, which is module-scope and has no fallback at all.
//
// The two differ because their histories do. This one dates from when data.js
// was the FIRST classic script in index.html, so a module-scope read would
// have named a script that had not run yet; that is no longer true (task 5588
// moved endpoint_staleness.js and datum.js ahead of this file, pinned in
// test_index_html.py), but the lazy read is kept because changing it is not
// this task's business and the fallback literal it carries is already pinned
// against the real value by endpoint_staleness.test.mjs. Do NOT copy this
// shape for a new dependency: it costs a duplicated constant plus a test to
// keep the two copies agreeing, which is exactly the drift a module-scope
// destructure removes.
const STALE_FAILURE_THRESHOLD_FALLBACK = 3;

function staleFailureThreshold() {
  if (typeof window !== 'undefined' && window.DF_ENDPOINT_STALENESS) {
    const n = Number(window.DF_ENDPOINT_STALENESS.STALE_FAILURE_THRESHOLD);
    if (Number.isFinite(n) && n > 0) return n;
  }
  return STALE_FAILURE_THRESHOLD_FALLBACK;
}

// Republish this endpoint's staleness record onto DF_DATA. Called from
// refreshOne's `finally`, so it covers the success path, the thrown-error
// path AND the `!resp.ok` early return alike — an endpoint that 503s for an
// hour is exactly as stale as one that times out for an hour.
function publishStaleness(stateKey, st) {
  if (typeof window === 'undefined' || !window.DF_DATA) return;
  window.DF_DATA.__stale[stateKey] = {
    failures: st.failures,
    lastSuccessAt: st.lastSuccessAt,
  };
}

// Record the provenance of the values this response just delivered. Called on
// the SUCCESS path ONLY — pointedly NOT from `finally`, where publishStaleness
// lives. See the __receipt seed block for why the asymmetry is the whole point.
function publishReceipt(stateKey, receipt) {
  if (typeof window === 'undefined' || !window.DF_DATA) return;
  window.DF_DATA.__receipt[stateKey] = receipt;
}

// `stateKey` names the flow-control, staleness and receipt entry this request
// owns, and defaults to pollKey(url) — so every poll-loop call is unchanged
// and every existing direct caller keeps working. An on-demand request passes
// its own key instead; see requestOnDemand for why it must.
async function refreshOne(url, keySpecs, state, deps, stateKey = pollKey(url)) {
  const st = stateFor(state, stateKey);
  if (st.inFlight) return; // already in flight for this endpoint — skip this tick, do not queue
  if (deps.now() < st.nextAllowedAt && !deps.ignoreBackoff) return; // still backed off
  st.inFlight = true;
  // Fall back inline (not via DEFAULT_POLL_DEPS) so a caller that hand-builds
  // a partial deps object — e.g. refreshOne invoked directly with just
  // {fetchImpl, now} — still gets a working deadline instead of throwing on
  // a missing dep.
  const setTimeoutFn = deps.setTimeoutImpl || ((fn, ms) => setTimeout(fn, ms));
  const clearTimeoutFn = deps.clearTimeoutImpl || (id => clearTimeout(id));
  let timeoutId;
  try {
    // Awaited INSIDE the in-flight window (st.inFlight is already true) so a
    // second tick firing while this endpoint is still jittering is skipped
    // by the check above, not free to sneak in a duplicate request.
    if (deps.jitterMaxMs > 0) {
      await deps.sleep(Math.floor(deps.random() * deps.jitterMaxMs));
    }
    // An explicitly injected deps.timeoutMs still wins over both defaults;
    // the reduced deadline is a DEFAULT selection, not an override.
    const timeoutMs = deps.timeoutMs
      ?? (st.failures >= staleFailureThreshold() ? STALE_TIMEOUT_MS : DEFAULT_TIMEOUT_MS);
    const controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    // Races the fetch against a deadline instead of relying on the fetch
    // implementation to honour AbortSignal itself (test stubs generally
    // don't) — the abort is still issued, so a real fetch's underlying
    // network request is actually cancelled, but the race is what
    // guarantees this function moves on regardless.
    const timedOut = new Promise((_, reject) => {
      timeoutId = setTimeoutFn(() => {
        if (controller) controller.abort();
        reject(new Error(`DF_DATA fetch timed out after ${timeoutMs}ms: ${url}`));
      }, timeoutMs);
    });
    const resp = await Promise.race([
      deps.fetchImpl(url, { credentials: 'same-origin', signal: controller ? controller.signal : undefined }),
      timedOut,
    ]);
    if (!resp.ok) {
      recordFailure(st, deps);
      return;
    }
    const body = await resp.json();
    // ONE clock reading for the whole response, so every key it carries shares
    // a single arrival instant — and so `lastSuccessAt` below cannot drift
    // from `receivedAt` by however long the applies took.
    const receipt = { servedAt: body.served_at ?? null, receivedAt: deps.now() };
    Object.entries(keySpecs).forEach(([k, spec]) => applyKey(k, body[k], spec, receipt));
    st.failures = 0;
    st.nextAllowedAt = 0;
    st.lastSuccessAt = receipt.receivedAt;
    publishReceipt(stateKey, receipt);
  } catch (err) {
    recordFailure(st, deps);
    // Network blip, or a timed-out/aborted request — keep the prior values
    // so the UI does not blank out.
    console.warn('DF_DATA fetch failed', url, err);
  } finally {
    clearTimeoutFn(timeoutId);
    st.inFlight = false;
    // In `finally` so the `!resp.ok` early return is covered too, not just
    // the success and thrown-error paths.
    publishStaleness(stateKey, st);
  }
}

// Module-scope window — updated by DF_REFRESH(win); 3 s polling reads from it
// so chip changes take effect on the next tick without restarting the loop.
let currentWin = '24h';

// Real (browser) deps; opts.deps overrides individual entries (tests inject
// a controllable clock/RNG/fetch instead of these).
//
// EVERY ENTRY HERE IS AN ENVIRONMENT CAPABILITY — a clock, an RNG, fetch, the
// timer pair. Do NOT add a policy value, and `timeoutMs` in particular. It was
// pinned here once and that silently disabled the reduced deadline entirely:
// refreshDFData merges these FIRST, so `deps.timeoutMs` was never undefined,
// so refreshOne's `deps.timeoutMs ?? (failures >= threshold ? STALE_TIMEOUT_MS
// : DEFAULT_TIMEOUT_MS)` could not fall through and STALE_TIMEOUT_MS was dead
// code in every browser. The node suite stayed green throughout, because its
// tests hand refreshOne a partial deps object with no timeoutMs and so take a
// path production never takes. Chrome 151 is what caught it (task 4884 step-19,
// #4791 acceptance 3): four consecutive ~30000ms aborts on a wedged endpoint
// whose banner already read "4 consecutive attempts failed".
//
// Now pinned from both ends by data_poll.test.mjs's "PRODUCTION deps merge"
// test — behaviourally through refreshDFData, and structurally against this
// literal.
const DEFAULT_POLL_DEPS = {
  now: () => Date.now(),
  random: () => Math.random(),
  sleep: ms => new Promise(r => setTimeout(r, ms)),
  fetchImpl: (u, i) => fetch(u, i),
  setTimeoutImpl: (fn, ms) => setTimeout(fn, ms),
  clearTimeoutImpl: id => clearTimeout(id),
};

// `opts.state`/`opts.deps` let callers supply isolated flow-control state
// and a controllable clock/RNG/fetch; production callers fall back to the
// shared DF_POLL_STATE singleton and the real fetch/timers.
//
// A chip change (explicit non-empty `win`, app.jsx:71) bypasses backoff ONLY
// for the 4 windowed endpoints whose URL actually changes on that chip click
// — the other 9 endpoints have no bearing on the chip and must keep
// respecting whatever backoff the TIMER path already accumulated for them,
// otherwise a chip click during an outage would re-hammer every endpoint,
// recreating exactly the load this task removes. A forced attempt that
// still fails does not touch failures/nextAllowedAt (see recordFailure), so
// repeated chip clicks cannot escalate the timer path's own backoff
// schedule. The in-flight check in refreshOne is unconditional regardless
// of ignoreBackoff, so a chip change still cannot stack a second concurrent
// request for an endpoint that's already running.
async function refreshDFData(win, opts) {
  const o = opts || {};
  const isChipChange = typeof win === 'string' && win;
  if (isChipChange) currentWin = win;
  const state = o.state || DF_POLL_STATE;
  const baseDeps = {
    ...DEFAULT_POLL_DEPS,
    ...o.deps,
    jitterMaxMs: o.jitterMaxMs ?? JITTER_MAX_MS,
  };
  await Promise.all(Object.entries(endpointsFor(currentWin)).map(([url, keySpecs]) => {
    const ignoreBackoff = !!(isChipChange && url.includes('?window='));
    return refreshOne(url, keySpecs, state, { ...baseDeps, ignoreBackoff });
  }));
  window.dispatchEvent(new CustomEvent('df-data-refresh'));
}

// Fetch one declared on-demand row for one parameter, through the SAME
// refreshOne every polled endpoint uses — so the timeout/abort deadline, the
// backoff, the in-flight guard and the receipt all come from one copy rather
// than a second fetch path that would drift from it.
//
// WHY IT NEEDS ITS OWN stateKey. pollKey strips the query string on purpose
// (the four ?window= endpoints must not get a fresh flow-control entry on
// every chip click), so an on-demand `/api/v2/dashboard/tasks?terminal=<p>`
// would otherwise land on the POLLED `/api/v2/dashboard/tasks` entry: it would
// set that endpoint's in-flight flag — making the poll loop skip the real
// tasks fetch for as long as a user's terminal request runs — reset or
// escalate its backoff, and write its __stale entry, reporting an endpoint
// stale that never failed. A user action must not be able to blind a tab.
//
// An unrecognised `name` throws rather than returning quietly: there is no url
// to build for it, and a silent no-op would make a typo indistinguishable from
// an empty result.
//
// NO JITTER. It exists to spread the 14 poll fetches across the interval; a
// single user-triggered request has nothing to spread against, and delaying it
// would only be latency the user sees.
async function requestOnDemand(name, param, opts) {
  const row = ON_DEMAND_KEYS[name];
  if (!row) {
    throw new Error(`DF_DATA: no on-demand key named '${name}' (declared: ${Object.keys(ON_DEMAND_KEYS).join(', ')})`);
  }
  const o = opts || {};
  const url = row.url(param);
  const deps = { ...DEFAULT_POLL_DEPS, ...o.deps, jitterMaxMs: 0 };
  await refreshOne(
    url,
    { [row.key(param)]: row.spec },
    o.state || DF_POLL_STATE,
    deps,
    `${pollKey(url)}#${name}:${param}`,
  );
}

window.DF_REFRESH = refreshDFData;
window.__DF_PAUSE = false;

// Isolated per-endpoint flow-control state (steps 3-8 give this Map real
// entries keyed by endpoint path). createPollState() also lets tests hand
// pollTick/refreshDFData a fresh Map instead of sharing this singleton.
function createPollState() {
  return new Map();
}
const DF_POLL_STATE = createPollState();

const POLL_INTERVAL_MS = 3000;

function pollTick(opts) {
  if (!window.__DF_PAUSE) refreshDFData(undefined, opts);
}

function startPolling(opts) {
  refreshDFData(undefined, opts);
  const handle = setInterval(() => pollTick(opts), POLL_INTERVAL_MS);
  return {
    stop() {
      clearInterval(handle);
    },
  };
}

// Auto-start only in a real browser document context. index.html loads this
// file as a classic <script> where `document` always exists, so the browser
// path above is unchanged; requiring this module under `node --test` (no
// `document` shim) leaves it inert instead of firing real fetches and
// leaving a live timer that would hang the test runner — see
// dashboard/tests/js/data_poll.test.mjs.
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  startPolling();
}

// Module-unique export const, never a bare `API` — see the
// shared-classic-script-scope note in graph_layout.js's header, enforced by
// dashboard/tests/js/classic_script_scope.test.mjs.
const DF_DATA_LOADER_API = {
  endpointsFor,
  applyKey,
  refreshOne,
  refreshDFData,
  pollTick,
  startPolling,
  createPollState,
  pollKey,
  datumFor,
  ON_DEMAND_KEYS,
  requestOnDemand,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = DF_DATA_LOADER_API;
}
if (typeof window !== 'undefined') {
  window.DF_DATA_LOADER = DF_DATA_LOADER_API;
}
