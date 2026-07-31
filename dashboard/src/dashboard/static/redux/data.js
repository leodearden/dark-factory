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

// Endpoint → DF_DATA keys map, parameterised on the active window chip.
// Only the four windowed endpoints append ?window=; the rest stay static.
function endpointsFor(win) {
  const w = encodeURIComponent(win);
  return {
    '/api/v2/dashboard/orchestrators':                ['ORCHESTRATORS', 'PROJECTS', 'ORCHESTRATORS_SPARK'],
    '/api/v2/dashboard/tasks':                        ['ACTIVE_TASKS', 'TASKS_OFFLINE', 'TASKS_OFFLINE_PROJECTS', 'DONE_COUNTS'],
    '/api/v2/dashboard/memory':                       ['MEMORY_STATUS'],
    '/api/v2/dashboard/memory-graphs':                ['MEMORY_TIMESERIES', 'MEMORY_OPS_BREAKDOWN'],
    '/api/v2/dashboard/recon':                        ['RECON_STATE', 'AGENTS'],
    [`/api/v2/dashboard/merge-queue?window=${w}`]:    ['MERGE_QUEUE'],
    [`/api/v2/dashboard/costs?window=${w}`]:          ['COSTS'],
    [`/api/v2/dashboard/performance?window=${w}`]:    ['PERFORMANCE'],
    [`/api/v2/dashboard/burndown?window=${w}`]:       ['BURNDOWN', 'BURNDOWN_BY_PROJECT'],
    '/api/v2/dashboard/curator':                      ['CURATOR_STATE'],
    '/api/v2/dashboard/scheduler':                    ['SCHEDULER'],
    '/api/v2/dashboard/escalations':                  ['ESCALATIONS'],
    '/api/v2/dashboard/escalation-analytics':         ['ESCALATION_ANALYTICS'],
  };
}

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
  ACTIVE_TASKS: [],
  TASKS_OFFLINE: false,
  TASKS_OFFLINE_PROJECTS: [],
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
  BURNDOWN: { labels: [], done: [], in_progress: [], blocked: [], pending: [], forecast_low: null, forecast_high: null },
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
};

function applyKey(key, value) {
  if (value === undefined || value === null) return;
  if (STABLE_ARRAY_KEYS.has(key) && Array.isArray(window.DF_DATA[key]) && Array.isArray(value)) {
    window.DF_DATA[key].length = 0;
    window.DF_DATA[key].push(...value);
  } else {
    window.DF_DATA[key] = value;
  }
}

// Flow-control state is keyed by endpoint PATH (query string stripped): four
// of the 13 endpoints carry ?window=<chip>, whose URL changes on every chip
// click (app.jsx:71 -> DF_REFRESH(win)). URL-keyed state would create a
// fresh entry on every chip change, silently resetting the in-flight flag
// (and, once backoff lands, its deadline) for those four endpoints.
function pollKey(url) {
  return url.split('?')[0];
}

function stateFor(state, url) {
  const key = pollKey(url);
  let st = state.get(key);
  if (!st) {
    st = { inFlight: false, failures: 0, nextAllowedAt: 0 };
    state.set(key, st);
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
const DEFAULT_TIMEOUT_MS = 30000; // 10x the poll interval

async function refreshOne(url, keys, state, deps) {
  const st = stateFor(state, url);
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
    const timeoutMs = deps.timeoutMs ?? DEFAULT_TIMEOUT_MS;
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
    keys.forEach(k => applyKey(k, body[k]));
    st.failures = 0;
    st.nextAllowedAt = 0;
  } catch (err) {
    recordFailure(st, deps);
    // Network blip, or a timed-out/aborted request — keep the prior values
    // so the UI does not blank out.
    console.warn('DF_DATA fetch failed', url, err);
  } finally {
    clearTimeoutFn(timeoutId);
    st.inFlight = false;
  }
}

// Module-scope window — updated by DF_REFRESH(win); 3 s polling reads from it
// so chip changes take effect on the next tick without restarting the loop.
let currentWin = '24h';

// Real (browser) deps; opts.deps overrides individual entries (tests inject
// a controllable clock/RNG/fetch instead of these).
const DEFAULT_POLL_DEPS = {
  now: () => Date.now(),
  random: () => Math.random(),
  sleep: ms => new Promise(r => setTimeout(r, ms)),
  fetchImpl: (u, i) => fetch(u, i),
  setTimeoutImpl: (fn, ms) => setTimeout(fn, ms),
  clearTimeoutImpl: id => clearTimeout(id),
  timeoutMs: DEFAULT_TIMEOUT_MS,
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
  await Promise.all(Object.entries(endpointsFor(currentWin)).map(([url, keys]) => {
    const ignoreBackoff = !!(isChipChange && url.includes('?window='));
    return refreshOne(url, keys, state, { ...baseDeps, ignoreBackoff });
  }));
  window.dispatchEvent(new CustomEvent('df-data-refresh'));
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
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = DF_DATA_LOADER_API;
}
if (typeof window !== 'undefined') {
  window.DF_DATA_LOADER = DF_DATA_LOADER_API;
}
